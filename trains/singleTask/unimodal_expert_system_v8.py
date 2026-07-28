"""Three-stage training and evaluation for V8 unimodal experts."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids


LOGGER = logging.getLogger("MMSA")


ACCEPTANCE_THRESHOLDS = {
    "text": {"mae": 0.705, "corr": 0.79, "spearman": 0.25, "auroc": 0.68},
    "audio": {"mae": 1.40, "corr": 0.18, "spearman": 0.15, "auroc": 0.62},
    "vision": {"mae": 1.40, "corr": 0.15, "spearman": 0.15, "auroc": 0.62},
}


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _safe_float(value, default=float("nan")) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average-tie ranks without a SciPy dependency."""
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = _rankdata(first)
    second_rank = _rankdata(second)
    if first_rank.std() < 1e-12 or second_rank.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def uncertainty_diagnostics(
    prediction: torch.Tensor,
    uncertainty: torch.Tensor,
    labels: torch.Tensor,
    error_scale: float,
) -> Dict[str, object]:
    prediction_np = prediction.view(-1).detach().cpu().numpy().astype(np.float64)
    uncertainty_np = uncertainty.view(-1).detach().cpu().numpy().astype(np.float64)
    labels_np = labels.view(-1).detach().cpu().numpy().astype(np.float64)
    error = np.abs(prediction_np - labels_np)
    target = np.tanh(error / max(float(error_scale), 1e-4))

    order = np.argsort(uncertainty_np, kind="mergesort")
    groups = np.array_split(order, 4)
    quartile_mae = [
        float(error[group].mean()) if len(group) else float("nan")
        for group in groups
    ]
    q1 = quartile_mae[0]
    q4 = quartile_mae[-1]
    q4_q1 = (
        float(q4 / q1)
        if math.isfinite(q1) and q1 > 1e-12 and math.isfinite(q4)
        else float("nan")
    )

    high_threshold = float(np.quantile(error, 0.70))
    high_error = (error >= high_threshold).astype(np.int64)
    try:
        auroc = float(roc_auc_score(high_error, uncertainty_np))
    except ValueError:
        auroc = float("nan")

    return {
        "error_spearman": _spearman(uncertainty_np, error),
        "error_target_spearman": _spearman(uncertainty_np, target),
        "high_error_auroc": auroc,
        "high_error_threshold": high_threshold,
        "uncertainty_loss": float(np.mean(np.abs(uncertainty_np - target))),
        "quartile_mae": quartile_mae,
        "q4_q1_ratio": q4_q1,
        "quartile_monotonic": bool(
            all(
                quartile_mae[index] <= quartile_mae[index + 1] + 1e-12
                for index in range(3)
            )
        ),
        "mean_uncertainty": float(uncertainty_np.mean()),
        "mean_absolute_error": float(error.mean()),
    }


def fit_train_normalizer(dataset, modality: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, object]]:
    """Fit per-feature statistics on valid training tokens only."""
    if modality == "text" and bool(getattr(dataset.args, "use_bert", False)):
        return None, None, {"enabled": False, "reason": "BERT output uses LayerNorm"}

    array = getattr(dataset, modality if modality != "vision" else "vision")
    values = torch.as_tensor(array, dtype=torch.float32)
    if values.ndim != 3:
        raise ValueError(f"Expected [N,T,D] {modality} array, got {tuple(values.shape)}")
    finite = torch.isfinite(values).all(dim=-1)
    nonzero = values.abs().sum(dim=-1) > 0
    mask = finite & nonzero
    flattened = values[mask]
    if flattened.numel() == 0:
        mean = torch.zeros(values.shape[-1])
        std = torch.ones(values.shape[-1])
    else:
        mean = flattened.mean(dim=0)
        std = flattened.std(dim=0, unbiased=False).clamp_min(1e-5)
    diagnostics = {
        "enabled": True,
        "valid_token_count": int(mask.sum().item()),
        "total_token_count": int(mask.numel()),
        "all_zero_sample_rate": float((~mask.any(dim=1)).float().mean().item()),
        "mean_abs_feature_mean": float(mean.abs().mean().item()),
        "median_feature_std": float(std.median().item()),
        "min_feature_std": float(std.min().item()),
        "max_feature_std": float(std.max().item()),
    }
    return mean, std, diagnostics


def dataset_diagnostics(dataloaders, modalities: Sequence[str]) -> Dict[str, object]:
    result: Dict[str, object] = {}
    for split_name, dataloader in dataloaders.items():
        dataset = dataloader.dataset
        labels = np.asarray(dataset.labels["M"], dtype=np.float64).reshape(-1)
        split = {
            "sample_count": int(len(labels)),
            "label_min": float(labels.min()),
            "label_max": float(labels.max()),
            "label_mean": float(labels.mean()),
            "label_std": float(labels.std()),
            "modalities": {},
        }
        for modality in modalities:
            if modality == "text" and bool(getattr(dataset.args, "use_bert", False)):
                text = torch.as_tensor(dataset.text)
                mask = text[:, 1, :].bool()
                finite_rate = 1.0
                feature_dim = int(dataset.args.feature_dims[0])
                nan_count = 0
                inf_count = 0
            else:
                array = getattr(dataset, modality if modality != "vision" else "vision")
                value = torch.as_tensor(array, dtype=torch.float32)
                mask = torch.isfinite(value).all(dim=-1) & (value.abs().sum(dim=-1) > 0)
                finite_rate = float(torch.isfinite(value).float().mean().item())
                feature_dim = int(value.shape[-1])
                nan_count = int(torch.isnan(value).sum().item())
                inf_count = int(torch.isinf(value).sum().item())
            lengths = mask.sum(dim=1).float()
            split["modalities"][modality] = {
                "feature_dim": feature_dim,
                "length_min": float(lengths.min().item()),
                "length_mean": float(lengths.mean().item()),
                "length_max": float(lengths.max().item()),
                "all_zero_sample_rate": float((lengths == 0).float().mean().item()),
                "finite_value_rate": finite_rate,
                "nan_count": nan_count,
                "inf_count": inf_count,
            }
        result[split_name] = split
    return result


class UnimodalExpertTrainerV8:
    def __init__(
        self,
        args,
        metrics_fn,
        modality: str,
        save_dir,
        prediction_epochs: int = 15,
        uncertainty_epochs: int = 6,
        joint_epochs: int = 6,
        prediction_patience: int = 6,
        joint_patience: int = 4,
        learning_rate: float = 1e-4,
        text_learning_rate: float = 2e-5,
        uncertainty_learning_rate: float = 3e-4,
        weight_decay: float = 1e-2,
        uncertainty_weight: float = 0.10,
        grad_clip: float = 1.0,
        max_prediction_degradation: float = 0.005,
        max_corr_degradation: float = 0.005,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.modality = modality
        self.batch_key = modality if modality != "vision" else "vision"
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.prediction_epochs = int(prediction_epochs)
        self.uncertainty_epochs = int(uncertainty_epochs)
        self.joint_epochs = int(joint_epochs)
        self.prediction_patience = int(prediction_patience)
        self.joint_patience = int(joint_patience)
        self.learning_rate = float(learning_rate)
        self.text_learning_rate = float(text_learning_rate)
        self.uncertainty_learning_rate = float(uncertainty_learning_rate)
        self.weight_decay = float(weight_decay)
        self.uncertainty_weight = float(uncertainty_weight)
        self.grad_clip = float(grad_clip)
        self.max_prediction_degradation = float(max_prediction_degradation)
        self.max_corr_degradation = float(max_corr_degradation)

    def _optimizer(self, model, stage: str):
        model.configure_stage(stage)
        trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        if not trainable:
            raise RuntimeError(f"No trainable parameters for stage={stage}")
        if stage == "uncertainty":
            return optim.AdamW(
                [parameter for _, parameter in trainable],
                lr=self.uncertainty_learning_rate,
                weight_decay=self.weight_decay,
            )

        text_parameters = [
            parameter for name, parameter in trainable
            if name.startswith("text_model.")
        ]
        other_parameters = [
            parameter for name, parameter in trainable
            if not name.startswith("text_model.")
        ]
        groups = []
        if other_parameters:
            groups.append({"params": other_parameters, "lr": self.learning_rate})
        if text_parameters:
            groups.append({"params": text_parameters, "lr": self.text_learning_rate})
        return optim.AdamW(groups, weight_decay=self.weight_decay)

    @staticmethod
    def _metric_dict(metrics_fn, prediction, labels):
        return {
            key: float(value)
            for key, value in metrics_fn(
                prediction.detach().cpu(), labels.detach().cpu()
            ).items()
        }

    @torch.no_grad()
    def collect(self, model, dataloader):
        model.eval()
        predictions = []
        uncertainties = []
        labels = []
        sample_ids = []
        missing = []
        for batch in tqdm(dataloader, leave=False):
            output = model(batch[self.batch_key].to(self.args.device))
            predictions.append(output["prediction"].detach().cpu())
            uncertainties.append(output["uncertainty"].detach().cpu())
            missing.append(output["all_missing"].detach().cpu())
            labels.append(batch["labels"]["M"].view(-1, 1).cpu())
            sample_ids.extend(normalize_batch_ids(batch.get("id")))
        return {
            "prediction": torch.cat(predictions, dim=0),
            "uncertainty": torch.cat(uncertainties, dim=0),
            "labels": torch.cat(labels, dim=0),
            "all_missing": torch.cat(missing, dim=0),
            "sample_ids": sample_ids,
        }

    def evaluate(self, model, dataloader) -> Dict[str, object]:
        collected = self.collect(model, dataloader)
        metrics = self._metric_dict(
            self.metrics_fn, collected["prediction"], collected["labels"]
        )
        uncertainty = uncertainty_diagnostics(
            collected["prediction"],
            collected["uncertainty"],
            collected["labels"],
            float(model.error_scale.item()),
        )
        return {
            "metrics": metrics,
            "uncertainty": uncertainty,
            "missing_sample_rate": float(
                collected["all_missing"].float().mean().item()
            ),
            "collected": collected,
        }

    @staticmethod
    def _prediction_is_better(candidate, best, tie_tolerance=0.003):
        if best is None:
            return True
        candidate_mae = _safe_float(candidate["MAE"], float("inf"))
        best_mae = _safe_float(best["MAE"], float("inf"))
        candidate_corr = _safe_float(candidate["Corr"], -float("inf"))
        best_corr = _safe_float(best["Corr"], -float("inf"))
        if candidate_mae < best_mae - float(tie_tolerance):
            return True
        if abs(candidate_mae - best_mae) <= float(tie_tolerance):
            if candidate_corr > best_corr + 1e-8:
                return True
            if abs(candidate_corr - best_corr) <= 1e-8 and candidate_mae < best_mae:
                return True
        return False

    def _train_epoch(self, model, dataloader, optimizer, stage: str):
        model.set_stage_mode(stage)
        totals = {"prediction": 0.0, "uncertainty": 0.0, "total": 0.0}
        count = 0
        for batch in tqdm(dataloader, leave=False):
            value = batch[self.batch_key].to(self.args.device)
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            output = model(value)
            prediction_loss = F.mse_loss(output["prediction"], labels)
            target = model.error_target(output["prediction"], labels)
            uncertainty_loss = F.smooth_l1_loss(output["uncertainty"], target)
            if stage == "prediction":
                total = prediction_loss
            elif stage == "uncertainty":
                total = uncertainty_loss
            elif stage == "joint":
                total = prediction_loss + self.uncertainty_weight * uncertainty_loss
            else:
                raise ValueError(stage)

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                self.grad_clip,
            )
            optimizer.step()
            totals["prediction"] += float(prediction_loss.detach().item())
            totals["uncertainty"] += float(uncertainty_loss.detach().item())
            totals["total"] += float(total.detach().item())
            count += 1
        return {key: value / max(1, count) for key, value in totals.items()}

    def _fit_prediction_stage(self, model, dataloaders):
        optimizer = self._optimizer(model, "prediction")
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-6
        )
        history = []
        best = None
        last_improvement = 0
        for epoch in range(1, self.prediction_epochs + 1):
            losses = self._train_epoch(
                model, dataloaders["train"], optimizer, "prediction"
            )
            valid = self.evaluate(model, dataloaders["valid"])
            metrics = valid["metrics"]
            scheduler.step(metrics["MAE"])
            row = {
                "stage": "prediction",
                "epoch": epoch,
                **{f"train_{key}": value for key, value in losses.items()},
                **{f"valid_{key}": value for key, value in metrics.items()},
            }
            history.append(row)
            LOGGER.info(
                "V8 %s Stage-A epoch=%d Valid MAE=%.4f Corr=%.4f",
                self.modality,
                epoch,
                metrics["MAE"],
                metrics["Corr"],
            )
            if self._prediction_is_better(metrics, None if best is None else best["metrics"]):
                best = {
                    "epoch": epoch,
                    "metrics": metrics,
                    "state": _cpu_state_dict(model),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.prediction_patience:
                break
        if best is None:
            raise RuntimeError("Prediction stage produced no checkpoint.")
        model.load_state_dict(best["state"])
        torch.save(best, self.save_dir / "prediction_stage_best.pth")
        return best, history

    def _fit_error_scale(self, model, train_loader) -> float:
        collected = self.collect(model, train_loader)
        errors = torch.abs(collected["prediction"] - collected["labels"]).view(-1)
        scale = float(torch.quantile(errors, 0.75).item())
        model.set_error_scale(scale)
        return scale

    def _fit_uncertainty_stage(self, model, dataloaders):
        optimizer = self._optimizer(model, "uncertainty")
        history = []
        best = None
        for epoch in range(1, self.uncertainty_epochs + 1):
            losses = self._train_epoch(
                model, dataloaders["train"], optimizer, "uncertainty"
            )
            valid = self.evaluate(model, dataloaders["valid"])
            diagnostic = valid["uncertainty"]
            spearman = _safe_float(diagnostic["error_spearman"], -float("inf"))
            row = {
                "stage": "uncertainty",
                "epoch": epoch,
                **{f"train_{key}": value for key, value in losses.items()},
                "valid_error_spearman": diagnostic["error_spearman"],
                "valid_high_error_auroc": diagnostic["high_error_auroc"],
                "valid_uncertainty_loss": diagnostic["uncertainty_loss"],
            }
            history.append(row)
            LOGGER.info(
                "V8 %s Stage-B epoch=%d Spearman=%.4f AUROC=%.4f",
                self.modality,
                epoch,
                diagnostic["error_spearman"],
                diagnostic["high_error_auroc"],
            )
            candidate_key = (
                spearman,
                -_safe_float(diagnostic["uncertainty_loss"], float("inf")),
            )
            if best is None or candidate_key > best["key"]:
                best = {
                    "epoch": epoch,
                    "key": candidate_key,
                    "diagnostics": diagnostic,
                    "state": _cpu_state_dict(model),
                }
        if best is None:
            raise RuntimeError("Uncertainty stage produced no checkpoint.")
        model.load_state_dict(best["state"])
        torch.save(best, self.save_dir / "uncertainty_stage_best.pth")
        return best, history

    def _fit_joint_stage(self, model, dataloaders):
        optimizer = self._optimizer(model, "joint")
        initial = self.evaluate(model, dataloaders["valid"])
        best = {
            "epoch": 0,
            "metrics": initial["metrics"],
            "uncertainty": initial["uncertainty"],
            "state": _cpu_state_dict(model),
        }
        history = []
        last_improvement = 0
        for epoch in range(1, self.joint_epochs + 1):
            losses = self._train_epoch(
                model, dataloaders["train"], optimizer, "joint"
            )
            valid = self.evaluate(model, dataloaders["valid"])
            metrics = valid["metrics"]
            history.append({
                "stage": "joint",
                "epoch": epoch,
                **{f"train_{key}": value for key, value in losses.items()},
                **{f"valid_{key}": value for key, value in metrics.items()},
                "valid_error_spearman": valid["uncertainty"]["error_spearman"],
                "valid_high_error_auroc": valid["uncertainty"]["high_error_auroc"],
            })
            LOGGER.info(
                "V8 %s Stage-C epoch=%d Valid MAE=%.4f Corr=%.4f Spearman=%.4f",
                self.modality,
                epoch,
                metrics["MAE"],
                metrics["Corr"],
                valid["uncertainty"]["error_spearman"],
            )
            if self._prediction_is_better(metrics, best["metrics"]):
                best = {
                    "epoch": epoch,
                    "metrics": metrics,
                    "uncertainty": valid["uncertainty"],
                    "state": _cpu_state_dict(model),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.joint_patience:
                break
        model.load_state_dict(best["state"])
        torch.save(best, self.save_dir / "joint_stage_best.pth")
        return best, history

    def _acceptance(self, test_result) -> Dict[str, object]:
        threshold = ACCEPTANCE_THRESHOLDS[self.modality]
        metrics = test_result["metrics"]
        uncertainty = test_result["uncertainty"]
        checks = {
            "mae": bool(metrics["MAE"] <= threshold["mae"]),
            "corr": bool(metrics["Corr"] >= threshold["corr"]),
            "error_spearman": bool(
                _safe_float(uncertainty["error_spearman"], -float("inf"))
                >= threshold["spearman"]
            ),
            "high_error_auroc": bool(
                _safe_float(uncertainty["high_error_auroc"], -float("inf"))
                >= threshold["auroc"]
            ),
            "quartile_monotonic": bool(uncertainty["quartile_monotonic"]),
        }
        return {
            "thresholds": threshold,
            "checks": checks,
            "all_pass": bool(all(checks.values())),
        }

    def train_and_evaluate(self, model, dataloaders, normalizer_info=None):
        stage_a, history_a = self._fit_prediction_stage(model, dataloaders)
        model.load_state_dict(stage_a["state"])
        error_scale = self._fit_error_scale(model, dataloaders["train"])
        stage_b, history_b = self._fit_uncertainty_stage(model, dataloaders)
        model.load_state_dict(stage_b["state"])
        stage_b_valid = self.evaluate(model, dataloaders["valid"])

        stage_c, history_c = self._fit_joint_stage(model, dataloaders)
        candidate_valid = stage_c["metrics"]
        reference_valid = stage_a["metrics"]
        joint_accepted = bool(
            candidate_valid["MAE"]
            <= reference_valid["MAE"] + self.max_prediction_degradation
            and candidate_valid["Corr"]
            >= reference_valid["Corr"] - self.max_corr_degradation
        )
        if joint_accepted:
            final_state = stage_c["state"]
            selected_stage = "joint"
        else:
            final_state = stage_b["state"]
            selected_stage = "uncertainty_head_only"
        model.load_state_dict(final_state)

        valid_result = self.evaluate(model, dataloaders["valid"])
        test_result = self.evaluate(model, dataloaders["test"])
        train_result = self.evaluate(model, dataloaders["train"])
        acceptance = self._acceptance(test_result)

        history = history_a + history_b + history_c
        pd.DataFrame(history).to_csv(
            self.save_dir / "unimodal_expert_v8_history.csv", index=False
        )
        collected = test_result["collected"]
        pd.DataFrame({
            "sample_id": collected["sample_ids"],
            "label": collected["labels"].view(-1).tolist(),
            "prediction": collected["prediction"].view(-1).tolist(),
            "uncertainty": collected["uncertainty"].view(-1).tolist(),
            "absolute_error": torch.abs(
                collected["prediction"] - collected["labels"]
            ).view(-1).tolist(),
            "all_missing": collected["all_missing"].view(-1).tolist(),
        }).to_csv(self.save_dir / "unimodal_expert_v8_test_predictions.csv", index=False)

        package = {
            "state": _cpu_state_dict(model),
            "modality": self.modality,
            "selected_stage": selected_stage,
            "error_scale": error_scale,
            "normalizer_info": normalizer_info or {},
            "stage_a_valid_metrics": stage_a["metrics"],
            "stage_b_valid_uncertainty": stage_b_valid["uncertainty"],
            "final_valid_metrics": valid_result["metrics"],
        }
        torch.save(package, self.save_dir / "unimodal_expert_v8_best.pth")

        summary = {
            "method": "staged_unimodal_expert_v8",
            "modality": self.modality,
            "protocol": (
                "Stage A trains sentiment prediction only. Stage B freezes the "
                "encoder/predictor and fits an error head using a train-residual "
                "Q75 scale. Stage C performs low-weight joint fine-tuning. "
                "Checkpoints are selected only on Valid; Test is evaluated once."
            ),
            "selected_stage": selected_stage,
            "joint_accepted": joint_accepted,
            "error_scale_q75_train": error_scale,
            "normalizer": normalizer_info or {},
            "stage_a": {
                "selected_epoch": stage_a["epoch"],
                "valid_metrics": stage_a["metrics"],
            },
            "stage_b": {
                "selected_epoch": stage_b["epoch"],
                "valid_uncertainty": stage_b_valid["uncertainty"],
            },
            "stage_c": {
                "selected_epoch": stage_c["epoch"],
                "valid_metrics": stage_c["metrics"],
            },
            "train": {
                "metrics": train_result["metrics"],
                "uncertainty": train_result["uncertainty"],
                "missing_sample_rate": train_result["missing_sample_rate"],
            },
            "valid": {
                "metrics": valid_result["metrics"],
                "uncertainty": valid_result["uncertainty"],
                "missing_sample_rate": valid_result["missing_sample_rate"],
            },
            "test": {
                "metrics": test_result["metrics"],
                "uncertainty": test_result["uncertainty"],
                "missing_sample_rate": test_result["missing_sample_rate"],
            },
            "acceptance": acceptance,
            "prediction_head_damage_guard": {
                "max_mae_degradation": self.max_prediction_degradation,
                "max_corr_degradation": self.max_corr_degradation,
                "stage_a_valid_metrics": reference_valid,
                "joint_candidate_valid_metrics": candidate_valid,
            },
        }
        (self.save_dir / "unimodal_expert_v8_summary.json").write_text(
            json.dumps(summary, indent=2, allow_nan=True), encoding="utf-8"
        )
        LOGGER.info(
            "V8 %s final stage=%s Test MAE=%.4f Corr=%.4f Acc7=%.4f "
            "Acc5=%.4f Acc2=%.4f F1=%.4f Spearman=%.4f AUROC=%.4f",
            self.modality,
            selected_stage,
            test_result["metrics"]["MAE"],
            test_result["metrics"]["Corr"],
            test_result["metrics"]["acc_7"],
            test_result["metrics"]["acc_5"],
            test_result["metrics"]["acc_2"],
            test_result["metrics"]["F1_score"],
            test_result["uncertainty"]["error_spearman"],
            test_result["uncertainty"]["high_error_auroc"],
        )
        return summary
