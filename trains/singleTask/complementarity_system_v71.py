"""Training and evaluation for complementarity-preserving distillation V7.1."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids
from .complementarity_v71 import (
    REGION_NAMES,
    apply_global_committee,
    apply_region_committee,
    balanced_sign_accuracy,
    committee_dispersion,
    complementarity_distillation_loss,
    fit_committee_cv,
    region_index,
    safe_corr,
    selection_stats,
)


logger = logging.getLogger("MMSA")


def _safe_metrics(metrics_fn, prediction, labels):
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _cache_index(sample_ids):
    mapping = {}
    for index, value in enumerate(sample_ids):
        key = str(value)
        if key in mapping:
            raise RuntimeError(f"Duplicate sample id in teacher cache: {key}")
        mapping[key] = index
    return mapping


def _batch_cache_indices(batch_ids, mapping):
    values = normalize_batch_ids(batch_ids)
    try:
        return torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    except KeyError as error:
        raise KeyError(f"Sample id missing from teacher cache: {error}") from error


class ComplementarityTrainerV71:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths: Sequence[Path],
        init_index: int,
        region_temperature: float = 0.55,
        committee_steps: int = 600,
        max_epochs: int = 20,
        early_stop: int = 6,
        head_lr: float = 3e-4,
        weight_decay: float = 1e-3,
        residual_clip: float = 0.35,
        confidence_temperature: float = 0.12,
        loss_weights=None,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache = teacher_cache
        self.teacher_paths = [Path(value) for value in teacher_paths]
        self.init_index = int(init_index)
        self.region_temperature = float(region_temperature)
        self.committee_steps = int(committee_steps)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.head_lr = float(head_lr)
        self.weight_decay = float(weight_decay)
        self.residual_clip = float(residual_clip)
        self.confidence_temperature = float(confidence_temperature)
        self.loss_weights = loss_weights or {
            "supervised": 1.00,
            "base_supervised": 0.10,
            "residual_distill": 0.75,
            "pairwise": 0.15,
            "ordinal": 0.10,
            "region": 0.25,
            "bias": 0.08,
            "uncertainty": 0.04,
            "disagreement_shrink": 0.08,
            "correction_shrink": 0.01,
        }
        self.index = {
            split: _cache_index(self.cache["splits"][split]["sample_ids"])
            for split in ("train", "valid", "test")
        }
        self.committee = self._fit_committee()

    def _fit_committee(self):
        valid = self.cache["splits"]["valid"]
        anchor = valid["predictions"][:, self.init_index]
        fitted = fit_committee_cv(
            valid["predictions"],
            valid["labels"],
            anchor,
            valid["sample_ids"],
            temperature=self.region_temperature,
            steps=self.committee_steps,
        )
        pd.DataFrame(fitted["cv_rows"]).to_csv(
            self.save_dir / "v71_committee_cv.csv", index=False
        )
        payload = {
            "selected": fitted["selected"],
            "selected_regularization": fitted["selected_regularization"],
            "global_cv_score": fitted["global_cv_score"],
            "region_cv_score": fitted["region_cv_score"],
            "global_weights": fitted["global_weights"].tolist(),
            "region_weights": fitted["region_weights"].tolist(),
            "region_temperature": self.region_temperature,
        }
        (self.save_dir / "v71_committee_weights.json").write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )
        logger.info(
            "V7.1 committee selected=%s global_cv=%.6f region_cv=%.6f reg=%.4f",
            fitted["selected"],
            fitted["global_cv_score"],
            fitted["region_cv_score"],
            fitted["selected_regularization"],
        )
        return fitted

    def _committee_predictions(self, split_name: str) -> Dict[str, torch.Tensor]:
        split = self.cache["splits"][split_name]
        predictions = split["predictions"].float()
        anchor = predictions[:, self.init_index]
        uniform = predictions.mean(dim=1)
        global_simplex = apply_global_committee(
            predictions, self.committee["global_weights"]
        )
        region_simplex = apply_region_committee(
            predictions,
            anchor,
            self.committee["region_weights"],
            self.region_temperature,
        )
        selected = (
            region_simplex
            if self.committee["selected"] == "region_simplex"
            else global_simplex
        )
        return {
            "uniform": uniform,
            "global_simplex": global_simplex,
            "region_simplex": region_simplex,
            "selected": selected,
            "anchor": anchor,
            "dispersion": committee_dispersion(predictions),
        }

    def _optimizer(self, model):
        parameters = [
            parameter
            for module in (
                model.adapter,
                model.global_head,
                model.region_heads,
                model.log_scale_head,
                model.ordinal_head,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        return optim.AdamW(
            parameters, lr=self.head_lr, weight_decay=self.weight_decay
        )

    def _train_epoch(self, model, dataloader, optimizer):
        model.set_train_mode()
        split = self.cache["splits"]["train"]
        totals = {}
        accumulation = max(1, int(getattr(self.args, "update_epochs", 1)))
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, leave=False), start=1):
            cpu_indices = _batch_cache_indices(
                batch.get("id"), self.index["train"]
            )
            teacher_predictions = split["predictions"][cpu_indices].to(
                self.args.device
            )
            anchor = teacher_predictions[:, self.init_index]
            if self.committee["selected"] == "region_simplex":
                teacher = apply_region_committee(
                    teacher_predictions,
                    anchor,
                    self.committee["region_weights"].to(self.args.device),
                    self.region_temperature,
                )
            else:
                teacher = apply_global_committee(
                    teacher_predictions,
                    self.committee["global_weights"].to(self.args.device),
                )
            dispersion = committee_dispersion(teacher_predictions)

            text = batch["text"].to(self.args.device)
            audio = batch["audio"].to(self.args.device)
            vision = batch["vision"].to(self.args.device)
            labels = batch["labels"]["M"].to(
                self.args.device
            ).view(-1, 1)

            student = model(text, audio, vision)
            losses = complementarity_distillation_loss(
                student,
                teacher,
                dispersion,
                labels,
                self.loss_weights,
                residual_clip=self.residual_clip,
                confidence_temperature=self.confidence_temperature,
            )
            (losses["total"] / accumulation).backward()
            if step % accumulation == 0 or step == len(dataloader):
                nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for parameter in model.parameters()
                        if parameter.requires_grad
                    ],
                    max_norm=2.0,
                )
                optimizer.step()
                optimizer.zero_grad()

            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().item()
                )

        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    @torch.no_grad()
    def collect(self, model, dataloader, split_name):
        """Collect Student outputs directly into teacher-cache order."""
        model.eval()
        mapping = self.index[split_name]
        split = self.cache["splits"][split_name]
        count = len(split["sample_ids"])
        buffers = None
        seen = torch.zeros(count, dtype=torch.bool)

        for batch in tqdm(dataloader, leave=False):
            indices = _batch_cache_indices(batch.get("id"), mapping)
            text = batch["text"].to(self.args.device)
            audio = batch["audio"].to(self.args.device)
            vision = batch["vision"].to(self.args.device)
            student = model(text, audio, vision)

            keys = (
                "base_prediction",
                "prediction",
                "correction",
                "log_scale",
                "feature",
                "region_probs",
            )
            if buffers is None:
                buffers = {}
                for key in keys:
                    value = student[key].detach().cpu()
                    buffers[key] = torch.empty(
                        (count,) + tuple(value.shape[1:]),
                        dtype=value.dtype,
                    )
            for key in keys:
                buffers[key][indices] = student[key].detach().cpu()
            seen[indices] = True

        if buffers is None or not bool(seen.all()):
            missing = int((~seen).sum().item())
            raise RuntimeError(
                f"Incomplete aligned Student collection for {split_name}: "
                f"missing={missing}"
            )

        return {
            "sample_ids": list(split["sample_ids"]),
            "labels": split["labels"].clone(),
            **buffers,
        }

    def _calibrate_student(self, collected):
        rows = []
        best = None
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            prediction = collected["base_prediction"] + float(alpha) * (
                collected["prediction"] - collected["base_prediction"]
            )
            stats = selection_stats(
                collected["base_prediction"],
                prediction,
                collected["labels"],
            )
            objective = (
                stats["mae"] + 0.05 * stats["harm_over_010_rate"]
            )
            row = {"alpha": alpha, "objective": objective, **stats}
            rows.append(row)
            if best is None or (
                row["objective"], row["mae"]
            ) < (best["objective"], best["mae"]):
                best = row
        return best, rows

    def train(self, model, dataloaders):
        model.freeze_backbone()
        optimizer = self._optimizer(model)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
        )
        history = []
        best = {
            "value": float("inf"),
            "epoch": 0,
            "state": None,
            "alpha": 0.0,
        }
        last_improvement = 0
        valid_committee = self._committee_predictions("valid")

        for epoch in range(1, self.max_epochs + 1):
            train_row = self._train_epoch(
                model, dataloaders["train"], optimizer
            )
            valid = self.collect(model, dataloaders["valid"], "valid")
            policy, _ = self._calibrate_student(valid)
            scheduler.step(policy["objective"])

            residual_prediction = (
                valid["prediction"] - valid["base_prediction"]
            )
            residual_target = (
                valid["labels"] - valid["base_prediction"]
            )
            row = {
                "epoch": epoch,
                **train_row,
                "valid_objective": float(policy["objective"]),
                "valid_mae": float(policy["mae"]),
                "valid_alpha": float(policy["alpha"]),
                "valid_base_mae": float(
                    torch.abs(
                        valid["base_prediction"] - valid["labels"]
                    ).mean().item()
                ),
                "valid_teacher_mae": float(
                    torch.abs(
                        valid_committee["selected"] - valid["labels"]
                    ).mean().item()
                ),
                "valid_global_simplex_mae": float(
                    torch.abs(
                        valid_committee["global_simplex"]
                        - valid["labels"]
                    ).mean().item()
                ),
                "valid_region_simplex_mae": float(
                    torch.abs(
                        valid_committee["region_simplex"]
                        - valid["labels"]
                    ).mean().item()
                ),
                "valid_residual_corr": safe_corr(
                    residual_prediction, residual_target
                ),
                "valid_sign_balanced_accuracy": balanced_sign_accuracy(
                    residual_prediction, residual_target
                ),
            }
            history.append(row)
            logger.info(
                "V7.1 epoch=%d train=%.4f valid=%.4f alpha=%.2f "
                "base=%.4f teacher=%.4f res_corr=%.4f",
                epoch,
                train_row.get("total", float("nan")),
                row["valid_mae"],
                row["valid_alpha"],
                row["valid_base_mae"],
                row["valid_teacher_mae"],
                row["valid_residual_corr"],
            )
            if policy["objective"] < best["value"] - 1e-6:
                best = {
                    "value": float(policy["objective"]),
                    "epoch": epoch,
                    "state": _cpu_state_dict(model),
                    "alpha": float(policy["alpha"]),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.early_stop:
                break

        if best["state"] is None:
            raise RuntimeError("V7.1 failed to select a validation checkpoint.")
        pd.DataFrame(history).to_csv(
            self.save_dir / "v71_training_history.csv", index=False
        )
        torch.save(
            best, self.save_dir / "complementarity_v71_best.pth"
        )
        return best

    def _region_rows(self, name, anchor, prediction, labels):
        regions = region_index(labels)
        rows = []
        for index, region_name in enumerate(REGION_NAMES):
            mask = regions == index
            if mask.any():
                rows.append({
                    "model": name,
                    "region": region_name,
                    "count": int(mask.sum().item()),
                    "anchor_mae": float(
                        torch.abs(anchor[mask] - labels[mask]).mean().item()
                    ),
                    "final_mae": float(
                        torch.abs(prediction[mask] - labels[mask]).mean().item()
                    ),
                    "anchor_bias": float(
                        (labels[mask] - anchor[mask]).mean().item()
                    ),
                    "final_bias": float(
                        (labels[mask] - prediction[mask]).mean().item()
                    ),
                })
        return rows

    def evaluate_and_save(self, model, dataloaders, best):
        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        test = self.collect(model, dataloaders["test"], "test")
        valid_committees = self._committee_predictions("valid")
        test_committees = self._committee_predictions("test")

        student_policy, student_rows = self._calibrate_student(valid)
        alpha = float(student_policy["alpha"])
        valid_student = valid["base_prediction"] + alpha * (
            valid["prediction"] - valid["base_prediction"]
        )
        test_student = test["base_prediction"] + alpha * (
            test["prediction"] - test["base_prediction"]
        )

        hybrid_rows = []
        best_hybrid = None
        for committee_name in (
            "uniform", "global_simplex", "region_simplex", "selected"
        ):
            for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
                prediction = (
                    float(beta) * valid_committees[committee_name]
                    + (1.0 - float(beta)) * valid_student
                )
                stats = selection_stats(
                    valid["base_prediction"],
                    prediction,
                    valid["labels"],
                )
                objective = (
                    stats["mae"] + 0.02 * stats["harm_over_010_rate"]
                )
                row = {
                    "committee": committee_name,
                    "beta": beta,
                    "objective": objective,
                    **stats,
                }
                hybrid_rows.append(row)
                if best_hybrid is None or (
                    row["objective"], row["mae"]
                ) < (
                    best_hybrid["objective"],
                    best_hybrid["mae"],
                ):
                    best_hybrid = row

        beta = float(best_hybrid["beta"])
        committee_name = best_hybrid["committee"]
        test_hybrid = (
            beta * test_committees[committee_name]
            + (1.0 - beta) * test_student
        )

        named_predictions = {
            "student_base": test["base_prediction"],
            "student_raw": test["prediction"],
            "student_calibrated": test_student,
            "committee_uniform": test_committees["uniform"],
            "committee_global_simplex": test_committees["global_simplex"],
            "committee_region_simplex": test_committees["region_simplex"],
            "committee_cv_selected": test_committees["selected"],
            "hybrid_valid_selected": test_hybrid,
        }

        result_rows = []
        region_rows = []
        results = {}
        for name, prediction in named_predictions.items():
            metrics = _safe_metrics(
                self.metrics_fn, prediction, test["labels"]
            )
            stats = selection_stats(
                test["base_prediction"], prediction, test["labels"]
            )
            results[name] = {
                "metrics": metrics,
                "selection": stats,
            }
            result_rows.append({"model": name, **metrics, **stats})
            region_rows.extend(
                self._region_rows(
                    name,
                    test["base_prediction"],
                    prediction,
                    test["labels"],
                )
            )

        teacher_rows = []
        test_split = self.cache["splits"]["test"]
        for index, path in enumerate(self.teacher_paths):
            prediction = test_split["predictions"][:, index]
            teacher_rows.append({
                "teacher_index": index,
                "checkpoint": str(path),
                **_safe_metrics(
                    self.metrics_fn,
                    prediction,
                    test_split["labels"],
                ),
            })

        prediction_frame = {
            "sample_id": test["sample_ids"],
            "label": test["labels"].view(-1).tolist(),
        }
        for index in range(len(self.teacher_paths)):
            prediction_frame[f"teacher_{index}"] = (
                test_split["predictions"][:, index, 0].tolist()
            )
        for name, prediction in named_predictions.items():
            prediction_frame[name] = prediction.view(-1).tolist()

        pd.DataFrame(student_rows).to_csv(
            self.save_dir / "v71_student_policy_calibration.csv",
            index=False,
        )
        pd.DataFrame(hybrid_rows).to_csv(
            self.save_dir / "v71_hybrid_calibration.csv", index=False
        )
        pd.DataFrame(result_rows).to_csv(
            self.save_dir / "v71_test_baseline_comparison.csv",
            index=False,
        )
        pd.DataFrame(teacher_rows).to_csv(
            self.save_dir / "v71_test_teacher_metrics.csv", index=False
        )
        pd.DataFrame(region_rows).to_csv(
            self.save_dir / "v71_test_region_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(prediction_frame).to_csv(
            self.save_dir / "complementarity_v71_predictions.csv",
            index=False,
        )

        summary = {
            "method": "complementarity_preserving_function_distillation_v7_1",
            "seed": int(getattr(self.args, "seed", 0)),
            "teacher_count": len(self.teacher_paths),
            "teacher_paths": [str(value) for value in self.teacher_paths],
            "base_teacher_index": self.init_index,
            "selected_epoch": int(best["epoch"]),
            "selected_valid_objective": float(best["value"]),
            "student_policy": {"alpha": alpha},
            "committee": {
                "selected": self.committee["selected"],
                "global_weights": self.committee["global_weights"].tolist(),
                "region_weights": self.committee["region_weights"].tolist(),
                "selected_regularization": self.committee[
                    "selected_regularization"
                ],
                "global_cv_score": self.committee["global_cv_score"],
                "region_cv_score": self.committee["region_cv_score"],
            },
            "hybrid_policy": {
                "committee": committee_name,
                "beta": beta,
                "valid_mae": float(best_hybrid["mae"]),
                "valid_objective": float(best_hybrid["objective"]),
            },
            "results": results,
            "loss_weights": dict(self.loss_weights),
        }
        (self.save_dir / "complementarity_v71_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

        best_name = min(
            results,
            key=lambda name: results[name]["metrics"]["MAE"],
        )
        logger.info(
            "V7.1 TEST best=%s MAE=%.4f base=%.4f global=%.4f "
            "region=%.4f student=%.4f hybrid=%.4f",
            best_name,
            results[best_name]["metrics"]["MAE"],
            results["student_base"]["metrics"]["MAE"],
            results["committee_global_simplex"]["metrics"]["MAE"],
            results["committee_region_simplex"]["metrics"]["MAE"],
            results["student_calibrated"]["metrics"]["MAE"],
            results["hybrid_valid_selected"]["metrics"]["MAE"],
        )
        return summary
