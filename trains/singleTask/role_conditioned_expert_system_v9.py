"""Leakage-controlled training and evaluation for V9 specialized experts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids
from .role_conditioned_experts_v9 import (
    REGION_NAMES,
    LossWeights,
    apply_teacher_weights,
    region_index,
    region_metrics,
    role_conditioned_loss,
)
from .model.RoleConditionedDLF import RoleConditionedDLF


logger = logging.getLogger("MMSA")


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _cache_index(sample_ids: Sequence[object]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for index, value in enumerate(sample_ids):
        key = str(value)
        if key in mapping:
            raise RuntimeError("Duplicate sample id in V9 teacher cache: %s" % key)
        mapping[key] = index
    return mapping


def _batch_cache_indices(batch_ids, mapping: Mapping[str, int]) -> torch.Tensor:
    values = normalize_batch_ids(batch_ids)
    try:
        return torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    except KeyError as error:
        raise KeyError("V9 sample id missing from teacher cache: %s" % error)


def _safe_metrics(metrics_fn, prediction, labels) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    x = x.view(-1).double()
    y = y.view(-1).double()
    if x.numel() < 2:
        return 0.0
    x_order = torch.argsort(x, stable=True)
    y_order = torch.argsort(y, stable=True)
    x_rank = torch.empty_like(x_order, dtype=torch.double)
    y_rank = torch.empty_like(y_order, dtype=torch.double)
    x_rank[x_order] = torch.arange(x.numel(), dtype=torch.double)
    y_rank[y_order] = torch.arange(y.numel(), dtype=torch.double)
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denominator = torch.sqrt(
        x_rank.square().sum() * y_rank.square().sum()
    ).clamp_min(1e-12)
    return float((x_rank * y_rank).sum().div(denominator).item())


class RoleConditionedExpertTrainerV9:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths: Sequence[Path],
        anchor_index: int,
        teacher_fit: Mapping[str, object],
        hidden_dim: int = 192,
        dropout: float = 0.15,
        residual_max: float = 0.45,
        head_epochs: int = 4,
        tail_epochs: int = 12,
        non_bert_epochs: int = 0,
        early_stop: int = 5,
        head_lr: float = 3e-4,
        tail_lr: float = 2e-5,
        backbone_lr: float = 5e-6,
        weight_decay: float = 1e-3,
        membership_floor: float = 0.25,
        membership_sigma_scale: float = 1.0,
        gain_margin: float = 0.08,
        gain_fraction: float = 0.20,
        global_tolerance: float = 0.012,
        loss_weights: LossWeights = LossWeights(),
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache = teacher_cache
        self.teacher_paths = [Path(value) for value in teacher_paths]
        self.anchor_index = int(anchor_index)
        self.teacher_fit = teacher_fit
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual_max = float(residual_max)
        self.stage_epochs: List[Tuple[str, int]] = [
            ("heads", int(head_epochs)),
            ("tail", int(tail_epochs)),
        ]
        if int(non_bert_epochs) > 0:
            self.stage_epochs.append(("non_bert", int(non_bert_epochs)))
        self.early_stop = int(early_stop)
        self.head_lr = float(head_lr)
        self.tail_lr = float(tail_lr)
        self.backbone_lr = float(backbone_lr)
        self.weight_decay = float(weight_decay)
        self.membership_floor = float(membership_floor)
        self.membership_sigma_scale = float(membership_sigma_scale)
        self.gain_margin = float(gain_margin)
        self.gain_fraction = float(gain_fraction)
        self.global_tolerance = float(global_tolerance)
        self.loss_weights = loss_weights
        self.index = {
            split: _cache_index(self.cache["splits"][split]["sample_ids"])
            for split in ("train", "valid", "test")
        }

        self.global_teacher_weights = teacher_fit["global_weights"].float()
        self.role_teacher_weights = teacher_fit["role_weights"].float()
        if self.role_teacher_weights.shape != (
            len(REGION_NAMES),
            len(self.teacher_paths),
        ):
            raise ValueError("V9 role teacher weight matrix has an invalid shape.")

    def _new_model(self) -> RoleConditionedDLF:
        model = RoleConditionedDLF(
            self.args,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            residual_max=self.residual_max,
        ).to(self.args.device)
        incompatible = model.load_backbone_checkpoint(
            self.teacher_paths[self.anchor_index], map_location=self.args.device
        )
        logger.info(
            "Loaded V9 anchor (missing=%d unexpected=%d): %s",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            self.teacher_paths[self.anchor_index],
        )
        return model

    def _optimizer(self, model: RoleConditionedDLF) -> optim.Optimizer:
        groups = model.parameter_groups(
            head_lr=self.head_lr,
            tail_lr=self.tail_lr,
            backbone_lr=self.backbone_lr,
        )
        return optim.AdamW(groups, weight_decay=self.weight_decay)

    def _split_targets(self, split_name: str, role: int):
        split = self.cache["splits"][split_name]
        predictions = split["predictions"].float()
        anchor = predictions[:, self.anchor_index]
        global_teacher = apply_teacher_weights(
            predictions, self.global_teacher_weights
        )
        role_teacher = apply_teacher_weights(
            predictions, self.role_teacher_weights[int(role)]
        )
        return anchor, global_teacher, role_teacher

    def _train_epoch(
        self,
        model: RoleConditionedDLF,
        dataloader,
        optimizer,
        role: int,
    ) -> Dict[str, float]:
        model.set_train_mode()
        anchor_all, global_teacher_all, role_teacher_all = self._split_targets(
            "train", role
        )
        totals: Dict[str, float] = {}
        accumulation = max(1, int(getattr(self.args, "update_epochs", 1)))
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, leave=False), start=1):
            cpu_indices = _batch_cache_indices(batch.get("id"), self.index["train"])
            anchor = anchor_all[cpu_indices].to(self.args.device)
            global_teacher = global_teacher_all[cpu_indices].to(self.args.device)
            role_teacher = role_teacher_all[cpu_indices].to(self.args.device)
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            losses = role_conditioned_loss(
                output,
                labels,
                anchor,
                global_teacher,
                role_teacher,
                role=role,
                membership_floor=self.membership_floor,
                membership_sigma_scale=self.membership_sigma_scale,
                gain_margin=self.gain_margin,
                gain_fraction=self.gain_fraction,
                weights=self.loss_weights,
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
        denominator = max(1, len(dataloader))
        return {name: value / denominator for name, value in totals.items()}

    @torch.no_grad()
    def collect(self, model, dataloader, split_name: str) -> Dict[str, object]:
        model.eval()
        split = self.cache["splits"][split_name]
        mapping = self.index[split_name]
        sample_count = len(split["sample_ids"])
        keys = (
            "base_prediction",
            "prediction",
            "correction",
            "region_probs",
            "region_expected",
            "predicted_abs_error",
        )
        buffers = None
        seen = torch.zeros(sample_count, dtype=torch.bool)
        for batch in tqdm(dataloader, leave=False):
            indices = _batch_cache_indices(batch.get("id"), mapping)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            if buffers is None:
                buffers = {
                    key: torch.empty(
                        (sample_count,) + tuple(output[key].shape[1:]),
                        dtype=output[key].dtype,
                    )
                    for key in keys
                }
            for key in keys:
                buffers[key][indices] = output[key].detach().cpu()
            seen[indices] = True
        if buffers is None or not bool(seen.all()):
            raise RuntimeError(
                "Incomplete V9 collection for %s: missing=%d"
                % (split_name, int((~seen).sum().item()))
            )
        return {
            "sample_ids": list(split["sample_ids"]),
            "labels": split["labels"].clone(),
            **buffers,
        }

    def _validation_row(
        self,
        collected: Mapping[str, torch.Tensor],
        role: int,
    ) -> Dict[str, float]:
        labels = collected["labels"]
        prediction = collected["prediction"]
        anchor = collected["base_prediction"]
        masks = region_index(labels) == int(role)
        if not masks.any():
            raise RuntimeError("Validation split has no samples for role %d." % role)
        global_mae = float(torch.abs(prediction - labels).mean().item())
        role_mae = float(torch.abs(prediction[masks] - labels[masks]).mean().item())
        anchor_global = float(torch.abs(anchor - labels).mean().item())
        anchor_role = float(torch.abs(anchor[masks] - labels[masks]).mean().item())
        realized_gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
        harm_rate = float((realized_gain < -0.10).float().mean().item())
        global_excess = max(
            0.0, global_mae - anchor_global - self.global_tolerance
        )
        objective = (
            0.65 * role_mae
            + 0.35 * global_mae
            + 2.0 * global_excess
            + 0.05 * harm_rate
        )
        class_prediction = collected["region_probs"].argmax(dim=1)
        class_target = region_index(labels)
        region_accuracy = float(
            (class_prediction == class_target).float().mean().item()
        )
        risk_spearman = _spearman(
            collected["predicted_abs_error"], torch.abs(prediction - labels)
        )
        return {
            "objective": objective,
            "global_mae": global_mae,
            "role_mae": role_mae,
            "anchor_global_mae": anchor_global,
            "anchor_role_mae": anchor_role,
            "role_gain": anchor_role - role_mae,
            "global_delta": global_mae - anchor_global,
            "harm_over_010_rate": harm_rate,
            "region_accuracy": region_accuracy,
            "risk_spearman": risk_spearman,
            "mean_abs_correction": float(
                torch.abs(collected["correction"]).mean().item()
            ),
        }

    def train_role(self, role: int, dataloaders) -> Dict[str, object]:
        role_name = REGION_NAMES[int(role)]
        role_dir = self.save_dir / role_name
        role_dir.mkdir(parents=True, exist_ok=True)

        role_seed = int(getattr(self.args, "seed", 0))
        torch.manual_seed(role_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(role_seed)
        model = self._new_model()
        history: List[Dict[str, object]] = []
        best = None
        epoch_number = 0
        no_improvement = 0

        for stage, stage_epochs in self.stage_epochs:
            if stage_epochs <= 0:
                continue
            model.set_stage(stage)
            optimizer = self._optimizer(model)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
            )
            for _ in range(stage_epochs):
                epoch_number += 1
                sampler = getattr(dataloaders["train"], "sampler", None)
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch_number)
                train_row = self._train_epoch(
                    model, dataloaders["train"], optimizer, role
                )
                valid = self.collect(model, dataloaders["valid"], "valid")
                valid_row = self._validation_row(valid, role)
                scheduler.step(valid_row["objective"])
                row: Dict[str, object] = {
                    "epoch": epoch_number,
                    "stage": stage,
                    **{"train_" + key: value for key, value in train_row.items()},
                    **{"valid_" + key: value for key, value in valid_row.items()},
                }
                history.append(row)
                logger.info(
                    "V9 role=%s epoch=%d stage=%s objective=%.6f "
                    "role_mae=%.6f role_gain=%+.6f global_delta=%+.6f",
                    role_name,
                    epoch_number,
                    stage,
                    valid_row["objective"],
                    valid_row["role_mae"],
                    valid_row["role_gain"],
                    valid_row["global_delta"],
                )
                if best is None or valid_row["objective"] < best["objective"] - 1e-6:
                    best = {
                        "objective": float(valid_row["objective"]),
                        "epoch": int(epoch_number),
                        "stage": stage,
                        "state": _cpu_state_dict(model),
                        "valid": dict(valid_row),
                    }
                    no_improvement = 0
                else:
                    no_improvement += 1
                if no_improvement >= self.early_stop:
                    break
            if no_improvement >= self.early_stop:
                break

        if best is None:
            raise RuntimeError("V9 failed to select a checkpoint for %s." % role_name)
        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        test = self.collect(model, dataloaders["test"], "test")
        pd.DataFrame(history).to_csv(role_dir / "training_history.csv", index=False)
        torch.save(
            {
                "method": "role_conditioned_expert_v9",
                "role": role_name,
                "role_index": int(role),
                "anchor_checkpoint": str(self.teacher_paths[self.anchor_index]),
                "selected_epoch": int(best["epoch"]),
                "selected_stage": str(best["stage"]),
                "selected_valid": dict(best["valid"]),
                "state_dict": best["state"],
            },
            role_dir / "role_conditioned_expert_v9_best.pth",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "role": int(role),
            "role_name": role_name,
            "best": best,
            "valid": valid,
            "test": test,
        }

    @staticmethod
    def _stack_outputs(role_results, split_name: str):
        predictions = torch.stack(
            [value[split_name]["prediction"] for value in role_results], dim=1
        )
        probabilities = torch.stack(
            [value[split_name]["region_probs"] for value in role_results], dim=1
        )
        risks = torch.stack(
            [value[split_name]["predicted_abs_error"] for value in role_results],
            dim=1,
        )
        return predictions, probabilities, risks

    @staticmethod
    def _category_coach(
        predictions: torch.Tensor,
        probabilities: torch.Tensor,
        risks: torch.Tensor,
        temperature: float,
        risk_weight: float,
    ):
        diagonal_confidence = torch.stack(
            [probabilities[:, role, role] for role in range(len(REGION_NAMES))],
            dim=1,
        )
        score = torch.log(diagonal_confidence.clamp_min(1e-8))
        score = score - float(risk_weight) * risks.squeeze(-1)
        weights = torch.softmax(score / max(float(temperature), 1e-6), dim=1)
        prediction = (predictions * weights.unsqueeze(-1)).sum(dim=1)
        return prediction, weights

    def _calibrate_coach(self, role_results):
        valid_predictions, valid_probabilities, valid_risks = self._stack_outputs(
            role_results, "valid"
        )
        labels = role_results[0]["valid"]["labels"]
        anchor = role_results[0]["valid"]["base_prediction"]
        rows = []
        best = None
        for temperature in (0.25, 0.5, 0.75, 1.0, 1.5, 2.0):
            for risk_weight in (0.0, 0.25, 0.5, 1.0):
                specialist, _ = self._category_coach(
                    valid_predictions,
                    valid_probabilities,
                    valid_risks,
                    temperature,
                    risk_weight,
                )
                for beta in (0.25, 0.5, 0.75, 1.0):
                    prediction = (1.0 - beta) * anchor + beta * specialist
                    mae = float(torch.abs(prediction - labels).mean().item())
                    gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
                    harm = float((gain < -0.10).float().mean().item())
                    objective = mae + 0.03 * harm
                    row = {
                        "temperature": float(temperature),
                        "risk_weight": float(risk_weight),
                        "beta": float(beta),
                        "mae": mae,
                        "harm_over_010_rate": harm,
                        "objective": objective,
                    }
                    rows.append(row)
                    if best is None or (row["objective"], row["mae"]) < (
                        best["objective"],
                        best["mae"],
                    ):
                        best = row
        if best is None:
            raise RuntimeError("V9 coach calibration generated no candidates.")
        return best, rows

    def train_all(self, dataloaders) -> Dict[str, object]:
        role_results = [
            self.train_role(role, dataloaders)
            for role in range(len(REGION_NAMES))
        ]
        coach_policy, coach_rows = self._calibrate_coach(role_results)
        pd.DataFrame(coach_rows).to_csv(
            self.save_dir / "v9_category_coach_calibration.csv", index=False
        )
        summary = self._evaluate_and_save(role_results, coach_policy)
        return summary

    def _capability_rows(self, role_results, split_name: str):
        rows = []
        labels = role_results[0][split_name]["labels"]
        anchor = role_results[0][split_name]["base_prediction"]
        rows.append({"expert": "anchor", **region_metrics(anchor, labels)})
        for result in role_results:
            rows.append(
                {
                    "expert": result["role_name"],
                    "designated_role": result["role_name"],
                    **region_metrics(result[split_name]["prediction"], labels),
                }
            )
        return rows

    def _evaluate_and_save(self, role_results, coach_policy):
        valid_capability = self._capability_rows(role_results, "valid")
        test_capability = self._capability_rows(role_results, "test")
        pd.DataFrame(valid_capability).to_csv(
            self.save_dir / "v9_valid_capability_matrix.csv", index=False
        )
        pd.DataFrame(test_capability).to_csv(
            self.save_dir / "v9_test_capability_matrix.csv", index=False
        )

        test_predictions, test_probabilities, test_risks = self._stack_outputs(
            role_results, "test"
        )
        test_labels = role_results[0]["test"]["labels"]
        test_anchor = role_results[0]["test"]["base_prediction"]
        specialist, coach_weights = self._category_coach(
            test_predictions,
            test_probabilities,
            test_risks,
            coach_policy["temperature"],
            coach_policy["risk_weight"],
        )
        coach_prediction = (
            (1.0 - coach_policy["beta"]) * test_anchor
            + coach_policy["beta"] * specialist
        )

        true_regions = region_index(test_labels)
        designated = torch.empty_like(test_anchor)
        for role in range(len(REGION_NAMES)):
            mask = true_regions == role
            designated[mask] = test_predictions[mask, role]
        sample_errors = torch.abs(test_predictions - test_labels.unsqueeze(1))
        best_indices = sample_errors.squeeze(-1).argmin(dim=1)
        sample_oracle = test_predictions[
            torch.arange(test_predictions.size(0)), best_indices
        ]

        named_predictions = {
            "anchor": test_anchor,
            "uniform_specialists": test_predictions.mean(dim=1),
            "category_coach_valid_selected": coach_prediction,
            "true_region_designated_oracle": designated,
            "sample_oracle_upper_bound": sample_oracle,
        }
        result_rows = []
        result_payload = {}
        for name, prediction in named_predictions.items():
            metrics = _safe_metrics(self.metrics_fn, prediction, test_labels)
            metrics.update(region_metrics(prediction, test_labels))
            result_rows.append({"model": name, **metrics})
            result_payload[name] = metrics
        pd.DataFrame(result_rows).to_csv(
            self.save_dir / "v9_test_summary.csv", index=False
        )

        prediction_frame: Dict[str, object] = {
            "sample_id": role_results[0]["test"]["sample_ids"],
            "label": test_labels.view(-1).tolist(),
            "anchor": test_anchor.view(-1).tolist(),
            "category_coach": coach_prediction.view(-1).tolist(),
            "true_region_designated_oracle": designated.view(-1).tolist(),
            "sample_oracle_upper_bound": sample_oracle.view(-1).tolist(),
        }
        for role, role_name in enumerate(REGION_NAMES):
            result = role_results[role]["test"]
            prediction_frame[role_name + "_prediction"] = (
                result["prediction"].view(-1).tolist()
            )
            prediction_frame[role_name + "_predicted_risk"] = (
                result["predicted_abs_error"].view(-1).tolist()
            )
            prediction_frame[role_name + "_coach_weight"] = (
                coach_weights[:, role].view(-1).tolist()
            )
            for category, category_name in enumerate(REGION_NAMES):
                prediction_frame[
                    role_name + "_p_" + category_name
                ] = result["region_probs"][:, category].tolist()
        pd.DataFrame(prediction_frame).to_csv(
            self.save_dir / "role_conditioned_experts_v9_predictions.csv",
            index=False,
        )

        valid_wins = {}
        for role, role_name in enumerate(REGION_NAMES):
            column = role_name + "_mae"
            specialist_rows = valid_capability[1:]
            best_row = min(specialist_rows, key=lambda row: row[column])
            valid_wins[role_name] = {
                "best_expert": best_row["expert"],
                "designated_expert": role_name,
                "designated_is_best": best_row["expert"] == role_name,
                "designated_mae": next(
                    row[column]
                    for row in specialist_rows
                    if row["expert"] == role_name
                ),
                "best_mae": best_row[column],
            }

        summary = {
            "method": "role_conditioned_complementary_distillation_v9",
            "dataset": str(self.args.dataset_name),
            "seed": int(getattr(self.args, "seed", 0)),
            "anchor_checkpoint": str(self.teacher_paths[self.anchor_index]),
            "anchor_index": self.anchor_index,
            "role_initialization_seed": int(getattr(self.args, "seed", 0)),
            "teacher_paths": [str(value) for value in self.teacher_paths],
            "teacher_fit": {
                "global_weights": self.global_teacher_weights.tolist(),
                "role_weights": self.role_teacher_weights.tolist(),
                "selected_regularizations": self.teacher_fit[
                    "selected_regularizations"
                ],
                "region_counts": self.teacher_fit["region_counts"],
            },
            "roles": {
                result["role_name"]: {
                    "selected_epoch": int(result["best"]["epoch"]),
                    "selected_stage": str(result["best"]["stage"]),
                    "selected_valid": dict(result["best"]["valid"]),
                }
                for result in role_results
            },
            "coach_policy": dict(coach_policy),
            "valid_specialization_wins": valid_wins,
            "test_results": result_payload,
            "loss_weights": self.loss_weights.__dict__,
            "selection_protocol": (
                "All checkpoints, teacher weights, coach temperature, risk weight, "
                "and shrinkage beta are selected using train/validation only. "
                "Test labels are used solely for final reporting and oracle analysis."
            ),
        }
        (self.save_dir / "role_conditioned_experts_v9_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
