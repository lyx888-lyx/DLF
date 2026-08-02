"""Strict region-gradient consolidation for V9.24.

This module trains one frozen-anchor model with a single scalar output. It does
not instantiate experts, a router, a consensus head, or a student. The only
trainable parameters are the pre-registered V9.23 fusion-tail parameters.

Each epoch computes exact dataset-mean MAE gradients for the global training
set and the five fixed semantic regions. Three strategies are supported:

* ``global_mae``: the ordinary global MAE gradient;
* ``equal_region_mean``: the raw mean of the five region-mean gradients;
* ``normalized_mgda``: the V9.23 minimum-norm convex combination of unit region
  gradients, rescaled to the global-gradient norm.

All updates use plain SGD without momentum or weight decay so that the actual
parameter update remains a positive scalar multiple of the analyzed gradient
direction.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from .gradient_conflict_audit_v923 import (
    REGION_TASK_NAMES,
    Gradient,
    GradientAuditConfigV923,
    ParameterRecordV923,
    collect_mean_mae_gradients,
    fit_normalized_mgda,
    gradient_dot,
    gradient_mean,
    gradient_norm,
    gradient_scale,
    select_parameter_records,
    selected_parameter_fingerprint,
)
from .role_conditioned_experts_v9 import REGION_NAMES, region_index

TRAINING_VERSION = "region_gradient_consolidation_v924_v1"
STRATEGIES = (
    "global_mae",
    "equal_region_mean",
    "normalized_mgda",
)
PRIMARY_STRATEGY = "normalized_mgda"


@dataclass(frozen=True)
class RegionGradientTrainingConfigV924:
    parameter_scope: str = "fusion_tail"
    learning_rate: float = 1e-3
    max_epochs: int = 40
    early_stop: int = 8
    min_validation_improvement: float = 1e-5
    gradient_clip_norm: float = 1.0
    minimum_mgda_direction_norm: float = 0.02
    validation_global_weight: float = 0.50
    validation_macro_region_weight: float = 0.50
    mgda_max_iterations: int = 2000
    mgda_ftol: float = 1e-12

    def validate(self) -> None:
        if self.parameter_scope != "fusion_tail":
            raise ValueError("V9.24 is pre-registered for fusion_tail only")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.max_epochs < 1 or self.early_stop < 1:
            raise ValueError("epoch limits must be positive")
        if self.gradient_clip_norm <= 0.0:
            raise ValueError("gradient_clip_norm must be positive")
        if self.minimum_mgda_direction_norm <= 0.0:
            raise ValueError("minimum_mgda_direction_norm must be positive")
        total = self.validation_global_weight + self.validation_macro_region_weight
        if abs(total - 1.0) > 1e-8:
            raise ValueError("validation objective weights must sum to one")


@dataclass(frozen=True)
class SuccessGateConfigV924:
    required_gain_vs_global: float = 0.003
    required_positive_outer_folds: int = 4
    maximum_worst_fold_degradation: float = 0.005
    require_macro_region_improvement: bool = True
    require_strong_negative_nondegradation: bool = True


def freeze_to_parameter_scope(
    model: torch.nn.Module,
    scope: str = "fusion_tail",
) -> list[ParameterRecordV923]:
    """Freeze the complete model except the fixed V9.23 parameter scope."""
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    records = select_parameter_records(model, scope)
    selected = {id(record.parameter) for record in records}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in selected)
        parameter.grad = None
    model.eval()
    if not records:
        raise RuntimeError("empty V9.24 trainable parameter scope")
    return records


def trainable_parameter_names(
    records: Sequence[ParameterRecordV923],
) -> list[str]:
    return [record.name for record in records]


def _copy_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _region_name_tensor(labels: torch.Tensor) -> list[str]:
    indices = region_index(labels.view(-1).detach().cpu()).tolist()
    return [REGION_NAMES[int(index)] for index in indices]


def regression_metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    anchor: torch.Tensor | None = None,
) -> Dict[str, object]:
    prediction = prediction.detach().cpu().float().view(-1)
    labels = labels.detach().cpu().float().view(-1)
    if len(prediction) != len(labels):
        raise ValueError("prediction/label length mismatch")
    regions = region_index(labels)
    region_mae: Dict[str, float] = {}
    for index, name in enumerate(REGION_NAMES):
        mask = regions == int(index)
        if not bool(mask.any()):
            raise RuntimeError(f"metric split has no samples for region {name}")
        region_mae[name] = float(torch.abs(prediction[mask] - labels[mask]).mean())
    result: Dict[str, object] = {
        "sample_count": int(len(labels)),
        "mae": float(torch.abs(prediction - labels).mean()),
        "macro_region_mae": float(np.mean(list(region_mae.values()))),
        "region_mae": region_mae,
    }
    if anchor is not None:
        anchor = anchor.detach().cpu().float().view(-1)
        if len(anchor) != len(labels):
            raise ValueError("anchor/label length mismatch")
        anchor_error = torch.abs(anchor - labels)
        error = torch.abs(prediction - labels)
        gain = anchor_error - error
        result.update(
            {
                "anchor_mae": float(anchor_error.mean()),
                "gain_vs_anchor": float(gain.mean()),
                "win_rate": float((gain > 0.0).float().mean()),
                "large_gain_rate_010": float((gain > 0.10).float().mean()),
                "large_harm_rate_010": float((gain < -0.10).float().mean()),
            }
        )
    return result


def validation_objective(
    metrics: Mapping[str, object],
    config: RegionGradientTrainingConfigV924,
) -> float:
    return (
        float(config.validation_global_weight) * float(metrics["mae"])
        + float(config.validation_macro_region_weight)
        * float(metrics["macro_region_mae"])
    )


def evaluate_model(
    model: torch.nn.Module,
    loader: Iterable[object],
    forward_batch: Callable[
        [torch.nn.Module, object],
        tuple[torch.Tensor, torch.Tensor],
    ],
    sample_metadata: Callable[[object], Mapping[str, Sequence[object]]] | None = None,
) -> Dict[str, object]:
    model.eval()
    predictions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    metadata: Dict[str, list[object]] = {}
    with torch.no_grad():
        for batch in loader:
            prediction, target = forward_batch(model, batch)
            predictions.append(prediction.detach().cpu().float().view(-1))
            labels.append(target.detach().cpu().float().view(-1))
            if sample_metadata is not None:
                local = sample_metadata(batch)
                for key, values in local.items():
                    metadata.setdefault(str(key), []).extend(list(values))
    if not predictions:
        raise RuntimeError("evaluation loader is empty")
    prediction = torch.cat(predictions)
    target = torch.cat(labels)
    if metadata and any(len(values) != len(target) for values in metadata.values()):
        raise RuntimeError("evaluation metadata length mismatch")
    return {
        "prediction": prediction,
        "labels": target,
        "regions": _region_name_tensor(target),
        "metadata": metadata,
        "metrics": regression_metrics(prediction, target),
    }


def _scale_and_clip_direction(
    direction: Gradient,
    target_norm: float,
    clip_norm: float,
) -> tuple[Gradient, Dict[str, float]]:
    raw_norm = gradient_norm(direction)
    if raw_norm <= 0.0:
        raise RuntimeError("zero update direction")
    scaled = gradient_scale(direction, float(target_norm) / raw_norm)
    scaled_norm = gradient_norm(scaled)
    clip_scale = min(1.0, float(clip_norm) / max(scaled_norm, 1e-12))
    clipped = gradient_scale(scaled, clip_scale)
    return clipped, {
        "raw_direction_norm": raw_norm,
        "target_direction_norm": float(target_norm),
        "preclip_direction_norm": scaled_norm,
        "clip_scale": clip_scale,
        "applied_direction_norm": gradient_norm(clipped),
    }


def build_training_direction(
    strategy: str,
    gradients: Mapping[str, Gradient],
    config: RegionGradientTrainingConfigV924,
) -> Dict[str, object]:
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown V9.24 strategy: {strategy}")
    global_gradient = gradients["global"]
    global_norm = gradient_norm(global_gradient)
    if global_norm <= 0.0:
        raise RuntimeError("global training gradient is zero")

    mgda_weights = None
    mgda_raw_norm = None
    if strategy == "global_mae":
        raw = global_gradient
        target_norm = global_norm
        task_weights = {
            name: float("nan") for name in REGION_TASK_NAMES
        }
    elif strategy == "equal_region_mean":
        raw = gradient_mean([gradients[name] for name in REGION_TASK_NAMES])
        target_norm = gradient_norm(raw)
        task_weights = {
            name: 1.0 / len(REGION_TASK_NAMES)
            for name in REGION_TASK_NAMES
        }
    else:
        audit_config = GradientAuditConfigV923(
            parameter_scope=config.parameter_scope,
            minimum_mgda_norm=config.minimum_mgda_direction_norm,
            mgda_max_iterations=config.mgda_max_iterations,
            mgda_ftol=config.mgda_ftol,
        )
        fitted = fit_normalized_mgda(gradients, audit_config)
        raw = fitted["direction"]
        mgda_raw_norm = float(fitted["direction_norm"])
        if mgda_raw_norm < float(config.minimum_mgda_direction_norm):
            raise RuntimeError(
                "MGDA direction norm fell below the pre-registered floor: "
                f"{mgda_raw_norm:.8f}"
            )
        target_norm = global_norm
        mgda_weights = np.asarray(fitted["weights"], dtype=np.float64)
        task_weights = {
            name: float(weight)
            for name, weight in zip(
                REGION_TASK_NAMES,
                mgda_weights.tolist(),
            )
        }

    direction, scaling = _scale_and_clip_direction(
        raw,
        target_norm=target_norm,
        clip_norm=config.gradient_clip_norm,
    )
    effects = {
        name: {
            "dot": gradient_dot(gradients[name], direction),
            "cosine": (
                gradient_dot(gradients[name], direction)
                / max(
                    gradient_norm(gradients[name])
                    * gradient_norm(direction),
                    1e-12,
                )
            ),
        }
        for name in ("global", *REGION_TASK_NAMES)
    }
    return {
        "direction": direction,
        "task_weights": task_weights,
        "mgda_raw_norm": mgda_raw_norm,
        "effects": effects,
        **scaling,
    }


def assign_gradient_direction(
    records: Sequence[ParameterRecordV923],
    direction: Gradient,
) -> None:
    if len(records) != len(direction):
        raise ValueError("record/direction length mismatch")
    for record, value in zip(records, direction):
        record.parameter.grad = value.to(
            device=record.parameter.device,
            dtype=record.parameter.dtype,
        ).clone()


def clear_selected_gradients(
    records: Sequence[ParameterRecordV923],
) -> None:
    for record in records:
        record.parameter.grad = None


def _is_better_validation(
    candidate: Mapping[str, object],
    best: Mapping[str, object] | None,
    min_improvement: float,
) -> bool:
    if best is None:
        return True
    candidate_key = (
        float(candidate["validation_objective"]),
        float(candidate["validation_mae"]),
        int(candidate["epoch"]),
    )
    best_key = (
        float(best["validation_objective"]),
        float(best["validation_mae"]),
        int(best["epoch"]),
    )
    if candidate_key[0] < best_key[0] - float(min_improvement):
        return True
    if abs(candidate_key[0] - best_key[0]) <= float(min_improvement):
        return candidate_key[1:] < best_key[1:]
    return False


def train_one_strategy(
    model: torch.nn.Module,
    strategy: str,
    train_loader: Iterable[object],
    valid_loader: Iterable[object],
    forward_batch: Callable[
        [torch.nn.Module, object],
        tuple[torch.Tensor, torch.Tensor],
    ],
    config: RegionGradientTrainingConfigV924,
) -> Dict[str, object]:
    """Train one pre-registered strategy from the supplied anchor state."""
    config.validate()
    if strategy not in STRATEGIES:
        raise ValueError(strategy)
    records = freeze_to_parameter_scope(model, config.parameter_scope)
    initial_fingerprint = selected_parameter_fingerprint(records)
    optimizer = torch.optim.SGD(
        [record.parameter for record in records],
        lr=float(config.learning_rate),
        momentum=0.0,
        weight_decay=0.0,
    )

    history: list[Dict[str, object]] = []
    weight_history: list[Dict[str, object]] = []
    best: Dict[str, object] | None = None
    best_state = _copy_state_dict(model)
    stale_epochs = 0
    stop_reason = "max_epochs"

    baseline_eval = evaluate_model(model, valid_loader, forward_batch)
    baseline_metrics = baseline_eval["metrics"]
    baseline_row: Dict[str, object] = {
        "epoch": 0,
        "strategy": strategy,
        "train_global_mae": None,
        "validation_mae": float(baseline_metrics["mae"]),
        "validation_macro_region_mae": float(
            baseline_metrics["macro_region_mae"]
        ),
        "validation_objective": validation_objective(
            baseline_metrics,
            config,
        ),
        "applied_direction_norm": 0.0,
        "global_direction_cosine": None,
        "all_region_first_order_improve": None,
    }
    history.append(baseline_row)
    best = dict(baseline_row)

    for epoch in range(1, int(config.max_epochs) + 1):
        clear_selected_gradients(records)
        collected = collect_mean_mae_gradients(
            model,
            train_loader,
            records,
            forward_batch,
        )
        try:
            built = build_training_direction(
                strategy,
                collected["gradients"],
                config,
            )
        except RuntimeError as error:
            if (
                strategy == "normalized_mgda"
                and "direction norm" in str(error)
            ):
                stop_reason = "mgda_direction_below_floor"
                break
            raise

        optimizer.zero_grad(set_to_none=True)
        assign_gradient_direction(records, built["direction"])
        optimizer.step()
        clear_selected_gradients(records)
        model.eval()

        valid_eval = evaluate_model(model, valid_loader, forward_batch)
        valid_metrics = valid_eval["metrics"]
        region_improvements = [
            built["effects"][name]["dot"] > 0.0
            for name in REGION_TASK_NAMES
        ]
        row = {
            "epoch": int(epoch),
            "strategy": strategy,
            "train_global_mae": float(
                collected["mean_losses"]["global"]
            ),
            **{
                f"train_region_mae_{name}": float(
                    collected["mean_losses"][name]
                )
                for name in REGION_TASK_NAMES
            },
            "validation_mae": float(valid_metrics["mae"]),
            "validation_macro_region_mae": float(
                valid_metrics["macro_region_mae"]
            ),
            "validation_objective": validation_objective(
                valid_metrics,
                config,
            ),
            "raw_direction_norm": float(built["raw_direction_norm"]),
            "target_direction_norm": float(
                built["target_direction_norm"]
            ),
            "applied_direction_norm": float(
                built["applied_direction_norm"]
            ),
            "clip_scale": float(built["clip_scale"]),
            "global_direction_cosine": float(
                built["effects"]["global"]["cosine"]
            ),
            "minimum_region_direction_cosine": float(
                min(
                    built["effects"][name]["cosine"]
                    for name in REGION_TASK_NAMES
                )
            ),
            "all_region_first_order_improve": bool(
                all(region_improvements)
            ),
            "mgda_raw_norm": built["mgda_raw_norm"],
        }
        history.append(row)
        for name in REGION_TASK_NAMES:
            weight_history.append(
                {
                    "epoch": int(epoch),
                    "strategy": strategy,
                    "region": name,
                    "weight": built["task_weights"][name],
                    "direction_cosine": float(
                        built["effects"][name]["cosine"]
                    ),
                    "direction_dot": float(
                        built["effects"][name]["dot"]
                    ),
                }
            )

        if _is_better_validation(
            row,
            best,
            min_improvement=config.min_validation_improvement,
        ):
            best = dict(row)
            best_state = _copy_state_dict(model)
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(config.early_stop):
                stop_reason = "early_stop"
                break

    model.load_state_dict(best_state, strict=True)
    model.eval()
    final_fingerprint = selected_parameter_fingerprint(records)
    return {
        "version": TRAINING_VERSION,
        "strategy": strategy,
        "config": asdict(config),
        "history": history,
        "weight_history": weight_history,
        "best_epoch": int(best["epoch"]),
        "best_validation_objective": float(
            best["validation_objective"]
        ),
        "best_validation_mae": float(best["validation_mae"]),
        "best_validation_macro_region_mae": float(
            best["validation_macro_region_mae"]
        ),
        "stop_reason": stop_reason,
        "initial_parameter_fingerprint": initial_fingerprint,
        "selected_parameter_fingerprint": final_fingerprint,
        "selected_parameter_names": trainable_parameter_names(records),
        "selected_parameter_count": int(
            sum(record.parameter.numel() for record in records)
        ),
        "best_state_dict": best_state,
    }


def paired_group_bootstrap_difference(
    frame,
    candidate_column: str,
    baseline_column: str,
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    """Bootstrap baseline-error minus candidate-error by conversation group."""
    import pandas as pd

    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    groups = sorted(frame["group_id"].astype(str).unique().tolist())
    if not groups:
        raise RuntimeError("no groups for bootstrap")
    grouped = {
        group: frame[frame["group_id"].astype(str) == group]
        for group in groups
    }
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(repetitions)):
        sampled = rng.choice(groups, size=len(groups), replace=True)
        rows = [grouped[str(group)] for group in sampled]
        local = pd.concat(rows, ignore_index=True)
        candidate_error = np.abs(
            local[candidate_column].to_numpy(dtype=np.float64)
            - local["label"].to_numpy(dtype=np.float64)
        )
        baseline_error = np.abs(
            local[baseline_column].to_numpy(dtype=np.float64)
            - local["label"].to_numpy(dtype=np.float64)
        )
        values.append(
            float(np.mean(baseline_error - candidate_error))
        )
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_gain_difference": float(array.mean()),
        "difference_ci_low": float(np.quantile(array, 0.025)),
        "difference_ci_high": float(np.quantile(array, 0.975)),
        "positive_probability": float(np.mean(array > 0.0)),
    }


__all__ = [
    "TRAINING_VERSION",
    "STRATEGIES",
    "PRIMARY_STRATEGY",
    "RegionGradientTrainingConfigV924",
    "SuccessGateConfigV924",
    "freeze_to_parameter_scope",
    "trainable_parameter_names",
    "regression_metrics",
    "validation_objective",
    "evaluate_model",
    "build_training_direction",
    "assign_gradient_direction",
    "clear_selected_gradients",
    "train_one_strategy",
    "paired_group_bootstrap_difference",
]
