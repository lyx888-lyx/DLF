"""Observable-attribute expert utilities for V9.30.

V9.30 repairs the central train/deploy mismatch of the historical label-region
experts. Expert applicability is computed only from predictions available at
inference time under deterministic modality masks. Labels are used only by
ordinary supervised expert fitting and by evaluation.

The primary deployable strategy is a single global convex weight vector fitted
on strict inner OOF predictions. No winner labels, regret targets, kNN router,
confidence router, or sample-oracle supervision are used.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from .cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _batch_to_device,
    _load_state,
    mode_to_mask,
)
from .static_dense_expert_consensus_v921 import fit_shrunk_mae_simplex

AUDIT_VERSION = "observable_attribute_experts_v930_v1"
EXPERT_NAMES = (
    "text_stable",
    "audio_informative",
    "vision_informative",
    "cross_modal_conflict",
)
ACTION_NAMES_V930 = ("anchor",) + EXPERT_NAMES
EXPERT_MODE = {
    "text_stable": "L",
    "audio_informative": "LA",
    "vision_informative": "LV",
    "cross_modal_conflict": "LAV",
}
MODE_NAMES = ("LAV", "L", "LA", "LV")


@dataclass(frozen=True)
class ObservableAttributeConfigV930:
    batch_size: int = 128
    feature_batch_size: int = 128
    num_workers: int = 2
    max_epochs: int = 18
    early_stop: int = 4
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    use_amp: bool = True
    applicability_floor: float = 0.10
    applicability_power: float = 2.0
    global_loss_weight: float = 0.25
    anchor_preservation_weight: float = 0.10
    validation_global_weight: float = 0.25
    shrinkage_lambda: float = 0.01
    attribute_anchor_prior: float = 1.0
    required_gain_vs_v921: float = 0.005
    required_nondegrading_folds: int = 4
    max_worst_fold_degradation: float = 0.005

    def validate(self) -> None:
        if self.batch_size <= 0 or self.feature_batch_size <= 0:
            raise ValueError("batch sizes must be positive")
        if self.max_epochs <= 0 or self.early_stop <= 0:
            raise ValueError("epoch limits must be positive")
        if self.learning_rate <= 0.0 or self.weight_decay < 0.0:
            raise ValueError("invalid optimizer settings")
        if not 0.0 <= self.applicability_floor < 1.0:
            raise ValueError("applicability_floor must be in [0,1)")
        if self.applicability_power <= 0.0:
            raise ValueError("applicability_power must be positive")
        for name in (
            "global_loss_weight",
            "anchor_preservation_weight",
            "validation_global_weight",
            "shrinkage_lambda",
            "attribute_anchor_prior",
        ):
            if float(getattr(self, name)) < 0.0:
                raise ValueError(f"{name} must be non-negative")


@dataclass(frozen=True)
class AttributeCalibratorV930:
    sorted_scores: Dict[str, tuple[float, ...]]
    applicability_floor: float
    applicability_power: float

    def serializable(self) -> Dict[str, object]:
        return asdict(self)


def _as_vector(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def _as_matrix(value, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0:
        raise ValueError(f"{name} must be a non-empty matrix")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def raw_attribute_scores(
    mode_predictions: Mapping[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    """Create label-free expert scores from deterministic masked predictions."""
    missing = [name for name in MODE_NAMES if name not in mode_predictions]
    if missing:
        raise KeyError(f"missing mode predictions: {missing}")
    values = {
        name: _as_vector(mode_predictions[name], name) for name in MODE_NAMES
    }
    if len({len(value) for value in values.values()}) != 1:
        raise ValueError("mode prediction lengths differ")

    lav = values["LAV"]
    text = values["L"]
    la = values["LA"]
    lv = values["LV"]
    audio_effect = lav - lv
    vision_effect = lav - la
    nontext_effect = np.abs(lav - text)
    paired_disagreement = np.abs(la - lv)
    opposite_effect = np.maximum(0.0, -(audio_effect * vision_effect))
    return {
        "text_stable": -nontext_effect,
        "audio_informative": np.abs(audio_effect) - np.abs(vision_effect),
        "vision_informative": np.abs(vision_effect) - np.abs(audio_effect),
        "cross_modal_conflict": opposite_effect + 0.5 * paired_disagreement,
    }


def fit_attribute_calibrator(
    raw_scores: Mapping[str, Sequence[float]],
    applicability_floor: float,
    applicability_power: float,
) -> AttributeCalibratorV930:
    if set(raw_scores) != set(EXPERT_NAMES):
        raise ValueError("raw score names differ from V9.30 experts")
    sorted_scores = {}
    for name in EXPERT_NAMES:
        value = np.sort(_as_vector(raw_scores[name], name), kind="mergesort")
        sorted_scores[name] = tuple(float(item) for item in value.tolist())
    return AttributeCalibratorV930(
        sorted_scores=sorted_scores,
        applicability_floor=float(applicability_floor),
        applicability_power=float(applicability_power),
    )


def _empirical_cdf(value: np.ndarray, reference: Sequence[float]) -> np.ndarray:
    ref = _as_vector(reference, "reference")
    rank = np.searchsorted(ref, value, side="right").astype(np.float64)
    q = (rank - 0.5) / max(1, len(ref))
    return np.clip(q, 0.0, 1.0)


def apply_attribute_calibrator(
    raw_scores: Mapping[str, Sequence[float]],
    calibrator: AttributeCalibratorV930,
) -> np.ndarray:
    columns = []
    floor = float(calibrator.applicability_floor)
    power = float(calibrator.applicability_power)
    for name in EXPERT_NAMES:
        raw = _as_vector(raw_scores[name], name)
        q = _empirical_cdf(raw, calibrator.sorted_scores[name])
        q = floor + (1.0 - floor) * np.power(q, power)
        columns.append(q)
    result = np.column_stack(columns)
    if not np.isfinite(result).all() or np.any(result < floor - 1e-12):
        raise FloatingPointError("invalid calibrated applicability")
    return result


def mode_prediction_rows(evaluator, loader, device) -> list[Dict[str, object]]:
    """Collect label-free signals plus identifiers; labels are not returned."""
    evaluator.eval()
    rows = []
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = _batch_to_device(batch, device)
            predictions = {}
            for mode in MODE_NAMES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                predictions[mode] = (
                    evaluator(text, audio, vision, mask)["output_logit"]
                    .detach()
                    .view(-1)
                    .cpu()
                    .numpy()
                )
            indices = batch["index"].view(-1).cpu().tolist()
            ids = [str(value) for value in list(batch["id"])]
            for offset, index in enumerate(indices):
                rows.append(
                    {
                        "sample_index": int(index),
                        "sample_id": ids[offset],
                        **{
                            f"prediction_{mode}": float(
                                predictions[mode][offset]
                            )
                            for mode in MODE_NAMES
                        },
                    }
                )
    return rows


def mode_prediction_arrays(
    rows: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not rows:
        raise ValueError("mode prediction rows are empty")
    ordered = sorted(rows, key=lambda row: int(row["sample_index"]))
    indices = [int(row["sample_index"]) for row in ordered]
    if len(set(indices)) != len(indices):
        raise RuntimeError("duplicate sample index in mode predictions")
    return {
        "indices": indices,
        "sample_ids": [str(row["sample_id"]) for row in ordered],
        "mode_predictions": {
            mode: np.asarray(
                [float(row[f"prediction_{mode}"]) for row in ordered],
                dtype=np.float64,
            )
            for mode in MODE_NAMES
        },
    }


def build_attribute_cache(
    evaluator,
    loader,
    device,
    calibrator: AttributeCalibratorV930 | None = None,
    config: ObservableAttributeConfigV930 | None = None,
) -> Dict[str, object]:
    arrays = mode_prediction_arrays(
        mode_prediction_rows(evaluator, loader, device)
    )
    raw = raw_attribute_scores(arrays["mode_predictions"])
    if calibrator is None:
        if config is None:
            raise ValueError("config is required when fitting a calibrator")
        calibrator = fit_attribute_calibrator(
            raw,
            config.applicability_floor,
            config.applicability_power,
        )
    applicability = apply_attribute_calibrator(raw, calibrator)
    return {
        **arrays,
        "raw_scores": raw,
        "applicability": applicability,
        "calibrator": calibrator,
    }


def applicability_lookup(cache: Mapping[str, object]) -> Dict[int, np.ndarray]:
    indices = list(cache["indices"])
    matrix = _as_matrix(cache["applicability"], "applicability")
    if len(indices) != len(matrix):
        raise ValueError("cache indices/applicability mismatch")
    return {
        int(index): matrix[offset].copy()
        for offset, index in enumerate(indices)
    }


def prediction_lookup(
    rows: Sequence[Mapping[str, object]],
) -> Dict[int, float]:
    result = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in result:
            raise RuntimeError(f"duplicate prediction index: {index}")
        result[index] = float(row["prediction"])
    return result


def _configure_trainable_parameters(model: nn.Module, expert_name: str) -> int:
    common = (
        "backbone.proj1",
        "backbone.proj2",
        "backbone.out_layer",
        "backbone.projector_",
        "mask_adapter",
    )
    role_specific = {
        "text_stable": ("backbone.trans_l_mem",),
        "audio_informative": (
            "backbone.trans_l_with_a",
            "backbone.trans_a_mem",
        ),
        "vision_informative": (
            "backbone.trans_l_with_v",
            "backbone.trans_v_mem",
        ),
        "cross_modal_conflict": (
            "backbone.trans_l_with_a",
            "backbone.trans_l_with_v",
            "backbone.trans_a_with_l",
            "backbone.trans_a_with_v",
            "backbone.trans_v_with_l",
            "backbone.trans_v_with_a",
            "backbone.trans_l_mem",
            "backbone.trans_a_mem",
            "backbone.trans_v_mem",
        ),
    }
    if expert_name not in role_specific:
        raise ValueError(f"unknown expert: {expert_name}")
    allowed = common + role_specific[expert_name]
    count = 0
    for name, parameter in model.named_parameters():
        trainable = any(token in name for token in allowed)
        parameter.requires_grad_(trainable)
        if trainable:
            count += parameter.numel()
    if count == 0:
        raise RuntimeError(f"no trainable parameters selected for {expert_name}")
    return int(count)


def _weighted_smooth_l1(
    prediction, labels, weights, beta: float = 0.2
):
    per_sample = F.smooth_l1_loss(
        prediction.view(-1),
        labels.view(-1),
        reduction="none",
        beta=beta,
    )
    value = weights.view(-1).to(per_sample)
    return (per_sample * value).sum() / value.sum().clamp_min(1e-8)


def _lookup_tensor(
    indices: torch.Tensor,
    mapping: Mapping[int, float | np.ndarray],
    device,
):
    values = [
        mapping[int(index)] for index in indices.view(-1).cpu().tolist()
    ]
    return torch.as_tensor(
        np.asarray(values), dtype=torch.float32, device=device
    )


def _weighted_mae(prediction, labels, weights) -> float:
    pred = _as_vector(prediction, "prediction")
    y = _as_vector(labels, "labels")
    w = _as_vector(weights, "weights")
    if not (len(pred) == len(y) == len(w)):
        raise ValueError("weighted MAE inputs differ")
    return float(
        (np.abs(pred - y) * w).sum() / max(1e-12, w.sum())
    )


def _new_wrapper(args, checkpoint: Path):
    model = MissingModalityWrapper(
        DLF(args).to(args.device),
        int(args.feature_dims[1]),
        int(args.feature_dims[2]),
    ).to(args.device)
    model.load_state_dict(_load_state(checkpoint, args.device), strict=True)
    return model


def train_observable_expert(
    args,
    train_loader,
    valid_loader,
    anchor_checkpoint: Path,
    expert_name: str,
    train_applicability: Mapping[int, np.ndarray],
    valid_applicability: Mapping[int, np.ndarray],
    train_anchor_prediction: Mapping[int, float],
    config: ObservableAttributeConfigV930,
    checkpoint: Path,
) -> Dict[str, object]:
    config.validate()
    expert_index = EXPERT_NAMES.index(expert_name)
    mode = EXPERT_MODE[expert_name]
    model = _new_wrapper(args, anchor_checkpoint)
    trainable_parameters = _configure_trainable_parameters(
        model, expert_name
    )
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    optimizer = optim.AdamW(
        parameters,
        lr=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
    )
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(
        enabled=bool(config.use_amp and torch.cuda.is_available())
    )
    best_objective = float("inf")
    best_epoch = 0
    history = []
    checkpoint.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, int(config.max_epochs) + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            text, audio, vision, labels = _batch_to_device(
                batch, args.device
            )
            indices = batch["index"].view(-1)
            local_q = _lookup_tensor(
                indices, train_applicability, args.device
            )[:, expert_index]
            anchor_target = _lookup_tensor(
                indices, train_anchor_prediction, args.device
            ).view(-1)
            mask = mode_to_mask(
                mode, labels.size(0), args.device, audio.dtype
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(
                enabled=bool(config.use_amp and torch.cuda.is_available())
            ):
                prediction = model(
                    text, audio, vision, mask
                )["output_logit"].view(-1)
                weighted = _weighted_smooth_l1(
                    prediction, labels, local_q
                )
                global_loss = F.smooth_l1_loss(
                    prediction, labels.view(-1), beta=0.2
                )
                preservation = _weighted_smooth_l1(
                    prediction,
                    anchor_target,
                    1.0
                    - local_q
                    + float(config.applicability_floor),
                )
                loss = (
                    weighted
                    + float(config.global_loss_weight) * global_loss
                    + float(config.anchor_preservation_weight)
                    * preservation
                )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite {expert_name} loss"
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if float(config.grad_clip_norm) > 0.0:
                nn.utils.clip_grad_norm_(
                    parameters, float(config.grad_clip_norm)
                )
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().item()))

        valid = predict_observable_expert(
            model, valid_loader, args.device, mode
        )
        valid_indices = valid["sample_indices"]
        q = np.asarray(
            [
                valid_applicability[int(index)][expert_index]
                for index in valid_indices
            ],
            dtype=np.float64,
        )
        weighted_mae = _weighted_mae(
            valid["prediction"], valid["labels"], q
        )
        global_mae = float(
            np.abs(valid["prediction"] - valid["labels"]).mean()
        )
        objective = (
            weighted_mae
            + float(config.validation_global_weight) * global_mae
        )
        scheduler.step(objective)
        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(train_losses)),
                "valid_weighted_mae": weighted_mae,
                "valid_global_mae": global_mae,
                "valid_objective": objective,
                "learning_rate": float(
                    optimizer.param_groups[0]["lr"]
                ),
            }
        )
        if objective < best_objective - 1e-6:
            best_objective = objective
            best_epoch = epoch
            torch.save(
                {
                    "version": AUDIT_VERSION,
                    "expert_name": expert_name,
                    "mode": mode,
                    "state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "config": asdict(config),
                    "trainable_parameters": trainable_parameters,
                },
                checkpoint,
            )
        if epoch - best_epoch >= int(config.early_stop):
            break

    if not checkpoint.is_file():
        raise RuntimeError(
            f"{expert_name} did not save a checkpoint"
        )
    payload = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(payload["state_dict"], strict=True)
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_valid_objective": best_objective,
        "trainable_parameters": trainable_parameters,
        "mode": mode,
    }


@torch.no_grad()
def predict_observable_expert(
    model, loader, device, mode: str
) -> Dict[str, object]:
    model.eval()
    predictions = []
    labels = []
    indices = []
    sample_ids = []
    for batch in loader:
        text, audio, vision, local_labels = _batch_to_device(
            batch, device
        )
        mask = mode_to_mask(
            mode, local_labels.size(0), device, audio.dtype
        )
        output = model(
            text, audio, vision, mask
        )["output_logit"].detach().view(-1)
        predictions.append(output.cpu())
        labels.append(local_labels.detach().view(-1).cpu())
        indices.extend(
            int(value)
            for value in batch["index"].view(-1).cpu().tolist()
        )
        sample_ids.extend(str(value) for value in list(batch["id"]))
    prediction = torch.cat(predictions).numpy().astype(np.float64)
    label = torch.cat(labels).numpy().astype(np.float64)
    if len(set(indices)) != len(indices):
        raise RuntimeError(
            "expert prediction contains duplicate indices"
        )
    return {
        "sample_indices": indices,
        "sample_ids": sample_ids,
        "prediction": prediction,
        "labels": label,
    }


def prediction_metrics(prediction, labels, baseline=None) -> Dict[str, float]:
    pred = _as_vector(prediction, "prediction")
    y = _as_vector(labels, "labels")
    if len(pred) != len(y):
        raise ValueError("prediction/labels mismatch")
    error = np.abs(pred - y)
    result = {
        "sample_count": int(len(y)),
        "mae": float(error.mean()),
        "median_absolute_error": float(np.median(error)),
        "p90_absolute_error": float(np.quantile(error, 0.90)),
    }
    if baseline is not None:
        base = _as_vector(baseline, "baseline")
        gain = np.abs(base - y) - error
        result.update(
            {
                "baseline_mae": float(np.abs(base - y).mean()),
                "gain_vs_baseline": float(gain.mean()),
                "win_rate_vs_baseline": float((gain > 0.0).mean()),
                "large_harm_rate_010": float((gain < -0.10).mean()),
            }
        )
    return result


def fit_fixed_convex(
    actions, labels, shrinkage_lambda: float
) -> np.ndarray:
    x = _as_matrix(actions, "actions")
    y = _as_vector(labels, "labels")
    if len(x) != len(y):
        raise ValueError("actions/labels mismatch")
    return fit_shrunk_mae_simplex(
        x, y, float(shrinkage_lambda)
    )


def inner_crossfit_fixed_convex(
    actions,
    labels,
    fold_index,
    shrinkage_lambda: float,
) -> Dict[str, object]:
    x = _as_matrix(actions, "actions")
    y = _as_vector(labels, "labels")
    folds = np.asarray(fold_index, dtype=np.int64).reshape(-1)
    if not (len(x) == len(y) == len(folds)) or np.any(folds < 0):
        raise ValueError("invalid inner cross-fit inputs")
    prediction = np.full(len(y), np.nan, dtype=np.float64)
    rows = []
    for fold in sorted(np.unique(folds).tolist()):
        train = folds != fold
        holdout = folds == fold
        weights = fit_fixed_convex(
            x[train], y[train], shrinkage_lambda
        )
        prediction[holdout] = x[holdout] @ weights
        rows.append(
            {
                "inner_fold": int(fold),
                "training_count": int(train.sum()),
                "holdout_count": int(holdout.sum()),
                "mae": float(
                    np.abs(
                        prediction[holdout] - y[holdout]
                    ).mean()
                ),
                "weights": weights,
            }
        )
    if not np.isfinite(prediction).all():
        raise FloatingPointError(
            "inner cross-fit prediction is incomplete"
        )
    final_weights = fit_fixed_convex(
        x, y, shrinkage_lambda
    )
    return {
        "prediction": prediction,
        "fold_rows": rows,
        "final_weights": final_weights,
    }


def attribute_weighted_prediction(
    actions,
    applicability,
    anchor_prior: float = 1.0,
) -> Dict[str, np.ndarray]:
    x = _as_matrix(actions, "actions")
    q = _as_matrix(applicability, "applicability")
    if (
        x.shape[1] != len(ACTION_NAMES_V930)
        or q.shape != (len(x), len(EXPERT_NAMES))
    ):
        raise ValueError("attribute weighting shape mismatch")
    raw = np.column_stack(
        [
            np.full(
                len(x), float(anchor_prior), dtype=np.float64
            ),
            q,
        ]
    )
    weights = raw / np.maximum(
        raw.sum(axis=1, keepdims=True), 1e-12
    )
    return {
        "prediction": (x * weights).sum(axis=1),
        "weights": weights,
    }


def residual_correlation(actions, labels) -> np.ndarray:
    x = _as_matrix(actions, "actions")
    y = _as_vector(labels, "labels")
    residual = x - y[:, None]
    matrix = np.corrcoef(residual, rowvar=False)
    return np.nan_to_num(
        matrix, nan=0.0, posinf=0.0, neginf=0.0
    )


def success_gate(
    fold_primary_gain: Sequence[float],
    aggregate_gain_vs_v921: float,
    config: ObservableAttributeConfigV930,
) -> Dict[str, object]:
    gain = _as_vector(fold_primary_gain, "fold_primary_gain")
    nondegrading = int((gain >= -1e-12).sum())
    worst_degradation = float(max(0.0, -gain.min()))
    passed = (
        float(aggregate_gain_vs_v921)
        >= float(config.required_gain_vs_v921)
        and nondegrading
        >= int(config.required_nondegrading_folds)
        and worst_degradation
        <= float(config.max_worst_fold_degradation)
    )
    return {
        "passed": bool(passed),
        "aggregate_gain_vs_v921": float(
            aggregate_gain_vs_v921
        ),
        "required_gain_vs_v921": float(
            config.required_gain_vs_v921
        ),
        "nondegrading_outer_folds": nondegrading,
        "required_nondegrading_outer_folds": int(
            config.required_nondegrading_folds
        ),
        "worst_fold_degradation": worst_degradation,
        "max_worst_fold_degradation": float(
            config.max_worst_fold_degradation
        ),
        "next_step": (
            "Only if passed, consider pre-registered deterministic "
            "attribute weighting. Do not train a winner router from "
            "oracle labels."
        ),
    }


__all__ = [
    "AUDIT_VERSION",
    "EXPERT_NAMES",
    "ACTION_NAMES_V930",
    "EXPERT_MODE",
    "MODE_NAMES",
    "ObservableAttributeConfigV930",
    "AttributeCalibratorV930",
    "raw_attribute_scores",
    "fit_attribute_calibrator",
    "apply_attribute_calibrator",
    "mode_prediction_rows",
    "mode_prediction_arrays",
    "build_attribute_cache",
    "applicability_lookup",
    "prediction_lookup",
    "train_observable_expert",
    "predict_observable_expert",
    "prediction_metrics",
    "fit_fixed_convex",
    "inner_crossfit_fixed_convex",
    "attribute_weighted_prediction",
    "residual_correlation",
    "success_gate",
]
