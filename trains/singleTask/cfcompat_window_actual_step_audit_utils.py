"""Window-level actual-step diagnostics for CFCompatKD v12.3.

v12.3 is diagnostic only.  It distinguishes three mechanism levels for each
predeclared v12 optimizer window:

1. raw post-surgery gradient direction ``g_update``;
2. the actual finite Adam parameter displacement ``delta_theta``;
3. the realized held-out Train-video OOF loss change after that finite step.

For an OOF loss gradient ``g_oof``:

* ``dot(g_update, g_oof) > 0`` means gradient descent on the raw surgery
  direction is first-order improving;
* ``dot(delta_theta, g_oof) < 0`` means the *actual* Adam parameter displacement
  is first-order improving;
* ``loss_after - loss_before < 0`` means the finite step actually improved the
  OOF loss.

No thresholds here are model-selection criteria.  Signs are descriptive
mechanism labels fixed before the formal v12.3 audit.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch


VERSION = "cfcompat_window_actual_step_train_oof_audit_v12p3"
METHOD = "DLF-v12-Exact-Replay-Window-Actual-Step-Train-OOF-Audit-v12.3"
OUTPUT_TAG = "cfcompat_window_actual_step_audit_v12p3"
DEV_SEED = 1113
N_FOLDS = 5

# Frozen before v12.3.  v12.2 located the fast function-movement / failure-onset
# transition in this interval.  Every optimizer window in every listed epoch is
# audited; no result-dependent window selection is allowed.
AUDIT_EPOCH_START = 4
AUDIT_EPOCH_END = 12
AUDIT_EPOCHS = tuple(range(AUDIT_EPOCH_START, AUDIT_EPOCH_END + 1))

KEY_OOF_GROUPS = (
    "OOF_TEACHER_BENEFICIAL",
    "OOF_TEACHER_NONBENEFICIAL",
    "OOF_S0_BENEFICIAL",
    "OOF_S0_NONBENEFICIAL",
)


def jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def clone_tensor_tuple(values: Sequence[torch.Tensor]) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().cpu().clone() for value in values)


def flatten_tensor_tuple(
    values: Sequence[torch.Tensor], dtype: torch.dtype = torch.float64
) -> torch.Tensor:
    if not values:
        raise ValueError("Cannot flatten an empty tensor tuple.")
    parts = [value.detach().reshape(-1).cpu().to(dtype) for value in values]
    vector = torch.cat(parts)
    if not torch.isfinite(vector).all():
        raise FloatingPointError("Flattened vector contains NaN/Inf.")
    return vector


def assign_parameter_tuple(
    parameters: Sequence[torch.nn.Parameter], values: Sequence[torch.Tensor]
) -> None:
    if len(parameters) != len(values):
        raise ValueError("Parameter snapshot length mismatch.")
    with torch.no_grad():
        for parameter, value in zip(parameters, values):
            if tuple(parameter.shape) != tuple(value.shape):
                raise ValueError("Parameter snapshot shape mismatch.")
            parameter.copy_(value.to(device=parameter.device, dtype=parameter.dtype))


def vector_dot(first: torch.Tensor, second: torch.Tensor) -> float:
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("Expected aligned flattened vectors.")
    return float(torch.dot(first.to(torch.float64), second.to(torch.float64)))


def vector_cosine(first: torch.Tensor, second: torch.Tensor) -> float:
    dot = vector_dot(first, second)
    norm_first = float(torch.linalg.vector_norm(first.to(torch.float64)))
    norm_second = float(torch.linalg.vector_norm(second.to(torch.float64)))
    if norm_first <= 0.0 or norm_second <= 0.0:
        return 0.0
    return float(dot / (norm_first * norm_second))


def step_mechanism_row(
    *,
    fold: int,
    epoch: int,
    window: int,
    oof_group: str,
    oof_count: int,
    surgery_gradient: torch.Tensor,
    actual_delta: torch.Tensor,
    oof_gradient_before: torch.Tensor,
    loss_before: float,
    loss_after: float,
) -> dict:
    """Classify one window/group without using any tunable magnitude threshold."""
    if oof_group not in KEY_OOF_GROUPS:
        raise ValueError("Unexpected OOF group: {}".format(oof_group))
    raw_dot = vector_dot(surgery_gradient, oof_gradient_before)
    raw_cosine = vector_cosine(surgery_gradient, oof_gradient_before)
    actual_first_order_change = vector_dot(actual_delta, oof_gradient_before)
    effective_direction = -actual_delta
    adam_effective_cosine = vector_cosine(effective_direction, oof_gradient_before)
    raw_to_adam_cosine = vector_cosine(surgery_gradient, effective_direction)
    finite_change = float(loss_after - loss_before)

    raw_harm = bool(raw_dot < 0.0)
    adam_harm = bool(actual_first_order_change > 0.0)
    finite_harm = bool(finite_change > 0.0)
    if raw_harm:
        mechanism = "RAW_SURGERY_DIRECTION_HARM"
    elif adam_harm:
        mechanism = "ADAM_TRANSFORM_HARM"
    elif finite_harm:
        mechanism = "NONLINEAR_FINITE_STEP_HARM"
    else:
        mechanism = "SAFE_OR_IMPROVING"

    return {
        "Fold": int(fold),
        "Epoch": int(epoch),
        "UpdateWindow": int(window),
        "OOFGroup": str(oof_group),
        "OOFN": int(oof_count),
        "raw_surgery_gradient_l2": float(torch.linalg.vector_norm(surgery_gradient)),
        "actual_delta_l2": float(torch.linalg.vector_norm(actual_delta)),
        "oof_gradient_l2": float(torch.linalg.vector_norm(oof_gradient_before)),
        "raw_surgery_dot_oof_gradient": raw_dot,
        "raw_surgery_cosine_oof_gradient": raw_cosine,
        "predicted_first_order_loss_change_raw_gradient_descent": float(-raw_dot),
        "actual_delta_dot_oof_gradient": actual_first_order_change,
        "adam_effective_direction_cosine_oof_gradient": adam_effective_cosine,
        "raw_surgery_to_adam_effective_cosine": raw_to_adam_cosine,
        "predicted_first_order_loss_change_actual_adam_step": actual_first_order_change,
        "oof_loss_before": float(loss_before),
        "oof_loss_after": float(loss_after),
        "actual_finite_oof_loss_change": finite_change,
        "raw_surgery_harm": raw_harm,
        "actual_adam_first_order_harm": adam_harm,
        "actual_finite_step_harm": finite_harm,
        "mechanism_class": mechanism,
    }


def aggregate_mechanism(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Fold", "Epoch", "UpdateWindow", "OOFGroup", "mechanism_class",
        "raw_surgery_harm", "actual_adam_first_order_harm", "actual_finite_step_harm",
        "raw_surgery_cosine_oof_gradient", "adam_effective_direction_cosine_oof_gradient",
        "raw_surgery_to_adam_effective_cosine", "actual_finite_oof_loss_change",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError("Window mechanism frame lacks columns: {}".format(sorted(missing)))
    rows = []
    for (epoch, group), local in frame.groupby(["Epoch", "OOFGroup"], sort=True):
        rows.append(
            {
                "Epoch": int(epoch),
                "OOFGroup": str(group),
                "FoldCount": int(local.Fold.nunique()),
                "WindowCount": int(len(local)),
                "raw_harm_window_fraction": float(local.raw_surgery_harm.astype(bool).mean()),
                "adam_first_order_harm_window_fraction": float(
                    local.actual_adam_first_order_harm.astype(bool).mean()
                ),
                "finite_harm_window_fraction": float(local.actual_finite_step_harm.astype(bool).mean()),
                "mean_raw_surgery_cosine": float(local.raw_surgery_cosine_oof_gradient.mean()),
                "mean_adam_effective_cosine": float(
                    local.adam_effective_direction_cosine_oof_gradient.mean()
                ),
                "mean_raw_to_adam_effective_cosine": float(
                    local.raw_surgery_to_adam_effective_cosine.mean()
                ),
                "mean_actual_finite_oof_loss_change": float(
                    local.actual_finite_oof_loss_change.mean()
                ),
                "raw_direction_harm_count": int(
                    local.mechanism_class.eq("RAW_SURGERY_DIRECTION_HARM").sum()
                ),
                "adam_transform_harm_count": int(
                    local.mechanism_class.eq("ADAM_TRANSFORM_HARM").sum()
                ),
                "nonlinear_finite_step_harm_count": int(
                    local.mechanism_class.eq("NONLINEAR_FINITE_STEP_HARM").sum()
                ),
                "safe_or_improving_count": int(
                    local.mechanism_class.eq("SAFE_OR_IMPROVING").sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def overall_mechanism_counts(frame: pd.DataFrame, group: str) -> dict:
    local = frame.loc[frame.OOFGroup.astype(str).eq(str(group))]
    if local.empty:
        raise RuntimeError("No window rows for {}.".format(group))
    return {
        "window_count": int(len(local)),
        "raw_direction_harm_count": int(
            local.mechanism_class.eq("RAW_SURGERY_DIRECTION_HARM").sum()
        ),
        "adam_transform_harm_count": int(
            local.mechanism_class.eq("ADAM_TRANSFORM_HARM").sum()
        ),
        "nonlinear_finite_step_harm_count": int(
            local.mechanism_class.eq("NONLINEAR_FINITE_STEP_HARM").sum()
        ),
        "safe_or_improving_count": int(
            local.mechanism_class.eq("SAFE_OR_IMPROVING").sum()
        ),
        "raw_harm_window_fraction": float(local.raw_surgery_harm.astype(bool).mean()),
        "adam_first_order_harm_window_fraction": float(
            local.actual_adam_first_order_harm.astype(bool).mean()
        ),
        "finite_harm_window_fraction": float(local.actual_finite_step_harm.astype(bool).mean()),
        "mean_raw_to_adam_effective_cosine": float(
            local.raw_surgery_to_adam_effective_cosine.mean()
        ),
    }


__all__ = [
    "AUDIT_EPOCH_END",
    "AUDIT_EPOCH_START",
    "AUDIT_EPOCHS",
    "DEV_SEED",
    "KEY_OOF_GROUPS",
    "METHOD",
    "N_FOLDS",
    "OUTPUT_TAG",
    "VERSION",
    "aggregate_mechanism",
    "assign_parameter_tuple",
    "clone_tensor_tuple",
    "flatten_tensor_tuple",
    "jsonable",
    "overall_mechanism_counts",
    "step_mechanism_row",
    "vector_cosine",
    "vector_dot",
]
