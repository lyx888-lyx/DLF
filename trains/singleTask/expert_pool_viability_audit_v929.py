"""Utilities for the strict V9.29 expert-pool viability ("life-or-death") audit.

The module operates only on saved scalar predictions and labels.  It deliberately
separates deployable references from label-cheating diagnostic upper bounds.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence

import numpy as np
from scipy.optimize import linprog

AUDIT_VERSION = "expert_pool_viability_audit_v929_v1"


@dataclass(frozen=True)
class ViabilityAuditConfigV929:
    target_mae: float = 0.70
    tolerance: float = 1e-7

    def validate(self) -> None:
        if not np.isfinite(self.target_mae) or self.target_mae <= 0.0:
            raise ValueError("target_mae must be positive and finite")
        if not np.isfinite(self.tolerance) or self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive and finite")


def as_vector(value, *, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def as_action_matrix(value, *, name: str = "actions") -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.ndim != 2 or result.shape[0] == 0 or result.shape[1] < 2:
        raise ValueError(f"{name} must have shape [N,K] with N>0 and K>=2")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return result


def validate_simplex(weights: Sequence[float], action_count: int) -> np.ndarray:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.shape != (int(action_count),):
        raise ValueError(f"expected {action_count} weights, got {value.shape}")
    if not np.isfinite(value).all() or np.any(value < -1e-8):
        raise ValueError("invalid convex weights")
    value = np.clip(value, 0.0, None)
    if value.sum() <= 0.0:
        raise ValueError("zero convex weight vector")
    value /= value.sum()
    if abs(float(value.sum()) - 1.0) > 1e-9:
        raise AssertionError("simplex normalization failed")
    return value


def fit_mae_simplex(actions, labels) -> np.ndarray:
    """Fit the exact in-sample convex MAE optimum by linear programming."""
    x = as_action_matrix(actions)
    y = as_vector(labels, name="labels")
    if len(x) != len(y):
        raise ValueError("actions/labels shape mismatch")
    sample_count, action_count = x.shape
    objective = np.concatenate(
        [
            np.zeros(action_count, dtype=np.float64),
            np.ones(sample_count, dtype=np.float64) / sample_count,
        ]
    )
    a_upper = np.zeros(
        (2 * sample_count, action_count + sample_count), dtype=np.float64
    )
    b_upper = np.zeros(2 * sample_count, dtype=np.float64)
    a_upper[:sample_count, :action_count] = x
    a_upper[:sample_count, action_count:] = -np.eye(sample_count)
    b_upper[:sample_count] = y
    a_upper[sample_count:, :action_count] = -x
    a_upper[sample_count:, action_count:] = -np.eye(sample_count)
    b_upper[sample_count:] = -y
    a_equal = np.zeros((1, action_count + sample_count), dtype=np.float64)
    a_equal[0, :action_count] = 1.0
    result = linprog(
        objective,
        A_ub=a_upper,
        b_ub=b_upper,
        A_eq=a_equal,
        b_eq=np.ones(1, dtype=np.float64),
        bounds=[(0.0, 1.0)] * action_count
        + [(0.0, None)] * sample_count,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"simplex MAE optimization failed: {result.message}")
    return validate_simplex(result.x[:action_count], action_count)


def mae(prediction, labels) -> float:
    pred = as_vector(prediction, name="prediction")
    y = as_vector(labels, name="labels")
    if len(pred) != len(y):
        raise ValueError("prediction/labels shape mismatch")
    return float(np.abs(pred - y).mean())


def prediction_metrics(prediction, labels, anchor=None) -> Dict[str, float]:
    pred = as_vector(prediction, name="prediction")
    y = as_vector(labels, name="labels")
    if len(pred) != len(y):
        raise ValueError("prediction/labels shape mismatch")
    error = np.abs(pred - y)
    result: Dict[str, float] = {
        "sample_count": int(len(y)),
        "mae": float(error.mean()),
        "median_absolute_error": float(np.median(error)),
        "p90_absolute_error": float(np.quantile(error, 0.90)),
        "max_absolute_error": float(error.max()),
    }
    if anchor is not None:
        base = as_vector(anchor, name="anchor")
        if len(base) != len(y):
            raise ValueError("anchor/labels shape mismatch")
        gain = np.abs(base - y) - error
        result.update(
            {
                "anchor_mae": float(np.abs(base - y).mean()),
                "gain_vs_anchor": float(gain.mean()),
                "win_rate_vs_anchor": float((gain > 0.0).mean()),
                "large_gain_rate_010": float((gain > 0.10).mean()),
                "large_harm_rate_010": float((gain < -0.10).mean()),
            }
        )
    return result


def compute_fold_upper_bounds(
    actions,
    labels,
    action_names: Sequence[str],
) -> Dict[str, object]:
    """Compute fold-local label-cheating single, convex, and sample oracles."""
    x = as_action_matrix(actions)
    y = as_vector(labels, name="labels")
    names = tuple(str(value) for value in action_names)
    if len(x) != len(y):
        raise ValueError("actions/labels shape mismatch")
    if len(names) != x.shape[1] or len(set(names)) != len(names):
        raise ValueError("action_names do not match action columns")

    errors = np.abs(x - y[:, None])
    action_mae = errors.mean(axis=0)
    best_single_index = int(action_mae.argmin())
    best_single_prediction = x[:, best_single_index]

    convex_weights = fit_mae_simplex(x, y)
    convex_prediction = x @ convex_weights

    sample_oracle_index = errors.argmin(axis=1).astype(np.int64)
    sample_oracle_prediction = x[
        np.arange(len(x), dtype=np.int64), sample_oracle_index
    ]

    tolerance = 1e-7
    if mae(convex_prediction, y) > mae(best_single_prediction, y) + tolerance:
        raise AssertionError("fold cheating convex solution is worse than one-hot single")
    if mae(sample_oracle_prediction, y) > mae(best_single_prediction, y) + tolerance:
        raise AssertionError("sample oracle is worse than fold-best single")

    return {
        "action_mae": action_mae,
        "best_single_index": best_single_index,
        "best_single_action": names[best_single_index],
        "best_single_prediction": best_single_prediction,
        "convex_weights": convex_weights,
        "convex_prediction": convex_prediction,
        "sample_oracle_index": sample_oracle_index,
        "sample_oracle_action": np.asarray(
            [names[index] for index in sample_oracle_index], dtype=object
        ),
        "sample_oracle_prediction": sample_oracle_prediction,
    }


def oracle_gap_closure(anchor_mae: float, method_mae: float, oracle_mae: float) -> float:
    denominator = float(anchor_mae) - float(oracle_mae)
    if denominator <= 0.0:
        return float("nan")
    return float((float(anchor_mae) - float(method_mae)) / denominator)


def make_viability_verdict(
    metrics: Dict[str, Dict[str, float]], target_mae: float
) -> Dict[str, object]:
    required = {
        "v921_convex_shrinkage",
        "best_global_fixed_action_cheating",
        "fold_best_single_cheating",
        "pooled_global_convex_cheating",
        "fold_local_convex_cheating",
        "sample_oracle",
    }
    missing = sorted(required - set(metrics))
    if missing:
        raise KeyError(f"missing aggregate metrics: {missing}")

    values = {name: float(metrics[name]["mae"]) for name in required}
    below = {name: value < float(target_mae) for name, value in values.items()}

    if below["best_global_fixed_action_cheating"] and not below[
        "v921_convex_shrinkage"
    ]:
        verdict = "single_action_family_has_target_capacity_but_v921_does_not"
    elif not below["pooled_global_convex_cheating"]:
        if below["fold_local_convex_cheating"]:
            verdict = "global_fixed_fusion_cannot_hit_target_fold_specific_cheating_can"
        elif below["sample_oracle"]:
            verdict = "fixed_fusion_cannot_hit_target_sample_oracle_can"
        else:
            verdict = "current_discrete_expert_pool_cannot_hit_target_by_selection"
    elif below["pooled_global_convex_cheating"] and not below[
        "v921_convex_shrinkage"
    ]:
        verdict = "fixed_fusion_has_target_capacity_only_with_outer_label_cheating"
    elif below["v921_convex_shrinkage"]:
        verdict = "deployable_v921_already_hits_target"
    else:
        verdict = "inconclusive_review_full_table"

    return {
        "target_mae": float(target_mae),
        "below_target": below,
        "verdict": verdict,
        "interpretation_guardrails": {
            "posthoc_single_is_not_deployable": True,
            "outer_fitted_convex_is_not_deployable": True,
            "sample_oracle_uses_true_labels": True,
            "convex_interpolation_can_outperform_discrete_sample_oracle": True,
        },
    }
