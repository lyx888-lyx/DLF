"""Strict nested static expert-consensus utilities for V9.21.

Only saved V9.19 predictions are used.  There is no region model, router,
per-sample gate, or sample-dependent expert weight.  Convex weights are fitted
on an outer fold's inner OOF predictions and then frozen for its unseen outer
holdout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linprog, minimize

from .model.GlobalDenseExpertConsensusV921 import (
    GlobalDenseExpertConsensusV921,
)
from .model.SemanticCostCoachV99 import ACTION_NAMES
from .no_train_decomposition_v920 import PoolView, normalize_pool

AUDIT_VERSION = "static_dense_expert_consensus_v921_v1"
PRIMARY_STRATEGY = "convex_shrinkage"


@dataclass(frozen=True)
class ConsensusConfigV921:
    shrinkage_lambda: float = 0.01
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 1111
    required_gain_vs_selected_single: float = 0.003
    required_nondegrading_outer_folds: int = 4
    max_worst_fold_degradation: float = 0.005


def _numpy_actions(pool: PoolView) -> tuple[np.ndarray, np.ndarray]:
    return (
        pool.actions.detach().cpu().double().numpy(),
        pool.labels.detach().cpu().double().numpy(),
    )


def validate_simplex(weights: Sequence[float], action_count: int) -> np.ndarray:
    value = np.asarray(weights, dtype=np.float64).reshape(-1)
    if value.shape != (int(action_count),):
        raise ValueError(f"expected {action_count} weights, got {value.shape}")
    if not np.isfinite(value).all() or np.any(value < -1e-7):
        raise ValueError("invalid convex weights")
    value = np.clip(value, 0.0, None)
    if value.sum() <= 0.0:
        raise ValueError("zero weight vector")
    value /= value.sum()
    if abs(float(value.sum()) - 1.0) > 1e-8:
        raise AssertionError("simplex normalization failed")
    return value


def fit_mae_simplex(actions: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Solve exact convex MAE stacking as a linear program."""
    x = np.asarray(actions, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("actions/labels shape mismatch")
    n, action_count = x.shape
    objective = np.concatenate(
        [np.zeros(action_count, dtype=np.float64), np.ones(n) / max(1, n)]
    )
    a_upper = np.zeros((2 * n, action_count + n), dtype=np.float64)
    b_upper = np.zeros(2 * n, dtype=np.float64)
    a_upper[:n, :action_count] = x
    a_upper[:n, action_count:] = -np.eye(n)
    b_upper[:n] = y
    a_upper[n:, :action_count] = -x
    a_upper[n:, action_count:] = -np.eye(n)
    b_upper[n:] = -y
    a_equal = np.zeros((1, action_count + n), dtype=np.float64)
    a_equal[0, :action_count] = 1.0
    result = linprog(
        objective,
        A_ub=a_upper,
        b_ub=b_upper,
        A_eq=a_equal,
        b_eq=np.ones(1),
        bounds=[(0.0, 1.0)] * action_count + [(0.0, None)] * n,
        method="highs",
    )
    if not result.success:
        raise RuntimeError(f"simplex MAE optimization failed: {result.message}")
    return validate_simplex(result.x[:action_count], action_count)


def fit_shrunk_mae_simplex(
    actions: np.ndarray,
    labels: np.ndarray,
    shrinkage_lambda: float,
) -> np.ndarray:
    """Fit one pre-registered MAE simplex with weak equal-weight shrinkage."""
    x = np.asarray(actions, dtype=np.float64)
    y = np.asarray(labels, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("actions/labels shape mismatch")
    action_count = x.shape[1]
    prior = np.full(action_count, 1.0 / action_count, dtype=np.float64)
    initial = fit_mae_simplex(x, y)
    lam = float(shrinkage_lambda)
    if lam < 0.0:
        raise ValueError("shrinkage_lambda must be non-negative")

    def objective(weights: np.ndarray) -> float:
        residual = x @ weights - y
        return float(np.abs(residual).mean() + lam * np.square(weights - prior).sum())

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * action_count,
        constraints=[{"type": "eq", "fun": lambda value: float(value.sum() - 1.0)}],
        options={"maxiter": 2000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise RuntimeError(f"shrunk simplex optimization failed: {result.message}")
    weights = validate_simplex(result.x, action_count)
    if objective(weights) > objective(initial) + 1e-7:
        raise RuntimeError("shrunk optimizer returned a worse objective than initialization")
    return weights


def predict_weighted(actions: torch.Tensor, weights: Sequence[float]) -> torch.Tensor:
    module = GlobalDenseExpertConsensusV921(weights)
    return module(actions).view(-1)


def predict_strategy(
    pool: PoolView,
    strategy: str,
    weights: Sequence[float] | None = None,
    action_index: int | None = None,
) -> torch.Tensor:
    actions = pool.actions
    if strategy == "anchor":
        return actions[:, 0]
    if strategy == "fixed_action":
        if action_index is None:
            raise ValueError("fixed_action requires action_index")
        return actions[:, int(action_index)]
    if strategy in {"convex_mae", "convex_shrinkage"}:
        if weights is None:
            raise ValueError(f"{strategy} requires weights")
        return predict_weighted(actions, weights)
    if strategy == "boundary_positive_mean":
        return actions[:, [2, 3]].mean(dim=1)
    if strategy == "all_action_mean":
        return actions.mean(dim=1)
    if strategy == "all_action_median":
        return actions.median(dim=1).values
    if strategy == "trimmed_mean":
        ordered = actions.sort(dim=1).values
        return ordered[:, 1:-1].mean(dim=1)
    raise ValueError(f"unknown static strategy: {strategy}")


def prediction_metrics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    anchor: torch.Tensor,
) -> Dict[str, object]:
    pred = prediction.detach().cpu().float().view(-1)
    y = labels.detach().cpu().float().view(-1)
    base = anchor.detach().cpu().float().view(-1)
    if not (len(pred) == len(y) == len(base)):
        raise ValueError("metric inputs do not align")
    selected_error = torch.abs(pred - y)
    anchor_error = torch.abs(base - y)
    gain = anchor_error - selected_error
    return {
        "sample_count": len(y),
        "anchor_mae": float(anchor_error.mean().item()),
        "mae": float(selected_error.mean().item()),
        "gain_vs_anchor": float(gain.mean().item()),
        "win_rate": float((gain > 0).float().mean().item()),
        "large_gain_rate_010": float((gain > 0.10).float().mean().item()),
        "large_harm_rate_010": float((gain < -0.10).float().mean().item()),
        "mean_gain": float(gain.mean().item()),
        "prediction": pred,
        "sample_gain": gain,
    }


def best_single_action(pool: PoolView) -> int:
    errors = torch.abs(pool.actions - pool.labels.view(-1, 1)).mean(dim=0)
    return int(errors.argmin().item())


def fit_consensus_weights(
    pool: PoolView, config: ConsensusConfigV921
) -> Dict[str, np.ndarray]:
    actions, labels = _numpy_actions(pool)
    return {
        "convex_mae": fit_mae_simplex(actions, labels),
        "convex_shrinkage": fit_shrunk_mae_simplex(
            actions, labels, config.shrinkage_lambda
        ),
    }


def inner_crossfit_consensus(
    payload: Mapping[str, object],
    config: ConsensusConfigV921,
) -> Dict[str, object]:
    """Cross-fit static weights across V9.19 inner expert stacks."""
    pool = normalize_pool(payload)
    fold_index = torch.as_tensor(payload["fold_index"]).view(-1).long()
    if len(fold_index) != len(pool.labels) or bool((fold_index < 0).any()):
        raise ValueError("inner pool fold_index is missing or invalid")
    unique_folds = sorted(int(value) for value in torch.unique(fold_index).tolist())
    outputs = {
        "selected_single": torch.full_like(pool.labels, float("nan")),
        "convex_mae": torch.full_like(pool.labels, float("nan")),
        "convex_shrinkage": torch.full_like(pool.labels, float("nan")),
    }
    rows = []
    for fold in unique_folds:
        holdout = torch.nonzero(fold_index == fold, as_tuple=False).view(-1)
        development = torch.nonzero(fold_index != fold, as_tuple=False).view(-1)
        dev_payload = {
            "labels": pool.labels.index_select(0, development).view(-1, 1),
            "anchor": pool.actions.index_select(0, development)[:, 0:1],
            "expert_predictions": pool.actions.index_select(0, development)[:, 1:],
            "sample_ids": [pool.sample_ids[i] for i in development.tolist()],
            "group_ids": [pool.group_ids[i] for i in development.tolist()],
            "action_names": ACTION_NAMES,
        }
        dev_pool = normalize_pool(dev_payload)
        selected = best_single_action(dev_pool)
        weights = fit_consensus_weights(dev_pool, config)
        holdout_actions = pool.actions.index_select(0, holdout)
        outputs["selected_single"][holdout] = holdout_actions[:, selected]
        for name in ("convex_mae", "convex_shrinkage"):
            outputs[name][holdout] = predict_weighted(holdout_actions, weights[name])
        for name, prediction in outputs.items():
            local_prediction = prediction.index_select(0, holdout)
            metrics = prediction_metrics(
                local_prediction,
                pool.labels.index_select(0, holdout),
                pool.actions.index_select(0, holdout)[:, 0],
            )
            rows.append(
                {
                    "inner_fold": fold,
                    "strategy": name,
                    "selected_single_action": ACTION_NAMES[selected],
                    **_scalar_metrics(metrics),
                }
            )
    if any(not torch.isfinite(value).all() for value in outputs.values()):
        raise FloatingPointError("inner cross-fit predictions are incomplete")
    aggregate = {
        name: prediction_metrics(value, pool.labels, pool.actions[:, 0])
        for name, value in outputs.items()
    }
    return {"pool": pool, "predictions": outputs, "fold_rows": rows, "aggregate": aggregate}


def _scalar_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"prediction", "sample_gain"}
    }


def group_bootstrap_gain_interval(
    sample_gain: Sequence[float] | torch.Tensor,
    group_ids: Sequence[object],
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    gains = np.asarray(torch.as_tensor(sample_gain).view(-1).double().numpy())
    groups = np.asarray([str(value) for value in group_ids], dtype=object)
    if len(gains) != len(groups):
        raise ValueError("bootstrap gains/groups mismatch")
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    indices = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        total = 0.0
        count = 0
        for group in sampled:
            rows = indices[group]
            total += float(gains[rows].sum())
            count += int(len(rows))
        values[repetition] = total / max(1, count)
    return {
        "gain_ci_low": float(np.quantile(values, 0.025)),
        "gain_ci_high": float(np.quantile(values, 0.975)),
        "bootstrap_positive_probability": float((values > 0.0).mean()),
    }


def residual_correlation_rows(
    actions: torch.Tensor,
    labels: torch.Tensor,
    kind: str,
) -> list[Dict[str, object]]:
    residual = actions.detach().cpu().double() - labels.detach().cpu().double().view(-1, 1)
    if kind == "absolute_error":
        residual = residual.abs()
    elif kind != "signed_residual":
        raise ValueError("kind must be signed_residual or absolute_error")
    matrix = np.corrcoef(residual.numpy(), rowvar=False)
    rows = []
    for left, left_name in enumerate(ACTION_NAMES):
        for right, right_name in enumerate(ACTION_NAMES):
            rows.append(
                {
                    "kind": kind,
                    "left_action": left_name,
                    "right_action": right_name,
                    "correlation": float(matrix[left, right]),
                }
            )
    return rows


def strategy_predictions(
    inner_pool: PoolView,
    outer_pool: PoolView,
    config: ConsensusConfigV921,
) -> Dict[str, object]:
    weights = fit_consensus_weights(inner_pool, config)
    selected_index = best_single_action(inner_pool)
    predictions = {
        "anchor": predict_strategy(outer_pool, "anchor"),
        **{
            f"fixed_{name}": predict_strategy(
                outer_pool, "fixed_action", action_index=index
            )
            for index, name in enumerate(ACTION_NAMES[1:], start=1)
        },
        "inner_selected_single": predict_strategy(
            outer_pool, "fixed_action", action_index=selected_index
        ),
        "boundary_positive_mean": predict_strategy(
            outer_pool, "boundary_positive_mean"
        ),
        "all_action_mean": predict_strategy(outer_pool, "all_action_mean"),
        "all_action_median": predict_strategy(outer_pool, "all_action_median"),
        "trimmed_mean": predict_strategy(outer_pool, "trimmed_mean"),
        "convex_mae": predict_strategy(
            outer_pool, "convex_mae", weights=weights["convex_mae"]
        ),
        "convex_shrinkage": predict_strategy(
            outer_pool,
            "convex_shrinkage",
            weights=weights["convex_shrinkage"],
        ),
    }
    metrics = {
        name: prediction_metrics(value, outer_pool.labels, outer_pool.actions[:, 0])
        for name, value in predictions.items()
    }
    return {
        "weights": weights,
        "selected_single_index": selected_index,
        "selected_single_action": ACTION_NAMES[selected_index],
        "predictions": predictions,
        "metrics": metrics,
    }


def aggregate_prediction_frames(frame: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    strategies = sorted(
        column.removeprefix("prediction_")
        for column in frame.columns
        if column.startswith("prediction_")
    )
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
    anchor = torch.tensor(frame["prediction_anchor"].to_numpy(), dtype=torch.float32)
    result = {}
    for strategy in strategies:
        prediction = torch.tensor(
            frame[f"prediction_{strategy}"].to_numpy(), dtype=torch.float32
        )
        result[strategy] = _scalar_metrics(prediction_metrics(prediction, labels, anchor))
    return result


def success_gate(
    fold_metrics: pd.DataFrame,
    aggregate_metrics: Mapping[str, Mapping[str, float]],
    config: ConsensusConfigV921,
) -> Dict[str, object]:
    if PRIMARY_STRATEGY not in aggregate_metrics:
        raise ValueError("primary consensus strategy missing")
    primary = aggregate_metrics[PRIMARY_STRATEGY]
    selected = aggregate_metrics["inner_selected_single"]
    aggregate_improvement = float(selected["mae"] - primary["mae"])
    pivot = fold_metrics.pivot(
        index="outer_fold", columns="strategy", values="mae"
    )
    delta = pivot["inner_selected_single"] - pivot[PRIMARY_STRATEGY]
    nondegrading = int((delta >= -1e-12).sum())
    worst_degradation = float(max(0.0, (-delta).max()))
    passed = (
        aggregate_improvement >= float(config.required_gain_vs_selected_single)
        and nondegrading >= int(config.required_nondegrading_outer_folds)
        and worst_degradation <= float(config.max_worst_fold_degradation)
    )
    return {
        "primary_strategy": PRIMARY_STRATEGY,
        "aggregate_gain_vs_inner_selected_single": aggregate_improvement,
        "required_gain_vs_inner_selected_single": float(
            config.required_gain_vs_selected_single
        ),
        "nondegrading_outer_folds": nondegrading,
        "required_nondegrading_outer_folds": int(
            config.required_nondegrading_outer_folds
        ),
        "worst_fold_degradation": worst_degradation,
        "max_worst_fold_degradation": float(config.max_worst_fold_degradation),
        "passed": bool(passed),
        "interpretation": (
            "Failure means do not add a nonlinear consensus head; retain the "
            "inner-OOF-selected single model or a parameter-free average."
        ),
    }


__all__ = [
    "AUDIT_VERSION",
    "PRIMARY_STRATEGY",
    "ConsensusConfigV921",
    "validate_simplex",
    "fit_mae_simplex",
    "fit_shrunk_mae_simplex",
    "predict_weighted",
    "predict_strategy",
    "prediction_metrics",
    "best_single_action",
    "fit_consensus_weights",
    "inner_crossfit_consensus",
    "group_bootstrap_gain_interval",
    "residual_correlation_rows",
    "strategy_predictions",
    "aggregate_prediction_frames",
    "success_gate",
]
