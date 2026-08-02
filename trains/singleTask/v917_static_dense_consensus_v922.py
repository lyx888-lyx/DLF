"""Static dense consensus compatibility audit for the frozen V9.17 expert pool.

No anchor or expert is trained. Validation predictions fit one global convex
weight vector, which is frozen before exploratory evaluation on the already
observed MOSI Test pool. There is no router or sample-dependent weight.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import torch

from .fixed_expert_region_audit_v917 import normalize_expert_pool
from .model.SemanticCostCoachV99 import ACTION_NAMES
from .no_train_decomposition_v920 import PoolView, region_index
from .static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    best_single_action,
    fit_consensus_weights,
    prediction_metrics,
    predict_strategy,
)

AUDIT_VERSION = "v917_static_dense_consensus_compatibility_v922_v1"
PRIMARY_STRATEGY = "convex_shrinkage"


@dataclass(frozen=True)
class CompatibilityConfigV922:
    shrinkage_lambda: float = 0.01
    validation_group_folds: int = 5
    bootstrap_repetitions: int = 2000
    bootstrap_seed: int = 1111


def normalize_v917_pool(
    payload: Mapping[str, object], source_protocol: str
) -> PoolView:
    normalized = normalize_expert_pool(payload, source_protocol)
    actions = normalized["actions"].detach().cpu().float()
    if actions.dim() == 3 and actions.size(-1) == 1:
        actions = actions.squeeze(-1)
    labels = normalized["labels"].detach().cpu().float().view(-1)
    if actions.shape != (len(labels), len(ACTION_NAMES)):
        raise ValueError("V9.17 normalized actions must have shape [N,5]")
    return PoolView(
        sample_ids=[str(value) for value in normalized["sample_ids"]],
        group_ids=[str(value) for value in normalized["group_ids"]],
        labels=labels,
        actions=actions,
        regions=region_index(labels),
    )


def subset_pool(pool: PoolView, indices: Sequence[int]) -> PoolView:
    index = torch.tensor([int(value) for value in indices], dtype=torch.long)
    return PoolView(
        sample_ids=[pool.sample_ids[i] for i in index.tolist()],
        group_ids=[pool.group_ids[i] for i in index.tolist()],
        labels=pool.labels.index_select(0, index),
        actions=pool.actions.index_select(0, index),
        regions=pool.regions.index_select(0, index),
    )


def balanced_group_folds(group_ids: Sequence[object], fold_count: int) -> list[list[int]]:
    groups: Dict[str, list[int]] = {}
    for index, value in enumerate(group_ids):
        groups.setdefault(str(value), []).append(index)
    if len(groups) < 2:
        raise ValueError("at least two conversation groups are required")
    k = max(2, min(int(fold_count), len(groups)))
    fold_groups: list[list[str]] = [[] for _ in range(k)]
    fold_sizes = [0] * k
    ordered = sorted(groups, key=lambda key: (-len(groups[key]), key))
    for group in ordered:
        target = min(range(k), key=lambda fold: (fold_sizes[fold], fold))
        fold_groups[target].append(group)
        fold_sizes[target] += len(groups[group])
    folds = [
        sorted(index for group in assigned for index in groups[group])
        for assigned in fold_groups
    ]
    if any(not fold for fold in folds):
        raise RuntimeError("group fold construction produced an empty fold")
    return folds


def scalar_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"prediction", "sample_gain"}
    }


def fit_full_validation(
    validation: PoolView, config: CompatibilityConfigV922
) -> Dict[str, object]:
    base_config = ConsensusConfigV921(
        shrinkage_lambda=float(config.shrinkage_lambda),
        bootstrap_repetitions=int(config.bootstrap_repetitions),
        bootstrap_seed=int(config.bootstrap_seed),
    )
    return {
        "weights": fit_consensus_weights(validation, base_config),
        "selected_single_index": best_single_action(validation),
    }


def strategy_predictions(
    fit: Mapping[str, object], pool: PoolView
) -> Dict[str, torch.Tensor]:
    selected = int(fit["selected_single_index"])
    weights = fit["weights"]
    return {
        "anchor": predict_strategy(pool, "anchor"),
        **{
            f"fixed_{name}": predict_strategy(
                pool, "fixed_action", action_index=index
            )
            for index, name in enumerate(ACTION_NAMES[1:], start=1)
        },
        "validation_selected_single": predict_strategy(
            pool, "fixed_action", action_index=selected
        ),
        "boundary_positive_mean": predict_strategy(
            pool, "boundary_positive_mean"
        ),
        "all_action_mean": predict_strategy(pool, "all_action_mean"),
        "all_action_median": predict_strategy(pool, "all_action_median"),
        "trimmed_mean": predict_strategy(pool, "trimmed_mean"),
        "convex_mae": predict_strategy(
            pool, "convex_mae", weights=weights["convex_mae"]
        ),
        "convex_shrinkage": predict_strategy(
            pool, "convex_shrinkage", weights=weights["convex_shrinkage"]
        ),
    }


def evaluate_predictions(
    pool: PoolView, predictions: Mapping[str, torch.Tensor]
) -> Dict[str, Dict[str, object]]:
    return {
        name: prediction_metrics(
            prediction, pool.labels, pool.actions[:, 0]
        )
        for name, prediction in predictions.items()
    }


def validation_group_crossfit(
    validation: PoolView, config: CompatibilityConfigV922
) -> Dict[str, object]:
    folds = balanced_group_folds(
        validation.group_ids, config.validation_group_folds
    )
    strategies = (
        "validation_selected_single",
        "convex_mae",
        "convex_shrinkage",
    )
    predictions = {
        name: torch.full_like(validation.labels, float("nan"))
        for name in strategies
    }
    fold_rows = []
    all_indices = set(range(len(validation.labels)))
    for fold, holdout in enumerate(folds):
        development = sorted(all_indices - set(holdout))
        dev_pool = subset_pool(validation, development)
        holdout_pool = subset_pool(validation, holdout)
        fit = fit_full_validation(dev_pool, config)
        local = strategy_predictions(fit, holdout_pool)
        holdout_index = torch.tensor(holdout, dtype=torch.long)
        for strategy in strategies:
            predictions[strategy][holdout_index] = local[strategy]
            metrics = prediction_metrics(
                local[strategy],
                holdout_pool.labels,
                holdout_pool.actions[:, 0],
            )
            fold_rows.append(
                {
                    "validation_fold": fold,
                    "strategy": strategy,
                    "development_count": len(development),
                    "holdout_count": len(holdout),
                    "selected_single_action": ACTION_NAMES[
                        int(fit["selected_single_index"])
                    ],
                    **scalar_metrics(metrics),
                }
            )
    if any(not torch.isfinite(value).all() for value in predictions.values()):
        raise FloatingPointError("validation cross-fit predictions are incomplete")
    aggregate = {
        strategy: prediction_metrics(
            prediction, validation.labels, validation.actions[:, 0]
        )
        for strategy, prediction in predictions.items()
    }
    return {
        "predictions": predictions,
        "fold_rows": fold_rows,
        "aggregate": aggregate,
        "fold_indices": folds,
    }


def paired_group_bootstrap_difference(
    gain_a: Sequence[float] | torch.Tensor,
    gain_b: Sequence[float] | torch.Tensor,
    group_ids: Sequence[object],
    repetitions: int,
    seed: int,
) -> Dict[str, float]:
    left = np.asarray(torch.as_tensor(gain_a).view(-1).double().numpy())
    right = np.asarray(torch.as_tensor(gain_b).view(-1).double().numpy())
    groups = np.asarray([str(value) for value in group_ids], dtype=object)
    if not (len(left) == len(right) == len(groups)):
        raise ValueError("paired bootstrap inputs do not align")
    delta = left - right
    unique = np.asarray(sorted(set(groups.tolist())), dtype=object)
    mapping = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(int(seed))
    values = np.empty(int(repetitions), dtype=np.float64)
    for repetition in range(int(repetitions)):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        total = 0.0
        count = 0
        for group in sampled:
            rows = mapping[group]
            total += float(delta[rows].sum())
            count += int(len(rows))
        values[repetition] = total / max(1, count)
    return {
        "mean_gain_difference": float(delta.mean()),
        "difference_ci_low": float(np.quantile(values, 0.025)),
        "difference_ci_high": float(np.quantile(values, 0.975)),
        "positive_probability": float((values > 0).mean()),
    }


__all__ = [
    "AUDIT_VERSION",
    "PRIMARY_STRATEGY",
    "CompatibilityConfigV922",
    "normalize_v917_pool",
    "subset_pool",
    "balanced_group_folds",
    "scalar_metrics",
    "fit_full_validation",
    "strategy_predictions",
    "evaluate_predictions",
    "validation_group_crossfit",
    "paired_group_bootstrap_difference",
]
