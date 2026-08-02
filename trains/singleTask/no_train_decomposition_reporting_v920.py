"""Reporting helpers for the V9.20 no-training decomposition audit."""

from __future__ import annotations

from typing import Dict, Mapping

import numpy as np
import pandas as pd
import torch

from .no_train_decomposition_v920 import (
    ACTION_NAMES,
    REGION_NAMES,
    SEMANTIC_ACTION_BY_REGION,
    PoolView,
)


def align_frame(frame: pd.DataFrame, pool: PoolView) -> pd.DataFrame:
    result = frame.copy()
    result["sample_id"] = result["sample_id"].astype(str)
    if result["sample_id"].duplicated().any():
        raise ValueError("prediction CSV contains duplicate sample IDs")
    indexed = result.set_index("sample_id", drop=False)
    if set(indexed.index) != set(pool.sample_ids):
        raise ValueError("pool/prediction sample-ID sets differ")
    return indexed.loc[pool.sample_ids].reset_index(drop=True)


def mapping_rows(
    outer_fold: int,
    inner_pool: PoolView,
    outer_pool: PoolView,
    mapping: tuple[int, ...],
) -> list[Dict[str, object]]:
    inner_error = torch.abs(inner_pool.actions - inner_pool.labels.view(-1, 1))
    outer_error = torch.abs(outer_pool.actions - outer_pool.labels.view(-1, 1))
    rows = []
    for region, action in enumerate(mapping):
        inner_mask = inner_pool.regions == region
        outer_mask = outer_pool.regions == region
        rows.append(
            {
                "outer_fold": int(outer_fold),
                "region_index": region,
                "region": REGION_NAMES[region],
                "development_selected_action_index": int(action),
                "development_selected_action": ACTION_NAMES[action],
                "inner_region_count": int(inner_mask.sum().item()),
                "inner_selected_mae": float(inner_error[inner_mask, action].mean())
                if bool(inner_mask.any())
                else float("nan"),
                "inner_anchor_mae": float(inner_error[inner_mask, 0].mean())
                if bool(inner_mask.any())
                else float("nan"),
                "outer_region_count": int(outer_mask.sum().item()),
                "outer_selected_mae": float(outer_error[outer_mask, action].mean())
                if bool(outer_mask.any())
                else float("nan"),
                "outer_anchor_mae": float(outer_error[outer_mask, 0].mean())
                if bool(outer_mask.any())
                else float("nan"),
                "outer_gain_vs_anchor": float(
                    (outer_error[outer_mask, 0] - outer_error[outer_mask, action]).mean()
                )
                if bool(outer_mask.any())
                else float("nan"),
            }
        )
    return rows


def _gather(pool: PoolView, indices: torch.Tensor) -> torch.Tensor:
    return pool.actions.gather(1, indices.view(-1, 1)).view(-1)


def build_outer_sample_frame(
    frame: pd.DataFrame,
    pool: PoolView,
    outer_fold: int,
    development_mapping: tuple[int, ...],
    posthoc_mapping: tuple[int, ...],
    counterfactual: Mapping[str, object],
    actual: Mapping[str, object],
) -> pd.DataFrame:
    errors = torch.abs(pool.actions - pool.labels.view(-1, 1))
    sample_oracle = errors.argmin(dim=1)
    semantic = torch.tensor(SEMANTIC_ACTION_BY_REGION).index_select(0, pool.regions)
    dev_region = torch.tensor(development_mapping).index_select(0, pool.regions)
    posthoc = torch.tensor(posthoc_mapping).index_select(0, pool.regions)
    frozen = counterfactual["frozen_policy"]
    expected_argmin = counterfactual["expected_cost_argmin"]

    result = frame.copy()
    result.insert(0, "outer_fold", int(outer_fold))
    strategies = {
        "sample_oracle": (sample_oracle, _gather(pool, sample_oracle)),
        "semantic_region": (semantic, _gather(pool, semantic)),
        "dev_locked_region": (dev_region, _gather(pool, dev_region)),
        "posthoc_region": (posthoc, _gather(pool, posthoc)),
        "counterfactual_router": (
            frozen["selected_action"],
            frozen["selected_prediction"],
        ),
        "expected_cost_argmin": (
            expected_argmin["selected_action"],
            expected_argmin["selected_prediction"],
        ),
    }
    for prefix, (indices, predictions) in strategies.items():
        result[f"{prefix}_action"] = [ACTION_NAMES[i] for i in indices.tolist()]
        result[f"{prefix}_prediction"] = predictions.numpy()
    result["counterfactual_router_triggered"] = (
        frozen["selected_action"] != 0
    ).numpy()
    result["proposed_specialist_action"] = [
        ACTION_NAMES[i] for i in counterfactual["proposed_action"].tolist()
    ]
    result["proposed_specialist_actual_gain"] = counterfactual[
        "proposed_actual_gain"
    ].numpy()
    result["recomputed_predicted_gain"] = counterfactual["predicted_gain"].numpy()
    result["actual_gated_prediction_recomputed"] = actual[
        "selected_prediction"
    ].numpy()
    return result


def strategy_metrics(
    samples: pd.DataFrame,
    prediction_column: str,
    action_column: str,
) -> Dict[str, float]:
    anchor_error = (samples["anchor_prediction"] - samples["label"]).abs()
    selected_error = (samples[prediction_column] - samples["label"]).abs()
    gain = anchor_error - selected_error
    triggered = samples[action_column].astype(str) != "anchor"
    return {
        "anchor_mae": float(anchor_error.mean()),
        "mae": float(selected_error.mean()),
        "gain_vs_anchor": float(gain.mean()),
        "coverage": float(triggered.mean()),
        "trigger_precision": float((gain[triggered] > 0).mean())
        if triggered.any()
        else 0.0,
        "mean_trigger_gain": float(gain[triggered].mean())
        if triggered.any()
        else 0.0,
        "harm_over_010_rate": float((gain < -0.10).mean()),
        "large_gain_rate_010": float((gain > 0.10).mean()),
    }


def aggregate_strategies(samples: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    specifications = {
        "actual_gated_router": ("selected_prediction", "selected_action"),
        "sample_oracle": ("sample_oracle_prediction", "sample_oracle_action"),
        "semantic_region_oracle": (
            "semantic_region_prediction",
            "semantic_region_action",
        ),
        "development_locked_true_region_oracle": (
            "dev_locked_region_prediction",
            "dev_locked_region_action",
        ),
        "posthoc_true_region_oracle": (
            "posthoc_region_prediction",
            "posthoc_region_action",
        ),
        "counterfactual_frozen_router": (
            "counterfactual_router_prediction",
            "counterfactual_router_action",
        ),
        "expected_cost_argmin": (
            "expected_cost_argmin_prediction",
            "expected_cost_argmin_action",
        ),
    }
    return {
        name: strategy_metrics(samples, prediction, action)
        for name, (prediction, action) in specifications.items()
    }


def weighted_region_consistency(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (region, action), group in frame.groupby(["region", "action"], sort=False):
        valid = group[group["sample_count"] > 0]
        weights = valid["sample_count"].astype(float)
        rows.append(
            {
                "region": region,
                "action": action,
                "folds_with_samples": int(len(valid)),
                "positive_gain_folds": int((valid["gain_vs_anchor"] > 0).sum()),
                "negative_gain_folds": int((valid["gain_vs_anchor"] < 0).sum()),
                "mean_fold_gain": float(valid["gain_vs_anchor"].mean()),
                "std_fold_gain": float(valid["gain_vs_anchor"].std(ddof=0)),
                "weighted_gain": float(
                    np.average(valid["gain_vs_anchor"], weights=weights)
                ),
                "weighted_mae": float(np.average(valid["mae"], weights=weights)),
                "total_samples": int(weights.sum()),
            }
        )
    return pd.DataFrame(rows)


def diagnose(
    folds: pd.DataFrame,
    strategies: Mapping[str, Mapping[str, float]],
    aggregate_actions: pd.DataFrame,
) -> tuple[Dict[str, object], Dict[str, object]]:
    sample_gain = strategies["sample_oracle"]["gain_vs_anchor"]
    region_gain = strategies["development_locked_true_region_oracle"][
        "gain_vs_anchor"
    ]
    router_gain = strategies["counterfactual_frozen_router"]["gain_vs_anchor"]
    actual_gain = strategies["actual_gated_router"]["gain_vs_anchor"]
    region_positive = int((folds["outer_dev_locked_region_oracle_gain"] > 0).sum())
    router_positive = int((folds["outer_counterfactual_router_gain"] > 0).sum())
    router_harm = strategies["counterfactual_frozen_router"]["harm_over_010_rate"]
    router_passes = router_gain >= 0.005 and router_positive >= 4 and router_harm <= 0.05
    gate_verdict = (
        "potentially_overconservative"
        if router_passes
        else "protective_or_not_the_primary_problem"
    )
    if sample_gain < 0.005:
        bottleneck = "fold_local_expert_capacity_is_too_small"
    elif region_gain <= 0 or region_positive < 3:
        bottleneck = "true_region_specialization_does_not_transfer_across_folds"
    elif not router_passes:
        bottleneck = "router_expected_gain_does_not_identify_real_expert_gain"
    else:
        bottleneck = "safety_gate_is_likely_overconservative"
    best = aggregate_actions.loc[aggregate_actions["mae"].idxmin()].to_dict()
    best = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in best.items()
    }
    decomposition = {
        "sample_oracle_capacity_gain": sample_gain,
        "development_locked_true_region_gain": region_gain,
        "counterfactual_frozen_router_gain": router_gain,
        "actual_gated_router_gain": actual_gain,
        "capacity_to_region_loss": sample_gain - region_gain,
        "region_to_router_loss": region_gain - router_gain,
        "router_to_gate_loss": router_gain - actual_gain,
        "region_retention_of_sample_oracle": region_gain / sample_gain
        if sample_gain > 0
        else None,
        "router_retention_of_region_oracle": router_gain / region_gain
        if region_gain > 0
        else None,
        "router_positive_outer_folds": router_positive,
        "region_oracle_positive_outer_folds": region_positive,
        "gate_verdict": gate_verdict,
        "primary_bottleneck": bottleneck,
    }
    return decomposition, best


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].copy()
    for column in data.columns:
        if pd.api.types.is_float_dtype(data[column]):
            data[column] = data[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
    lines = [
        "|" + "|".join(columns) + "|",
        "|" + "|".join(["---"] * len(columns)) + "|",
    ]
    lines.extend(
        "|" + "|".join(str(value) for value in row) + "|"
        for row in data.itertuples(index=False, name=None)
    )
    return "\n".join(lines)


__all__ = [
    "align_frame",
    "mapping_rows",
    "build_outer_sample_frame",
    "strategy_metrics",
    "aggregate_strategies",
    "weighted_region_consistency",
    "diagnose",
    "markdown_table",
]
