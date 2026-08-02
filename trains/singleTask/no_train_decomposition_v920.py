"""No-training decomposition utilities for the V9.19 same-stack experiment.

This module only reads saved predictions/expert pools.  It separates:
1. fold-local expert capacity (sample oracle),
2. transfer of coarse true-region specialization,
3. router identifiability/calibration,
4. the effect of the pre-registered safety gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .model.SemanticCostCoachV99 import ACTION_NAMES

AUDIT_VERSION = "same_stack_no_train_decomposition_v920_v1"
REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)
SEMANTIC_ACTION_BY_REGION = (1, 0, 2, 3, 4)


@dataclass(frozen=True)
class PoolView:
    sample_ids: list[str]
    group_ids: list[str]
    labels: torch.Tensor
    actions: torch.Tensor
    regions: torch.Tensor


def region_index(labels: torch.Tensor) -> torch.Tensor:
    values = labels.detach().cpu().float().view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def normalize_pool(payload: Mapping[str, object]) -> PoolView:
    labels = payload["labels"].detach().cpu().float().view(-1)
    anchor = payload["anchor"].detach().cpu().float().view(-1, 1)
    experts = payload["expert_predictions"].detach().cpu().float()
    if experts.dim() == 3 and experts.size(-1) == 1:
        experts = experts.squeeze(-1)
    if experts.dim() != 2 or experts.size(1) != len(ACTION_NAMES) - 1:
        raise ValueError("expert_predictions must have shape [N,4] or [N,4,1]")
    actions = torch.cat([anchor, experts], dim=1)
    if len(actions) != len(labels) or not torch.isfinite(actions).all():
        raise ValueError("invalid pool action predictions")
    sample_ids = [str(value) for value in payload["sample_ids"]]
    group_ids = [str(value) for value in payload["group_ids"]]
    if len(sample_ids) != len(labels) or len(group_ids) != len(labels):
        raise ValueError("pool identifiers do not align with labels")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("pool contains duplicate sample IDs")
    action_names = tuple(payload.get("action_names", ACTION_NAMES))
    if action_names != tuple(ACTION_NAMES):
        raise ValueError(f"unexpected action order: {action_names}")
    return PoolView(
        sample_ids=sample_ids,
        group_ids=group_ids,
        labels=labels,
        actions=actions,
        regions=region_index(labels),
    )


def selection_metrics(
    actions: torch.Tensor,
    labels: torch.Tensor,
    selected_action: torch.Tensor,
) -> Dict[str, object]:
    actions = actions.detach().cpu().float()
    labels = labels.detach().cpu().float().view(-1)
    selected_action = selected_action.detach().cpu().long().view(-1)
    selected_prediction = actions.gather(1, selected_action.view(-1, 1)).view(-1)
    errors = torch.abs(actions - labels.view(-1, 1))
    anchor_error = errors[:, 0]
    selected_error = torch.abs(selected_prediction - labels)
    gain = anchor_error - selected_error
    triggered = selected_action != 0
    return {
        "anchor_mae": float(anchor_error.mean().item()),
        "mae": float(selected_error.mean().item()),
        "gain_vs_anchor": float(gain.mean().item()),
        "coverage": float(triggered.float().mean().item()),
        "trigger_precision": (
            float((gain[triggered] > 0).float().mean().item())
            if bool(triggered.any())
            else 0.0
        ),
        "mean_trigger_gain": (
            float(gain[triggered].mean().item()) if bool(triggered.any()) else 0.0
        ),
        "harm_over_010_rate": float((gain < -0.10).float().mean().item()),
        "large_gain_rate_010": float((gain > 0.10).float().mean().item()),
        "action_counts": {
            name: int((selected_action == index).sum().item())
            for index, name in enumerate(ACTION_NAMES)
        },
        "selected_action": selected_action,
        "selected_prediction": selected_prediction,
        "sample_gain": gain,
    }


def action_metric_rows(
    pool: PoolView, split: str, outer_fold: int
) -> list[Dict[str, object]]:
    errors = torch.abs(pool.actions - pool.labels.view(-1, 1))
    anchor_error = errors[:, 0]
    oracle_index = errors.argmin(dim=1)
    rows = []
    for action, name in enumerate(ACTION_NAMES):
        gain = anchor_error - errors[:, action]
        rows.append(
            {
                "split": split,
                "outer_fold": int(outer_fold),
                "action": name,
                "sample_count": len(pool.labels),
                "mae": float(errors[:, action].mean().item()),
                "gain_vs_anchor": float(gain.mean().item()),
                "win_rate": float((gain > 0).float().mean().item()),
                "large_gain_rate_010": float((gain > 0.10).float().mean().item()),
                "large_harm_rate_010": float((gain < -0.10).float().mean().item()),
                "sample_oracle_rate": float(
                    (oracle_index == action).float().mean().item()
                ),
            }
        )
    return rows


def region_action_rows(
    pool: PoolView, split: str, outer_fold: int
) -> list[Dict[str, object]]:
    errors = torch.abs(pool.actions - pool.labels.view(-1, 1))
    anchor_error = errors[:, 0]
    rows = []
    for region, region_name in enumerate(REGION_NAMES):
        mask = pool.regions == region
        for action, action_name in enumerate(ACTION_NAMES):
            if bool(mask.any()):
                gain = anchor_error[mask] - errors[mask, action]
                row = {
                    "mae": float(errors[mask, action].mean().item()),
                    "gain_vs_anchor": float(gain.mean().item()),
                    "win_rate": float((gain > 0).float().mean().item()),
                    "large_gain_rate_010": float(
                        (gain > 0.10).float().mean().item()
                    ),
                    "large_harm_rate_010": float(
                        (gain < -0.10).float().mean().item()
                    ),
                }
            else:
                row = {
                    "mae": float("nan"),
                    "gain_vs_anchor": float("nan"),
                    "win_rate": float("nan"),
                    "large_gain_rate_010": float("nan"),
                    "large_harm_rate_010": float("nan"),
                }
            rows.append(
                {
                    "split": split,
                    "outer_fold": int(outer_fold),
                    "region_index": region,
                    "region": region_name,
                    "sample_count": int(mask.sum().item()),
                    "action_index": action,
                    "action": action_name,
                    **row,
                }
            )
    return rows


def best_action_by_region(pool: PoolView) -> tuple[int, ...]:
    errors = torch.abs(pool.actions - pool.labels.view(-1, 1))
    mapping = []
    for region in range(len(REGION_NAMES)):
        mask = pool.regions == region
        if not bool(mask.any()):
            mapping.append(0)
        else:
            mapping.append(int(errors[mask].mean(dim=0).argmin().item()))
    return tuple(mapping)


def select_by_region(pool: PoolView, mapping: Sequence[int]) -> Dict[str, object]:
    if len(mapping) != len(REGION_NAMES):
        raise ValueError("region mapping must contain five action indices")
    lookup = torch.tensor(tuple(int(value) for value in mapping), dtype=torch.long)
    selected = lookup.index_select(0, pool.regions)
    return selection_metrics(pool.actions, pool.labels, selected)


def oracle_metrics(pool: PoolView) -> Dict[str, object]:
    errors = torch.abs(pool.actions - pool.labels.view(-1, 1))
    oracle_action = errors.argmin(dim=1)
    sample_oracle = selection_metrics(pool.actions, pool.labels, oracle_action)
    semantic = select_by_region(pool, SEMANTIC_ACTION_BY_REGION)
    posthoc_mapping = best_action_by_region(pool)
    posthoc_region = select_by_region(pool, posthoc_mapping)
    return {
        "sample_oracle": sample_oracle,
        "semantic_region_oracle": semantic,
        "posthoc_region_mapping": posthoc_mapping,
        "posthoc_region_oracle": posthoc_region,
    }


def frame_actions(frame: pd.DataFrame) -> torch.Tensor:
    columns = [f"prediction_{name}" for name in ACTION_NAMES]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    return torch.tensor(frame[columns].to_numpy(), dtype=torch.float32)


def frame_expected_costs(frame: pd.DataFrame) -> torch.Tensor:
    columns = [f"expected_cost_{name}" for name in ACTION_NAMES]
    missing = [name for name in columns if name not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    return torch.tensor(frame[columns].to_numpy(), dtype=torch.float32)


def selected_indices_from_names(values: Sequence[object]) -> torch.Tensor:
    action_to_index = {name: index for index, name in enumerate(ACTION_NAMES)}
    try:
        return torch.tensor(
            [action_to_index[str(value)] for value in values], dtype=torch.long
        )
    except KeyError as error:
        raise ValueError(f"unknown selected action: {error}") from error


def actual_gated_metrics(frame: pd.DataFrame) -> Dict[str, object]:
    actions = frame_actions(frame)
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
    selected = selected_indices_from_names(frame["selected_action"].tolist())
    return selection_metrics(actions, labels, selected)


def counterfactual_router(
    frame: pd.DataFrame,
    fold_summary: Mapping[str, object],
) -> Dict[str, object]:
    """Apply the frozen outer policy while ignoring the fold-level reject switch."""
    actions = frame_actions(frame)
    expected = frame_expected_costs(frame)
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
    policy = fold_summary["fixed_policy"]
    allowed_names = tuple(policy["allowed_specialists"])
    action_to_index = {name: index for index, name in enumerate(ACTION_NAMES)}
    allowed = torch.tensor(
        [action_to_index[name] for name in allowed_names], dtype=torch.long
    )
    local_cost = expected.index_select(1, allowed)
    local_offset = local_cost.argmin(dim=1)
    proposed = allowed.index_select(0, local_offset)
    predicted_gain = expected[:, 0] - local_cost.min(dim=1).values
    confidence = torch.tensor(
        frame["region_confidence"].to_numpy(), dtype=torch.float32
    )
    cutoff_value = fold_summary["outer_holdout_metrics"].get("gain_cutoff")
    cutoff = float("inf") if cutoff_value is None else float(cutoff_value)
    threshold = float(policy["min_region_confidence"])
    trigger = (predicted_gain >= cutoff) & (confidence >= threshold)
    selected = torch.where(trigger, proposed, torch.zeros_like(proposed))
    frozen = selection_metrics(actions, labels, selected)

    all_action = expected.argmin(dim=1)
    expected_argmin = selection_metrics(actions, labels, all_action)
    proposed_metrics = selection_metrics(actions, labels, proposed)
    proposed_actual_gain = proposed_metrics["sample_gain"]
    stored_predicted = torch.tensor(
        frame["predicted_gain"].to_numpy(), dtype=torch.float32
    )
    if not torch.allclose(predicted_gain, stored_predicted, atol=2e-5):
        raise ValueError("stored predicted_gain differs from expected-cost calculation")

    return {
        "frozen_policy": frozen,
        "expected_cost_argmin": expected_argmin,
        "always_proposed_specialist": proposed_metrics,
        "proposed_action": proposed,
        "predicted_gain": predicted_gain,
        "proposed_actual_gain": proposed_actual_gain,
        "region_confidence": confidence,
        "gain_cutoff": cutoff,
        "confidence_threshold": threshold,
    }


def correlation_or_nan(left: np.ndarray, right: np.ndarray, method: str) -> float:
    frame = pd.DataFrame({"left": left, "right": right}).dropna()
    if len(frame) < 3 or frame["left"].nunique() < 2 or frame["right"].nunique() < 2:
        return float("nan")
    return float(frame["left"].corr(frame["right"], method=method))


def routing_diagnostics(
    frame: pd.DataFrame,
    counterfactual: Mapping[str, object],
) -> Dict[str, float]:
    predicted = counterfactual["predicted_gain"].numpy()
    actual = counterfactual["proposed_actual_gain"].numpy()
    proposed = counterfactual["proposed_action"]
    actions = frame_actions(frame)
    labels = torch.tensor(frame["label"].to_numpy(), dtype=torch.float32)
    oracle = torch.abs(actions - labels.view(-1, 1)).argmin(dim=1)
    region_prob_columns = [f"prob_{name}" for name in REGION_NAMES]
    probabilities = torch.tensor(
        frame[region_prob_columns].to_numpy(), dtype=torch.float32
    )
    true_region = torch.tensor(frame["true_region"].to_numpy(), dtype=torch.long)
    predicted_region = probabilities.argmax(dim=1)
    return {
        "predicted_gain_pearson": correlation_or_nan(predicted, actual, "pearson"),
        "predicted_gain_spearman": correlation_or_nan(predicted, actual, "spearman"),
        "proposed_positive_rate": float((actual > 0).mean()),
        "proposed_mean_actual_gain": float(actual.mean()),
        "proposed_harm_over_010_rate": float((actual < -0.10).mean()),
        "proposed_matches_sample_oracle": float(
            (proposed == oracle).float().mean().item()
        ),
        "region_accuracy": float(
            (predicted_region == true_region).float().mean().item()
        ),
        "region_adjacent_accuracy": float(
            ((predicted_region - true_region).abs() <= 1).float().mean().item()
        ),
        "mean_max_region_probability": float(
            probabilities.max(dim=1).values.mean().item()
        ),
    }


def predicted_gain_bin_rows(
    frame: pd.DataFrame,
    counterfactual: Mapping[str, object],
    outer_fold: int,
    bins: int = 5,
) -> list[Dict[str, object]]:
    predicted = counterfactual["predicted_gain"].numpy()
    actual = counterfactual["proposed_actual_gain"].numpy()
    data = pd.DataFrame(
        {
            "predicted_gain": predicted,
            "actual_gain": actual,
            "region_confidence": counterfactual["region_confidence"].numpy(),
        }
    )
    try:
        data["bin"] = pd.qcut(
            data["predicted_gain"], q=int(bins), labels=False, duplicates="drop"
        )
    except ValueError:
        data["bin"] = 0
    rows = []
    for bin_index, group in data.groupby("bin", dropna=False):
        rows.append(
            {
                "outer_fold": int(outer_fold),
                "gain_bin": int(bin_index) if pd.notna(bin_index) else -1,
                "sample_count": int(len(group)),
                "predicted_gain_min": float(group["predicted_gain"].min()),
                "predicted_gain_max": float(group["predicted_gain"].max()),
                "mean_predicted_gain": float(group["predicted_gain"].mean()),
                "mean_actual_gain": float(group["actual_gain"].mean()),
                "actual_positive_rate": float((group["actual_gain"] > 0).mean()),
                "actual_harm_over_010_rate": float(
                    (group["actual_gain"] < -0.10).mean()
                ),
                "mean_region_confidence": float(
                    group["region_confidence"].mean()
                ),
            }
        )
    return rows


def compact_metrics(metrics: Mapping[str, object]) -> Dict[str, object]:
    excluded = {"selected_action", "selected_prediction", "sample_gain"}
    return {key: value for key, value in metrics.items() if key not in excluded}


__all__ = [
    "AUDIT_VERSION",
    "ACTION_NAMES",
    "REGION_NAMES",
    "SEMANTIC_ACTION_BY_REGION",
    "PoolView",
    "normalize_pool",
    "selection_metrics",
    "action_metric_rows",
    "region_action_rows",
    "best_action_by_region",
    "select_by_region",
    "oracle_metrics",
    "actual_gated_metrics",
    "counterfactual_router",
    "routing_diagnostics",
    "predicted_gain_bin_rows",
    "compact_metrics",
]
