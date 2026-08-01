"""Fixed expert-by-region capability audit for V9.17."""

from __future__ import annotations

from typing import Dict, Mapping, Sequence

import torch

from .model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SPECIALIST_NAMES,
    stack_action_predictions,
)
from .oof_group_splits_v92 import canonical_sample_id, conversation_group_id
from .role_conditioned_experts_v9 import REGION_NAMES, region_index

AUDIT_VERSION = "fixed_expert_region_audit_v917_v1"
DEFAULT_THRESHOLDS = (0.05, 0.10)
DESIGNATED_ACTION_BY_REGION = {
    "strong_negative": "strong_negative",
    "negative": "anchor",
    "boundary": "boundary",
    "positive": "positive",
    "strong_positive": "strong_positive",
}


def _column(value: torch.Tensor, name: str) -> torch.Tensor:
    value = value.detach().cpu().float()
    if value.dim() == 1:
        value = value.view(-1, 1)
    if value.dim() != 2 or value.size(1) != 1:
        raise ValueError(f"{name} must have shape [N,1]")
    if not torch.isfinite(value).all():
        raise FloatingPointError(f"{name} contains non-finite values")
    return value


def normalize_expert_pool(
    pool: Mapping[str, object],
    source_protocol: str,
) -> Dict[str, object]:
    """Convert strict-OOF or frozen deployment pools to one aligned action tensor."""
    required = {"sample_ids", "labels", "anchor"}
    if not required.issubset(pool):
        raise ValueError(f"expert pool missing {sorted(required - set(pool))}")

    sample_ids = [canonical_sample_id(value) for value in pool["sample_ids"]]
    if len(sample_ids) != len(set(sample_ids)):
        raise RuntimeError("duplicate sample IDs in expert pool")
    labels = _column(pool["labels"], "labels")
    anchor = _column(pool["anchor"], "anchor")

    if "expert_predictions" in pool:
        experts = pool["expert_predictions"].detach().cpu().float()
        if experts.dim() == 2:
            experts = experts.unsqueeze(-1)
        if experts.dim() != 3 or experts.shape[1:] != (len(SPECIALIST_NAMES), 1):
            raise ValueError("expert_predictions must have shape [N,4,1]")
        actions = stack_action_predictions(anchor, experts)
    elif "experts" in pool:
        expert_columns = []
        for name in SPECIALIST_NAMES:
            if name not in pool["experts"]:
                raise ValueError(f"frozen pool missing expert: {name}")
            expert_columns.append(
                _column(pool["experts"][name]["prediction"], f"{name}_prediction")
            )
        experts = torch.stack(expert_columns, dim=1)
        actions = stack_action_predictions(anchor, experts)
    else:
        raise ValueError("pool has neither expert_predictions nor experts")

    if len(sample_ids) != len(labels) or len(actions) != len(labels):
        raise RuntimeError("sample, label, and action lengths differ")
    if not torch.isfinite(actions).all():
        raise FloatingPointError("action predictions contain non-finite values")

    supplied_names = tuple(pool.get("action_names", ACTION_NAMES))
    if supplied_names != tuple(ACTION_NAMES):
        raise ValueError(
            f"unexpected action order {supplied_names}; expected {ACTION_NAMES}"
        )

    group_ids = list(pool.get("group_ids", ()))
    if group_ids:
        if len(group_ids) != len(sample_ids):
            raise RuntimeError("group ID count does not match samples")
        group_ids = [str(value) for value in group_ids]
    else:
        group_ids = [conversation_group_id(value) for value in sample_ids]

    result: Dict[str, object] = {
        "sample_ids": sample_ids,
        "group_ids": group_ids,
        "labels": labels,
        "actions": actions,
        "action_names": tuple(ACTION_NAMES),
        "source_protocol": str(source_protocol),
    }
    if "fold_index" in pool:
        fold_index = pool["fold_index"].detach().cpu().long().view(-1)
        if len(fold_index) != len(sample_ids):
            raise RuntimeError("fold index count does not match samples")
        result["fold_index"] = fold_index
    return result


def grouped_bootstrap_mean(
    values: torch.Tensor,
    group_ids: Sequence[object],
    repeats: int,
    seed: int,
) -> Dict[str, float]:
    """Conversation-group bootstrap for mean expert advantage."""
    values = values.detach().cpu().float().view(-1)
    if len(values) != len(group_ids):
        raise ValueError("bootstrap values/group IDs length mismatch")
    if values.numel() == 0:
        return {
            "bootstrap_mean": float("nan"),
            "gain_ci_low": float("nan"),
            "gain_ci_high": float("nan"),
            "bootstrap_positive_rate": float("nan"),
        }

    mapping: Dict[str, list[int]] = {}
    for index, value in enumerate(group_ids):
        mapping.setdefault(str(value), []).append(int(index))
    groups = sorted(mapping)
    if not groups:
        raise RuntimeError("bootstrap has no groups")

    generator = torch.Generator().manual_seed(int(seed))
    means = []
    for _ in range(max(1, int(repeats))):
        sampled = torch.randint(
            len(groups),
            (len(groups),),
            generator=generator,
        ).tolist()
        indices = [
            index
            for group_index in sampled
            for index in mapping[groups[int(group_index)]]
        ]
        means.append(float(values[indices].mean().item()))
    samples = torch.tensor(means, dtype=torch.float32)
    return {
        "bootstrap_mean": float(samples.mean().item()),
        "gain_ci_low": float(torch.quantile(samples, 0.025).item()),
        "gain_ci_high": float(torch.quantile(samples, 0.975).item()),
        "bootstrap_positive_rate": float((samples > 0.0).float().mean().item()),
    }


def _metric_row(
    split: str,
    source_protocol: str,
    region_name: str,
    region_id: int,
    action_name: str,
    action_id: int,
    errors: torch.Tensor,
    advantages: torch.Tensor,
    oracle_index: torch.Tensor,
    mask: torch.Tensor,
    group_ids: Sequence[object],
    thresholds: Sequence[float],
    bootstrap_repeats: int,
    bootstrap_seed: int,
) -> Dict[str, object]:
    local_error = errors[mask, action_id]
    local_advantage = advantages[mask, action_id]
    local_groups = [
        group_ids[index]
        for index in torch.nonzero(mask, as_tuple=False).view(-1).tolist()
    ]
    anchor_mae = float(errors[mask, 0].mean().item())
    row: Dict[str, object] = {
        "split": str(split),
        "source_protocol": str(source_protocol),
        "region": str(region_name),
        "region_index": int(region_id),
        "sample_count": int(mask.sum().item()),
        "group_count": int(len(set(str(value) for value in local_groups))),
        "expert": str(action_name),
        "expert_index": int(action_id),
        "designated_for_region": bool(
            DESIGNATED_ACTION_BY_REGION[region_name] == action_name
        ),
        "anchor_mae": anchor_mae,
        "expert_mae": float(local_error.mean().item()),
        "gain_vs_anchor": float(local_advantage.mean().item()),
        "median_gain_vs_anchor": float(local_advantage.median().item()),
        "win_rate": float((local_advantage > 1e-8).float().mean().item()),
        "tie_rate": float((local_advantage.abs() <= 1e-8).float().mean().item()),
        "loss_rate": float((local_advantage < -1e-8).float().mean().item()),
        "oracle_best_rate": float(
            (oracle_index[mask] == int(action_id)).float().mean().item()
        ),
    }
    for threshold in thresholds:
        suffix = f"{float(threshold):.2f}".replace(".", "")
        row[f"large_gain_rate_{suffix}"] = float(
            (local_advantage > float(threshold)).float().mean().item()
        )
        row[f"large_harm_rate_{suffix}"] = float(
            (local_advantage < -float(threshold)).float().mean().item()
        )
    row.update(
        grouped_bootstrap_mean(
            local_advantage,
            local_groups,
            repeats=int(bootstrap_repeats),
            seed=int(bootstrap_seed),
        )
    )
    return row


def build_region_audit(
    normalized_pool: Mapping[str, object],
    split: str,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    bootstrap_repeats: int = 500,
    bootstrap_seed: int = 917,
) -> Dict[str, object]:
    """Build long metrics, designated-region summary, and sample-level rows."""
    labels = normalized_pool["labels"].detach().cpu().float().view(-1)
    actions = normalized_pool["actions"].detach().cpu().float().squeeze(-1)
    if actions.shape != (len(labels), len(ACTION_NAMES)):
        raise ValueError("normalized actions must have shape [N,5]")
    errors = torch.abs(actions - labels.view(-1, 1))
    advantages = errors[:, :1] - errors
    regions = region_index(labels)
    oracle_index = errors.argmin(dim=1)
    oracle_prediction = actions.gather(1, oracle_index.view(-1, 1)).view(-1)
    oracle_gain = errors[:, 0] - errors.gather(
        1, oracle_index.view(-1, 1)
    ).view(-1)

    metrics_rows = []
    designated_rows = []
    for region_id, region_name in enumerate(REGION_NAMES):
        mask = regions == int(region_id)
        if not bool(mask.any()):
            raise RuntimeError(f"split {split} has no samples in region {region_name}")
        for action_id, action_name in enumerate(ACTION_NAMES):
            row = _metric_row(
                split=split,
                source_protocol=normalized_pool["source_protocol"],
                region_name=region_name,
                region_id=region_id,
                action_name=action_name,
                action_id=action_id,
                errors=errors,
                advantages=advantages,
                oracle_index=oracle_index,
                mask=mask,
                group_ids=normalized_pool["group_ids"],
                thresholds=thresholds,
                bootstrap_repeats=bootstrap_repeats,
                bootstrap_seed=(
                    int(bootstrap_seed)
                    + 1009 * int(region_id)
                    + 9173 * int(action_id)
                ),
            )
            metrics_rows.append(row)
            if row["designated_for_region"]:
                designated_rows.append(
                    {
                        "split": row["split"],
                        "source_protocol": row["source_protocol"],
                        "region": row["region"],
                        "region_index": row["region_index"],
                        "sample_count": row["sample_count"],
                        "group_count": row["group_count"],
                        "designated_expert": row["expert"],
                        "anchor_mae": row["anchor_mae"],
                        "designated_expert_mae": row["expert_mae"],
                        "improvement": row["gain_vs_anchor"],
                        "win_rate": row["win_rate"],
                        "large_gain_rate_005": row.get("large_gain_rate_005"),
                        "large_gain_rate_010": row.get("large_gain_rate_010"),
                        "large_harm_rate_005": row.get("large_harm_rate_005"),
                        "large_harm_rate_010": row.get("large_harm_rate_010"),
                        "gain_ci_low": row["gain_ci_low"],
                        "gain_ci_high": row["gain_ci_high"],
                        "bootstrap_positive_rate": row[
                            "bootstrap_positive_rate"
                        ],
                        "oracle_best_rate": row["oracle_best_rate"],
                    }
                )

    global_rows = []
    all_mask = torch.ones(len(labels), dtype=torch.bool)
    for action_id, action_name in enumerate(ACTION_NAMES):
        global_rows.append(
            _metric_row(
                split=split,
                source_protocol=normalized_pool["source_protocol"],
                region_name="all",
                region_id=-1,
                action_name=action_name,
                action_id=action_id,
                errors=errors,
                advantages=advantages,
                oracle_index=oracle_index,
                mask=all_mask,
                group_ids=normalized_pool["group_ids"],
                thresholds=thresholds,
                bootstrap_repeats=bootstrap_repeats,
                bootstrap_seed=(
                    int(bootstrap_seed) + 104729 + 9173 * int(action_id)
                ),
            )
        )

    sample_rows = []
    fold_index = normalized_pool.get("fold_index")
    for index, sample_id in enumerate(normalized_pool["sample_ids"]):
        region_id = int(regions[index].item())
        row: Dict[str, object] = {
            "split": str(split),
            "source_protocol": str(normalized_pool["source_protocol"]),
            "sample_id": str(sample_id),
            "group_id": str(normalized_pool["group_ids"][index]),
            "label": float(labels[index].item()),
            "true_region": REGION_NAMES[region_id],
            "true_region_index": region_id,
            "oracle_expert": ACTION_NAMES[int(oracle_index[index].item())],
            "oracle_prediction": float(oracle_prediction[index].item()),
            "oracle_gain_vs_anchor": float(oracle_gain[index].item()),
        }
        if fold_index is not None:
            row["outer_fold"] = int(fold_index[index].item())
        for action_id, action_name in enumerate(ACTION_NAMES):
            row[f"{action_name}_prediction"] = float(
                actions[index, action_id].item()
            )
            row[f"{action_name}_abs_error"] = float(
                errors[index, action_id].item()
            )
            row[f"{action_name}_gain_vs_anchor"] = float(
                advantages[index, action_id].item()
            )
        sample_rows.append(row)

    oracle_region_rows = []
    for region_id, region_name in enumerate(REGION_NAMES):
        mask = regions == int(region_id)
        oracle_region_rows.append(
            {
                "split": str(split),
                "source_protocol": str(normalized_pool["source_protocol"]),
                "region": region_name,
                "region_index": int(region_id),
                "sample_count": int(mask.sum().item()),
                "anchor_mae": float(errors[mask, 0].mean().item()),
                "sample_oracle_mae": float(
                    errors[mask]
                    .gather(1, oracle_index[mask].view(-1, 1))
                    .mean()
                    .item()
                ),
                "sample_oracle_gain": float(oracle_gain[mask].mean().item()),
                **{
                    f"oracle_count_{action_name}": int(
                        (oracle_index[mask] == action_id).sum().item()
                    )
                    for action_id, action_name in enumerate(ACTION_NAMES)
                },
            }
        )

    return {
        "metrics_rows": metrics_rows,
        "global_rows": global_rows,
        "designated_rows": designated_rows,
        "sample_rows": sample_rows,
        "oracle_region_rows": oracle_region_rows,
    }


def specialization_rows(
    metrics_rows: Sequence[Mapping[str, object]],
    global_rows: Sequence[Mapping[str, object]],
) -> list[Dict[str, object]]:
    """Summarize each action's designated and empirically best true region."""
    result = []
    for split in sorted({str(row["split"]) for row in metrics_rows}):
        split_metrics = [row for row in metrics_rows if row["split"] == split]
        split_global = [row for row in global_rows if row["split"] == split]
        for action_name in ACTION_NAMES:
            local = [row for row in split_metrics if row["expert"] == action_name]
            global_row = next(
                row for row in split_global if row["expert"] == action_name
            )
            best = max(
                local,
                key=lambda row: (
                    float(row["gain_vs_anchor"]),
                    float(row["win_rate"]),
                    -int(row["region_index"]),
                ),
            )
            designated_region = next(
                (
                    region
                    for region, action in DESIGNATED_ACTION_BY_REGION.items()
                    if action == action_name
                ),
                None,
            )
            designated = next(
                (
                    row
                    for row in local
                    if row["region"] == designated_region
                ),
                None,
            )
            result.append(
                {
                    "split": split,
                    "expert": action_name,
                    "designated_region": designated_region,
                    "designated_region_gain": (
                        float(designated["gain_vs_anchor"])
                        if designated is not None
                        else None
                    ),
                    "designated_region_win_rate": (
                        float(designated["win_rate"])
                        if designated is not None
                        else None
                    ),
                    "designated_region_gain_ci_low": (
                        float(designated["gain_ci_low"])
                        if designated is not None
                        else None
                    ),
                    "designated_region_gain_ci_high": (
                        float(designated["gain_ci_high"])
                        if designated is not None
                        else None
                    ),
                    "best_true_region": best["region"],
                    "best_region_gain": float(best["gain_vs_anchor"]),
                    "best_region_win_rate": float(best["win_rate"]),
                    "positive_gain_region_count": int(
                        sum(float(row["gain_vs_anchor"]) > 0.0 for row in local)
                    ),
                    "global_mae": float(global_row["expert_mae"]),
                    "global_gain_vs_anchor": float(
                        global_row["gain_vs_anchor"]
                    ),
                    "global_gain_ci_low": float(global_row["gain_ci_low"]),
                    "global_gain_ci_high": float(global_row["gain_ci_high"]),
                    "global_bootstrap_positive_rate": float(
                        global_row["bootstrap_positive_rate"]
                    ),
                }
            )
    return result


__all__ = [
    "ACTION_NAMES",
    "SPECIALIST_NAMES",
    "REGION_NAMES",
    "AUDIT_VERSION",
    "DEFAULT_THRESHOLDS",
    "DESIGNATED_ACTION_BY_REGION",
    "normalize_expert_pool",
    "grouped_bootstrap_mean",
    "build_region_audit",
    "specialization_rows",
]
