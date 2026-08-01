"""Reporting and serialization helpers for V9.18."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping

import torch

from .model.SemanticCostCoachV99 import ACTION_NAMES
from .region_probability_cost_router_v918 import policy_result_for_pool


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, float) and (
        value != value or value in (float("inf"), float("-inf"))
    ):
        return None
    return value


def subset_pool(pool: Mapping[str, object], indices: torch.Tensor) -> Dict[str, object]:
    indices = indices.view(-1).long()
    result: Dict[str, object] = {}
    sample_count = len(pool["labels"])
    for key, value in pool.items():
        if isinstance(value, torch.Tensor) and value.dim() > 0 and len(value) == sample_count:
            result[key] = value.index_select(0, indices)
        elif key in {"sample_ids", "group_ids"}:
            result[key] = [value[index] for index in indices.tolist()]
        else:
            result[key] = value
    return result


def confusion_rows(split: str, probabilities: torch.Tensor, true_regions: torch.Tensor):
    predicted = probabilities.argmax(dim=1)
    rows = []
    for truth in range(5):
        mask = true_regions == truth
        for pred in range(5):
            count = int((mask & (predicted == pred)).sum().item())
            rows.append({
                "split": split,
                "true_region": truth,
                "predicted_region": pred,
                "count": count,
                "row_rate": count / max(1, int(mask.sum().item())),
            })
    return rows


def sample_rows(split: str, pool, result):
    rows = []
    probs = result["region_probabilities"]
    expected = result["expected_costs"]
    selected = result["selected_action"]
    labels = pool["labels"].view(-1)
    actions = pool["actions_2d"]
    true_regions = pool["region_index"]
    region_names = (
        "strong_negative", "negative", "boundary", "positive", "strong_positive"
    )
    for index, sample_id in enumerate(pool["sample_ids"]):
        action_index = int(selected[index].item())
        row = {
            "split": split,
            "sample_id": sample_id,
            "group_id": pool["group_ids"][index],
            "label": float(labels[index].item()),
            "true_region": int(true_regions[index].item()),
            "selected_action": ACTION_NAMES[action_index],
            "selected_action_index": action_index,
            "anchor_prediction": float(actions[index, 0].item()),
            "selected_prediction": float(result["selected_prediction"][index].item()),
            "predicted_gain": float(result["predicted_gain"][index].item()),
            "region_confidence": float(result["region_confidence"][index].item()),
            "triggered": bool(result["trigger"][index].item()),
        }
        for region, name in enumerate(region_names):
            row[f"prob_{name}"] = float(probs[index, region].item())
        for action, name in enumerate(ACTION_NAMES):
            row[f"prediction_{name}"] = float(actions[index, action].item())
            row[f"expected_cost_{name}"] = float(expected[index, action].item())
        rows.append(row)
    return rows


def anchor_fallback_result(model, pool, device, batch_size, temperature, cost_matrix):
    return policy_result_for_pool(
        model, pool, device, batch_size, temperature, cost_matrix, 1e9, 1.0
    )


__all__ = [
    "sha256", "jsonable", "subset_pool", "confusion_rows", "sample_rows",
    "anchor_fallback_result",
]
