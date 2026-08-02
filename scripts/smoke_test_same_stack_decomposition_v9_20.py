"""Dependency-light synthetic smoke test for V9.20 decomposition logic."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.no_train_decomposition_v920 import (  # noqa: E402
    ACTION_NAMES,
    REGION_NAMES,
    action_metric_rows,
    best_action_by_region,
    counterfactual_router,
    normalize_pool,
    oracle_metrics,
    predicted_gain_bin_rows,
    region_action_rows,
    routing_diagnostics,
    select_by_region,
)


def make_payload():
    labels = torch.tensor(
        [-2.2, -1.0, 0.0, 1.0, 2.2, -2.4, -0.8, 0.2, 1.2, 2.4]
    ).view(-1, 1)
    anchor = labels + 0.30
    expert_error = torch.tensor(
        [
            [0.05, 0.50, 0.50, 0.50],
            [0.40, 0.18, 0.12, 0.45],
            [0.45, 0.05, 0.12, 0.45],
            [0.45, 0.12, 0.05, 0.40],
            [0.45, 0.45, 0.40, 0.05],
            [0.05, 0.50, 0.50, 0.50],
            [0.40, 0.18, 0.12, 0.45],
            [0.45, 0.05, 0.12, 0.45],
            [0.45, 0.12, 0.05, 0.40],
            [0.45, 0.45, 0.40, 0.05],
        ]
    )
    experts = labels.unsqueeze(1) + expert_error.unsqueeze(-1)
    return {
        "sample_ids": [f"g{i // 2}|s{i}" for i in range(len(labels))],
        "group_ids": [f"g{i // 2}" for i in range(len(labels))],
        "labels": labels,
        "anchor": anchor,
        "expert_predictions": experts,
        "action_names": ACTION_NAMES,
    }


def make_frame(pool):
    probabilities = torch.eye(5).index_select(0, pool.regions)
    cost = torch.tensor(
        [
            [0.30, 0.05, 0.50, 0.50, 0.50],
            [0.30, 0.40, 0.18, 0.12, 0.45],
            [0.30, 0.45, 0.05, 0.12, 0.45],
            [0.30, 0.45, 0.12, 0.05, 0.40],
            [0.30, 0.45, 0.45, 0.40, 0.05],
        ]
    )
    expected = probabilities @ cost
    predicted_gain = expected[:, 0] - expected[:, 1:].min(dim=1).values
    rows = []
    for index in range(len(pool.labels)):
        row = {
            "sample_id": pool.sample_ids[index],
            "group_id": pool.group_ids[index],
            "label": float(pool.labels[index]),
            "true_region": int(pool.regions[index]),
            "selected_action": "anchor",
            "selected_prediction": float(pool.actions[index, 0]),
            "anchor_prediction": float(pool.actions[index, 0]),
            "predicted_gain": float(predicted_gain[index]),
            "region_confidence": float(probabilities[index].max()),
        }
        for region, name in enumerate(REGION_NAMES):
            row[f"prob_{name}"] = float(probabilities[index, region])
        for action, name in enumerate(ACTION_NAMES):
            row[f"prediction_{name}"] = float(pool.actions[index, action])
            row[f"expected_cost_{name}"] = float(expected[index, action])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    pool = normalize_pool(make_payload())
    assert pool.actions.shape == (10, 5)
    assert len(action_metric_rows(pool, "outer", 0)) == 5
    assert len(region_action_rows(pool, "outer", 0)) == 25

    mapping = best_action_by_region(pool)
    assert mapping == (1, 3, 2, 3, 4)
    mapped = select_by_region(pool, mapping)
    assert mapped["gain_vs_anchor"] > 0
    oracle = oracle_metrics(pool)
    assert oracle["sample_oracle"]["gain_vs_anchor"] >= mapped["gain_vs_anchor"]

    frame = make_frame(pool)
    summary = {
        "fixed_policy": {
            "allowed_specialists": list(ACTION_NAMES[1:]),
            "min_region_confidence": 0.55,
        },
        "outer_holdout_metrics": {"gain_cutoff": 0.03},
    }
    counterfactual = counterfactual_router(frame, summary)
    assert counterfactual["frozen_policy"]["gain_vs_anchor"] > 0
    assert counterfactual["frozen_policy"]["coverage"] == 1.0
    diagnostics = routing_diagnostics(frame, counterfactual)
    assert diagnostics["region_accuracy"] == 1.0
    assert diagnostics["predicted_gain_spearman"] > 0
    bins = predicted_gain_bin_rows(frame, counterfactual, 0, bins=3)
    assert len(bins) >= 1
    print("V9.20 NO-TRAINING DECOMPOSITION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
