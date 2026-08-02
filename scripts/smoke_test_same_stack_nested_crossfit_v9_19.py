"""Dependency-light smoke test for V9.19 merge and fixed policy."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES
from trains.singleTask.same_stack_nested_router_v919 import (
    FixedPolicyV919,
    apply_fixed_policy,
    calibrate_gain_cutoff,
    merge_pools,
)


def make_pool(indices, inner_fold):
    n = len(indices)
    labels = torch.tensor([-2.2, -1.0, 0.0, 1.0, 2.2][:n]).view(-1, 1)
    anchor = labels + 0.30
    expert = torch.stack(
        [
            labels
            + torch.tensor([0.05, 0.40, 0.50, 0.50, 0.50])[:n].view(-1, 1),
            labels
            + torch.tensor([0.50, 0.20, 0.05, 0.15, 0.50])[:n].view(-1, 1),
            labels
            + torch.tensor([0.50, 0.15, 0.15, 0.05, 0.50])[:n].view(-1, 1),
            labels
            + torch.tensor([0.50, 0.50, 0.50, 0.40, 0.05])[:n].view(-1, 1),
        ],
        dim=1,
    )
    return {
        "sample_indices": list(indices),
        "sample_ids": [f"g{value}|s" for value in indices],
        "group_ids": [f"g{value}" for value in indices],
        "labels": labels,
        "anchor": anchor,
        "function_space": torch.zeros(n, 4),
        "expert_predictions": expert,
        "expert_confidences": torch.ones(n, 4, 1) * 0.8,
        "expert_corrections": expert - anchor.unsqueeze(1),
        "action_names": ACTION_NAMES,
        "feature_space": "auxiliary_prediction_logits_v1",
        "inner_fold": inner_fold,
    }


def main():
    left = make_pool([0, 2, 4], 0)
    right = make_pool([1, 3], 1)
    merged = merge_pools([left, right], [0, 1, 2, 3, 4])
    assert merged["sample_indices"] == [0, 1, 2, 3, 4]
    assert merged["expert_predictions"].shape == (5, 4, 1)

    probabilities = torch.eye(5)
    cost = torch.tensor(
        [
            [0.30, 0.05, 0.50, 0.50, 0.50],
            [0.30, 0.40, 0.20, 0.15, 0.50],
            [0.30, 0.50, 0.05, 0.15, 0.50],
            [0.30, 0.50, 0.15, 0.05, 0.40],
            [0.30, 0.50, 0.50, 0.50, 0.05],
        ]
    )
    expected = probabilities @ cost
    actions = torch.cat(
        [merged["anchor"], merged["expert_predictions"].squeeze(-1)],
        dim=1,
    )
    policy = FixedPolicyV919(
        gain_margin=0.03,
        min_region_confidence=0.55,
        max_coverage=0.40,
    )
    cutoff = calibrate_gain_cutoff(probabilities, expected, policy)
    result = apply_fixed_policy(
        probabilities,
        expected,
        actions,
        merged["labels"],
        policy,
        gain_cutoff=cutoff,
    )
    assert result["coverage"] <= 0.400001
    assert result["gain_vs_anchor"] > 0
    assert sum(result["action_counts"].values()) == 5
    print("V9.19 FULL SAME-STACK NESTED CROSSFIT SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
