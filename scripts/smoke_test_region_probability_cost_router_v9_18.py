"""CPU smoke test for V9.18 ordinal region probability × action-cost routing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.OrdinalRegionCostRouterV918 import (  # noqa: E402
    OrdinalRegionCostRouterV918,
    cumulative_targets,
    ordinal_region_loss,
    region_probabilities_from_logits,
)
from trains.singleTask.region_probability_cost_router_v918 import (  # noqa: E402
    action_cost_matrix,
    apply_policy,
    expected_action_costs,
    normalize_router_pool,
)


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def synthetic_pool():
    labels = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).view(-1, 1)
    anchor = torch.tensor([-1.2, -0.7, 0.6, 0.2, 1.4]).view(-1, 1)
    expert_predictions = torch.tensor(
        [
            [-1.9, -1.0, -0.8, -1.1],
            [-0.8, -0.9, -1.0, -0.7],
            [0.4, 0.0, 0.1, 0.5],
            [0.1, 0.7, 1.0, 0.4],
            [1.3, 1.1, 1.2, 1.9],
        ]
    ).unsqueeze(-1)
    corrections = expert_predictions.squeeze(-1) - anchor
    return {
        "sample_ids": [f"video{i}|segment0" for i in range(5)],
        "group_ids": [f"video{i}" for i in range(5)],
        "labels": labels,
        "anchor": anchor,
        "function_space": torch.arange(40, dtype=torch.float32).view(5, 8) / 10.0,
        "expert_predictions": expert_predictions,
        "expert_confidences": torch.full((5, 4, 1), 0.7),
        "expert_corrections": corrections.unsqueeze(-1),
        "action_names": (
            "anchor", "strong_negative", "boundary", "positive", "strong_positive"
        ),
        "fold_index": torch.arange(5),
    }


def main():
    pool = normalize_router_pool(synthetic_pool(), "synthetic_strict_oof")
    require(pool["router_features"].shape == (5, 28), "unexpected router feature dimension")
    require(pool["region_index"].tolist() == [0, 1, 2, 3, 4], "region mapping failed")

    features = pool["router_features"]
    mean = features.mean(dim=0)
    scale = features.std(dim=0, unbiased=False).clamp_min(1e-4)
    model = OrdinalRegionCostRouterV918(mean, scale, hidden_dim=16, dropout=0.0)
    output = model(features)
    probabilities = region_probabilities_from_logits(output["logits"], 1.0)
    require(probabilities.shape == (5, 5), "probability shape mismatch")
    require(
        torch.allclose(probabilities.sum(dim=1), torch.ones(5), atol=1e-6),
        "probabilities do not sum to one",
    )
    require(bool((probabilities >= 0).all()), "negative probability")
    cutpoints = output["cutpoints"]
    require(bool((cutpoints[1:] > cutpoints[:-1]).all()), "cutpoints are not ordered")

    targets = cumulative_targets(pool["labels"])
    require(
        targets.tolist()
        == [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ],
        "ordinal targets failed",
    )
    loss = ordinal_region_loss(output, pool["labels"], torch.ones(4))["total"]
    loss.backward()
    require(torch.isfinite(loss), "loss is non-finite")

    indices = torch.arange(5)
    costs, counts = action_cost_matrix(
        pool["actions_2d"], pool["labels"], pool["region_index"], indices
    )
    require(costs.shape == (5, 5), "cost matrix shape mismatch")
    require(counts.tolist() == [1, 1, 1, 1, 1], "region counts mismatch")

    perfect_probabilities = torch.eye(5)
    expected = expected_action_costs(perfect_probabilities, costs)
    routed = apply_policy(
        perfect_probabilities,
        expected,
        pool["actions_2d"],
        pool["labels"],
        gain_margin=0.0,
        min_region_confidence=0.9,
    )
    require(
        routed["mae"] < routed["anchor_mae"],
        "perfect-region cost routing did not improve",
    )
    require(routed["coverage"] > 0.0, "router never triggered")
    require(
        routed["action_counts"]["strong_negative"] == 1,
        "strong-negative action missing",
    )
    require(routed["action_counts"]["boundary"] >= 1, "boundary action missing")
    require(routed["action_counts"]["positive"] >= 1, "positive action missing")
    require(
        routed["action_counts"]["strong_positive"] == 1,
        "strong-positive action missing",
    )

    print("V9.18 REGION-PROBABILITY ACTION-COST SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
