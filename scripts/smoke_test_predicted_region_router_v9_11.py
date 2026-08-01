"""CPU structural smoke test for V9.11 predicted-region routing."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.PredictedRegionRouterV911 import (  # noqa: E402
    REGION_ROUTER_VERSION,
    SEMANTIC_REGION_ACTION_MAP,
    PredictedRegionRouterV911,
    action_indices_from_regions,
    predicted_region_router_loss,
    region_classification_metrics,
    region_index,
    route_by_regions,
)
from trains.singleTask.model.SemanticCostCoachV99 import (  # noqa: E402
    SIGNATURE_DIM,
    SPECIALIST_NAMES,
)
from trains.singleTask.predicted_region_router_crossfit_v911 import (  # noqa: E402
    estimate_empirical_region_map,
    pool_action_tensors,
)


def main():
    assert REGION_ROUTER_VERSION == "predicted_region_router_v1"
    assert SEMANTIC_REGION_ACTION_MAP == (1, 0, 2, 3, 4)
    torch.manual_seed(31)
    n = 40
    anchor = torch.linspace(-2.0, 2.0, n).view(-1, 1)
    experts = anchor.unsqueeze(1) + 0.30 * torch.randn(n, 4, 1)
    signatures = torch.randn(n, 4, SIGNATURE_DIM)
    signatures[:, :, 0] = experts[:, :, 0]
    signatures[:, :, 1] = experts[:, :, 0] - anchor
    signatures[:, :, 2] = signatures[:, :, 1].abs()
    function_space = torch.randn(n, 4)
    labels = anchor + 0.35 * torch.randn_like(anchor)

    strict_pool = {
        "anchor": anchor,
        "function_space": function_space,
        "expert_predictions": experts,
        "expert_signatures": signatures,
    }
    strict_features, strict_actions, strict_signatures = pool_action_tensors(
        strict_pool
    )
    frozen_pool = {
        "anchor": anchor,
        "function_space": function_space,
        "experts": {
            name: {
                "prediction": experts[:, index],
                "signature": signatures[:, index],
            }
            for index, name in enumerate(SPECIALIST_NAMES)
        },
    }
    frozen_features, frozen_actions, frozen_signatures = pool_action_tensors(
        frozen_pool
    )
    assert torch.allclose(strict_features, frozen_features)
    assert torch.allclose(strict_actions, frozen_actions)
    assert torch.allclose(strict_signatures, frozen_signatures)
    assert strict_actions.shape == (n, 5, 1)

    model = PredictedRegionRouterV911(
        input_dim=strict_features.size(1),
        hidden_dim=32,
        dropout=0.0,
    )
    output = model(strict_features, anchor)
    assert output["region_probs"].shape == (n, 5)
    assert torch.allclose(
        output["region_probs"].sum(dim=1), torch.ones(n), atol=1e-5
    )
    losses = predicted_region_router_loss(output, labels)
    losses["total"].backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    metrics = region_classification_metrics(output["region_probs"], labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0

    regions = region_index(labels)
    route = route_by_regions(strict_actions, regions, SEMANTIC_REGION_ACTION_MAP)
    expected_actions = action_indices_from_regions(
        regions, SEMANTIC_REGION_ACTION_MAP
    )
    assert torch.equal(route["action_indices"], expected_actions)
    assert route["prediction"].shape == (n, 1)

    empirical_map, rows = estimate_empirical_region_map(
        strict_actions, labels, range(n), prior_strength=5.0
    )
    assert len(empirical_map) == 5
    assert len(rows) == 5
    assert all(0 <= index < 5 for index in empirical_map)

    print("V9.11 PREDICTED REGION ROUTER SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
