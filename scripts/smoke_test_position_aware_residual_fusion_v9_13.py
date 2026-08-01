"""CPU smoke test for V9.13 validation-only fusion primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.SemanticCostCoachV99 import SPECIALIST_NAMES  # noqa: E402
from trains.singleTask.position_aware_residual_fusion_v913 import (  # noqa: E402
    FUSION_VERSION,
    apply_config,
    fit_validation_fusion,
    pool_tensors,
    position_aware_prediction,
    simplex_grid,
)


def synthetic_pool(n: int = 60):
    torch.manual_seed(913)
    labels = torch.linspace(-2.8, 2.8, n).view(-1, 1)
    anchor = labels + 0.20 * torch.sin(torch.arange(n).float()).view(-1, 1)
    centers = {
        "strong_negative": -2.25,
        "boundary": 0.0,
        "positive": 1.0,
        "strong_positive": 2.25,
    }
    experts = {}
    for offset, name in enumerate(SPECIALIST_NAMES):
        distance = torch.abs(labels - centers[name])
        correction = -0.65 * (anchor - labels) * torch.exp(-distance)
        prediction = anchor + correction + 0.01 * (offset + 1)
        experts[name] = {
            "prediction": prediction,
            "confidence": torch.exp(-distance),
        }
    return {
        "sample_ids": [f"sample-{index}" for index in range(n)],
        "labels": labels,
        "anchor": anchor,
        "experts": experts,
    }


def main():
    assert FUSION_VERSION == "position_aware_residual_fusion_v913_v1"
    pool = synthetic_pool()
    values = pool_tensors(pool)
    assert values["actions"].shape == (60, 5)
    assert values["confidence"].shape == (60, 4)

    prediction, weights = position_aware_prediction(
        values["actions"],
        values["confidence"],
        position_source="anchor",
        tau=0.75,
        rho=0.30,
        confidence_power=1.0,
    )
    assert prediction.shape == (60, 1)
    assert weights.shape == (60, 4)
    assert bool((weights >= 0.0).all())
    assert bool((weights.sum(dim=1) <= 0.300001).all())

    grid = simplex_grid(0.25)
    assert len(grid) == 70
    assert all(abs(sum(row) - 1.0) < 1e-8 for row in grid)

    fit = fit_validation_fusion(
        pool,
        beta_grid=(0.0, 0.25, 0.5, 0.75, 1.0),
        simplex_step=0.25,
        position_sources=("anchor", "median"),
        tau_grid=(0.5, 1.0),
        rho_grid=(0.1, 0.3),
        confidence_powers=(0.0, 1.0),
        minimum_validation_gain=0.0,
        maximum_validation_harm=1.0,
    )
    assert fit["selected_by_validation_only"] is True
    assert fit["simplex_candidate_count"] == 70
    selected_prediction, _ = apply_config(pool, fit["selected"]["config"])
    assert selected_prediction.shape == (60, 1)
    assert fit["selected"]["mae"] <= fit["anchor"]["mae"] + 1e-8

    print("V9.13 POSITION-AWARE RESIDUAL FUSION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
