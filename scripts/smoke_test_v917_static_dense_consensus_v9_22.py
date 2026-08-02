"""Synthetic smoke test for V9.22 V9.17 frozen-expert consensus audit."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SPECIALIST_NAMES,
)
from trains.singleTask.v917_static_dense_consensus_v922 import (
    CompatibilityConfigV922,
    fit_full_validation,
    normalize_v917_pool,
    strategy_predictions,
    validation_group_crossfit,
)


def make_pool(n: int, prefix: str):
    labels = torch.linspace(-2.5, 2.5, n).view(-1, 1)
    index = torch.arange(n, dtype=torch.float32).view(-1, 1)
    anchor = labels + 0.30 + 0.03 * torch.sin(index * 0.2)
    residuals = {
        "strong_negative": -0.20 + 0.04 * torch.cos(index * 0.13),
        "boundary": 0.10 + 0.03 * torch.sin(index * 0.17),
        "positive": -0.08 + 0.03 * torch.cos(index * 0.11),
        "strong_positive": 0.18 + 0.04 * torch.sin(index * 0.07),
    }
    return {
        "sample_ids": [f"{prefix}_{i}|segment" for i in range(n)],
        "group_ids": [f"{prefix}_g{i // 6}" for i in range(n)],
        "labels": labels,
        "anchor": anchor,
        "experts": {
            name: {"prediction": labels + residuals[name]}
            for name in SPECIALIST_NAMES
        },
        "action_names": ACTION_NAMES,
    }


def main():
    validation = normalize_v917_pool(make_pool(60, "valid"), "synthetic_valid")
    test = normalize_v917_pool(make_pool(35, "test"), "synthetic_test")
    config = CompatibilityConfigV922(
        shrinkage_lambda=0.01,
        validation_group_folds=5,
        bootstrap_repetitions=50,
        bootstrap_seed=1111,
    )
    crossfit = validation_group_crossfit(validation, config)
    assert set(crossfit["aggregate"]) == {
        "validation_selected_single",
        "convex_mae",
        "convex_shrinkage",
    }
    fit = fit_full_validation(validation, config)
    for values in fit["weights"].values():
        assert len(values) == len(ACTION_NAMES)
        assert min(values) >= -1e-8
        assert abs(float(sum(values)) - 1.0) < 1e-7
    predictions = strategy_predictions(fit, test)
    assert len(predictions["convex_shrinkage"]) == len(test.labels)
    assert torch.isfinite(predictions["convex_shrinkage"]).all()
    assert not torch.allclose(
        predictions["convex_shrinkage"], test.actions[:, 0]
    )
    print("V9.22 V9.17 STATIC CONSENSUS SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
