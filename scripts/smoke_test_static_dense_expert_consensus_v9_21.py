"""Synthetic smoke test for V9.21 static dense expert consensus."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.model.GlobalDenseExpertConsensusV921 import (
    GlobalDenseExpertConsensusV921,
)
from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    fit_consensus_weights,
    group_bootstrap_gain_interval,
    inner_crossfit_consensus,
    prediction_metrics,
    strategy_predictions,
)


def make_payload(n: int, offset: int = 0, include_folds: bool = True):
    index = torch.arange(n, dtype=torch.float32)
    labels = torch.linspace(-2.5, 2.5, n)
    residuals = torch.stack(
        [
            0.30 + 0.10 * torch.sin(index * 0.17),
            -0.24 + 0.08 * torch.cos(index * 0.11),
            0.13 + 0.05 * torch.sin(index * 0.23),
            -0.11 + 0.05 * torch.cos(index * 0.19),
            0.22 + 0.06 * torch.sin(index * 0.07),
        ],
        dim=1,
    )
    actions = labels.view(-1, 1) + residuals
    payload = {
        "sample_ids": [f"g{(i + offset) // 4}|s{i + offset}" for i in range(n)],
        "group_ids": [f"g{(i + offset) // 4}" for i in range(n)],
        "labels": labels.view(-1, 1),
        "anchor": actions[:, 0:1],
        "expert_predictions": actions[:, 1:].unsqueeze(-1),
        "action_names": ACTION_NAMES,
    }
    if include_folds:
        payload["fold_index"] = torch.arange(n, dtype=torch.long) % 3
    return payload


def main():
    config = ConsensusConfigV921(
        shrinkage_lambda=0.01,
        bootstrap_repetitions=200,
        bootstrap_seed=17,
    )
    inner_payload = make_payload(180, 0, True)
    outer_payload = make_payload(90, 1000, False)
    inner_pool = normalize_pool(inner_payload)
    outer_pool = normalize_pool(outer_payload)

    weights = fit_consensus_weights(inner_pool, config)
    for name, value in weights.items():
        assert value.shape == (len(ACTION_NAMES),), name
        assert np.all(value >= -1e-8), name
        assert abs(float(value.sum()) - 1.0) < 1e-8, name
        module = GlobalDenseExpertConsensusV921(value)
        assert module(outer_pool.actions).shape == (len(outer_pool.labels), 1)

    result = strategy_predictions(inner_pool, outer_pool, config)
    assert result["metrics"]["convex_mae"]["mae"] < result["metrics"]["anchor"]["mae"]
    assert result["metrics"]["convex_shrinkage"]["mae"] < result["metrics"]["anchor"]["mae"]
    assert result["selected_single_action"] in ACTION_NAMES

    crossfit = inner_crossfit_consensus(inner_payload, config)
    assert len(crossfit["fold_rows"]) == 9
    for prediction in crossfit["predictions"].values():
        assert torch.isfinite(prediction).all()

    primary = result["metrics"]["convex_shrinkage"]
    interval = group_bootstrap_gain_interval(
        primary["sample_gain"],
        outer_pool.group_ids,
        repetitions=200,
        seed=23,
    )
    assert interval["gain_ci_high"] >= interval["gain_ci_low"]
    assert 0.0 <= interval["bootstrap_positive_probability"] <= 1.0

    recomputed = prediction_metrics(
        result["predictions"]["convex_shrinkage"],
        outer_pool.labels,
        outer_pool.actions[:, 0],
    )
    assert abs(recomputed["mae"] - primary["mae"]) < 1e-8
    print("V9.21 STATIC DENSE EXPERT CONSENSUS SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
