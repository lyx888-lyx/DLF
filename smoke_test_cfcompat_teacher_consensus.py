"""Synthetic tests for the locked CFCompatKD teacher-consensus utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trains.singleTask.cfcompat_teacher_consensus_utils import (
    CANDIDATE_RUNS,
    CONSENSUS_RUN,
    FORMAL_SEEDS,
    MEAN_RUN,
    TEACHER_SEEDS,
    aggregate_candidate_gate,
    consensus_for_indices,
    consensus_statistics,
    make_consensus_cache,
    method_gate,
    verdict_from_gates,
)


def metric_row(seed, run, j_value):
    return {
        "Seed": int(seed),
        "Run": run,
        "J_valid": float(j_value),
        "valid_LAV_MAE": float(j_value + 0.001),
        "valid_LA_MAE": float(j_value - 0.001),
        "valid_LV_MAE": float(j_value),
        "valid_L_MAE": float(j_value + 0.001),
    }


def main():
    teacher_predictions = np.asarray(
        [
            [-1.0, 0.0, 1.0, 2.0],
            [-0.8, 0.2, 1.2, 1.8],
            [-1.2, -0.1, 0.9, 2.1],
            [-0.9, 0.1, 1.1, 2.2],
            [-1.1, -0.2, 0.8, 1.9],
        ],
        dtype=np.float64,
    )
    mean, variance, variance_median, weight = consensus_statistics(
        teacher_predictions
    )
    assert mean.shape == variance.shape == weight.shape == (4,)
    assert variance_median > 0.0
    assert np.isfinite(weight).all()
    assert ((weight > 0.0) & (weight <= 1.0)).all()
    np.testing.assert_allclose(
        weight,
        1.0 / (1.0 + variance / variance_median),
        atol=0.0,
        rtol=0.0,
    )

    metadata = pd.DataFrame(
        {
            "sample_index": np.arange(4, dtype=int),
            "sample_id": ["s0", "s1", "s2", "s3"],
            "label": [-1.0, 0.0, 1.0, 2.0],
        }
    )
    cache, cached_median = make_consensus_cache(
        metadata,
        {
            int(seed): teacher_predictions[position]
            for position, seed in enumerate(TEACHER_SEEDS)
        },
    )
    assert cached_median == variance_median
    cache_by_index = {
        int(row.sample_index): row._asdict()
        for row in cache.itertuples(index=False)
    }
    target, local_variance, local_weight = consensus_for_indices(
        cache_by_index,
        [3, 1],
        torch.device("cpu"),
        torch.float32,
    )
    np.testing.assert_allclose(target.numpy(), mean[[3, 1]], atol=1e-6, rtol=0)
    np.testing.assert_allclose(
        local_variance.numpy(), variance[[3, 1]], atol=1e-6, rtol=0
    )
    np.testing.assert_allclose(
        local_weight.numpy(), weight[[3, 1]], atol=1e-6, rtol=0
    )

    compatibility = torch.tensor([0.25, 0.75], dtype=torch.float32)
    mean_gate = method_gate(MEAN_RUN, compatibility, local_weight)
    consensus_gate = method_gate(CONSENSUS_RUN, compatibility, local_weight)
    torch.testing.assert_close(mean_gate, compatibility)
    torch.testing.assert_close(consensus_gate, compatibility * local_weight)
    assert torch.all(consensus_gate <= mean_gate)

    baselines = {
        1111: metric_row(1111, "baseline", 0.680),
        1114: metric_row(1114, "baseline", 0.670),
    }
    candidates = []
    epochs = []
    for run, values in (
        (MEAN_RUN, {1111: 0.671, 1114: 0.662}),
        (CONSENSUS_RUN, {1111: 0.669, 1114: 0.660}),
    ):
        for seed in FORMAL_SEEDS:
            candidates.append(metric_row(seed, run, values[seed]))
            for epoch, offset in ((1, 0.001), (2, 0.002), (3, 0.000)):
                epochs.append(
                    {
                        "Seed": int(seed),
                        "Run": run,
                        "Epoch": int(epoch),
                        "J_valid": float(values[seed] + offset),
                    }
                )
    gates = {
        run: aggregate_candidate_gate(run, candidates, baselines, epochs)
        for run in CANDIDATE_RUNS
    }
    assert gates[MEAN_RUN]["passed"]
    assert gates[CONSENSUS_RUN]["passed"]
    assert (
        verdict_from_gates(gates)
        == "PROMOTE_TEACHER_CONSENSUS_TO_MOSEI_SINGLE_SEED_VALID_SCREEN"
    )
    print("CFCompatKD teacher-consensus utility smoke test passed")


if __name__ == "__main__":
    main()
