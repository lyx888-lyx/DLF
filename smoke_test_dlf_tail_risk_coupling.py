"""Synthetic smoke tests for DLF tail-risk coupling utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trains.singleTask.dlf_tail_risk_utils import (
    FORMAL_SEEDS,
    MODES,
    add_risk_events,
    bootstrap_diagnostics,
    compute_bin_and_run_metrics,
    coupling_gate,
    joint_video_bootstrap,
    tail_head_definition,
)


def synthetic_distribution():
    counts = {-3: 10, -2: 14, -1: 80, 0: 100, 1: 90, 2: 50, 3: 12}
    rows = []
    for sentiment_bin, count in counts.items():
        rows.append({
            "Dataset": "mosi",
            "Split": "train",
            "LabelFamily": "sentiment_7",
            "Class": sentiment_bin,
            "Count": count,
        })
    return pd.DataFrame(rows)


def synthetic_predictions():
    labels = np.asarray(
        [-3, -3, -2, -2, -1, -1, 0, 0, 1, 1, 2, 2, 3, 3],
        dtype=float,
    )
    bins = labels.astype(int)
    sample_indices = np.arange(len(labels), dtype=int)
    rows = []
    tail = {-3, -2, 3}
    for seed in FORMAL_SEEDS:
        for mode in MODES:
            for index, (label, sentiment_bin) in enumerate(zip(labels, bins)):
                if int(sentiment_bin) in tail:
                    prediction = -label if label != 0 else 2.5
                    prediction += 0.01 * (seed - FORMAL_SEEDS[0])
                else:
                    prediction = label + 0.05
                # Both dense synthetic videos contain every sentiment bin.
                video_id = "video{}".format(index % 2)
                rows.append({
                    "Seed": int(seed),
                    "Split": "valid",
                    "Mode": mode,
                    "sample_index": int(sample_indices[index]),
                    "sample_id": "{}[{}]".format(video_id, index),
                    "video_id": video_id,
                    "label": float(label),
                    "prediction": float(prediction),
                    "sentiment_bin": int(sentiment_bin),
                })
    return pd.DataFrame(rows)


def sparse_video_events(events, definition):
    """Create draws that are sometimes undefined to test rejection sampling."""
    sparse = events.copy()
    tail = set(int(value) for value in definition["tail_bins"])
    head = set(int(value) for value in definition["head_bins"])

    def video_for_bin(value):
        value = int(value)
        if value in tail:
            return "tail_video"
        if value in head:
            return "head_video"
        return "middle_video"

    sparse["video_id"] = sparse.sentiment_bin.map(video_for_bin)
    sparse["sample_id"] = [
        "{}[{}]".format(video, index)
        for video, index in zip(sparse.video_id, sparse.sample_index)
    ]
    return sparse


def main():
    distribution = synthetic_distribution()
    definition = tail_head_definition(distribution)
    assert set(definition["tail_bins"]) == {-3, -2, 3}
    assert set(definition["head_bins"]) == {-1, 0, 1}
    events = add_risk_events(synthetic_predictions())
    assert len(events) == len(FORMAL_SEEDS) * len(MODES) * 14
    assert events.high_cost_event.astype(int).sum() > 0
    bins, runs = compute_bin_and_run_metrics(events, definition)
    assert len(bins) == len(FORMAL_SEEDS) * len(MODES) * 7
    assert len(runs) == len(FORMAL_SEEDS) * len(MODES)
    assert (runs.tail_head_macro_mae_gap.astype(float) > 0).all()
    assert (runs.tail_head_high_cost_rate_gap.astype(float) > 0).all()

    first = joint_video_bootstrap(events, definition, replicates=100, seed=123)
    second = joint_video_bootstrap(events, definition, replicates=100, seed=123)
    pd.testing.assert_frame_equal(first, second)
    assert np.isfinite(
        first[
            [
                "mean_tail_head_macro_mae_gap",
                "mean_tail_head_high_cost_rate_gap",
            ]
        ].to_numpy(dtype=float)
    ).all()
    diagnostics = bootstrap_diagnostics(first)
    assert diagnostics["valid_replicates"] == 100
    assert diagnostics["invalid_draw_count"] == 0

    # The sparse construction omits all Tail or all Head bins in some draws.
    # The implementation must reject those draws and still return 100 finite,
    # deterministic accepted replicates.
    sparse = sparse_video_events(events, definition)
    sparse_first = joint_video_bootstrap(
        sparse, definition, replicates=100, seed=456
    )
    sparse_second = joint_video_bootstrap(
        sparse, definition, replicates=100, seed=456
    )
    pd.testing.assert_frame_equal(sparse_first, sparse_second)
    sparse_diagnostics = bootstrap_diagnostics(sparse_first)
    assert sparse_diagnostics["valid_replicates"] == 100
    assert sparse_diagnostics["invalid_draw_count"] > 0
    assert sparse_diagnostics["total_draw_attempts"] > 100
    assert np.isfinite(
        sparse_first[
            [
                "mean_tail_head_macro_mae_gap",
                "mean_tail_head_high_cost_rate_gap",
            ]
        ].to_numpy(dtype=float)
    ).all()

    gate = coupling_gate(runs, first, long_tail_present=True)
    assert gate["mean_tail_head_macro_mae_gap"] > 0
    assert gate["mean_tail_head_high_cost_rate_gap"] > 0
    assert gate["bootstrap_diagnostics"]["valid_replicates"] == 100
    print("DLF tail-risk coupling utility smoke test passed")


if __name__ == "__main__":
    main()
