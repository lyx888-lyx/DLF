"""Synthetic smoke tests for the CFCompatKD mechanism-audit utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trains.singleTask.cfcompat_mechanism_utils import (
    MAX_SAMPLES_PER_VIDEO,
    calibration_stats,
    gradient_pair_stats,
    infer_mechanism_flags,
    parse_video_id,
    rank_compatibility_proxy,
    select_probe_positions,
    summarize_mode_deltas,
)


def main():
    assert parse_video_id("abc$_$12") == "abc"
    assert parse_video_id("abc[12]") == "abc"

    rows = []
    labels = (-2.0, 0.0, 2.0)
    for index in range(120):
        rows.append({
            "dataset_position": index,
            "sample_index": index,
            "sample_id": "video{}$_${}".format(index // 2, index),
            "video_id": "video{}".format(index // 2),
            "label": labels[index % 3],
        })
    selected = select_probe_positions(
        pd.DataFrame(rows), target_count=64, max_per_video=2
    )
    assert len(selected) == 64
    assert selected.sample_index.nunique() == 64
    assert selected.groupby("video_id").size().max() <= MAX_SAMPLES_PER_VIDEO

    proxy = rank_compatibility_proxy([0.4, 0.1, 0.2, 0.3])
    assert np.all((proxy > 0) & (proxy < 1))
    assert proxy[1] > proxy[0]

    left = [torch.tensor([1.0, 0.0]), None]
    right = [torch.tensor([-1.0, 0.0]), torch.tensor([2.0])]
    stats = gradient_pair_stats(left, right)
    assert stats["finite"]
    assert stats["conflict"]
    assert abs(stats["cosine"] + 1.0 / np.sqrt(5.0)) < 1e-7

    calibration = calibration_stats(
        [-1.0, 0.0, 1.0], [-0.5, 0.0, 0.5]
    )
    assert abs(calibration["slope_pred_on_label"] - 0.5) < 1e-9

    mode_rows = pd.DataFrame([
        {"Seed": 1111, "Mode": "LAV", "delta_abs_error": -0.2},
        {"Seed": 1111, "Mode": "LAV", "delta_abs_error": 0.1},
    ])
    summary = summarize_mode_deltas(mode_rows)
    assert len(summary) == 1
    assert abs(summary.iloc[0].mean_delta_abs_error + 0.05) < 1e-9

    shrinkage = pd.DataFrame([
        {
            "prediction_std_ratio_sam_to_baseline": 0.8,
            "prediction_abs_mean_ratio_sam_to_baseline": 0.8,
        },
        {
            "prediction_std_ratio_sam_to_baseline": 0.9,
            "prediction_abs_mean_ratio_sam_to_baseline": 0.9,
        },
    ])
    gradient_delta = pd.DataFrame([
        {
            "Seed": 1111,
            "Pair": "missing_vs_kd",
            "ParameterGroup": "fusion_head",
            "conflict_fraction_change_sam_minus_baseline": -0.2,
        },
        {
            "Seed": 1114,
            "Pair": "missing_vs_kd",
            "ParameterGroup": "fusion_head",
            "conflict_fraction_change_sam_minus_baseline": -0.1,
        },
    ])
    flags = infer_mechanism_flags(
        {
            "spearman_delta_J_proxy": 0.1,
            "same_direction_fraction": 0.4,
        },
        shrinkage,
        gradient_delta,
    )
    assert flags["prediction_shrinkage_supported"]
    assert not flags["sample_effect_consistent_across_seeds"]
    assert flags["sam_gradient_conflict_change_consistent"]

    print("CFCompatKD mechanism utility smoke tests passed")


if __name__ == "__main__":
    main()
