"""Synthetic smoke tests for CFCompatKD video-aware sampling."""

from __future__ import annotations

import numpy as np

from trains.singleTask.video_aware_sampling_utils import (
    SAMPLES_PER_VIDEO,
    VideoAwareBatchSampler,
    batch_video_statistics,
    candidate_test_gate,
    candidate_valid_gate,
    parse_video_id,
)


def test_video_id_parser():
    assert parse_video_id("videoA$_$17") == "videoA"
    assert parse_video_id("speaker/topic[3]") == "speaker/topic"


def test_sampler_exact_coverage_and_determinism():
    videos = (
        ["v0"] * 9
        + ["v1"] * 8
        + ["v2"] * 7
        + ["v3"] * 6
        + ["v4"] * 5
    )
    left = VideoAwareBatchSampler(
        videos, batch_size=8, seed=1114, samples_per_video=SAMPLES_PER_VIDEO
    )
    right = VideoAwareBatchSampler(
        videos, batch_size=8, seed=1114, samples_per_video=SAMPLES_PER_VIDEO
    )
    left_batches = left.preview(epochs=2)
    right_batches = right.preview(epochs=2)
    assert left_batches == right_batches

    per_epoch = int(np.ceil(len(videos) / 8.0))
    for epoch in range(2):
        batches = left_batches[epoch * per_epoch : (epoch + 1) * per_epoch]
        flat = [index for batch in batches for index in batch]
        assert sorted(flat) == list(range(len(videos)))
        assert len(flat) == len(set(flat))
        assert all(len(batch) <= 8 for batch in batches)

    first_stats = batch_video_statistics(
        [videos[index] for index in left_batches[0]]
    )
    assert first_stats["video_count"] >= 2
    assert first_stats["repeated_sample_fraction"] > 0.5


def test_valid_and_test_gates():
    baseline = {
        "J_valid": 0.700,
        "valid_LAV_MAE": 0.700,
        "valid_LA_MAE": 0.700,
        "valid_LV_MAE": 0.700,
        "valid_L_MAE": 0.700,
        "J_test_at_valid_best": 0.710,
        "test_at_valid_best_LAV_MAE": 0.710,
        "test_at_valid_best_LA_MAE": 0.710,
        "test_at_valid_best_LV_MAE": 0.710,
        "test_at_valid_best_L_MAE": 0.710,
    }
    candidate = {
        "J_valid": 0.692,
        "valid_LAV_MAE": 0.691,
        "valid_LA_MAE": 0.693,
        "valid_LV_MAE": 0.693,
        "valid_L_MAE": 0.693,
        "J_test_at_valid_best": 0.704,
        "test_at_valid_best_LAV_MAE": 0.704,
        "test_at_valid_best_LA_MAE": 0.704,
        "test_at_valid_best_LV_MAE": 0.704,
        "test_at_valid_best_L_MAE": 0.704,
    }
    epochs = [
        {"J_valid": 0.696},
        {"J_valid": 0.695},
        {"J_valid": 0.692},
    ]
    valid_gate = candidate_valid_gate(candidate, baseline, epochs)
    assert valid_gate["passed"]
    test_gate = candidate_test_gate(candidate, baseline)
    assert test_gate["passed"]


def main():
    test_video_id_parser()
    test_sampler_exact_coverage_and_determinism()
    test_valid_and_test_gates()
    print("CFCompat video-aware sampling smoke test passed")


if __name__ == "__main__":
    main()
