"""Synthetic smoke tests for CFCompatKD Video-VREx utilities."""

from __future__ import annotations

import numpy as np
import torch

from trains.singleTask.video_vrex_utils import (
    VideoAwareBatchSampler,
    candidate_test_gate,
    candidate_valid_gate,
    parse_video_id,
    video_risk_variance,
)


def test_video_id_parser():
    assert parse_video_id("03bSnISJMiM$_$11") == "03bSnISJMiM"
    assert parse_video_id("video-name_abc[17]") == "video-name_abc"
    try:
        parse_video_id("unsegmented")
    except ValueError:
        pass
    else:
        raise AssertionError("Unsegmented IDs must not silently define domains.")


def test_sampler_exact_coverage_and_determinism():
    videos = []
    for video, count in (("a", 13), ("b", 12), ("c", 11), ("d", 10), ("e", 9)):
        videos.extend([video] * count)
    first = VideoAwareBatchSampler(videos, batch_size=16, seed=1111, samples_per_video=4)
    second = VideoAwareBatchSampler(videos, batch_size=16, seed=1111, samples_per_video=4)
    first_batches = first.preview(epochs=2)
    second_batches = second.preview(epochs=2)
    assert first_batches == second_batches
    one_epoch = first.preview(epochs=1)
    flat = [index for batch in one_epoch for index in batch]
    assert sorted(flat) == list(range(len(videos)))
    assert len(flat) == len(set(flat))
    assert all(len(batch) <= 16 for batch in one_epoch)


def test_vrex_gradient_and_value():
    full = torch.tensor([[0.0], [1.0], [2.0], [3.0]], requires_grad=True)
    missing = torch.tensor([[0.5], [1.5], [1.0], [4.0]], requires_grad=True)
    labels = torch.zeros(4, 1)
    result = video_risk_variance(full, missing, labels, ["a", "a", "b", "b"])
    assert result.diagnostics["vrex_active"] is True
    assert result.diagnostics["eligible_video_count"] == 2
    assert float(result.penalty.detach()) > 0
    result.penalty.backward()
    assert full.grad is not None and float(full.grad.abs().sum()) > 0
    assert missing.grad is not None and float(missing.grad.abs().sum()) > 0

    inactive = video_risk_variance(
        torch.zeros(3, 1, requires_grad=True),
        torch.zeros(3, 1, requires_grad=True),
        torch.zeros(3, 1),
        ["a", "b", "c"],
    )
    assert inactive.diagnostics["vrex_active"] is False
    assert float(inactive.penalty.detach()) == 0.0


def test_gates():
    baseline = {
        "J_valid": 0.680,
        "valid_LAV_MAE": 0.680,
        "valid_LA_MAE": 0.681,
        "valid_LV_MAE": 0.682,
        "valid_L_MAE": 0.683,
        "J_test_at_valid_best": 0.710,
        "test_at_valid_best_LAV_MAE": 0.709,
        "test_at_valid_best_LA_MAE": 0.711,
        "test_at_valid_best_LV_MAE": 0.712,
        "test_at_valid_best_L_MAE": 0.713,
    }
    candidate = {
        "J_valid": 0.674,
        "valid_LAV_MAE": 0.674,
        "valid_LA_MAE": 0.675,
        "valid_LV_MAE": 0.676,
        "valid_L_MAE": 0.677,
        "J_test_at_valid_best": 0.704,
        "test_at_valid_best_LAV_MAE": 0.704,
        "test_at_valid_best_LA_MAE": 0.705,
        "test_at_valid_best_LV_MAE": 0.706,
        "test_at_valid_best_L_MAE": 0.707,
    }
    epochs = [{"J_valid": 0.676}, {"J_valid": 0.675}, {"J_valid": 0.679}]
    assert candidate_valid_gate(candidate, baseline, epochs)["passed"] is True
    assert candidate_test_gate(candidate, baseline)["passed"] is True


def main():
    test_video_id_parser()
    test_sampler_exact_coverage_and_determinism()
    test_vrex_gradient_and_value()
    test_gates()
    print("CFCompat Video-VREx smoke test passed")


if __name__ == "__main__":
    main()
