"""CPU smoke tests for V9.32 task-decomposed multitask utilities."""

from __future__ import annotations

import numpy as np
import torch

from trains.singleTask.task_decomposed_multitask_v932 import (
    ORDINAL_THRESHOLDS,
    TaskDecomposedConfigV932,
    auxiliary_losses,
    ordinal_monotonic_penalty,
    ordinal_targets,
    paired_prediction_metrics,
    success_gate,
    variant_loss_weights,
)


def main():
    config = TaskDecomposedConfigV932()
    config.validate()

    assert variant_loss_weights("regression_only", config) == {
        "ordinal": 0.0,
        "intensity": 0.0,
    }
    assert variant_loss_weights("ordinal_only", config)["ordinal"] > 0
    assert variant_loss_weights("intensity_only", config)["intensity"] > 0
    both = variant_loss_weights("ordinal_intensity", config)
    assert both["ordinal"] > 0 and both["intensity"] > 0

    labels = torch.tensor([[-3.0], [-1.0], [0.0], [1.5], [3.0]])
    targets = ordinal_targets(labels)
    expected = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=torch.float32,
    )
    assert tuple(ORDINAL_THRESHOLDS) == (-2.0, -1.0, 0.0, 1.0, 2.0)
    assert torch.equal(targets, expected)

    monotone_logits = torch.tensor(
        [[4.0, 3.0, 2.0, 1.0, 0.0]]
    )
    reversed_logits = torch.tensor(
        [[0.0, 1.0, 2.0, 3.0, 4.0]]
    )
    assert ordinal_monotonic_penalty(monotone_logits).item() == 0.0
    assert ordinal_monotonic_penalty(reversed_logits).item() > 0.9

    intensity_prediction = labels.abs().clone()
    perfect_logits = torch.where(
        targets > 0.5,
        torch.full_like(targets, 12.0),
        torch.full_like(targets, -12.0),
    )
    losses = auxiliary_losses(
        intensity_prediction, perfect_logits, labels
    )
    assert losses["intensity"].item() < 1e-8
    assert losses["ordinal"].item() < 1e-4
    assert losses["ordinal_monotonic"].item() == 0.0

    baseline = np.array([0.0, 0.0, 0.0, 0.0])
    labels_np = np.array([1.0, -1.0, 0.5, -0.5])
    improved = np.array([0.8, -0.8, 0.4, -0.4])
    metrics = paired_prediction_metrics(improved, labels_np, baseline)
    assert metrics["gain_vs_baseline"] > 0
    assert metrics["win_rate_vs_baseline"] == 1.0

    passing = success_gate(
        [0.01, 0.02, 0.01, 0.03, 0.0],
        0.014,
        config,
    )
    failing = success_gate(
        [0.01, -0.01, 0.01, -0.02, 0.0],
        0.001,
        config,
    )
    assert passing["passed"] is True
    assert failing["passed"] is False

    print("V9.32 SMOKE TEST PASSED")
    print("deployment prediction: regression head only")
    print("auxiliary targets available at train time: True")
    print("test-time label partition required: False")


if __name__ == "__main__":
    main()
