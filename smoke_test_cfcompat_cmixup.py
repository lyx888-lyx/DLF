"""Synthetic smoke tests for CFCompatKD mode-consistent C-Mixup."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from trains.singleTask.cmixup_regression_utils import (
    FusionInputCapture,
    ModeConsistentCMixupSampler,
    compute_mode_consistent_cmixup_loss,
    predict_from_dlf_fusion,
    promotion_gate,
)


class ToyBackbone(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.output_dropout = 0.0
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.out_layer = nn.Linear(dim, 1)

    def forward(self, fusion):
        hidden = self.proj2(F.relu(self.proj1(fusion), inplace=True))
        return self.out_layer(hidden + fusion)


class ToyStudent(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.backbone = ToyBackbone(dim)


def main():
    torch.manual_seed(7)
    labels = torch.tensor([[-2.0], [-1.8], [0.0], [0.1], [1.0], [1.2]])
    modes = ["LA", "LA", "LV", "LV", "L", "L"]
    sampler = ModeConsistentCMixupSampler(
        seed=1111, alpha=2.0, bandwidth=0.5
    )
    batch = sampler.sample(labels, modes, labels.device, labels.dtype)
    assert bool(batch.active_mask.all())
    for index, partner in enumerate(batch.partner_indices.tolist()):
        assert index != partner
        assert modes[index] == modes[partner]

    probabilities = sampler._probabilities(
        np.asarray([0.0, 0.1, 2.0]), 0.5
    )
    assert probabilities[0, 1] > probabilities[0, 2]

    backbone = ToyBackbone()
    fusion = torch.randn(4, 6)
    backbone.eval()
    direct = backbone(fusion)
    replay = predict_from_dlf_fusion(backbone, fusion)
    assert torch.allclose(direct, replay, atol=1e-7, rtol=0.0)

    with FusionInputCapture(backbone.proj1) as capture:
        _ = backbone(fusion)
        captured = capture.pop()
        capture.assert_empty()
    assert torch.equal(captured, fusion)

    student = ToyStudent()
    student.train()
    full = torch.randn(6, 6, requires_grad=True)
    missing = torch.randn(6, 6, requires_grad=True)
    loss, diagnostics, _ = compute_mode_consistent_cmixup_loss(
        student,
        full,
        missing,
        labels,
        modes,
        ModeConsistentCMixupSampler(2222),
        nn.L1Loss(),
    )
    loss.backward()
    assert float(student.backbone.proj1.weight.grad.abs().sum()) > 0
    assert float(student.backbone.out_layer.weight.grad.abs().sum()) > 0
    assert diagnostics["mix_active_fraction"] == 1.0

    baseline = {
        "J_valid": 0.700,
        "valid_LAV_MAE": 0.690,
        "valid_LA_MAE": 0.710,
        "valid_LV_MAE": 0.705,
        "valid_L_MAE": 0.715,
    }
    candidate = {
        "J_valid": 0.694,
        "valid_LAV_MAE": 0.684,
        "valid_LA_MAE": 0.710,
        "valid_LV_MAE": 0.704,
        "valid_L_MAE": 0.714,
    }
    rows = [{"J_valid": 0.696}, {"J_valid": 0.695}]
    assert promotion_gate(candidate, baseline, rows)["passed"]

    print("CFCompat C-Mixup smoke test passed")


if __name__ == "__main__":
    main()
