"""Synthetic smoke test for the one permitted full-view-only C-Mixup repair."""

import torch
import torch.nn as nn

from train_cfcompat_cmixup_full_only import full_only_cmixup_loss
from trains.singleTask.cmixup_regression_utils import ModeConsistentCMixupSampler


class TinyBackbone(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.proj1 = nn.Linear(dim, dim)
        self.proj2 = nn.Linear(dim, dim)
        self.out_layer = nn.Linear(dim, 1)
        self.output_dropout = 0.0


class TinyStudent(nn.Module):
    def __init__(self, dim=6):
        super().__init__()
        self.backbone = TinyBackbone(dim)


def main():
    torch.manual_seed(7)
    student = TinyStudent()
    full = torch.randn(8, 6, requires_grad=True)
    missing = torch.randn(8, 6, requires_grad=True)
    labels = torch.tensor(
        [-2.0, -1.2, -0.4, 0.0, 0.5, 1.0, 1.8, 2.5]
    ).view(-1, 1)
    modes = ["LA", "LV", "L", "LA", "LV", "L", "LA", "LV"]
    sampler = ModeConsistentCMixupSampler(seed=1111, alpha=2.0, bandwidth=0.5)

    loss, diagnostics, batch = full_only_cmixup_loss(
        student=student,
        full_fusion=full,
        missing_fusion=missing,
        labels=labels,
        modes=modes,
        sampler=sampler,
        criterion=nn.L1Loss(),
    )
    if not torch.isfinite(loss) or float(loss.detach()) <= 0:
        raise RuntimeError("Full-view C-Mixup loss is invalid.")
    loss.backward()

    if full.grad is None or float(full.grad.abs().sum()) <= 0:
        raise RuntimeError("Full-view C-Mixup did not reach LAV fusion features.")
    if missing.grad is not None and float(missing.grad.abs().sum()) != 0:
        raise RuntimeError("Missing fusion features received mixed-label gradients.")
    if diagnostics["mix_missing_loss"] != 0.0:
        raise RuntimeError("Missing mixed-label loss must remain exactly zero.")
    if diagnostics["mix_active_fraction"] < 0.99:
        raise RuntimeError("Whole-batch C-Mixup should activate every sample.")
    if batch.mode_counts != {"LA": 0, "LV": 0, "L": 8}:
        raise RuntimeError("The complete-view partner pool was unexpectedly split.")
    if torch.any(batch.partner_indices == torch.arange(8)):
        raise RuntimeError("C-Mixup sampled a self partner.")

    print("CFCompat full-view C-Mixup v2 smoke test passed")
    print("mixed missing-view loss: 0")
    print("whole-batch label-KDE pool: verified")
    print("missing fusion mixed-label gradient: absent")


if __name__ == "__main__":
    main()
