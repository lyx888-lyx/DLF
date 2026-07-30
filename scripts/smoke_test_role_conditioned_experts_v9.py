"""CPU-only synthetic smoke test for the V9 mathematical core."""

from __future__ import annotations

import sys
from pathlib import Path

# When a file under ``scripts/`` is executed directly, Python puts that
# directory—not the repository root—at sys.path[0].  Bootstrap the root before
# importing the local ``trains`` package so the test works from any cwd.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from trains.singleTask.role_conditioned_experts_v9 import (
    LossWeights,
    apply_teacher_weights,
    fit_role_teacher_weights,
    region_index,
    role_conditioned_loss,
)


def main():
    torch.manual_seed(7)
    sample_count = 90
    teacher_count = 5
    labels = torch.linspace(-3.0, 3.0, sample_count).view(-1, 1)
    biases = torch.tensor([-0.10, -0.04, 0.00, 0.05, 0.11]).view(1, -1, 1)
    noise = 0.16 * torch.randn(sample_count, teacher_count, 1)
    predictions = labels.unsqueeze(1) + biases + noise
    sample_ids = ["synthetic-%03d" % index for index in range(sample_count)]

    fitted = fit_role_teacher_weights(
        predictions,
        labels,
        sample_ids,
        steps=20,
        min_region_samples=5,
    )
    assert fitted["global_weights"].shape == (teacher_count,)
    assert fitted["role_weights"].shape == (5, teacher_count)
    assert torch.allclose(fitted["global_weights"].sum(), torch.tensor(1.0), atol=1e-5)
    assert torch.allclose(
        fitted["role_weights"].sum(dim=1), torch.ones(5), atol=1e-5
    )

    batch = 20
    batch_labels = labels[:batch].clone()
    anchor = predictions[:batch, 0].clone()
    global_teacher = apply_teacher_weights(
        predictions[:batch], fitted["global_weights"]
    )
    for role in range(5):
        role_teacher = apply_teacher_weights(
            predictions[:batch], fitted["role_weights"][role]
        )
        prediction = anchor.clone().requires_grad_(True)
        correction = torch.zeros_like(prediction, requires_grad=True)
        logits = torch.randn(batch, 5, requires_grad=True)
        probs = torch.softmax(logits, dim=1)
        centers = prediction.new_tensor((-2.25, -1.0, 0.0, 1.0, 2.25))
        expected = (probs * centers.view(1, -1)).sum(dim=1, keepdim=True)
        risk = torch.full_like(prediction, 0.5, requires_grad=True)
        fake_backbone = {
            key: prediction
            for key in (
                "logits_c",
                "logits_l_hetero",
                "logits_a_hetero",
                "logits_v_hetero",
            )
        }
        outputs = {
            "prediction": prediction,
            "base_prediction": anchor,
            "correction": correction,
            "region_logits": logits,
            "region_probs": probs,
            "region_expected": expected,
            "predicted_abs_error": risk,
            "backbone": fake_backbone,
        }
        losses = role_conditioned_loss(
            outputs,
            batch_labels,
            anchor,
            global_teacher,
            role_teacher,
            role=role,
            membership_floor=0.25,
            membership_sigma_scale=1.0,
            gain_margin=0.08,
            gain_fraction=0.20,
            weights=LossWeights(),
        )
        assert torch.isfinite(losses["total"])
        losses["total"].backward()
        assert prediction.grad is not None
        assert logits.grad is not None
        assert risk.grad is not None

    region_counts = torch.bincount(region_index(labels), minlength=5)
    assert bool((region_counts > 0).all())
    print("V9 SYNTHETIC SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
