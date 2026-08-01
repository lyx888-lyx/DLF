"""CPU smoke test for V9.16 small-scale absolute-target primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.SmallScaleExpertStudentV916 import (  # noqa: E402
    STUDENT_VERSION,
    project_absolute_targets,
    small_scale_absolute_distillation_loss,
)


def main():
    assert STUDENT_VERSION == "small_scale_expert_student_v916_v1"
    torch.manual_seed(916)
    n = 12
    residual_max = 0.03
    auxiliary_max = 0.15

    base = torch.linspace(-1.0, 1.0, n).view(-1, 1)
    labels = base + 0.08 * torch.randn(n, 1)
    teacher = base + 0.40 * torch.randn(n, 1)
    experts = base + 0.50 * torch.randn(n, 4)
    relevance = torch.softmax(torch.randn(n, 4), dim=1)

    projected = project_absolute_targets(base, teacher, residual_max)
    delta = projected["prediction"] - base
    assert bool((delta.abs() <= residual_max + 1e-7).all())
    assert 0.0 <= float(projected["saturation_rate"]) <= 1.0

    correction_parameter = torch.nn.Parameter(torch.zeros(n, 1))
    auxiliary_parameter = torch.nn.Parameter(torch.zeros(n, 4))
    correction = residual_max * torch.tanh(correction_parameter)
    auxiliary_correction = auxiliary_max * torch.tanh(auxiliary_parameter)
    output = {
        "base_prediction": base,
        "correction": correction,
        "prediction": base + correction,
        "auxiliary_corrections": auxiliary_correction,
        "auxiliary_predictions": base + auxiliary_correction,
    }
    losses = small_scale_absolute_distillation_loss(
        output,
        labels,
        teacher,
        experts,
        relevance,
        distill_weight=1.0,
        auxiliary_weight=0.02,
        residual_max=residual_max,
        auxiliary_residual_max=auxiliary_max,
        correction_penalty=0.01,
    )
    losses["total"].backward()
    assert correction_parameter.grad is not None
    assert auxiliary_parameter.grad is not None
    assert torch.isfinite(losses["total"])
    assert float(losses["teacher_target_saturation_rate"]) > 0.0

    # Label-only must not back-propagate through the specialist heads.
    correction_parameter.grad = None
    auxiliary_parameter.grad = None
    correction = residual_max * torch.tanh(correction_parameter)
    auxiliary_correction = auxiliary_max * torch.tanh(auxiliary_parameter)
    output = {
        "base_prediction": base,
        "correction": correction,
        "prediction": base + correction,
        "auxiliary_corrections": auxiliary_correction,
        "auxiliary_predictions": base + auxiliary_correction,
    }
    label_only = small_scale_absolute_distillation_loss(
        output,
        labels,
        teacher,
        experts,
        relevance,
        distill_weight=0.0,
        auxiliary_weight=0.02,
        residual_max=residual_max,
        auxiliary_residual_max=auxiliary_max,
        correction_penalty=0.01,
    )
    label_only["total"].backward()
    assert correction_parameter.grad is not None
    assert auxiliary_parameter.grad is None or torch.allclose(
        auxiliary_parameter.grad,
        torch.zeros_like(auxiliary_parameter.grad),
    )

    print("V9.16 STRICT OOF SMALL-SCALE DISTILLATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
