"""CPU smoke test for V9.15 absolute-target distillation primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.AbsoluteTargetExpertStudentV915 import (  # noqa: E402
    STUDENT_VERSION,
    absolute_target_distillation_loss,
)
from trains.singleTask.model.SemanticCostCoachV99 import (  # noqa: E402
    ACTION_NAMES,
    SPECIALIST_NAMES,
)
from trains.singleTask.strict_oof_absolute_target_distillation_v915 import (  # noqa: E402
    TeacherConfigV914,
    apply_beta,
    build_soft_teacher_v914,
    calibrate_beta,
)


def synthetic_strict_pool(n: int = 24):
    torch.manual_seed(915)
    labels = torch.linspace(-2.0, 2.0, n).view(-1, 1)
    anchor = labels + 0.4 * torch.sin(torch.linspace(0.0, 5.0, n)).view(-1, 1)
    offsets = torch.tensor([-0.25, -0.05, 0.08, 0.22]).view(1, 4, 1)
    experts = labels.unsqueeze(1) + offsets
    return {
        "version": "synthetic_strict_pool",
        "sample_ids": [f"sample_{index}" for index in range(n)],
        "group_ids": [f"group_{index // 2}" for index in range(n)],
        "labels": labels,
        "anchor": anchor,
        "fold_index": torch.arange(n) % 5,
        "expert_predictions": experts,
        "action_names": ACTION_NAMES,
        "provenance": {
            "is_fully_nested_teacher_stack": True,
            "holdout_label_isolation": True,
            "historical_full_train_teacher_reuse": False,
        },
    }


def main():
    assert STUDENT_VERSION == "absolute_target_expert_student_v915_v1"
    teacher = build_soft_teacher_v914(
        synthetic_strict_pool(),
        TeacherConfigV914(),
    )
    assert teacher["teacher_prediction"].shape == (24, 1)
    assert teacher["expert_predictions"].shape == (24, 4)
    assert teacher["expert_relevance"].shape == (24, 4)
    assert tuple(teacher["specialist_names"]) == tuple(SPECIALIST_NAMES)

    n = 12
    base = torch.linspace(-1.0, 1.0, n).view(-1, 1)
    correction = torch.zeros(n, 1, requires_grad=True)
    auxiliary_corrections = torch.zeros(n, 4, requires_grad=True)
    output = {
        "prediction": base + correction,
        "correction": correction,
        "auxiliary_predictions": base + auxiliary_corrections,
        "auxiliary_corrections": auxiliary_corrections,
    }
    labels = base + 0.05
    teacher_prediction = base + 0.04
    expert_predictions = base + torch.tensor([0.02, 0.04, 0.06, 0.08])
    relevance = torch.full((n, 4), 0.25)
    losses = absolute_target_distillation_loss(
        output,
        labels,
        teacher_prediction,
        expert_predictions,
        relevance,
        distill_weight=0.5,
        auxiliary_weight=0.05,
        correction_penalty=0.10,
    )
    losses["total"].backward()
    assert correction.grad is not None
    assert auxiliary_corrections.grad is not None
    assert torch.isfinite(losses["total"])

    outputs = {
        "anchor": torch.tensor([[0.0], [0.0]]),
        "correction": torch.tensor([[0.2], [0.2]]),
        "labels": torch.tensor([[0.1], [0.1]]),
    }
    selected = calibrate_beta(outputs, (0.0, 0.5, 1.0))
    assert selected["beta"] == 0.5
    assert selected["mae"] < 1e-7
    applied = apply_beta(outputs, 0.5)
    assert torch.allclose(applied["prediction"], outputs["labels"])

    print("V9.15 STRICT OOF ABSOLUTE-TARGET DISTILLATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
