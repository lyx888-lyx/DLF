"""CPU smoke test for V9.14 strict OOF distillation primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.SemanticCostCoachV99 import (  # noqa: E402
    ACTION_NAMES,
    SPECIALIST_NAMES,
)
from trains.singleTask.model.StrictOOFExpertStudentV914 import (  # noqa: E402
    STUDENT_VERSION,
    strict_oof_distillation_loss,
)
from trains.singleTask.strict_oof_expert_distillation_v914 import (  # noqa: E402
    TEACHER_VERSION,
    TeacherConfigV914,
    align_teacher_to_dataset,
    build_soft_teacher_v914,
)


class FakeDataset:
    def __init__(self, ids):
        self.ids = list(ids)


def synthetic_pool(n: int = 24):
    torch.manual_seed(914)
    labels = torch.linspace(-2.0, 2.0, n).view(-1, 1)
    anchor = labels + 0.30 * torch.randn(n, 1)
    experts = torch.stack(
        [
            anchor + 0.15 * torch.randn(n, 1),
            anchor + 0.12 * torch.randn(n, 1),
            anchor + 0.10 * torch.randn(n, 1),
            anchor + 0.18 * torch.randn(n, 1),
        ],
        dim=1,
    )
    return {
        "version": "v98_strict_nested_v93_architecture_frontier_pool",
        "sample_ids": [f"sample-{index}" for index in range(n)],
        "group_ids": [f"group-{index // 2}" for index in range(n)],
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
    assert STUDENT_VERSION == "strict_oof_expert_student_v914_v1"
    assert TEACHER_VERSION == "strict_oof_soft_teacher_v914_v1"
    pool = synthetic_pool()
    teacher = build_soft_teacher_v914(
        pool,
        TeacherConfigV914(
            temperature=0.15,
            gain_margin=0.02,
            gain_scale=0.10,
            max_alpha=0.50,
        ),
    )
    n = len(pool["sample_ids"])
    assert teacher["teacher_correction"].shape == (n, 1)
    assert teacher["expert_corrections"].shape == (
        n,
        len(SPECIALIST_NAMES),
    )
    assert teacher["expert_relevance"].shape == (
        n,
        len(SPECIALIST_NAMES),
    )
    assert torch.allclose(
        teacher["expert_relevance"].sum(dim=1),
        torch.ones(n),
        atol=1e-6,
    )
    anchor_error = torch.abs(teacher["anchor"] - teacher["labels"])
    teacher_error = torch.abs(
        teacher["teacher_prediction"] - teacher["labels"]
    )
    assert bool((teacher_error <= anchor_error + 1e-6).all())
    assert float(teacher["alpha"].max()) <= 0.500001

    reverse_ids = list(reversed(pool["sample_ids"]))
    aligned = align_teacher_to_dataset(teacher, FakeDataset(reverse_ids))
    assert aligned["sample_ids"] == reverse_ids
    assert aligned["sample_id_alignment_checked"] is True

    correction = torch.zeros(n, 1, requires_grad=True)
    auxiliary = torch.zeros(
        n, len(SPECIALIST_NAMES), requires_grad=True
    )
    output = {
        "prediction": teacher["anchor"] + correction,
        "correction": correction,
        "auxiliary_corrections": auxiliary,
    }
    losses = strict_oof_distillation_loss(
        output,
        teacher["labels"],
        teacher["teacher_correction"],
        teacher["expert_corrections"],
        teacher["expert_relevance"],
        distill_weight=0.25,
        auxiliary_weight=0.10,
    )
    losses["total"].backward()
    assert correction.grad is not None
    assert auxiliary.grad is not None
    assert torch.isfinite(losses["total"])
    print("V9.14 STRICT OOF EXPERT DISTILLATION SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
