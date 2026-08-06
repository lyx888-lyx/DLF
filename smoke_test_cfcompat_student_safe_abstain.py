"""Synthetic unit smoke tests for current-Student safe abstention."""
from __future__ import annotations

import torch

from trains.singleTask.cfcompat_student_safe_abstain_utils import (
    student_projection_summary,
    student_safe_project_teacher,
)


def main():
    student = torch.tensor([[-1.0], [-1.0], [-1.0], [0.5], [0.8]])
    teacher = torch.tensor([[-1.7], [-0.4], [-2.8], [-1.0], [0.5]])
    labels = torch.tensor([[-2.0], [-2.0], [-2.0], [0.5], [1.0]])

    projected, diagnostics = student_safe_project_teacher(
        student, teacher, labels
    )
    expected = torch.tensor([[-1.7], [-1.0], [-2.0], [0.5], [0.8]])
    torch.testing.assert_close(projected, expected)

    assert diagnostics["active"].tolist() == [True, False, True, False, False]
    assert diagnostics["abstained"].tolist() == [False, True, False, True, True]
    assert diagnostics["wrong_direction"].tolist() == [False, True, False, False, True]
    assert diagnostics["overshoot"].tolist() == [False, False, True, False, False]
    assert diagnostics["zero_width"].tolist() == [False, False, False, True, False]

    records = []
    for offset in range(student.size(0)):
        records.append(
            {
                key: (
                    bool(value[offset])
                    if value.dtype == torch.bool
                    else float(value[offset])
                )
                for key, value in diagnostics.items()
            }
        )
    summary = student_projection_summary(records)
    assert summary["sample_count"] == 5
    assert abs(summary["active_fraction"] - 0.4) <= 1e-12
    assert abs(summary["abstain_fraction"] - 0.6) <= 1e-12
    assert abs(
        summary["projected_to_baseline_fraction"]
        - summary["abstain_fraction"]
    ) <= 1e-12

    print("Current-Student safe-abstention utility smoke test passed")


if __name__ == "__main__":
    main()
