"""Fast synthetic checks for student-safe dynamic-utility CFCompatKD v3."""
from __future__ import annotations

import math

import torch

from trains.singleTask.cfcompat_student_safe_utility_utils import (
    RESIDUAL_CFCOMPAT_ALPHA,
    dynamic_difficulty,
    dynamic_utility,
    residual_compatibility,
    student_safe_project_teacher,
    utility_projection_summary,
)


def main():
    student = torch.tensor([[0.0], [0.5], [0.0]], dtype=torch.float32)
    teacher = torch.tensor([[-1.0], [1.5], [0.25]], dtype=torch.float32)
    labels = torch.tensor([[1.0], [1.0], [1.0]], dtype=torch.float32)
    compatibility = torch.tensor([0.2, 0.5, 0.8], dtype=torch.float32)
    tau = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)

    safe, projection = student_safe_project_teacher(student, teacher, labels)
    expected_safe = torch.tensor([[0.0], [1.0], [0.25]], dtype=torch.float32)
    if not torch.allclose(safe, expected_safe, atol=0.0, rtol=0.0):
        raise AssertionError("Student-safe projection synthetic targets changed.")
    if projection["active"].tolist() != [False, True, True]:
        raise AssertionError("Student-safe active mask changed.")

    utility = dynamic_utility(student, safe, labels)
    expected_utility = torch.tensor([0.0, 1.0, 0.25], dtype=torch.float32)
    if not torch.allclose(utility, expected_utility, atol=1e-6, rtol=0.0):
        raise AssertionError("Dynamic utility formula changed.")

    difficulty = dynamic_difficulty(student, labels, tau)
    expected_difficulty = torch.tensor(
        [2.0 / 3.0, 0.5, 2.0 / 3.0], dtype=torch.float32
    )
    if not torch.allclose(difficulty, expected_difficulty, atol=1e-6, rtol=0.0):
        raise AssertionError("Dynamic difficulty formula changed.")

    residual = residual_compatibility(compatibility)
    expected_residual = torch.tensor([0.6, 0.75, 0.9], dtype=torch.float32)
    if not torch.allclose(residual, expected_residual, atol=1e-6, rtol=0.0):
        raise AssertionError("Residual CFCompat transform changed.")
    if not math.isclose(RESIDUAL_CFCOMPAT_ALPHA, 0.5, abs_tol=0.0):
        raise AssertionError("Frozen residual alpha changed.")

    active = projection["active"].to(dtype=torch.float32)
    utility_gate = active * utility * difficulty
    residual_gate = utility_gate * residual
    if torch.any(residual_gate > utility_gate + 1e-7):
        raise AssertionError("Residual prior may not amplify dynamic utility gate.")
    if not math.isclose(float(residual_gate[0]), 0.0, abs_tol=0.0):
        raise AssertionError("Unsafe synthetic sample did not abstain.")

    records = []
    modes = ("LA", "LV", "L")
    for index, mode in enumerate(modes):
        record = {
            key: (
                bool(value[index].item())
                if value.dtype == torch.bool
                else float(value[index].item())
            )
            for key, value in projection.items()
        }
        record.update(
            {
                "mode": mode,
                "utility": float(utility[index]),
                "difficulty": float(difficulty[index]),
                "compatibility": float(compatibility[index]),
                "residual_compatibility": float(residual[index]),
                "final_gate": float(residual_gate[index]),
                "difficulty_tau": float(tau[index]),
            }
        )
        records.append(record)

    summary = utility_projection_summary(records)
    if not math.isclose(
        summary["active_fraction"] + summary["abstain_fraction"],
        1.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Synthetic active/abstain accounting failed.")
    if not 0.0 < summary["mean_final_gate"] < 1.0:
        raise AssertionError("Synthetic final gate summary is invalid.")
    if not 0.0 < summary["gate_effective_sample_size"] <= len(records):
        raise AssertionError("Synthetic gate ESS is invalid.")

    print("Student-safe dynamic-utility v3 utility smoke test passed")


if __name__ == "__main__":
    main()
