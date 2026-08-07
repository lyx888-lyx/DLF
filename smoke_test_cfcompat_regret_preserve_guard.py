"""Synthetic smoke for Regret-Aware Beneficial-Teacher Guard v4.1."""
from __future__ import annotations

import numpy as np
import torch

from trains.singleTask.cfcompat_regret_preserve_guard_utils import (
    STRONG_DISTILL_MARGIN,
    WEAK_DISTILL_SCALE,
    regret_guard_projection_summary,
    regret_preserve_guard_decision,
    tiered_kd_loss,
)


def assert_close(actual, expected, tol=1e-7):
    if abs(float(actual) - float(expected)) > tol:
        raise AssertionError("{} != {}".format(actual, expected))


def main():
    # label=0 for all cases.
    # 0 strong: baseline=1.0, teacher=0.2, student=1.2
    # 1 weak:   baseline=0.2, teacher=0.19, student=0.3 (advantage=0.01)
    # 2 pause:  baseline=1.0, teacher=0.4, student=0.2 (Student already better)
    # 3 preserve: Teacher not better; Student regressed from baseline.
    # 4 abstain: Teacher not better; Student has not materially regressed.
    student = torch.tensor([[1.20], [0.30], [0.20], [0.60], [0.21]], dtype=torch.float32)
    teacher = torch.tensor([[0.20], [0.19], [0.40], [0.40], [0.40]], dtype=torch.float32)
    baseline = torch.tensor([[1.00], [0.20], [1.00], [0.20], [0.20]], dtype=torch.float32)
    labels = torch.zeros_like(student)
    compat = torch.tensor([0.8, 0.6, 0.9, 0.4, 0.5], dtype=torch.float32)

    decision = regret_preserve_guard_decision(student, teacher, baseline, labels, compat)
    expected = {
        "strong_distill": [True, False, False, False, False],
        "weak_distill": [False, True, False, False, False],
        "beneficial_pause": [False, False, True, False, False],
        "preserve": [False, False, False, True, False],
        "abstain": [False, False, False, False, True],
    }
    for key, values in expected.items():
        actual = decision[key].cpu().numpy().astype(bool)
        if not np.array_equal(actual, np.asarray(values, dtype=bool)):
            raise AssertionError("Synthetic v4.1 state changed for {}: {}".format(key, actual))

    if STRONG_DISTILL_MARGIN != 0.02 or WEAK_DISTILL_SCALE != 0.25:
        raise AssertionError("Frozen v4.1 constants changed.")

    # Verify that weak scaling cannot be canceled by denominator normalization.
    s = torch.tensor([1.0, 1.0], dtype=torch.float32)
    t = torch.tensor([0.0, 0.0], dtype=torch.float32)
    ones = torch.ones(2, dtype=torch.float32)
    zeros = torch.zeros(2, dtype=torch.float32)
    strong_loss, _, _, _ = tiered_kd_loss(s, t, ones, zeros)
    weak_loss, _, _, _ = tiered_kd_loss(s, t, zeros, ones)
    assert_close(weak_loss, WEAK_DISTILL_SCALE * strong_loss)

    records = []
    projection = decision["teacher_projection"]
    eligible = decision["strong_gate"] + decision["weak_gate"]
    effective = decision["strong_gate"] + WEAK_DISTILL_SCALE * decision["weak_gate"]
    for i in range(len(student)):
        record = {
            key: (bool(value[i]) if value.dtype == torch.bool else float(value[i]))
            for key, value in projection.items()
        }
        record.update({
            "mode": ("LA", "LV", "L", "LA", "LV")[i],
            "strong_distill": bool(decision["strong_distill"][i]),
            "weak_distill": bool(decision["weak_distill"][i]),
            "beneficial_pause": bool(decision["beneficial_pause"][i]),
            "preserve": bool(decision["preserve"][i]),
            "guard_abstain": bool(decision["abstain"][i]),
            "teacher_better_any": bool(decision["teacher_better_any"][i]),
            "strong_candidate": bool(decision["strong_candidate"][i]),
            "weak_candidate": bool(decision["weak_candidate"][i]),
            "baseline_direction_correct": bool(decision["baseline_direction_correct"][i]),
            "current_regressed": bool(decision["current_regressed"][i]),
            "baseline_error": float(decision["baseline_error"][i]),
            "teacher_error": float(decision["teacher_error"][i]),
            "current_error": float(decision["current_error"][i]),
            "teacher_advantage_vs_baseline": float(decision["teacher_advantage_vs_baseline"][i]),
            "current_regret_vs_baseline": float(decision["current_regret_vs_baseline"][i]),
            "compatibility": float(compat[i]),
            "mild_compatibility": float(decision["mild_compatibility"][i]),
            "strong_gate": float(decision["strong_gate"][i]),
            "weak_gate": float(decision["weak_gate"][i]),
            "eligible_distill_mass": float(eligible[i]),
            "effective_distill_gate": float(effective[i]),
            "preserve_gate": float(decision["preserve_gate"][i]),
        })
        records.append(record)
    summary = regret_guard_projection_summary(records)
    for key in (
        "strong_distill_fraction", "weak_distill_fraction", "beneficial_pause_fraction",
        "preserve_fraction", "guard_abstain_fraction",
    ):
        assert_close(summary[key], 0.2)

    print("Regret-aware beneficial-Teacher guard v4.1 utility smoke test passed")


if __name__ == "__main__":
    main()
