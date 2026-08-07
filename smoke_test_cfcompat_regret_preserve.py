"""Synthetic unit smoke for Regret-Aware Preserve-or-Distill CFCompatKD v4."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trains.singleTask.cfcompat_regret_preserve_utils import (
    DISTILL_MARGIN,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    PRESERVE_MARGIN,
    negative_transfer_summary,
    regret_preserve_decision,
    regret_projection_summary,
)
from trains.singleTask.cfcompat_safe_projection_utils import derive_valid_events


def assert_close(actual, expected, tolerance=1e-7):
    if abs(float(actual) - float(expected)) > tolerance:
        raise AssertionError("{} != {}".format(actual, expected))


def main():
    # Five deliberately different cases:
    # 0: Teacher clearly improves the frozen baseline -> DISTILL.
    # 1: Teacher is poor and current Student regressed -> PRESERVE.
    # 2: Neither meaningful Teacher benefit nor Student regret -> ABSTAIN.
    # 3: Teacher beats baseline, but current Student already beats Teacher and
    #    Teacher is wrong-direction relative to current Student -> ABSTAIN.
    # 4: Frozen baseline is better but lies across the label -> PRESERVE target
    #    must clip to the label rather than overshoot it.
    student = torch.tensor([[1.20], [0.80], [0.21], [0.10], [1.00]], dtype=torch.float32)
    teacher = torch.tensor([[0.30], [1.00], [0.30], [0.50], [1.20]], dtype=torch.float32)
    baseline = torch.tensor([[1.00], [0.20], [0.20], [1.00], [-0.20]], dtype=torch.float32)
    labels = torch.zeros_like(student)
    compatibility = torch.tensor([0.8, 0.4, 0.6, 0.9, 0.2], dtype=torch.float32)

    decision = regret_preserve_decision(
        student, teacher, baseline, labels, compatibility
    )
    expected_distill = np.asarray([True, False, False, False, False])
    expected_preserve = np.asarray([False, True, False, False, True])
    expected_abstain = np.asarray([False, False, True, True, False])
    if not np.array_equal(decision["distill"].cpu().numpy(), expected_distill):
        raise AssertionError("Synthetic DISTILL decisions changed.")
    if not np.array_equal(decision["preserve"].cpu().numpy(), expected_preserve):
        raise AssertionError("Synthetic PRESERVE decisions changed.")
    if not np.array_equal(decision["abstain"].cpu().numpy(), expected_abstain):
        raise AssertionError("Synthetic ABSTAIN decisions changed.")

    # Case 0 keeps a safe Teacher target.  Case 4 clips the across-label frozen
    # baseline to y=0 rather than preserving an overshoot.
    assert_close(decision["teacher_safe_target"][0], 0.30)
    assert_close(decision["preserve_safe_target"][4], 0.0)
    assert_close(
        decision["distill_gate"][0],
        MILD_CFCOMPAT_BASE + MILD_CFCOMPAT_SCALE * 0.8,
    )
    assert_close(decision["preserve_gate"][1], 1.0)
    assert_close(decision["preserve_gate"][4], 1.0)

    if DISTILL_MARGIN != 0.02 or PRESERVE_MARGIN != 0.02:
        raise AssertionError("Frozen v4 decision margins changed.")

    # Build records accepted by the aggregate accounting function.
    records = []
    projection = decision["teacher_projection"]
    for offset in range(len(student)):
        record = {
            key: (
                bool(value[offset].cpu())
                if value.dtype == torch.bool
                else float(value[offset].cpu())
            )
            for key, value in projection.items()
        }
        record.update(
            {
                "mode": ("LA", "LV", "L", "LA", "LV")[offset],
                "distill": bool(decision["distill"][offset]),
                "preserve": bool(decision["preserve"][offset]),
                "decision_abstain": bool(decision["abstain"][offset]),
                "teacher_beneficial": bool(decision["teacher_beneficial"][offset]),
                "current_regressed": bool(decision["current_regressed"][offset]),
                "baseline_error": float(decision["baseline_error"][offset]),
                "teacher_error": float(decision["teacher_error"][offset]),
                "current_error": float(decision["current_error"][offset]),
                "teacher_advantage_vs_baseline": float(decision["teacher_advantage_vs_baseline"][offset]),
                "current_regret_vs_baseline": float(decision["current_regret_vs_baseline"][offset]),
                "compatibility": float(compatibility[offset]),
                "mild_compatibility": float(decision["mild_compatibility"][offset]),
                "distill_gate": float(decision["distill_gate"][offset]),
                "preserve_gate": float(decision["preserve_gate"][offset]),
                "teacher_safe_target": float(decision["teacher_safe_target"][offset]),
                "preserve_safe_target": float(decision["preserve_safe_target"][offset]),
                "student_prediction": float(student[offset]),
                "label": float(labels[offset]),
            }
        )
        records.append(record)
    summary = regret_projection_summary(records)
    assert_close(summary["distill_fraction"], 0.2)
    assert_close(summary["preserve_fraction"], 0.4)
    assert_close(summary["decision_abstain_fraction"], 0.4)

    # Four samples per modality keep the baseline quartile derivation valid.
    # For each missing mode the candidate regrets are +0.05, -0.10, +0.20, 0;
    # therefore 2/4 missing-modality events exceed the frozen +0.02 threshold.
    rows = []
    candidate_by_sample = (0.25, 0.10, 0.40, 0.20)
    for sample_index, prediction in enumerate(candidate_by_sample):
        for mode in ("LAV", "LA", "LV", "L"):
            rows.append(
                {
                    "Seed": 1113,
                    "Run": "candidate",
                    "Mode": mode,
                    "sample_index": sample_index,
                    "sample_id": "x{}".format(sample_index),
                    "label": 0.0,
                    "baseline_prediction": 0.2,
                    "candidate_prediction": prediction,
                    "teacher_prediction": 0.1,
                    "Split": "valid",
                    "SelectedBy": "validation_J",
                }
            )
    events = derive_valid_events(pd.DataFrame(rows))
    negative = negative_transfer_summary(events)
    missing = negative.loc[negative.Mode.eq("MISSING_ALL")].iloc[0]
    assert_close(missing.negative_transfer_rate, 0.5)

    print("Regret-aware preserve-or-distill v4 utility smoke test passed")


if __name__ == "__main__":
    main()
