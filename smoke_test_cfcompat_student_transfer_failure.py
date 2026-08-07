"""Synthetic smoke test for CFCompatKD v5.2 Student-transfer diagnostic."""
from __future__ import annotations

import numpy as np
import pandas as pd

import analyze_cfcompat_student_transfer_failure_v5p2 as diag


def make_raw(run: str, n: int = 24, delta: float = 0.0):
    rows = []
    rng = np.random.default_rng(sum(ord(c) for c in run) + 17)
    labels = np.linspace(-2.0, 2.0, n)
    for index in range(n):
        label = float(labels[index])
        for mode_i, mode in enumerate(diag.MODES):
            direction = 1.0 if index % 2 == 0 else -1.0
            baseline = label + (0.35 + 0.05 * mode_i) * direction
            teacher = label + (0.10 if index % 3 else 0.55) * direction
            if index % 5 == 1:
                candidate = baseline + 0.12 * direction + delta + rng.normal(0, 0.01)
            elif index % 7 == 2:
                candidate = baseline + delta + rng.normal(0, 0.01)
            else:
                candidate = baseline - 0.12 * direction + delta + rng.normal(0, 0.01)
            rows.append({
                "Seed": diag.SEED, "Run": run, "Mode": mode,
                "sample_index": index, "sample_id": f"s{index}", "label": label,
                "baseline_prediction": baseline, "candidate_prediction": candidate,
                "teacher_prediction": teacher, "Split": "valid",
            })
    return pd.DataFrame(rows)


def make_gate(candidate_raw: pd.DataFrame):
    rows = []
    for row in candidate_raw.itertuples(index=False):
        advantage = abs(row.baseline_prediction - row.label) - abs(row.teacher_prediction - row.label)
        p = 0.72 if advantage >= diag.BENEFIT_MARGIN else 0.28
        b = row.baseline_prediction
        f = b + 0.03
        t = row.teacher_prediction
        s0 = b + 0.02
        rows.append({
            "sample_index": row.sample_index, "sample_id": row.sample_id, "mode": row.Mode,
            "split": "valid", "label": row.label,
            "baseline_missing_prediction": b, "baseline_full_prediction": f,
            "teacher_full_prediction": t, "initial_student_missing_prediction": s0,
            "teacher_advantage_vs_baseline": advantage,
            "beneficial_label": int(advantage >= diag.BENEFIT_MARGIN),
            "full_train_benefit_probability": p,
            "teacher_minus_baseline_missing": t-b,
            "student0_minus_baseline_missing": s0-b,
            "teacher_minus_student0": t-s0,
            "baseline_full_minus_missing": f-b,
            "teacher_minus_baseline_full": t-f,
            "abs_teacher_minus_baseline_missing": abs(t-b),
            "abs_student0_minus_baseline_missing": abs(s0-b),
            "abs_teacher_minus_student0": abs(t-s0),
            "abs_baseline_full_minus_missing": abs(f-b),
            "abs_teacher_minus_baseline_full": abs(t-f),
            "abs_baseline_missing_prediction": abs(b),
            "abs_teacher_full_prediction": abs(t),
            "abs_initial_student_missing_prediction": abs(s0),
        })
    return pd.DataFrame(rows)


def main():
    candidate_raw = make_raw(diag.V4_RUN)
    refs_raw = pd.concat([
        make_raw("cfcompat_replay", delta=0.03),
        make_raw("student_safe_uniform", delta=-0.02),
    ], ignore_index=True)
    candidate = diag.prepare_events(diag.validate_raw(candidate_raw, [diag.V4_RUN], "candidate"))
    refs = diag.prepare_events(diag.validate_raw(refs_raw, diag.REF_RUNS, "refs"))
    gate = diag.validate_gate(make_gate(candidate_raw))
    joined = diag.add_characteristics(diag.join_gate(candidate, gate), 0.5)

    assert len(joined) == 24 * 3
    assert (joined.teacher_beneficial & ~joined.student_improved_any).any()
    all_events = pd.concat([joined, refs], ignore_index=True, sort=False)
    assert not diag.run_summary(all_events).empty
    assert not diag.quadrant_summary(all_events).empty
    compare = diag.compare_runs(joined, refs)
    assert len(compare) == len(joined)
    features = diag.feature_summary(joined)
    rules = diag.failure_rules(joined, min_support=2)
    assert not features.empty and not rules.empty
    assert not diag.joint_summary(joined).empty
    assert len(diag.top_failures(joined, 10)) == 10
    print("PASS: CFCompatKD v5.2 Student-transfer failure diagnostic smoke test")


if __name__ == "__main__":
    main()
