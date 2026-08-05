"""Synthetic smoke tests for Safe-CFCompatKD projection and gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from trains.singleTask.cfcompat_safe_projection_utils import (
    FORMAL_SEEDS,
    RUNS,
    aggregate_candidate_gate,
    derive_valid_events,
    group_summary,
    overall_from_events,
    safe_project_teacher,
)


def test_projection_cases():
    baseline = torch.tensor([[-1.0], [-1.0], [-1.0], [0.5]])
    teacher = torch.tensor([[-1.7], [-0.4], [-2.8], [-1.0]])
    labels = torch.tensor([[-2.0], [-2.0], [-2.0], [0.5]])
    projected, diagnostics = safe_project_teacher(baseline, teacher, labels)
    expected = torch.tensor([[-1.7], [-1.0], [-2.0], [0.5]])
    torch.testing.assert_close(projected, expected)
    assert diagnostics["unchanged"].tolist() == [True, False, False, False]
    assert diagnostics["wrong_direction"].tolist() == [False, True, False, False]
    assert diagnostics["overshoot"].tolist() == [False, False, True, False]
    assert diagnostics["zero_width"].tolist() == [False, False, False, True]


def synthetic_events():
    rows = []
    for seed in FORMAL_SEEDS:
        for run_index, run in enumerate(RUNS):
            for mode_index, mode in enumerate(("LAV", "LA", "LV", "L")):
                for sample_index in range(8):
                    label = float(-2.0 + 0.5 * sample_index)
                    baseline = label + (-1.0 if sample_index >= 6 else 0.05 * (sample_index + 1))
                    teacher = label + (0.1 if sample_index >= 4 else 0.4)
                    candidate = baseline
                    if run == "cfcompat_replay":
                        candidate = baseline + 0.20 * np.sign(teacher - baseline)
                    elif run == "safe_uniform":
                        candidate = baseline + 0.24 * np.sign(label - baseline)
                    else:
                        candidate = baseline + 0.28 * np.sign(label - baseline)
                    rows.append(
                        {
                            "Seed": seed,
                            "Run": run,
                            "Mode": mode,
                            "sample_index": sample_index,
                            "sample_id": f"s{sample_index}",
                            "label": label,
                            "baseline_prediction": baseline,
                            "candidate_prediction": candidate,
                            "teacher_prediction": teacher,
                        }
                    )
    events = derive_valid_events(pd.DataFrame(rows))
    assert len(events) == len(FORMAL_SEEDS) * len(RUNS) * 4 * 8
    overall = overall_from_events(events)
    assert set(overall.Mode) == {"LAV", "LA", "LV", "L", "J"}
    groups = group_summary(events)
    assert {"Q1_easy", "Q4_hard"}.issubset(set(groups.GroupValue))


def synthetic_gate():
    grid = []
    epochs = []
    groups = []
    for seed in FORMAL_SEEDS:
        for run in RUNS:
            j = {
                "cfcompat_replay": 0.700,
                "safe_uniform": 0.694,
                "safe_cfcompat": 0.690,
            }[run]
            row = {"Seed": seed, "Run": run, "J_valid": j}
            for mode in ("LAV", "LA", "LV", "L"):
                row[f"valid_{mode}_MAE"] = j
            grid.append(row)
            for epoch in range(1, 4):
                epochs.append(
                    {
                        "Seed": seed,
                        "Run": run,
                        "Epoch": epoch,
                        "J_valid": j + (0.001 if epoch == 1 else 0.0),
                    }
                )
            values = {
                "cfcompat_replay": (-0.060, 0.080, 0.300, 0.070),
                "safe_uniform": (-0.025, 0.066, 0.240, 0.069),
                "safe_cfcompat": (-0.020, 0.070, 0.200, 0.069),
            }[run]
            q1, q4, harmful, good = values
            groups.extend(
                [
                    {
                        "Seed": seed,
                        "Run": run,
                        "GroupType": "baseline_error_quartile",
                        "GroupValue": "Q1_easy",
                        "gain_vs_DLF": q1,
                        "harmful_imitation_rate": harmful,
                    },
                    {
                        "Seed": seed,
                        "Run": run,
                        "GroupType": "baseline_error_quartile",
                        "GroupValue": "Q4_hard",
                        "gain_vs_DLF": q4,
                        "harmful_imitation_rate": harmful,
                    },
                    {
                        "Seed": seed,
                        "Run": run,
                        "GroupType": "all",
                        "GroupValue": "ALL",
                        "gain_vs_DLF": 0.0,
                        "harmful_imitation_rate": harmful,
                    },
                    {
                        "Seed": seed,
                        "Run": run,
                        "GroupType": "teacher_condition",
                        "GroupValue": "better_and_correct",
                        "gain_vs_DLF": good,
                        "harmful_imitation_rate": harmful,
                    },
                ]
            )
    group_frame = pd.DataFrame(groups)
    gate = aggregate_candidate_gate("safe_cfcompat", grid, epochs, group_frame)
    assert gate["passed"]
    assert np.isclose(gate["mean_gain_valid_J_vs_CFCompatKD"], 0.010)


def main():
    test_projection_cases()
    synthetic_events()
    synthetic_gate()
    print("Safe-CFCompatKD projection utility smoke test passed")


if __name__ == "__main__":
    main()
