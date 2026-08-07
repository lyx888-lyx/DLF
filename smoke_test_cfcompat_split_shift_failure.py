"""Synthetic smoke test for the v5.1 split-shift failure diagnostic."""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyze_cfcompat_split_shift_failure_v5p1 import run_diagnostic
from trains.singleTask.cfcompat_crossfit_transfer_risk_utils import add_transfer_risk_features
from trains.singleTask.missing_utils import MISSING_MODES


def _frame(split: str, n_samples: int, shifted: bool) -> pd.DataFrame:
    rows = []
    for index in range(n_samples):
        label = -1.5 + 3.0 * index / max(n_samples - 1, 1)
        for mode_index, mode in enumerate(MISSING_MODES):
            base = 0.35 * label + 0.05 * mode_index
            gap = (0.65 if not shifted else -0.65) * label + 0.03 * mode_index
            teacher = base + gap
            if shifted:
                base += 0.6
            rows.append(
                {
                    "sample_index": index,
                    "sample_id": f"{split}-{index}",
                    "mode": mode,
                    "split": split,
                    "label": label,
                    "baseline_missing_prediction": base,
                    "baseline_full_prediction": base + 0.1,
                    "teacher_full_prediction": teacher,
                    "initial_student_missing_prediction": base + 0.02,
                }
            )
    frame = add_transfer_risk_features(pd.DataFrame(rows))
    beneficial = frame["beneficial_label"].to_numpy(dtype=float)
    if shifted:
        probability = np.where(beneficial > 0.5, 0.25, 0.75)
        frame["full_train_benefit_probability"] = probability
    else:
        probability = np.where(beneficial > 0.5, 0.80, 0.20)
        frame["oof_benefit_probability"] = probability
    return frame


def main():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        train_path = root / "train.csv"
        valid_path = root / "valid.csv"
        output = root / "out"
        _frame("train", 80, shifted=False).to_csv(train_path, index=False)
        _frame("valid", 40, shifted=True).to_csv(valid_path, index=False)
        summary = run_diagnostic(
            train_path,
            valid_path,
            output,
            min_rule_support=5,
            top_k_failures_count=20,
        )
        required = {
            "sample_failure_table.csv",
            "feature_shift_summary.csv",
            "benefit_relationship_shift.csv",
            "feature_benefit_relationship_bins.csv",
            "mode_failure_summary.csv",
            "failure_interaction_rules.csv",
            "top_gate_failure_samples.csv",
            "split_shift_failure_summary.json",
        }
        produced = {path.name for path in output.iterdir()}
        assert required.issubset(produced), (required - produced)
        assert summary["test_accessed"] is False
        assert summary["student_training_performed"] is False
        assert summary["counts"]["train_events"] == 80 * len(MISSING_MODES)
        assert summary["counts"]["valid_events"] == 40 * len(MISSING_MODES)
        assert summary["gate_error_rates"]["valid_full_train"] > summary["gate_error_rates"]["train_oof"]
        print("PASS: CFCompat v5.1 split-shift failure diagnostic smoke test")


if __name__ == "__main__":
    main()
