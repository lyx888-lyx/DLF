#!/usr/bin/env python3
"""Create explicitly versioned V9.4 outputs from the inherited V9.2 writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OLD_MODEL = "positive_residual_specialist_v92"
NEW_MODEL = "oof_positive_residual_v94"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("result/oof_positive_residual_v94/mosi/seed_1111"),
    )
    return parser.parse_args()


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    run_dir = parse_args().run_dir

    comparison = pd.read_csv(require(run_dir / "v92_test_comparison.csv"))
    comparison["model"] = comparison["model"].replace({OLD_MODEL: NEW_MODEL})
    comparison.to_csv(run_dir / "v94_test_comparison.csv", index=False)

    regions = pd.read_csv(require(run_dir / "v92_test_region_diagnostics.csv"))
    regions["model"] = regions["model"].replace({OLD_MODEL: NEW_MODEL})
    regions.to_csv(run_dir / "v94_test_region_diagnostics.csv", index=False)

    predictions = pd.read_csv(require(run_dir / "v92_test_predictions.csv"))
    predictions = predictions.rename(columns={"v92_selected": "v94_selected"})
    predictions.to_csv(run_dir / "v94_test_predictions.csv", index=False)

    final_valid = pd.read_csv(require(run_dir / "v92_final_valid_policy.csv"))
    final_valid.to_csv(run_dir / "v94_final_valid_policy.csv", index=False)

    summary_path = require(run_dir / "oof_positive_residual_v94_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary.get("test_results", {})
    if OLD_MODEL in results:
        results[NEW_MODEL] = results.pop(OLD_MODEL)
    summary["test_results"] = results
    summary["versioned_outputs"] = {
        "comparison": "v94_test_comparison.csv",
        "regions": "v94_test_region_diagnostics.csv",
        "predictions": "v94_test_predictions.csv",
        "valid_policy": "v94_final_valid_policy.csv",
        "crossfit_search": "v94_crossfit_calibrator_search.csv",
        "oof_assignments": "v94_oof_assignments.csv",
        "magnitude_history": "v94_magnitude_history.csv",
        "gate_history": "v94_gate_history.csv",
        "policy_search": "v94_valid_policy_search.csv",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    selected = summary["selected_policy"]
    selected_result = results[NEW_MODEL]
    print(f"V9.4 outputs finalized in {run_dir}")
    print(
        "V9.4 final "
        f"source={selected['source']} gate_mode={selected.get('gate_mode', 'none')} "
        f"gamma={selected['gamma']} residual_contributed="
        f"{summary['residual_training_contributed']} "
        f"Test(MAE={selected_result['MAE']:.4f} "
        f"worst={selected_result['worst_region_mae']:.4f} "
        f"op={selected_result['ordinary_positive_mae']:.4f} "
        f"nonpos={selected_result['nonpositive_mae']:.4f}) "
        f"reference_MAE={results['original_v71_hybrid']['MAE']:.4f}"
    )


if __name__ == "__main__":
    main()
