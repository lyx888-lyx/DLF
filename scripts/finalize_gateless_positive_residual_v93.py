#!/usr/bin/env python3
"""Create explicitly versioned V9.3 output files from the inherited V9.2 writer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


OLD_MODEL = "positive_residual_specialist_v92"
NEW_MODEL = "gateless_positive_residual_v93"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("result/gateless_positive_residual_v93/mosi/seed_1111"),
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
    comparison.to_csv(run_dir / "v93_test_comparison.csv", index=False)

    regions = pd.read_csv(require(run_dir / "v92_test_region_diagnostics.csv"))
    regions["model"] = regions["model"].replace({OLD_MODEL: NEW_MODEL})
    regions.to_csv(run_dir / "v93_test_region_diagnostics.csv", index=False)

    predictions = pd.read_csv(require(run_dir / "v92_test_predictions.csv"))
    predictions = predictions.rename(columns={"v92_selected": "v93_selected"})
    predictions.to_csv(run_dir / "v93_test_predictions.csv", index=False)

    final_valid = pd.read_csv(require(run_dir / "v92_final_valid_policy.csv"))
    final_valid.to_csv(run_dir / "v93_final_valid_policy.csv", index=False)

    summary_path = require(run_dir / "gateless_positive_residual_v93_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    results = summary.get("test_results", {})
    if OLD_MODEL in results:
        results[NEW_MODEL] = results.pop(OLD_MODEL)
    summary["test_results"] = results
    summary["versioned_outputs"] = {
        "comparison": "v93_test_comparison.csv",
        "regions": "v93_test_region_diagnostics.csv",
        "predictions": "v93_test_predictions.csv",
        "valid_policy": "v93_final_valid_policy.csv",
        "history": "v93_training_history.csv",
        "policy_search": "v93_valid_policy_search.csv",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"V9.3 outputs finalized in {run_dir}")


if __name__ == "__main__":
    main()
