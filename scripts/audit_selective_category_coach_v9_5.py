"""Engineering and scientific audit for V9.5."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "result" / "selective_category_coach_v95" / "mosi" / "seed_1111"),
    )
    parser.add_argument("--require-deployable-gain", type=float, default=None)
    return parser.parse_args()


def require(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    cli = parse_args()
    root = Path(cli.root)
    summary = json.loads(require(root / "selective_category_coach_v95_summary.json").read_text())
    oof = pd.read_csv(require(root / "v95_region_oof_predictions.csv"))
    if len(oof) != 1284 or oof.sample_id.nunique() != 1284:
        raise RuntimeError("V9.5 region OOF must contain 1284 unique samples")
    probability_columns = [
        "p_strong_negative", "p_negative", "p_boundary", "p_positive", "p_strong_positive"
    ]
    probabilities = oof[probability_columns].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.allclose(
        probabilities.sum(axis=1), 1.0, atol=1e-5
    ):
        raise RuntimeError("invalid V9.5 OOF region probabilities")
    checkpoint = torch.load(
        require(summary["region_calibrator"]["checkpoint"]), map_location="cpu"
    )
    if not isinstance(checkpoint.get("state_dict"), dict):
        raise RuntimeError("invalid V9.5 calibrator checkpoint")
    calibration = pd.read_csv(require(root / "v95_route_calibration.csv"))
    if not {"anchor", "selective_category"}.issubset(set(calibration["mode"])):
        raise RuntimeError("V9.5 calibration is incomplete")
    predictions = pd.read_csv(require(root / "selective_category_coach_v95_predictions.csv"))
    test_summary = pd.read_csv(require(root / "v95_test_summary.csv")).set_index("model")
    required_models = {
        "anchor", "selective_category_coach_valid_selected",
        "true_region_expert_policy", "sample_oracle_upper_bound",
    }
    if not required_models.issubset(test_summary.index):
        raise RuntimeError("V9.5 test summary is incomplete")
    if len(predictions) != 686 or predictions.sample_id.nunique() != 686:
        raise RuntimeError("V9.5 prediction file must contain 686 unique Test samples")
    anchor = float(test_summary.loc["anchor", "MAE"])
    coach = float(test_summary.loc["selective_category_coach_valid_selected", "MAE"])
    gain = anchor - coach
    print("ENGINEERING AUDIT PASSED")
    print("region OOF samples:", len(oof))
    print("region OOF metrics:", summary["region_calibrator"]["oof_metrics"])
    print("anchor-threshold OOF metrics:", summary["region_calibrator"]["anchor_metrics"])
    print("validation-selected route:", summary["route_policy"])
    print("test activation rate:", summary["test_activation_rate"])
    print("test selected experts:", summary["test_selected_expert_counts"])
    print(f"test anchor MAE: {anchor:.6f}")
    print(f"test selective coach MAE: {coach:.6f}")
    print(f"test gain: {gain:+.6f}")
    print(
        "test true-region MAE:",
        float(test_summary.loc["true_region_expert_policy", "MAE"]),
    )
    if cli.require_deployable_gain is not None and gain < cli.require_deployable_gain:
        raise RuntimeError(
            f"deployable gain {gain:.6f} is below required {cli.require_deployable_gain:.6f}"
        )
    if gain <= 0:
        print("WARNING: V9.5 deployable coach did not beat Anchor on Test")


if __name__ == "__main__":
    main()
