"""Engineering and scientific audit for the V9.3 ordinal advantage coach."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
REGIONS = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)
SPECIALISTS = ("strong_negative", "boundary", "positive", "strong_positive")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(REPO_ROOT / "result" / "ordinal_advantage_coach_v93" / "mosi" / "seed_1111"),
    )
    parser.add_argument("--require-deployable-gain", type=float, default=0.0)
    return parser.parse_args()


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def audit_manifest(path: Path, expected_samples: int):
    frame = pd.read_csv(require(path))
    required = {"outer_fold", "sample_index", "group_id", "partition"}
    if not required.issubset(frame.columns):
        raise ValueError(f"manifest missing columns: {sorted(required - set(frame.columns))}")
    holdout_counts = frame.loc[frame.partition == "outer_holdout"].groupby("sample_index").size()
    if len(holdout_counts) != expected_samples or not holdout_counts.eq(1).all():
        raise RuntimeError(f"every sample must be exactly one outer holdout in {path}")
    for fold, local in frame.groupby("outer_fold"):
        group_partitions = local.groupby("group_id").partition.nunique()
        if not group_partitions.eq(1).all():
            raise RuntimeError(f"group leakage in {path}, fold {fold}")


def main():
    cli = parse_args()
    root = Path(cli.root)
    summary_path = require(root / "ordinal_advantage_coach_v93_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    region_oof = pd.read_csv(require(root / "v93_region_oof_predictions.csv"))
    if len(region_oof) != 1284 or region_oof.sample_id.nunique() != 1284:
        raise RuntimeError("region OOF cache must contain 1284 unique MOSI train samples")
    probability_columns = ["p_" + name for name in REGIONS]
    probabilities = region_oof[probability_columns].to_numpy(dtype=np.float64)
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-7):
        raise FloatingPointError("invalid region OOF probabilities")
    if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("region OOF probabilities do not sum to one")
    audit_manifest(root / "v93_region_group_manifest.csv", len(region_oof))

    valid_pool = pd.read_csv(require(root / "v93_valid_expert_pool_predictions.csv"))
    test_pool = pd.read_csv(require(root / "v93_test_expert_pool_predictions.csv"))
    if valid_pool.sample_id.duplicated().any() or test_pool.sample_id.duplicated().any():
        raise RuntimeError("expert-pool sample IDs are not unique")
    for frame, split in ((valid_pool, "valid"), (test_pool, "test")):
        required_columns = {"sample_id", "label", "anchor"}
        for name in SPECIALISTS:
            required_columns.update({name + "_prediction", name + "_confidence"})
        if not required_columns.issubset(frame.columns):
            raise ValueError(f"{split} expert pool is incomplete")
        numeric = frame[list(required_columns - {"sample_id"})].to_numpy(dtype=np.float64)
        if not np.isfinite(numeric).all():
            raise FloatingPointError(f"non-finite {split} expert-pool value")

    advantage = pd.read_csv(require(root / "v93_advantage_oof_predictions.csv"))
    if advantage.sample_id.nunique() != len(advantage):
        raise RuntimeError("advantage OOF sample IDs are not unique")
    audit_manifest(root / "v93_advantage_group_manifest.csv", len(advantage))
    for name in SPECIALISTS:
        columns = [name + "_predicted_gain", name + "_win_probability", name + "_realized_gain"]
        values = advantage[columns].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError(f"invalid advantage OOF outputs for {name}")
        probability = advantage[name + "_win_probability"].to_numpy(dtype=np.float64)
        if np.any((probability < 0) | (probability > 1)):
            raise RuntimeError(f"win probability outside [0,1] for {name}")

    calibration = pd.read_csv(require(root / "v93_route_calibration.csv"))
    if not {"anchor", "anchor_score_region", "hard_ordinal", "ordinal_advantage"}.issubset(set(calibration["mode"])):
        raise RuntimeError("route calibration is missing required baselines")
    if not np.isfinite(calibration[["mae", "objective", "harm_over_010_rate"]].to_numpy()).all():
        raise FloatingPointError("non-finite route calibration")

    predictions = pd.read_csv(require(root / "ordinal_advantage_coach_v93_predictions.csv"))
    if len(predictions) != len(test_pool) or predictions.sample_id.tolist() != test_pool.sample_id.tolist():
        raise RuntimeError("final prediction sample order does not match test expert pool")
    test_summary = pd.read_csv(require(root / "v93_test_summary.csv"))
    by_name = test_summary.set_index("model")
    required_models = {
        "anchor",
        "anchor_score_region_valid_selected",
        "hard_ordinal_valid_selected",
        "ordinal_advantage_coach_valid_selected",
        "true_region_expert_policy",
        "sample_oracle_upper_bound",
    }
    if not required_models.issubset(by_name.index):
        raise RuntimeError("test summary is incomplete")

    checkpoint_paths = [summary["region_coach"]["checkpoint"]]
    checkpoint_paths.extend(
        payload["checkpoint"] for payload in summary["advantage_heads"].values()
    )
    for value in checkpoint_paths:
        payload = torch.load(require(Path(value)), map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError(f"malformed V9.3 checkpoint: {value}")

    anchor_mae = float(by_name.loc["anchor", "MAE"])
    anchor_score_mae = float(by_name.loc["anchor_score_region_valid_selected", "MAE"])
    hard_mae = float(by_name.loc["hard_ordinal_valid_selected", "MAE"])
    deployable_mae = float(by_name.loc["ordinal_advantage_coach_valid_selected", "MAE"])
    true_region_mae = float(by_name.loc["true_region_expert_policy", "MAE"])
    deployable_gain = anchor_mae - deployable_mae
    if deployable_gain + 1e-12 < float(cli.require_deployable_gain):
        raise RuntimeError(
            f"deployable gain {deployable_gain:.6f} is below required {cli.require_deployable_gain:.6f}"
        )

    print("ENGINEERING AUDIT PASSED")
    print("region OOF samples:", len(region_oof))
    print("region OOF macro F1:", summary["region_coach"]["oof_metrics"]["macro_f1"])
    print("validation-selected route:", summary["route_policy"])
    print(f"test anchor MAE: {anchor_mae:.6f}")
    print(f"test anchor-score region MAE: {anchor_score_mae:.6f}")
    print(f"test hard ordinal MAE: {hard_mae:.6f}")
    print(f"test deployable advantage coach MAE: {deployable_mae:.6f}")
    print(f"test true-region expert policy MAE: {true_region_mae:.6f}")
    if deployable_gain <= 0:
        print("WARNING: deployable coach did not beat the anchor on Test")


if __name__ == "__main__":
    main()
