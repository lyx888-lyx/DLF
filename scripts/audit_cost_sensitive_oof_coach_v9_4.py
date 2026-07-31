"""Engineering and scientific audit for V9.4 fully nested action-cost routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ACTION_NAMES = (
    "anchor",
    "strong_negative",
    "boundary",
    "positive",
    "strong_positive",
)
SPECIALIST_NAMES = ACTION_NAMES[1:]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(
            REPO_ROOT
            / "result"
            / "cost_sensitive_oof_coach_v94"
            / "mosi"
            / "seed_1111"
        ),
    )
    parser.add_argument("--require-deployable-gain", type=float, default=None)
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
    per_fold_sample = frame.groupby(["outer_fold", "sample_index"]).size()
    if not per_fold_sample.eq(1).all():
        raise RuntimeError("manifest has repeated fold/sample rows")
    holdout = frame.loc[frame.partition == "outer_holdout"]
    holdout_counts = holdout.groupby("sample_index").size()
    if len(holdout_counts) != expected_samples or not holdout_counts.eq(1).all():
        raise RuntimeError("every sample must be exactly one top outer holdout")
    for fold, local in frame.groupby("outer_fold"):
        if local.sample_index.nunique() != expected_samples:
            raise RuntimeError(f"outer fold {fold} does not record every sample")
        if not local.groupby("group_id").partition.nunique().eq(1).all():
            raise RuntimeError(f"video-group leakage in outer fold {fold}")


def main():
    cli = parse_args()
    root = Path(cli.root)
    pool_root = root / "oof_expert_pool"
    pool_path = require(pool_root / "fully_nested_oof_expert_pool_v94.pth")
    pool = torch.load(pool_path, map_location="cpu")
    required_pool = {
        "sample_ids",
        "labels",
        "anchor",
        "function_space",
        "expert_names",
        "action_names",
        "expert_predictions",
        "expert_confidences",
        "fold_index",
        "fold_metadata",
        "protocol",
    }
    if not required_pool.issubset(pool):
        raise ValueError("nested pool is incomplete")
    if tuple(pool["action_names"]) != ACTION_NAMES:
        raise RuntimeError("action schema mismatch")
    if tuple(pool["expert_names"]) != SPECIALIST_NAMES:
        raise RuntimeError("specialist schema mismatch")
    n = len(pool["sample_ids"])
    if n != 1284 or len(set(pool["sample_ids"])) != n:
        raise RuntimeError("MOSI nested pool must contain 1284 unique samples")
    expected_shapes = {
        "labels": (n, 1),
        "anchor": (n, 1),
        "expert_predictions": (n, 4, 1),
        "expert_confidences": (n, 4, 1),
    }
    for key, shape in expected_shapes.items():
        if tuple(pool[key].shape) != shape or not torch.isfinite(pool[key]).all():
            raise RuntimeError(f"invalid nested pool tensor {key}")
    confidence = pool["expert_confidences"].numpy()
    if np.any((confidence < 0) | (confidence > 1)):
        raise RuntimeError("specialist confidence outside [0,1]")
    audit_manifest(pool_root / "v94_top_outer_manifest.csv", n)
    if len(pool["fold_metadata"]) != int(pool["fold_index"].unique().numel()):
        raise RuntimeError("outer fold metadata count mismatch")
    for metadata in pool["fold_metadata"]:
        inner_cache = require(Path(metadata["inner_oof_cache"]))
        inner = torch.load(inner_cache, map_location="cpu")
        if not torch.isfinite(inner["oof_prediction"]).all():
            raise FloatingPointError(f"non-finite inner OOF cache: {inner_cache}")

    summary_path = require(root / "cost_sensitive_oof_coach_v94_summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    calibration = pd.read_csv(require(root / "v94_route_calibration.csv"))
    if not {"anchor", "cost_sensitive"}.issubset(set(calibration["mode"])):
        raise RuntimeError("route calibration is incomplete")
    numeric = calibration[
        ["mae", "objective", "harm_over_010_rate", "activation_rate"]
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise FloatingPointError("route calibration contains non-finite values")

    test_summary = pd.read_csv(require(root / "v94_test_summary.csv"))
    by_name = test_summary.set_index("model")
    required_models = {
        "anchor",
        "anchor_score_region_valid_selected",
        "ordinal_region_valid_selected",
        "cost_sensitive_oof_coach_valid_selected",
        "true_region_expert_policy",
        "sample_oracle_upper_bound",
    }
    if not required_models.issubset(by_name.index):
        raise RuntimeError("V9.4 test summary is incomplete")
    predictions = pd.read_csv(
        require(root / "cost_sensitive_oof_coach_v94_predictions.csv")
    )
    if len(predictions) != 686 or predictions.sample_id.nunique() != 686:
        raise RuntimeError("MOSI Test predictions must contain 686 unique samples")
    if not set(predictions.selected_action.unique()).issubset(set(range(5))):
        raise RuntimeError("invalid selected action index")

    checkpoints = [Path(summary["coach_checkpoint"])]
    for role in SPECIALIST_NAMES:
        checkpoints.append(
            root / "final_specialists" / f"specialist_{role}_v94.pth"
        )
    for path in checkpoints:
        payload = torch.load(require(path), map_location="cpu")
        if not isinstance(payload, dict) or not isinstance(payload.get("state_dict"), dict):
            raise ValueError(f"malformed checkpoint: {path}")

    anchor_mae = float(by_name.loc["anchor", "MAE"])
    coach_mae = float(
        by_name.loc["cost_sensitive_oof_coach_valid_selected", "MAE"]
    )
    true_region_mae = float(by_name.loc["true_region_expert_policy", "MAE"])
    oracle_mae = float(by_name.loc["sample_oracle_upper_bound", "MAE"])
    deployable_gain = anchor_mae - coach_mae
    if (
        cli.require_deployable_gain is not None
        and deployable_gain + 1e-12 < float(cli.require_deployable_gain)
    ):
        raise RuntimeError(
            f"deployable gain {deployable_gain:.6f} is below required "
            f"{float(cli.require_deployable_gain):.6f}"
        )

    print("ENGINEERING AUDIT PASSED")
    print("nested OOF samples:", n)
    print("top outer folds:", int(pool["fold_index"].unique().numel()))
    print("action schema:", list(pool["action_names"]))
    print("enabled specialists:", summary["enabled_specialists"])
    print("validation-selected route:", summary["route_policy"])
    print(f"test anchor MAE: {anchor_mae:.6f}")
    print(f"test deployable coach MAE: {coach_mae:.6f}")
    print(f"test deployable gain: {deployable_gain:+.6f}")
    print(f"test true-region MAE: {true_region_mae:.6f}")
    print(f"test sample oracle MAE: {oracle_mae:.6f}")
    if deployable_gain <= 0:
        print("WARNING: deployable V9.4 coach did not beat Anchor on Test")


if __name__ == "__main__":
    main()
