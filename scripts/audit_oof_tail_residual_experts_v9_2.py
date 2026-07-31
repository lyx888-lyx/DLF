"""Engineering and scientific audit for V9.2 grouped OOF tail experts."""

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
        default=str(
            REPO_ROOT
            / "result"
            / "oof_tail_residual_experts_v92"
            / "mosi"
            / "seed_1111"
        ),
    )
    return parser.parse_args()


def require(condition, message, errors):
    if not condition:
        errors.append(message)


def main():
    root = Path(parse_args().root)
    errors = []
    warnings = []
    oof_root = root / "oof_cfcompat"
    cache_path = oof_root / "nested_grouped_oof_cfcompat_cache_v92.pth"
    manifest_path = oof_root / "v92_nested_group_manifest.csv"
    summary_path = root / "oof_tail_residual_experts_v92_summary.json"
    prediction_path = root / "oof_tail_residual_experts_v92_predictions.csv"
    valid_matrix_path = root / "v92_valid_tail_capability_matrix.csv"
    test_matrix_path = root / "v92_test_tail_capability_matrix.csv"
    direction_path = root / "v92_oof_valid_direction_agreement.csv"
    for path in (
        cache_path,
        manifest_path,
        summary_path,
        prediction_path,
        valid_matrix_path,
        test_matrix_path,
        direction_path,
    ):
        require(path.is_file(), f"missing required output: {path.name}", errors)
    if errors:
        raise SystemExit("AUDIT FAILED\n- " + "\n- ".join(errors))

    cache = torch.load(cache_path, map_location="cpu")
    require(
        cache.get("version") == "nested_grouped_oof_cfcompat_v9_2",
        "unexpected OOF version",
        errors,
    )
    sample_count = len(cache.get("sample_ids", []))
    require(sample_count > 0, "empty OOF cache", errors)
    require(cache["labels"].shape == (sample_count, 1), "invalid OOF labels shape", errors)
    require(cache["oof_prediction"].shape == (sample_count, 1), "invalid OOF prediction shape", errors)
    require(cache["oof_feature"].shape[0] == sample_count, "invalid OOF feature count", errors)
    require(torch.isfinite(cache["oof_prediction"]).all(), "non-finite OOF prediction", errors)
    require(torch.isfinite(cache["oof_feature"]).all(), "non-finite OOF feature", errors)
    require(bool((cache["fold_index"] >= 0).all()), "unassigned outer fold", errors)
    require(len(set(cache["sample_ids"])) == sample_count, "duplicate OOF sample IDs", errors)

    manifest = pd.read_csv(manifest_path)
    folds = sorted(manifest.outer_fold.unique().tolist())
    for fold in folds:
        local = manifest.loc[manifest.outer_fold == fold]
        group_partitions = local.groupby("group_id").partition.nunique()
        require((group_partitions == 1).all(), f"group leakage inside outer fold {fold}", errors)
    holdout = manifest.loc[manifest.partition == "outer_holdout"]
    holdout_counts = holdout.groupby("sample_index").size()
    require(len(holdout_counts) == sample_count, "not every sample has an outer holdout", errors)
    require((holdout_counts == 1).all(), "sample appears in multiple outer holdouts", errors)
    group_holdout_counts = holdout.groupby("group_id").outer_fold.nunique()
    require((group_holdout_counts == 1).all(), "group appears in multiple outer holdout folds", errors)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    require(
        summary.get("method") == "nested_grouped_oof_cfcompat_tail_residual_v9_2",
        "unexpected method",
        errors,
    )
    require(int(summary.get("oof_outer_folds", 0)) >= 2, "invalid OOF fold count", errors)
    require(len(summary.get("candidates", {})) == 3, "summary must contain three candidates", errors)
    for role in ("shared_tail", "strong_negative", "strong_positive"):
        checkpoint = root / role / "oof_tail_residual_expert_v92_best.pth"
        history = root / role / "training_history.csv"
        require(checkpoint.is_file(), f"missing checkpoint for {role}", errors)
        require(history.is_file(), f"missing history for {role}", errors)
        candidate = summary.get("candidates", {}).get(role, {})
        if candidate.get("anchor_fallback"):
            warnings.append(f"{role} fell back to anchor")

    predictions = pd.read_csv(prediction_path)
    require(predictions.sample_id.astype(str).is_unique, "duplicate test sample IDs", errors)
    numeric = predictions.select_dtypes(include=[np.number]).to_numpy()
    require(np.isfinite(numeric).all(), "non-finite test predictions", errors)

    direction = pd.read_csv(direction_path)
    require(
        set(direction.role) == {"shared_tail", "strong_negative", "strong_positive"},
        "direction audit incomplete",
        errors,
    )
    mismatched = direction.loc[~direction.same_direction.astype(bool)]
    for row in mismatched.itertuples(index=False):
        warnings.append(
            f"OOF/Validation residual direction mismatch for {row.role}: "
            f"oof={row.oof_train_mean_residual:.6f} valid={row.valid_mean_residual:.6f}"
        )

    if errors:
        raise SystemExit("AUDIT FAILED\n- " + "\n- ".join(errors))

    print("ENGINEERING AUDIT PASSED")
    print("OOF samples:", sample_count)
    print("outer folds:", len(folds))
    print("anchor index:", summary.get("anchor_index"))
    for tail, policy in summary.get("validation_selected_tail_policy", {}).items():
        print(
            f"{tail} -> {policy.get('expert')}, "
            f"validation gain={float(policy.get('valid_gain', 0.0)):.6f}"
        )
    results = summary["test_results"]
    print("test anchor MAE: %.6f" % results["anchor"]["MAE"])
    print("test deployable OOF gate MAE: %.6f" % results["oof_gate_valid_selected"]["MAE"])
    print(
        "test true-region tail-policy MAE: %.6f"
        % results["true_region_valid_selected_tail_policy"]["MAE"]
    )
    for warning in warnings:
        print("WARNING:", warning)


if __name__ == "__main__":
    main()
