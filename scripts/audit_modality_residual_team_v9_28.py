"""Engineering and leakage audit for V9.28 outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.modality_residual_team_v928 import (
    AUDIT_VERSION,
    HEAD_NAMES,
    PRIMARY_STRATEGY,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size == 0:
        raise RuntimeError(f"empty output: {path}")
    return path


def main():
    cli = parse_args()
    output = Path(cli.v919_root) / "v928_modality_residual_team_audit"
    required = [
        "v928_report.md",
        "v928_summary.json",
        "v928_fold_metrics.csv",
        "v928_aggregate_metrics.csv",
        "v928_coefficient_inventory.csv",
        "v928_residual_head_metrics.csv",
        "v928_outer_fold_primary.csv",
        "v928_outer_predictions.csv",
        "v928_split_manifest.csv",
        "v928_source_integrity.csv",
        "v928_feature_schema.csv",
    ]
    for filename in required:
        require(output / filename)

    summary = json.loads(
        (output / "v928_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("version") != AUDIT_VERSION:
        raise RuntimeError("unexpected V9.28 version")
    if summary.get("router_or_sample_dependent_weights") is not False:
        raise RuntimeError("V9.28 must not contain a router")
    residual_inputs = summary.get("residual_head_inputs", {})
    if residual_inputs.get("text_tokens_or_embeddings_available") is not False:
        raise RuntimeError("residual heads received forbidden text features")
    provenance = summary.get("provenance", {})
    for key in (
        "inner_corrections_are_complete_crossfit_predictions",
        "global_coefficients_frozen_per_outer_fold",
        "nonnegative_coefficients_can_shrink_to_zero",
        "no_router",
        "no_per_sample_gate",
    ):
        if provenance.get(key) is not True:
            raise RuntimeError(f"missing provenance guarantee: {key}")
    for key in (
        "outer_holdout_labels_used_for_training",
        "outer_holdout_labels_used_for_coefficient_fit",
    ):
        if provenance.get(key) is not False:
            raise RuntimeError(f"outer-label leakage flag is invalid: {key}")

    features = pd.read_csv(output / "v928_feature_schema.csv")
    forbidden = features["uses_text_token_or_embedding"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    if bool(forbidden.any()):
        raise RuntimeError("feature schema exposes text token or embedding")
    allowed_anchor_names = {
        "text_anchor_scalar",
        "text_anchor_abs",
        "text_anchor_squared",
    }
    for name in features["feature_name"].astype(str):
        if "text" in name.lower() and name not in allowed_anchor_names:
            raise RuntimeError(f"forbidden residual feature: {name}")

    coefficients = pd.read_csv(output / "v928_coefficient_inventory.csv")
    primary = coefficients[coefficients["strategy"] == PRIMARY_STRATEGY]
    expected = cli.outer_folds * len(HEAD_NAMES)
    if len(primary) != expected:
        raise RuntimeError(
            f"expected {expected} primary coefficients, found {len(primary)}"
        )
    if set(primary["head"].astype(str)) != set(HEAD_NAMES):
        raise RuntimeError("primary coefficient heads are incomplete")
    values = primary["coefficient"].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all() or np.any(values < -1e-9):
        raise RuntimeError("invalid non-negative coefficient")
    upper = float(summary["configs"]["team"]["coefficient_upper_bound"])
    if np.any(values > upper + 1e-8):
        raise RuntimeError("coefficient exceeds registered upper bound")

    split = pd.read_csv(output / "v928_split_manifest.csv")
    if set(split["outer_fold"].astype(int)) != set(range(cli.outer_folds)):
        raise RuntimeError("outer-fold split manifest is incomplete")
    if bool((split.groupby("outer_fold")["inner_fold"].nunique() < 3).any()):
        raise RuntimeError("each outer fold requires at least three inner folds")
    if bool((split["sample_count"] <= 0).any()) or bool(
        (split["group_count"] <= 0).any()
    ):
        raise RuntimeError("empty split in manifest")

    sources = pd.read_csv(output / "v928_source_integrity.csv")
    if len(sources) != 2 * cli.outer_folds:
        raise RuntimeError("source integrity rows are incomplete")
    if bool((sources["dataset_label_max_abs"] > 1e-6).any()):
        raise RuntimeError("dataset/pool label alignment failed")
    if bool(sources["sha256"].astype(str).str.len().ne(64).any()):
        raise RuntimeError("invalid source checksum")

    predictions = pd.read_csv(output / "v928_outer_predictions.csv")
    if predictions["sample_id"].duplicated().any():
        raise RuntimeError("outer prediction table duplicates sample IDs")
    required_prediction_columns = {
        "text_anchor",
        "v921_convex_shrinkage",
        *(f"correction_{name}" for name in HEAD_NAMES),
        f"prediction_{PRIMARY_STRATEGY}",
    }
    missing = required_prediction_columns - set(predictions.columns)
    if missing:
        raise RuntimeError(f"prediction columns missing: {sorted(missing)}")
    numeric = predictions[list(required_prediction_columns)].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(numeric).all():
        raise RuntimeError("non-finite outer prediction output")

    models = list((output / "residual_models").glob("outer_fold_*/*_residual_v928.pth"))
    if len(models) != cli.outer_folds * len(HEAD_NAMES):
        raise RuntimeError("residual model inventory is incomplete")

    print("V9.28 ENGINEERING AUDIT PASSED")
    print("outer_folds:", cli.outer_folds)
    print("outer_samples:", len(predictions))
    print("primary_coefficients:", len(primary))
    print("residual_models:", len(models))
    print(
        "incremental_modality_signal_supported:",
        summary.get("incremental_modality_signal_supported"),
    )
    print(
        "deployment_replacement_supported:",
        summary.get("deployment_replacement_supported"),
    )


if __name__ == "__main__":
    main()
