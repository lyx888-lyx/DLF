#!/usr/bin/env python3
"""Audit alignment, cross-fitting and attribution outputs for V9.4."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REFERENCE = "original_v71_hybrid"
CANDIDATE = "oof_positive_residual_v94"
RESIDUAL_SOURCES = {"oof_magnitude_only", "oof_gated_residual"}
LEARNED_SOURCES = RESIDUAL_SOURCES | {"oof_calibrated_reference"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("result/oof_positive_residual_v94/mosi/seed_1111"),
    )
    parser.add_argument("--expected-reference-mae", type=float, default=None)
    parser.add_argument("--reference-tolerance", type=float, default=2e-4)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def finite_frame(frame: pd.DataFrame, columns):
    for column in columns:
        if column not in frame.columns:
            raise RuntimeError(f"Missing column: {column}")
        values = pd.to_numeric(frame[column], errors="coerce")
        if not np.isfinite(values.to_numpy()).all():
            raise RuntimeError(f"Non-finite values in column: {column}")


def max_abs_difference(left, right):
    return float(np.max(np.abs(np.asarray(left, dtype=float) - np.asarray(right, dtype=float))))


def main():
    cli = parse_args()
    run_dir = cli.run_dir

    summary = json.loads(
        require(run_dir / "oof_positive_residual_v94_summary.json").read_text(
            encoding="utf-8"
        )
    )
    comparison = pd.read_csv(require(run_dir / "v94_test_comparison.csv"))
    regions = pd.read_csv(require(run_dir / "v94_test_region_diagnostics.csv"))
    predictions = pd.read_csv(require(run_dir / "v94_test_predictions.csv"))
    assignments = pd.read_csv(require(run_dir / "v94_oof_assignments.csv"))
    calibrator_search = pd.read_csv(
        require(run_dir / "v94_crossfit_calibrator_search.csv")
    )
    magnitude_history = pd.read_csv(require(run_dir / "v94_magnitude_history.csv"))
    gate_history = pd.read_csv(require(run_dir / "v94_gate_history.csv"))
    policy_search = pd.read_csv(require(run_dir / "v94_valid_policy_search.csv"))

    if summary.get("method") != "crossfit_calibrated_staged_positive_residual_v9_4":
        raise RuntimeError(f"Unexpected method: {summary.get('method')}")
    if "calibration layer" not in str(summary.get("oof_scope", "")):
        raise RuntimeError("Summary does not state the limited OOF scope.")

    required_models = {REFERENCE, CANDIDATE}
    observed_models = set(comparison["model"].astype(str))
    missing_models = required_models - observed_models
    if missing_models:
        raise RuntimeError(f"Missing Test comparison models: {sorted(missing_models)}")
    comparison = comparison.set_index("model")
    reference_mae = float(comparison.loc[REFERENCE, "MAE"])
    candidate_mae = float(comparison.loc[CANDIDATE, "MAE"])
    if cli.expected_reference_mae is not None:
        difference = abs(reference_mae - cli.expected_reference_mae)
        if difference > cli.reference_tolerance:
            raise RuntimeError(
                "Reference MAE mismatch: "
                f"observed={reference_mae:.8f}, "
                f"expected={cli.expected_reference_mae:.8f}, "
                f"difference={difference:.8f}"
            )

    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate Test sample ids.")
    finite_frame(
        predictions,
        [
            "label",
            "original_v71_hybrid",
            "v94_selected",
            "positive_correction",
            "gate_probability",
            "magnitude",
        ],
    )
    if (predictions["positive_correction"] < -1e-8).any():
        raise RuntimeError("Negative positive_correction values.")
    max_correction = float(summary.get("max_correction", 0.5))
    if (predictions["magnitude"] > max_correction + 1e-6).any():
        raise RuntimeError("Magnitude exceeds configured max_correction.")
    if ((predictions["gate_probability"] < -1e-8) | (
        predictions["gate_probability"] > 1.0 + 1e-8
    )).any():
        raise RuntimeError("Gate probabilities are outside [0, 1].")

    if assignments["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate Train sample ids in OOF assignments.")
    finite_frame(
        assignments,
        [
            "fold",
            "label",
            "original_reference",
            "oof_calibrated_reference",
            "positive_residual_target",
        ],
    )
    group_fold_counts = assignments.groupby("group_key")["fold"].nunique()
    if (group_fold_counts > 1).any():
        examples = group_fold_counts[group_fold_counts > 1].head().index.tolist()
        raise RuntimeError(f"Group leakage across OOF folds: {examples}")
    expected_folds = int(summary["crossfit_folds"])
    observed_folds = sorted(assignments["fold"].astype(int).unique().tolist())
    if observed_folds != list(range(expected_folds)):
        raise RuntimeError(
            f"OOF folds mismatch: observed={observed_folds}, expected={list(range(expected_folds))}"
        )
    target = assignments["positive_residual_target"].to_numpy(dtype=float)
    if target.min() < -1e-8 or target.max() > max_correction + 1e-6:
        raise RuntimeError("OOF residual targets are outside the configured bounds.")

    finite_frame(
        calibrator_search,
        [
            "l2",
            "blend",
            "objective",
            "mae",
            "worst_region_mae",
            "ordinary_positive_mae",
        ],
    )
    if not np.isclose(calibrator_search["blend"].to_numpy(dtype=float), 0.0).any():
        raise RuntimeError("Calibrator search omitted the zero-blend fallback.")
    calibrator = summary.get("calibrator")
    if not calibrator:
        raise RuntimeError("Missing serialized calibrator.")
    selected_l2 = float(calibrator["l2"])
    selected_blend = float(calibrator["blend"])
    selected_match = calibrator_search[
        np.isclose(calibrator_search["l2"], selected_l2)
        & np.isclose(calibrator_search["blend"], selected_blend)
    ]
    if selected_match.empty:
        raise RuntimeError("Serialized calibrator was not present in the search table.")

    finite_frame(
        magnitude_history,
        [
            "epoch",
            "valid_overall_huber",
            "valid_active_huber",
            "valid_false_magnitude_mean",
            "valid_magnitude_mean",
            "valid_magnitude_max",
        ],
    )
    if gate_history.empty:
        raise RuntimeError("Gate history is empty.")
    finite_frame(
        gate_history,
        [
            "epoch",
            "valid_bce",
            "valid_average_precision",
            "valid_recall",
            "valid_specificity",
            "valid_balanced_accuracy",
            "valid_predicted_active_rate",
        ],
    )
    finite_frame(
        policy_search,
        [
            "mae",
            "worst_region_mae",
            "nonpositive_mae",
            "gamma",
            "threshold",
        ],
    )

    selected = summary["selected_policy"]
    source = str(selected["source"])
    gamma = float(selected["gamma"])
    residual_contributed = bool(summary["residual_training_contributed"])
    calibration_contributed = bool(summary["calibration_contributed"])
    training_contributed = bool(summary["training_contributed"])

    if source in RESIDUAL_SOURCES:
        if gamma <= 0.0 or not residual_contributed:
            raise RuntimeError("Residual family selected without positive gamma/contribution flag.")
    elif residual_contributed:
        raise RuntimeError("Residual contribution flag is true for a non-residual source.")
    if training_contributed != (source in LEARNED_SOURCES):
        raise RuntimeError("training_contributed is inconsistent with selected source.")
    if calibration_contributed and selected_blend <= 0.0:
        raise RuntimeError("Calibration contribution declared with zero calibrator blend.")

    if source == "legacy_reference":
        difference = max_abs_difference(
            predictions["v94_selected"], predictions["original_v71_hybrid"]
        )
        if difference > cli.prediction_tolerance:
            raise RuntimeError(f"Legacy reference mismatch: {difference}")
    elif source == "legacy_beta_search":
        if "best_valid_legacy_beta_search" not in predictions.columns:
            raise RuntimeError("Missing beta-search attribution column.")
        difference = max_abs_difference(
            predictions["v94_selected"], predictions["best_valid_legacy_beta_search"]
        )
        if difference > cli.prediction_tolerance:
            raise RuntimeError(f"Beta-search prediction mismatch: {difference}")
    elif source == "zero_shrinkage":
        if "best_valid_zero_shrinkage" not in predictions.columns:
            raise RuntimeError("Missing zero-shrinkage attribution column.")
        difference = max_abs_difference(
            predictions["v94_selected"], predictions["best_valid_zero_shrinkage"]
        )
        if difference > cli.prediction_tolerance:
            raise RuntimeError(f"Zero-shrinkage prediction mismatch: {difference}")
    elif source == "oof_gated_residual":
        mode = str(selected.get("gate_mode", "soft"))
        probability = predictions["gate_probability"].to_numpy(dtype=float)
        magnitude = predictions["magnitude"].to_numpy(dtype=float)
        if mode == "hard":
            correction = (
                probability >= float(selected.get("threshold", 0.5))
            ).astype(float) * magnitude
        else:
            correction = probability * magnitude
        if np.max(correction) <= 0.0:
            raise RuntimeError("Selected gated residual is identically zero.")
    elif source == "oof_magnitude_only":
        if predictions["magnitude"].max() <= 0.0:
            raise RuntimeError("Selected magnitude-only residual is identically zero.")
    elif source != "oof_calibrated_reference":
        raise RuntimeError(f"Unknown selected source: {source}")

    ordinary = regions[regions["region"] == "ordinary_positive"].set_index("model")
    for model in (REFERENCE, CANDIDATE):
        if model not in ordinary.index:
            raise RuntimeError(f"Missing ordinary-positive diagnostics: {model}")

    print("AUDIT PASSED")
    print(
        f"calibrator=l2:{selected_l2} blend:{selected_blend} "
        f"folds:{expected_folds} Train-OOF-MAE="
        f"{np.mean(np.abs(assignments['oof_calibrated_reference'] - assignments['label'])):.6f}"
    )
    print(
        "magnitude_valid: "
        f"best_active={magnitude_history['valid_active_huber'].min():.6f}, "
        f"best_overall={magnitude_history['valid_overall_huber'].min():.6f}, "
        f"max_output={magnitude_history['valid_magnitude_max'].max():.6f}"
    )
    print(
        "gate_valid: "
        f"best_AP={gate_history['valid_average_precision'].max():.4f}, "
        f"best_balanced_accuracy={gate_history['valid_balanced_accuracy'].max():.4f}, "
        f"selected_epoch={summary['gate_epoch']}"
    )
    print(
        "selected_policy="
        f"{source} gate_mode={selected.get('gate_mode', 'none')} "
        f"threshold={selected.get('threshold', 0.0)} gamma={gamma}"
    )
    print(
        f"calibration_contributed={calibration_contributed} "
        f"residual_training_contributed={residual_contributed}"
    )
    print(f"reference_mae={reference_mae:.6f}")
    print(f"candidate_mae={candidate_mae:.6f}")
    print(f"overall_mae_gain={reference_mae - candidate_mae:+.6f}")
    print(
        "ordinary_positive: "
        f"count={int(ordinary.loc[REFERENCE, 'count'])}, "
        f"mae_gain="
        f"{float(ordinary.loc[REFERENCE, 'mae']) - float(ordinary.loc[CANDIDATE, 'mae']):+.6f}, "
        f"reference_bias={float(ordinary.loc[REFERENCE, 'signed_bias']):+.6f}, "
        f"candidate_bias={float(ordinary.loc[CANDIDATE, 'signed_bias']):+.6f}, "
        f"reference_cross_zero={float(ordinary.loc[REFERENCE, 'cross_zero_rate']):.4f}, "
        f"candidate_cross_zero={float(ordinary.loc[CANDIDATE, 'cross_zero_rate']):.4f}"
    )


if __name__ == "__main__":
    main()
