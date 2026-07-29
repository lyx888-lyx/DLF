#!/usr/bin/env python3
"""Audit V9.2 outputs before interpreting any Test result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REFERENCE = "original_v71_hybrid"
CANDIDATE = "positive_residual_specialist_v92"
ORDINARY_POSITIVE = "ordinary_positive"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("result/positive_residual_specialist_v92/mosi/seed_1111"),
    )
    parser.add_argument("--expected-reference-mae", type=float, default=None)
    parser.add_argument("--reference-tolerance", type=float, default=1e-4)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def require_file(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def finite_series(frame: pd.DataFrame, column: str):
    if column not in frame:
        raise KeyError(f"Missing column: {column}")
    values = pd.to_numeric(frame[column], errors="coerce")
    if not values.map(math.isfinite).all():
        raise RuntimeError(f"Non-finite values in {column}")
    return values


def max_abs_difference(left, right):
    return float((pd.to_numeric(left) - pd.to_numeric(right)).abs().max())


def main():
    cli = parse_args()
    run_dir = cli.run_dir
    summary = json.loads(
        require_file(
            run_dir / "positive_residual_specialist_v92_summary.json"
        ).read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(require_file(run_dir / "v92_test_comparison.csv"))
    regions = pd.read_csv(
        require_file(run_dir / "v92_test_region_diagnostics.csv")
    )
    predictions = pd.read_csv(
        require_file(run_dir / "v92_test_predictions.csv")
    )
    history = pd.read_csv(
        require_file(run_dir / "v92_training_history.csv")
    )
    weights = pd.read_csv(
        require_file(run_dir / "v92_train_region_weights.csv")
    )

    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate sample_id in V9.2 Test predictions")
    for column in (
        "label",
        "original_v71_hybrid",
        "v92_selected",
        "legacy_student",
        "positive_correction",
        "gate_probability",
        "magnitude",
    ):
        finite_series(predictions, column)

    max_correction = float(summary["max_correction"])
    correction = pd.to_numeric(predictions["positive_correction"])
    magnitude = pd.to_numeric(predictions["magnitude"])
    gate = pd.to_numeric(predictions["gate_probability"])
    if float(correction.min()) < -1e-8:
        raise RuntimeError("V9.2 emitted a negative correction")
    if float(correction.max()) > max_correction + 1e-6:
        raise RuntimeError("V9.2 correction exceeded max_correction")
    if float(magnitude.min()) < -1e-8 or float(magnitude.max()) > max_correction + 1e-6:
        raise RuntimeError("V9.2 magnitude is outside its configured range")
    if float(gate.min()) < -1e-8 or float(gate.max()) > 1.0 + 1e-8:
        raise RuntimeError("V9.2 gate probability is outside [0, 1]")
    if float((correction - gate * magnitude).abs().max()) > 1e-5:
        raise RuntimeError("positive_correction != gate_probability * magnitude")

    model_rows = comparison.set_index("model")
    for model in (REFERENCE, CANDIDATE):
        if model not in model_rows.index:
            raise RuntimeError(f"Missing comparison row: {model}")
    reference_mae = float(model_rows.loc[REFERENCE, "MAE"])
    candidate_mae = float(model_rows.loc[CANDIDATE, "MAE"])
    if cli.expected_reference_mae is not None:
        difference = abs(reference_mae - cli.expected_reference_mae)
        if difference > cli.reference_tolerance:
            raise RuntimeError(
                "Frozen V7.1 reference was not reproduced: "
                f"observed={reference_mae:.8f}, "
                f"expected={cli.expected_reference_mae:.8f}, "
                f"difference={difference:.8f}"
            )

    ordinary = regions[
        regions["region"] == ORDINARY_POSITIVE
    ].set_index("model")
    for model in (REFERENCE, CANDIDATE):
        if model not in ordinary.index:
            raise RuntimeError(
                f"Missing ordinary-positive diagnostics: {model}"
            )
    ref_count = int(ordinary.loc[REFERENCE, "count"])
    cand_count = int(ordinary.loc[CANDIDATE, "count"])
    if ref_count != cand_count or ref_count <= 0:
        raise RuntimeError(
            "Ordinary-positive count mismatch: "
            f"reference={ref_count}, candidate={cand_count}"
        )

    if len(weights) != 5 or (weights["count"] <= 0).any():
        raise RuntimeError("Invalid Train region counts")
    weighted_mean = float(
        (weights["count"] * weights["weight"]).sum()
        / weights["count"].sum()
    )
    if abs(weighted_mean - 1.0) > 1e-4:
        raise RuntimeError(
            f"Region weights are not sample-normalized: {weighted_mean}"
        )

    if history.empty:
        raise RuntimeError("V9.2 training history is empty")
    for column in (
        "expert_candidate_mae",
        "expert_candidate_ordinary_positive_mae",
        "expert_candidate_nonpositive_mae",
        "residual_huber",
        "gate_accuracy",
        "valid_correction_mean",
        "selected_mae",
        "selected_beta",
        "selected_shrinkage",
        "selected_gamma",
    ):
        finite_series(history, column)

    selected = summary["selected_policy"]
    selected_epoch = int(summary["selected_epoch"])
    source = str(selected["source"])
    gamma = float(selected["gamma"])
    training_contributed = bool(summary["training_contributed"])
    expected_column = {
        "legacy_reference": "original_v71_hybrid",
        "legacy_beta_search": "best_valid_legacy_beta_search",
        "zero_shrinkage": "best_valid_zero_shrinkage",
        "positive_residual": "best_valid_positive_residual",
    }.get(source)
    if expected_column is None:
        raise RuntimeError(f"Unknown selected source: {source}")
    if expected_column not in predictions:
        raise RuntimeError(
            f"Missing prediction column for selected source: {expected_column}"
        )
    difference = max_abs_difference(
        predictions["v92_selected"], predictions[expected_column]
    )
    if difference > cli.prediction_tolerance:
        raise RuntimeError(
            "Saved selected prediction does not match its declared family: "
            f"source={source}, max_abs_difference={difference}"
        )

    if source == "positive_residual":
        if selected_epoch <= 0 or gamma <= 0.0:
            raise RuntimeError(
                "positive_residual selected without trained epoch and positive gamma"
            )
        if not training_contributed:
            raise RuntimeError(
                "Summary failed to mark selected positive residual as training"
            )
    elif training_contributed:
        raise RuntimeError(
            "training_contributed=True for a non-specialist policy"
        )

    best_op = float(history["expert_candidate_ordinary_positive_mae"].min())
    first_op = float(history.iloc[0]["expert_candidate_ordinary_positive_mae"])
    best_overall = float(history["expert_candidate_mae"].min())
    first_overall = float(history.iloc[0]["expert_candidate_mae"])

    print("AUDIT PASSED")
    print(f"selected_epoch={selected_epoch}")
    print(
        "selected_policy="
        f"{source} committee={selected['committee']} beta={selected['beta']} "
        f"shrinkage={selected['shrinkage']} gamma={selected['gamma']}"
    )
    print(f"training_contributed={training_contributed}")
    print(f"reference_mae={reference_mae:.6f}")
    print(f"candidate_mae={candidate_mae:.6f}")
    print(f"overall_mae_gain={reference_mae - candidate_mae:+.6f}")
    print(
        "specialist_valid: "
        f"overall_first={first_overall:.6f}, overall_best={best_overall:.6f}, "
        f"ordinary_positive_first={first_op:.6f}, "
        f"ordinary_positive_best={best_op:.6f}"
    )
    print(
        "correction: "
        f"mean={float(correction.mean()):.6f}, "
        f"max={float(correction.max()):.6f}, "
        f"gate_mean={float(gate.mean()):.6f}"
    )
    print(
        "ordinary_positive: "
        f"count={ref_count}, "
        f"mae_gain="
        f"{float(ordinary.loc[REFERENCE, 'mae']) - float(ordinary.loc[CANDIDATE, 'mae']):+.6f}, "
        f"reference_bias={float(ordinary.loc[REFERENCE, 'signed_bias']):+.6f}, "
        f"candidate_bias={float(ordinary.loc[CANDIDATE, 'signed_bias']):+.6f}, "
        f"reference_cross_zero={float(ordinary.loc[REFERENCE, 'cross_zero_rate']):.4f}, "
        f"candidate_cross_zero={float(ordinary.loc[CANDIDATE, 'cross_zero_rate']):.4f}"
    )


if __name__ == "__main__":
    main()
