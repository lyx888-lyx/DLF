#!/usr/bin/env python3
"""Audit V9.3 outputs before interpreting MOSI Test performance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REFERENCE = "original_v71_hybrid"
CANDIDATE = "gateless_positive_residual_v93"
ORDINARY_POSITIVE = "ordinary_positive"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("result/gateless_positive_residual_v93/mosi/seed_1111"),
    )
    parser.add_argument("--expected-reference-mae", type=float, default=None)
    parser.add_argument("--reference-tolerance", type=float, default=1e-4)
    parser.add_argument("--prediction-tolerance", type=float, default=1e-6)
    return parser.parse_args()


def require(path: Path):
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
    summary = json.loads(require(
        run_dir / "gateless_positive_residual_v93_summary.json"
    ).read_text(encoding="utf-8"))
    comparison = pd.read_csv(require(run_dir / "v93_test_comparison.csv"))
    regions = pd.read_csv(require(run_dir / "v93_test_region_diagnostics.csv"))
    predictions = pd.read_csv(require(run_dir / "v93_test_predictions.csv"))
    history = pd.read_csv(require(run_dir / "v93_training_history.csv"))
    weights = pd.read_csv(require(run_dir / "v92_train_region_weights.csv"))

    if summary.get("method") != "gateless_positive_residual_specialist_v9_3":
        raise RuntimeError(f"Unexpected method: {summary.get('method')}")
    if summary.get("gate_removed") is not True:
        raise RuntimeError("V9.3 summary does not confirm gate removal")

    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate sample_id in V9.3 Test predictions")
    for column in (
        "label",
        "original_v71_hybrid",
        "v93_selected",
        "positive_correction",
        "magnitude",
        "gate_probability",
    ):
        finite_series(predictions, column)

    correction = pd.to_numeric(predictions["positive_correction"])
    max_correction = float(summary["max_correction"])
    if float(correction.min()) < -1e-8:
        raise RuntimeError("V9.3 produced a negative correction")
    if float(correction.max()) > max_correction + 1e-6:
        raise RuntimeError(
            f"Correction exceeds max_correction: {correction.max()} > {max_correction}"
        )
    if float((predictions["gate_probability"] - 1.0).abs().max()) > 1e-6:
        raise RuntimeError("Gateless V9.3 did not save unit gate probabilities")
    if max_abs_difference(predictions["positive_correction"], predictions["magnitude"]) > 1e-6:
        raise RuntimeError("V9.3 correction is not equal to learned magnitude")

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
            raise RuntimeError(f"Missing ordinary-positive row: {model}")
    ref_count = int(ordinary.loc[REFERENCE, "count"])
    cand_count = int(ordinary.loc[CANDIDATE, "count"])
    if ref_count != cand_count or ref_count <= 0:
        raise RuntimeError(
            f"Ordinary-positive count mismatch: {ref_count} vs {cand_count}"
        )

    if len(weights) != 5 or (weights["count"] <= 0).any():
        raise RuntimeError("Invalid Train region counts")
    weighted_mean = float(
        (weights["count"] * weights["weight"]).sum() / weights["count"].sum()
    )
    if abs(weighted_mean - 1.0) > 1e-4:
        raise RuntimeError(f"Region weights are not normalized: {weighted_mean}")

    selected = summary["selected_policy"]
    source = str(selected["source"])
    gamma = float(selected["gamma"])
    training_contributed = bool(summary["training_contributed"])
    if source == "positive_residual":
        if gamma <= 0.0 or not training_contributed:
            raise RuntimeError("Selected residual is not marked as a contribution")
    elif training_contributed:
        raise RuntimeError("Non-residual policy marked training_contributed=True")

    expected_column = {
        "legacy_reference": "original_v71_hybrid",
        "legacy_beta_search": "best_valid_legacy_beta_search",
        "zero_shrinkage": "best_valid_zero_shrinkage",
        "positive_residual": "best_valid_positive_residual",
    }.get(source)
    if expected_column is None or expected_column not in predictions:
        raise RuntimeError(f"Missing selected-family prediction for {source}")
    prediction_difference = max_abs_difference(
        predictions["v93_selected"], predictions[expected_column]
    )
    if prediction_difference > cli.prediction_tolerance:
        raise RuntimeError(
            "Selected prediction does not match declared family: "
            f"source={source}, max_abs_difference={prediction_difference}"
        )

    if history.empty:
        raise RuntimeError("V9.3 training history is empty")
    for column in (
        "valid_residual_huber",
        "valid_active_huber",
        "valid_false_correction_mean",
        "valid_correction_mean",
        "valid_correction_max",
    ):
        finite_series(history, column)

    print("AUDIT PASSED")
    print(f"selected_epoch={summary['selected_epoch']}")
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
        "residual_valid: "
        f"first={float(history.iloc[0]['valid_residual_huber']):.6f}, "
        f"best={float(history['valid_residual_huber'].min()):.6f}, "
        f"last={float(history.iloc[-1]['valid_residual_huber']):.6f}, "
        f"active_best={float(history['valid_active_huber'].min()):.6f}"
    )
    print(
        "correction_test: "
        f"mean={float(correction.mean()):.6f}, "
        f"max={float(correction.max()):.6f}, "
        f"p95={float(correction.quantile(0.95)):.6f}"
    )
    print(
        "ordinary_positive: "
        f"count={ref_count}, "
        f"mae_gain={float(ordinary.loc[REFERENCE, 'mae']) - float(ordinary.loc[CANDIDATE, 'mae']):+.6f}, "
        f"reference_bias={float(ordinary.loc[REFERENCE, 'signed_bias']):+.6f}, "
        f"candidate_bias={float(ordinary.loc[CANDIDATE, 'signed_bias']):+.6f}, "
        f"reference_cross_zero={float(ordinary.loc[REFERENCE, 'cross_zero_rate']):.4f}, "
        f"candidate_cross_zero={float(ordinary.loc[CANDIDATE, 'cross_zero_rate']):.4f}"
    )


if __name__ == "__main__":
    main()
