#!/usr/bin/env python3
"""Audit V9.1 outputs before interpreting Test performance."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REFERENCE = "original_v71_hybrid"
CANDIDATE = "decoupled_region_mixture_v91"
ORDINARY_POSITIVE = "ordinary_positive"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "result/decoupled_region_mixture_v91/mosi/seed_1111"
        ),
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


def max_abs_difference(left, right):
    return float((pd.to_numeric(left) - pd.to_numeric(right)).abs().max())


def main():
    cli = parse_args()
    run_dir = cli.run_dir
    summary = json.loads(
        require_file(
            run_dir / "decoupled_region_mixture_v91_summary.json"
        ).read_text(encoding="utf-8")
    )
    comparison = pd.read_csv(
        require_file(run_dir / "v91_test_comparison.csv")
    )
    regions = pd.read_csv(
        require_file(run_dir / "v91_test_region_diagnostics.csv")
    )
    predictions = pd.read_csv(
        require_file(run_dir / "v91_test_predictions.csv")
    )
    weights = pd.read_csv(
        require_file(run_dir / "v91_train_region_weights.csv")
    )
    history = pd.read_csv(
        require_file(run_dir / "v91_training_history.csv")
    )

    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate sample_id in V9.1 Test predictions")

    numeric_columns = (
        "label",
        "original_v71_hybrid",
        "v91_selected",
        "selected_beta_only",
        "selected_zero_shrinkage",
        "selected_trained_mixture",
        "legacy_student",
        "direct_trained_mixture",
        "gate_negative",
        "gate_neutral",
        "gate_positive",
    )
    for column in numeric_columns:
        finite_series(predictions, column)

    gate_sum = (
        predictions["gate_negative"]
        + predictions["gate_neutral"]
        + predictions["gate_positive"]
    )
    if float((gate_sum - 1.0).abs().max()) > 1e-5:
        raise RuntimeError("Polarity probabilities do not sum to one")

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

    selected = summary["selected_policy"]
    selected_epoch = int(summary["selected_epoch"])
    source = str(selected["source"])
    gamma = float(selected["gamma"])
    training_contributed = bool(summary["training_contributed"])

    if source == "trained_mixture":
        if selected_epoch <= 0 or gamma <= 0.0:
            raise RuntimeError(
                "trained_mixture was selected without a trained epoch "
                "and positive gamma"
            )
        if not training_contributed:
            raise RuntimeError(
                "Summary failed to mark a selected trained mixture"
            )
    elif training_contributed:
        raise RuntimeError(
            "training_contributed=True for a non-trained policy"
        )

    expected_column = {
        "legacy_reference": "original_v71_hybrid",
        "legacy_beta_search": "selected_beta_only",
        "zero_shrinkage": "selected_zero_shrinkage",
        "trained_mixture": "selected_trained_mixture",
    }.get(source)
    if expected_column is None:
        raise RuntimeError(f"Unknown selected source: {source}")
    prediction_difference = max_abs_difference(
        predictions["v91_selected"], predictions[expected_column]
    )
    if prediction_difference > cli.prediction_tolerance:
        raise RuntimeError(
            "Saved selected prediction does not match its declared family: "
            f"source={source}, max_abs_difference={prediction_difference}"
        )

    if history.empty:
        raise RuntimeError("V9.1 training history is empty")
    for column in (
        "direct_mixture_mae",
        "selected_mae",
        "selected_beta",
        "selected_gamma",
    ):
        finite_series(history, column)

    print("AUDIT PASSED")
    print(f"selected_epoch={selected_epoch}")
    print(
        "selected_policy="
        f"{source} committee={selected['committee']} "
        f"beta={selected['beta']} gamma={selected['gamma']}"
    )
    print(f"training_contributed={training_contributed}")
    print(f"reference_mae={reference_mae:.6f}")
    print(f"candidate_mae={candidate_mae:.6f}")
    print(f"overall_mae_gain={reference_mae - candidate_mae:+.6f}")
    print(
        "direct_mixture_valid: "
        f"first={float(history.iloc[0]['direct_mixture_mae']):.6f}, "
        f"best={float(history['direct_mixture_mae'].min()):.6f}, "
        f"last={float(history.iloc[-1]['direct_mixture_mae']):.6f}"
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
