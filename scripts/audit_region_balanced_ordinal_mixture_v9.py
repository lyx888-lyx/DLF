#!/usr/bin/env python3
"""Audit V9 outputs before interpreting any Test result."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


REFERENCE = "original_v71_hybrid"
CANDIDATE = "region_balanced_ordinal_mixture_v9"
ORDINARY_POSITIVE = "ordinary_positive"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("result/region_balanced_ordinal_mixture_v9/mosi/seed_1111"),
    )
    parser.add_argument("--expected-reference-mae", type=float, default=None)
    parser.add_argument("--reference-tolerance", type=float, default=1e-4)
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


def main():
    cli = parse_args()
    run_dir = cli.run_dir
    summary = json.loads(require_file(
        run_dir / "region_balanced_ordinal_mixture_v9_summary.json"
    ).read_text(encoding="utf-8"))
    comparison = pd.read_csv(require_file(run_dir / "v9_test_comparison.csv"))
    regions = pd.read_csv(require_file(run_dir / "v9_test_region_diagnostics.csv"))
    predictions = pd.read_csv(require_file(run_dir / "v9_test_predictions.csv"))
    weights = pd.read_csv(require_file(run_dir / "v9_train_region_weights.csv"))

    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("Duplicate sample_id in V9 Test predictions")
    for column in (
        "label",
        "original_v71_hybrid",
        "v9_selected",
        "raw_v9_prediction",
        "mixture_value",
        "gate_negative",
        "gate_neutral",
        "gate_positive",
    ):
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
                f"observed={reference_mae:.8f}, expected={cli.expected_reference_mae:.8f}, "
                f"difference={difference:.8f}"
            )

    ordinary = regions[regions["region"] == ORDINARY_POSITIVE].set_index("model")
    for model in (REFERENCE, CANDIDATE):
        if model not in ordinary.index:
            raise RuntimeError(f"Missing ordinary-positive diagnostics: {model}")
    ref_count = int(ordinary.loc[REFERENCE, "count"])
    cand_count = int(ordinary.loc[CANDIDATE, "count"])
    if ref_count != cand_count or ref_count <= 0:
        raise RuntimeError(
            f"Ordinary-positive count mismatch: reference={ref_count}, candidate={cand_count}"
        )

    if len(weights) != 5 or (weights["count"] <= 0).any():
        raise RuntimeError("Invalid Train region counts")
    weighted_mean = float(
        (weights["count"] * weights["weight"]).sum() / weights["count"].sum()
    )
    if abs(weighted_mean - 1.0) > 1e-4:
        raise RuntimeError(f"Region weights are not sample-normalized: {weighted_mean}")

    selected = summary["selected_policy"]
    print("AUDIT PASSED")
    print(f"selected_epoch={summary['selected_epoch']}")
    print(
        "selected_policy="
        f"{selected['source']} committee={selected['committee']} "
        f"beta={selected['beta']} alpha={selected['alpha']}"
    )
    print(f"reference_mae={reference_mae:.6f}")
    print(f"candidate_mae={candidate_mae:.6f}")
    print(f"overall_mae_gain={reference_mae - candidate_mae:+.6f}")
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
