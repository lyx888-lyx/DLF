"""Engineering and scientific audit for V9.1 tail residual experts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ("shared_tail", "strong_negative", "strong_positive")
TAIL_REGIONS = ("strong_negative", "strong_positive")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(
            REPO_ROOT
            / "result"
            / "tail_residual_experts_v91"
            / "mosi"
            / "seed_1111"
        ),
    )
    parser.add_argument(
        "--require-tail-gain",
        type=float,
        default=0.0,
        help="Optional minimum validation gain required in each tail region.",
    )
    return parser.parse_args()


def _require(condition, message, errors):
    if not condition:
        errors.append(message)


def _finite(frame, columns, errors):
    for column in columns:
        _require(column in frame.columns, "missing column: %s" % column, errors)
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            _require(np.isfinite(values).all(), "non-finite values in %s" % column, errors)


def main() -> None:
    cli = parse_args()
    root = Path(cli.root)
    errors = []
    warnings = []

    summary_path = root / "tail_residual_experts_v91_summary.json"
    prediction_path = root / "tail_residual_experts_v91_predictions.csv"
    valid_matrix_path = root / "v91_valid_tail_capability_matrix.csv"
    test_matrix_path = root / "v91_test_tail_capability_matrix.csv"
    diagnostic_path = root / "v91_anchor_residual_diagnostics.csv"
    gate_path = root / "v91_anchor_gate_calibration.csv"
    for path in (
        summary_path,
        prediction_path,
        valid_matrix_path,
        test_matrix_path,
        diagnostic_path,
        gate_path,
    ):
        _require(path.is_file(), "missing output: %s" % path.name, errors)
    if errors:
        raise SystemExit("AUDIT FAILED\n- " + "\n- ".join(errors))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    valid_matrix = pd.read_csv(valid_matrix_path)
    test_matrix = pd.read_csv(test_matrix_path)
    diagnostics = pd.read_csv(diagnostic_path)
    gate = pd.read_csv(gate_path)

    _require(
        summary.get("method") == "bidirectional_tail_residual_experts_v9_1",
        "unexpected method",
        errors,
    )
    _require(isinstance(summary.get("anchor_index"), int), "missing anchor index", errors)
    _require(predictions["sample_id"].astype(str).is_unique, "duplicate sample ids", errors)
    _finite(
        predictions,
        ["label", "anchor", "anchor_score_gate", "true_region_tail_policy"],
        errors,
    )

    expected_experts = {"anchor", *CANDIDATES}
    _require(set(valid_matrix["expert"].astype(str)) == expected_experts, "invalid validation matrix experts", errors)
    _require(set(test_matrix["expert"].astype(str)) == expected_experts, "invalid test matrix experts", errors)
    _finite(
        valid_matrix,
        ["global_mae", "strong_negative_mae", "strong_positive_mae"],
        errors,
    )
    _finite(
        test_matrix,
        ["global_mae", "strong_negative_mae", "strong_positive_mae"],
        errors,
    )

    for candidate in CANDIDATES:
        checkpoint = root / candidate / "tail_residual_expert_v91_best.pth"
        history = root / candidate / "training_history.csv"
        _require(checkpoint.is_file(), "missing checkpoint for %s" % candidate, errors)
        _require(history.is_file(), "missing history for %s" % candidate, errors)
        _finite(
            predictions,
            [
                candidate + "_prediction",
                candidate + "_correction",
                candidate + "_applicability",
            ],
            errors,
        )
        if candidate in summary.get("candidates", {}):
            selected = summary["candidates"][candidate]["selected_valid"]
            _require(float(selected.get("max_anchor_diff", 1.0)) <= 5e-5, "%s anchor drift" % candidate, errors)
            if bool(summary["candidates"][candidate].get("anchor_fallback")):
                warnings.append("%s fell back to the immutable anchor" % candidate)

    _require(
        set(diagnostics["split"].astype(str)) >= {"train", "valid", "test"},
        "residual diagnostics are missing a split",
        errors,
    )
    _require(
        set(diagnostics["role"].astype(str)) >= set(CANDIDATES),
        "residual diagnostics are missing a tail role",
        errors,
    )
    _finite(gate, ["mae", "objective", "harm_over_010_rate"], errors)

    policy = summary.get("validation_selected_tail_policy", {})
    _require(set(policy) == set(TAIL_REGIONS), "tail policy is incomplete", errors)
    for region in TAIL_REGIONS:
        if region not in policy:
            continue
        expert = policy[region].get("expert")
        gain = float(policy[region].get("valid_gain", float("nan")))
        _require(expert in expected_experts, "invalid expert for %s" % region, errors)
        _require(np.isfinite(gain), "non-finite policy gain for %s" % region, errors)
        if gain < float(cli.require_tail_gain):
            errors.append(
                "%s validation gain %.6f < required %.6f"
                % (region, gain, float(cli.require_tail_gain))
            )

    if errors:
        raise SystemExit("AUDIT FAILED\n- " + "\n- ".join(errors))

    results = summary["test_results"]
    print("ENGINEERING AUDIT PASSED")
    print("anchor index:", summary["anchor_index"])
    for region in TAIL_REGIONS:
        entry = policy[region]
        print(
            "%s -> %s, validation gain=%.6f"
            % (region, entry["expert"], entry["valid_gain"])
        )
    print("test anchor MAE: %.6f" % results["anchor"]["MAE"])
    print(
        "test deployable anchor-score gate MAE: %.6f"
        % results["anchor_score_gate_valid_selected"]["MAE"]
    )
    print(
        "test true-region tail-policy MAE: %.6f"
        % results["true_region_valid_selected_tail_policy"]["MAE"]
    )
    for warning in warnings:
        print("WARNING: " + warning)


if __name__ == "__main__":
    main()
