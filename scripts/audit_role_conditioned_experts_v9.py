"""Engineering and scientific audit for V9 role-conditioned experts."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(
            REPO_ROOT
            / "result"
            / "role_conditioned_experts_v9"
            / "mosi"
            / "seed_1111"
        ),
    )
    parser.add_argument(
        "--require-designated-wins",
        type=int,
        default=0,
        help="Optional scientific gate; engineering checks are always enforced.",
    )
    parser.add_argument(
        "--max-global-delta",
        type=float,
        default=0.02,
        help="Diagnostic threshold for each specialist's validation degradation.",
    )
    return parser.parse_args()


def _require(condition, message, errors):
    if not condition:
        errors.append(message)


def _finite_frame(frame, columns, errors):
    for column in columns:
        _require(column in frame.columns, "missing column: %s" % column, errors)
        if column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").to_numpy()
            _require(np.isfinite(values).all(), "non-finite values in %s" % column, errors)


def main():
    cli = parse_args()
    root = Path(cli.root)
    errors = []
    warnings = []

    summary_path = root / "role_conditioned_experts_v9_summary.json"
    prediction_path = root / "role_conditioned_experts_v9_predictions.csv"
    valid_matrix_path = root / "v9_valid_capability_matrix.csv"
    test_matrix_path = root / "v9_test_capability_matrix.csv"
    _require(summary_path.is_file(), "missing V9 summary", errors)
    _require(prediction_path.is_file(), "missing V9 predictions", errors)
    _require(valid_matrix_path.is_file(), "missing validation capability matrix", errors)
    _require(test_matrix_path.is_file(), "missing test capability matrix", errors)
    if errors:
        raise SystemExit("AUDIT FAILED\n- " + "\n- ".join(errors))

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(prediction_path)
    valid_matrix = pd.read_csv(valid_matrix_path)
    test_matrix = pd.read_csv(test_matrix_path)

    _require(summary.get("method") == "role_conditioned_complementary_distillation_v9", "unexpected method", errors)
    _require(len(summary.get("roles", {})) == 5, "summary must contain five roles", errors)
    _require(predictions["sample_id"].astype(str).is_unique, "duplicate sample ids", errors)
    _finite_frame(predictions, ["label", "anchor", "category_coach"], errors)

    teacher_fit = summary.get("teacher_fit", {})
    global_weights = np.asarray(teacher_fit.get("global_weights", []), dtype=float)
    role_weights = np.asarray(teacher_fit.get("role_weights", []), dtype=float)
    _require(global_weights.ndim == 1 and global_weights.size >= 3, "invalid global teacher weights", errors)
    _require(role_weights.shape == (5, global_weights.size), "invalid role teacher weight matrix", errors)
    if global_weights.size:
        _require(np.all(global_weights >= -1e-8), "negative global teacher weight", errors)
        _require(abs(global_weights.sum() - 1.0) < 1e-5, "global weights do not sum to one", errors)
    if role_weights.size:
        _require(np.all(role_weights >= -1e-8), "negative role teacher weight", errors)
        _require(np.allclose(role_weights.sum(axis=1), 1.0, atol=1e-5), "role weights do not sum to one", errors)

    for role in REGION_NAMES:
        checkpoint = root / role / "role_conditioned_expert_v9_best.pth"
        history = root / role / "training_history.csv"
        _require(checkpoint.is_file(), "missing checkpoint for %s" % role, errors)
        _require(history.is_file(), "missing history for %s" % role, errors)
        probability_columns = [role + "_p_" + category for category in REGION_NAMES]
        _finite_frame(predictions, probability_columns, errors)
        if all(column in predictions.columns for column in probability_columns):
            probability_sum = predictions[probability_columns].sum(axis=1).to_numpy()
            _require(np.allclose(probability_sum, 1.0, atol=1e-4), "%s probabilities do not sum to one" % role, errors)
        _finite_frame(
            predictions,
            [role + "_prediction", role + "_predicted_risk", role + "_coach_weight"],
            errors,
        )
        if role in summary.get("roles", {}):
            selected = summary["roles"][role].get("selected_valid", {})
            global_delta = float(selected.get("global_delta", math.inf))
            role_gain = float(selected.get("role_gain", -math.inf))
            if global_delta > cli.max_global_delta:
                warnings.append(
                    "%s validation global delta %.6f exceeds %.6f"
                    % (role, global_delta, cli.max_global_delta)
                )
            if role_gain <= 0:
                warnings.append(
                    "%s did not improve over the anchor in its designated validation region (gain=%.6f)"
                    % (role, role_gain)
                )

    coach_weight_columns = [role + "_coach_weight" for role in REGION_NAMES]
    if all(column in predictions.columns for column in coach_weight_columns):
        coach_sum = predictions[coach_weight_columns].sum(axis=1).to_numpy()
        _require(np.allclose(coach_sum, 1.0, atol=1e-4), "coach weights do not sum to one", errors)

    expected_rows = {"anchor", *REGION_NAMES}
    _require(set(valid_matrix["expert"].astype(str)) == expected_rows, "validation matrix rows are incomplete", errors)
    _require(set(test_matrix["expert"].astype(str)) == expected_rows, "test matrix rows are incomplete", errors)

    wins = summary.get("valid_specialization_wins", {})
    designated_wins = sum(
        int(bool(value.get("designated_is_best"))) for value in wins.values()
    )
    _require(len(wins) == 5, "specialization win table is incomplete", errors)
    if designated_wins < int(cli.require_designated_wins):
        errors.append(
            "designated validation wins %d < required %d"
            % (designated_wins, cli.require_designated_wins)
        )

    if errors:
        raise SystemExit("AUDIT FAILED\n- " + "\n- ".join(errors))

    print("ENGINEERING AUDIT PASSED")
    print("designated validation wins: %d/5" % designated_wins)
    coach_mae = summary["test_results"]["category_coach_valid_selected"]["MAE"]
    anchor_mae = summary["test_results"]["anchor"]["MAE"]
    oracle_mae = summary["test_results"]["true_region_designated_oracle"]["MAE"]
    print("test anchor MAE: %.6f" % anchor_mae)
    print("test category-coach MAE: %.6f" % coach_mae)
    print("test true-region designated oracle MAE: %.6f" % oracle_mae)
    if designated_wins >= 3 and oracle_mae < anchor_mae:
        print("SPECIALIZATION SIGNAL: PRESENT")
    else:
        print("SPECIALIZATION SIGNAL: NOT YET ESTABLISHED")
    for warning in warnings:
        print("WARNING: " + warning)


if __name__ == "__main__":
    main()
