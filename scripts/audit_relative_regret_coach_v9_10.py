"""Engineering and result audit for V9.10 relative-regret coaching."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/relative_regret_coach_v910/mosi/seed_1111",
    )
    return parser.parse_args()


def require(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    cli = parse_args()
    root = Path(cli.root)
    summary_path = require(root / "relative_regret_coach_v910_summary.json")
    oof_path = require(root / "v910_relative_regret_oof_predictions.csv")
    policy_path = require(root / "v910_oof_policy_profiles.csv")
    valid_path = require(root / "v910_validation_profile_selection.csv")
    test_path = require(root / "v910_test_summary.csv")
    prediction_path = require(root / "relative_regret_coach_v910_predictions.csv")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("regret_version") != "relative_regret_v1":
        raise AssertionError("unexpected V9.10 regret version")
    semantic = summary.get("semantic_pool", {})
    provenance = semantic.get("provenance", {}) or {}
    if provenance.get("is_fully_nested_teacher_stack") is not True:
        raise AssertionError("semantic OOF teacher stack is not fully nested")
    if provenance.get("historical_full_train_teacher_reuse") is not False:
        raise AssertionError("historical full-Train teacher reuse detected")
    if provenance.get("holdout_label_isolation") is not True:
        raise AssertionError("holdout-label isolation is not asserted")

    coach = summary.get("relative_regret_coach", {})
    checkpoints = [Path(value) for value in coach.get("ensemble_checkpoints", [])]
    if len(checkpoints) != 5:
        raise AssertionError(f"expected five ensemble checkpoints, got {len(checkpoints)}")
    for checkpoint in checkpoints:
        require(checkpoint)

    oof = pd.read_csv(oof_path)
    if len(oof) != 1284:
        raise AssertionError(f"unexpected OOF sample count: {len(oof)}")
    if oof["sample_id"].duplicated().any():
        raise AssertionError("duplicate OOF sample ids")
    numeric = oof.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise FloatingPointError("non-finite V9.10 OOF diagnostic")

    policies = pd.read_csv(policy_path)
    validation = pd.read_csv(valid_path)
    pd.read_csv(test_path)
    predictions = pd.read_csv(prediction_path)
    if len(predictions) != 686:
        raise AssertionError(f"unexpected Test sample count: {len(predictions)}")
    if "anchor" not in validation["profile"].astype(str).tolist():
        raise AssertionError("Validation selection lacks Anchor fallback")

    results = summary.get("test_results", {})
    anchor_mae = float(results["anchor"]["MAE"])
    deploy_mae = float(results["relative_regret_valid_selected"]["MAE"])
    hard_mae = float(results["minimum_predicted_regret_action_all"]["MAE"])
    risk_mae = float(results["risk_adjusted_regret_action_all"]["MAE"])
    oracle_mae = float(results["sample_oracle_upper_bound"]["MAE"])

    print("ENGINEERING AUDIT PASSED")
    print(f"relative-regret OOF samples: {len(oof)}")
    print("teacher stack fully nested:", provenance.get("is_fully_nested_teacher_stack"))
    print("historical full-Train teacher reuse:", provenance.get("historical_full_train_teacher_reuse"))
    print("holdout-label isolation:", provenance.get("holdout_label_isolation"))
    print("ensemble checkpoints:", len(checkpoints))
    print("crossfit best epochs:", coach.get("crossfit_best_epochs"))
    print("OOF delta MAE:", coach.get("oof_per_expert_delta_mae"))
    print("OOF delta Spearman:", coach.get("oof_per_expert_delta_spearman"))
    print("OOF Beat-Anchor accuracy:", coach.get("oof_per_expert_beat_accuracy"))
    print("OOF selected-action accuracy:", coach.get("oof_selected_action_accuracy"))
    print("OOF selected MAE:", coach.get("oof_regret_selected_mae"))
    print("OOF unrestricted gain:", coach.get("oof_unrestricted_gain"))
    print("OOF benefit rate:", coach.get("oof_selected_benefit_rate"))
    print("OOF severe-harm rate:", coach.get("oof_selected_harm_over_010_rate"))
    for row in policies.to_dict("records"):
        print(
            "  profile=%s gain=%+.6f lower=%+.6f harm=%.4f activation=%.4f eligible=%s"
            % (
                row["profile"],
                row["oof_gain"],
                row["bootstrap_gain_lower"],
                row["harm_over_010_rate"],
                row["activation_rate"],
                row["oof_eligible"],
            )
        )
    print("validation selected profile:", summary.get("selected_profile"))
    print("test activation rate:", summary.get("test_activation_rate"))
    print("test proposed actions:", summary.get("test_proposed_action_counts"))
    print("test deployed actions:", summary.get("test_deployed_action_counts"))
    print(f"test anchor MAE: {anchor_mae:.6f}")
    print(f"test hard relative-regret MAE: {hard_mae:.6f}")
    print(f"test risk-adjusted relative-regret MAE: {risk_mae:.6f}")
    print(f"test deployable relative-regret MAE: {deploy_mae:.6f}")
    print(f"test deployable gain: {anchor_mae - deploy_mae:+.6f}")
    print(f"test sample-oracle MAE: {oracle_mae:.6f}")
    if deploy_mae >= anchor_mae:
        print("WARNING: V9.10 deployable coach did not beat Anchor on Test")


if __name__ == "__main__":
    main()
