"""Engineering audit for completed V9.27 artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trains.singleTask.paired_perturbation_relative_stability_v927 import (
    AUDIT_VERSION,
    BASE_METHOD,
    PRIMARY_METHOD,
    STABILITY_ONLY_METHOD,
)


EXPECTED_FILES = (
    "v927_report.md",
    "v927_summary.json",
    "v927_binary_metrics.csv",
    "v927_continuous_metrics.csv",
    "v927_ranked_subset_metrics.csv",
    "v927_outer_fold_safe_gain.csv",
    "v927_inner_risk_predictions.csv",
    "v927_outer_risk_predictions.csv",
    "v927_risk_split_manifest.csv",
    "v927_source_integrity.csv",
    "v927_model_inventory.csv",
    "v927_feature_schema.csv",
    "v927_identity_reconstruction.csv",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main():
    cli = parse_args()
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v927_paired_perturbation_audit"
    )
    missing = [name for name in EXPECTED_FILES if not (output / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing V9.27 artifacts: {missing}")

    summary = json.loads(
        (output / "v927_summary.json").read_text(encoding="utf-8")
    )
    if summary.get("version") != AUDIT_VERSION:
        raise RuntimeError("unexpected V9.27 version")
    provenance = summary.get("provenance", {})
    required_false = (
        "base_experts_trained",
        "base_experts_or_predictions_modified",
        "v921_weights_refit_on_outer_labels",
        "outer_labels_used_for_fit_or_calibration",
        "raw_waveform_or_pixels_perturbed",
        "text_perturbed",
        "router_or_action_selection_executed",
        "prediction_replacement_or_mixing_executed",
        "official_validation_or_test_used",
    )
    for key in required_false:
        if provenance.get(key) is not False:
            raise RuntimeError(f"provenance must be false: {key}")
    required_true = (
        "risk_heads_trained",
        "inner_oof_only",
        "disjoint_calibration_fold",
        "pre_extracted_audio_vision_features_perturbed",
        "same_perturbation_for_experts_and_v921",
    )
    for key in required_true:
        if provenance.get(key) is not True:
            raise RuntimeError(f"provenance must be true: {key}")

    perturbations = summary.get("perturbations", [])
    names = [row.get("name") for row in perturbations]
    expected_names = [
        "identity",
        "audio_gain_095",
        "audio_gain_105",
        "audio_noise_a",
        "audio_noise_b",
        "audio_mask_025",
        "audio_mask_065",
        "vision_gain_095",
        "vision_gain_105",
        "vision_noise_a",
        "vision_noise_b",
        "vision_mask_025",
        "vision_mask_065",
    ]
    if names != expected_names:
        raise RuntimeError(f"perturbation registry changed: {names}")

    integrity = pd.read_csv(output / "v927_source_integrity.csv")
    expected_stacks = int(cli.outer_folds) * 4
    if len(integrity) != expected_stacks:
        raise RuntimeError(
            f"expected {expected_stacks} stack caches, found {len(integrity)}"
        )
    if integrity["cache_path"].duplicated().any():
        raise RuntimeError("duplicate perturbation cache path")
    for column in (
        "identity_action_max_abs",
        "identity_confidence_max_abs",
        "identity_correction_max_abs",
    ):
        if not (integrity[column] <= 5e-4 + 1e-12).all():
            raise RuntimeError(f"identity reconstruction failed: {column}")
    for path in integrity["pool_path"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    for path in integrity["cache_path"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    reconstruction = pd.read_csv(
        output / "v927_identity_reconstruction.csv"
    )
    if len(reconstruction) != cli.outer_folds:
        raise RuntimeError("identity reconstruction fold count mismatch")
    if not (
        reconstruction[
            [
                "inner_v921_identity_max_abs",
                "outer_v921_identity_max_abs",
            ]
        ]
        <= 5e-4 + 1e-12
    ).all().all():
        raise RuntimeError("V9.21 identity reconstruction failed")

    inventory = pd.read_csv(output / "v927_model_inventory.csv")
    expected_models = int(cli.outer_folds) * 4 * 3
    if len(inventory) != expected_models:
        raise RuntimeError(
            f"expected {expected_models} risk models, found {len(inventory)}"
        )
    if set(inventory["method"]) != {
        BASE_METHOD,
        STABILITY_ONLY_METHOD,
        PRIMARY_METHOD,
    }:
        raise RuntimeError("risk model methods changed")
    expected_counts = {
        BASE_METHOD: 35,
        STABILITY_ONLY_METHOD: 8,
        PRIMARY_METHOD: 43,
    }
    for method, count in expected_counts.items():
        local = inventory[inventory["method"] == method]
        if set(local["feature_count"].astype(int)) != {count}:
            raise RuntimeError(
                f"{method} feature count changed: "
                f"{sorted(set(local['feature_count']))}"
            )
    for path in inventory["model_path"]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    schema = pd.read_csv(output / "v927_feature_schema.csv")
    stability_names = {
        "paired_expert_std",
        "paired_baseline_std",
        "paired_log_std_ratio",
        "paired_relative_gap_std",
        "paired_relative_direction_flip_rate",
        "paired_applicability_flip_rate",
        "paired_expert_max_change",
        "paired_baseline_max_change",
    }
    stability_schema = schema[
        schema["method"] == STABILITY_ONLY_METHOD
    ]
    if set(stability_schema["feature_name"]) != stability_names:
        raise RuntimeError("paired-stability feature schema changed")

    binary = pd.read_csv(output / "v927_binary_metrics.csv")
    continuous = pd.read_csv(output / "v927_continuous_metrics.csv")
    ranked = pd.read_csv(output / "v927_ranked_subset_metrics.csv")
    for method in (BASE_METHOD, STABILITY_ONLY_METHOD, PRIMARY_METHOD):
        for target in (
            "win_vs_baseline",
            "large_gain_010",
            "large_harm_030",
        ):
            local = binary[
                (binary["method"] == method)
                & (binary["expert"] == "all")
                & (binary["target"] == target)
            ]
            if len(local) != 1:
                raise RuntimeError(
                    f"missing aggregate binary row: {method}/{target}"
                )
        local = continuous[
            (continuous["method"] == method)
            & (continuous["expert"] == "all")
            & (continuous["target"] == "gain_vs_baseline")
        ]
        if len(local) != 1:
            raise RuntimeError(
                f"missing aggregate gain row: {method}"
            )
        local = ranked[
            (ranked["method"] == method)
            & (ranked["outer_fold"].astype(str) == "aggregate")
            & (ranked["selector"] == "selected_risk_top")
        ]
        if len(local) != 1:
            raise RuntimeError(
                f"missing aggregate ranked row: {method}"
            )

    print("V9.27 PAIRED PERTURBATION ENGINEERING AUDIT PASSED")
    print("base experts trained or modified: False")
    print("raw waveform or pixels perturbed: False")
    print("pre-extracted model inputs perturbed: True")
    print("router or action selection executed: False")
    print(
        "risk signal supported:",
        bool(summary.get("risk_signal_supported")),
    )


if __name__ == "__main__":
    main()
