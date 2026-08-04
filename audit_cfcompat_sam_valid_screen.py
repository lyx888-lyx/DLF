"""Independent artifact audit for the CFCompatKD + SAM Valid-only screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_sam_utils import (
    FORMAL_SEEDS,
    METHOD,
    SAM_RHOS,
    VERSION,
    aggregate_valid_gate,
    replay_gate,
    select_rho,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES


MODES = ("LAV",) + MISSING_MODES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def require_close(left, right, name, tolerance=1e-7):
    if abs(float(left) - float(right)) > tolerance:
        raise RuntimeError(
            f"{name} mismatch: {left} vs {right}"
        )


def metrics_from_predictions(frame):
    labels = frame["label"].to_numpy(dtype=float)
    return {
        mode: float(
            np.mean(
                np.abs(
                    frame[f"{mode}_pred"].to_numpy(dtype=float)
                    - labels
                )
            )
        )
        for mode in MODES
    }


def nested_equal(actual, expected, tolerance=1e-10):
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            nested_equal(actual[key], expected[key], tolerance)
            for key in expected
        )
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, np.integer)):
        return int(actual) == int(expected)
    if isinstance(expected, (float, np.floating)):
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    required = {
        "summary": root / "sam_valid_screen_summary.json",
        "source": root / "sam_source_manifest.json",
        "grid": root / "sam_valid_grid_summary.csv",
        "epochs": root / "sam_all_epoch_metrics.csv",
        "predictions": root / "sam_all_valid_predictions.csv",
        "report": root / "sam_valid_screen_report.md",
    }
    missing = [
        str(path) for path in required.values() if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing SAM artifacts:\n" + "\n".join(missing)
        )

    summary = json.loads(
        required["summary"].read_text(encoding="utf-8")
    )
    source = json.loads(
        required["source"].read_text(encoding="utf-8")
    )
    grid = pd.read_csv(required["grid"])
    epochs = pd.read_csv(required["epochs"])
    predictions = pd.read_csv(required["predictions"])
    checks = {}

    checks["version_and_method"] = bool(
        summary["version"] == VERSION
        and summary["method"] == METHOD
        and source["version"] == VERSION
        and source["method"] == METHOD
    )
    checks["frozen_seed_and_rho_grid"] = bool(
        tuple(summary["protocol"]["formal_seeds"])
        == FORMAL_SEEDS
        and tuple(
            float(value)
            for value in summary["protocol"]["rho_grid"]
        )
        == SAM_RHOS
        and tuple(source["formal_seeds"]) == FORMAL_SEEDS
        and tuple(float(value) for value in source["rho_grid"])
        == SAM_RHOS
    )

    expected_runs = {
        (seed, 0.0) for seed in FORMAL_SEEDS
    } | {
        (seed, rho)
        for seed in FORMAL_SEEDS
        for rho in SAM_RHOS
    }
    observed_runs = {
        (int(row.Seed), float(row.Rho))
        for row in grid.itertuples(index=False)
    }
    checks["complete_grid"] = bool(
        observed_runs == expected_runs
        and len(grid) == len(expected_runs)
    )

    hash_ok = True
    for row in grid.to_dict("records"):
        for path_key, hash_key in (
            ("MainCheckpoint", "MainCheckpointSHA256"),
            ("TeacherCheckpoint", "TeacherSHA256"),
            ("EvaluatorCheckpoint", "EvaluatorSHA256"),
            ("CompatibilityCache", "CompatibilityCacheSHA256"),
        ):
            path = Path(str(row[path_key]))
            hash_ok = (
                hash_ok
                and path.is_file()
                and checkpoint_sha256(path) == row[hash_key]
            )
        if str(row["Optimizer"]) == "Adam":
            baseline_path = Path(str(row["BaselineResult"]))
            hash_ok = (
                hash_ok
                and baseline_path.is_file()
                and checkpoint_sha256(baseline_path)
                == row["BaselineResultSHA256"]
            )
    checks["all_asset_hashes"] = bool(hash_ok)

    source_hash_ok = True
    for item in source["checkpoints"]:
        path = Path(item["path"])
        source_hash_ok = (
            source_hash_ok
            and path.is_file()
            and checkpoint_sha256(path) == item["sha256"]
        )
    checks["source_checkpoint_hashes"] = bool(source_hash_ok)

    recomputed_metrics = True
    for row in grid.to_dict("records"):
        local = predictions.loc[
            predictions.Seed.astype(int).eq(int(row["Seed"]))
            & predictions.Run.astype(str).eq(str(row["Run"]))
        ]
        if local.empty:
            recomputed_metrics = False
            break
        mae = metrics_from_predictions(local)
        for mode in MODES:
            require_close(
                mae[mode],
                row[f"valid_{mode}_MAE"],
                f"seed{row['Seed']} {row['Run']} {mode}",
            )
        j_value = 0.5 * mae["LAV"] + 0.5 * np.mean(
            [mae[mode] for mode in MISSING_MODES]
        )
        require_close(
            j_value,
            row["J_valid"],
            f"seed{row['Seed']} {row['Run']} J",
        )
    checks["valid_metrics_recomputed"] = recomputed_metrics

    replay_ok = True
    baseline_rows = {}
    for seed in FORMAL_SEEDS:
        baseline = summary["baselines"][str(seed)]
        replay = next(
            row
            for row in summary["baseline_replays"]
            if int(row["Seed"]) == seed
        )
        computed = replay_gate(replay, baseline)
        replay_ok = (
            replay_ok
            and nested_equal(
                computed,
                summary["baseline_replay_gates"][str(seed)],
            )
            and computed["passed"]
        )
        baseline_rows[seed] = baseline
    checks["baseline_replays_recomputed"] = bool(replay_ok)

    candidate_rows = [row for row in summary["candidates"]]
    selected_rho = select_rho(candidate_rows)
    checks["selected_by_mean_valid_only"] = math.isclose(
        float(summary["selected_rho"]),
        float(selected_rho),
        abs_tol=0.0,
    )
    recomputed_gate = aggregate_valid_gate(
        selected_rho,
        candidate_rows,
        baseline_rows,
        epochs.to_dict("records"),
    )
    checks["dual_seed_valid_gate_recomputed"] = nested_equal(
        summary["valid_gate"], recomputed_gate
    )

    sam_epochs = epochs.loc[
        epochs.Optimizer.astype(str).eq("SAM(Adam)")
    ]
    adam_epochs = epochs.loc[
        epochs.Optimizer.astype(str).eq("Adam")
    ]
    checks["sam_gradient_contract"] = bool(
        not sam_epochs.empty
        and (sam_epochs.sam_first_grad_norm.astype(float) > 0).all()
        and (sam_epochs.sam_second_grad_norm.astype(float) > 0).all()
        and np.isfinite(
            sam_epochs.sam_first_grad_norm.astype(float)
        ).all()
        and np.isfinite(
            sam_epochs.sam_second_grad_norm.astype(float)
        ).all()
        and (
            adam_epochs.sam_first_grad_norm.astype(float) == 0
        ).all()
        and (
            adam_epochs.sam_second_grad_norm.astype(float) == 0
        ).all()
    )
    checks["sam_two_pass_rng_and_window_contract"] = bool(
        summary["protocol"]["update_epochs"] == 10
        and summary["protocol"][
            "same_microbatch_window_two_pass"
        ]
        and summary["protocol"]["same_missing_masks_two_pass"]
        and summary["protocol"]["same_dropout_rng_two_pass"]
        and summary["protocol"]["base_optimizer"] == "Adam"
    )
    checks["original_objective_preserved"] = bool(
        summary["protocol"][
            "original_cfcompat_objective_unchanged"
        ]
    )
    checks["no_inference_change"] = bool(
        summary["protocol"]["additional_inference_parameters"]
        == 0
        and source["additional_inference_parameters"] == 0
    )

    known_test_files = [
        root / "sam_test_predictions.csv",
        root / "selected_test_predictions.csv",
        root / "test_predictions.csv",
    ]
    allowed_test_metadata = {"TestConstructed"}
    test_columns = [
        column
        for column in (
            list(grid.columns)
            + list(epochs.columns)
            + list(predictions.columns)
        )
        if "test" in str(column).lower()
        and str(column) not in allowed_test_metadata
    ]
    test_constructed_values = grid.TestConstructed.map(
        lambda value: str(value).strip().lower() == "true"
    )
    checks["official_test_forbidden"] = bool(
        not summary["protocol"]["official_test_constructed"]
        and not summary["protocol"]["official_test_authorized"]
        and not source["test_constructed"]
        and source["test_loader_construction_count"] == 0
        and source["test_loader_traversal_count"] == 0
        and not any(path.exists() for path in known_test_files)
        and not test_columns
        and not test_constructed_values.any()
    )
    checks["next_stage_is_group_screen_only"] = bool(
        summary["valid_gate"]["next_required_stage"]
        in ("train_only_group_stability_screen", "stop_sam")
        and summary["protocol"]["next_stage_on_pass"]
        == "train_only_group_stability_screen"
    )
    expected_verdict = (
        "PROMOTE_SAM_TO_TRAIN_ONLY_GROUP_SCREEN"
        if recomputed_gate["passed"]
        else "STOP_SAM_DUAL_SEED_VALID_FAILED"
    )
    checks["verdict_recomputed"] = (
        summary["verdict"] == expected_verdict
    )

    passed = bool(all(checks.values()))
    payload = {
        "passed": passed,
        "verdict": summary["verdict"],
        "checks": checks,
    }
    (
        root / "sam_valid_screen_audit_check.json"
    ).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError(
            "CFCompatKD + SAM independent audit failed."
        )
    print("CFCompatKD + SAM independent audit passed")
    for key, value in checks.items():
        print(f"{key}: {value}")
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
