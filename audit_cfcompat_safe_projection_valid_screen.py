"""Independent audit for the Safe-CFCompatKD held-out-seed Valid screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_safe_projection_utils import (
    FORMAL_SEEDS,
    METHOD,
    RUNS,
    VERSION,
    aggregate_candidate_gate,
    derive_valid_events,
    group_summary,
    jsonable,
    overall_from_events,
    replay_gate,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def same_frame(actual, expected, keys, tolerance=1e-9):
    left = actual.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(
                left[column].to_numpy(dtype=float),
                right[column].to_numpy(dtype=float),
                atol=tolerance,
                rtol=0.0,
                equal_nan=True,
            ):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def nested_close(actual, expected, tolerance=1e-9):
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            nested_close(actual[key], expected[key], tolerance) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            nested_close(left, right, tolerance)
            for left, right in zip(actual, expected)
        )
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int):
        return int(actual) == expected
    if isinstance(expected, float):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return abs(float(actual) - expected) <= tolerance
    return actual == expected


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    files = {
        "grid": root / "safe_projection_valid_grid_summary.csv",
        "epochs": root / "safe_projection_all_epoch_metrics.csv",
        "raw_events": root / "safe_projection_raw_valid_events.csv",
        "events": root / "safe_projection_valid_events.csv",
        "overall": root / "safe_projection_overall_metrics.csv",
        "groups": root / "safe_projection_group_metrics.csv",
        "references": root / "safe_projection_stage3_references.csv",
        "manifest": root / "safe_projection_source_manifest.json",
        "summary": root / "safe_projection_valid_screen_summary.json",
        "report": root / "safe_projection_valid_screen_report.md",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing Safe-CFCompat artifacts:\n" + "\n".join(missing))

    grid = pd.read_csv(files["grid"])
    epochs = pd.read_csv(files["epochs"])
    raw_events = pd.read_csv(files["raw_events"])
    recorded_events = pd.read_csv(files["events"])
    recorded_overall = pd.read_csv(files["overall"])
    recorded_groups = pd.read_csv(files["groups"])
    references = pd.read_csv(files["references"])
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))

    checks = {}
    checks["version_and_method"] = bool(
        manifest["version"] == VERSION
        and manifest["method"] == METHOD
        and summary["version"] == VERSION
        and summary["method"] == METHOD
    )
    checks["frozen_protocol"] = bool(
        tuple(manifest["formal_seeds"]) == FORMAL_SEEDS
        and tuple(manifest["runs"]) == RUNS
        and manifest["decision_split"] == "official_valid_only"
        and not manifest["official_test_constructed"]
        and manifest["test_loader_construction_count"] == 0
        and manifest["test_loader_traversal_count"] == 0
        and summary["protocol"]["formal_seeds"] == list(FORMAL_SEEDS)
        and summary["protocol"]["runs"] == list(RUNS)
        and not summary["protocol"]["official_test_constructed"]
        and not summary["protocol"]["official_test_authorized"]
        and summary["protocol"]["lambda_kd"] == 1.0
        and summary["protocol"]["update_epochs"] == 10
    )

    artifact_hashes = True
    for name, metadata in manifest["artifacts"].items():
        path = Path(metadata["path"])
        artifact_hashes = bool(
            artifact_hashes
            and path.is_file()
            and path.resolve() == (root / name).resolve()
            and checkpoint_sha256(path) == metadata["sha256"]
        )
    checks["artifact_hashes"] = artifact_hashes

    source_binding = True
    for record in manifest["source_records"]:
        for path_key, sha_key in (
            ("checkpoint", "checkpoint_sha256"),
            ("teacher_checkpoint", "teacher_sha256"),
            ("evaluator_checkpoint", "evaluator_sha256"),
            ("evaluator_source", "evaluator_source_sha256"),
            ("compatibility_cache", "compatibility_cache_sha256"),
            ("compatibility_config", "compatibility_config_sha256"),
        ):
            path = Path(record[path_key])
            source_binding = bool(
                source_binding
                and path.is_file()
                and checkpoint_sha256(path) == record[sha_key]
            )
    for record in manifest["stage3_references"]:
        path = Path(record["path"])
        source_binding = bool(
            source_binding
            and path.is_file()
            and checkpoint_sha256(path) == record["sha256"]
        )
    checks["source_and_checkpoint_binding"] = source_binding

    checks["complete_nine_run_grid"] = bool(
        len(grid) == len(FORMAL_SEEDS) * len(RUNS)
        and not grid.duplicated(["Seed", "Run"]).any()
        and set(grid.Seed.astype(int)) == set(FORMAL_SEEDS)
        and set(grid.Run.astype(str)) == set(RUNS)
    )
    checks["valid_only_outputs"] = bool(
        set(raw_events.Split.astype(str)) == {"valid"}
        and set(raw_events.SelectedBy.astype(str)) == {"validation_J"}
        and not grid.TestConstructed.astype(bool).any()
        and set(raw_events.Seed.astype(int)) == set(FORMAL_SEEDS)
        and set(raw_events.Run.astype(str)) == set(RUNS)
        and set(raw_events.Mode.astype(str)) == {"LAV", "LA", "LV", "L"}
        and not raw_events.duplicated(["Seed", "Run", "Mode", "sample_index"]).any()
    )
    valid_count = int(raw_events.sample_index.nunique())
    checks["complete_valid_prediction_grid"] = bool(
        valid_count == 229
        and len(raw_events) == len(FORMAL_SEEDS) * len(RUNS) * 4 * valid_count
    )

    recomputed_events = derive_valid_events(
        raw_events[
            [
                "Seed",
                "Run",
                "Mode",
                "sample_index",
                "sample_id",
                "label",
                "baseline_prediction",
                "candidate_prediction",
                "teacher_prediction",
                "Split",
                "SelectedBy",
            ]
        ]
    )
    checks["valid_events_recomputed"] = same_frame(
        recorded_events,
        recomputed_events,
        ["Seed", "Run", "Mode", "sample_index"],
    )
    recomputed_overall = overall_from_events(recomputed_events)
    recomputed_groups = group_summary(recomputed_events)
    checks["overall_metrics_recomputed"] = same_frame(
        recorded_overall,
        recomputed_overall,
        ["Seed", "Run", "Mode"],
    )
    checks["group_metrics_recomputed"] = same_frame(
        recorded_groups,
        recomputed_groups,
        ["Seed", "Run", "GroupType", "GroupValue"],
    )

    grid_binding = True
    for row in grid.itertuples(index=False):
        local = recomputed_overall.loc[
            recomputed_overall.Seed.astype(int).eq(int(row.Seed))
            & recomputed_overall.Run.astype(str).eq(str(row.Run))
        ].set_index("Mode")
        if set(local.index) != {"LAV", "LA", "LV", "L", "J"}:
            grid_binding = False
            continue
        grid_binding = bool(
            grid_binding
            and abs(float(row.J_valid) - float(local.loc["J", "candidate_MAE"])) <= 1e-9
        )
        for mode in ("LAV", "LA", "LV", "L"):
            grid_binding = bool(
                grid_binding
                and abs(
                    float(getattr(row, f"valid_{mode}_MAE"))
                    - float(local.loc[mode, "candidate_MAE"])
                )
                <= 1e-9
            )
    checks["grid_metrics_bound_to_predictions"] = grid_binding

    replay_checks = {}
    replay_passed = True
    for seed in FORMAL_SEEDS:
        reference_rows = references.loc[references.Seed.astype(int).eq(seed)]
        replay_rows = grid.loc[
            grid.Seed.astype(int).eq(seed)
            & grid.Run.astype(str).eq("cfcompat_replay")
        ]
        if len(reference_rows) != 1 or len(replay_rows) != 1:
            replay_passed = False
            continue
        check = replay_gate(
            replay_rows.iloc[0].to_dict(), reference_rows.iloc[0].to_dict()
        )
        replay_checks[str(seed)] = check
        replay_passed = bool(replay_passed and check["passed"])
    checks["baseline_replays_exact"] = replay_passed
    checks["baseline_replay_summary_matches"] = nested_close(
        replay_checks, summary["baseline_replay_gates"]
    )

    recomputed_gates = {
        run: aggregate_candidate_gate(
            run,
            grid.to_dict("records"),
            epochs.to_dict("records"),
            recomputed_groups,
        )
        for run in ("safe_uniform", "safe_cfcompat")
    }
    checks["candidate_gates_recomputed"] = nested_close(
        recomputed_gates, summary["candidate_gates"]
    )
    if recomputed_gates["safe_cfcompat"]["passed"]:
        verdict = "PROMOTE_SAFE_CFCompatKD_TO_MOSEI_SINGLE_SEED_VALID_SCREEN"
    elif recomputed_gates["safe_uniform"]["passed"]:
        verdict = "SAFE_PROJECTION_ABLATION_PASSED_CFCompat_EXTENSION_FAILED"
    else:
        verdict = "STOP_SAFE_CFCompatKD_HELDOUT_VALID_FAILED"
    checks["verdict_recomputed"] = bool(summary["verdict"] == verdict)

    projection_accounting = True
    for row in grid.itertuples(index=False):
        count = int(row.projection_sample_count)
        expected = 0 if row.Run == "cfcompat_replay" else 1284 * int(row.TrainEpochCount)
        fractions = [
            float(row.projection_wrong_direction_fraction),
            float(row.projection_overshoot_fraction),
            float(row.projection_zero_width_fraction),
            float(row.projection_unchanged_fraction),
            float(row.projection_projected_to_baseline_fraction),
            float(row.projection_projected_to_label_fraction),
        ]
        projection_accounting = bool(
            projection_accounting
            and count == expected
            and all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in fractions)
            and math.isfinite(float(row.projection_mean_teacher_target_abs_shift))
            and float(row.projection_mean_teacher_target_abs_shift) >= 0.0
            and math.isfinite(float(row.projection_mean_safe_interval_width))
            and float(row.projection_mean_safe_interval_width) >= 0.0
        )
    checks["projection_accounting"] = projection_accounting

    checks["finite_metrics"] = bool(
        np.isfinite(
            raw_events[
                [
                    "label",
                    "baseline_prediction",
                    "candidate_prediction",
                    "teacher_prediction",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            grid[
                [
                    "J_valid",
                    "valid_LAV_MAE",
                    "valid_LA_MAE",
                    "valid_LV_MAE",
                    "valid_L_MAE",
                ]
            ].to_numpy(dtype=float)
        ).all()
    )
    checks["no_test_named_artifacts"] = bool(
        not any("test" in path.name.lower() for path in root.iterdir())
    )

    passed = bool(all(checks.values()))
    payload = {
        "version": VERSION,
        "passed": passed,
        "verdict": verdict,
        "checks": checks,
        "recomputed_candidate_gates": jsonable(recomputed_gates),
    }
    output = root / "safe_projection_audit_check.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("Independent Safe-CFCompat audit failed: {}".format(failed))
    print("Safe-CFCompatKD independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
