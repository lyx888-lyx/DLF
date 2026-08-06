"""Independent audit for student-safe dynamic-utility residual-CFCompat v3."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_safe_projection_utils import (
    derive_valid_events,
    group_summary,
    overall_from_events,
    replay_gate,
)
from trains.singleTask.cfcompat_student_safe_utility_utils import (
    CANDIDATE_RUNS,
    FORMAL_SEEDS,
    METHOD,
    PRIMARY_RUN,
    RESIDUAL_CFCOMPAT_ALPHA,
    RUNS,
    UNIFORM_RUN,
    UTILITY_RUN,
    VERSION,
    candidate_gate,
    frozen_thresholds,
    jsonable,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES


PREFIX = "student_safe_utility_v3"
GRID_BINDING_TOLERANCE = 1e-6


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
        if (
            pd.api.types.is_numeric_dtype(left[column])
            and pd.api.types.is_numeric_dtype(right[column])
        ):
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


def verdict_from_gates(gates):
    if gates[PRIMARY_RUN]["passed"]:
        return "PROMOTE_UTILITY_RESIDUAL_CFCOMPAT_TO_1111_1114_EXTENSION"
    if gates[UTILITY_RUN]["passed"]:
        return "DYNAMIC_UTILITY_PASSED_RESIDUAL_CFCOMPAT_FAILED"
    if gates[UNIFORM_RUN]["passed"]:
        return "STUDENT_SAFE_UNIFORM_PASSED_DYNAMIC_UTILITY_FAILED"
    return "STOP_STUDENT_SAFE_UTILITY_V3_FAILED"


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    files = {
        "grid": root / f"{PREFIX}_valid_grid_summary.csv",
        "epochs": root / f"{PREFIX}_all_epoch_metrics.csv",
        "raw_events": root / f"{PREFIX}_raw_valid_events.csv",
        "events": root / f"{PREFIX}_valid_events.csv",
        "overall": root / f"{PREFIX}_overall_metrics.csv",
        "groups": root / f"{PREFIX}_group_metrics.csv",
        "references": root / f"{PREFIX}_stage3_references.csv",
        "manifest": root / f"{PREFIX}_source_manifest.json",
        "summary": root / f"{PREFIX}_valid_screen_summary.json",
        "report": root / f"{PREFIX}_valid_screen_report.md",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing student-safe utility v3 artifacts:\n" + "\n".join(missing)
        )

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
    protocol = summary["protocol"]
    checks["frozen_protocol"] = bool(
        tuple(manifest["formal_seeds"]) == FORMAL_SEEDS
        and tuple(manifest["runs"]) == RUNS
        and manifest["decision_split"] == "official_valid_only"
        and not manifest["official_test_constructed"]
        and manifest["test_loader_construction_count"] == 0
        and manifest["test_loader_traversal_count"] == 0
        and manifest["exploratory_after_v1_v2"]
        and manifest["difficulty_scale_source"]
        == "initial_student_train_only_per_mode_median_absolute_error"
        and abs(
            float(manifest["residual_cfcompat_alpha"])
            - RESIDUAL_CFCOMPAT_ALPHA
        ) <= 1e-12
        and protocol["formal_seeds"] == list(FORMAL_SEEDS)
        and protocol["runs"] == list(RUNS)
        and protocol["exploratory_after_v1_v2"]
        and not protocol["official_test_constructed"]
        and not protocol["official_test_authorized"]
        and protocol["lambda_kd"] == 1.0
        and protocol["update_epochs"] == 10
        and protocol["abstention"]
        == "gate_zero_when_projected_target_equals_current_student"
        and protocol["difficulty_scale_source"]
        == "initial_student_train_only_per_mode_median_absolute_error"
        and protocol["primary_gate"]
        == "active_times_utility_times_difficulty_times_residual_cfcompat"
        and abs(
            float(protocol["residual_cfcompat_alpha"])
            - RESIDUAL_CFCOMPAT_ALPHA
        ) <= 1e-12
        and nested_close(
            protocol["frozen_candidate_thresholds"], frozen_thresholds()
        )
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

    checks["complete_twelve_run_grid"] = bool(
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
        and not raw_events.duplicated(
            ["Seed", "Run", "Mode", "sample_index"]
        ).any()
    )
    valid_count = int(raw_events.sample_index.nunique())
    checks["complete_valid_prediction_grid"] = bool(
        valid_count == 229
        and len(raw_events) == len(FORMAL_SEEDS) * len(RUNS) * 4 * valid_count
    )

    required_raw = [
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
    recomputed_events = derive_valid_events(raw_events[required_raw])
    recomputed_overall = overall_from_events(recomputed_events)
    recomputed_groups = group_summary(recomputed_events)
    checks["valid_events_recomputed"] = same_frame(
        recorded_events,
        recomputed_events,
        ["Seed", "Run", "Mode", "sample_index"],
    )
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

    binding_rows = []
    grid_binding = True
    for row in grid.itertuples(index=False):
        local = recomputed_overall.loc[
            recomputed_overall.Seed.astype(int).eq(int(row.Seed))
            & recomputed_overall.Run.astype(str).eq(str(row.Run))
        ].set_index("Mode")
        if set(local.index) != {"LAV", "LA", "LV", "L", "J"}:
            grid_binding = False
            continue
        pairs = [
            ("J_valid", float(row.J_valid), float(local.loc["J", "candidate_MAE"]))
        ]
        for mode in ("LAV", "LA", "LV", "L"):
            pairs.append(
                (
                    f"valid_{mode}_MAE",
                    float(getattr(row, f"valid_{mode}_MAE")),
                    float(local.loc[mode, "candidate_MAE"]),
                )
            )
        for metric, grid_value, prediction_value in pairs:
            difference = abs(grid_value - prediction_value)
            binding_rows.append(
                {
                    "Seed": int(row.Seed),
                    "Run": str(row.Run),
                    "Metric": metric,
                    "GridValue": grid_value,
                    "PredictionValue": prediction_value,
                    "AbsoluteDifference": difference,
                }
            )
            grid_binding = bool(
                grid_binding and difference <= GRID_BINDING_TOLERANCE
            )
    checks["grid_metrics_bound_to_predictions"] = grid_binding
    pd.DataFrame(binding_rows).to_csv(
        root / f"{PREFIX}_grid_binding_diagnostics.csv", index=False
    )

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
        run: candidate_gate(
            run,
            grid.to_dict("records"),
            epochs.to_dict("records"),
            recomputed_groups,
        )
        for run in CANDIDATE_RUNS
    }
    checks["candidate_gates_recomputed"] = nested_close(
        recomputed_gates, summary["candidate_gates"]
    )
    verdict = verdict_from_gates(recomputed_gates)
    checks["verdict_recomputed"] = bool(summary["verdict"] == verdict)

    tau_consistency = True
    for seed in FORMAL_SEEDS:
        local = grid.loc[
            grid.Seed.astype(int).eq(seed)
            & grid.Run.astype(str).isin(CANDIDATE_RUNS)
        ]
        if len(local) != len(CANDIDATE_RUNS):
            tau_consistency = False
            continue
        for mode in MISSING_MODES:
            values = local["DifficultyTau{}".format(mode)].to_numpy(dtype=float)
            tau_consistency = bool(
                tau_consistency
                and np.isfinite(values).all()
                and np.all(values > 0.0)
                and float(values.max() - values.min()) <= 1e-9
            )
    replay_rows = grid.loc[grid.Run.astype(str).eq("cfcompat_replay")]
    for mode in MISSING_MODES:
        tau_consistency = bool(
            tau_consistency
            and replay_rows["DifficultyTau{}".format(mode)].isna().all()
        )
    checks["train_only_difficulty_scales_consistent"] = tau_consistency

    projection_accounting = True
    required_projection_columns = [
        "projection_mean_utility",
        "projection_mean_active_utility",
        "projection_mean_difficulty",
        "projection_mean_compatibility",
        "projection_mean_residual_compatibility",
        "projection_mean_final_gate",
        "projection_gate_effective_sample_size",
        "projection_active_compatibility_utility_pearson",
        "projection_active_compatibility_utility_spearman",
    ]
    if any(column not in grid.columns for column in required_projection_columns):
        projection_accounting = False

    for row in grid.itertuples(index=False):
        count = int(row.projection_sample_count)
        if row.Run == "cfcompat_replay":
            projection_accounting = bool(
                projection_accounting
                and count == 0
                and math.isnan(float(row.projection_mean_utility))
            )
            continue

        expected = 1284 * int(row.TrainEpochCount)
        active = float(row.projection_active_fraction)
        abstain = float(row.projection_abstain_fraction)
        projected_to_student = float(
            row.projection_projected_to_baseline_fraction
        )
        fractions = [
            float(row.projection_wrong_direction_fraction),
            float(row.projection_overshoot_fraction),
            float(row.projection_zero_width_fraction),
            float(row.projection_unchanged_fraction),
            float(row.projection_projected_to_label_fraction),
            active,
            abstain,
            float(row.projection_mean_utility),
            float(row.projection_mean_active_utility),
            float(row.projection_mean_difficulty),
            float(row.projection_mean_compatibility),
            float(row.projection_mean_residual_compatibility),
            float(row.projection_mean_final_gate),
        ]
        ess = float(row.projection_gate_effective_sample_size)
        pearson = float(row.projection_active_compatibility_utility_pearson)
        spearman = float(row.projection_active_compatibility_utility_spearman)
        projection_accounting = bool(
            projection_accounting
            and count == expected
            and all(math.isfinite(value) for value in fractions)
            and all(0.0 <= value <= 1.0 for value in fractions[:11])
            and RESIDUAL_CFCOMPAT_ALPHA
            < float(row.projection_mean_residual_compatibility)
            < 1.0
            and 0.0 <= float(row.projection_mean_final_gate) <= active + 1e-9
            and abs(active + abstain - 1.0) <= 1e-9
            and abs(abstain - projected_to_student) <= 1e-9
            and math.isfinite(ess)
            and 0.0 < ess <= count + 1e-9
            and math.isfinite(pearson)
            and -1.0 <= pearson <= 1.0
            and math.isfinite(spearman)
            and -1.0 <= spearman <= 1.0
        )
        if row.Run == UNIFORM_RUN:
            projection_accounting = bool(
                projection_accounting
                and abs(
                    float(row.projection_mean_final_gate) - active
                ) <= 1e-9
            )
        if row.Run == PRIMARY_RUN:
            projection_accounting = bool(
                projection_accounting
                and abs(
                    float(row.ResidualCFCompatAlpha)
                    - RESIDUAL_CFCOMPAT_ALPHA
                ) <= 1e-12
                and bool(row.CompatibilityUsed)
                and bool(row.DynamicUtilityUsed)
                and bool(row.DynamicDifficultyUsed)
            )
        if row.Run == UTILITY_RUN:
            projection_accounting = bool(
                projection_accounting
                and not bool(row.CompatibilityUsed)
                and bool(row.DynamicUtilityUsed)
                and bool(row.DynamicDifficultyUsed)
            )
    checks["projection_utility_and_abstention_accounting"] = projection_accounting

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
        "grid_binding_tolerance": GRID_BINDING_TOLERANCE,
        "checks": checks,
        "recomputed_candidate_gates": jsonable(recomputed_gates),
    }
    output = root / f"{PREFIX}_audit_check.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(
            "Independent student-safe utility v3 audit failed: {}".format(failed)
        )
    print("Student-Safe Dynamic-Utility v3 independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
