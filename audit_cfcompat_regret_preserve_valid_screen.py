"""Independent audit for Regret-Aware Preserve-or-Distill CFCompatKD v4."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_regret_preserve_utils import (
    DEV_SEED,
    DISTILL_MARGIN,
    LAMBDA_PRESERVE,
    METHOD,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    PRESERVE_MARGIN,
    RUN,
    RUNS,
    VERSION,
    dev_candidate_gate,
    frozen_thresholds,
    jsonable,
    negative_transfer_summary,
)
from trains.singleTask.cfcompat_safe_projection_utils import (
    derive_valid_events,
    group_summary,
    overall_from_events,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES


PREFIX = "regret_preserve_v4"
GRID_BINDING_TOLERANCE = 1e-6
PROJECTION_TOLERANCE = 1e-12


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
    if expected is None:
        return actual is None
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(
            nested_close(actual[key], expected[key], tolerance) for key in expected
        )
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(
            nested_close(a, b, tolerance) for a, b in zip(actual, expected)
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


def clip(values, lower, upper):
    return np.maximum(np.minimum(values, upper), lower)


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    files = {
        "grid": root / f"{PREFIX}_candidate_grid.csv",
        "epochs": root / f"{PREFIX}_all_epoch_metrics.csv",
        "candidate_raw": root / f"{PREFIX}_candidate_raw_valid_events.csv",
        "reference_grid": root / f"{PREFIX}_reference_grid.csv",
        "reference_raw": root / f"{PREFIX}_reference_raw_valid_events.csv",
        "events": root / f"{PREFIX}_combined_valid_events.csv",
        "overall": root / f"{PREFIX}_overall_metrics.csv",
        "groups": root / f"{PREFIX}_group_metrics.csv",
        "negative": root / f"{PREFIX}_negative_transfer_metrics.csv",
        "train_baseline": root / f"{PREFIX}_train_baseline_cache.csv",
        "train_decisions": root / f"{PREFIX}_train_decisions.csv",
        "manifest": root / f"{PREFIX}_source_manifest.json",
        "summary": root / f"{PREFIX}_valid_screen_summary.json",
        "report": root / f"{PREFIX}_valid_screen_report.md",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing v4 audit artifacts:\n" + "\n".join(missing))

    grid = pd.read_csv(files["grid"])
    epochs = pd.read_csv(files["epochs"])
    candidate_raw = pd.read_csv(files["candidate_raw"])
    reference_grid = pd.read_csv(files["reference_grid"])
    reference_raw = pd.read_csv(files["reference_raw"])
    recorded_events = pd.read_csv(files["events"])
    recorded_overall = pd.read_csv(files["overall"])
    recorded_groups = pd.read_csv(files["groups"])
    recorded_negative = pd.read_csv(files["negative"])
    train_baseline = pd.read_csv(files["train_baseline"])
    train_decisions = pd.read_csv(files["train_decisions"])
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
        manifest["development_seed"] == DEV_SEED
        and tuple(manifest["runs"]) == RUNS
        and manifest["decision_split"] == "official_valid_only"
        and not manifest["official_test_constructed"]
        and manifest["test_loader_construction_count"] == 0
        and manifest["test_loader_traversal_count"] == 0
        and protocol["development_seed"] == DEV_SEED
        and protocol["new_trajectories_trained"] == 1
        and tuple(protocol["runs"]) == RUNS
        and protocol["reference_runs"] == ["cfcompat_replay", "student_safe_uniform"]
        and protocol["regret_anchor"] == "frozen_validation_best_moddrop_missing_prediction"
        and protocol["lambda_kd"] == 1.0
        and protocol["lambda_preserve"] == LAMBDA_PRESERVE
        and protocol["update_epochs"] == 10
        and not protocol["official_test_constructed"]
        and not protocol["official_test_authorized"]
        and nested_close(protocol["frozen_thresholds"], frozen_thresholds())
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

    reference_hashes = True
    for metadata in manifest["reference_sources"].values():
        path = Path(metadata["path"])
        reference_hashes = bool(
            reference_hashes
            and path.is_file()
            and checkpoint_sha256(path) == metadata["sha256"]
        )
    checks["frozen_v2_reference_hashes"] = reference_hashes

    source = manifest["source_record"]
    source_hashes = True
    for path_key, sha_key in (
        ("checkpoint", "checkpoint_sha256"),
        ("teacher_checkpoint", "teacher_sha256"),
        ("evaluator_checkpoint", "evaluator_sha256"),
        ("evaluator_source", "evaluator_source_sha256"),
        ("compatibility_cache", "compatibility_cache_sha256"),
        ("compatibility_config", "compatibility_config_sha256"),
        ("train_baseline_cache", "train_baseline_cache_sha256"),
        ("train_decision_records", "train_decision_records_sha256"),
    ):
        path = Path(source[path_key])
        source_hashes = bool(
            source_hashes and path.is_file() and checkpoint_sha256(path) == source[sha_key]
        )
    checks["source_and_checkpoint_binding"] = source_hashes

    checks["single_seed_single_new_trajectory"] = bool(
        len(grid) == 1
        and int(grid.iloc[0].Seed) == DEV_SEED
        and str(grid.iloc[0].Run) == RUN
        and set(epochs.Seed.astype(int)) == {DEV_SEED}
        and set(epochs.Run.astype(str)) == {RUN}
    )
    checks["frozen_reference_grid_shape"] = bool(
        len(reference_grid) == 2
        and set(reference_grid.Seed.astype(int)) == {DEV_SEED}
        and set(reference_grid.Run.astype(str)) == {"cfcompat_replay", "student_safe_uniform"}
    )

    # Rebind the copied reference artifacts to the original audited v2 files.
    original_grid_path = Path(manifest["reference_sources"]["grid"]["path"])
    original_raw_path = Path(manifest["reference_sources"]["raw"]["path"])
    original_grid = pd.read_csv(original_grid_path)
    original_grid = original_grid.loc[
        original_grid.Seed.astype(int).eq(DEV_SEED)
        & original_grid.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    original_raw = pd.read_csv(original_raw_path)
    original_raw = original_raw.loc[
        original_raw.Seed.astype(int).eq(DEV_SEED)
        & original_raw.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    checks["reference_grid_exact_copy"] = same_frame(
        reference_grid, original_grid, ["Seed", "Run"], tolerance=1e-12
    )
    checks["reference_predictions_exact_copy"] = same_frame(
        reference_raw,
        original_raw,
        ["Seed", "Run", "Mode", "sample_index"],
        tolerance=1e-12,
    )

    checks["valid_only_outputs"] = bool(
        set(candidate_raw.Split.astype(str)) == {"valid"}
        and set(reference_raw.Split.astype(str)) == {"valid"}
        and set(candidate_raw.SelectedBy.astype(str)) == {"validation_J"}
        and set(reference_raw.SelectedBy.astype(str)) == {"validation_J"}
        and not grid.TestConstructed.astype(bool).any()
    )
    checks["complete_valid_prediction_grid"] = bool(
        len(candidate_raw) == 4 * 229
        and len(reference_raw) == 2 * 4 * 229
        and candidate_raw.sample_index.nunique() == 229
        and set(candidate_raw.Mode.astype(str)) == {"LAV", "LA", "LV", "L"}
        and not candidate_raw.duplicated(["Seed", "Run", "Mode", "sample_index"]).any()
    )

    combined_raw = pd.concat([reference_raw, candidate_raw], ignore_index=True, sort=False)
    recomputed_events = derive_valid_events(combined_raw)
    recomputed_overall = overall_from_events(recomputed_events)
    recomputed_groups = group_summary(recomputed_events)
    recomputed_negative = negative_transfer_summary(recomputed_events)
    checks["valid_events_recomputed"] = same_frame(
        recorded_events,
        recomputed_events,
        ["Seed", "Run", "Mode", "sample_index"],
    )
    checks["overall_metrics_recomputed"] = same_frame(
        recorded_overall, recomputed_overall, ["Seed", "Run", "Mode"]
    )
    checks["group_metrics_recomputed"] = same_frame(
        recorded_groups,
        recomputed_groups,
        ["Seed", "Run", "GroupType", "GroupValue"],
    )
    checks["negative_transfer_recomputed"] = same_frame(
        recorded_negative, recomputed_negative, ["Seed", "Run", "Mode"]
    )

    local = recomputed_overall.loc[
        recomputed_overall.Seed.astype(int).eq(DEV_SEED)
        & recomputed_overall.Run.astype(str).eq(RUN)
    ].set_index("Mode")
    binding = abs(float(grid.iloc[0].J_valid) - float(local.loc["J", "candidate_MAE"]))
    mode_bindings = [binding]
    for mode in ("LAV",) + MISSING_MODES:
        mode_bindings.append(
            abs(
                float(grid.iloc[0]["valid_{}_MAE".format(mode)])
                - float(local.loc[mode, "candidate_MAE"])
            )
        )
    checks["grid_metrics_bound_to_predictions"] = bool(
        all(value <= GRID_BINDING_TOLERANCE for value in mode_bindings)
    )

    # Train-only frozen ModDrop baseline cache integrity.
    required_baseline = ["sample_index", "sample_id", "label"] + [
        "baseline_{}_pred".format(mode) for mode in MISSING_MODES
    ]
    checks["train_baseline_cache_complete"] = bool(
        list(train_baseline.columns) == required_baseline
        and len(train_baseline) == 1284
        and train_baseline.sample_index.nunique() == 1284
        and np.isfinite(
            train_baseline[["label"] + ["baseline_{}_pred".format(mode) for mode in MISSING_MODES]].to_numpy(dtype=float)
        ).all()
    )

    # Independently recompute every three-way Train decision from recorded raw values.
    expected_count = 1284 * int(grid.iloc[0].TrainEpochCount)
    decision_ok = bool(
        len(train_decisions) == expected_count
        and set(train_decisions["mode"].astype(str)) == set(MISSING_MODES)
        and int(train_decisions.Epoch.min()) == 1
        and int(train_decisions.Epoch.max()) == int(grid.iloc[0].TrainEpochCount)
        and all(
            int((train_decisions.Epoch.astype(int) == epoch).sum()) == 1284
            for epoch in range(1, int(grid.iloc[0].TrainEpochCount) + 1)
        )
    )
    if decision_ok:
        cache = train_baseline.set_index("sample_index")
        student = train_decisions.student_prediction.to_numpy(dtype=float)
        teacher = train_decisions.teacher_prediction.to_numpy(dtype=float)
        baseline = train_decisions.baseline_prediction.to_numpy(dtype=float)
        label = train_decisions.label.to_numpy(dtype=float)
        compat = train_decisions.compatibility.to_numpy(dtype=float)
        lower = np.minimum(student, label)
        upper = np.maximum(student, label)
        teacher_safe = clip(teacher, lower, upper)
        preserve_safe = clip(baseline, lower, upper)
        baseline_error = np.abs(baseline - label)
        teacher_error = np.abs(teacher - label)
        current_error = np.abs(student - label)
        teacher_advantage = baseline_error - teacher_error
        current_regret = current_error - baseline_error
        active = ~np.isclose(teacher_safe, student, atol=PROJECTION_TOLERANCE, rtol=0.0)
        teacher_beneficial = teacher_advantage >= DISTILL_MARGIN
        current_regressed = current_regret >= PRESERVE_MARGIN
        distill = teacher_beneficial & active
        preserve = (~distill) & current_regressed
        abstain = ~(distill | preserve)
        mild = MILD_CFCOMPAT_BASE + MILD_CFCOMPAT_SCALE * compat
        distill_gate = distill.astype(float) * mild
        preserve_gate = preserve.astype(float)

        bound_baseline = np.asarray(
            [
                float(cache.loc[int(index), "baseline_{}_pred".format(mode)])
                for index, mode in zip(
                    train_decisions.sample_index.astype(int),
                    train_decisions["mode"].astype(str),
                )
            ],
            dtype=float,
        )
        bound_label = np.asarray(
            [float(cache.loc[int(index), "label"]) for index in train_decisions.sample_index.astype(int)],
            dtype=float,
        )
        comparisons = [
            np.allclose(baseline, bound_baseline, atol=1e-7, rtol=0.0),
            np.allclose(label, bound_label, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.teacher_safe_target, teacher_safe, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.preserve_safe_target, preserve_safe, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.baseline_error, baseline_error, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.teacher_error, teacher_error, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.current_error, current_error, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.teacher_advantage_vs_baseline, teacher_advantage, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.current_regret_vs_baseline, current_regret, atol=1e-7, rtol=0.0),
            np.array_equal(train_decisions.teacher_beneficial.astype(bool).to_numpy(), teacher_beneficial),
            np.array_equal(train_decisions.current_regressed.astype(bool).to_numpy(), current_regressed),
            np.array_equal(train_decisions.distill.astype(bool).to_numpy(), distill),
            np.array_equal(train_decisions.preserve.astype(bool).to_numpy(), preserve),
            np.array_equal(train_decisions.decision_abstain.astype(bool).to_numpy(), abstain),
            np.allclose(train_decisions.mild_compatibility, mild, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.distill_gate, distill_gate, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.preserve_gate, preserve_gate, atol=1e-7, rtol=0.0),
        ]
        decision_ok = bool(all(comparisons))
    checks["three_way_train_decisions_recomputed"] = decision_ok

    # Bind aggregate decision fractions in the grid to raw Train decisions.
    aggregate_ok = False
    if len(train_decisions):
        distill_fraction = float(train_decisions.distill.astype(bool).mean())
        preserve_fraction = float(train_decisions.preserve.astype(bool).mean())
        abstain_fraction = float(train_decisions.decision_abstain.astype(bool).mean())
        aggregate_ok = bool(
            abs(float(grid.iloc[0].projection_distill_fraction) - distill_fraction) <= 1e-9
            and abs(float(grid.iloc[0].projection_preserve_fraction) - preserve_fraction) <= 1e-9
            and abs(float(grid.iloc[0].projection_decision_abstain_fraction) - abstain_fraction) <= 1e-9
            and abs(distill_fraction + preserve_fraction + abstain_fraction - 1.0) <= 1e-12
        )
    checks["three_way_aggregate_accounting"] = aggregate_ok

    recomputed_gate = dev_candidate_gate(
        grid.iloc[0].to_dict(),
        reference_grid,
        recomputed_groups,
        recomputed_negative,
        epochs.to_dict("records"),
    )
    checks["candidate_gate_recomputed"] = nested_close(
        jsonable(recomputed_gate), summary["candidate_gate"]
    )
    verdict = (
        "PROMOTE_REGRET_PRESERVE_TO_3SEED_VALID_SCREEN"
        if recomputed_gate["passed"]
        else "STOP_REGRET_PRESERVE_SINGLE_SEED_DEV_FAILED"
    )
    checks["verdict_recomputed"] = bool(summary["verdict"] == verdict)

    checks["finite_metrics"] = bool(
        np.isfinite(
            grid[["J_valid", "valid_LAV_MAE", "valid_LA_MAE", "valid_LV_MAE", "valid_L_MAE"]].to_numpy(dtype=float)
        ).all()
        and np.isfinite(
            train_decisions[
                [
                    "student_prediction",
                    "baseline_prediction",
                    "teacher_prediction",
                    "label",
                    "compatibility",
                    "distill_gate",
                    "preserve_gate",
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
        "recomputed_candidate_gate": jsonable(recomputed_gate),
    }
    output = root / f"{PREFIX}_audit_check.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("Independent regret-preserve v4 audit failed: {}".format(failed))

    print("Regret-Aware Preserve-or-Distill v4 independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
