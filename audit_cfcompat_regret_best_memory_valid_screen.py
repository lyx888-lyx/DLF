"""Independent audit for Regret-Aware Best-So-Far Memory CFCompatKD v4.2."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_regret_best_memory_utils import (
    DEV_SEED,
    MEMORY_UPDATE_EPS,
    METHOD,
    MILD_CFCOMPAT_BASE_V4P2,
    MILD_CFCOMPAT_SCALE_V4P2,
    PRESERVE_MARGIN_V4P2,
    RUN,
    RUNS,
    STRONG_DISTILL_MARGIN,
    VERSION,
    WEAK_DISTILL_SCALE,
    dev_candidate_gate,
    frozen_thresholds,
    negative_transfer_summary,
)
from trains.singleTask.cfcompat_safe_projection_utils import (
    derive_valid_events,
    group_summary,
    overall_from_events,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES


PREFIX = "regret_best_memory_v4p2"
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
                left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float),
                atol=tolerance, rtol=0.0, equal_nan=True,
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


def clip(value, lower, upper):
    return max(min(value, upper), lower)


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    files = {
        "grid": root / f"{PREFIX}_candidate_grid.csv",
        "epochs": root / f"{PREFIX}_all_epoch_metrics.csv",
        "candidate_raw": root / f"{PREFIX}_candidate_raw_valid_events.csv",
        "v2_grid": root / f"{PREFIX}_v2_reference_grid.csv",
        "v2_raw": root / f"{PREFIX}_v2_reference_raw_valid_events.csv",
        "v4_grid": root / f"{PREFIX}_v4_reference_grid.csv",
        "v4_raw": root / f"{PREFIX}_v4_reference_raw_valid_events.csv",
        "v4p1_grid": root / f"{PREFIX}_v4p1_reference_grid.csv",
        "v4p1_raw": root / f"{PREFIX}_v4p1_reference_raw_valid_events.csv",
        "events": root / f"{PREFIX}_combined_valid_events.csv",
        "overall": root / f"{PREFIX}_overall_metrics.csv",
        "groups": root / f"{PREFIX}_group_metrics.csv",
        "transfer": root / f"{PREFIX}_transfer_metrics.csv",
        "baseline": root / f"{PREFIX}_train_baseline_cache.csv",
        "memory": root / f"{PREFIX}_final_memory.csv",
        "decisions": root / f"{PREFIX}_train_decisions.csv",
        "manifest": root / f"{PREFIX}_source_manifest.json",
        "summary": root / f"{PREFIX}_valid_screen_summary.json",
        "report": root / f"{PREFIX}_valid_screen_report.md",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing v4.2 audit artifacts:\n" + "\n".join(missing))

    grid = pd.read_csv(files["grid"])
    epochs = pd.read_csv(files["epochs"])
    candidate_raw = pd.read_csv(files["candidate_raw"])
    v2_grid = pd.read_csv(files["v2_grid"])
    v2_raw = pd.read_csv(files["v2_raw"])
    v4_grid = pd.read_csv(files["v4_grid"])
    v4_raw = pd.read_csv(files["v4_raw"])
    v4p1_grid = pd.read_csv(files["v4p1_grid"])
    v4p1_raw = pd.read_csv(files["v4p1_raw"])
    recorded_events = pd.read_csv(files["events"])
    recorded_overall = pd.read_csv(files["overall"])
    recorded_groups = pd.read_csv(files["groups"])
    recorded_transfer = pd.read_csv(files["transfer"])
    baseline = pd.read_csv(files["baseline"])
    final_memory = pd.read_csv(files["memory"])
    decisions = pd.read_csv(files["decisions"])
    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))

    checks = {}
    checks["version_and_method"] = bool(
        manifest["version"] == VERSION and manifest["method"] == METHOD
        and summary["version"] == VERSION and summary["method"] == METHOD
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
        and protocol["regret_anchor"] == "per_sample_per_missing_mode_best_so_far_prediction_memory"
        and protocol["memory_update_split"] == "train_only"
        and not protocol["official_test_constructed"]
        and not protocol["official_test_authorized"]
        and nested_close(protocol["frozen_thresholds"], frozen_thresholds())
    )

    artifact_hashes = True
    for name, metadata in manifest["artifacts"].items():
        path = Path(metadata["path"])
        artifact_hashes = bool(
            artifact_hashes and path.is_file() and path.resolve() == (root / name).resolve()
            and checkpoint_sha256(path) == metadata["sha256"]
        )
    checks["artifact_hashes"] = artifact_hashes

    reference_hashes = True
    for source_group in ("v2_reference_sources", "v4_reference_sources", "v4p1_reference_sources"):
        for metadata in manifest[source_group].values():
            path = Path(metadata["path"])
            reference_hashes = bool(
                reference_hashes and path.is_file() and checkpoint_sha256(path) == metadata["sha256"]
            )
    checks["frozen_reference_hashes"] = reference_hashes

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
        ("final_best_memory", "final_best_memory_sha256"),
        ("train_decision_records", "train_decision_records_sha256"),
    ):
        path = Path(source[path_key])
        source_hashes = bool(
            source_hashes and path.is_file() and checkpoint_sha256(path) == source[sha_key]
        )
    checks["source_and_checkpoint_binding"] = source_hashes

    checks["single_seed_single_new_trajectory"] = bool(
        len(grid) == 1 and int(grid.iloc[0].Seed) == DEV_SEED and str(grid.iloc[0].Run) == RUN
        and set(epochs.Seed.astype(int)) == {DEV_SEED} and set(epochs.Run.astype(str)) == {RUN}
    )
    checks["reference_grid_shapes"] = bool(
        len(v2_grid) == 2 and len(v4_grid) == 1 and len(v4p1_grid) == 1
    )

    # Exact copies of frozen reference artifacts.
    ref_copy_ok = True
    copy_specs = (
        (v2_grid, manifest["v2_reference_sources"]["grid"]["path"], ["Seed", "Run"], {"cfcompat_replay", "student_safe_uniform"}),
        (v2_raw, manifest["v2_reference_sources"]["raw"]["path"], ["Seed", "Run", "Mode", "sample_index"], {"cfcompat_replay", "student_safe_uniform"}),
        (v4_grid, manifest["v4_reference_sources"]["grid"]["path"], ["Seed", "Run"], {"regret_preserve_cfcompat"}),
        (v4_raw, manifest["v4_reference_sources"]["raw"]["path"], ["Seed", "Run", "Mode", "sample_index"], {"regret_preserve_cfcompat"}),
        (v4p1_grid, manifest["v4p1_reference_sources"]["grid"]["path"], ["Seed", "Run"], {"regret_preserve_guard_cfcompat"}),
        (v4p1_raw, manifest["v4p1_reference_sources"]["raw"]["path"], ["Seed", "Run", "Mode", "sample_index"], {"regret_preserve_guard_cfcompat"}),
    )
    for copied, source_path, keys, runs in copy_specs:
        original = pd.read_csv(source_path)
        original = original.loc[
            original.Seed.astype(int).eq(DEV_SEED) & original.Run.astype(str).isin(runs)
        ].copy()
        ref_copy_ok = bool(ref_copy_ok and same_frame(copied, original, keys, tolerance=1e-12))
    checks["reference_artifacts_exact_copy"] = ref_copy_ok

    checks["valid_only_outputs"] = bool(
        set(candidate_raw.Split.astype(str)) == {"valid"}
        and set(candidate_raw.SelectedBy.astype(str)) == {"validation_J"}
        and not grid.TestConstructed.astype(bool).any()
        and all(set(frame.Split.astype(str)) == {"valid"} for frame in (v2_raw, v4_raw, v4p1_raw))
    )
    checks["complete_valid_prediction_grid"] = bool(
        len(candidate_raw) == 4 * 229 and candidate_raw.sample_index.nunique() == 229
        and set(candidate_raw.Mode.astype(str)) == {"LAV", "LA", "LV", "L"}
        and not candidate_raw.duplicated(["Seed", "Run", "Mode", "sample_index"]).any()
    )

    combined_raw = pd.concat([v2_raw, v4_raw, v4p1_raw, candidate_raw], ignore_index=True, sort=False)
    recomputed_events = derive_valid_events(combined_raw)
    recomputed_overall = overall_from_events(recomputed_events)
    recomputed_groups = group_summary(recomputed_events)
    recomputed_transfer = negative_transfer_summary(recomputed_events)
    checks["valid_events_recomputed"] = same_frame(recorded_events, recomputed_events, ["Seed", "Run", "Mode", "sample_index"])
    checks["overall_metrics_recomputed"] = same_frame(recorded_overall, recomputed_overall, ["Seed", "Run", "Mode"])
    checks["group_metrics_recomputed"] = same_frame(recorded_groups, recomputed_groups, ["Seed", "Run", "GroupType", "GroupValue"])
    checks["transfer_metrics_recomputed"] = same_frame(recorded_transfer, recomputed_transfer, ["Seed", "Run", "Mode"])

    local = recomputed_overall.loc[
        recomputed_overall.Seed.astype(int).eq(DEV_SEED) & recomputed_overall.Run.astype(str).eq(RUN)
    ].set_index("Mode")
    binding = [abs(float(grid.iloc[0].J_valid) - float(local.loc["J", "candidate_MAE"]))]
    for mode in ("LAV",) + MISSING_MODES:
        binding.append(abs(float(grid.iloc[0]["valid_{}_MAE".format(mode)]) - float(local.loc[mode, "candidate_MAE"])))
    checks["grid_metrics_bound_to_predictions"] = bool(all(value <= GRID_BINDING_TOLERANCE for value in binding))

    required_baseline = ["sample_index", "sample_id", "label"] + [
        "baseline_{}_pred".format(mode) for mode in MISSING_MODES
    ]
    checks["train_baseline_cache_complete"] = bool(
        list(baseline.columns) == required_baseline and len(baseline) == 1284
        and baseline.sample_index.nunique() == 1284
    )
    checks["final_memory_shape"] = bool(
        len(final_memory) == 1284 * len(MISSING_MODES)
        and set(final_memory["mode"].astype(str)) == set(MISSING_MODES)
        and not final_memory.duplicated(["sample_index", "mode"]).any()
    )

    # Independently replay the stateful best-memory update and every route.
    expected_count = 1284 * int(grid.iloc[0].TrainEpochCount)
    memory_ok = bool(
        len(decisions) == expected_count
        and int(decisions.event_ordinal.min()) == 1
        and int(decisions.event_ordinal.max()) == expected_count
        and all(int((decisions.Epoch.astype(int) == epoch).sum()) == 1284 for epoch in range(1, int(grid.iloc[0].TrainEpochCount) + 1))
    )
    state = {}
    if memory_ok:
        for row in baseline.itertuples(index=False):
            for mode in MISSING_MODES:
                pred = float(getattr(row, "baseline_{}_pred".format(mode)))
                err = abs(pred - float(row.label))
                state[(int(row.sample_index), mode)] = [pred, err, 0, 0]

        ordered = decisions.sort_values("event_ordinal", kind="mergesort")
        for row in ordered.itertuples(index=False):
            key = (int(row.sample_index), str(row.mode))
            if key not in state:
                memory_ok = False
                break
            before_pred, before_err, update_count, last_event = state[key]
            label = float(row.label)
            current = float(row.student_prediction)
            teacher = float(row.teacher_prediction)
            compat = float(row.compatibility)
            current_err = abs(current - label)
            do_update = current_err + MEMORY_UPDATE_EPS < before_err
            after_pred = current if do_update else before_pred
            after_err = current_err if do_update else before_err
            after_count = update_count + (1 if do_update else 0)
            after_last = int(row.event_ordinal) if do_update else last_event

            lower, upper = min(current, label), max(current, label)
            teacher_safe = clip(teacher, lower, upper)
            preserve_safe = clip(after_pred, lower, upper)
            active = not np.isclose(teacher_safe, current, atol=PROJECTION_TOLERANCE, rtol=0.0)
            teacher_err = abs(teacher - label)
            advantage = after_err - teacher_err
            regret = current_err - after_err
            direction_correct = (teacher - after_pred) * (label - after_pred) > 0.0
            teacher_better = advantage > 0.0
            strong_candidate = advantage >= STRONG_DISTILL_MARGIN
            weak_candidate = teacher_better and advantage < STRONG_DISTILL_MARGIN and direction_correct
            strong = strong_candidate and active
            weak = weak_candidate and active
            current_regressed = regret >= PRESERVE_MARGIN_V4P2
            preserve = (not teacher_better) and current_regressed
            abstain = not (strong or weak or preserve)
            mild = MILD_CFCOMPAT_BASE_V4P2 + MILD_CFCOMPAT_SCALE_V4P2 * compat
            strong_gate = mild if strong else 0.0
            weak_gate = mild if weak else 0.0
            eligible = strong_gate + weak_gate
            effective = strong_gate + WEAK_DISTILL_SCALE * weak_gate
            preserve_gate = 1.0 if preserve else 0.0

            comparisons = (
                abs(float(row.memory_before_prediction) - before_pred) <= 1e-7,
                abs(float(row.memory_before_error) - before_err) <= 1e-7,
                bool(row.memory_updated) == do_update,
                abs(float(row.memory_after_prediction) - after_pred) <= 1e-7,
                abs(float(row.memory_after_error) - after_err) <= 1e-7,
                abs(float(row.memory_improvement) - (before_err - after_err)) <= 1e-7,
                int(row.memory_update_count_after) == after_count,
                abs(float(row.memory_error) - after_err) <= 1e-7,
                abs(float(row.teacher_error) - teacher_err) <= 1e-7,
                abs(float(row.current_error) - current_err) <= 1e-7,
                abs(float(row.teacher_advantage_vs_memory) - advantage) <= 1e-7,
                abs(float(row.current_regret_vs_memory) - regret) <= 1e-7,
                bool(row.strong_distill) == strong,
                bool(row.weak_distill) == weak,
                bool(row.preserve) == preserve,
                bool(row.memory_abstain) == abstain,
                abs(float(row.teacher_safe_target) - teacher_safe) <= 1e-7,
                abs(float(row.preserve_safe_target) - preserve_safe) <= 1e-7,
                abs(float(row.mild_compatibility) - mild) <= 1e-7,
                abs(float(row.strong_gate) - strong_gate) <= 1e-7,
                abs(float(row.weak_gate) - weak_gate) <= 1e-7,
                abs(float(row.eligible_distill_mass) - eligible) <= 1e-7,
                abs(float(row.effective_distill_gate) - effective) <= 1e-7,
                abs(float(row.preserve_gate) - preserve_gate) <= 1e-7,
            )
            if not all(comparisons):
                memory_ok = False
                break
            state[key] = [after_pred, after_err, after_count, after_last]
    checks["best_memory_and_routes_recomputed"] = memory_ok

    final_ok = memory_ok
    if final_ok:
        final_indexed = final_memory.set_index(["sample_index", "mode"])
        for key, (prediction, error, count, last_event) in state.items():
            if key not in final_indexed.index:
                final_ok = False
                break
            row = final_indexed.loc[key]
            if not (
                abs(float(row.best_prediction) - prediction) <= 1e-7
                and abs(float(row.best_error) - error) <= 1e-7
                and int(row.update_count) == int(count)
                and int(row.last_update_event) == int(last_event)
                and float(row.best_error) <= float(row.initial_error) + 1e-12
            ):
                final_ok = False
                break
    checks["final_memory_recomputed"] = final_ok

    aggregate_ok = False
    if len(decisions):
        fractions = {
            "strong_distill_fraction": float(decisions.strong_distill.astype(bool).mean()),
            "weak_distill_fraction": float(decisions.weak_distill.astype(bool).mean()),
            "preserve_fraction": float(decisions.preserve.astype(bool).mean()),
            "memory_abstain_fraction": float(decisions.memory_abstain.astype(bool).mean()),
            "memory_update_fraction": float(decisions.memory_updated.astype(bool).mean()),
        }
        aggregate_ok = all(
            abs(float(grid.iloc[0]["projection_{}".format(key)]) - value) <= 1e-9
            for key, value in fractions.items()
        )
        aggregate_ok = bool(
            aggregate_ok and abs(
                fractions["strong_distill_fraction"] + fractions["weak_distill_fraction"]
                + fractions["preserve_fraction"] + fractions["memory_abstain_fraction"] - 1.0
            ) <= 1e-12
        )
    checks["memory_aggregate_accounting"] = aggregate_ok

    recomputed_gate = dev_candidate_gate(
        grid.iloc[0].to_dict(), v4_grid, v4p1_grid,
        recomputed_groups, recomputed_transfer, epochs.to_dict("records")
    )
    checks["candidate_gate_recomputed"] = nested_close(recomputed_gate, summary["candidate_gate"])
    verdict = (
        "PROMOTE_REGRET_BEST_MEMORY_V4P2_TO_3SEED_VALID_SCREEN"
        if recomputed_gate["passed"]
        else "STOP_REGRET_BEST_MEMORY_V4P2_SINGLE_SEED_DEV_FAILED"
    )
    checks["verdict_recomputed"] = bool(summary["verdict"] == verdict)

    checks["finite_metrics"] = bool(
        np.isfinite(grid[["J_valid", "valid_LAV_MAE", "valid_LA_MAE", "valid_LV_MAE", "valid_L_MAE"]].to_numpy(dtype=float)).all()
        and np.isfinite(decisions[[
            "student_prediction", "teacher_prediction", "memory_before_prediction",
            "memory_after_prediction", "memory_before_error", "memory_after_error",
            "compatibility", "strong_gate", "weak_gate", "preserve_gate",
        ]].to_numpy(dtype=float)).all()
        and np.isfinite(final_memory[["initial_prediction", "initial_error", "best_prediction", "best_error"]].to_numpy(dtype=float)).all()
    )
    checks["no_test_named_artifacts"] = bool(not any("test" in path.name.lower() for path in root.iterdir()))

    passed = bool(all(checks.values()))
    payload = {
        "version": VERSION,
        "passed": passed,
        "verdict": verdict,
        "grid_binding_tolerance": GRID_BINDING_TOLERANCE,
        "checks": checks,
        "recomputed_candidate_gate": recomputed_gate,
    }
    output = root / f"{PREFIX}_audit_check.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("Independent best-memory v4.2 audit failed: {}".format(failed))

    print("Regret-Aware Best-So-Far Memory v4.2 independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
