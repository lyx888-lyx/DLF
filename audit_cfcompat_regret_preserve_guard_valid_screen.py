"""Independent audit for Regret-Aware Beneficial-Teacher Guard v4.1."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_regret_preserve_guard_utils import (
    DEV_SEED,
    METHOD,
    RUN,
    RUNS,
    VERSION,
    STRONG_DISTILL_MARGIN,
    WEAK_DISTILL_SCALE,
    PRESERVE_MARGIN_V4P1,
    MILD_CFCOMPAT_BASE_V4P1,
    MILD_CFCOMPAT_SCALE_V4P1,
    dev_candidate_gate,
    frozen_thresholds,
    negative_transfer_summary,
)
from trains.singleTask.cfcompat_safe_projection_utils import derive_valid_events, group_summary, overall_from_events
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES

PREFIX = "regret_guard_v4p1"
TOL = 1e-9
GRID_TOL = 1e-6
PROJECTION_TOL = 1e-12


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def same_frame(actual, expected, keys, tolerance=TOL):
    left = actual.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(right[column]):
            if not np.allclose(left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float), atol=tolerance, rtol=0.0, equal_nan=True):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def nested_close(actual, expected, tolerance=TOL):
    if expected is None:
        return actual is None
    if isinstance(expected, dict):
        return isinstance(actual, dict) and set(actual) == set(expected) and all(nested_close(actual[k], expected[k], tolerance) for k in expected)
    if isinstance(expected, list):
        return isinstance(actual, list) and len(actual) == len(expected) and all(nested_close(a, b, tolerance) for a, b in zip(actual, expected))
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
    root = Path(parse_args().result_dir)
    files = {
        "grid": root / f"{PREFIX}_candidate_grid.csv",
        "epochs": root / f"{PREFIX}_all_epoch_metrics.csv",
        "candidate_raw": root / f"{PREFIX}_candidate_raw_valid_events.csv",
        "v2_grid": root / f"{PREFIX}_v2_reference_grid.csv",
        "v2_raw": root / f"{PREFIX}_v2_reference_raw_valid_events.csv",
        "v4_grid": root / f"{PREFIX}_v4_reference_grid.csv",
        "v4_raw": root / f"{PREFIX}_v4_reference_raw_valid_events.csv",
        "events": root / f"{PREFIX}_combined_valid_events.csv",
        "overall": root / f"{PREFIX}_overall_metrics.csv",
        "groups": root / f"{PREFIX}_group_metrics.csv",
        "transfer": root / f"{PREFIX}_transfer_metrics.csv",
        "train_baseline": root / f"{PREFIX}_train_baseline_cache.csv",
        "train_decisions": root / f"{PREFIX}_train_decisions.csv",
        "manifest": root / f"{PREFIX}_source_manifest.json",
        "summary": root / f"{PREFIX}_valid_screen_summary.json",
        "report": root / f"{PREFIX}_valid_screen_report.md",
    }
    missing = [str(p) for p in files.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError("Missing v4.1 audit artifacts:\n" + "\n".join(missing))

    grid = pd.read_csv(files["grid"])
    epochs = pd.read_csv(files["epochs"])
    candidate_raw = pd.read_csv(files["candidate_raw"])
    v2_grid = pd.read_csv(files["v2_grid"])
    v2_raw = pd.read_csv(files["v2_raw"])
    v4_grid = pd.read_csv(files["v4_grid"])
    v4_raw = pd.read_csv(files["v4_raw"])
    recorded_events = pd.read_csv(files["events"])
    recorded_overall = pd.read_csv(files["overall"])
    recorded_groups = pd.read_csv(files["groups"])
    recorded_transfer = pd.read_csv(files["transfer"])
    train_baseline = pd.read_csv(files["train_baseline"])
    train_decisions = pd.read_csv(files["train_decisions"])
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
        and protocol["reference_runs"] == ["cfcompat_replay", "student_safe_uniform", "regret_preserve_cfcompat"]
        and protocol["routing_policy"] == "strong_weak_distill_beneficial_pause_preserve_abstain"
        and protocol["weak_kd_normalization"] == "eligible_unscaled_cfcompat_mass_denominator_effective_scaled_numerator"
        and protocol["update_epochs"] == 10
        and not protocol["official_test_constructed"] and not protocol["official_test_authorized"]
        and nested_close(protocol["frozen_thresholds"], frozen_thresholds())
    )

    artifact_ok = True
    for name, meta in manifest["artifacts"].items():
        p = Path(meta["path"])
        artifact_ok = bool(artifact_ok and p.is_file() and p.resolve() == (root / name).resolve() and checkpoint_sha256(p) == meta["sha256"])
    checks["artifact_hashes"] = artifact_ok

    ref_hashes = True
    for group in ("v2_reference_sources", "v4_reference_sources"):
        for meta in manifest[group].values():
            p = Path(meta["path"])
            ref_hashes = bool(ref_hashes and p.is_file() and checkpoint_sha256(p) == meta["sha256"])
    checks["frozen_reference_hashes"] = ref_hashes

    source_ok = True
    source = manifest["source_record"]
    for path_key, sha_key in (
        ("checkpoint", "checkpoint_sha256"), ("teacher_checkpoint", "teacher_sha256"),
        ("evaluator_checkpoint", "evaluator_sha256"), ("evaluator_source", "evaluator_source_sha256"),
        ("compatibility_cache", "compatibility_cache_sha256"), ("compatibility_config", "compatibility_config_sha256"),
        ("train_baseline_cache", "train_baseline_cache_sha256"), ("train_decision_records", "train_decision_records_sha256"),
    ):
        p = Path(source[path_key])
        source_ok = bool(source_ok and p.is_file() and checkpoint_sha256(p) == source[sha_key])
    checks["source_and_checkpoint_binding"] = source_ok

    checks["single_seed_single_new_trajectory"] = bool(
        len(grid) == 1 and int(grid.iloc[0].Seed) == DEV_SEED and str(grid.iloc[0].Run) == RUN
        and set(epochs.Seed.astype(int)) == {DEV_SEED} and set(epochs.Run.astype(str)) == {RUN}
    )
    checks["reference_grid_shapes"] = bool(
        len(v2_grid) == 2 and set(v2_grid.Run.astype(str)) == {"cfcompat_replay", "student_safe_uniform"}
        and len(v4_grid) == 1 and str(v4_grid.iloc[0].Run) == "regret_preserve_cfcompat"
    )

    # Exact copies from already audited reference artifacts.
    original_v2_grid = pd.read_csv(Path(manifest["v2_reference_sources"]["grid"]["path"]))
    original_v2_grid = original_v2_grid.loc[
        original_v2_grid.Seed.astype(int).eq(DEV_SEED)
        & original_v2_grid.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    original_v2_raw = pd.read_csv(Path(manifest["v2_reference_sources"]["raw"]["path"]))
    original_v2_raw = original_v2_raw.loc[
        original_v2_raw.Seed.astype(int).eq(DEV_SEED)
        & original_v2_raw.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    checks["v2_reference_exact_copy"] = bool(
        same_frame(v2_grid, original_v2_grid, ["Seed", "Run"], 1e-12)
        and same_frame(v2_raw, original_v2_raw, ["Seed", "Run", "Mode", "sample_index"], 1e-12)
    )
    original_v4_grid = pd.read_csv(Path(manifest["v4_reference_sources"]["grid"]["path"]))
    original_v4_raw = pd.read_csv(Path(manifest["v4_reference_sources"]["raw"]["path"]))
    checks["v4_reference_exact_copy"] = bool(
        same_frame(v4_grid, original_v4_grid, ["Seed", "Run"], 1e-12)
        and same_frame(v4_raw, original_v4_raw, ["Seed", "Run", "Mode", "sample_index"], 1e-12)
    )

    checks["valid_only_outputs"] = bool(
        set(candidate_raw.Split.astype(str)) == {"valid"} and set(v2_raw.Split.astype(str)) == {"valid"}
        and set(v4_raw.Split.astype(str)) == {"valid"} and not grid.TestConstructed.astype(bool).any()
    )
    checks["complete_valid_prediction_grid"] = bool(
        len(candidate_raw) == 4 * 229 and len(v2_raw) == 2 * 4 * 229 and len(v4_raw) == 4 * 229
        and candidate_raw.sample_index.nunique() == 229
        and not candidate_raw.duplicated(["Seed", "Run", "Mode", "sample_index"]).any()
    )

    combined_raw = pd.concat([v2_raw, v4_raw, candidate_raw], ignore_index=True, sort=False)
    events = derive_valid_events(combined_raw)
    overall = overall_from_events(events)
    groups = group_summary(events)
    transfer = negative_transfer_summary(events)
    checks["valid_events_recomputed"] = same_frame(recorded_events, events, ["Seed", "Run", "Mode", "sample_index"])
    checks["overall_metrics_recomputed"] = same_frame(recorded_overall, overall, ["Seed", "Run", "Mode"])
    checks["group_metrics_recomputed"] = same_frame(recorded_groups, groups, ["Seed", "Run", "GroupType", "GroupValue"])
    checks["transfer_metrics_recomputed"] = same_frame(recorded_transfer, transfer, ["Seed", "Run", "Mode"])

    local = overall.loc[(overall.Seed.astype(int) == DEV_SEED) & (overall.Run.astype(str) == RUN)].set_index("Mode")
    diffs = [abs(float(grid.iloc[0].J_valid) - float(local.loc["J", "candidate_MAE"]))]
    for mode in ("LAV",) + MISSING_MODES:
        diffs.append(abs(float(grid.iloc[0]["valid_{}_MAE".format(mode)]) - float(local.loc[mode, "candidate_MAE"])))
    checks["grid_metrics_bound_to_predictions"] = bool(all(x <= GRID_TOL for x in diffs))

    required_baseline = ["sample_index", "sample_id", "label"] + ["baseline_{}_pred".format(mode) for mode in MISSING_MODES]
    checks["train_baseline_cache_complete"] = bool(
        list(train_baseline.columns) == required_baseline and len(train_baseline) == 1284
        and train_baseline.sample_index.nunique() == 1284
    )

    expected_count = 1284 * int(grid.iloc[0].TrainEpochCount)
    decision_ok = bool(
        len(train_decisions) == expected_count
        and set(train_decisions["mode"].astype(str)) == set(MISSING_MODES)
        and int(train_decisions.Epoch.min()) == 1
        and int(train_decisions.Epoch.max()) == int(grid.iloc[0].TrainEpochCount)
        and all(int((train_decisions.Epoch.astype(int) == e).sum()) == 1284 for e in range(1, int(grid.iloc[0].TrainEpochCount) + 1))
    )
    if decision_ok:
        cache = train_baseline.set_index("sample_index")
        student = train_decisions.student_prediction.to_numpy(dtype=float)
        teacher = train_decisions.teacher_prediction.to_numpy(dtype=float)
        baseline = train_decisions.baseline_prediction.to_numpy(dtype=float)
        label = train_decisions.label.to_numpy(dtype=float)
        compat = train_decisions.compatibility.to_numpy(dtype=float)
        lower, upper = np.minimum(student, label), np.maximum(student, label)
        teacher_safe = np.maximum(np.minimum(teacher, upper), lower)
        preserve_safe = np.maximum(np.minimum(baseline, upper), lower)
        baseline_error = np.abs(baseline - label)
        teacher_error = np.abs(teacher - label)
        current_error = np.abs(student - label)
        advantage = baseline_error - teacher_error
        regret = current_error - baseline_error
        direction = (teacher - baseline) * (label - baseline) > 0.0
        active = ~np.isclose(teacher_safe, student, atol=PROJECTION_TOL, rtol=0.0)
        better = advantage > 0.0
        strong_candidate = advantage >= STRONG_DISTILL_MARGIN
        weak_candidate = better & (advantage < STRONG_DISTILL_MARGIN) & direction
        strong = strong_candidate & active
        weak = weak_candidate & active
        pause = better & (~active)
        regressed = regret >= PRESERVE_MARGIN_V4P1
        preserve = (~better) & regressed
        abstain = ~(strong | weak | pause | preserve)
        mild = MILD_CFCOMPAT_BASE_V4P1 + MILD_CFCOMPAT_SCALE_V4P1 * compat
        strong_gate = strong.astype(float) * mild
        weak_gate = weak.astype(float) * mild
        eligible = strong_gate + weak_gate
        effective = strong_gate + WEAK_DISTILL_SCALE * weak_gate
        preserve_gate = preserve.astype(float)
        bound_baseline = np.asarray([
            float(cache.loc[int(index), "baseline_{}_pred".format(mode)])
            for index, mode in zip(train_decisions.sample_index.astype(int), train_decisions["mode"].astype(str))
        ])
        comparisons = [
            np.allclose(baseline, bound_baseline, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.teacher_safe_target, teacher_safe, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.preserve_safe_target, preserve_safe, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.teacher_advantage_vs_baseline, advantage, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.current_regret_vs_baseline, regret, atol=1e-7, rtol=0.0),
            np.array_equal(train_decisions.strong_distill.astype(bool).to_numpy(), strong),
            np.array_equal(train_decisions.weak_distill.astype(bool).to_numpy(), weak),
            np.array_equal(train_decisions.beneficial_pause.astype(bool).to_numpy(), pause),
            np.array_equal(train_decisions.preserve.astype(bool).to_numpy(), preserve),
            np.array_equal(train_decisions.guard_abstain.astype(bool).to_numpy(), abstain),
            np.allclose(train_decisions.strong_gate, strong_gate, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.weak_gate, weak_gate, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.eligible_distill_mass, eligible, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.effective_distill_gate, effective, atol=1e-7, rtol=0.0),
            np.allclose(train_decisions.preserve_gate, preserve_gate, atol=1e-7, rtol=0.0),
        ]
        decision_ok = bool(all(comparisons))
    checks["guard_train_decisions_recomputed"] = decision_ok

    aggregate_ok = False
    if len(train_decisions):
        values = {
            "strong_distill": float(train_decisions.strong_distill.astype(bool).mean()),
            "weak_distill": float(train_decisions.weak_distill.astype(bool).mean()),
            "beneficial_pause": float(train_decisions.beneficial_pause.astype(bool).mean()),
            "preserve": float(train_decisions.preserve.astype(bool).mean()),
            "guard_abstain": float(train_decisions.guard_abstain.astype(bool).mean()),
        }
        aggregate_ok = bool(
            abs(float(grid.iloc[0].projection_strong_distill_fraction) - values["strong_distill"]) <= TOL
            and abs(float(grid.iloc[0].projection_weak_distill_fraction) - values["weak_distill"]) <= TOL
            and abs(float(grid.iloc[0].projection_beneficial_pause_fraction) - values["beneficial_pause"]) <= TOL
            and abs(float(grid.iloc[0].projection_preserve_fraction) - values["preserve"]) <= TOL
            and abs(float(grid.iloc[0].projection_guard_abstain_fraction) - values["guard_abstain"]) <= TOL
            and abs(sum(values.values()) - 1.0) <= 1e-12
        )
    checks["guard_aggregate_accounting"] = aggregate_ok

    recomputed_gate = dev_candidate_gate(grid.iloc[0].to_dict(), v4_grid, v2_grid, groups, transfer, epochs.to_dict("records"))
    checks["candidate_gate_recomputed"] = nested_close(recomputed_gate, summary["candidate_gate"])
    verdict = "PROMOTE_REGRET_GUARD_V4P1_TO_3SEED_VALID_SCREEN" if recomputed_gate["passed"] else "STOP_REGRET_GUARD_V4P1_SINGLE_SEED_DEV_FAILED"
    checks["verdict_recomputed"] = summary["verdict"] == verdict

    numeric_cols = ["student_prediction", "baseline_prediction", "teacher_prediction", "label", "compatibility", "strong_gate", "weak_gate", "effective_distill_gate", "preserve_gate"]
    checks["finite_metrics"] = bool(
        np.isfinite(grid[["J_valid", "valid_LAV_MAE", "valid_LA_MAE", "valid_LV_MAE", "valid_L_MAE"]].to_numpy(dtype=float)).all()
        and np.isfinite(train_decisions[numeric_cols].to_numpy(dtype=float)).all()
    )
    checks["no_test_named_artifacts"] = not any("test" in p.name.lower() for p in root.iterdir())

    passed = bool(all(checks.values()))
    payload = {"version": VERSION, "passed": passed, "verdict": verdict, "checks": checks, "recomputed_candidate_gate": recomputed_gate}
    output = root / f"{PREFIX}_audit_check.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("Independent v4.1 audit failed: {}".format([k for k, v in checks.items() if not v]))
    print("Regret-Aware Beneficial-Teacher Guard v4.1 independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
