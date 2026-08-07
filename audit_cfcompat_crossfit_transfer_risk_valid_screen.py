"""Independent audit for Cross-Fitted Transfer-Risk CFCompatKD v5."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_crossfit_transfer_risk_utils import (
    DEV_SEED,
    METHOD,
    RUN,
    RUNS,
    VERSION,
    dev_candidate_gate,
    fit_crossfit_transfer_risk_gate,
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


PREFIX = "crossfit_transfer_risk_v5"
TOL = 1e-7


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
            if not np.allclose(
                left[column].to_numpy(dtype=float), right[column].to_numpy(dtype=float),
                atol=tolerance, rtol=0.0, equal_nan=True,
            ):
                return False
        elif not left[column].astype(str).equals(right[column].astype(str)):
            return False
    return True


def nested_close(actual, expected, tolerance=TOL):
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


def raw_feature_columns():
    return [
        "sample_index", "sample_id", "mode", "split", "label",
        "baseline_missing_prediction", "baseline_full_prediction",
        "teacher_full_prediction", "initial_student_missing_prediction",
    ]


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
        "events": root / f"{PREFIX}_combined_valid_events.csv",
        "overall": root / f"{PREFIX}_overall_metrics.csv",
        "groups": root / f"{PREFIX}_group_metrics.csv",
        "transfer": root / f"{PREFIX}_transfer_metrics.csv",
        "decisions": root / f"{PREFIX}_train_decisions.csv",
        "gate_train": root / f"{PREFIX}_train_oof_gate.csv",
        "gate_valid": root / f"{PREFIX}_valid_gate_diagnostic.csv",
        "gate_summary": root / f"{PREFIX}_gate_summary.json",
        "manifest": root / f"{PREFIX}_source_manifest.json",
        "summary": root / f"{PREFIX}_valid_screen_summary.json",
        "report": root / f"{PREFIX}_valid_screen_report.md",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing v5 audit artifacts:\n" + "\n".join(missing))

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
    decisions = pd.read_csv(files["decisions"])
    gate_train = pd.read_csv(files["gate_train"])
    gate_valid = pd.read_csv(files["gate_valid"])
    gate_summary = json.loads(files["gate_summary"].read_text(encoding="utf-8"))
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
        and not manifest["official_test_constructed"]
        and manifest["test_loader_construction_count"] == 0
        and protocol["development_seed"] == DEV_SEED
        and protocol["new_trajectories_trained"] == 1
        and tuple(protocol["runs"]) == RUNS
        and protocol["decision_split"] == "official_valid_only"
        and protocol["train_routing_probability_source"] == "grouped_out_of_fold_only"
        and not protocol["label_used_as_gate_feature"]
        and not protocol["official_test_constructed"]
        and not protocol["official_test_authorized"]
        and nested_close(protocol["frozen_thresholds"], frozen_thresholds())
    )

    artifact_hashes = True
    for metadata in manifest["artifacts"].values():
        path = Path(metadata["path"])
        artifact_hashes = bool(
            artifact_hashes and path.is_file() and checkpoint_sha256(path) == metadata["sha256"]
        )
    for metadata in manifest["gate_artifacts"].values():
        path = Path(metadata["path"])
        artifact_hashes = bool(
            artifact_hashes and path.is_file() and checkpoint_sha256(path) == metadata["sha256"]
        )
    checks["artifact_hashes"] = artifact_hashes

    reference_hashes = True
    for group in ("v2_reference_sources", "v4_reference_sources"):
        for metadata in manifest[group].values():
            path = Path(metadata["path"])
            reference_hashes = bool(
                reference_hashes and path.is_file() and checkpoint_sha256(path) == metadata["sha256"]
            )
    checks["frozen_reference_hashes"] = reference_hashes

    checks["single_seed_single_new_trajectory"] = bool(
        len(grid) == 1 and int(grid.iloc[0].Seed) == DEV_SEED and str(grid.iloc[0].Run) == RUN
        and set(epochs.Seed.astype(int)) == {DEV_SEED}
        and set(epochs.Run.astype(str)) == {RUN}
    )
    checks["reference_grid_shapes"] = bool(
        len(v2_grid) == 2 and set(v2_grid.Run.astype(str)) == {"cfcompat_replay", "student_safe_uniform"}
        and len(v4_grid) == 1 and str(v4_grid.iloc[0].Run) == "regret_preserve_cfcompat"
    )

    ref_copy_ok = True
    v2_grid_source = pd.read_csv(manifest["v2_reference_sources"]["grid"]["path"])
    v2_grid_source = v2_grid_source.loc[
        v2_grid_source.Seed.astype(int).eq(DEV_SEED)
        & v2_grid_source.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    v2_raw_source = pd.read_csv(manifest["v2_reference_sources"]["raw"]["path"])
    v2_raw_source = v2_raw_source.loc[
        v2_raw_source.Seed.astype(int).eq(DEV_SEED)
        & v2_raw_source.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    v4_grid_source = pd.read_csv(manifest["v4_reference_sources"]["grid"]["path"])
    v4_raw_source = pd.read_csv(manifest["v4_reference_sources"]["raw"]["path"])
    ref_copy_ok = bool(
        same_frame(v2_grid, v2_grid_source, ["Seed", "Run"])
        and same_frame(v2_raw, v2_raw_source, ["Seed", "Run", "Mode", "sample_index"])
        and same_frame(v4_grid, v4_grid_source, ["Seed", "Run"])
        and same_frame(v4_raw, v4_raw_source, ["Seed", "Run", "Mode", "sample_index"])
    )
    checks["reference_artifacts_exact_copy"] = ref_copy_ok

    # Refit the grouped cross-fit gate from only the frozen raw predictors.
    train_raw = gate_train.loc[:, raw_feature_columns()].copy()
    valid_raw = gate_valid.loc[:, raw_feature_columns()].copy()
    recomputed_train, recomputed_valid, recomputed_gate_summary = fit_crossfit_transfer_risk_gate(
        train_raw, valid_raw
    )
    gate_probability_ok = bool(
        np.allclose(
            gate_train.oof_benefit_probability.to_numpy(float),
            recomputed_train.oof_benefit_probability.to_numpy(float),
            atol=TOL, rtol=0.0,
        )
        and np.array_equal(
            gate_train.crossfit_fold.to_numpy(int), recomputed_train.crossfit_fold.to_numpy(int)
        )
        and np.allclose(
            gate_valid.full_train_benefit_probability.to_numpy(float),
            recomputed_valid.full_train_benefit_probability.to_numpy(float),
            atol=TOL, rtol=0.0,
        )
    )
    checks["grouped_crossfit_gate_recomputed"] = gate_probability_ok
    recorded_gate_core = dict(gate_summary)
    for key in (
        "test_constructed", "train_labels_used_only_for_gate_supervision",
        "train_routing_probability_source", "valid_gate_probability_source", "frozen_thresholds",
    ):
        recorded_gate_core.pop(key, None)
    checks["gate_metrics_recomputed"] = nested_close(recorded_gate_core, recomputed_gate_summary)
    checks["gate_prescreen_recomputed"] = bool(
        gate_summary["prescreen_passed"] == recomputed_gate_summary["prescreen_passed"]
        and nested_close(gate_summary["prescreen_checks"], recomputed_gate_summary["prescreen_checks"])
    )
    checks["group_fold_isolation"] = bool(
        int(gate_train.groupby("sample_index").crossfit_fold.nunique().max()) == 1
        and gate_train.sample_index.nunique() == 1284
        and len(gate_train) == 1284 * len(MISSING_MODES)
    )

    checks["valid_only_outputs"] = bool(
        set(candidate_raw.Split.astype(str)) == {"valid"}
        and set(candidate_raw.SelectedBy.astype(str)) == {"validation_J"}
        and not grid.TestConstructed.astype(bool).any()
        and set(gate_valid.split.astype(str)) == {"valid"}
        and set(gate_train.split.astype(str)) == {"train"}
    )
    checks["complete_valid_prediction_grid"] = bool(
        len(candidate_raw) == 4 * 229 and candidate_raw.sample_index.nunique() == 229
        and set(candidate_raw.Mode.astype(str)) == {"LAV", "LA", "LV", "L"}
        and not candidate_raw.duplicated(["Seed", "Run", "Mode", "sample_index"]).any()
    )

    combined_raw = pd.concat([v2_raw, v4_raw, candidate_raw], ignore_index=True, sort=False)
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
    checks["grid_metrics_bound_to_predictions"] = bool(all(value <= 1e-6 for value in binding))

    expected_decisions = 1284 * int(grid.iloc[0].TrainEpochCount)
    decision_ok = bool(
        len(decisions) == expected_decisions
        and int(decisions.event_ordinal.min()) == 1
        and int(decisions.event_ordinal.max()) == expected_decisions
    )
    gate_index = gate_train.set_index(["sample_index", "mode"])
    if decision_ok:
        for row in decisions.itertuples(index=False):
            key = (int(row.sample_index), str(row.mode))
            if key not in gate_index.index:
                decision_ok = False
                break
            gate_row = gate_index.loc[key]
            probability = float(gate_row.oof_benefit_probability)
            risk_weight = max(2.0 * probability - 1.0, 0.0)
            student = float(row.student_prediction)
            teacher = float(row.teacher_prediction)
            label = float(row.label)
            lower, upper = min(student, label), max(student, label)
            safe = max(min(teacher, upper), lower)
            active = not np.isclose(safe, student, atol=1e-12, rtol=0.0)
            effective = risk_weight if active else 0.0
            comparisons = (
                int(row.crossfit_fold) == int(gate_row.crossfit_fold),
                abs(float(row.oof_benefit_probability) - probability) <= TOL,
                abs(float(row.risk_weight) - risk_weight) <= TOL,
                bool(row.safe_teacher_active) == active,
                abs(float(row.teacher_safe_target) - safe) <= TOL,
                abs(float(row.eligible_distill_mass) - (1.0 if active else 0.0)) <= TOL,
                abs(float(row.effective_gate) - effective) <= TOL,
            )
            if not all(comparisons):
                decision_ok = False
                break
    checks["train_oof_routing_recomputed"] = decision_ok

    recomputed_gate = dev_candidate_gate(
        grid.iloc[0].to_dict(), v2_grid, v4_grid,
        recomputed_groups, recomputed_transfer, epochs.to_dict("records"), recomputed_gate_summary,
    )
    checks["candidate_gate_recomputed"] = nested_close(summary["candidate_gate"], recomputed_gate)
    expected_verdict = (
        "PROMOTE_CROSSFIT_TRANSFER_RISK_V5_TO_3SEED_VALID_SCREEN"
        if recomputed_gate["passed"]
        else "STOP_CROSSFIT_TRANSFER_RISK_V5_SINGLE_SEED_DEV_FAILED"
    )
    checks["verdict_recomputed"] = bool(summary["verdict"] == expected_verdict)

    numeric_frames = [grid, epochs, candidate_raw, recorded_overall, recorded_groups, recorded_transfer, decisions]
    finite_ok = True
    for frame in numeric_frames:
        numeric = frame.select_dtypes(include=[np.number])
        finite_ok = bool(finite_ok and np.isfinite(numeric.to_numpy(dtype=np.float64)).all())
    checks["finite_metrics"] = finite_ok
    checks["no_test_named_artifacts"] = bool(
        not any("test" in path.name.lower() for path in root.rglob("*"))
        and not manifest["official_test_constructed"]
    )

    passed = bool(all(checks.values()))
    payload = {
        "passed": passed,
        "version": VERSION,
        "method": METHOD,
        "checks": checks,
        "verdict": summary["verdict"],
    }
    output = root / f"{PREFIX}_audit_check.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("Cross-Fitted Transfer-Risk v5 independent audit")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])
    print("official Test was not constructed")
    if not passed:
        raise RuntimeError("Independent v5 audit failed: {}".format([key for key, value in checks.items() if not value]))


if __name__ == "__main__":
    main()
