"""Independent audit for the CFCompatKD teacher-consensus Valid screen."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.cfcompat_stability_utils import expected_missing_sequence_sha
from trains.singleTask.cfcompat_teacher_consensus_utils import (
    BASELINE_RUN,
    CANDIDATE_RUNS,
    CONSENSUS_RUN,
    FORMAL_SEEDS,
    MEAN_RUN,
    METHOD,
    RUNS,
    TEACHER_SEEDS,
    VERSION,
    aggregate_candidate_gate,
    load_consensus_cache,
    replay_gate,
    verdict_from_gates,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def nested_close(actual, expected, tolerance=1e-10):
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            nested_close(actual[key], expected[key], tolerance) for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            nested_close(left, right, tolerance) for left, right in zip(actual, expected)
        )
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, int):
        return int(actual) == int(expected)
    if isinstance(expected, float):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return abs(float(actual) - float(expected)) <= tolerance
    return actual == expected


def bool_series(series):
    return series.map(lambda value: str(value).strip().lower() == "true")


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    required = {
        "grid": root / "teacher_consensus_valid_grid_summary.csv",
        "epochs": root / "teacher_consensus_all_epoch_metrics.csv",
        "predictions": root / "teacher_consensus_all_valid_predictions.csv",
        "manifest": root / "teacher_consensus_source_manifest.json",
        "summary": root / "teacher_consensus_valid_screen_summary.json",
        "report": root / "teacher_consensus_valid_screen_report.md",
        "cache": root / "teacher_consensus_train_cache" / "mosi_train_teacher_consensus.csv",
        "cache_config": root
        / "teacher_consensus_train_cache"
        / "mosi_train_teacher_consensus_config.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing teacher-consensus artifacts:\n" + "\n".join(missing)
        )
    grid = pd.read_csv(required["grid"])
    epochs = pd.read_csv(required["epochs"])
    predictions = pd.read_csv(required["predictions"])
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    checks = {}

    checks["version_and_method"] = bool(
        manifest["version"] == VERSION
        and manifest["method"] == METHOD
        and summary["version"] == VERSION
        and summary["method"] == METHOD
    )
    checks["frozen_protocol"] = bool(
        tuple(manifest["formal_seeds"]) == FORMAL_SEEDS
        and tuple(manifest["teacher_seeds"]) == TEACHER_SEEDS
        and tuple(manifest["runs"]) == RUNS
        and manifest["candidate_selection"]
        == "none_each_candidate_has_an_independent_gate"
        and tuple(summary["protocol"]["formal_seeds"]) == FORMAL_SEEDS
        and tuple(summary["protocol"]["teacher_seeds"]) == TEACHER_SEEDS
        and tuple(summary["protocol"]["candidate_runs"]) == CANDIDATE_RUNS
        and summary["protocol"]["candidate_selection"] == "none"
        and summary["protocol"]["base_optimizer"] == "Adam"
        and int(summary["protocol"]["update_epochs"]) == 10
        and summary["protocol"]["teacher_cache_source"] == "official_train_only"
        and summary["protocol"]["official_test_constructed"] is False
        and summary["protocol"]["official_test_authorized"] is False
        and int(summary["protocol"]["additional_inference_parameters"]) == 0
        and manifest["test_constructed"] is False
        and int(manifest["test_loader_construction_count"]) == 0
        and int(manifest["test_loader_traversal_count"]) == 0
        and int(manifest["additional_inference_parameters"]) == 0
    )

    artifact_hashes_ok = True
    for name, metadata in manifest["artifacts"].items():
        path = Path(metadata["path"])
        artifact_hashes_ok = bool(
            artifact_hashes_ok
            and path.is_file()
            and path.resolve() == (root / name).resolve()
            and checkpoint_sha256(path) == metadata["sha256"]
        )
    cache_metadata = manifest["teacher_consensus_cache"]
    artifact_hashes_ok = bool(
        artifact_hashes_ok
        and Path(cache_metadata["csv"]).resolve() == required["cache"].resolve()
        and Path(cache_metadata["config"]).resolve()
        == required["cache_config"].resolve()
        and checkpoint_sha256(required["cache"]) == cache_metadata["csv_sha256"]
        and checkpoint_sha256(required["cache_config"])
        == cache_metadata["config_sha256"]
    )
    checks["artifact_hashes"] = artifact_hashes_ok

    source_ok = True
    for record in manifest["baseline_sources"]:
        path = Path(record["path"])
        source_ok = bool(
            source_ok
            and path.is_file()
            and checkpoint_sha256(path) == record["sha256"]
        )
    for record in manifest["checkpoints"]:
        path = Path(record["path"])
        source_ok = bool(
            source_ok
            and path.is_file()
            and checkpoint_sha256(path) == record["sha256"]
        )
    checks["source_and_checkpoint_binding"] = source_ok

    cache_frame, _, cache_config = load_consensus_cache(
        {
            "csv": required["cache"],
            "config": required["cache_config"],
        }
    )
    checks["teacher_consensus_cache_recomputed"] = bool(
        len(cache_frame) == 1284
        and tuple(cache_config["teacher_seeds"]) == TEACHER_SEEDS
        and int(summary["teacher_consensus"]["sample_count"]) == 1284
        and math.isclose(
            float(summary["teacher_consensus"]["variance_median"]),
            float(cache_config["variance_median"]),
            rel_tol=0.0,
            abs_tol=1e-10,
        )
        and summary["teacher_consensus"]["weight_formula"]
        == cache_config["consensus_weight_formula"]
    )

    expected_grid = {(seed, run) for seed in FORMAL_SEEDS for run in RUNS}
    observed_grid = {
        (int(row.Seed), str(row.Run)) for row in grid.itertuples(index=False)
    }
    checks["complete_six_run_grid"] = bool(
        observed_grid == expected_grid and len(grid) == len(expected_grid)
    )
    checks["valid_only_outputs"] = bool(
        set(predictions.Split.astype(str)) == {"valid"}
        and {
            (int(row.Seed), str(row.Run))
            for row in predictions[["Seed", "Run"]]
            .drop_duplicates()
            .itertuples(index=False)
        }
        == expected_grid
        and not any(column.lower().startswith("test_") for column in grid.columns)
        and not any(column.lower().startswith("test_") for column in epochs.columns)
        and not any(column.lower().startswith("test_") for column in predictions.columns)
        and not bool_series(grid.TestConstructed).any()
    )
    checks["valid_prediction_binding"] = bool(
        all(
            len(local) == 229
            and local.sample_index.astype(int).nunique() == 229
            and set(local.sample_index.astype(int)) == set(range(229))
            for _, local in predictions.groupby(["Seed", "Run"], sort=True)
        )
    )

    run_integrity = True
    for row in grid.itertuples(index=False):
        batch_sizes = [int(value) for value in json.loads(row.BatchSizeSequenceJSON)]
        expected_sha, expected_count = expected_missing_sequence_sha(
            int(row.Seed), int(row.TrainEpochCount), batch_sizes
        )
        run_integrity = bool(
            run_integrity
            and row.MissingSequenceSHA256 == expected_sha
            and int(row.MissingSequenceCount) == int(expected_count)
            and int(row.FrozenTeacherGradientCount) == 0
            and row.TeacherConsensusCacheSHA256
            == checkpoint_sha256(required["cache"])
            and math.isclose(
                float(row.TeacherVarianceMedian),
                float(cache_config["variance_median"]),
                rel_tol=0.0,
                abs_tol=1e-10,
            )
        )
        if str(row.Run) == BASELINE_RUN:
            run_integrity = bool(
                run_integrity
                and str(row.LiveTeacherUsedDuringTraining).lower() == "true"
                and int(row.TeacherEnsembleSeedCount) == 1
                and row.TeacherTarget == "single_seed_teacher"
                and row.DistillationGate == "counterfactual_compatibility"
            )
        elif str(row.Run) == MEAN_RUN:
            run_integrity = bool(
                run_integrity
                and str(row.LiveTeacherUsedDuringTraining).lower() == "false"
                and int(row.TeacherEnsembleSeedCount) == len(TEACHER_SEEDS)
                and row.TeacherTarget == "five_teacher_mean"
                and row.DistillationGate == "counterfactual_compatibility"
            )
        elif str(row.Run) == CONSENSUS_RUN:
            run_integrity = bool(
                run_integrity
                and str(row.LiveTeacherUsedDuringTraining).lower() == "false"
                and int(row.TeacherEnsembleSeedCount) == len(TEACHER_SEEDS)
                and row.TeacherTarget == "five_teacher_mean"
                and row.DistillationGate
                == "counterfactual_compatibility_times_teacher_consensus"
            )
    checks["run_objective_and_missing_sequence"] = run_integrity

    baseline_replay_checks = {}
    replay_ok = True
    for seed in FORMAL_SEEDS:
        row = grid.loc[
            grid.Seed.astype(int).eq(int(seed))
            & grid.Run.astype(str).eq(BASELINE_RUN)
        ].iloc[0].to_dict()
        reference = summary["baselines"][str(seed)]
        check = replay_gate(row, reference)
        baseline_replay_checks[str(seed)] = check
        replay_ok = bool(
            replay_ok
            and check["passed"]
            and nested_close(check, summary["baseline_replay_gates"][str(seed)])
        )
    checks["baseline_replays_exact"] = replay_ok

    grid_records = grid.to_dict("records")
    epoch_records = epochs.to_dict("records")
    recomputed_gates = {
        run: aggregate_candidate_gate(
            run,
            grid_records,
            {
                int(seed): summary["baselines"][str(seed)]
                for seed in FORMAL_SEEDS
            },
            epoch_records,
        )
        for run in CANDIDATE_RUNS
    }
    checks["candidate_gates_recomputed"] = nested_close(
        recomputed_gates, summary["candidate_gates"]
    )
    expected_verdict = verdict_from_gates(recomputed_gates)
    checks["verdict_recomputed"] = bool(summary["verdict"] == expected_verdict)

    checks["finite_metrics"] = bool(
        np.isfinite(
            grid.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
        and np.isfinite(
            epochs.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
        and np.isfinite(
            predictions.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
        ).all()
    )
    checks["no_test_named_artifacts"] = bool(
        not any("test" in path.name.lower() for path in root.rglob("*"))
    )

    passed = bool(all(checks.values()))
    payload = {
        "version": VERSION,
        "passed": passed,
        "verdict": summary["verdict"],
        "checks": checks,
        "recomputed_candidate_gates": recomputed_gates,
        "recomputed_baseline_replay_gates": baseline_replay_checks,
    }
    output = root / "teacher_consensus_audit_check.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(
            "CFCompatKD teacher-consensus independent audit failed: {}".format(failed)
        )
    print("CFCompatKD teacher-consensus independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
