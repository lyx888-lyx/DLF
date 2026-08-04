"""Independent audit for the frozen DLF tail-risk coupling analysis."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.dlf_role_specialization_utils import long_tail_status
from trains.singleTask.dlf_tail_risk_utils import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FORMAL_SEEDS,
    METHOD,
    MODES,
    VERSION,
    add_risk_events,
    compute_bin_and_run_metrics,
    coupling_gate,
    joint_video_bootstrap,
    sha256_file,
    tail_head_definition,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def nested_close(actual, expected, tolerance=1e-10):
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            nested_close(actual[key], expected[key], tolerance)
            for key in expected
        )
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(
            nested_close(left, right, tolerance)
            for left, right in zip(actual, expected)
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


def same_frame(actual: pd.DataFrame, expected: pd.DataFrame, keys, tolerance=1e-10):
    left = actual.sort_values(keys, kind="mergesort").reset_index(drop=True)
    right = expected.sort_values(keys, kind="mergesort").reset_index(drop=True)
    if list(left.columns) != list(right.columns) or len(left) != len(right):
        return False
    for column in left.columns:
        if pd.api.types.is_numeric_dtype(left[column]) and pd.api.types.is_numeric_dtype(
            right[column]
        ):
            a = left[column].to_numpy(dtype=float)
            b = right[column].to_numpy(dtype=float)
            if not np.allclose(a, b, atol=tolerance, rtol=0.0, equal_nan=True):
                return False
        else:
            if not left[column].astype(str).equals(right[column].astype(str)):
                return False
    return True


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    required = {
        "events": root / "tail_risk_sample_events.csv",
        "bins": root / "tail_risk_bin_metrics.csv",
        "runs": root / "tail_risk_run_summary.csv",
        "bootstrap": root / "tail_risk_video_bootstrap.csv",
        "summary": root / "tail_risk_coupling_summary.json",
        "manifest": root / "tail_risk_source_manifest.json",
        "report": root / "tail_risk_coupling_report.md",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing tail-risk audit artifacts:\n" + "\n".join(missing)
        )
    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    manifest = json.loads(required["manifest"].read_text(encoding="utf-8"))
    recorded_events = pd.read_csv(required["events"])
    recorded_bins = pd.read_csv(required["bins"])
    recorded_runs = pd.read_csv(required["runs"])
    recorded_bootstrap = pd.read_csv(required["bootstrap"])
    checks = {}

    checks["version_and_method"] = bool(
        summary["version"] == VERSION
        and summary["method"] == METHOD
        and manifest["version"] == VERSION
        and manifest["method"] == METHOD
    )
    checks["frozen_protocol"] = bool(
        tuple(manifest["formal_seeds"]) == FORMAL_SEEDS
        and tuple(manifest["modes"]) == MODES
        and int(manifest["bootstrap_replicates"]) == BOOTSTRAP_REPLICATES
        and int(manifest["bootstrap_seed"]) == BOOTSTRAP_SEED
        and not manifest["official_test_constructed"]
        and manifest["test_loader_construction_count"] == 0
        and manifest["test_loader_traversal_count"] == 0
        and not manifest["model_loaded"]
        and not manifest["optimizer_constructed"]
        and not manifest["backward_called"]
        and not manifest["model_parameters_updated"]
        and not summary["protocol"]["official_test_constructed"]
        and not summary["protocol"]["official_test_authorized"]
        and not summary["protocol"]["model_training_performed"]
    )

    artifact_hashes = True
    for name, metadata in manifest["artifacts"].items():
        path = Path(metadata["path"])
        artifact_hashes = bool(
            artifact_hashes
            and path.is_file()
            and path.resolve() == (root / name).resolve()
            and sha256_file(path) == metadata["sha256"]
        )
    checks["artifact_hashes"] = artifact_hashes

    source_binding = True
    for metadata in manifest["source_binding"].values():
        path = Path(metadata["path"])
        source_binding = bool(
            source_binding
            and path.is_file()
            and sha256_file(path) == metadata["sha256"]
        )
    checks["source_binding"] = source_binding

    source_paths = {
        key: Path(value["path"])
        for key, value in manifest["source_binding"].items()
    }
    source_audit = json.loads(
        source_paths["role_audit_v2"].read_text(encoding="utf-8")
    )
    source_summary = json.loads(
        source_paths["role_summary"].read_text(encoding="utf-8")
    )
    source_manifest = json.loads(
        source_paths["role_source"].read_text(encoding="utf-8")
    )
    checks["source_v2_audit_passed"] = bool(source_audit.get("passed", False))
    checks["source_test_forbidden"] = bool(
        not source_manifest["official_test_constructed"]
        and source_manifest["test_loader_construction_count"] == 0
        and source_manifest["test_loader_traversal_count"] == 0
        and not source_summary["protocol"]["official_test_constructed"]
    )

    predictions = pd.read_csv(source_paths["predictions"])
    distribution = pd.read_csv(source_paths["distribution"])
    samples = pd.read_csv(source_paths["samples"])
    checks["train_valid_source_only"] = bool(
        set(predictions.Split.astype(str)) == {"train", "valid"}
        and set(samples.Split.astype(str)) == {"train", "valid"}
        and "test" not in set(predictions.Split.astype(str))
        and "test" not in set(samples.Split.astype(str))
    )

    definition = tail_head_definition(distribution)
    recomputed_events = add_risk_events(predictions)
    recomputed_bins, recomputed_runs = compute_bin_and_run_metrics(
        recomputed_events, definition
    )
    recomputed_bootstrap = joint_video_bootstrap(
        recomputed_events,
        definition,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    long_tail = long_tail_status(distribution, "mosi")
    recomputed_gate = coupling_gate(
        recomputed_runs,
        recomputed_bootstrap,
        long_tail["tail_present"],
    )

    event_keys = ["Seed", "Mode", "sample_index"]
    bin_keys = ["Seed", "Mode", "sentiment_bin"]
    run_keys = ["Seed", "Mode"]
    bootstrap_keys = ["Replicate"]
    checks["sample_events_recomputed"] = same_frame(
        recorded_events, recomputed_events, event_keys
    )
    checks["bin_metrics_recomputed"] = same_frame(
        recorded_bins, recomputed_bins, bin_keys
    )
    checks["run_summary_recomputed"] = same_frame(
        recorded_runs, recomputed_runs, run_keys
    )
    checks["video_bootstrap_recomputed"] = same_frame(
        recorded_bootstrap, recomputed_bootstrap, bootstrap_keys
    )
    checks["frequency_definition_recomputed"] = nested_close(
        definition, summary["frequency_definition"]
    )
    checks["long_tail_recomputed"] = nested_close(
        long_tail, summary["long_tail"]
    )
    checks["coupling_gate_recomputed"] = nested_close(
        recomputed_gate, summary["coupling_gate"]
    )
    checks["verdict_recomputed"] = bool(
        summary["verdict"] == recomputed_gate["verdict"]
    )
    checks["complete_eight_run_grid"] = bool(
        {
            (int(row.Seed), str(row.Mode))
            for row in recorded_runs.itertuples(index=False)
        }
        == {(seed, mode) for seed in FORMAL_SEEDS for mode in MODES}
    )
    checks["no_test_named_artifacts"] = bool(
        not any("test" in path.name.lower() for path in root.iterdir())
    )

    passed = bool(all(checks.values()))
    payload = {
        "version": VERSION,
        "passed": passed,
        "verdict": summary["verdict"],
        "checks": checks,
    }
    (root / "tail_risk_audit_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError(
            "DLF tail-risk independent audit failed: {}".format(failed)
        )
    print("DLF tail-risk independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
