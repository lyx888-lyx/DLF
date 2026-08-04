"""Independent artifact audit for CFCompatKD video-aware sampling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES
from trains.singleTask.video_aware_sampling_utils import (
    METHOD,
    SAMPLES_PER_VIDEO,
    VERSION,
    candidate_test_gate,
    candidate_valid_gate,
)


MODES = ("LAV", "LA", "LV", "L")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def require_close(left, right, name, tolerance=1e-7):
    if abs(float(left) - float(right)) > tolerance:
        raise RuntimeError(
            "{} mismatch: {} vs {}".format(name, left, right)
        )


def metrics_from_predictions(frame):
    labels = frame["label"].to_numpy(dtype=float)
    return {
        mode: float(np.mean(np.abs(
            frame["{}_pred".format(mode)].to_numpy(dtype=float) - labels
        )))
        for mode in MODES
    }


def compare_nested(actual, expected, tolerance=1e-10):
    if isinstance(expected, dict):
        if set(actual) != set(expected):
            return False
        return all(
            compare_nested(actual[key], expected[key], tolerance)
            for key in expected
        )
    if isinstance(expected, bool):
        return actual is expected
    if isinstance(expected, (int, np.integer)):
        return int(actual) == int(expected)
    if isinstance(expected, (float, np.floating)):
        return abs(float(actual) - float(expected)) <= tolerance
    if expected is None:
        return actual is None
    return actual == expected


def expected_verdict(root, summary, valid_gate, test_gate):
    seed = int(summary["seed"])
    if not valid_gate["passed"]:
        return "STOP_VIDEO_AWARE_SAMPLING_VALID_GATE_FAILED"
    if test_gate is None or not test_gate["passed"]:
        return "STOP_VIDEO_AWARE_SAMPLING_TEST_GATE_FAILED"
    if seed == 1114:
        return "PROMOTE_SEED1114_RUN_SEED1111_FORMAL"

    other = root.parent / "seed1114" / "video_aware_sampling_summary.json"
    if other.is_file():
        payload = json.loads(other.read_text(encoding="utf-8"))
        other_passed = bool(
            payload.get("valid_gate", {}).get("passed")
            and payload.get("test_gate", {}).get("passed")
        )
    else:
        other_passed = False
    return (
        "PROMOTE_MOSI_TWO_SEEDS_RUN_MOSEI"
        if other_passed
        else "PROMOTE_SEED1111_WAIT_FOR_SEED1114"
    )


def main():
    args = parse_args()
    root = Path(args.result_dir)
    required = {
        "summary": root / "video_aware_sampling_summary.json",
        "source": root / "video_aware_sampling_source_manifest.json",
        "epochs": root / "video_aware_sampling_all_epoch_metrics.csv",
        "valid": root / "video_aware_sampling_all_valid_predictions.csv",
        "comparison": root / "video_aware_sampling_comparison.csv",
        "report": root / "video_aware_sampling_report.md",
        "domain": root / "domain_audit" / "video_domain_audit.json",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing video-aware sampling artifacts:\n" + "\n".join(missing)
        )

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    source = json.loads(required["source"].read_text(encoding="utf-8"))
    domain = json.loads(required["domain"].read_text(encoding="utf-8"))
    epochs = pd.read_csv(required["epochs"])
    valid = pd.read_csv(required["valid"])
    comparison = pd.read_csv(required["comparison"])

    checks = {}
    checks["version_and_method"] = (
        summary["version"] == VERSION
        and summary["method"] == METHOD
        and source["version"] == VERSION
        and source["method"] == METHOD
    )
    checks["baseline_result_hash"] = (
        checkpoint_sha256(Path(source["baseline_result"]))
        == source["baseline_result_sha256"]
    )
    checks["domain_audit_hash"] = (
        checkpoint_sha256(Path(source["domain_audit"]))
        == source["domain_audit_sha256"]
    )
    checks["candidate_checkpoint_hash"] = (
        checkpoint_sha256(Path(source["candidate_checkpoint"]))
        == source["candidate_checkpoint_sha256"]
    )
    checks["teacher_hash"] = (
        checkpoint_sha256(Path(source["teacher_checkpoint"]))
        == source["teacher_sha256"]
    )
    checks["evaluator_hash"] = (
        checkpoint_sha256(Path(source["evaluator_checkpoint"]))
        == source["evaluator_sha256"]
    )
    checks["cache_hash"] = (
        checkpoint_sha256(Path(source["compatibility_cache"]))
        == source["compatibility_cache_sha256"]
    )
    checks["video_domain_audit_passed"] = bool(
        domain["passed"]
        and domain["sampler_viable"]
        and domain["sampler_exact_coverage"]
        and domain["sampler_duplicate_count"] == 0
        and all(not values for values in domain["split_video_overlaps"].values())
    )

    baseline_frame = pd.read_csv(source["baseline_result"])
    baseline_rows = baseline_frame.loc[
        baseline_frame.Seed.astype(int).eq(int(summary["seed"]))
    ]
    if len(baseline_rows) != 1:
        raise RuntimeError("Baseline seed row is not unique.")
    baseline_row = baseline_rows.iloc[0]
    for key, value in summary["baseline"].items():
        if key in baseline_row and isinstance(value, (int, float)):
            require_close(baseline_row[key], value, "baseline {}".format(key))
    checks["baseline_binding_recomputed"] = True

    replay = summary["baseline_replay"]
    replay_run = str(replay["Run"])
    replay_valid = valid.loc[valid.Run.astype(str).eq(replay_run)].copy()
    if replay_valid.empty:
        raise RuntimeError("Baseline replay validation predictions are absent.")
    replay_mae = metrics_from_predictions(replay_valid)
    for mode in MODES:
        require_close(
            replay_mae[mode],
            replay["valid_{}_MAE".format(mode)],
            "replay valid {} MAE".format(mode),
        )
    replay_j = 0.5 * replay_mae["LAV"] + 0.5 * np.mean([
        replay_mae[mode] for mode in MISSING_MODES
    ])
    require_close(replay_j, replay["J_valid"], "replay Valid J")
    differences = {
        "J_valid": abs(float(replay["J_valid"]) - float(summary["baseline"]["J_valid"])),
    }
    for mode in MODES:
        key = "valid_{}_MAE".format(mode)
        differences[key] = abs(
            float(replay[key]) - float(summary["baseline"][key])
        )
    replay_gate = summary["baseline_replay_gate"]
    expected_replay_gate = {
        "passed": bool(
            int(replay["BestValidEpoch"])
            == int(summary["baseline"]["BestValidEpoch"])
            and all(
                value <= float(replay_gate["tolerance"])
                for value in differences.values()
            )
        ),
        "epoch_match": bool(
            int(replay["BestValidEpoch"])
            == int(summary["baseline"]["BestValidEpoch"])
        ),
        "tolerance": float(replay_gate["tolerance"]),
        "differences": differences,
    }
    checks["baseline_replay_gate_recomputed"] = bool(
        compare_nested(replay_gate, expected_replay_gate)
        and replay_gate["passed"]
    )

    candidate = summary["candidate"]
    candidate_run = str(candidate["Run"])
    candidate_valid = valid.loc[valid.Run.astype(str).eq(candidate_run)].copy()
    if candidate_valid.empty:
        raise RuntimeError("Candidate validation predictions are absent.")
    candidate_mae = metrics_from_predictions(candidate_valid)
    for mode in MODES:
        require_close(
            candidate_mae[mode],
            candidate["valid_{}_MAE".format(mode)],
            "candidate valid {} MAE".format(mode),
        )
    candidate_j = 0.5 * candidate_mae["LAV"] + 0.5 * np.mean([
        candidate_mae[mode] for mode in MISSING_MODES
    ])
    require_close(candidate_j, candidate["J_valid"], "candidate Valid J")
    checks["candidate_valid_metrics_recomputed"] = True

    candidate_epochs = epochs.loc[epochs.Run.astype(str).eq(candidate_run)]
    valid_gate = candidate_valid_gate(
        candidate,
        summary["baseline"],
        candidate_epochs.to_dict("records"),
    )
    checks["valid_gate_recomputed"] = compare_nested(
        summary["valid_gate"], valid_gate
    )

    discovery = summary["discovery_replay_gate"]
    if discovery.get("available"):
        checks["discovery_replay_binding"] = bool(
            discovery.get("passed")
            and checkpoint_sha256(Path(discovery["source"]))
            == discovery["source_sha256"]
        )
    else:
        checks["discovery_replay_binding"] = True

    test_path = root / "video_aware_sampling_test_predictions.csv"
    test_gate = summary["test_gate"]
    if valid_gate["passed"]:
        if test_gate is None or not test_path.is_file():
            raise RuntimeError("Passing Valid gate must produce frozen Test artifacts.")
        test = pd.read_csv(test_path)
        test_mae = metrics_from_predictions(test)
        for mode in MODES:
            require_close(
                test_mae[mode],
                candidate["test_at_valid_best_{}_MAE".format(mode)],
                "candidate test {} MAE".format(mode),
            )
        test_j = 0.5 * test_mae["LAV"] + 0.5 * np.mean([
            test_mae[mode] for mode in MISSING_MODES
        ])
        require_close(test_j, candidate["J_test_at_valid_best"], "candidate Test J")
        expected_test_gate = candidate_test_gate(candidate, summary["baseline"])
        checks["test_gate_recomputed"] = compare_nested(
            test_gate, expected_test_gate
        )
        checks["test_protocol"] = bool(
            summary["protocol"]["test_constructed"]
            and summary["protocol"]["test_loader_traversal_count"] == 1
            and source["test_constructed_after_valid_gate"]
            and source["test_loader_traversal_count"] == 1
        )
    else:
        checks["test_gate_recomputed"] = test_gate is None
        checks["test_protocol"] = bool(
            not test_path.exists()
            and not summary["protocol"]["test_constructed"]
            and summary["protocol"]["test_loader_traversal_count"] == 0
            and not source["test_constructed_after_valid_gate"]
            and source["test_loader_traversal_count"] == 0
        )

    checks["verdict_recomputed"] = (
        summary["verdict"]
        == expected_verdict(root, summary, valid_gate, test_gate)
    )
    checks["sampler_contract"] = bool(
        candidate["VideoAwareSampler"]
        and int(candidate["SamplesPerVideo"]) == SAMPLES_PER_VIDEO
        and summary["protocol"]["samples_per_video"] == SAMPLES_PER_VIDEO
        and summary["protocol"]["exact_epoch_coverage"]
        and not summary["protocol"]["oversampling"]
        and not summary["protocol"]["replacement"]
    )
    checks["original_objective_preserved"] = bool(
        not summary["protocol"]["new_loss"]
        and summary["protocol"]["original_cfcompat_objective_unchanged"]
    )
    checks["no_inference_change"] = bool(
        summary["protocol"]["additional_inference_parameters"] == 0
        and source["additional_inference_parameters"] == 0
    )

    test_flags = comparison.loc[
        comparison.metric.astype(str).str.startswith("test_"),
        "decision_metric",
    ]
    checks["comparison_flags"] = bool(
        test_flags.empty or test_flags.astype(bool).eq(False).all()
    )

    passed = bool(all(checks.values()))
    payload = {
        "passed": passed,
        "verdict": summary["verdict"],
        "checks": checks,
    }
    (root / "video_aware_sampling_audit_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("Video-aware sampling independent audit failed.")

    print("CFCompat video-aware sampling independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
