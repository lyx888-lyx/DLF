"""Independent artifact audit for CFCompatKD source-video V-REx."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES
from trains.singleTask.video_vrex_utils import (
    METHOD,
    VERSION,
    VREX_LAMBDAS,
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
        mode: float(
            np.mean(
                np.abs(
                    frame["{}_pred".format(mode)].to_numpy(dtype=float)
                    - labels
                )
            )
        )
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
    return actual == expected


def main():
    args = parse_args()
    root = Path(args.result_dir)
    required = {
        "summary": root / "video_vrex_summary.json",
        "source": root / "video_vrex_source_manifest.json",
        "grid": root / "video_vrex_grid_summary.csv",
        "epochs": root / "video_vrex_all_epoch_metrics.csv",
        "valid": root / "video_vrex_all_valid_predictions.csv",
        "report": root / "video_vrex_report.md",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing Video-VREx artifacts:\n" + "\n".join(missing)
        )

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    source = json.loads(required["source"].read_text(encoding="utf-8"))
    grid = pd.read_csv(required["grid"])
    epochs = pd.read_csv(required["epochs"])
    valid = pd.read_csv(required["valid"])

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
    checks["selected_checkpoint_hash"] = (
        checkpoint_sha256(Path(source["selected_checkpoint"]))
        == source["selected_checkpoint_sha256"]
    )
    checks["teacher_hash"] = (
        checkpoint_sha256(Path(source["teacher_checkpoint"]))
        == source["teacher_sha256"]
    )
    checks["evaluator_hash"] = (
        checkpoint_sha256(Path(source["evaluator_checkpoint"]))
        == source["evaluator_sha256"]
    )

    domain = json.loads(Path(source["domain_audit"]).read_text(encoding="utf-8"))
    checks["video_domain_audit_passed"] = bool(
        domain["passed"]
        and domain["vrex_batch_viable"]
        and domain["sampler_exact_coverage"]
        and domain["sampler_duplicate_count"] == 0
        and all(not values for values in domain["split_video_overlaps"].values())
    )

    seed = int(summary["seed"])
    observed_positive = sorted(
        grid.loc[grid.EligibleForSelection.astype(bool), "LambdaVREx"]
        .astype(float)
        .tolist()
    )
    if seed == 1111:
        expected_positive = sorted(float(value) for value in VREX_LAMBDAS)
        checks["frozen_positive_lambda_grid"] = (
            observed_positive == expected_positive
            and sorted(float(value) for value in summary["lambda_grid"])
            == expected_positive
        )
    else:
        declared = [float(value) for value in summary["lambda_grid"]]
        checks["frozen_positive_lambda_grid"] = bool(
            len(declared) == 1
            and len(observed_positive) == 1
            and np.isclose(declared[0], observed_positive[0])
            and any(np.isclose(declared[0], value) for value in VREX_LAMBDAS)
        )

    checks["one_sampler_control"] = bool(
        len(grid.loc[np.isclose(grid.LambdaVREx.astype(float), 0.0)]) == 1
        and not grid.loc[
            np.isclose(grid.LambdaVREx.astype(float), 0.0),
            "EligibleForSelection",
        ].astype(bool).any()
    )

    selected = summary["selected_candidate"]
    selected_run = str(selected["Run"])
    selected_lambda = float(selected["LambdaVREx"])
    selected_grid = grid.loc[grid.Run.astype(str).eq(selected_run)]
    if len(selected_grid) != 1:
        raise RuntimeError("Selected Video-VREx row is not unique.")
    checks["selected_is_positive_lambda"] = any(
        np.isclose(selected_lambda, value) for value in VREX_LAMBDAS
    )
    eligible_grid = grid.loc[grid.EligibleForSelection.astype(bool)].copy()
    expected_selected = eligible_grid.sort_values(
        ["J_valid", "LambdaVREx"], kind="mergesort"
    ).iloc[0]
    checks["selected_by_valid_J_only"] = (
        str(expected_selected.Run) == selected_run
        and int(grid.SelectedByValidJ.astype(bool).sum()) == 1
        and bool(selected_grid.iloc[0].SelectedByValidJ)
    )

    selected_valid = valid.loc[valid.Run.astype(str).eq(selected_run)].copy()
    if selected_valid.empty:
        raise RuntimeError("Selected validation predictions are absent.")
    valid_mae = metrics_from_predictions(selected_valid)
    for mode in MODES:
        require_close(
            valid_mae[mode],
            selected["valid_{}_MAE".format(mode)],
            "selected valid {} MAE".format(mode),
        )
    valid_j = 0.5 * valid_mae["LAV"] + 0.5 * np.mean(
        [valid_mae[mode] for mode in MISSING_MODES]
    )
    require_close(valid_j, selected["J_valid"], "selected Valid J")
    checks["selected_valid_metrics_recomputed"] = True

    baseline_frame = pd.read_csv(source["baseline_result"])
    baseline_rows = baseline_frame.loc[
        baseline_frame.Seed.astype(int).eq(seed)
    ]
    if len(baseline_rows) != 1:
        raise RuntimeError("Baseline seed row is not unique.")
    baseline_row = baseline_rows.iloc[0]
    for key, value in summary["baseline"].items():
        if key in baseline_row and isinstance(value, (int, float)):
            require_close(baseline_row[key], value, "baseline {}".format(key))
    checks["baseline_binding_recomputed"] = True

    replay = summary["baseline_replay"]
    replay_gate = summary["baseline_replay_gate"]
    replay_differences = {
        "J_valid": abs(float(replay["J_valid"]) - float(summary["baseline"]["J_valid"])),
    }
    for mode in MODES:
        key = "valid_{}_MAE".format(mode)
        replay_differences[key] = abs(
            float(replay[key]) - float(summary["baseline"][key])
        )
    recomputed_replay = {
        "passed": bool(
            int(replay["BestValidEpoch"])
            == int(summary["baseline"]["BestValidEpoch"])
            and all(
                value <= float(replay_gate["tolerance"])
                for value in replay_differences.values()
            )
        ),
        "epoch_match": bool(
            int(replay["BestValidEpoch"])
            == int(summary["baseline"]["BestValidEpoch"])
        ),
        "tolerance": float(replay_gate["tolerance"]),
        "differences": replay_differences,
    }
    checks["baseline_replay_gate_recomputed"] = compare_nested(
        replay_gate, recomputed_replay
    ) and replay_gate["passed"]

    selected_epochs = epochs.loc[epochs.Run.astype(str).eq(selected_run)]
    valid_gate = candidate_valid_gate(
        selected,
        summary["baseline"],
        selected_epochs.to_dict("records"),
    )
    checks["valid_gate_recomputed"] = compare_nested(
        summary["valid_gate"], valid_gate
    )

    test_path = root / "video_vrex_selected_test_predictions.csv"
    test_gate = summary["test_gate"]
    if valid_gate["passed"]:
        if not test_path.is_file() or test_gate is None:
            raise RuntimeError("Passing Valid gate must produce frozen test artifacts.")
        test = pd.read_csv(test_path)
        test_mae = metrics_from_predictions(test)
        for mode in MODES:
            require_close(
                test_mae[mode],
                selected["test_at_valid_best_{}_MAE".format(mode)],
                "selected test {} MAE".format(mode),
            )
        test_j = 0.5 * test_mae["LAV"] + 0.5 * np.mean(
            [test_mae[mode] for mode in MISSING_MODES]
        )
        require_close(test_j, selected["J_test_at_valid_best"], "selected Test J")
        recomputed_test_gate = candidate_test_gate(selected, summary["baseline"])
        checks["test_gate_recomputed"] = compare_nested(
            test_gate, recomputed_test_gate
        )
        expected_verdict = (
            "PROMOTE_SEED1111_RUN_SEED1114"
            if test_gate["passed"] and seed == 1111
            else (
                "PROMOTE_TWO_SEEDS_RUN_MOSEI"
                if test_gate["passed"] and seed == 1114
                else "STOP_VIDEO_VREX_TEST_GATE_FAILED"
            )
        )
        checks["test_protocol"] = bool(
            summary["protocol"]["test_constructed"]
            and summary["protocol"]["test_loader_traversal_count"] == 1
            and source["test_constructed_after_valid_gate"]
            and source.get("test_loader_traversal_count") == 1
        )
    else:
        checks["test_gate_recomputed"] = test_gate is None
        checks["test_protocol"] = bool(
            not test_path.exists()
            and not summary["protocol"]["test_constructed"]
            and summary["protocol"]["test_loader_traversal_count"] == 0
            and not source["test_constructed_after_valid_gate"]
            and source.get("test_loader_traversal_count") == 0
        )
        expected_verdict = "STOP_VIDEO_VREX_VALID_GATE_FAILED"

    checks["verdict_recomputed"] = summary["verdict"] == expected_verdict
    checks["no_inference_change"] = bool(
        summary["protocol"]["additional_inference_parameters"] == 0
        and source["additional_inference_parameters"] == 0
    )
    checks["original_objective_preserved"] = bool(
        summary["protocol"]["original_cfcompat_objective_unchanged"]
    )

    passed = bool(all(checks.values()))
    payload = {
        "passed": passed,
        "verdict": summary["verdict"],
        "checks": checks,
    }
    (root / "video_vrex_audit_check.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not passed:
        raise RuntimeError("Video-VREx independent audit failed.")

    print("CFCompat Video-VREx independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
