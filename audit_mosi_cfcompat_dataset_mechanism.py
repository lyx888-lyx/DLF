"""Independent audit for the MOSI dataset and CFCompatKD mechanism study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.mosi_cfcompat_audit_utils import (
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    FORMAL_SEEDS,
    METHOD,
    MODES,
    VERSION,
    dataset_limitation_flags,
    dataset_split_summary,
    group_mechanism_summary,
    joint_video_bootstrap,
    mechanism_assessment,
    modality_marginal_value,
    opportunity_ranking,
    overall_prediction_summary,
    prediction_events,
    sha256_file,
    split_shift_summary,
    video_summary,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def same_frame(actual: pd.DataFrame, expected: pd.DataFrame, keys, tolerance=1e-9) -> bool:
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
            nested_close(left, right, tolerance) for left, right in zip(actual, expected)
        )
    if isinstance(expected, bool):
        return bool(actual) is expected
    if isinstance(expected, int):
        return int(actual) == int(expected)
    if isinstance(expected, float):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return abs(float(actual) - expected) <= tolerance
    return actual == expected


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    files = {
        "samples": root / "dataset_samples_train_valid.csv",
        "split": root / "dataset_split_summary.csv",
        "videos": root / "dataset_video_summary.csv",
        "feature_samples": root / "modality_feature_sample_quality.csv",
        "feature_summary": root / "modality_feature_summary.csv",
        "events": root / "valid_prediction_events.csv",
        "overall": root / "overall_prediction_summary.csv",
        "groups": root / "group_mechanism_summary.csv",
        "modality": root / "modality_marginal_value.csv",
        "bootstrap": root / "video_bootstrap.csv",
        "opportunities": root / "opportunity_ranking.csv",
        "source_metrics": root / "source_metric_rows.csv",
        "controls": root / "historical_control_valid_metrics.csv",
        "manifest": root / "source_manifest.json",
        "summary": root / "audit_summary.json",
        "report": root / "audit_report.md",
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audit artifacts:\n" + "\n".join(missing))

    manifest = json.loads(files["manifest"].read_text(encoding="utf-8"))
    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    samples = pd.read_csv(files["samples"])
    recorded_split = pd.read_csv(files["split"])
    recorded_videos = pd.read_csv(files["videos"])
    feature_samples = pd.read_csv(files["feature_samples"])
    recorded_feature_summary = pd.read_csv(files["feature_summary"])
    recorded_events = pd.read_csv(files["events"])
    recorded_overall = pd.read_csv(files["overall"])
    recorded_groups = pd.read_csv(files["groups"])
    recorded_modality = pd.read_csv(files["modality"])
    recorded_bootstrap = pd.read_csv(files["bootstrap"])
    recorded_opportunities = pd.read_csv(files["opportunities"])
    source_metrics = pd.read_csv(files["source_metrics"])

    checks = {}
    checks["version_and_method"] = bool(
        manifest["version"] == VERSION
        and manifest["method"] == METHOD
        and summary["version"] == VERSION
        and summary["method"] == METHOD
    )
    checks["frozen_protocol"] = bool(
        tuple(manifest["formal_seeds"]) == FORMAL_SEEDS
        and manifest["splits_constructed"] == ["train", "valid"]
        and not manifest["official_test_constructed"]
        and manifest["test_loader_construction_count"] == 0
        and manifest["test_loader_traversal_count"] == 0
        and not manifest["model_training_performed"]
        and not manifest["optimizer_constructed"]
        and not manifest["backward_called"]
        and not summary["protocol"]["official_test_constructed"]
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
    for record in manifest["source_records"]:
        for path_key, sha_key in (
            ("baseline_result", "baseline_result_sha256"),
            ("baseline_checkpoint", "baseline_checkpoint_sha256"),
            ("cfcompat_result", "cfcompat_result_sha256"),
            ("cfcompat_checkpoint", "cfcompat_checkpoint_sha256"),
            ("teacher_checkpoint", "teacher_checkpoint_sha256"),
        ):
            path = Path(record[path_key])
            source_binding = bool(
                source_binding and path.is_file() and sha256_file(path) == record[sha_key]
            )
    for record in manifest.get("historical_control_bindings", []):
        path = Path(record["path"])
        source_binding = bool(
            source_binding and path.is_file() and sha256_file(path) == record["sha256"]
        )
    checks["source_and_checkpoint_binding"] = source_binding

    checks["train_valid_only"] = bool(
        set(samples.Split.astype(str)) == {"train", "valid"}
        and "test" not in set(samples.Split.astype(str))
        and set(recorded_events.Seed.astype(int)) == set(FORMAL_SEEDS)
        and set(recorded_events.Mode.astype(str)) == set(MODES)
    )
    valid_count = int((samples.Split.astype(str) == "valid").sum())
    checks["complete_valid_prediction_grid"] = bool(
        len(recorded_events) == len(FORMAL_SEEDS) * len(MODES) * valid_count
        and not recorded_events.duplicated(["Seed", "Mode", "sample_index"]).any()
    )

    recomputed_split = dataset_split_summary(samples)
    recomputed_videos = video_summary(samples)
    recomputed_shift = split_shift_summary(samples)
    checks["dataset_split_summary_recomputed"] = same_frame(
        recorded_split, recomputed_split, ["Split"]
    )
    checks["video_summary_recomputed"] = same_frame(
        recorded_videos, recomputed_videos, ["Split", "video_id"]
    )
    checks["dataset_shift_recomputed"] = nested_close(
        recomputed_shift, summary["dataset_shift"]
    )

    recomputed_feature_summary = (
        feature_samples.groupby(["Split", "Modality"], as_index=False)
        .agg(
            sample_count=("sample_index", "size"),
            finite_fraction_mean=("finite_fraction", "mean"),
            effective_length_mean=("effective_length", "mean"),
            effective_length_std=("effective_length", "std"),
            zero_fraction_mean=("zero_fraction", "mean"),
            mean_abs_value=("mean_abs_value", "mean"),
            sample_std_mean=("sample_std", "mean"),
            all_zero_fraction=("all_zero", "mean"),
            near_constant_fraction=("near_constant", "mean"),
        )
    )
    checks["feature_summary_recomputed"] = same_frame(
        recorded_feature_summary,
        recomputed_feature_summary,
        ["Split", "Modality"],
    )

    raw_columns = [
        "Seed", "Mode", "sample_index", "sample_id", "video_id", "segment_id",
        "label", "baseline_prediction", "cfcompat_prediction", "teacher_prediction",
        "evaluator_LAV_prediction", "evaluator_shift", "compatibility_proxy", "token_count",
    ]
    recomputed_events = prediction_events(recorded_events[raw_columns])
    checks["prediction_events_recomputed"] = same_frame(
        recorded_events,
        recomputed_events,
        ["Seed", "Mode", "sample_index"],
    )
    recomputed_overall = overall_prediction_summary(recomputed_events)
    recomputed_groups = group_mechanism_summary(recomputed_events)
    recomputed_modality = modality_marginal_value(recomputed_events)
    recomputed_bootstrap = joint_video_bootstrap(
        recomputed_events,
        replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
    )
    recomputed_opportunities = opportunity_ranking(recomputed_groups)
    checks["overall_metrics_recomputed"] = same_frame(
        recorded_overall, recomputed_overall, ["Seed", "Mode"]
    )
    checks["group_mechanisms_recomputed"] = same_frame(
        recorded_groups, recomputed_groups, ["Seed", "GroupType", "GroupValue"]
    )
    checks["modality_value_recomputed"] = same_frame(
        recorded_modality, recomputed_modality, ["Seed", "Method", "Comparison"]
    )
    checks["video_bootstrap_recomputed"] = same_frame(
        recorded_bootstrap, recomputed_bootstrap, ["Replicate"]
    )
    checks["opportunity_ranking_recomputed"] = same_frame(
        recorded_opportunities,
        recomputed_opportunities,
        ["GroupType", "GroupValue"],
    )

    limitations = dataset_limitation_flags(
        recomputed_split,
        recomputed_shift,
        recomputed_modality,
        recomputed_overall,
    )
    mechanism = mechanism_assessment(
        recomputed_events,
        recomputed_overall,
        recomputed_bootstrap,
    )
    checks["dataset_limitations_recomputed"] = nested_close(
        limitations, summary["dataset_limitations"]
    )
    checks["mechanism_assessment_recomputed"] = nested_close(
        mechanism, summary["mechanism_assessment"]
    )
    checks["verdict_recomputed"] = bool(
        summary["verdict"] == mechanism["verdict"]
    )

    source_metric_binding = True
    j_rows = recomputed_overall.loc[recomputed_overall.Mode.eq("J")].set_index("Seed")
    for row in source_metrics.itertuples(index=False):
        seed = int(row.Seed)
        if seed not in j_rows.index:
            source_metric_binding = False
            continue
        if math.isfinite(float(row.baseline_source_J)):
            source_metric_binding = bool(
                source_metric_binding
                and abs(float(row.baseline_source_J) - float(j_rows.loc[seed, "baseline_MAE"])) <= 1e-4
            )
        if math.isfinite(float(row.cfcompat_source_J)):
            source_metric_binding = bool(
                source_metric_binding
                and abs(float(row.cfcompat_source_J) - float(j_rows.loc[seed, "cfcompat_MAE"])) <= 1e-4
            )
    checks["source_valid_metrics_reproduced"] = source_metric_binding
    checks["finite_metrics"] = bool(
        np.isfinite(
            recorded_events[
                [
                    "label", "baseline_prediction", "cfcompat_prediction",
                    "teacher_prediction", "evaluator_shift", "compatibility_proxy",
                    "baseline_error", "cfcompat_error", "cfcompat_gain",
                ]
            ].to_numpy(dtype=float)
        ).all()
        and np.isfinite(recorded_bootstrap.select_dtypes(include=[np.number]).to_numpy(dtype=float)).all()
    )
    checks["no_test_named_outputs"] = bool(
        not any("test" in path.name.lower() for path in root.iterdir())
    )

    passed = bool(all(checks.values()))
    payload = {
        "version": VERSION,
        "passed": passed,
        "verdict": mechanism["verdict"],
        "checks": checks,
    }
    output = root / "independent_audit_check.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        failed = [key for key, value in checks.items() if not value]
        raise RuntimeError("Independent MOSI CFCompat audit failed: {}".format(failed))
    print("MOSI CFCompatKD independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", mechanism["verdict"])


if __name__ == "__main__":
    main()
