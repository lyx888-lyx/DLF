"""Strict v2 artifact audit for the frozen DLF role-specialization study.

This audit intentionally recomputes the role gate, probe metrics, long-tail
status, and baseline bin-risk tables from lower-level artifacts. It does not
construct a model or loader and therefore can be run without repeating feature
extraction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from trains.singleTask.dlf_role_specialization_utils import (
    FORMAL_SEEDS,
    METHOD,
    MODES,
    NEUTRAL_TAU,
    REPRESENTATIONS,
    SENTIMENT_BINS,
    VERSION,
    bin_risk_metrics,
    effective_number_weights,
    long_tail_status,
    ordinal_probe_metrics,
    polarity_probe_metrics,
    role_alignment_gate,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def truthy(value) -> bool:
    return str(value).strip().lower() == "true"


def nested_close(actual, expected, tolerance=1e-10) -> bool:
    if isinstance(expected, Mapping):
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
        return int(actual) == expected
    if isinstance(expected, float):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return abs(float(actual) - expected) <= tolerance
    return actual == expected


def same_numeric(left, right, tolerance=1e-7) -> bool:
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    return bool(
        left.shape == right.shape
        and np.allclose(left, right, atol=tolerance, rtol=0, equal_nan=True)
    )


def unique_row(frame: pd.DataFrame, mask: pd.Series, name: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise RuntimeError("{} is not unique: {} rows".format(name, len(selected)))
    return selected.iloc[0]


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    required_names = (
        "label_samples_train_valid.csv",
        "label_distribution.csv",
        "effective_number_weights.csv",
        "dataset_availability.csv",
        "baseline_predictions_train_valid.csv",
        "representation_manifest.csv",
        "model_state_audit.csv",
        "probe_metrics.csv",
        "probe_valid_predictions.csv",
        "role_alignment_comparisons.csv",
        "baseline_bin_risk.csv",
        "baseline_risk_summary.csv",
        "role_specialization_source_manifest.json",
        "role_specialization_summary.json",
        "role_specialization_report.md",
    )
    missing = [str(root / name) for name in required_names if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError("Missing role-audit artifacts:\n" + "\n".join(missing))

    summary = json.loads(
        (root / "role_specialization_summary.json").read_text(encoding="utf-8")
    )
    source = json.loads(
        (root / "role_specialization_source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    samples = pd.read_csv(root / "label_samples_train_valid.csv")
    distribution = pd.read_csv(root / "label_distribution.csv")
    weights = pd.read_csv(root / "effective_number_weights.csv")
    availability = pd.read_csv(root / "dataset_availability.csv")
    predictions = pd.read_csv(root / "baseline_predictions_train_valid.csv")
    representations = pd.read_csv(root / "representation_manifest.csv")
    states = pd.read_csv(root / "model_state_audit.csv")
    probe_metrics = pd.read_csv(root / "probe_metrics.csv")
    probe_predictions = pd.read_csv(root / "probe_valid_predictions.csv")
    comparisons = pd.read_csv(root / "role_alignment_comparisons.csv")
    risk_bins = pd.read_csv(root / "baseline_bin_risk.csv")
    risk_summary = pd.read_csv(root / "baseline_risk_summary.csv")

    checks = {}
    checks["version_and_method"] = bool(
        summary["version"] == VERSION
        and summary["method"] == METHOD
        and source["version"] == VERSION
        and source["method"] == METHOD
    )
    checks["frozen_protocol"] = bool(
        tuple(source["formal_seeds"]) == FORMAL_SEEDS
        and tuple(summary["protocol"]["modes"]) == MODES
        and tuple(summary["protocol"]["representation_families"])
        == REPRESENTATIONS
        and float(summary["protocol"]["neutral_tau"]) == NEUTRAL_TAU
        and summary["protocol"]["fit_split"] == "official_train_only"
        and summary["protocol"]["evaluation_split"] == "official_valid_only"
        and summary["protocol"][
            "specific_representation_uses_present_modalities_only"
        ]
    )

    artifact_ok = True
    for name, metadata in source["artifacts"].items():
        path = Path(metadata["path"])
        artifact_ok = bool(
            artifact_ok
            and path.is_file()
            and path.resolve() == (root / name).resolve()
            and sha256_file(path) == metadata["sha256"]
        )
    checks["artifact_hashes"] = artifact_ok

    binding_ok = True
    for key in ("summary", "audit", "grid"):
        path = Path(source["source"]["{}_path".format(key)])
        binding_ok = bool(
            binding_ok
            and path.is_file()
            and sha256_file(path)
            == source["source"]["{}_sha256".format(key)]
        )
    for row in source["source"]["baseline_rows"]:
        checkpoint = Path(row["ResolvedCheckpoint"])
        binding_ok = bool(
            binding_ok
            and checkpoint.is_file()
            and checkpoint_sha256(checkpoint) == row["MainCheckpointSHA256"]
        )
    checks["source_and_checkpoint_binding"] = binding_ok

    checks["official_test_forbidden"] = bool(
        set(samples.Split.astype(str)) == {"train", "valid"}
        and set(predictions.Split.astype(str)) == {"train", "valid"}
        and set(representations.Split.astype(str)) == {"train", "valid"}
        and not source["official_test_constructed"]
        and source["test_loader_construction_count"] == 0
        and source["test_loader_traversal_count"] == 0
        and not summary["protocol"]["official_test_constructed"]
        and not summary["protocol"]["official_test_authorized"]
        and not any(root.glob("*test*prediction*"))
    )
    checks["no_model_training"] = bool(
        not source["optimizer_constructed"]
        and not source["backward_called"]
        and not source["model_parameters_updated"]
        and not summary["protocol"]["model_parameters_updated"]
        and states.ParametersUnchanged.map(truthy).all()
        and (states.TrainableParameterCountDuringAudit.astype(int) == 0).all()
        and (
            states.StateSHA256Before.astype(str)
            == states.StateSHA256After.astype(str)
        ).all()
    )

    # Recompute all distribution counts from the sample-level table.
    expected_classes = {
        "polarity_3": tuple(range(3)),
        "intensity_4": tuple(range(4)),
        "sentiment_7": tuple(SENTIMENT_BINS),
    }
    distribution_ok = True
    for (dataset, split), local_samples in samples.groupby(
        ["Dataset", "Split"], sort=True
    ):
        for family, column in (
            ("polarity_3", "polarity"),
            ("intensity_4", "intensity"),
            ("sentiment_7", "sentiment_bin"),
        ):
            local = distribution.loc[
                distribution.Dataset.astype(str).eq(str(dataset))
                & distribution.Split.astype(str).eq(str(split))
                & distribution.LabelFamily.astype(str).eq(family)
            ].sort_values("Class")
            distribution_ok = bool(
                distribution_ok
                and tuple(local.Class.astype(int)) == expected_classes[family]
            )
            for class_value in expected_classes[family]:
                observed = int(
                    local_samples[column].astype(int).eq(int(class_value)).sum()
                )
                recorded = int(
                    unique_row(
                        local,
                        local.Class.astype(int).eq(int(class_value)),
                        "distribution class",
                    )["Count"]
                )
                distribution_ok = distribution_ok and observed == recorded
    checks["distribution_counts_recomputed"] = bool(distribution_ok)

    # Recompute effective-number weights from train counts only.
    weight_ok = True
    available_datasets = availability.loc[
        availability.Available.map(truthy)
    ].Dataset.astype(str)
    for dataset in available_datasets:
        train = samples.loc[
            samples.Dataset.astype(str).eq(dataset)
            & samples.Split.astype(str).eq("train")
        ]
        for family, column, classes in (
            ("polarity_3", "polarity", range(3)),
            ("intensity_4", "intensity", range(4)),
            ("sentiment_7", "sentiment_bin", SENTIMENT_BINS),
        ):
            counts = {
                int(value): int(train[column].astype(int).eq(int(value)).sum())
                for value in classes
            }
            expected = effective_number_weights(counts).sort_values("class")
            recorded = weights.loc[
                weights.Dataset.astype(str).eq(dataset)
                & weights.LabelFamily.astype(str).eq(family)
            ].sort_values("class")
            weight_ok = bool(
                weight_ok
                and len(expected) == len(recorded)
                and np.array_equal(
                    expected["class"].to_numpy(dtype=int),
                    recorded["class"].to_numpy(dtype=int),
                )
            )
            for column_name in (
                "count",
                "beta",
                "effective_weight",
                "effective_weight_capped",
            ):
                weight_ok = bool(
                    weight_ok
                    and same_numeric(expected[column_name], recorded[column_name], 1e-10)
                )
    checks["effective_weights_recomputed"] = bool(weight_ok)

    # Verify representation files, dimensions, uniqueness, and model-state coverage.
    expected_representation_runs = {
        (seed, split, mode)
        for seed in FORMAL_SEEDS
        for split in ("train", "valid")
        for mode in MODES
    }
    observed_representation_runs = {
        (int(row.Seed), str(row.Split), str(row.Mode))
        for row in representations.itertuples(index=False)
    }
    representation_ok = observed_representation_runs == expected_representation_runs
    expected_specific_dims = {"LAV": 150, "LA": 100, "LV": 100, "L": 50}
    for row in representations.itertuples(index=False):
        path = Path(row.Path)
        if not path.is_file() or sha256_file(path) != row.SHA256:
            representation_ok = False
            continue
        payload = np.load(path, allow_pickle=False)
        representation_ok = bool(
            representation_ok
            and len(payload["label"]) == int(row.SampleCount)
            and payload["shared"].shape == (int(row.SampleCount), 150)
            and payload["specific_present"].shape
            == (int(row.SampleCount), expected_specific_dims[str(row.Mode)])
            and payload["final_fusion"].shape == (int(row.SampleCount), 300)
            and len(np.unique(payload["sample_index"])) == int(row.SampleCount)
        )
    checks["representation_files_and_dimensions"] = bool(representation_ok)

    # Recompute both probe tasks from per-sample predictions.
    probe_ok = True
    for (seed, mode, representation), local in probe_predictions.groupby(
        ["Seed", "Mode", "Representation"], sort=True
    ):
        polarity = polarity_probe_metrics(
            local.true_polarity.to_numpy(dtype=int),
            local.pred_polarity.to_numpy(dtype=int),
        )
        ordinal = ordinal_probe_metrics(
            local.true_intensity.to_numpy(dtype=int),
            local.pred_intensity.to_numpy(dtype=int),
        )
        polarity_row = unique_row(
            probe_metrics,
            probe_metrics.Seed.astype(int).eq(int(seed))
            & probe_metrics.Mode.astype(str).eq(str(mode))
            & probe_metrics.Representation.astype(str).eq(str(representation))
            & probe_metrics.Task.astype(str).eq("polarity_3"),
            "polarity probe row",
        )
        ordinal_row = unique_row(
            probe_metrics,
            probe_metrics.Seed.astype(int).eq(int(seed))
            & probe_metrics.Mode.astype(str).eq(str(mode))
            & probe_metrics.Representation.astype(str).eq(str(representation))
            & probe_metrics.Task.astype(str).eq("absolute_intensity_ordinal_4"),
            "ordinal probe row",
        )
        for name, value in polarity.items():
            probe_ok = probe_ok and abs(float(polarity_row[name]) - value) <= 1e-7
        for name, value in ordinal.items():
            probe_ok = probe_ok and abs(float(ordinal_row[name]) - value) <= 1e-7
        probabilities = local[["p_gt_0p5", "p_gt_1p5", "p_gt_2p5"]].to_numpy(
            dtype=float
        )
        probe_ok = bool(
            probe_ok
            and np.isfinite(probabilities).all()
            and ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
            and (probabilities[:, 0] + 1e-12 >= probabilities[:, 1]).all()
            and (probabilities[:, 1] + 1e-12 >= probabilities[:, 2]).all()
        )
    checks["probe_metrics_and_ordinal_monotonicity"] = bool(probe_ok)

    recomputed_role = role_alignment_gate(comparisons)
    checks["role_gate_recomputed"] = nested_close(
        recomputed_role, summary["role_alignment"]
    )
    recomputed_tail = {
        dataset: long_tail_status(distribution, dataset)
        for dataset in available_datasets
    }
    checks["long_tail_recomputed"] = nested_close(
        recomputed_tail, summary["long_tail"]
    )

    # Strictly recompute every bin-risk row and every aggregate from predictions.
    risk_ok = True
    expected_risk_runs = {
        (seed, split, mode)
        for seed in FORMAL_SEEDS
        for split in ("train", "valid")
        for mode in MODES
    }
    observed_risk_runs = {
        (int(row.Seed), str(row.Split), str(row.Mode))
        for row in risk_summary.itertuples(index=False)
    }
    risk_ok = risk_ok and observed_risk_runs == expected_risk_runs
    for seed, split, mode in sorted(expected_risk_runs):
        local_predictions = predictions.loc[
            predictions.Seed.astype(int).eq(int(seed))
            & predictions.Split.astype(str).eq(str(split))
            & predictions.Mode.astype(str).eq(str(mode))
        ]
        expected_bins, expected_summary = bin_risk_metrics(local_predictions)
        recorded_bins = risk_bins.loc[
            risk_bins.Seed.astype(int).eq(int(seed))
            & risk_bins.Split.astype(str).eq(str(split))
            & risk_bins.Mode.astype(str).eq(str(mode))
        ].sort_values("sentiment_bin")
        expected_bins = expected_bins.sort_values("sentiment_bin")
        risk_ok = bool(
            risk_ok
            and len(local_predictions) > 0
            and len(recorded_bins) == len(SENTIMENT_BINS)
            and tuple(recorded_bins.sentiment_bin.astype(int)) == tuple(SENTIMENT_BINS)
            and int(recorded_bins["count"].sum()) == len(local_predictions)
            and np.array_equal(
                expected_bins.sentiment_bin.to_numpy(dtype=int),
                recorded_bins.sentiment_bin.to_numpy(dtype=int),
            )
        )
        for column in (
            "count",
            "mae",
            "polarity_flip_rate",
            "neutral_escape_rate",
        ):
            risk_ok = bool(
                risk_ok
                and same_numeric(expected_bins[column], recorded_bins[column], 1e-10)
            )
        recorded_summary = unique_row(
            risk_summary,
            risk_summary.Seed.astype(int).eq(int(seed))
            & risk_summary.Split.astype(str).eq(str(split))
            & risk_summary.Mode.astype(str).eq(str(mode)),
            "risk summary row",
        )
        for name, value in expected_summary.items():
            risk_ok = bool(
                risk_ok and abs(float(recorded_summary[name]) - float(value)) <= 1e-10
            )
    checks["baseline_risk_recomputed"] = bool(risk_ok)

    if not recomputed_tail["mosi"]["tail_present"]:
        expected_verdict = "STOP_LONG_TAIL_PREMISE_NOT_SUPPORTED"
    elif recomputed_role["passed"]:
        expected_verdict = "PROMOTE_ROLE_SPECIALIZATION_STAGE_B"
    elif recomputed_role["status"] == "PARTIAL_ROLE_ALIGNMENT_NEEDS_REVIEW":
        expected_verdict = "PARTIAL_ROLE_ALIGNMENT_DO_NOT_TRAIN_YET"
    else:
        expected_verdict = "STOP_ROLE_SPECIALIZATION_HYPOTHESIS"
    checks["verdict_recomputed"] = summary["verdict"] == expected_verdict

    passed = bool(all(checks.values()))
    payload = {
        "version": "dlf_role_specialization_audit_v2",
        "passed": passed,
        "verdict": summary["verdict"],
        "role_status": recomputed_role["status"],
        "checks": checks,
    }
    output = root / "role_specialization_audit_check_v2.json"
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])
    print("role status:", recomputed_role["status"])
    print("audit v2:", output)
    if not passed:
        raise RuntimeError("DLF role-specialization v2 independent audit failed.")
    print("DLF role-specialization v2 independent audit passed")


if __name__ == "__main__":
    main()
