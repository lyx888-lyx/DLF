"""Independent artifact audit for the frozen DLF role-specialization study."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

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
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        return int(actual) == expected
    if isinstance(expected, float):
        if math.isnan(expected):
            return math.isnan(float(actual))
        return abs(float(actual) - expected) <= tolerance
    return actual == expected


def require_close(left, right, name, tolerance=1e-7):
    if abs(float(left) - float(right)) > tolerance:
        raise RuntimeError("{} mismatch: {} vs {}".format(name, left, right))


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    required_names = [
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
    ]
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
    representation_manifest = pd.read_csv(root / "representation_manifest.csv")
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

    artifact_hashes_ok = True
    for name, metadata in source["artifacts"].items():
        path = Path(metadata["path"])
        artifact_hashes_ok = bool(
            artifact_hashes_ok
            and path.is_file()
            and path.resolve() == (root / name).resolve()
            and sha256_file(path) == metadata["sha256"]
        )
    checks["artifact_hashes"] = artifact_hashes_ok

    source_binding_ok = True
    for key in ("summary", "audit", "grid"):
        path = Path(source["source"]["{}_path".format(key)])
        source_binding_ok = bool(
            source_binding_ok
            and path.is_file()
            and sha256_file(path)
            == source["source"]["{}_sha256".format(key)]
        )
    for row in source["source"]["baseline_rows"]:
        path = Path(row["ResolvedCheckpoint"])
        source_binding_ok = bool(
            source_binding_ok
            and path.is_file()
            and checkpoint_sha256(path) == row["MainCheckpointSHA256"]
        )
    checks["source_and_checkpoint_binding"] = source_binding_ok

    checks["train_valid_only"] = bool(
        set(samples.Split.astype(str)) == {"train", "valid"}
        and set(predictions.Split.astype(str)) == {"train", "valid"}
        and set(representation_manifest.Split.astype(str)) == {"train", "valid"}
        and "test" not in " ".join(samples.columns).lower()
        and "test" not in " ".join(predictions.columns).lower()
    )
    checks["official_test_forbidden"] = bool(
        not source["official_test_constructed"]
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
        and states.ParametersUnchanged.map(
            lambda value: str(value).strip().lower() == "true"
        ).all()
        and (states.TrainableParameterCountDuringAudit.astype(int) == 0).all()
    )

    expected_state_seeds = sorted(FORMAL_SEEDS)
    checks["complete_model_state_audit"] = bool(
        sorted(states.Seed.astype(int).tolist()) == expected_state_seeds
        and (
            states.StateSHA256Before.astype(str)
            == states.StateSHA256After.astype(str)
        ).all()
    )

    distribution_ok = True
    expected_classes = {
        "polarity_3": tuple(range(3)),
        "intensity_4": tuple(range(4)),
        "sentiment_7": SENTIMENT_BINS,
    }
    for (dataset, split), local_samples in samples.groupby(
        ["Dataset", "Split"], sort=True
    ):
        for family, column in (
            ("polarity_3", "polarity"),
            ("intensity_4", "intensity"),
            ("sentiment_7", "sentiment_bin"),
        ):
            local_distribution = distribution.loc[
                distribution.Dataset.astype(str).eq(str(dataset))
                & distribution.Split.astype(str).eq(str(split))
                & distribution.LabelFamily.astype(str).eq(family)
            ].sort_values("Class")
            distribution_ok = bool(
                distribution_ok
                and tuple(local_distribution.Class.astype(int))
                == tuple(expected_classes[family])
            )
            for class_value in expected_classes[family]:
                observed = int(
                    local_samples[column].astype(int).eq(int(class_value)).sum()
                )
                recorded = int(
                    local_distribution.loc[
                        local_distribution.Class.astype(int).eq(int(class_value)),
                        "Count",
                    ].iloc[0]
                )
                distribution_ok = distribution_ok and observed == recorded
    checks["distribution_counts_recomputed"] = bool(distribution_ok)

    weight_ok = True
    for dataset in availability.loc[
        availability.Available.map(lambda value: str(value).lower() == "true")
    ].Dataset.astype(str):
        train_samples = samples.loc[
            samples.Dataset.astype(str).eq(dataset)
            & samples.Split.astype(str).eq("train")
        ]
        for family, column, classes in (
            ("polarity_3", "polarity", range(3)),
            ("intensity_4", "intensity", range(4)),
            ("sentiment_7", "sentiment_bin", SENTIMENT_BINS),
        ):
            counts = {
                int(class_value): int(
                    train_samples[column].astype(int).eq(int(class_value)).sum()
                )
                for class_value in classes
            }
            recomputed = effective_number_weights(counts).sort_values("class")
            recorded = weights.loc[
                weights.Dataset.astype(str).eq(dataset)
                & weights.LabelFamily.astype(str).eq(family)
            ].sort_values("class")
            if len(recomputed) != len(recorded):
                weight_ok = False
                continue
            for column_name in (
                "count",
                "beta",
                "effective_weight",
                "effective_weight_capped",
            ):
                left = recomputed[column_name].to_numpy(dtype=float)
                right = recorded[column_name].to_numpy(dtype=float)
                weight_ok = bool(
                    weight_ok and np.allclose(left, right, atol=1e-10, rtol=0)
                )
    checks["effective_weights_recomputed"] = bool(weight_ok)

    expected_representation_runs = {
        (seed, split, mode)
        for seed in FORMAL_SEEDS
        for split in ("train", "valid")
        for mode in MODES
    }
    observed_representation_runs = {
        (int(row.Seed), str(row.Split), str(row.Mode))
        for row in representation_manifest.itertuples(index=False)
    }
    representation_ok = observed_representation_runs == expected_representation_runs
    expected_specific_dims = {"LAV": 150, "LA": 100, "LV": 100, "L": 50}
    for row in representation_manifest.itertuples(index=False):
        path = Path(row.Path)
        if not path.is_file() or sha256_file(path) != row.SHA256:
            representation_ok = False
            continue
        payload = np.load(path, allow_pickle=False)
        representation_ok = bool(
            representation_ok
            and len(payload["label"]) == int(row.SampleCount)
            and payload["shared"].shape[1] == int(row.SharedDim) == 150
            and payload["specific_present"].shape[1]
            == int(row.SpecificPresentDim)
            == expected_specific_dims[str(row.Mode)]
            and payload["final_fusion"].shape[1]
            == int(row.FinalFusionDim)
            == 300
            and len(np.unique(payload["sample_index"])) == len(payload["sample_index"])
        )
    checks["representation_files_and_dimensions"] = bool(representation_ok)

    expected_probe_runs = {
        (seed, mode, representation, task)
        for seed in FORMAL_SEEDS
        for mode in MODES
        for representation in (
            ("shared", "specific_present", "final_fusion", "specific_l")
            + (("specific_a",) if "A" in mode else tuple())
            + (("specific_v",) if "V" in mode else tuple())
        )
        for task in ("polarity_3", "absolute_intensity_ordinal_4")
    }
    observed_probe_runs = {
        (int(row.Seed), str(row.Mode), str(row.Representation), str(row.Task))
        for row in probe_metrics.itertuples(index=False)
    }
    checks["complete_fixed_probe_grid"] = bool(
        observed_probe_runs == expected_probe_runs
    )
    checks["fixed_probe_hyperparameters"] = bool(
        float(source["probe_hyperparameters"]["C"]) == 1.0
        and source["probe_hyperparameters"]["class_weight"] == "balanced"
        and int(source["probe_hyperparameters"]["max_iter"]) == 2000
        and int(source["probe_hyperparameters"]["random_state"]) == 0
        and not source["probe_hyperparameters"]["selection_on_valid"]
    )

    probe_metrics_ok = True
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
        polarity_row = probe_metrics.loc[
            probe_metrics.Seed.astype(int).eq(int(seed))
            & probe_metrics.Mode.astype(str).eq(str(mode))
            & probe_metrics.Representation.astype(str).eq(str(representation))
            & probe_metrics.Task.astype(str).eq("polarity_3")
        ].iloc[0]
        ordinal_row = probe_metrics.loc[
            probe_metrics.Seed.astype(int).eq(int(seed))
            & probe_metrics.Mode.astype(str).eq(str(mode))
            & probe_metrics.Representation.astype(str).eq(str(representation))
            & probe_metrics.Task.astype(str).eq("absolute_intensity_ordinal_4")
        ].iloc[0]
        for name, value in polarity.items():
            require_close(value, polarity_row[name], "polarity {}".format(name))
        for name, value in ordinal.items():
            require_close(value, ordinal_row[name], "ordinal {}".format(name))
        probabilities = local[["p_gt_0p5", "p_gt_1p5", "p_gt_2p5"]].to_numpy(
            dtype=float
        )
        probe_metrics_ok = bool(
            probe_metrics_ok
            and np.isfinite(probabilities).all()
            and ((probabilities >= 0) & (probabilities <= 1)).all()
            and (probabilities[:, 0] + 1e-12 >= probabilities[:, 1]).all()
            and (probabilities[:, 1] + 1e-12 >= probabilities[:, 2]).all()
        )
    checks["probe_metrics_and_ordinal_monotonicity"] = bool(probe_metrics_ok)

    recomputed_role = role_alignment_gate(comparisons)
    checks["role_gate_recomputed"] = nested_close(
        recomputed_role, summary["role_alignment"]
    )

    recomputed_tail = {}
    for dataset in availability.loc[
        availability.Available.map(lambda value: str(value).lower() == "true")
    ].Dataset.astype(str):
        recomputed_tail[dataset] = long_tail_status(distribution, dataset)
    checks["long_tail_recomputed"] = nested_close(
        recomputed_tail, summary["long_tail"]
    )

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
    for key, local in risk_bins.groupby(["Seed", "Split", "Mode"], sort=True):
        risk_ok = bool(
            risk_ok
            and tuple(local.sort_values("sentiment_bin").sentiment_bin.astype(int))
            == SENTIMENT_BINS
            and int(local.Count.sum())
            == len(
                predictions.loc[
                    predictions.Seed.astype(int).eq(int(key[0]))
                    & predictions.Split.astype(str).eq(str(key[1]))
                    & predictions.Mode.astype(str).eq(str(key[2]))
                ]
            )
        )
    checks["baseline_risk_grid_complete"] = bool(risk_ok)

    if not recomputed_tail["mosi"]["tail_present"]:
        expected_verdict = "STOP_LONG_TAIL_PREMISE_NOT_SUPPORTED"
    elif recomputed_role["passed"]:
        expected_verdict = "PROMOTE_ROLE_SPECIALIZATION_STAGE_B"
    elif recomputed_role["status"] == "PARTIAL_ROLE_ALIGNMENT_NEEDS_REVIEW":
        expected_verdict = "PARTIAL_ROLE_ALIGNMENT_DO_NOT_TRAIN_YET"
    else:
        expected_verdict = "STOP_ROLE_SPECIALIZATION_HYPOTHESIS"
    checks["verdict_recomputed"] = summary["verdict"] == expected_verdict

    count_checks = {
        "label_samples": len(samples),
        "distribution_rows": len(distribution),
        "weight_rows": len(weights),
        "baseline_prediction_rows": len(predictions),
        "representation_files": len(representation_manifest),
        "probe_metric_rows": len(probe_metrics),
        "probe_prediction_rows": len(probe_predictions),
        "role_comparison_rows": len(comparisons),
        "risk_bin_rows": len(risk_bins),
        "risk_summary_rows": len(risk_summary),
    }
    checks["summary_counts"] = all(
        int(summary["counts"][name]) == int(value)
        for name, value in count_checks.items()
    )

    passed = bool(all(checks.values()))
    payload = {
        "passed": passed,
        "verdict": summary["verdict"],
        "checks": checks,
    }
    path = root / "role_specialization_audit_check.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("DLF role-specialization independent audit failed.")
    print("DLF role-specialization independent audit passed")
    for key, value in checks.items():
        print("{}: {}".format(key, value))
    print("verdict:", summary["verdict"])


if __name__ == "__main__":
    main()
