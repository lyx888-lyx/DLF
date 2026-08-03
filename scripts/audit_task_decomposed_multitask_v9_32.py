"""Independent audit for V9.32 task-decomposed multitask DLF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.task_decomposed_multitask_v932 import (
    VARIANT_NAMES,
    VERSION,
    TaskDecomposedConfigV932,
    paired_prediction_metrics,
    success_gate,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def close(left, right, atol=1e-7):
    return np.allclose(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        atol=atol,
        rtol=atol,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    root = Path(parser.parse_args().result_dir)
    files = {
        "summary": root / "v932_summary.json",
        "predictions": root / "v932_outer_predictions.csv",
        "fold": root / "v932_metrics_by_fold.csv",
        "aggregate": root / "v932_aggregate_metrics.csv",
        "checkpoints": root / "v932_checkpoint_manifest.csv",
        "sources": root / "v932_source_manifest.csv",
    }
    for path in files.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    predictions = pd.read_csv(files["predictions"])
    fold = pd.read_csv(files["fold"])
    aggregate = pd.read_csv(files["aggregate"])
    checkpoints = pd.read_csv(files["checkpoints"])
    sources = pd.read_csv(files["sources"])
    checks = {}

    if summary.get("version") != VERSION:
        raise AssertionError("version mismatch")
    if predictions["sample_id"].astype(str).duplicated().any():
        raise AssertionError("duplicate outer sample IDs")
    checks["unique_outer_predictions"] = True

    for row in sources.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file() or sha256(path) != str(row.sha256):
            raise AssertionError(f"source checksum mismatch: {path}")
    checks["source_checksums_verified"] = True

    for row in checkpoints.itertuples(index=False):
        path = Path(row.checkpoint)
        if not path.is_file() or sha256(path) != str(row.sha256):
            raise AssertionError(f"checkpoint checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu")
        if not (
            payload.get("version") == VERSION
            and payload.get("variant") == str(row.variant)
            and payload.get("deployment_output")
            == "backbone_regression_head"
            and payload.get("auxiliary_outputs_used_at_inference") is False
            and payload.get("winner_router_present") is False
            and payload.get("scalar_residual_correction_present") is False
            and payload.get("source_manifest", {}).get(
                "outer_labels_used_for_training"
            )
            is False
        ):
            raise AssertionError(f"checkpoint provenance mismatch: {path}")
    checks["checkpoint_provenance_verified"] = True

    for outer_fold, local in predictions.groupby("outer_fold", sort=True):
        payload_path = (
            root / f"outer_fold_{int(outer_fold)}" / "v932_outer_payload.pth"
        )
        payload = torch.load(payload_path, map_location="cpu")
        provenance = payload.get("provenance", {})
        if not (
            payload.get("version") == VERSION
            and provenance.get("deployment_prediction")
            == "backbone_regression_head_only"
            and provenance.get("intensity_used_in_deployment_prediction")
            is False
            and provenance.get("ordinal_used_in_deployment_prediction")
            is False
            and provenance.get("outer_labels_used_for_training") is False
            and provenance.get("winner_router_present") is False
            and provenance.get("label_defined_sample_experts_present")
            is False
            and provenance.get("scalar_residual_correction_present")
            is False
        ):
            raise AssertionError("fold provenance mismatch")
        local = local.reset_index(drop=True)
        if list(map(str, payload["sample_ids"])) != local[
            "sample_id"
        ].astype(str).tolist():
            raise AssertionError("fold sample order mismatch")
        if not close(
            torch.as_tensor(payload["labels"]).view(-1).numpy(),
            local["label"].to_numpy(),
        ):
            raise AssertionError("fold labels mismatch")
        for strategy, values in payload["strategies"].items():
            if not close(
                torch.as_tensor(values).view(-1).numpy(),
                local[f"prediction_{strategy}"].to_numpy(),
            ):
                raise AssertionError(
                    f"fold prediction mismatch: {strategy}"
                )
    checks["fold_payloads_recomputed"] = True

    labels = predictions["label"].to_numpy(dtype=np.float64)
    control = predictions["prediction_regression_only"].to_numpy(
        dtype=np.float64
    )
    v921 = predictions[
        "prediction_old_v921_convex_shrinkage"
    ].to_numpy(dtype=np.float64)
    strategies = sorted(
        column.removeprefix("prediction_")
        for column in predictions.columns
        if column.startswith("prediction_")
    )

    expected_aggregate = []
    expected_fold = []
    for strategy in strategies:
        prediction = predictions[f"prediction_{strategy}"].to_numpy(
            dtype=np.float64
        )
        metric_control = paired_prediction_metrics(
            prediction, labels, control
        )
        metric_v921 = paired_prediction_metrics(prediction, labels, v921)
        expected_aggregate.append(
            {
                "strategy": strategy,
                "mae": np.abs(prediction - labels).mean(),
                "gain_vs_regression_only": metric_control[
                    "gain_vs_baseline"
                ],
                "gain_vs_v921": metric_v921["gain_vs_baseline"],
            }
        )
        for outer_fold, local in predictions.groupby(
            "outer_fold", sort=True
        ):
            local_labels = local["label"].to_numpy(dtype=np.float64)
            local_prediction = local[
                f"prediction_{strategy}"
            ].to_numpy(dtype=np.float64)
            local_control = local[
                "prediction_regression_only"
            ].to_numpy(dtype=np.float64)
            local_v921 = local[
                "prediction_old_v921_convex_shrinkage"
            ].to_numpy(dtype=np.float64)
            expected_fold.append(
                {
                    "outer_fold": int(outer_fold),
                    "strategy": strategy,
                    "mae": np.abs(
                        local_prediction - local_labels
                    ).mean(),
                    "gain_vs_regression_only":
                    paired_prediction_metrics(
                        local_prediction, local_labels, local_control
                    )["gain_vs_baseline"],
                    "gain_vs_v921": paired_prediction_metrics(
                        local_prediction, local_labels, local_v921
                    )["gain_vs_baseline"],
                }
            )

    for row in expected_aggregate:
        stored = aggregate[aggregate["strategy"] == row["strategy"]].iloc[0]
        for key in ("mae", "gain_vs_regression_only", "gain_vs_v921"):
            if not close([stored[key]], [row[key]]):
                raise AssertionError(f"aggregate mismatch: {row['strategy']} {key}")
    checks["aggregate_metrics_recomputed"] = True

    for row in expected_fold:
        stored = fold[
            (fold["outer_fold"].astype(int) == row["outer_fold"])
            & (fold["strategy"] == row["strategy"])
        ].iloc[0]
        for key in ("mae", "gain_vs_regression_only", "gain_vs_v921"):
            if not close([stored[key]], [row[key]]):
                raise AssertionError(
                    f"fold mismatch: {row['outer_fold']} "
                    f"{row['strategy']} {key}"
                )
    checks["fold_metrics_recomputed"] = True

    config = TaskDecomposedConfigV932(**summary["config"])
    pivot = fold.pivot(
        index="outer_fold", columns="strategy", values="mae"
    )
    primary = summary["primary_variant"]
    primary_mae = float(
        aggregate.loc[aggregate["strategy"] == primary, "mae"].iloc[0]
    )
    control_mae = float(
        aggregate.loc[
            aggregate["strategy"] == "regression_only", "mae"
        ].iloc[0]
    )
    v921_mae = float(
        aggregate.loc[
            aggregate["strategy"] == "old_v921_convex_shrinkage", "mae"
        ].iloc[0]
    )
    internal = success_gate(
        (pivot["regression_only"] - pivot[primary]).to_numpy(),
        control_mae - primary_mae,
        config,
    )
    relevance = success_gate(
        (
            pivot["old_v921_convex_shrinkage"] - pivot[primary]
        ).to_numpy(),
        v921_mae - primary_mae,
        config,
    )
    if internal != summary["internal_auxiliary_gate"]:
        raise AssertionError("internal gate mismatch")
    if relevance != summary["project_relevance_gate"]:
        raise AssertionError("project relevance gate mismatch")
    checks["pre_registered_gates_recomputed"] = True

    provenance = summary.get("provenance", {})
    if not (
        tuple(summary.get("variants", ())) == tuple(VARIANT_NAMES)
        and primary == "ordinal_intensity"
        and provenance.get("training_labels_define_auxiliary_targets_only")
        is True
        and provenance.get("test_samples_follow_identical_computation_path")
        is True
        and provenance.get("deployment_output")
        == "backbone_regression_head_only"
        and provenance.get("intensity_or_ordinal_hard_composition") is False
        and provenance.get("outer_labels_used_for_training") is False
        and provenance.get("winner_router_present") is False
        and provenance.get("label_defined_sample_experts_present") is False
        and provenance.get("scalar_residual_correction_present") is False
        and provenance.get("outer_results_used_to_choose_primary_variant")
        is False
    ):
        raise AssertionError("summary provenance mismatch")
    checks["no_router_partition_residual_or_hard_composition"] = True

    result = {
        "version": VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "primary_variant": primary,
        "primary_mae": primary_mae,
        "regression_only_mae": control_mae,
        "old_v921_mae": v921_mae,
        "internal_auxiliary_gate": internal,
        "project_relevance_gate": relevance,
    }
    (root / "v932_audit_check.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("V9.32 INDEPENDENT AUDIT PASSED")
    for key, value in checks.items():
        print(f"{key}: {value}")
    print("audit:", root / "v932_audit_check.json")


if __name__ == "__main__":
    main()
