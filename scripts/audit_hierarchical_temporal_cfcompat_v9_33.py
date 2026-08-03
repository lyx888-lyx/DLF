"""Independent artifact audit for V9.33 hierarchical temporal feasibility."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.hierarchical_temporal_cfcompat_v933 import (
    PRIMARY_VARIANT,
    VARIANT_NAMES,
    VERSION,
    HierarchicalTemporalConfigV933,
    group_bootstrap_gain_interval,
    paired_metrics,
    success_gate,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    cli = parse_args()
    root = Path(cli.result_dir)
    summary = json.loads(
        (root / "v933_summary.json").read_text(encoding="utf-8")
    )
    predictions = pd.read_csv(root / "v933_outer_predictions.csv")
    folds = pd.read_csv(root / "v933_metrics_by_fold.csv")
    aggregate = pd.read_csv(root / "v933_aggregate_metrics.csv")
    bootstrap = pd.read_csv(root / "v933_group_bootstrap_gain_ci.csv")
    checkpoints = pd.read_csv(root / "v933_checkpoint_manifest.csv")
    sources = pd.read_csv(root / "v933_source_manifest.csv")
    context = pd.read_csv(root / "v933_context_manifest.csv").fillna(-1)
    config = HierarchicalTemporalConfigV933(**summary["config"])
    config.validate()
    checks = {}

    if summary["version"] != VERSION:
        raise AssertionError("summary version mismatch")
    if tuple(summary["variants"]) != tuple(VARIANT_NAMES):
        raise AssertionError("variant registry mismatch")
    if summary["primary_variant"] != PRIMARY_VARIANT:
        raise AssertionError("primary variant mismatch")
    checks["version_and_registry_verified"] = True

    if predictions.sample_id.astype(str).duplicated().any():
        raise AssertionError("duplicate outer sample IDs")
    if len(predictions) != int(summary["sample_count"]):
        raise AssertionError("sample count mismatch")
    if predictions.groupby("sample_id").outer_fold.nunique().max() != 1:
        raise AssertionError("sample appears in multiple outer folds")
    checks["unique_outer_predictions"] = True

    for row in sources.itertuples(index=False):
        path = Path(str(row.path))
        if not path.is_file() or sha256(path) != str(row.sha256):
            raise AssertionError(f"source checksum mismatch: {path}")
    for row in checkpoints.itertuples(index=False):
        path = Path(str(row.checkpoint))
        if not path.is_file() or sha256(path) != str(row.sha256):
            raise AssertionError(f"checkpoint checksum mismatch: {path}")
        payload = torch.load(path, map_location="cpu")
        if payload.get("version") != VERSION:
            raise AssertionError("checkpoint version mismatch")
        if payload.get("variant") != str(row.variant):
            raise AssertionError("checkpoint variant mismatch")
        if payload.get("deployment_output") != "single_augmented_dlf_tail":
            raise AssertionError("checkpoint deployment path mismatch")
        if payload.get("expert_router_present") is not False:
            raise AssertionError("checkpoint claims an expert router")
        if payload.get("scalar_output_residual_present") is not False:
            raise AssertionError("checkpoint claims a scalar output residual")
    checks["source_and_checkpoint_checksums_verified"] = True

    context_columns = sorted(
        column
        for column in context.columns
        if column.startswith("ordered_context_index_")
    )
    wrong_columns = sorted(
        column
        for column in context.columns
        if column.startswith("wrong_context_index_")
    )
    if (
        len(context_columns) != config.context_length
        or len(wrong_columns) != config.context_length
    ):
        raise AssertionError("context manifest width mismatch")
    identity = {
        (int(row.outer_fold), str(row.partition), int(row.sample_index)): (
            str(row.video_id),
            int(row.segment_index),
        )
        for row in context.itertuples(index=False)
    }
    for row in context.itertuples(index=False):
        current_video = str(row.video_id)
        current_segment = int(row.segment_index)
        for slot, column in enumerate(context_columns, start=1):
            index = int(getattr(row, column))
            if index < 0:
                continue
            key = (int(row.outer_fold), str(row.partition), index)
            if key not in identity:
                raise AssertionError("ordered context crosses partition")
            video, segment = identity[key]
            expected_offset = config.context_length - slot + 1
            if (
                video != current_video
                or segment != current_segment - expected_offset
            ):
                raise AssertionError(
                    "ordered context is not exact causal same-video history"
                )
        for column in wrong_columns:
            index = int(getattr(row, column))
            if index < 0:
                continue
            key = (int(row.outer_fold), str(row.partition), index)
            if key not in identity:
                raise AssertionError("wrong context crosses partition")
            video, _ = identity[key]
            if video == current_video:
                raise AssertionError(
                    "wrong context came from the current video"
                )
    checks["causal_and_wrong_context_bindings_verified"] = True

    strategy_names = sorted(
        column.removeprefix("prediction_")
        for column in predictions.columns
        if column.startswith("prediction_")
    )
    labels = predictions.label.to_numpy(dtype=float)
    current_prediction = predictions.prediction_current_only.to_numpy(
        dtype=float
    )
    v921_prediction = predictions[
        "prediction_old_v921_convex_shrinkage"
    ].to_numpy(dtype=float)
    recomputed_aggregate = []
    for strategy in strategy_names:
        prediction = predictions[
            f"prediction_{strategy}"
        ].to_numpy(dtype=float)
        current_metrics = paired_metrics(
            prediction, labels, current_prediction
        )
        v921_metrics = paired_metrics(
            prediction, labels, v921_prediction
        )
        recomputed_aggregate.append(
            {
                "strategy": strategy,
                "primary": strategy == PRIMARY_VARIANT,
                "mae": current_metrics["mae"],
                "gain_vs_current_only": current_metrics[
                    "gain_vs_baseline"
                ],
                "gain_vs_v921": v921_metrics["gain_vs_baseline"],
                "win_rate_vs_current_only": current_metrics["win_rate"],
                "large_harm_rate_010_vs_current_only": current_metrics[
                    "large_harm_rate_010"
                ],
            }
        )
    recomputed_aggregate = pd.DataFrame(recomputed_aggregate).sort_values(
        "strategy"
    )
    observed_aggregate = aggregate.sort_values("strategy")
    for column in (
        "mae",
        "gain_vs_current_only",
        "gain_vs_v921",
        "win_rate_vs_current_only",
        "large_harm_rate_010_vs_current_only",
    ):
        if not np.allclose(
            recomputed_aggregate[column],
            observed_aggregate[column],
            atol=1e-9,
        ):
            raise AssertionError(f"aggregate mismatch: {column}")
    checks["aggregate_metrics_recomputed"] = True

    fold_rows = []
    for outer_fold, local in predictions.groupby("outer_fold", sort=True):
        local_labels = local.label.to_numpy(dtype=float)
        local_current = local.prediction_current_only.to_numpy(dtype=float)
        local_v921 = local[
            "prediction_old_v921_convex_shrinkage"
        ].to_numpy(dtype=float)
        for strategy in strategy_names:
            prediction = local[
                f"prediction_{strategy}"
            ].to_numpy(dtype=float)
            metric_current = paired_metrics(
                prediction, local_labels, local_current
            )
            metric_v921 = paired_metrics(
                prediction, local_labels, local_v921
            )
            fold_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "strategy": strategy,
                    "mae": metric_current["mae"],
                    "gain_vs_current_only": metric_current[
                        "gain_vs_baseline"
                    ],
                    "gain_vs_v921": metric_v921["gain_vs_baseline"],
                }
            )
    recomputed_folds = pd.DataFrame(fold_rows).sort_values(
        ["outer_fold", "strategy"]
    )
    observed_folds = folds.sort_values(["outer_fold", "strategy"])
    for column in ("mae", "gain_vs_current_only", "gain_vs_v921"):
        if not np.allclose(
            recomputed_folds[column], observed_folds[column], atol=1e-9
        ):
            raise AssertionError(f"fold metric mismatch: {column}")
    checks["fold_metrics_recomputed"] = True

    for outer_fold in sorted(predictions.outer_fold.unique()):
        payload_path = (
            root
            / f"outer_fold_{int(outer_fold)}"
            / "v933_outer_payload.pth"
        )
        payload = torch.load(payload_path, map_location="cpu")
        if payload.get("version") != VERSION:
            raise AssertionError("outer payload version mismatch")
        local = predictions[predictions.outer_fold == outer_fold]
        if payload["sample_ids"] != local.sample_id.astype(str).tolist():
            raise AssertionError("outer payload ID mismatch")
        if not np.allclose(
            torch.as_tensor(payload["labels"]).view(-1).numpy(),
            local.label.to_numpy(dtype=float),
            atol=1e-6,
        ):
            raise AssertionError("outer payload label mismatch")
        for strategy, values in payload["strategies"].items():
            if not np.allclose(
                torch.as_tensor(values).view(-1).numpy(),
                local[f"prediction_{strategy}"].to_numpy(dtype=float),
                atol=1e-6,
            ):
                raise AssertionError(
                    f"outer payload strategy mismatch: {strategy}"
                )
        provenance = payload.get("provenance", {})
        if not (
            provenance.get("outer_labels_used_for_training") is False
            and provenance.get("expert_router_present") is False
            and provenance.get("scalar_output_residual_present") is False
            and provenance.get("deployment_output")
            == "single_augmented_dlf_tail"
            and provenance.get(
                "ordered_context_is_strictly_past_same_video"
            )
            is True
            and provenance.get(
                "unaligned_audio_visual_used_before_scalar_head"
            )
            is True
        ):
            raise AssertionError("outer payload provenance mismatch")
    checks["fold_payloads_recomputed"] = True

    groups = predictions.group_id.astype(str).tolist()
    expected_bootstrap = []
    configured_seed = int(summary.get("seed", 1111))
    for offset, strategy in enumerate(strategy_names):
        prediction = predictions[
            f"prediction_{strategy}"
        ].to_numpy(dtype=float)
        for baseline_name, baseline in (
            ("current_only", current_prediction),
            ("old_v921_convex_shrinkage", v921_prediction),
        ):
            gain = np.abs(baseline - labels) - np.abs(
                prediction - labels
            )
            expected_bootstrap.append(
                {
                    "strategy": strategy,
                    "baseline": baseline_name,
                    **group_bootstrap_gain_interval(
                        gain,
                        groups,
                        int(summary["bootstrap_repetitions"]),
                        configured_seed
                        + 7919 * (offset + 1)
                        + (
                            0
                            if baseline_name == "current_only"
                            else 104729
                        ),
                    ),
                }
            )
    expected_bootstrap = pd.DataFrame(expected_bootstrap).sort_values(
        ["baseline", "strategy"]
    )
    observed_bootstrap = bootstrap.sort_values(
        ["baseline", "strategy"]
    )
    for column in (
        "gain_ci_low",
        "gain_ci_high",
        "bootstrap_positive_probability",
    ):
        if not np.allclose(
            expected_bootstrap[column],
            observed_bootstrap[column],
            atol=1e-9,
        ):
            raise AssertionError(f"bootstrap mismatch: {column}")
    checks["bootstrap_intervals_recomputed"] = True

    by_strategy = aggregate.set_index("strategy")
    pivot = folds.pivot(
        index="outer_fold", columns="strategy", values="mae"
    )
    primary_mae = float(by_strategy.loc[PRIMARY_VARIANT, "mae"])
    current_mae = float(by_strategy.loc["current_only", "mae"])
    v921_mae = float(
        by_strategy.loc["old_v921_convex_shrinkage", "mae"]
    )
    wrong_mae = float(
        by_strategy.loc["hierarchical_wrong_context", "mae"]
    )
    reversed_mae = float(
        by_strategy.loc["hierarchical_reversed_time", "mae"]
    )
    gates = {
        "primary_gain_gate": success_gate(
            (pivot["current_only"] - pivot[PRIMARY_VARIANT]).to_numpy(),
            current_mae - primary_mae,
            config.required_gain_vs_current,
            config,
        ),
        "project_relevance_gate": success_gate(
            (
                pivot["old_v921_convex_shrinkage"]
                - pivot[PRIMARY_VARIANT]
            ).to_numpy(),
            v921_mae - primary_mae,
            config.required_gain_vs_current,
            config,
        ),
        "ordered_context_gate": success_gate(
            (
                pivot["hierarchical_wrong_context"]
                - pivot[PRIMARY_VARIANT]
            ).to_numpy(),
            wrong_mae - primary_mae,
            config.required_structure_gain,
            config,
        ),
        "temporal_order_gate": success_gate(
            (
                pivot["hierarchical_reversed_time"]
                - pivot[PRIMARY_VARIANT]
            ).to_numpy(),
            reversed_mae - primary_mae,
            config.required_structure_gain,
            config,
        ),
    }
    for name, value in gates.items():
        if value != summary[name]:
            raise AssertionError(f"summary gate mismatch: {name}")
    checks["pre_registered_gates_recomputed"] = True

    provenance = summary.get("provenance", {})
    if not (
        provenance.get("outer_labels_used_for_training") is False
        and provenance.get("expert_router_present") is False
        and provenance.get("label_defined_experts_present") is False
        and provenance.get("scalar_output_residual_present") is False
        and provenance.get("deployment_output")
        == "single_augmented_dlf_tail"
        and provenance.get(
            "current_and_context_features_extracted_from_frozen_cfcompat"
        )
        is True
        and provenance.get(
            "unaligned_audio_visual_used_before_scalar_head"
        )
        is True
        and provenance.get(
            "ordered_context_is_strictly_past_same_video"
        )
        is True
        and provenance.get(
            "wrong_context_is_deterministic_other_video_control"
        )
        is True
        and provenance.get("reversed_time_is_within_sample_control")
        is True
        and provenance.get("outer_results_used_to_choose_primary_variant")
        is False
    ):
        raise AssertionError("summary provenance mismatch")
    checks["no_expert_router_or_scalar_output_patch"] = True

    result = {
        "version": VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "primary_variant": PRIMARY_VARIANT,
        "primary_mae": primary_mae,
        "current_only_mae": current_mae,
        "old_v921_mae": v921_mae,
        "verdict": summary["verdict"],
        **gates,
    }
    (root / "v933_audit_check.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("V9.33 INDEPENDENT AUDIT PASSED")
    for key, value in checks.items():
        print(f"{key}: {value}")
    print("verdict:", summary["verdict"])
    print("audit:", root / "v933_audit_check.json")


if __name__ == "__main__":
    main()
