"""Strict grouped V9.33 hierarchical temporal information experiment."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.hierarchical_temporal_cfcompat_v933 import (
    PRIMARY_VARIANT,
    VARIANT_NAMES,
    VERSION,
    HierarchicalTemporalConfigV933,
    HierarchicalTemporalDatasetV933,
    build_partition_context_bindings,
    extract_anchor_representation_cache,
    group_bootstrap_gain_interval,
    make_temporal_loader,
    paired_metrics,
    predict_hierarchical_temporal_model,
    success_gate,
    train_hierarchical_temporal_model,
)
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.oof_group_splits_v92 import (
    canonical_sample_id,
    conversation_group_id,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    strategy_predictions,
)
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi", "mosei"), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--context-length", type=int, default=3)
    parser.add_argument("--temporal-hidden-dim", type=int, default=64)
    parser.add_argument("--context-hidden-dim", type=int, default=96)
    parser.add_argument("--branch-dropout", type=float, default=0.10)
    parser.add_argument("--temporal-kernel-size", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--wrong-context-preservation-weight", type=float, default=0.10
    )
    parser.add_argument("--missing-temporal-weight", type=float, default=0.10)
    parser.add_argument("--required-gain-vs-current", type=float, default=0.005)
    parser.add_argument("--required-structure-gain", type=float, default=0.002)
    parser.add_argument("--required-nondegrading-folds", type=int, default=4)
    parser.add_argument("--max-worst-fold-degradation", type=float, default=0.005)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_1d(value) -> np.ndarray:
    return torch.as_tensor(value).detach().cpu().double().view(-1).numpy()


def payload_ids(payload: Mapping[str, object]) -> list[str]:
    return [canonical_sample_id(value) for value in payload["sample_ids"]]


def payload_indices(payload: Mapping[str, object]) -> list[int]:
    values = [int(value) for value in payload.get("sample_indices", ())]
    if not values or len(values) != len(set(values)):
        raise RuntimeError("invalid V9.19 sample indices")
    return values


def markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    local = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in local.columns:
        if pd.api.types.is_float_dtype(local[column]):
            local[column] = local[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return "\n".join(
        [
            "|" + "|".join(local.columns) + "|",
            "|" + "|".join(["---"] * len(local.columns)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in local.itertuples(index=False, name=None)
            ],
        ]
    )


def make_config(cli) -> HierarchicalTemporalConfigV933:
    config = HierarchicalTemporalConfigV933(
        context_length=cli.context_length,
        temporal_hidden_dim=cli.temporal_hidden_dim,
        context_hidden_dim=cli.context_hidden_dim,
        branch_dropout=cli.branch_dropout,
        temporal_kernel_size=cli.temporal_kernel_size,
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        grad_clip_norm=cli.grad_clip_norm,
        use_amp=not cli.disable_amp,
        wrong_context_preservation_weight=cli.wrong_context_preservation_weight,
        missing_temporal_weight=cli.missing_temporal_weight,
        required_gain_vs_current=cli.required_gain_vs_current,
        required_structure_gain=cli.required_structure_gain,
        required_nondegrading_folds=cli.required_nondegrading_folds,
        max_worst_fold_degradation=cli.max_worst_fold_degradation,
    )
    config.validate()
    return config


def build_unaligned_args(aligned_args, config_path: Path, dataset: str):
    payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    dataset_root = Path(payload["datasetCommonParams"]["dataset_root_dir"])
    relative = payload["datasetCommonParams"][dataset]["unaligned"]["featurePath"]
    args = copy.deepcopy(aligned_args)
    args["featurePath"] = str(dataset_root / relative)
    args["need_data_aligned"] = False
    args["need_model_aligned"] = False
    args["feature_dims"] = payload["datasetCommonParams"][dataset]["unaligned"][
        "feature_dims"
    ]
    return args


def context_manifest_rows(
    fold: int,
    partition: str,
    bindings: Mapping[int, Mapping[str, object]],
    sample_ids: Sequence[object],
):
    rows = []
    for index, binding in sorted(bindings.items()):
        row = {
            "outer_fold": int(fold),
            "partition": partition,
            "sample_index": int(index),
            "sample_id": canonical_sample_id(sample_ids[index]),
            "video_id": str(binding["video_id"]),
            "segment_index": int(binding["segment_index"]),
        }
        for slot, value in enumerate(binding["ordered_indices"], start=1):
            row[f"ordered_context_index_{slot}"] = int(value)
        for slot, value in enumerate(binding["wrong_indices"], start=1):
            row[f"wrong_context_index_{slot}"] = int(value)
        rows.append(row)
    return rows


def load_or_build_cache(
    cache_path: Path,
    args,
    dataset,
    indices,
    checkpoint: Path,
    checkpoint_sha: str,
    config: HierarchicalTemporalConfigV933,
    seed: int,
    resume: bool,
):
    expected = {
        "version": VERSION,
        "checkpoint_sha256": checkpoint_sha,
        "indices": sorted(set(int(value) for value in indices)),
    }
    if resume and cache_path.is_file():
        cached = torch.load(cache_path, map_location="cpu")
        if cached.get("source") == expected:
            return cached["payload"], True
    payload = extract_anchor_representation_cache(
        args,
        dataset,
        expected["indices"],
        checkpoint,
        config.batch_size,
        config.num_workers,
        seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"source": expected, "payload": payload}, cache_path)
    return payload, False


def main():
    cli = parse_args()
    setup_seed(cli.seed)
    config = make_config(cli)
    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = assign_gpu([cli.gpu])
    args["mode"] = "train"
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = int(cli.seed)
    args["cur_seed"] = int(cli.seed)
    args["batch_size"] = int(config.batch_size)
    unaligned_args = build_unaligned_args(args, Path(cli.config), cli.dataset)

    aligned_dataset = MMDataset(args, mode="train")
    unaligned_dataset = MMDataset(unaligned_args, mode="train")
    if len(aligned_dataset) != len(unaligned_dataset):
        raise RuntimeError("aligned and unaligned sample counts differ")
    all_sample_ids = [canonical_sample_id(value) for value in aligned_dataset.ids]
    unaligned_ids = [canonical_sample_id(value) for value in unaligned_dataset.ids]
    if all_sample_ids != unaligned_ids:
        raise RuntimeError("aligned and unaligned sample IDs are not identical")
    if not np.allclose(
        np.asarray(aligned_dataset.labels["M"]).reshape(-1),
        np.asarray(unaligned_dataset.labels["M"]).reshape(-1),
        atol=1e-6,
    ):
        raise RuntimeError("aligned and unaligned labels differ")

    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v933_hierarchical_temporal_cfcompat"
    )
    output.mkdir(parents=True, exist_ok=True)
    consensus_config = ConsensusConfigV921()

    fold_prediction_frames = []
    fold_metric_rows = []
    checkpoint_rows = []
    source_rows = []
    context_rows = []

    for outer_fold in range(int(cli.outer_folds)):
        fold_dir = root / f"outer_fold_{outer_fold}"
        inner_pool_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        outer_stack_dir = fold_dir / "outer_deployment_stack"
        outer_pool_path = outer_stack_dir / "same_stack_target_pool_v919.pth"
        anchor_checkpoint = outer_stack_dir / "cfcompat_student_best_inner_valid.pth"
        for path in (inner_pool_path, outer_pool_path, anchor_checkpoint):
            if not path.is_file():
                raise FileNotFoundError(path)
        anchor_sha = file_sha256(anchor_checkpoint)
        outer_pool_sha = file_sha256(outer_pool_path)
        inner_pool_sha = file_sha256(inner_pool_path)
        unaligned_sha = file_sha256(Path(unaligned_args.featurePath))
        for role, path, digest in (
            ("inner_oof_pool", inner_pool_path, inner_pool_sha),
            ("outer_pool", outer_pool_path, outer_pool_sha),
            ("anchor_checkpoint", anchor_checkpoint, anchor_sha),
            (
                "unaligned_feature_file",
                Path(unaligned_args.featurePath),
                unaligned_sha,
            ),
        ):
            source_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "role": role,
                    "path": str(path),
                    "sha256": digest,
                }
            )

        inner_payload = torch.load(inner_pool_path, map_location="cpu")
        outer_payload = torch.load(outer_pool_path, map_location="cpu")
        recipe = outer_payload.get("recipe", {})
        train_indices = [int(value) for value in recipe.get("train_indices", ())]
        valid_indices = [int(value) for value in recipe.get("valid_indices", ())]
        target_indices = payload_indices(outer_payload)
        if not train_indices or not valid_indices:
            raise RuntimeError(f"fold {outer_fold} lacks train/valid indices")
        if set(train_indices) & set(valid_indices):
            raise RuntimeError("train/valid overlap")
        if set(train_indices + valid_indices) & set(target_indices):
            raise RuntimeError("outer holdout entered training")

        old_inner = normalize_pool(inner_payload)
        old_outer = normalize_pool(outer_payload)
        old_reference = strategy_predictions(
            old_inner, old_outer, consensus_config
        )
        old_v921 = tensor_1d(
            old_reference["predictions"]["convex_shrinkage"]
        )
        source_anchor = tensor_1d(outer_payload["anchor"])
        target_labels = tensor_1d(outer_payload["labels"])
        target_ids = payload_ids(outer_payload)
        if target_ids != [all_sample_ids[index] for index in target_indices]:
            raise RuntimeError("V9.33 target ID order differs from source pool")

        fold_output = output / f"outer_fold_{outer_fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        union_indices = sorted(set(train_indices + valid_indices + target_indices))
        cache_path = fold_output / "v933_frozen_anchor_representation_cache.pth"
        cache, cache_reused = load_or_build_cache(
            cache_path,
            args,
            aligned_dataset,
            union_indices,
            anchor_checkpoint,
            anchor_sha,
            config,
            int(cli.seed) + 1009 * (outer_fold + 1),
            resume=not cli.no_resume,
        )
        cache_rows = cache["rows"]
        tail_state = cache["tail_state"]
        source_rows.append(
            {
                "outer_fold": int(outer_fold),
                "role": "representation_cache",
                "path": str(cache_path),
                "sha256": file_sha256(cache_path),
            }
        )

        partition_specs = {
            "inner_train": train_indices,
            "inner_valid": valid_indices,
            "outer_holdout": target_indices,
        }
        bindings = {}
        for partition, indices in partition_specs.items():
            local = build_partition_context_bindings(
                all_sample_ids, indices, config.context_length
            )
            bindings[partition] = local
            context_rows.extend(
                context_manifest_rows(
                    outer_fold, partition, local, all_sample_ids
                )
            )

        temporal_datasets = {
            partition: HierarchicalTemporalDatasetV933(
                unaligned_dataset,
                indices,
                cache_rows,
                bindings[partition],
                config.context_length,
            )
            for partition, indices in partition_specs.items()
        }
        source_manifest = {
            "version": VERSION,
            "outer_fold": int(outer_fold),
            "seed": int(cli.seed),
            "anchor_checkpoint": str(anchor_checkpoint),
            "anchor_checkpoint_sha256": anchor_sha,
            "outer_pool_sha256": outer_pool_sha,
            "inner_pool_sha256": inner_pool_sha,
            "unaligned_feature_path": str(unaligned_args.featurePath),
            "unaligned_feature_sha256": unaligned_sha,
            "representation_cache_sha256": file_sha256(cache_path),
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "target_indices": target_indices,
            "outer_labels_used_for_training": False,
        }

        variant_predictions = {}
        primary_diagnostics = None
        for variant in VARIANT_NAMES:
            variant_seed = int(cli.seed) + 100003 * (outer_fold + 1)
            setup_seed(variant_seed)
            train_loader = make_temporal_loader(
                temporal_datasets["inner_train"],
                config.batch_size,
                config.num_workers,
                True,
                variant_seed,
            )
            valid_loader = make_temporal_loader(
                temporal_datasets["inner_valid"],
                config.batch_size,
                config.num_workers,
                False,
                variant_seed,
            )
            target_loader = make_temporal_loader(
                temporal_datasets["outer_holdout"],
                config.batch_size,
                config.num_workers,
                False,
                variant_seed,
            )
            checkpoint = fold_output / f"{variant}_best_v933.pth"
            fitted = train_hierarchical_temporal_model(
                tail_state,
                int(unaligned_dataset.audio.shape[2]),
                int(unaligned_dataset.vision.shape[2]),
                variant,
                config,
                train_loader,
                valid_loader,
                args.device,
                checkpoint,
                source_manifest,
                variant_seed,
                resume=not cli.no_resume,
            )
            prediction = predict_hierarchical_temporal_model(
                fitted["model"], target_loader, args.device, diagnostics=True
            )
            if prediction["sample_id"].tolist() != target_ids:
                raise RuntimeError(f"target order changed for {variant}")
            if prediction["sample_index"].tolist() != target_indices:
                raise RuntimeError(f"target indices changed for {variant}")
            if not np.allclose(prediction["label"], target_labels, atol=1e-6):
                raise RuntimeError(f"target labels changed for {variant}")
            variant_predictions[variant] = prediction["prediction"]
            if variant == PRIMARY_VARIANT:
                primary_diagnostics = prediction
            pd.DataFrame(fitted["history"]).to_csv(
                fold_output / f"{variant}_history_v933.csv", index=False
            )
            checkpoint_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "variant": variant,
                    "checkpoint": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                    "best_epoch": int(fitted["best_epoch"]),
                    "best_valid_mae": float(fitted["best_valid_mae"]),
                    "reused": bool(fitted["reused"]),
                    "representation_cache_reused": bool(cache_reused),
                }
            )
            del fitted["model"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if primary_diagnostics is None:
            raise RuntimeError("primary diagnostics were not produced")

        strategies = {
            "source_cfcompat_anchor": source_anchor,
            "old_v921_convex_shrinkage": old_v921,
            **variant_predictions,
            "hierarchical_wrong_context": primary_diagnostics[
                "wrong_context_prediction"
            ],
            "hierarchical_reversed_time": primary_diagnostics[
                "reversed_time_prediction"
            ],
            "hierarchical_both_invalid": primary_diagnostics[
                "both_invalid_prediction"
            ],
        }
        current = variant_predictions["current_only"]
        for strategy, prediction in strategies.items():
            metric_current = paired_metrics(
                prediction, target_labels, current
            )
            metric_v921 = paired_metrics(
                prediction, target_labels, old_v921
            )
            fold_metric_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "strategy": strategy,
                    "primary": strategy == PRIMARY_VARIANT,
                    "mae": metric_current["mae"],
                    "gain_vs_current_only": metric_current[
                        "gain_vs_baseline"
                    ],
                    "gain_vs_v921": metric_v921["gain_vs_baseline"],
                    "win_rate_vs_current_only": metric_current["win_rate"],
                    "large_harm_rate_010_vs_current_only": metric_current[
                        "large_harm_rate_010"
                    ],
                }
            )

        frame = pd.DataFrame(
            {
                "outer_fold": int(outer_fold),
                "sample_index": target_indices,
                "sample_id": target_ids,
                "group_id": [
                    conversation_group_id(value) for value in target_ids
                ],
                "video_id": primary_diagnostics["video_id"],
                "segment_index": primary_diagnostics["segment_index"],
                "label": target_labels,
                "context_length": primary_diagnostics["context_length"],
                "audio_delta_norm": primary_diagnostics["audio_delta_norm"],
                "vision_delta_norm": primary_diagnostics["vision_delta_norm"],
                "context_delta_norm": primary_diagnostics["context_delta_norm"],
            }
        )
        for strategy, prediction in strategies.items():
            frame[f"prediction_{strategy}"] = prediction
        fold_prediction_frames.append(frame)
        torch.save(
            {
                "version": VERSION,
                "outer_fold": int(outer_fold),
                "source_manifest": source_manifest,
                "sample_indices": target_indices,
                "sample_ids": target_ids,
                "labels": torch.tensor(target_labels, dtype=torch.float32),
                "strategies": {
                    key: torch.tensor(value, dtype=torch.float32)
                    for key, value in strategies.items()
                },
                "provenance": {
                    "outer_labels_used_for_training": False,
                    "expert_router_present": False,
                    "scalar_output_residual_present": False,
                    "deployment_output": "single_augmented_dlf_tail",
                    "ordered_context_is_strictly_past_same_video": True,
                    "unaligned_audio_visual_used_before_scalar_head": True,
                    "wrong_context_and_reversed_time_are_diagnostics": True,
                },
            },
            fold_output / "v933_outer_payload.pth",
        )

    predictions = pd.concat(fold_prediction_frames, ignore_index=True)
    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("V9.33 outer predictions duplicate IDs")
    folds = pd.DataFrame(fold_metric_rows).sort_values(
        ["outer_fold", "mae", "strategy"]
    )
    checkpoints = pd.DataFrame(checkpoint_rows).sort_values(
        ["outer_fold", "variant"]
    )
    sources = pd.DataFrame(source_rows).sort_values(
        ["outer_fold", "role"]
    )
    context_manifest = pd.DataFrame(context_rows).sort_values(
        ["outer_fold", "partition", "sample_index"]
    )

    labels = predictions["label"].to_numpy(dtype=np.float64)
    groups = predictions["group_id"].astype(str).tolist()
    strategy_names = sorted(
        column.removeprefix("prediction_")
        for column in predictions.columns
        if column.startswith("prediction_")
    )
    current = predictions["prediction_current_only"].to_numpy(dtype=np.float64)
    v921 = predictions[
        "prediction_old_v921_convex_shrinkage"
    ].to_numpy(dtype=np.float64)
    aggregate_rows = []
    bootstrap_rows = []
    for offset, strategy in enumerate(strategy_names):
        prediction = predictions[
            f"prediction_{strategy}"
        ].to_numpy(dtype=np.float64)
        metric_current = paired_metrics(prediction, labels, current)
        metric_v921 = paired_metrics(prediction, labels, v921)
        aggregate_rows.append(
            {
                "strategy": strategy,
                "primary": strategy == PRIMARY_VARIANT,
                "mae": metric_current["mae"],
                "gain_vs_current_only": metric_current["gain_vs_baseline"],
                "gain_vs_v921": metric_v921["gain_vs_baseline"],
                "win_rate_vs_current_only": metric_current["win_rate"],
                "large_harm_rate_010_vs_current_only": metric_current[
                    "large_harm_rate_010"
                ],
            }
        )
        for baseline_name, baseline in (
            ("current_only", current),
            ("old_v921_convex_shrinkage", v921),
        ):
            gain = np.abs(baseline - labels) - np.abs(prediction - labels)
            bootstrap_rows.append(
                {
                    "strategy": strategy,
                    "baseline": baseline_name,
                    **group_bootstrap_gain_interval(
                        gain,
                        groups,
                        int(cli.bootstrap_repetitions),
                        int(cli.seed)
                        + 7919 * (offset + 1)
                        + (
                            0
                            if baseline_name == "current_only"
                            else 104729
                        ),
                    ),
                }
            )

    aggregate = pd.DataFrame(aggregate_rows).sort_values("mae")
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(
        ["baseline", "strategy"]
    )
    pivot = folds.pivot(
        index="outer_fold", columns="strategy", values="mae"
    )
    primary_mae = float(
        aggregate.loc[
            aggregate.strategy == PRIMARY_VARIANT, "mae"
        ].iloc[0]
    )
    current_mae = float(
        aggregate.loc[
            aggregate.strategy == "current_only", "mae"
        ].iloc[0]
    )
    v921_mae = float(
        aggregate.loc[
            aggregate.strategy == "old_v921_convex_shrinkage", "mae"
        ].iloc[0]
    )
    wrong_mae = float(
        aggregate.loc[
            aggregate.strategy == "hierarchical_wrong_context", "mae"
        ].iloc[0]
    )
    reversed_mae = float(
        aggregate.loc[
            aggregate.strategy == "hierarchical_reversed_time", "mae"
        ].iloc[0]
    )
    primary_gate = success_gate(
        (pivot["current_only"] - pivot[PRIMARY_VARIANT]).to_numpy(),
        current_mae - primary_mae,
        config.required_gain_vs_current,
        config,
    )
    relevance_gate = success_gate(
        (
            pivot["old_v921_convex_shrinkage"]
            - pivot[PRIMARY_VARIANT]
        ).to_numpy(),
        v921_mae - primary_mae,
        config.required_gain_vs_current,
        config,
    )
    context_gate = success_gate(
        (
            pivot["hierarchical_wrong_context"]
            - pivot[PRIMARY_VARIANT]
        ).to_numpy(),
        wrong_mae - primary_mae,
        config.required_structure_gain,
        config,
    )
    temporal_gate = success_gate(
        (
            pivot["hierarchical_reversed_time"]
            - pivot[PRIMARY_VARIANT]
        ).to_numpy(),
        reversed_mae - primary_mae,
        config.required_structure_gain,
        config,
    )
    if (
        primary_gate["passed"]
        and context_gate["passed"]
        and temporal_gate["passed"]
    ):
        verdict = (
            "hierarchical_temporal_information_supported_"
            "build_end_to_end_cfcompat"
        )
    elif primary_gate["passed"] and (
        context_gate["passed"] or temporal_gate["passed"]
    ):
        verdict = (
            "partial_temporal_information_supported_"
            "expand_only_supported_axis"
        )
    else:
        verdict = (
            "hierarchical_temporal_information_not_supported_do_not_scale"
        )

    summary = {
        "version": VERSION,
        "method": "pre_scalar_hierarchical_temporal_information",
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "sample_count": int(len(predictions)),
        "outer_fold_count": int(cli.outer_folds),
        "bootstrap_repetitions": int(cli.bootstrap_repetitions),
        "variants": list(VARIANT_NAMES),
        "primary_variant": PRIMARY_VARIANT,
        "config": asdict(config),
        "primary_mae": primary_mae,
        "current_only_mae": current_mae,
        "old_v921_mae": v921_mae,
        "wrong_context_mae": wrong_mae,
        "reversed_time_mae": reversed_mae,
        "gain_vs_current_only": current_mae - primary_mae,
        "gain_vs_old_v921": v921_mae - primary_mae,
        "ordered_context_structure_gain": wrong_mae - primary_mae,
        "temporal_order_structure_gain": reversed_mae - primary_mae,
        "primary_gain_gate": primary_gate,
        "project_relevance_gate": relevance_gate,
        "ordered_context_gate": context_gate,
        "temporal_order_gate": temporal_gate,
        "verdict": verdict,
        "provenance": {
            "outer_labels_used_for_training": False,
            "expert_router_present": False,
            "label_defined_experts_present": False,
            "scalar_output_residual_present": False,
            "deployment_output": "single_augmented_dlf_tail",
            "current_and_context_features_extracted_from_frozen_cfcompat": True,
            "unaligned_audio_visual_used_before_scalar_head": True,
            "ordered_context_is_strictly_past_same_video": True,
            "wrong_context_is_deterministic_other_video_control": True,
            "reversed_time_is_within_sample_control": True,
            "outer_results_used_to_choose_primary_variant": False,
        },
    }

    predictions.to_csv(output / "v933_outer_predictions.csv", index=False)
    folds.to_csv(output / "v933_metrics_by_fold.csv", index=False)
    aggregate.to_csv(output / "v933_aggregate_metrics.csv", index=False)
    bootstrap.to_csv(
        output / "v933_group_bootstrap_gain_ci.csv", index=False
    )
    checkpoints.to_csv(
        output / "v933_checkpoint_manifest.csv", index=False
    )
    sources.to_csv(output / "v933_source_manifest.csv", index=False)
    context_manifest.to_csv(
        output / "v933_context_manifest.csv", index=False
    )
    (output / "v933_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report = [
        "# V9.33 Hierarchical Temporal CFCompat Feasibility",
        "",
        "This stage changes the representation before the single DLF scalar head. It does not select experts and does not add a scalar residual to an already produced prediction.",
        "",
        "## Aggregate metrics",
        "",
        markdown_table(
            aggregate,
            [
                "strategy",
                "primary",
                "mae",
                "gain_vs_current_only",
                "gain_vs_v921",
                "win_rate_vs_current_only",
                "large_harm_rate_010_vs_current_only",
            ],
        ),
        "",
        "## Fold metrics",
        "",
        markdown_table(
            folds,
            [
                "outer_fold",
                "strategy",
                "mae",
                "gain_vs_current_only",
                "gain_vs_v921",
            ],
        ),
        "",
        "## Pre-registered decision",
        "",
        f"- Primary variant: `{PRIMARY_VARIANT}`",
        f"- Current-only MAE: `{current_mae:.6f}`",
        f"- Primary MAE: `{primary_mae:.6f}`",
        f"- Gain versus current-only: `{current_mae - primary_mae:+.6f}`",
        f"- Old V9.21 MAE: `{v921_mae:.6f}`",
        f"- Gain versus old V9.21: `{v921_mae - primary_mae:+.6f}`",
        f"- Ordered-context structure gain: `{wrong_mae - primary_mae:+.6f}`",
        f"- Temporal-order structure gain: `{reversed_mae - primary_mae:+.6f}`",
        f"- Primary gate: `{primary_gate['passed']}`",
        f"- Ordered-context gate: `{context_gate['passed']}`",
        f"- Temporal-order gate: `{temporal_gate['passed']}`",
        f"- Verdict: `{verdict}`",
        "",
        "Do not scale this into an end-to-end MOSEI model unless the correct ordered inputs beat both invalid controls and the current-only control under the fixed gates.",
    ]
    (output / "v933_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.33 HIERARCHICAL TEMPORAL FEASIBILITY COMPLETE")
    print(
        "primary / current-only / old V9.21 MAE:",
        f"{primary_mae:.6f}",
        f"{current_mae:.6f}",
        f"{v921_mae:.6f}",
    )
    print(
        "gain vs current / V9.21:",
        f"{current_mae-primary_mae:+.6f}",
        f"{v921_mae-primary_mae:+.6f}",
    )
    print(
        "context / temporal structure gain:",
        f"{wrong_mae-primary_mae:+.6f}",
        f"{reversed_mae-primary_mae:+.6f}",
    )
    print("verdict:", verdict)
    print("report:", output / "v933_report.md")


if __name__ == "__main__":
    main()
