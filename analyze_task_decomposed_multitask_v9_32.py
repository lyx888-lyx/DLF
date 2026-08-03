"""Strict grouped V9.32 task-decomposed multitask DLF experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.oof_group_splits_v92 import (
    build_subset_loader,
    canonical_sample_id,
    conversation_group_id,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    group_bootstrap_gain_interval,
    strategy_predictions,
)
from trains.singleTask.task_decomposed_multitask_v932 import (
    ORDINAL_THRESHOLDS,
    VARIANT_NAMES,
    VERSION,
    TaskDecomposedConfigV932,
    paired_prediction_metrics,
    predict_task_model,
    prediction_diagnostics,
    success_gate,
    train_task_model,
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--early-stop", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument(
        "--trainable-scope",
        choices=("fusion_tail", "full"),
        default="fusion_tail",
    )
    parser.add_argument("--auxiliary-hidden-dim", type=int, default=128)
    parser.add_argument("--auxiliary-dropout", type=float, default=0.15)
    parser.add_argument("--ordinal-weight", type=float, default=0.20)
    parser.add_argument("--intensity-weight", type=float, default=0.10)
    parser.add_argument("--ordinal-monotonic-weight", type=float, default=0.05)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--required-gain-vs-control", type=float, default=0.005)
    parser.add_argument("--required-nondegrading-folds", type=int, default=4)
    parser.add_argument("--max-worst-fold-degradation", type=float, default=0.005)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def tensor_1d(value) -> np.ndarray:
    return torch.as_tensor(value).detach().cpu().double().view(-1).numpy()


def payload_ids(payload: Mapping[str, object]) -> list[str]:
    return [canonical_sample_id(value) for value in payload["sample_ids"]]


def payload_indices(payload: Mapping[str, object]) -> list[int]:
    if "sample_indices" not in payload:
        raise RuntimeError("V9.19 pool lacks sample_indices")
    values = [int(value) for value in payload["sample_indices"]]
    if len(values) != len(payload["sample_ids"]) or len(set(values)) != len(values):
        raise RuntimeError("invalid V9.19 sample indices")
    return values


def make_config(cli) -> TaskDecomposedConfigV932:
    config = TaskDecomposedConfigV932(
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        grad_clip_norm=cli.grad_clip_norm,
        update_epochs=cli.update_epochs,
        use_amp=not cli.disable_amp,
        trainable_scope=cli.trainable_scope,
        auxiliary_hidden_dim=cli.auxiliary_hidden_dim,
        auxiliary_dropout=cli.auxiliary_dropout,
        ordinal_weight=cli.ordinal_weight,
        intensity_weight=cli.intensity_weight,
        ordinal_monotonic_weight=cli.ordinal_monotonic_weight,
        required_gain_vs_control=cli.required_gain_vs_control,
        required_nondegrading_folds=cli.required_nondegrading_folds,
        max_worst_fold_degradation=cli.max_worst_fold_degradation,
    )
    config.validate()
    return config


def main():
    cli = parse_args()
    setup_seed(cli.seed)
    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = assign_gpu([cli.gpu])
    args["mode"] = "train"
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = int(cli.seed)
    args["cur_seed"] = int(cli.seed)
    args["batch_size"] = int(cli.batch_size)

    config = make_config(cli)
    dataset = MMDataset(args, mode="train")
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v932_task_decomposed_multitask"
    )
    output.mkdir(parents=True, exist_ok=True)
    consensus_config = ConsensusConfigV921()

    prediction_frames = []
    fold_rows = []
    checkpoint_rows = []
    source_rows = []

    for outer_fold in range(int(cli.outer_folds)):
        fold_dir = root / f"outer_fold_{outer_fold}"
        inner_pool_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        outer_stack_dir = fold_dir / "outer_deployment_stack"
        outer_pool_path = outer_stack_dir / "same_stack_target_pool_v919.pth"
        anchor_checkpoint = (
            outer_stack_dir / "cfcompat_student_best_inner_valid.pth"
        )
        for role, path in (
            ("inner_oof_pool", inner_pool_path),
            ("outer_pool", outer_pool_path),
            ("anchor_checkpoint", anchor_checkpoint),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "role": role,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )

        inner_payload = torch.load(inner_pool_path, map_location="cpu")
        outer_payload = torch.load(outer_pool_path, map_location="cpu")
        recipe = outer_payload.get("recipe", {})
        train_indices = [int(value) for value in recipe.get("train_indices", ())]
        valid_indices = [int(value) for value in recipe.get("valid_indices", ())]
        target_indices = payload_indices(outer_payload)
        if not train_indices or not valid_indices:
            raise RuntimeError(f"fold {outer_fold} lacks train/valid recipe")
        if set(train_indices) & set(valid_indices):
            raise RuntimeError("V9.32 train/valid overlap")
        if set(train_indices + valid_indices) & set(target_indices):
            raise RuntimeError("V9.32 outer holdout entered training")

        old_inner = normalize_pool(inner_payload)
        old_outer = normalize_pool(outer_payload)
        old_reference = strategy_predictions(
            old_inner, old_outer, consensus_config
        )
        old_v921 = tensor_1d(
            old_reference["predictions"]["convex_shrinkage"]
        )
        source_anchor = tensor_1d(outer_payload["anchor"])
        labels = tensor_1d(outer_payload["labels"])
        sample_ids = payload_ids(outer_payload)
        group_ids = [conversation_group_id(value) for value in sample_ids]
        if not (
            len(labels)
            == len(sample_ids)
            == len(old_v921)
            == len(source_anchor)
        ):
            raise RuntimeError("V9.32 source array alignment failed")

        fold_output = output / f"outer_fold_{outer_fold}"
        fold_output.mkdir(parents=True, exist_ok=True)
        source_manifest = {
            "version": VERSION,
            "outer_fold": int(outer_fold),
            "anchor_checkpoint": str(anchor_checkpoint),
            "anchor_checkpoint_sha256": file_sha256(anchor_checkpoint),
            "outer_pool": str(outer_pool_path),
            "outer_pool_sha256": file_sha256(outer_pool_path),
            "train_indices": train_indices,
            "valid_indices": valid_indices,
            "target_indices": target_indices,
            "outer_labels_used_for_training": False,
        }

        variant_predictions: Dict[str, np.ndarray] = {}
        variant_aux: Dict[str, Mapping[str, object]] = {}
        for variant in VARIANT_NAMES:
            variant_seed = int(cli.seed) + 100003 * (outer_fold + 1)
            setup_seed(variant_seed)
            train_loader = build_subset_loader(
                dataset,
                train_indices,
                config.batch_size,
                config.num_workers,
                True,
                variant_seed,
            )
            valid_loader = build_subset_loader(
                dataset,
                valid_indices,
                config.batch_size,
                config.num_workers,
                False,
                variant_seed,
            )
            target_loader = build_subset_loader(
                dataset,
                target_indices,
                config.batch_size,
                config.num_workers,
                False,
                variant_seed,
            )
            checkpoint = fold_output / f"{variant}_best_v932.pth"
            fitted = train_task_model(
                args,
                train_loader,
                valid_loader,
                anchor_checkpoint,
                variant,
                config,
                checkpoint,
                source_manifest,
                variant_seed,
                resume=not cli.no_resume,
            )
            predicted = predict_task_model(
                fitted["model"], target_loader, args.device
            )
            predicted_ids = [
                canonical_sample_id(value)
                for value in predicted["sample_ids"]
            ]
            if predicted_ids != sample_ids:
                raise RuntimeError(
                    f"V9.32 target order changed for {variant}"
                )
            if [int(value) for value in predicted["sample_indices"]] != target_indices:
                raise RuntimeError(
                    f"V9.32 target indices changed for {variant}"
                )
            predicted_labels = tensor_1d(predicted["labels"])
            if not np.allclose(predicted_labels, labels, atol=1e-6):
                raise RuntimeError(
                    f"V9.32 target labels changed for {variant}"
                )
            diagnostics = prediction_diagnostics(predicted)
            variant_predictions[variant] = tensor_1d(
                predicted["regression"]
            )
            variant_aux[variant] = predicted
            pd.DataFrame(fitted["history"]).to_csv(
                fold_output / f"{variant}_history_v932.csv", index=False
            )
            checkpoint_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "variant": variant,
                    "checkpoint": str(checkpoint),
                    "sha256": file_sha256(checkpoint),
                    "best_epoch": int(fitted["best_epoch"]),
                    "best_valid_mae": float(fitted["best_valid_mae"]),
                    "trainable_parameters": int(
                        fitted["trainable_parameters"]
                    ),
                    "reused": bool(fitted["reused"]),
                    **{
                        f"outer_{key}": value
                        for key, value in diagnostics.items()
                    },
                }
            )
            del fitted["model"]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        strategies = {
            "source_cfcompat_anchor": source_anchor,
            "old_v921_convex_shrinkage": old_v921,
            **variant_predictions,
        }
        control = variant_predictions["regression_only"]
        for strategy, prediction in strategies.items():
            metrics_control = paired_prediction_metrics(
                prediction, labels, control
            )
            metrics_v921 = paired_prediction_metrics(
                prediction, labels, old_v921
            )
            fold_rows.append(
                {
                    "outer_fold": int(outer_fold),
                    "strategy": strategy,
                    "deployable": True,
                    "primary": strategy == "ordinal_intensity",
                    "mae": float(np.abs(prediction - labels).mean()),
                    "gain_vs_regression_only": metrics_control[
                        "gain_vs_baseline"
                    ],
                    "gain_vs_v921": metrics_v921["gain_vs_baseline"],
                    "win_rate_vs_regression_only": metrics_control[
                        "win_rate_vs_baseline"
                    ],
                    "large_harm_rate_010_vs_regression_only": metrics_control[
                        "large_harm_rate_010"
                    ],
                }
            )

        frame = pd.DataFrame(
            {
                "outer_fold": int(outer_fold),
                "sample_index": target_indices,
                "sample_id": sample_ids,
                "group_id": group_ids,
                "label": labels,
            }
        )
        for strategy, prediction in strategies.items():
            frame[f"prediction_{strategy}"] = prediction
        for variant in VARIANT_NAMES:
            payload = variant_aux[variant]
            frame[f"intensity_{variant}"] = tensor_1d(
                payload["intensity"]
            )
            probabilities = torch.as_tensor(
                payload["ordinal_probabilities"]
            ).detach().cpu().double().numpy()
            for index, threshold in enumerate(ORDINAL_THRESHOLDS):
                token = str(threshold).replace("-", "neg").replace(".", "p")
                frame[
                    f"ordinal_prob_gt_{token}_{variant}"
                ] = probabilities[:, index]
        prediction_frames.append(frame)

        torch.save(
            {
                "version": VERSION,
                "outer_fold": int(outer_fold),
                "source_manifest": source_manifest,
                "sample_indices": target_indices,
                "sample_ids": sample_ids,
                "group_ids": group_ids,
                "labels": torch.tensor(labels, dtype=torch.float32),
                "strategies": {
                    name: torch.tensor(value, dtype=torch.float32)
                    for name, value in strategies.items()
                },
                "provenance": {
                    "deployment_prediction": "backbone_regression_head_only",
                    "intensity_used_in_deployment_prediction": False,
                    "ordinal_used_in_deployment_prediction": False,
                    "outer_labels_used_for_training": False,
                    "winner_router_present": False,
                    "label_defined_sample_experts_present": False,
                    "scalar_residual_correction_present": False,
                },
            },
            fold_output / "v932_outer_payload.pth",
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("V9.32 aggregated outer predictions duplicate IDs")
    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["outer_fold", "mae", "strategy"]
    )
    checkpoints = pd.DataFrame(checkpoint_rows).sort_values(
        ["outer_fold", "variant"]
    )
    sources = pd.DataFrame(source_rows).sort_values(
        ["outer_fold", "role"]
    )

    labels = predictions["label"].to_numpy(dtype=np.float64)
    control = predictions[
        "prediction_regression_only"
    ].to_numpy(dtype=np.float64)
    v921 = predictions[
        "prediction_old_v921_convex_shrinkage"
    ].to_numpy(dtype=np.float64)
    strategy_names = [
        column.removeprefix("prediction_")
        for column in predictions.columns
        if column.startswith("prediction_")
    ]

    aggregate_rows = []
    bootstrap_rows = []
    for offset, strategy in enumerate(sorted(strategy_names)):
        prediction = predictions[
            f"prediction_{strategy}"
        ].to_numpy(dtype=np.float64)
        metrics_control = paired_prediction_metrics(
            prediction, labels, control
        )
        metrics_v921 = paired_prediction_metrics(
            prediction, labels, v921
        )
        aggregate_rows.append(
            {
                "strategy": strategy,
                "primary": strategy == "ordinal_intensity",
                "mae": float(np.abs(prediction - labels).mean()),
                "gain_vs_regression_only": metrics_control[
                    "gain_vs_baseline"
                ],
                "gain_vs_v921": metrics_v921["gain_vs_baseline"],
                "win_rate_vs_regression_only": metrics_control[
                    "win_rate_vs_baseline"
                ],
                "large_harm_rate_010_vs_regression_only": metrics_control[
                    "large_harm_rate_010"
                ],
            }
        )
        for baseline_name, baseline in (
            ("regression_only", control),
            ("old_v921_convex_shrinkage", v921),
        ):
            gain = np.abs(baseline - labels) - np.abs(
                prediction - labels
            )
            interval = group_bootstrap_gain_interval(
                gain,
                predictions["group_id"].tolist(),
                repetitions=int(cli.bootstrap_repetitions),
                seed=int(cli.seed)
                + 7919 * (offset + 1)
                + (0 if baseline_name == "regression_only" else 104729),
            )
            bootstrap_rows.append(
                {
                    "strategy": strategy,
                    "baseline": baseline_name,
                    **interval,
                }
            )

    aggregate = pd.DataFrame(aggregate_rows).sort_values("mae")
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values(
        ["baseline", "strategy"]
    )
    primary = "ordinal_intensity"
    primary_mae = float(
        aggregate.loc[
            aggregate["strategy"] == primary, "mae"
        ].iloc[0]
    )
    control_mae = float(
        aggregate.loc[
            aggregate["strategy"] == "regression_only", "mae"
        ].iloc[0]
    )
    v921_mae = float(
        aggregate.loc[
            aggregate["strategy"] == "old_v921_convex_shrinkage",
            "mae",
        ].iloc[0]
    )
    pivot = fold_metrics.pivot(
        index="outer_fold", columns="strategy", values="mae"
    )
    gains_control = (
        pivot["regression_only"] - pivot[primary]
    ).to_numpy()
    gains_v921 = (
        pivot["old_v921_convex_shrinkage"] - pivot[primary]
    ).to_numpy()
    internal_gate = success_gate(
        gains_control, control_mae - primary_mae, config
    )
    relevance_gate = success_gate(
        gains_v921, v921_mae - primary_mae, config
    )

    summary = {
        "version": VERSION,
        "method": "task_decomposed_auxiliary_supervision",
        "dataset": cli.dataset,
        "root": str(root),
        "output_dir": str(output),
        "config": asdict(config),
        "sample_count": int(len(predictions)),
        "outer_fold_count": int(cli.outer_folds),
        "bootstrap_repetitions": int(cli.bootstrap_repetitions),
        "variants": list(VARIANT_NAMES),
        "primary_variant": primary,
        "primary_mae": primary_mae,
        "regression_only_mae": control_mae,
        "old_v921_mae": v921_mae,
        "gain_vs_regression_only": control_mae - primary_mae,
        "gain_vs_old_v921": v921_mae - primary_mae,
        "internal_auxiliary_gate": internal_gate,
        "project_relevance_gate": relevance_gate,
        "provenance": {
            "training_labels_define_auxiliary_targets_only": True,
            "test_samples_follow_identical_computation_path": True,
            "deployment_output": "backbone_regression_head_only",
            "intensity_or_ordinal_hard_composition": False,
            "outer_labels_used_for_training": False,
            "winner_router_present": False,
            "label_defined_sample_experts_present": False,
            "scalar_residual_correction_present": False,
            "outer_results_used_to_choose_primary_variant": False,
        },
    }

    predictions.to_csv(output / "v932_outer_predictions.csv", index=False)
    fold_metrics.to_csv(output / "v932_metrics_by_fold.csv", index=False)
    aggregate.to_csv(output / "v932_aggregate_metrics.csv", index=False)
    bootstrap.to_csv(
        output / "v932_group_bootstrap_gain_ci.csv", index=False
    )
    checkpoints.to_csv(
        output / "v932_checkpoint_manifest.csv", index=False
    )
    sources.to_csv(output / "v932_source_manifest.csv", index=False)
    (output / "v932_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = [
        "# V9.32 Task-Decomposed Multitask DLF",
        "",
        "Every train and test sample follows the same DLF path. Intensity and ordinal outputs are auxiliary supervision only; the deployed prediction is the original regression head.",
        "",
        "## Aggregate metrics",
        "",
        markdown_table(
            aggregate,
            [
                "strategy",
                "primary",
                "mae",
                "gain_vs_regression_only",
                "gain_vs_v921",
                "win_rate_vs_regression_only",
                "large_harm_rate_010_vs_regression_only",
            ],
        ),
        "",
        "## Fold metrics",
        "",
        markdown_table(
            fold_metrics[
                fold_metrics["strategy"].isin(
                    [
                        "regression_only",
                        "ordinal_only",
                        "intensity_only",
                        "ordinal_intensity",
                        "old_v921_convex_shrinkage",
                    ]
                )
            ],
            [
                "outer_fold",
                "strategy",
                "mae",
                "gain_vs_regression_only",
                "gain_vs_v921",
            ],
        ),
        "",
        "## Pre-registered decision",
        "",
        f"- Primary variant: `{primary}`",
        f"- Regression-only MAE: `{control_mae:.6f}`",
        f"- Primary MAE: `{primary_mae:.6f}`",
        f"- Gain versus regression-only: `{control_mae - primary_mae:+.6f}`",
        f"- Old V9.21 MAE: `{v921_mae:.6f}`",
        f"- Gain versus old V9.21: `{v921_mae - primary_mae:+.6f}`",
        f"- Auxiliary-effect gate passed: `{internal_gate['passed']}`",
        f"- Project-relevance gate passed: `{relevance_gate['passed']}`",
        "",
        "Do not select the best outer-test ablation after seeing these results. The ordinal+intensity variant was fixed as primary before evaluation.",
    ]
    (output / "v932_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.32 TASK-DECOMPOSED MULTITASK COMPLETE")
    print("deployed output: backbone regression head only")
    print("winner router: False")
    print("label-defined sample experts: False")
    print(
        "regression-only / ordinal+intensity / old V9.21 MAE:",
        f"{control_mae:.6f}",
        f"{primary_mae:.6f}",
        f"{v921_mae:.6f}",
    )
    print(
        "gain vs regression-only / old V9.21:",
        f"{control_mae - primary_mae:+.6f}",
        f"{v921_mae - primary_mae:+.6f}",
    )
    print("auxiliary-effect gate:", internal_gate["passed"])
    print("project-relevance gate:", relevance_gate["passed"])
    print("report:", output / "v932_report.md")


if __name__ == "__main__":
    main()
