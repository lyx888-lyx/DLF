"""Run the V9.23 no-training gradient-space conflict audit."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _load_state,
    mode_to_mask,
)
from trains.singleTask.gradient_conflict_audit_v923 import (
    AUDIT_VERSION,
    REGION_TASK_NAMES,
    GradientAuditConfigV923,
    build_candidate_directions,
    collect_mean_mae_gradients,
    direction_effect_rows,
    fold_geometry_summary,
    global_region_decomposition_error,
    pairwise_geometry_rows,
    select_parameter_records,
    selected_parameter_fingerprint,
    task_stat_rows,
)
from trains.singleTask.oof_group_splits_v92 import build_subset_loader
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.same_stack_expert_factory_v919 import sha256
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compute global and five-region MAE gradients at frozen V9.19 "
            "CFCompat anchors; no optimizer or parameter update is used."
        )
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument(
        "--parameter-scope",
        choices=("fusion_tail", "output_head"),
        default="fusion_tail",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--common-cosine-margin",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--minimum-mgda-norm",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--required-feasible-outer-folds",
        type=int,
        default=4,
    )
    return parser.parse_args()


def _markdown_table(
    frame: pd.DataFrame,
    columns: list[str],
) -> str:
    view = frame.loc[:, columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: (
                    "" if pd.isna(value) else f"{float(value):.6f}"
                )
            )
    return "\n".join(
        [
            "|" + "|".join(columns) + "|",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in view.itertuples(index=False, name=None)
            ],
        ]
    )


def _build_args(cli):
    args = get_config_regression(
        "DLF",
        cli.dataset,
        Path(cli.config),
    )
    args["device"] = assign_gpu([cli.gpu])
    args["mode"] = "train"
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = int(cli.seed)
    args["cur_seed"] = int(cli.seed)
    args["batch_size"] = int(cli.batch_size)
    return args


def _load_development_indices(
    fold_dir: Path,
) -> tuple[list[int], dict[str, int]]:
    manifest_path = fold_dir / "outer_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = pd.read_csv(
        manifest_path,
        usecols=[
            "sample_index",
            "sample_id",
            "group_id",
            "partition",
        ],
    )
    expected = {"inner_train", "inner_valid", "outer_holdout"}
    observed = set(manifest["partition"].astype(str))
    if observed != expected:
        raise RuntimeError(
            f"outer manifest partitions differ: {sorted(observed)}"
        )
    development = manifest[
        manifest["partition"].isin(["inner_train", "inner_valid"])
    ].copy()
    holdout = manifest[
        manifest["partition"] == "outer_holdout"
    ].copy()
    development_indices = sorted(
        development["sample_index"].astype(int).tolist()
    )
    if set(development_indices) & set(
        holdout["sample_index"].astype(int).tolist()
    ):
        raise RuntimeError(
            "outer holdout entered development gradient audit"
        )
    if len(development_indices) != len(set(development_indices)):
        raise RuntimeError("duplicate development sample indices")
    counts = {
        "development_count": len(development_indices),
        "development_group_count": int(
            development["group_id"].astype(str).nunique()
        ),
        "outer_holdout_count": len(holdout),
        "outer_holdout_group_count": int(
            holdout["group_id"].astype(str).nunique()
        ),
    }
    return development_indices, counts


def _forward_batch(device):
    def forward(model, batch):
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["labels"]["M"].to(device).view(-1)
        mask = mode_to_mask(
            "LAV",
            labels.size(0),
            device=device,
            dtype=audio.dtype,
        )
        output = model(text, audio, vision, mask)
        return output["output_logit"].view(-1), labels

    return forward


def _pairwise_consistency(
    pairwise: pd.DataFrame,
) -> pd.DataFrame:
    regions = list(REGION_TASK_NAMES)
    unique = pairwise[
        pairwise["left_task"].isin(regions)
        & pairwise["right_task"].isin(regions)
    ].copy()
    unique = unique[
        unique.apply(
            lambda row: (
                regions.index(str(row["left_task"]))
                < regions.index(str(row["right_task"]))
            ),
            axis=1,
        )
    ]
    rows = []
    for (left, right), frame in unique.groupby(
        ["left_task", "right_task"],
        sort=True,
    ):
        rows.append(
            {
                "left_task": left,
                "right_task": right,
                "mean_cosine": float(frame["cosine"].mean()),
                "std_cosine": float(
                    frame["cosine"].std(ddof=0)
                ),
                "min_cosine": float(frame["cosine"].min()),
                "max_cosine": float(frame["cosine"].max()),
                "negative_outer_folds": int(
                    (frame["cosine"] < 0.0).sum()
                ),
                "positive_outer_folds": int(
                    (frame["cosine"] > 0.0).sum()
                ),
                "stable_conflict_4of5": bool(
                    (frame["cosine"] < 0.0).sum() >= 4
                ),
                "stable_alignment_4of5": bool(
                    (frame["cosine"] > 0.0).sum() >= 4
                ),
            }
        )
    return pd.DataFrame(rows)


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
    )
    setup_seed(cli.seed)
    args = _build_args(cli)
    dataset = MMDataset(args, mode="train")
    if "seq_lens" in args:
        args["seq_lens"] = dataset.get_seq_len()

    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v923_gradient_space_conflict_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    config = GradientAuditConfigV923(
        parameter_scope=str(cli.parameter_scope),
        common_cosine_margin=float(cli.common_cosine_margin),
        minimum_mgda_norm=float(cli.minimum_mgda_norm),
        required_feasible_outer_folds=int(
            cli.required_feasible_outer_folds
        ),
    )

    task_rows = []
    pairwise_rows = []
    layerwise_rows = []
    direction_task_rows = []
    direction_summary_rows = []
    mgda_weight_rows = []
    fold_summary_rows = []
    inventory_rows = []
    checkpoint_rows = []

    for fold in range(int(cli.outer_folds)):
        fold_dir = root / f"outer_fold_{fold}"
        stack_dir = fold_dir / "outer_deployment_stack"
        checkpoint_path = (
            stack_dir / "cfcompat_student_best_inner_valid.pth"
        )
        if not checkpoint_path.is_file():
            raise FileNotFoundError(checkpoint_path)
        development_indices, split_counts = (
            _load_development_indices(fold_dir)
        )
        loader = build_subset_loader(
            dataset,
            development_indices,
            int(cli.batch_size),
            int(cli.num_workers),
            False,
            int(cli.seed) + 170003 * (fold + 1),
        )

        model = MissingModalityWrapper(
            DLF(args).to(args.device),
            int(args.feature_dims[1]),
            int(args.feature_dims[2]),
        ).to(args.device)
        model.load_state_dict(
            _load_state(checkpoint_path, args.device),
            strict=True,
        )
        records = select_parameter_records(
            model,
            config.parameter_scope,
        )
        before_checkpoint_hash = sha256(checkpoint_path)
        before_parameter_hash = selected_parameter_fingerprint(records)
        collected = collect_mean_mae_gradients(
            model,
            loader,
            records,
            _forward_batch(args.device),
        )
        after_parameter_hash = selected_parameter_fingerprint(records)
        after_checkpoint_hash = sha256(checkpoint_path)
        if before_parameter_hash != after_parameter_hash:
            raise RuntimeError(
                "gradient audit changed selected parameters"
            )
        if before_checkpoint_hash != after_checkpoint_hash:
            raise RuntimeError(
                "gradient audit changed anchor checkpoint"
            )

        gradients = collected["gradients"]
        fold_pairwise, fold_layerwise = pairwise_geometry_rows(
            gradients,
            records,
        )
        built = build_candidate_directions(gradients, config)
        (
            fold_direction_tasks,
            fold_direction_summary,
        ) = direction_effect_rows(
            gradients,
            built["directions"],
            tolerance=float(config.conflict_tolerance),
        )
        decomposition_error = global_region_decomposition_error(
            gradients,
            collected["sample_counts"],
        )
        if decomposition_error > 5e-5:
            raise RuntimeError(
                "global gradient does not decompose into region gradients: "
                f"{decomposition_error}"
            )
        fold_summary = fold_geometry_summary(
            fold_pairwise,
            fold_direction_summary,
            built["mgda"],
            decomposition_error,
            config,
        )

        for row in task_stat_rows(
            gradients,
            collected["sample_counts"],
            collected["mean_losses"],
        ):
            task_rows.append({"outer_fold": fold, **row})
        pairwise_rows.extend(
            {"outer_fold": fold, **row}
            for row in fold_pairwise
        )
        layerwise_rows.extend(
            {"outer_fold": fold, **row}
            for row in fold_layerwise
        )
        direction_task_rows.extend(
            {"outer_fold": fold, **row}
            for row in fold_direction_tasks
        )
        direction_summary_rows.extend(
            {"outer_fold": fold, **row}
            for row in fold_direction_summary
        )
        for task, weight in zip(
            built["mgda"]["task_names"],
            built["mgda"]["weights"].tolist(),
        ):
            mgda_weight_rows.append(
                {
                    "outer_fold": fold,
                    "task": task,
                    "weight": float(weight),
                }
            )
        fold_summary_rows.append(
            {
                "outer_fold": fold,
                **split_counts,
                "batch_count": int(collected["batch_count"]),
                "selected_parameter_count": len(records),
                "selected_parameter_elements": int(
                    sum(
                        record.parameter.numel()
                        for record in records
                    )
                ),
                **fold_summary,
            }
        )
        for group in sorted(
            {record.group for record in records}
        ):
            group_records = [
                record
                for record in records
                if record.group == group
            ]
            inventory_rows.append(
                {
                    "outer_fold": fold,
                    "parameter_group": group,
                    "tensor_count": len(group_records),
                    "parameter_elements": int(
                        sum(
                            record.parameter.numel()
                            for record in group_records
                        )
                    ),
                }
            )
        checkpoint_rows.append(
            {
                "outer_fold": fold,
                "checkpoint": str(checkpoint_path),
                "checkpoint_sha256_before": before_checkpoint_hash,
                "checkpoint_sha256_after": after_checkpoint_hash,
                "selected_parameter_sha256_before": (
                    before_parameter_hash
                ),
                "selected_parameter_sha256_after": (
                    after_parameter_hash
                ),
                "unchanged": True,
            }
        )
        del model, gradients, collected
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    task_frame = pd.DataFrame(task_rows)
    pairwise_frame = pd.DataFrame(pairwise_rows)
    layerwise_frame = pd.DataFrame(layerwise_rows)
    direction_task_frame = pd.DataFrame(direction_task_rows)
    direction_summary_frame = pd.DataFrame(
        direction_summary_rows
    )
    mgda_weights_frame = pd.DataFrame(mgda_weight_rows)
    fold_summary_frame = pd.DataFrame(
        fold_summary_rows
    ).sort_values("outer_fold")
    inventory_frame = pd.DataFrame(inventory_rows)
    checkpoint_frame = pd.DataFrame(checkpoint_rows)
    consistency_frame = _pairwise_consistency(pairwise_frame)

    feasible_folds = int(
        fold_summary_frame["fold_geometry_feasible"].sum()
    )
    training_recommended = bool(
        feasible_folds
        >= int(config.required_feasible_outer_folds)
    )
    aggregate = {
        "version": AUDIT_VERSION,
        "method": "frozen_anchor_region_gradient_geometry",
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "outer_fold_count": int(cli.outer_folds),
        "parameter_scope": config.parameter_scope,
        "config": asdict(config),
        "feasible_outer_folds": feasible_folds,
        "required_feasible_outer_folds": int(
            config.required_feasible_outer_folds
        ),
        "gradient_space_training_recommended": (
            training_recommended
        ),
        "mean_region_pair_conflict_fraction": float(
            fold_summary_frame[
                "region_pair_conflict_fraction"
            ].mean()
        ),
        "mean_global_conflicting_region_count": float(
            fold_summary_frame[
                "global_conflicting_region_count"
            ].mean()
        ),
        "median_mgda_minimum_region_cosine": float(
            fold_summary_frame[
                "mgda_minimum_region_cosine"
            ].median()
        ),
        "median_mgda_direction_norm": float(
            fold_summary_frame["mgda_direction_norm"].median()
        ),
        "maximum_gradient_decomposition_error": float(
            fold_summary_frame[
                "global_region_decomposition_relative_error"
            ].max()
        ),
        "interpretation": (
            "A positive verdict only establishes first-order geometric "
            "plausibility for a future single-model experiment. No model "
            "was trained and no outer holdout was evaluated."
            if training_recommended
            else "Do not train a gradient-consolidated unified model from "
            "this evidence; common descent was not stable across outer "
            "folds."
        ),
        "provenance": {
            "models_trained": False,
            "optimizer_created": False,
            "backward_called": False,
            "autograd_grad_only": True,
            "parameters_updated": False,
            "frozen_v919_cfcompat_anchors_loaded": True,
            "development_partitions_only": True,
            "outer_holdout_samples_entered_gradient_computation": False,
            "outer_holdout_predictions_or_labels_loaded": False,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "router_or_fusion_head_used": False,
            "hyperparameters_selected_from_audit_results": False,
        },
    }

    task_frame.to_csv(
        output / "v923_task_gradient_stats.csv",
        index=False,
    )
    pairwise_frame.to_csv(
        output / "v923_pairwise_gradient_geometry.csv",
        index=False,
    )
    layerwise_frame.to_csv(
        output / "v923_layerwise_gradient_geometry.csv",
        index=False,
    )
    consistency_frame.to_csv(
        output / "v923_pairwise_consistency.csv",
        index=False,
    )
    direction_task_frame.to_csv(
        output / "v923_direction_task_effects.csv",
        index=False,
    )
    direction_summary_frame.to_csv(
        output / "v923_direction_summary_by_fold.csv",
        index=False,
    )
    mgda_weights_frame.to_csv(
        output / "v923_mgda_weights.csv",
        index=False,
    )
    fold_summary_frame.to_csv(
        output / "v923_fold_summary.csv",
        index=False,
    )
    inventory_frame.to_csv(
        output / "v923_parameter_inventory.csv",
        index=False,
    )
    checkpoint_frame.to_csv(
        output / "v923_checkpoint_integrity.csv",
        index=False,
    )
    (
        output / "v923_gradient_conflict_summary.json"
    ).write_text(
        json.dumps(
            jsonable(aggregate),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    report = [
        "# V9.23 No-Training Gradient-Space Conflict Audit",
        "",
        "No optimizer is created and no parameter is updated. Gradients "
        "are computed only on each V9.19 outer fold's inner-train and "
        "inner-valid development partitions.",
        "",
        "## Fold geometry",
        "",
        _markdown_table(
            fold_summary_frame,
            [
                "outer_fold",
                "region_pair_conflict_fraction",
                "global_conflicting_region_count",
                "equal_mean_all_regions_improve",
                "pcgrad_all_regions_improve",
                "mgda_all_regions_improve",
                "mgda_global_improves",
                "mgda_minimum_region_cosine",
                "mgda_direction_norm",
                "fold_geometry_feasible",
            ],
        ),
        "",
        "## Region-pair consistency",
        "",
        _markdown_table(
            consistency_frame,
            [
                "left_task",
                "right_task",
                "mean_cosine",
                "std_cosine",
                "negative_outer_folds",
                "positive_outer_folds",
                "stable_conflict_4of5",
                "stable_alignment_4of5",
            ],
        ),
        "",
        "## Verdict",
        "",
        f"- Feasible folds: `{feasible_folds}/{int(cli.outer_folds)}`",
        f"- Required feasible folds: "
        f"`{config.required_feasible_outer_folds}`",
        f"- Gradient-space training recommended: "
        f"`{training_recommended}`",
        f"- Median MGDA minimum region cosine: "
        f"`{aggregate['median_mgda_minimum_region_cosine']:.6f}`",
        f"- Median MGDA direction norm: "
        f"`{aggregate['median_mgda_direction_norm']:.6f}`",
        "",
        aggregate["interpretation"],
    ]
    (
        output / "v923_gradient_conflict_report.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        "V9.23 NO-TRAINING GRADIENT CONFLICT AUDIT COMPLETE"
    )
    print("models trained: False")
    print("optimizer created: False")
    print("parameters updated: False")
    print("parameter scope:", config.parameter_scope)
    print(
        "feasible folds:",
        f"{feasible_folds}/{int(cli.outer_folds)}",
    )
    print(
        "gradient-space training recommended:",
        training_recommended,
    )
    print(
        "report:",
        output / "v923_gradient_conflict_report.md",
    )


if __name__ == "__main__":
    main()
