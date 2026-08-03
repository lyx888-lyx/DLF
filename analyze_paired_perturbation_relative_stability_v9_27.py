"""Run the strict V9.27 paired-perturbation relative-stability audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.expert_self_risk_v925 import (
    BINARY_TARGET_NAMES,
    CONTINUOUS_TARGET_NAMES,
    EXPERT_NAMES,
    ExpertSelfRiskConfigV925,
    add_rank_selection_flags,
    binary_metric_row,
    build_expert_features,
    build_expert_targets,
    continuous_metric_row,
    crossfit_risk_predictions,
    fit_outer_risk_bundle,
    group_bootstrap_selected_gain,
    predict_risk_bundle,
    selection_metric_row,
)
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.paired_perturbation_relative_stability_v927 import (
    AUDIT_VERSION,
    BASE_METHOD,
    PRIMARY_METHOD,
    STABILITY_ONLY_METHOD,
    PairedPerturbationConfigV927,
    build_relative_stability_features,
    collect_stack_perturbations,
    perturbation_specs,
    weighted_baseline,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.same_stack_expert_factory_v919 import sha256
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    fit_consensus_weights,
    inner_crossfit_consensus,
    strategy_predictions,
)
from trains.singleTask.support_stability_self_knowledge_v926 import (
    average_precision,
)
from utils import assign_gpu, setup_seed


METHODS = (BASE_METHOD, STABILITY_ONLY_METHOD, PRIMARY_METHOD)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--noise-scale", type=float, default=0.01)
    parser.add_argument("--temporal-mask-fraction", type=float, default=0.05)
    parser.add_argument("--shrinkage-lambda", type=float, default=0.01)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1111)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def markdown_table(frame, columns):
    if frame.empty:
        return "_No rows._"
    local = frame.loc[:, [column for column in columns if column in frame]].copy()
    for column in local.columns:
        if pd.api.types.is_float_dtype(local[column]):
            local[column] = local[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
    headers = list(local.columns)
    return "\n".join(
        [
            "|" + "|".join(headers) + "|",
            "|" + "|".join(["---"] * len(headers)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in local.itertuples(index=False, name=None)
            ],
        ]
    )


def _subset_pool(view, indices):
    indices = torch.as_tensor(indices, dtype=torch.long)
    actions = view.actions.index_select(0, indices)
    return normalize_pool(
        {
            "labels": view.labels.index_select(0, indices).view(-1, 1),
            "anchor": actions[:, 0:1],
            "expert_predictions": actions[:, 1:].unsqueeze(-1),
            "sample_ids": [view.sample_ids[i] for i in indices.tolist()],
            "group_ids": [view.group_ids[i] for i in indices.tolist()],
        }
    )


def _max_abs(left, right):
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} != {right.shape}")
    return float(np.max(np.abs(left - right)))


def _assemble_inner_perturbations(
    args,
    dataset,
    fold_dir,
    inner_payload,
    consensus_config,
    perturb_config,
    resume,
    integrity_rows,
):
    fold_index = np.asarray(
        torch.as_tensor(inner_payload["fold_index"]).view(-1).cpu().numpy(),
        dtype=np.int64,
    )
    inner_view = normalize_pool(inner_payload)
    unique_folds = sorted(int(value) for value in np.unique(fold_index))
    id_to_position = {
        str(value): index
        for index, value in enumerate(inner_payload["sample_ids"])
    }
    variant_count = len(perturbation_specs(perturb_config))
    n = len(inner_payload["sample_ids"])
    actions = np.full((variant_count, n, 5), np.nan, dtype=np.float64)
    confidences = np.full((variant_count, n, 4), np.nan, dtype=np.float64)
    corrections = np.full_like(confidences, np.nan)
    baseline = np.full((variant_count, n), np.nan, dtype=np.float64)
    cache_paths = []

    for inner_fold in unique_folds:
        print(
            f"  collecting inner stack {inner_fold} for {fold_dir.name}",
            flush=True,
        )
        stack_dir = fold_dir / f"inner_stack_{inner_fold}"
        pool_path = stack_dir / "same_stack_target_pool_v919.pth"
        if not pool_path.is_file():
            raise FileNotFoundError(pool_path)
        pool_payload = torch.load(pool_path, map_location="cpu")
        cache_path = stack_dir / "paired_perturbation_pool_v927.pth"
        cache = collect_stack_perturbations(
            args,
            dataset,
            stack_dir,
            pool_payload,
            perturb_config,
            cache_path=cache_path,
            resume=resume,
        )
        cache_paths.append(cache_path)
        development = np.flatnonzero(fold_index != inner_fold)
        weights = fit_consensus_weights(
            _subset_pool(inner_view, development), consensus_config
        )["convex_shrinkage"]
        local_baseline = weighted_baseline(cache["actions"], weights)
        local_actions = torch.as_tensor(cache["actions"]).cpu().numpy()
        local_confidences = (
            torch.as_tensor(cache["expert_confidences"]).cpu().numpy()
        )
        local_corrections = (
            torch.as_tensor(cache["expert_corrections"]).cpu().numpy()
        )
        positions = []
        for sample_id in cache["sample_ids"]:
            key = str(sample_id)
            if key not in id_to_position:
                raise RuntimeError(f"inner perturbation id missing: {key}")
            positions.append(id_to_position[key])
        positions = np.asarray(positions, dtype=np.int64)
        if np.any(fold_index[positions] != inner_fold):
            raise RuntimeError("inner stack rows do not match fold_index")
        actions[:, positions] = local_actions
        confidences[:, positions] = local_confidences
        corrections[:, positions] = local_corrections
        baseline[:, positions] = local_baseline
        integrity_rows.append(
            {
                "outer_fold": int(fold_dir.name.rsplit("_", 1)[1]),
                "stack": f"inner_stack_{inner_fold}",
                "pool_path": str(pool_path),
                "pool_sha256": sha256(pool_path),
                "cache_path": str(cache_path),
                "cache_sha256": sha256(cache_path),
                "sample_count": len(positions),
                "identity_action_max_abs": cache[
                    "identity_reconstruction_max_abs"
                ]["actions"],
                "identity_confidence_max_abs": cache[
                    "identity_reconstruction_max_abs"
                ]["confidences"],
                "identity_correction_max_abs": cache[
                    "identity_reconstruction_max_abs"
                ]["corrections"],
            }
        )

    for name, value in (
        ("inner actions", actions),
        ("inner confidences", confidences),
        ("inner corrections", corrections),
        ("inner baseline", baseline),
    ):
        if not np.isfinite(value).all():
            raise FloatingPointError(f"incomplete {name}")

    reference = (
        inner_crossfit_consensus(inner_payload, consensus_config)["predictions"][
            "convex_shrinkage"
        ]
        .detach()
        .cpu()
        .numpy()
    )
    baseline_diff = _max_abs(baseline[0], reference)
    if baseline_diff > perturb_config.identity_tolerance:
        raise RuntimeError(
            f"inner V9.21 identity reconstruction mismatch: {baseline_diff}"
        )
    return {
        "actions": torch.tensor(actions, dtype=torch.float32),
        "expert_confidences": torch.tensor(
            confidences, dtype=torch.float32
        ),
        "expert_corrections": torch.tensor(
            corrections, dtype=torch.float32
        ),
        "baseline": baseline,
        "fold_index": fold_index,
        "cache_paths": cache_paths,
        "identity_baseline_max_abs": baseline_diff,
    }


def _collect_outer_perturbations(
    args,
    dataset,
    fold_dir,
    inner_payload,
    consensus_config,
    perturb_config,
    resume,
    integrity_rows,
):
    print(
        f"  collecting outer deployment stack for {fold_dir.name}",
        flush=True,
    )
    stack_dir = fold_dir / "outer_deployment_stack"
    pool_path = stack_dir / "same_stack_target_pool_v919.pth"
    if not pool_path.is_file():
        raise FileNotFoundError(pool_path)
    outer_payload = torch.load(pool_path, map_location="cpu")
    cache_path = stack_dir / "paired_perturbation_pool_v927.pth"
    cache = collect_stack_perturbations(
        args,
        dataset,
        stack_dir,
        outer_payload,
        perturb_config,
        cache_path=cache_path,
        resume=resume,
    )
    inner_view = normalize_pool(inner_payload)
    outer_view = normalize_pool(outer_payload)
    weights = fit_consensus_weights(
        inner_view, consensus_config
    )["convex_shrinkage"]
    baseline = weighted_baseline(cache["actions"], weights)
    reference = strategy_predictions(
        inner_view, outer_view, consensus_config
    )["predictions"]["convex_shrinkage"].detach().cpu().numpy()
    baseline_diff = _max_abs(baseline[0], reference)
    if baseline_diff > perturb_config.identity_tolerance:
        raise RuntimeError(
            f"outer V9.21 identity reconstruction mismatch: {baseline_diff}"
        )
    outer_fold = int(fold_dir.name.rsplit("_", 1)[1])
    integrity_rows.append(
        {
            "outer_fold": outer_fold,
            "stack": "outer_deployment_stack",
            "pool_path": str(pool_path),
            "pool_sha256": sha256(pool_path),
            "cache_path": str(cache_path),
            "cache_sha256": sha256(cache_path),
            "sample_count": len(outer_payload["sample_ids"]),
            "identity_action_max_abs": cache[
                "identity_reconstruction_max_abs"
            ]["actions"],
            "identity_confidence_max_abs": cache[
                "identity_reconstruction_max_abs"
            ]["confidences"],
            "identity_correction_max_abs": cache[
                "identity_reconstruction_max_abs"
            ]["corrections"],
        }
    )
    return outer_payload, cache, baseline, baseline_diff


def _prediction_frame(
    payload,
    outer_fold,
    split,
    method,
    expert,
    base_features,
    stability_features,
    targets,
    predictions,
    risk_folds=None,
):
    n = len(payload["sample_ids"])
    result = {
        "outer_fold": np.full(n, int(outer_fold)),
        "split": np.full(n, split),
        "method": np.full(n, method),
        "expert": np.full(n, expert),
        "sample_id": [str(value) for value in payload["sample_ids"]],
        "group_id": [str(value) for value in payload["group_ids"]],
        "label": torch.as_tensor(payload["labels"]).view(-1).cpu().numpy(),
        "anchor_prediction": base_features["anchor_prediction"],
        "baseline_prediction": base_features["baseline_prediction"],
        "expert_prediction": base_features["expert_prediction"],
        "native_confidence": base_features["native_confidence"],
        **{
            name: stability_features[name]
            for name in (
                "expert_std",
                "baseline_std",
                "log_std_ratio",
                "relative_gap_std",
                "relative_direction_flip_rate",
                "applicability_flip_rate",
                "expert_max_change",
                "baseline_max_change",
            )
        },
        **targets,
        **predictions,
    }
    if risk_folds is not None:
        result["risk_fold_index"] = np.asarray(
            risk_folds, dtype=np.int64
        )
    return pd.DataFrame(result)


def _metric_frames(frame, risk_config):
    binary_rows = []
    continuous_rows = []
    for method in METHODS:
        local_method = frame[frame["method"] == method]
        groups = [
            ("all", local_method),
            *list(local_method.groupby("expert", sort=True)),
        ]
        for expert, local in groups:
            for target in BINARY_TARGET_NAMES:
                metrics = binary_metric_row(
                    local[target],
                    local[f"pred_{target}_prob"],
                    risk_config.calibration_bins,
                )
                binary_rows.append(
                    {
                        "split": "outer_holdout",
                        "method": method,
                        "expert": expert,
                        "target": target,
                        "average_precision": average_precision(
                            local[target],
                            local[f"pred_{target}_prob"],
                        ),
                        **metrics,
                    }
                )
            for target in CONTINUOUS_TARGET_NAMES:
                continuous_rows.append(
                    {
                        "split": "outer_holdout",
                        "method": method,
                        "expert": expert,
                        "target": target,
                        **continuous_metric_row(
                            local[target], local[f"pred_{target}"]
                        ),
                    }
                )
    return pd.DataFrame(binary_rows), pd.DataFrame(continuous_rows)


def _ranked_metrics(frame, risk_config):
    rows = []
    flagged = []
    for method in METHODS:
        local = add_rank_selection_flags(
            frame[frame["method"] == method].copy(),
            risk_config.safe_rank_fraction,
            group_columns=("outer_fold", "expert"),
        )
        flagged.append(local)
        for outer_fold, fold_frame in [
            ("aggregate", local),
            *list(local.groupby("outer_fold", sort=True)),
        ]:
            for selector in (
                "selected_native_top",
                "selected_risk_top",
            ):
                rows.append(
                    {
                        "split": "outer_holdout",
                        "method": method,
                        "outer_fold": outer_fold,
                        "expert": "all",
                        "selector": selector,
                        **selection_metric_row(
                            fold_frame, selector, risk_config
                        ),
                    }
                )
    return pd.concat(flagged, ignore_index=True), pd.DataFrame(rows)


def _find_metric(frame, method, target, column):
    row = frame[
        (frame["method"] == method)
        & (frame["expert"] == "all")
        & (frame["target"] == target)
    ]
    if len(row) != 1:
        raise RuntimeError(
            f"metric row missing: {method}/{target}/{column}"
        )
    return float(row.iloc[0][column])


def _find_rank(frame, method, selector):
    row = frame[
        (frame["method"] == method)
        & (frame["outer_fold"].astype(str) == "aggregate")
        & (frame["expert"] == "all")
        & (frame["selector"] == selector)
    ]
    if len(row) != 1:
        raise RuntimeError(f"rank row missing: {method}/{selector}")
    return row.iloc[0].to_dict()


def main():
    cli = parse_args()
    setup_seed(cli.seed)
    v919_root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else v919_root / "v927_paired_perturbation_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "risk_models"
    model_dir.mkdir(parents=True, exist_ok=True)

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
    dataset = MMDataset(args, mode="train")
    if "seq_lens" in args:
        args["seq_lens"] = dataset.get_seq_len()

    perturb_config = PairedPerturbationConfigV927(
        batch_size=cli.batch_size,
        num_workers=cli.num_workers,
        noise_scale=cli.noise_scale,
        temporal_mask_fraction=cli.temporal_mask_fraction,
        base_seed=cli.seed,
    )
    perturb_config.validate()
    risk_config = ExpertSelfRiskConfigV925()
    risk_config.validate()
    consensus_config = ConsensusConfigV921(
        shrinkage_lambda=cli.shrinkage_lambda,
        bootstrap_repetitions=cli.bootstrap_repetitions,
        bootstrap_seed=cli.bootstrap_seed,
    )

    inner_frames = []
    outer_frames = []
    split_rows = []
    integrity_rows = []
    model_rows = []
    feature_rows = []
    fold_reconstruction_rows = []

    for outer_fold in range(cli.outer_folds):
        print(f"V9.27 outer fold {outer_fold}: collecting perturbations")
        fold_dir = v919_root / f"outer_fold_{outer_fold}"
        inner_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        if not inner_path.is_file():
            raise FileNotFoundError(inner_path)
        inner_payload = torch.load(inner_path, map_location="cpu")
        inner_perturb = _assemble_inner_perturbations(
            args,
            dataset,
            fold_dir,
            inner_payload,
            consensus_config,
            perturb_config,
            resume=not cli.no_resume,
            integrity_rows=integrity_rows,
        )
        (
            outer_payload,
            outer_perturb,
            outer_baseline,
            outer_baseline_diff,
        ) = _collect_outer_perturbations(
            args,
            dataset,
            fold_dir,
            inner_payload,
            consensus_config,
            perturb_config,
            resume=not cli.no_resume,
            integrity_rows=integrity_rows,
        )
        fold_reconstruction_rows.append(
            {
                "outer_fold": outer_fold,
                "inner_v921_identity_max_abs": inner_perturb[
                    "identity_baseline_max_abs"
                ],
                "outer_v921_identity_max_abs": outer_baseline_diff,
            }
        )
        inner_baseline = inner_perturb["baseline"]
        inner_fold_index = inner_perturb["fold_index"]

        for expert in EXPERT_NAMES:
            inner_base = build_expert_features(
                inner_payload, inner_baseline[0], expert
            )
            outer_base = build_expert_features(
                outer_payload, outer_baseline[0], expert
            )
            inner_stability = build_relative_stability_features(
                inner_perturb,
                inner_baseline,
                expert,
                perturb_config,
            )
            outer_stability = build_relative_stability_features(
                outer_perturb,
                outer_baseline,
                expert,
                perturb_config,
            )
            inner_targets = build_expert_targets(
                inner_payload,
                inner_baseline[0],
                expert,
                risk_config,
            )
            outer_targets = build_expert_targets(
                outer_payload,
                outer_baseline[0],
                expert,
                risk_config,
            )
            matrices = {
                BASE_METHOD: (
                    inner_base["matrix"],
                    outer_base["matrix"],
                    inner_base["feature_names"],
                ),
                STABILITY_ONLY_METHOD: (
                    inner_stability["matrix"],
                    outer_stability["matrix"],
                    inner_stability["feature_names"],
                ),
                PRIMARY_METHOD: (
                    np.column_stack(
                        [inner_base["matrix"], inner_stability["matrix"]]
                    ),
                    np.column_stack(
                        [outer_base["matrix"], outer_stability["matrix"]]
                    ),
                    [
                        *inner_base["feature_names"],
                        *inner_stability["feature_names"],
                    ],
                ),
            }

            for method, (
                inner_matrix,
                outer_matrix,
                feature_names,
            ) in matrices.items():
                crossfit = crossfit_risk_predictions(
                    inner_matrix,
                    inner_targets,
                    inner_fold_index,
                    risk_config,
                )
                outer_fit = fit_outer_risk_bundle(
                    inner_matrix,
                    inner_targets,
                    inner_fold_index,
                    risk_config,
                )
                outer_predictions = predict_risk_bundle(
                    outer_fit["bundle"],
                    outer_matrix,
                    risk_config,
                )
                model_path = (
                    model_dir
                    / f"outer_fold_{outer_fold}_{expert}_{method}_v927.pth"
                )
                torch.save(
                    {
                        "version": AUDIT_VERSION,
                        "outer_fold": outer_fold,
                        "expert": expert,
                        "method": method,
                        "feature_names": feature_names,
                        "bundle": outer_fit["bundle"],
                        "provenance": {
                            "base_experts_trained": False,
                            "outer_labels_used_for_fit_or_calibration": False,
                            "inner_oof_only": True,
                            "disjoint_calibration_fold": True,
                            "router_or_action_selection_executed": False,
                        },
                    },
                    model_path,
                )
                model_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "expert": expert,
                        "method": method,
                        "model_path": str(model_path),
                        "model_sha256": sha256(model_path),
                        "calibration_fold": outer_fit[
                            "calibration_fold"
                        ],
                        "fit_sample_count": outer_fit[
                            "fit_sample_count"
                        ],
                        "calibration_sample_count": outer_fit[
                            "calibration_sample_count"
                        ],
                        "feature_count": len(feature_names),
                    }
                )
                split_rows.extend(
                    {
                        "outer_fold": outer_fold,
                        "expert": expert,
                        "method": method,
                        **row,
                    }
                    for row in crossfit["split_rows"]
                )
                feature_rows.extend(
                    {
                        "outer_fold": outer_fold,
                        "expert": expert,
                        "method": method,
                        "feature_index": index,
                        "feature_name": name,
                    }
                    for index, name in enumerate(feature_names)
                )
                inner_frames.append(
                    _prediction_frame(
                        inner_payload,
                        outer_fold,
                        "inner_oof",
                        method,
                        expert,
                        inner_base,
                        inner_stability,
                        inner_targets,
                        crossfit["predictions"],
                        risk_folds=inner_fold_index,
                    )
                )
                outer_frames.append(
                    _prediction_frame(
                        outer_payload,
                        outer_fold,
                        "outer_holdout",
                        method,
                        expert,
                        outer_base,
                        outer_stability,
                        outer_targets,
                        outer_predictions,
                    )
                )

    inner_frame = pd.concat(inner_frames, ignore_index=True)
    outer_frame = pd.concat(outer_frames, ignore_index=True)
    binary_frame, continuous_frame = _metric_frames(
        outer_frame, risk_config
    )
    flagged_outer, ranked_frame = _ranked_metrics(
        outer_frame, risk_config
    )

    primary_top = _find_rank(
        ranked_frame, PRIMARY_METHOD, "selected_risk_top"
    )
    base_top = _find_rank(
        ranked_frame, BASE_METHOD, "selected_risk_top"
    )
    native_top = _find_rank(
        ranked_frame, PRIMARY_METHOD, "selected_native_top"
    )
    primary_selected = flagged_outer[
        flagged_outer["method"] == PRIMARY_METHOD
    ].copy()
    base_selected = flagged_outer[
        flagged_outer["method"] == BASE_METHOD
    ].copy()
    primary_bootstrap = group_bootstrap_selected_gain(
        primary_selected,
        "selected_risk_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed,
    )
    base_bootstrap = group_bootstrap_selected_gain(
        base_selected,
        "selected_risk_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed + 17,
    )
    native_bootstrap = group_bootstrap_selected_gain(
        primary_selected,
        "selected_native_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed + 31,
    )

    fold_rows = []
    for outer_fold, local in primary_selected.groupby(
        "outer_fold", sort=True
    ):
        metrics = selection_metric_row(
            local, "selected_risk_top", risk_config
        )
        fold_rows.append(
            {
                "outer_fold": int(outer_fold),
                "paired_top20_gain": metrics[
                    "mean_gain_vs_baseline"
                ],
                "paired_top20_win_rate": metrics["win_rate"],
                "paired_top20_large_harm_rate": metrics[
                    "large_harm_rate_030"
                ],
                "paired_expert_std_mean": float(
                    local.loc[
                        local["selected_risk_top"], "expert_std"
                    ].mean()
                ),
                "paired_baseline_std_mean": float(
                    local.loc[
                        local["selected_risk_top"], "baseline_std"
                    ].mean()
                ),
                "paired_direction_flip_rate_mean": float(
                    local.loc[
                        local["selected_risk_top"],
                        "relative_direction_flip_rate",
                    ].mean()
                ),
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    positive_folds = int((fold_frame["paired_top20_gain"] > 0).sum())

    primary_win_auc = _find_metric(
        binary_frame, PRIMARY_METHOD, "win_vs_baseline", "auc"
    )
    base_win_auc = _find_metric(
        binary_frame, BASE_METHOD, "win_vs_baseline", "auc"
    )
    stability_win_auc = _find_metric(
        binary_frame,
        STABILITY_ONLY_METHOD,
        "win_vs_baseline",
        "auc",
    )
    primary_harm_auc = _find_metric(
        binary_frame, PRIMARY_METHOD, "large_harm_030", "auc"
    )
    primary_harm_ap = _find_metric(
        binary_frame,
        PRIMARY_METHOD,
        "large_harm_030",
        "average_precision",
    )
    primary_gain_spearman = _find_metric(
        continuous_frame,
        PRIMARY_METHOD,
        "gain_vs_baseline",
        "spearman",
    )
    base_gain_spearman = _find_metric(
        continuous_frame,
        BASE_METHOD,
        "gain_vs_baseline",
        "spearman",
    )
    stability_gain_spearman = _find_metric(
        continuous_frame,
        STABILITY_ONLY_METHOD,
        "gain_vs_baseline",
        "spearman",
    )
    win_auc_gain = primary_win_auc - base_win_auc
    gain_spearman_gain = (
        primary_gain_spearman - base_gain_spearman
    )
    gain_over_native = (
        float(primary_top["mean_gain_vs_baseline"])
        - float(native_top["mean_gain_vs_baseline"])
    )

    passed = bool(
        primary_win_auc >= 0.60
        and primary_gain_spearman >= 0.20
        and primary_harm_auc >= 0.60
        and primary_bootstrap["gain_ci_low"] > 0.0
        and positive_folds >= 4
        and gain_over_native >= 0.002
        and float(primary_top["large_harm_rate_030"])
        <= float(native_top["large_harm_rate_030"])
        and win_auc_gain >= 0.02
        and gain_spearman_gain >= 0.05
    )
    interpretation = (
        "Paired model-input perturbation contains deployable relative expert "
        "advantage signal; only a separately pre-registered gate on untouched "
        "data would be justified."
        if passed
        else (
            "Do not build a dynamic gate from V9.27: paired perturbation "
            "stability did not identify stable expert advantage over V9.21."
        )
    )
    success_gate = {
        "passed": passed,
        "primary_method": PRIMARY_METHOD,
        "primary_win_auc": primary_win_auc,
        "stability_only_win_auc": stability_win_auc,
        "v925_base_win_auc": base_win_auc,
        "win_auc_gain_over_v925": win_auc_gain,
        "required_win_auc": 0.60,
        "required_win_auc_gain_over_v925": 0.02,
        "primary_gain_spearman": primary_gain_spearman,
        "stability_only_gain_spearman": stability_gain_spearman,
        "v925_base_gain_spearman": base_gain_spearman,
        "gain_spearman_gain_over_v925": gain_spearman_gain,
        "required_gain_spearman": 0.20,
        "required_gain_spearman_gain_over_v925": 0.05,
        "primary_large_harm_auc": primary_harm_auc,
        "primary_large_harm_average_precision": primary_harm_ap,
        "primary_top20": primary_top,
        "v925_base_top20": base_top,
        "native_confidence_top20": native_top,
        "primary_top20_bootstrap": primary_bootstrap,
        "v925_base_top20_bootstrap": base_bootstrap,
        "native_top20_bootstrap": native_bootstrap,
        "gain_over_native_top20": gain_over_native,
        "positive_outer_folds": positive_folds,
        "required_positive_outer_folds": 4,
    }

    summary = {
        "version": AUDIT_VERSION,
        "dataset": cli.dataset,
        "seed": cli.seed,
        "risk_signal_supported": passed,
        "interpretation": interpretation,
        "success_gate": success_gate,
        "perturbation_config": asdict(perturb_config),
        "perturbations": [
            asdict(spec) for spec in perturbation_specs(perturb_config)
        ],
        "risk_config": asdict(risk_config),
        "consensus_config": asdict(consensus_config),
        "provenance": {
            "base_experts_trained": False,
            "base_experts_or_predictions_modified": False,
            "v921_weights_refit_on_outer_labels": False,
            "risk_heads_trained": True,
            "inner_oof_only": True,
            "disjoint_calibration_fold": True,
            "outer_labels_used_for_fit_or_calibration": False,
            "pre_extracted_audio_vision_features_perturbed": True,
            "raw_waveform_or_pixels_perturbed": False,
            "text_perturbed": False,
            "same_perturbation_for_experts_and_v921": True,
            "router_or_action_selection_executed": False,
            "prediction_replacement_or_mixing_executed": False,
            "official_validation_or_test_used": False,
        },
    }

    binary_frame.to_csv(output / "v927_binary_metrics.csv", index=False)
    continuous_frame.to_csv(
        output / "v927_continuous_metrics.csv", index=False
    )
    ranked_frame.to_csv(
        output / "v927_ranked_subset_metrics.csv", index=False
    )
    fold_frame.to_csv(
        output / "v927_outer_fold_safe_gain.csv", index=False
    )
    inner_frame.to_csv(
        output / "v927_inner_risk_predictions.csv", index=False
    )
    flagged_outer.to_csv(
        output / "v927_outer_risk_predictions.csv", index=False
    )
    pd.DataFrame(split_rows).to_csv(
        output / "v927_risk_split_manifest.csv", index=False
    )
    pd.DataFrame(integrity_rows).to_csv(
        output / "v927_source_integrity.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(
        output / "v927_model_inventory.csv", index=False
    )
    pd.DataFrame(feature_rows).to_csv(
        output / "v927_feature_schema.csv", index=False
    )
    pd.DataFrame(fold_reconstruction_rows).to_csv(
        output / "v927_identity_reconstruction.csv", index=False
    )
    (output / "v927_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    aggregate_binary = binary_frame[
        binary_frame["expert"] == "all"
    ].copy()
    aggregate_continuous = continuous_frame[
        continuous_frame["expert"] == "all"
    ].copy()
    aggregate_ranked = ranked_frame[
        (ranked_frame["outer_fold"].astype(str) == "aggregate")
        & (ranked_frame["expert"] == "all")
    ].copy()
    report = [
        "# V9.27 Paired Perturbation Relative-Stability Audit",
        "",
        (
            "Frozen V9.19 checkpoints are rerun on one identity input and 12 "
            "fixed mild audio/vision feature-sequence perturbations. Frozen "
            "V9.21 convex-shrinkage weights are applied to each perturbed "
            "action matrix. Text is unchanged. This is not waveform/pixel "
            "augmentation, routing, prediction replacement, or mixture."
        ),
        "",
        "## Aggregate binary comparison",
        "",
        markdown_table(
            aggregate_binary[
                aggregate_binary["target"].isin(
                    (
                        "win_vs_baseline",
                        "large_gain_010",
                        "large_harm_030",
                    )
                )
            ],
            [
                "method",
                "target",
                "positive_rate",
                "auc",
                "average_precision",
                "brier",
                "ece",
            ],
        ),
        "",
        "## Aggregate continuous comparison",
        "",
        markdown_table(
            aggregate_continuous,
            [
                "method",
                "target",
                "target_mean",
                "prediction_mean",
                "mae",
                "pearson",
                "spearman",
            ],
        ),
        "",
        "## Fixed top-20% diagnostic strata",
        "",
        markdown_table(
            aggregate_ranked,
            [
                "method",
                "selector",
                "coverage",
                "mean_gain_vs_baseline",
                "win_rate",
                "large_harm_rate_030",
                "mean_absolute_error",
                "membership_rate",
            ],
        ),
        "",
        "## Outer-fold paired-stability-ranked gain",
        "",
        markdown_table(
            fold_frame,
            [
                "outer_fold",
                "paired_top20_gain",
                "paired_top20_win_rate",
                "paired_top20_large_harm_rate",
                "paired_expert_std_mean",
                "paired_baseline_std_mean",
                "paired_direction_flip_rate_mean",
            ],
        ),
        "",
        "## Verdict",
        "",
        f"- Risk signal supported: `{passed}`",
        f"- Primary win AUC: `{primary_win_auc:.6f}`",
        f"- Stability-only win AUC: `{stability_win_auc:.6f}`",
        f"- V9.25 base win AUC: `{base_win_auc:.6f}`",
        f"- Win-AUC gain over V9.25: `{win_auc_gain:.6f}`",
        f"- Primary gain Spearman: `{primary_gain_spearman:.6f}`",
        f"- Stability-only gain Spearman: `{stability_gain_spearman:.6f}`",
        f"- V9.25 base gain Spearman: `{base_gain_spearman:.6f}`",
        (
            f"- Large-harm AUC / AP: `{primary_harm_auc:.6f}` / "
            f"`{primary_harm_ap:.6f}`"
        ),
        (
            f"- Primary top-20% gain: "
            f"`{float(primary_top['mean_gain_vs_baseline']):.6f}`"
        ),
        (
            "- Primary top-20% gain CI: "
            f"`[{primary_bootstrap['gain_ci_low']:.6f}, "
            f"{primary_bootstrap['gain_ci_high']:.6f}]`"
        ),
        f"- Positive outer folds: `{positive_folds}/{cli.outer_folds}`",
        "",
        interpretation,
    ]
    (output / "v927_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.27 PAIRED PERTURBATION AUDIT COMPLETE")
    print("base experts trained or modified: False")
    print("raw waveform or pixels perturbed: False")
    print("pre-extracted model inputs perturbed: True")
    print("router or action selection executed: False")
    print("risk signal supported:", passed)
    print("report:", output / "v927_report.md")


if __name__ == "__main__":
    main()
