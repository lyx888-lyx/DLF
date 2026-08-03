"""Run the strict V9.28 text-anchor plus modality-residual team audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.function_space_features_v92 import (
    FEATURE_KEYS,
    FEATURE_SPACE_VERSION,
)
from trains.singleTask.modality_residual_team_v928 import (
    AUDIT_VERSION,
    HEAD_NAMES,
    PRIMARY_STRATEGY,
    TEXT_FEATURE_KEY,
    ResidualHeadConfigV928,
    ResidualTeamConfigV928,
    apply_strategy,
    build_modality_summary_matrix,
    crossfit_residual_predictions,
    fit_residual_head,
    fit_strategy_weights,
    predict_residual_head,
    prediction_metrics,
    serializable_bundle,
    strategy_definitions,
)
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.oof_group_splits_v92 import canonical_sample_id
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.same_stack_expert_factory_v919 import sha256
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    group_bootstrap_gain_interval,
    inner_crossfit_consensus,
    strategy_predictions,
)
from utils import assign_gpu, setup_seed


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
    parser.add_argument("--temporal-bins", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--correction-max", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--correction-l1", type=float, default=0.01)
    parser.add_argument("--coefficient-l2", type=float, default=0.01)
    parser.add_argument("--coefficient-upper-bound", type=float, default=1.0)
    parser.add_argument("--shrinkage-lambda", type=float, default=0.01)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1111)
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame, columns) -> str:
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


def vector(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64).reshape(-1)


def matrix(value) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    result = np.asarray(value, dtype=np.float64)
    if result.ndim == 3 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 2:
        raise ValueError(f"expected matrix, got {result.shape}")
    return result


def text_anchor(payload) -> np.ndarray:
    keys = tuple(payload.get("feature_keys", FEATURE_KEYS))
    if payload.get("feature_space") != FEATURE_SPACE_VERSION:
        raise RuntimeError(
            f"unexpected V9.19 function space: {payload.get('feature_space')}"
        )
    if not keys or keys[0] != TEXT_FEATURE_KEY:
        raise RuntimeError(
            f"V9.28 requires first function-space key {TEXT_FEATURE_KEY}, got {keys}"
        )
    features = matrix(payload["function_space"])
    if features.shape[1] < 1:
        raise ValueError("function_space has no text coordinate")
    result = features[:, 0].copy()
    if not np.isfinite(result).all():
        raise FloatingPointError("text anchor is non-finite")
    return result


def dataset_id_map(dataset) -> dict[str, int]:
    result = {}
    for index, value in enumerate(list(dataset.ids)):
        key = canonical_sample_id(value)
        if key in result:
            raise RuntimeError(f"duplicate dataset sample id: {key}")
        result[key] = int(index)
    return result


def map_ids(sample_ids, mapping) -> list[int]:
    values = []
    for value in sample_ids:
        key = canonical_sample_id(value)
        if key not in mapping:
            raise KeyError(f"sample id missing from dataset: {key}")
        values.append(mapping[key])
    return values


def check_dataset_labels(dataset, indices, pool_labels, tolerance=1e-6):
    dataset_labels = np.asarray(dataset.labels["M"], dtype=np.float64).reshape(-1)
    selected = dataset_labels[np.asarray(indices, dtype=np.int64)]
    labels = vector(pool_labels)
    if selected.shape != labels.shape:
        raise ValueError("dataset and pool labels have different shapes")
    delta = float(np.max(np.abs(selected - labels)))
    if delta > tolerance:
        raise RuntimeError(f"dataset/pool label mismatch: {delta}")
    return delta


def correlation(left, right) -> float:
    value = spearmanr(
        np.asarray(left, dtype=np.float64).reshape(-1),
        np.asarray(right, dtype=np.float64).reshape(-1),
    ).statistic
    return float(value) if np.isfinite(value) else 0.0


def scalar_metrics(metrics):
    return {key: value for key, value in metrics.items() if np.isscalar(value)}


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

    dataset = MMDataset(args, mode="train")
    id_map = dataset_id_map(dataset)
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v928_modality_residual_team_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    model_root = output / "residual_models"
    model_root.mkdir(parents=True, exist_ok=True)

    head_config = ResidualHeadConfigV928(
        temporal_bins=cli.temporal_bins,
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        correction_max=cli.correction_max,
        epochs=cli.epochs,
        batch_size=cli.batch_size,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        correction_l1=cli.correction_l1,
    )
    team_config = ResidualTeamConfigV928(
        head=head_config,
        coefficient_l2=cli.coefficient_l2,
        coefficient_upper_bound=cli.coefficient_upper_bound,
        bootstrap_repetitions=cli.bootstrap_repetitions,
        bootstrap_seed=cli.bootstrap_seed,
    )
    team_config.validate()
    consensus_config = ConsensusConfigV921(
        shrinkage_lambda=cli.shrinkage_lambda,
        bootstrap_repetitions=cli.bootstrap_repetitions,
        bootstrap_seed=cli.bootstrap_seed,
    )

    metric_rows = []
    coefficient_rows = []
    head_rows = []
    split_rows = []
    source_rows = []
    feature_rows = []
    prediction_frames = []
    fold_primary_rows = []

    for outer_fold in range(cli.outer_folds):
        fold_dir = root / f"outer_fold_{outer_fold}"
        inner_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        outer_path = (
            fold_dir
            / "outer_deployment_stack"
            / "same_stack_target_pool_v919.pth"
        )
        if not inner_path.is_file() or not outer_path.is_file():
            raise FileNotFoundError(
                f"missing V9.19 pool for outer fold {outer_fold}: "
                f"{inner_path} / {outer_path}"
            )
        inner_payload = torch.load(inner_path, map_location="cpu")
        outer_payload = torch.load(outer_path, map_location="cpu")
        inner_view = normalize_pool(inner_payload)
        outer_view = normalize_pool(outer_payload)
        inner_folds = np.asarray(
            torch.as_tensor(inner_payload["fold_index"]).view(-1).numpy(),
            dtype=np.int64,
        )
        if len(inner_folds) != len(inner_view.labels) or bool((inner_folds < 0).any()):
            raise ValueError("inner OOF fold assignment is invalid")

        inner_text = text_anchor(inner_payload)
        outer_text = text_anchor(outer_payload)
        inner_labels = vector(inner_view.labels)
        outer_labels = vector(outer_view.labels)
        inner_indices = map_ids(inner_view.sample_ids, id_map)
        outer_indices = map_ids(outer_view.sample_ids, id_map)
        inner_label_delta = check_dataset_labels(
            dataset, inner_indices, inner_view.labels
        )
        outer_label_delta = check_dataset_labels(
            dataset, outer_indices, outer_view.labels
        )

        inner_consensus = inner_crossfit_consensus(
            inner_payload, consensus_config
        )
        inner_v921 = vector(
            inner_consensus["predictions"]["convex_shrinkage"]
        )
        outer_consensus = strategy_predictions(
            inner_view, outer_view, consensus_config
        )
        outer_v921 = vector(
            outer_consensus["predictions"]["convex_shrinkage"]
        )

        inner_corrections = []
        outer_corrections = []
        fold_model_dir = model_root / f"outer_fold_{outer_fold}"
        fold_model_dir.mkdir(parents=True, exist_ok=True)

        for head_index, head_name in enumerate(HEAD_NAMES):
            inner_features = build_modality_summary_matrix(
                dataset,
                inner_indices,
                head_name,
                head_config.temporal_bins,
            )
            outer_features = build_modality_summary_matrix(
                dataset,
                outer_indices,
                head_name,
                head_config.temporal_bins,
            )
            if inner_features.shape[1] != outer_features.shape[1]:
                raise RuntimeError("inner/outer residual feature dimension changed")
            for feature_index in range(inner_features.shape[1]):
                feature_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "head": head_name,
                        "feature_index": feature_index,
                        "feature_name": f"{head_name}_summary_{feature_index}",
                        "uses_text_token_or_embedding": False,
                    }
                )
            for offset, name in enumerate(
                ("text_anchor_scalar", "text_anchor_abs", "text_anchor_squared"),
                start=inner_features.shape[1],
            ):
                feature_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "head": head_name,
                        "feature_index": offset,
                        "feature_name": name,
                        "uses_text_token_or_embedding": False,
                    }
                )

            crossfit = crossfit_residual_predictions(
                inner_features,
                inner_text,
                inner_labels,
                inner_folds,
                head_config,
                args.device,
                seed=(
                    cli.seed
                    + 100003 * (outer_fold + 1)
                    + 1009 * (head_index + 1)
                ),
            )
            inner_correction = crossfit["prediction"]
            inner_corrections.append(inner_correction)
            for row in crossfit["fold_rows"]:
                head_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "split": "inner_oof",
                        "head": head_name,
                        **row,
                    }
                )

            bundle = fit_residual_head(
                inner_features,
                inner_text,
                inner_labels,
                head_config,
                args.device,
                seed=(
                    cli.seed
                    + 200003 * (outer_fold + 1)
                    + 2017 * (head_index + 1)
                ),
            )
            outer_correction = predict_residual_head(
                bundle,
                outer_features,
                outer_text,
                args.device,
            )
            outer_corrections.append(outer_correction)
            checkpoint = fold_model_dir / f"{head_name}_residual_v928.pth"
            torch.save(serializable_bundle(bundle), checkpoint)
            pd.DataFrame(bundle["history"]).assign(
                outer_fold=outer_fold, head=head_name
            ).to_csv(
                fold_model_dir / f"{head_name}_training_history.csv",
                index=False,
            )
            outer_residual = outer_labels - outer_text
            head_rows.append(
                {
                    "outer_fold": outer_fold,
                    "split": "outer_holdout",
                    "head": head_name,
                    "inner_fold": "aggregate",
                    "training_count": len(inner_labels),
                    "holdout_count": len(outer_labels),
                    "correction_mean": float(outer_correction.mean()),
                    "correction_abs_mean": float(
                        np.abs(outer_correction).mean()
                    ),
                    "residual_spearman": correlation(
                        outer_correction, outer_residual
                    ),
                    "zeroish_rate_abs_lt_002": float(
                        (np.abs(outer_correction) < 0.02).mean()
                    ),
                    "checkpoint": str(checkpoint),
                    "checkpoint_sha256": sha256(checkpoint),
                }
            )
            del bundle
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        inner_corrections = np.column_stack(inner_corrections)
        outer_corrections = np.column_stack(outer_corrections)
        weights = fit_strategy_weights(
            inner_corrections,
            inner_text,
            inner_labels,
            team_config,
        )

        frame = pd.DataFrame(
            {
                "outer_fold": outer_fold,
                "sample_id": outer_view.sample_ids,
                "group_id": outer_view.group_ids,
                "label": outer_labels,
                "text_anchor": outer_text,
                "v921_convex_shrinkage": outer_v921,
                **{
                    f"correction_{name}": outer_corrections[:, index]
                    for index, name in enumerate(HEAD_NAMES)
                },
            }
        )
        for strategy, weight in weights.items():
            inner_prediction = apply_strategy(
                inner_text, inner_corrections, weight
            )
            outer_prediction = apply_strategy(
                outer_text, outer_corrections, weight
            )
            frame[f"prediction_{strategy}"] = outer_prediction
            for split, prediction, labels, text, v921 in (
                (
                    "inner_oof",
                    inner_prediction,
                    inner_labels,
                    inner_text,
                    inner_v921,
                ),
                (
                    "outer_holdout",
                    outer_prediction,
                    outer_labels,
                    outer_text,
                    outer_v921,
                ),
            ):
                metric_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "split": split,
                        "strategy": strategy,
                        **scalar_metrics(
                            prediction_metrics(
                                prediction,
                                labels,
                                text,
                                v921,
                            )
                        ),
                    }
                )
            for head_index, head_name in enumerate(HEAD_NAMES):
                coefficient_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "strategy": strategy,
                        "head": head_name,
                        "coefficient": float(weight[head_index]),
                    }
                )

        metric_rows.append(
            {
                "outer_fold": outer_fold,
                "split": "outer_holdout",
                "strategy": "v921_convex_shrinkage",
                **scalar_metrics(
                    prediction_metrics(
                        outer_v921,
                        outer_labels,
                        outer_text,
                        outer_v921,
                    )
                ),
            }
        )
        prediction_frames.append(frame)
        primary_metrics = prediction_metrics(
            frame[f"prediction_{PRIMARY_STRATEGY}"],
            outer_labels,
            outer_text,
            outer_v921,
        )
        fold_primary_rows.append(
            {
                "outer_fold": outer_fold,
                **scalar_metrics(primary_metrics),
                **{
                    f"weight_{name}": float(
                        weights[PRIMARY_STRATEGY][index]
                    )
                    for index, name in enumerate(HEAD_NAMES)
                },
            }
        )
        for inner_fold in sorted(np.unique(inner_folds).tolist()):
            mask = inner_folds == int(inner_fold)
            groups = {
                str(inner_view.group_ids[index])
                for index in np.flatnonzero(mask)
            }
            split_rows.append(
                {
                    "outer_fold": outer_fold,
                    "inner_fold": int(inner_fold),
                    "sample_count": int(mask.sum()),
                    "group_count": len(groups),
                }
            )
        source_rows.extend(
            [
                {
                    "outer_fold": outer_fold,
                    "source": "inner_oof_v919",
                    "path": str(inner_path),
                    "sha256": sha256(inner_path),
                    "sample_count": len(inner_labels),
                    "dataset_label_max_abs": inner_label_delta,
                },
                {
                    "outer_fold": outer_fold,
                    "source": "outer_holdout_v919",
                    "path": str(outer_path),
                    "sha256": sha256(outer_path),
                    "sample_count": len(outer_labels),
                    "dataset_label_max_abs": outer_label_delta,
                },
            ]
        )

    predictions = pd.concat(prediction_frames, ignore_index=True)
    fold_metrics = pd.DataFrame(fold_primary_rows)
    metric_frame = pd.DataFrame(metric_rows)
    coefficient_frame = pd.DataFrame(coefficient_rows)
    head_frame = pd.DataFrame(head_rows)
    split_frame = pd.DataFrame(split_rows)
    source_frame = pd.DataFrame(source_rows)
    feature_frame = pd.DataFrame(feature_rows)

    aggregate_rows = []
    labels = predictions["label"].to_numpy(dtype=np.float64)
    text = predictions["text_anchor"].to_numpy(dtype=np.float64)
    v921 = predictions["v921_convex_shrinkage"].to_numpy(dtype=np.float64)
    for strategy in strategy_definitions():
        prediction = predictions[f"prediction_{strategy}"].to_numpy(
            dtype=np.float64
        )
        aggregate_rows.append(
            {
                "strategy": strategy,
                **scalar_metrics(
                    prediction_metrics(prediction, labels, text, v921)
                ),
            }
        )
    aggregate_rows.append(
        {
            "strategy": "v921_convex_shrinkage",
            **scalar_metrics(prediction_metrics(v921, labels, text, v921)),
        }
    )
    aggregate_frame = pd.DataFrame(aggregate_rows)

    primary_prediction = predictions[
        f"prediction_{PRIMARY_STRATEGY}"
    ].to_numpy(dtype=np.float64)
    text_gain = np.abs(text - labels) - np.abs(primary_prediction - labels)
    v921_gain = np.abs(v921 - labels) - np.abs(primary_prediction - labels)
    bootstrap_text = group_bootstrap_gain_interval(
        text_gain,
        predictions["group_id"].tolist(),
        team_config.bootstrap_repetitions,
        team_config.bootstrap_seed,
    )
    bootstrap_v921 = group_bootstrap_gain_interval(
        v921_gain,
        predictions["group_id"].tolist(),
        team_config.bootstrap_repetitions,
        team_config.bootstrap_seed + 17,
    )
    primary_aggregate = aggregate_frame[
        aggregate_frame.strategy == PRIMARY_STRATEGY
    ].iloc[0]
    positive_folds = int((fold_metrics.gain_vs_text > 0.0).sum())
    worst_degradation = float(max(0.0, -fold_metrics.gain_vs_text.min()))
    incremental_supported = bool(
        float(primary_aggregate.gain_vs_text)
        >= team_config.required_gain_vs_text
        and float(bootstrap_text["gain_ci_low"])
        > team_config.required_gain_ci_low
        and positive_folds >= team_config.required_positive_outer_folds
        and worst_degradation <= team_config.max_worst_fold_degradation
    )
    competitive_with_v921 = bool(
        float(primary_aggregate.gain_vs_v921)
        >= team_config.competitive_margin_vs_v921
    )
    deployment_replacement_supported = bool(
        incremental_supported and competitive_with_v921
    )

    success_gate = {
        "primary_strategy": PRIMARY_STRATEGY,
        "primary_mae": float(primary_aggregate.mae),
        "text_anchor_mae": float(primary_aggregate.text_anchor_mae),
        "v921_mae": float(primary_aggregate.v921_mae),
        "gain_vs_text": float(primary_aggregate.gain_vs_text),
        "required_gain_vs_text": team_config.required_gain_vs_text,
        "gain_vs_text_bootstrap": bootstrap_text,
        "required_gain_ci_low": team_config.required_gain_ci_low,
        "positive_outer_folds": positive_folds,
        "required_positive_outer_folds": team_config.required_positive_outer_folds,
        "worst_fold_degradation": worst_degradation,
        "max_worst_fold_degradation": team_config.max_worst_fold_degradation,
        "gain_vs_v921": float(primary_aggregate.gain_vs_v921),
        "competitive_margin_vs_v921": team_config.competitive_margin_vs_v921,
        "gain_vs_v921_bootstrap": bootstrap_v921,
        "incremental_modality_signal_supported": incremental_supported,
        "competitive_with_v921": competitive_with_v921,
        "deployment_replacement_supported": deployment_replacement_supported,
    }
    interpretation = (
        "V9.28 supports a deployment-stable non-text residual signal and is "
        "competitive with V9.21. A separate untouched dataset is still needed "
        "for confirmation."
        if deployment_replacement_supported
        else (
            "V9.28 found deployment-stable audio/visual residual information, "
            "but the fixed residual team did not yet match V9.21. Treat this as "
            "mechanism evidence, not a replacement model."
            if incremental_supported
            else (
                "V9.28 did not establish deployment-stable audio/visual residual "
                "information beyond the strict text anchor. Do not add these "
                "residual heads to the deployed predictor."
            )
        )
    )

    metric_frame.to_csv(output / "v928_fold_metrics.csv", index=False)
    aggregate_frame.to_csv(output / "v928_aggregate_metrics.csv", index=False)
    coefficient_frame.to_csv(
        output / "v928_coefficient_inventory.csv", index=False
    )
    head_frame.to_csv(output / "v928_residual_head_metrics.csv", index=False)
    fold_metrics.to_csv(output / "v928_outer_fold_primary.csv", index=False)
    predictions.to_csv(output / "v928_outer_predictions.csv", index=False)
    split_frame.to_csv(output / "v928_split_manifest.csv", index=False)
    source_frame.to_csv(output / "v928_source_integrity.csv", index=False)
    feature_frame.to_csv(output / "v928_feature_schema.csv", index=False)

    summary = {
        "version": AUDIT_VERSION,
        "dataset": cli.dataset,
        "seed": cli.seed,
        "outer_folds": cli.outer_folds,
        "text_anchor_source": (
            "V9.19 strict fold-local function_space[logits_l_hetero]"
        ),
        "residual_head_inputs": {
            "audio": "audio sequence summary + scalar text anchor",
            "vision": "vision sequence summary + scalar text anchor",
            "audio_visual": (
                "audio/vision sequence summaries + scalar text anchor"
            ),
            "text_tokens_or_embeddings_available": False,
        },
        "router_or_sample_dependent_weights": False,
        "configs": {
            "head": asdict(head_config),
            "team": asdict(team_config),
            "consensus": asdict(consensus_config),
        },
        "success_gate": success_gate,
        "incremental_modality_signal_supported": incremental_supported,
        "deployment_replacement_supported": deployment_replacement_supported,
        "interpretation": interpretation,
        "provenance": {
            "inner_corrections_are_complete_crossfit_predictions": True,
            "outer_holdout_labels_used_for_training": False,
            "outer_holdout_labels_used_for_coefficient_fit": False,
            "global_coefficients_frozen_per_outer_fold": True,
            "nonnegative_coefficients_can_shrink_to_zero": True,
            "no_router": True,
            "no_per_sample_gate": True,
            "exploratory_after_prior_mosi_outer_results": True,
        },
    }
    (output / "v928_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    aggregate_view = aggregate_frame[
        aggregate_frame.strategy.isin(
            [
                "text_anchor",
                "text_plus_audio",
                "text_plus_vision",
                "text_plus_audio_and_vision",
                PRIMARY_STRATEGY,
                "v921_convex_shrinkage",
            ]
        )
    ]
    coefficient_view = coefficient_frame[
        coefficient_frame.strategy == PRIMARY_STRATEGY
    ].pivot(index="outer_fold", columns="head", values="coefficient").reset_index()
    report = [
        "# V9.28 Modality Residual Team Audit",
        "",
        (
            "The strict V9.19 text-only function-space coordinate is the frozen "
            "anchor. Audio and vision residual heads receive no text tokens or "
            "text embeddings. Inner OOF corrections fit one non-negative global "
            "coefficient vector, which is frozen for the outer holdout. There is "
            "no Router, sample gate, or sample-dependent mixture."
        ),
        "",
        "## Aggregate comparison",
        "",
        markdown_table(
            aggregate_view,
            [
                "strategy",
                "mae",
                "text_anchor_mae",
                "gain_vs_text",
                "win_rate_vs_text",
                "harm_over_010_rate_vs_text",
                "v921_mae",
                "gain_vs_v921",
            ],
        ),
        "",
        "## Primary outer-fold results",
        "",
        markdown_table(
            fold_metrics,
            [
                "outer_fold",
                "mae",
                "gain_vs_text",
                "gain_vs_v921",
                "win_rate_vs_text",
                "harm_over_010_rate_vs_text",
                "weight_audio",
                "weight_vision",
                "weight_audio_visual",
            ],
        ),
        "",
        "## Primary fixed coefficients",
        "",
        markdown_table(
            coefficient_view,
            ["outer_fold", "audio", "vision", "audio_visual"],
        ),
        "",
        "## Residual-head outer diagnostics",
        "",
        markdown_table(
            head_frame[head_frame.split == "outer_holdout"],
            [
                "outer_fold",
                "head",
                "correction_abs_mean",
                "residual_spearman",
                "zeroish_rate_abs_lt_002",
            ],
        ),
        "",
        "## Verdict",
        "",
        f"- Incremental non-text signal supported: `{incremental_supported}`",
        f"- Competitive with V9.21: `{competitive_with_v921}`",
        f"- Deployment replacement supported: `{deployment_replacement_supported}`",
        f"- Primary MAE: `{float(primary_aggregate.mae):.6f}`",
        f"- Text-anchor MAE: `{float(primary_aggregate.text_anchor_mae):.6f}`",
        f"- V9.21 MAE: `{float(primary_aggregate.v921_mae):.6f}`",
        f"- Gain versus text: `{float(primary_aggregate.gain_vs_text):+.6f}`",
        (
            "- Gain-versus-text CI: "
            f"`[{bootstrap_text['gain_ci_low']:.6f}, "
            f"{bootstrap_text['gain_ci_high']:.6f}]`"
        ),
        f"- Gain versus V9.21: `{float(primary_aggregate.gain_vs_v921):+.6f}`",
        f"- Positive outer folds versus text: `{positive_folds}/{cli.outer_folds}`",
        "",
        interpretation,
        "",
        (
            "This MOSI experiment is exploratory because its design follows "
            "earlier inspection of the same outer folds. Untouched MOSEI or a "
            "separately locked holdout is required for confirmation."
        ),
    ]
    (output / "v928_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.28 MODALITY RESIDUAL TEAM AUDIT COMPLETE")
    print("text tokens or embeddings given to residual heads: False")
    print("router or sample-dependent weights: False")
    print("incremental modality signal supported:", incremental_supported)
    print("deployment replacement supported:", deployment_replacement_supported)
    print("report:", output / "v928_report.md")


if __name__ == "__main__":
    main()
