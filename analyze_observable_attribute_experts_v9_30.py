"""Strict nested V9.30 observable-attribute expert experiment.

The script reuses each V9.19 stack's fold-local ModDrop and CFCompat
checkpoints, but replaces all true-label-region specialists with experts whose
roles and soft training weights are computable at inference time.
"""

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
from trains.singleTask.cfcompat_fold_training_v92 import (
    DLF,
    MissingModalityWrapper,
    _load_state,
    predict_wrapper_lav,
)
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.observable_attribute_experts_v930 import (
    ACTION_NAMES_V930,
    AUDIT_VERSION,
    EXPERT_NAMES,
    MODE_NAMES,
    ObservableAttributeConfigV930,
    applicability_lookup,
    attribute_weighted_prediction,
    build_attribute_cache,
    fit_fixed_convex,
    inner_crossfit_fixed_convex,
    prediction_lookup,
    prediction_metrics,
    predict_observable_expert,
    residual_correlation,
    success_gate,
    train_observable_expert,
)
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
    parser.add_argument("--feature-batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=18)
    parser.add_argument("--early-stop", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--applicability-floor", type=float, default=0.10)
    parser.add_argument("--applicability-power", type=float, default=2.0)
    parser.add_argument("--global-loss-weight", type=float, default=0.25)
    parser.add_argument("--anchor-preservation-weight", type=float, default=0.10)
    parser.add_argument("--validation-global-weight", type=float, default=0.25)
    parser.add_argument("--shrinkage-lambda", type=float, default=0.01)
    parser.add_argument("--attribute-anchor-prior", type=float, default=1.0)
    parser.add_argument("--required-gain-vs-v921", type=float, default=0.005)
    parser.add_argument("--required-nondegrading-folds", type=int, default=4)
    parser.add_argument("--max-worst-fold-degradation", type=float, default=0.005)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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


def dataset_index_map(dataset) -> Dict[str, int]:
    result = {}
    for index, value in enumerate(list(dataset.ids)):
        key = canonical_sample_id(value)
        if key in result:
            raise RuntimeError(f"duplicate dataset sample ID: {key}")
        result[key] = int(index)
    return result


def payload_indices(payload: Mapping[str, object], id_map: Mapping[str, int]) -> list[int]:
    if "sample_indices" in payload:
        values = [int(value) for value in payload["sample_indices"]]
    else:
        values = [
            id_map[canonical_sample_id(value)] for value in payload["sample_ids"]
        ]
    if len(values) != len(payload["sample_ids"]) or len(set(values)) != len(values):
        raise RuntimeError("invalid payload sample indices")
    return values


def load_wrapper(args, checkpoint: Path):
    model = MissingModalityWrapper(
        DLF(args).to(args.device),
        int(args.feature_dims[1]),
        int(args.feature_dims[2]),
    ).to(args.device)
    model.load_state_dict(_load_state(checkpoint, args.device), strict=True)
    model.eval()
    return model


def rows_by_index(rows: Sequence[Mapping[str, object]]) -> Dict[int, Mapping[str, object]]:
    result = {}
    for row in rows:
        index = int(row["sample_index"])
        if index in result:
            raise RuntimeError(f"duplicate row index: {index}")
        result[index] = row
    return result


def discover_v919_stack(fold_dir: Path, target_indices: Sequence[int]) -> Path:
    expected = set(int(value) for value in target_indices)
    matches = []
    for path in fold_dir.rglob("same_stack_target_pool_v919.pth"):
        payload = torch.load(path, map_location="cpu")
        local = set(int(value) for value in payload.get("sample_indices", ()))
        if local == expected and len(local) == len(expected):
            matches.append(path.parent)
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one V9.19 stack for {len(expected)} targets, found {matches}"
        )
    return matches[0]


def cache_frame(cache: Mapping[str, object], stack_name: str) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "stack": stack_name,
            "sample_index": cache["indices"],
            "sample_id": cache["sample_ids"],
            **{
                f"mode_prediction_{mode}": cache["mode_predictions"][mode]
                for mode in MODE_NAMES
            },
            **{
                f"raw_{name}": cache["raw_scores"][name]
                for name in EXPERT_NAMES
            },
            **{
                f"applicability_{name}": np.asarray(cache["applicability"])[:, index]
                for index, name in enumerate(EXPERT_NAMES)
            },
        }
    )
    return frame


def train_stack(
    args,
    dataset,
    source_stack_dir: Path,
    target_indices: Sequence[int],
    output_dir: Path,
    config: ObservableAttributeConfigV930,
    seed: int,
    resume: bool,
):
    source_pool_path = source_stack_dir / "same_stack_target_pool_v919.pth"
    source_payload = torch.load(source_pool_path, map_location="cpu")
    recipe = source_payload.get("recipe", {})
    train_indices = [int(value) for value in recipe.get("train_indices", ())]
    valid_indices = [int(value) for value in recipe.get("valid_indices", ())]
    target_indices = [int(value) for value in target_indices]
    if not train_indices or not valid_indices:
        raise RuntimeError(f"V9.19 recipe lacks train/valid indices: {source_stack_dir}")
    if set(train_indices) & set(valid_indices):
        raise RuntimeError("observable expert train/valid overlap")
    if set(train_indices + valid_indices) & set(target_indices):
        raise RuntimeError("observable expert target entered fitting")

    output_dir.mkdir(parents=True, exist_ok=True)
    pool_path = output_dir / "observable_attribute_target_pool_v930.pth"
    source_manifest = {
        "source_stack_dir": str(source_stack_dir),
        "source_pool_sha256": file_sha256(source_pool_path),
        "config": asdict(config),
        "train_indices": train_indices,
        "valid_indices": valid_indices,
        "target_indices": target_indices,
        "seed": int(seed),
    }
    if resume and pool_path.is_file():
        cached = torch.load(pool_path, map_location="cpu")
        if cached.get("source_manifest") == source_manifest:
            return cached

    evaluator_checkpoint = source_stack_dir / "moddrop_evaluator_best_inner_valid.pth"
    anchor_checkpoint = source_stack_dir / "cfcompat_student_best_inner_valid.pth"
    for path in (evaluator_checkpoint, anchor_checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)

    evaluator = load_wrapper(args, evaluator_checkpoint)
    anchor_model = load_wrapper(args, anchor_checkpoint)
    train_signal_loader = build_subset_loader(
        dataset,
        train_indices,
        config.feature_batch_size,
        config.num_workers,
        False,
        seed,
    )
    valid_signal_loader = build_subset_loader(
        dataset,
        valid_indices,
        config.feature_batch_size,
        config.num_workers,
        False,
        seed,
    )
    target_loader = build_subset_loader(
        dataset,
        target_indices,
        config.feature_batch_size,
        config.num_workers,
        False,
        seed,
    )
    train_cache = build_attribute_cache(
        evaluator, train_signal_loader, args.device, config=config
    )
    valid_cache = build_attribute_cache(
        evaluator,
        valid_signal_loader,
        args.device,
        calibrator=train_cache["calibrator"],
    )
    target_cache = build_attribute_cache(
        evaluator,
        target_loader,
        args.device,
        calibrator=train_cache["calibrator"],
    )
    train_app = applicability_lookup(train_cache)
    valid_app = applicability_lookup(valid_cache)

    anchor_train_rows = predict_wrapper_lav(
        anchor_model, train_signal_loader, args.device, False
    )
    anchor_target_rows = predict_wrapper_lav(
        anchor_model, target_loader, args.device, False
    )
    train_anchor = prediction_lookup(anchor_train_rows)
    target_anchor_map = rows_by_index(anchor_target_rows)
    if set(target_anchor_map) != set(target_indices):
        raise RuntimeError("anchor target predictions do not match requested indices")

    histories = []
    target_expert_maps = {}
    checkpoint_rows = []
    for expert_index, expert_name in enumerate(EXPERT_NAMES):
        train_loader = build_subset_loader(
            dataset,
            train_indices,
            config.batch_size,
            config.num_workers,
            True,
            int(seed) + 1009 * (expert_index + 1),
        )
        valid_loader = build_subset_loader(
            dataset,
            valid_indices,
            config.batch_size,
            config.num_workers,
            False,
            seed,
        )
        checkpoint = output_dir / f"{expert_name}_expert_v930.pth"
        fitted = train_observable_expert(
            args,
            train_loader,
            valid_loader,
            anchor_checkpoint,
            expert_name,
            train_app,
            valid_app,
            train_anchor,
            config,
            checkpoint,
        )
        local_target_loader = build_subset_loader(
            dataset,
            target_indices,
            config.feature_batch_size,
            config.num_workers,
            False,
            seed,
        )
        predicted = predict_observable_expert(
            fitted["model"], local_target_loader, args.device, fitted["mode"]
        )
        target_expert_maps[expert_name] = {
            int(index): float(value)
            for index, value in zip(
                predicted["sample_indices"], predicted["prediction"]
            )
        }
        histories.extend(
            {"expert": expert_name, **row} for row in fitted["history"]
        )
        checkpoint_rows.append(
            {
                "expert": expert_name,
                "mode": fitted["mode"],
                "checkpoint": str(checkpoint),
                "sha256": file_sha256(checkpoint),
                "best_epoch": fitted["best_epoch"],
                "best_valid_objective": fitted["best_valid_objective"],
                "trainable_parameters": fitted["trainable_parameters"],
            }
        )
        del fitted["model"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    ordered = target_indices
    labels = np.asarray(
        [float(target_anchor_map[index]["label"]) for index in ordered],
        dtype=np.float64,
    )
    sample_ids = [
        canonical_sample_id(target_anchor_map[index]["sample_id"])
        for index in ordered
    ]
    anchor = np.asarray(
        [float(target_anchor_map[index]["prediction"]) for index in ordered],
        dtype=np.float64,
    )
    experts = np.column_stack(
        [
            np.asarray(
                [target_expert_maps[name][index] for index in ordered],
                dtype=np.float64,
            )
            for name in EXPERT_NAMES
        ]
    )
    target_position = {
        int(index): offset for offset, index in enumerate(target_cache["indices"])
    }
    positions = [target_position[index] for index in ordered]
    applicability = np.asarray(
        target_cache["applicability"], dtype=np.float64
    )[positions]
    mode_predictions = {
        mode: np.asarray(
            target_cache["mode_predictions"][mode], dtype=np.float64
        )[positions]
        for mode in MODE_NAMES
    }
    raw_scores = {
        name: np.asarray(
            target_cache["raw_scores"][name], dtype=np.float64
        )[positions]
        for name in EXPERT_NAMES
    }
    actions = np.column_stack([anchor, experts])
    if not np.isfinite(actions).all() or not np.isfinite(applicability).all():
        raise FloatingPointError("V9.30 stack output is non-finite")

    pd.DataFrame(histories).to_csv(
        output_dir / "expert_training_history_v930.csv", index=False
    )
    pd.DataFrame(checkpoint_rows).to_csv(
        output_dir / "expert_checkpoints_v930.csv", index=False
    )
    cache_frame(train_cache, "train").to_csv(
        output_dir / "train_attribute_cache_v930.csv", index=False
    )
    cache_frame(valid_cache, "valid").to_csv(
        output_dir / "valid_attribute_cache_v930.csv", index=False
    )
    cache_frame(target_cache, "target").to_csv(
        output_dir / "target_attribute_cache_v930.csv", index=False
    )

    payload = {
        "version": AUDIT_VERSION,
        "method": "observable_attributes_then_define_experts",
        "source_manifest": source_manifest,
        "sample_indices": ordered,
        "sample_ids": sample_ids,
        "group_ids": [conversation_group_id(value) for value in sample_ids],
        "labels": torch.tensor(labels, dtype=torch.float32).view(-1, 1),
        "anchor": torch.tensor(anchor, dtype=torch.float32).view(-1, 1),
        "expert_predictions": torch.tensor(
            experts, dtype=torch.float32
        ).unsqueeze(-1),
        "expert_applicability": torch.tensor(
            applicability, dtype=torch.float32
        ),
        "mode_predictions": {
            mode: torch.tensor(value, dtype=torch.float32)
            for mode, value in mode_predictions.items()
        },
        "raw_attribute_scores": {
            name: torch.tensor(value, dtype=torch.float32)
            for name, value in raw_scores.items()
        },
        "action_names": ACTION_NAMES_V930,
        "expert_names": EXPERT_NAMES,
        "attribute_calibrator": train_cache["calibrator"].serializable(),
        "provenance": {
            "attribute_inputs": [
                f"prediction_{mode}" for mode in MODE_NAMES
            ],
            "labels_used_to_compute_attributes": False,
            "labels_used_for_expert_supervised_loss": True,
            "target_labels_used_for_training": False,
            "target_labels_used_for_attribute_calibration": False,
            "winner_or_regret_router_trained": False,
            "expert_modes_fixed_before_target_evaluation": True,
        },
    }
    torch.save(payload, pool_path)
    del evaluator, anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def payload_arrays(payload: Mapping[str, object]):
    anchor = torch.as_tensor(payload["anchor"]).view(-1).double().numpy()
    experts = torch.as_tensor(payload["expert_predictions"]).double().numpy()
    if experts.ndim == 3 and experts.shape[-1] == 1:
        experts = experts[..., 0]
    labels = torch.as_tensor(payload["labels"]).view(-1).double().numpy()
    applicability = (
        torch.as_tensor(payload["expert_applicability"]).double().numpy()
    )
    return np.column_stack([anchor, experts]), labels, applicability


def aggregate_inner_stack_payloads(
    template_payload: Mapping[str, object],
    results_by_fold: Mapping[int, Mapping[str, object]],
    id_map: Mapping[str, int],
):
    indices = payload_indices(template_payload, id_map)
    fold_index = (
        torch.as_tensor(template_payload["fold_index"])
        .view(-1)
        .long()
        .numpy()
    )
    position = {index: offset for offset, index in enumerate(indices)}
    actions = np.full(
        (len(indices), len(ACTION_NAMES_V930)), np.nan, dtype=np.float64
    )
    labels = np.full(len(indices), np.nan, dtype=np.float64)
    applicability = np.full(
        (len(indices), len(EXPERT_NAMES)), np.nan, dtype=np.float64
    )
    sample_ids = [None] * len(indices)
    group_ids = [None] * len(indices)
    for fold, payload in results_by_fold.items():
        local_actions, local_labels, local_app = payload_arrays(payload)
        local_indices = [int(value) for value in payload["sample_indices"]]
        for local_offset, index in enumerate(local_indices):
            target = position[index]
            if int(fold_index[target]) != int(fold):
                raise RuntimeError("inner fold assignment changed")
            actions[target] = local_actions[local_offset]
            labels[target] = local_labels[local_offset]
            applicability[target] = local_app[local_offset]
            sample_ids[target] = str(payload["sample_ids"][local_offset])
            group_ids[target] = str(payload["group_ids"][local_offset])
    if not np.isfinite(actions).all() or not np.isfinite(labels).all():
        raise FloatingPointError("inner observable pool is incomplete")
    return {
        "indices": indices,
        "fold_index": fold_index,
        "actions": actions,
        "labels": labels,
        "applicability": applicability,
        "sample_ids": sample_ids,
        "group_ids": group_ids,
    }


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

    config = ObservableAttributeConfigV930(
        batch_size=cli.batch_size,
        feature_batch_size=cli.feature_batch_size,
        num_workers=cli.num_workers,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        grad_clip_norm=cli.grad_clip_norm,
        use_amp=not cli.disable_amp,
        applicability_floor=cli.applicability_floor,
        applicability_power=cli.applicability_power,
        global_loss_weight=cli.global_loss_weight,
        anchor_preservation_weight=cli.anchor_preservation_weight,
        validation_global_weight=cli.validation_global_weight,
        shrinkage_lambda=cli.shrinkage_lambda,
        attribute_anchor_prior=cli.attribute_anchor_prior,
        required_gain_vs_v921=cli.required_gain_vs_v921,
        required_nondegrading_folds=cli.required_nondegrading_folds,
        max_worst_fold_degradation=cli.max_worst_fold_degradation,
    )
    config.validate()
    dataset = MMDataset(args, mode="train")
    id_map = dataset_index_map(dataset)
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v930_observable_attribute_experts"
    )
    output.mkdir(parents=True, exist_ok=True)
    consensus_reference = ConsensusConfigV921(
        shrinkage_lambda=cli.shrinkage_lambda
    )

    fold_rows = []
    weight_rows = []
    prediction_frames = []
    source_rows = []
    correlation_rows = []
    applicability_rows = []

    for outer_fold in range(int(cli.outer_folds)):
        fold_dir = root / f"outer_fold_{outer_fold}"
        old_inner_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        old_outer_path = (
            fold_dir
            / "outer_deployment_stack"
            / "same_stack_target_pool_v919.pth"
        )
        for role, path in (
            ("inner_oof_pool", old_inner_path),
            ("outer_pool", old_outer_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_rows.append(
                {
                    "outer_fold": outer_fold,
                    "role": role,
                    "path": str(path),
                    "sha256": file_sha256(path),
                }
            )
        old_inner_payload = torch.load(old_inner_path, map_location="cpu")
        old_outer_payload = torch.load(old_outer_path, map_location="cpu")
        old_inner = normalize_pool(old_inner_payload)
        old_outer = normalize_pool(old_outer_payload)
        old_reference = strategy_predictions(
            old_inner, old_outer, consensus_reference
        )
        old_v921 = (
            old_reference["predictions"]["convex_shrinkage"]
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )

        inner_indices = payload_indices(old_inner_payload, id_map)
        inner_fold_index = (
            torch.as_tensor(old_inner_payload["fold_index"])
            .view(-1)
            .long()
            .numpy()
        )
        inner_results = {}
        for inner_fold in sorted(np.unique(inner_fold_index).tolist()):
            target = [
                index
                for index, fold in zip(inner_indices, inner_fold_index)
                if int(fold) == int(inner_fold)
            ]
            source_stack = discover_v919_stack(fold_dir, target)
            inner_results[int(inner_fold)] = train_stack(
                args,
                dataset,
                source_stack,
                target,
                output
                / f"outer_fold_{outer_fold}"
                / f"inner_fold_{inner_fold}",
                config,
                cli.seed
                + 100003 * (outer_fold + 1)
                + 1009 * (inner_fold + 1),
                resume=not cli.no_resume,
            )
        inner = aggregate_inner_stack_payloads(
            old_inner_payload, inner_results, id_map
        )

        outer_indices = payload_indices(old_outer_payload, id_map)
        source_outer_stack = discover_v919_stack(fold_dir, outer_indices)
        outer_payload = train_stack(
            args,
            dataset,
            source_outer_stack,
            outer_indices,
            output
            / f"outer_fold_{outer_fold}"
            / "outer_deployment_stack",
            config,
            cli.seed + 500009 * (outer_fold + 1),
            resume=not cli.no_resume,
        )
        outer_actions, outer_labels, outer_app = payload_arrays(outer_payload)
        if list(outer_payload["sample_ids"]) != list(old_outer.sample_ids):
            raise RuntimeError("V9.30 and V9.19 outer target order differ")

        inner_consensus = inner_crossfit_fixed_convex(
            inner["actions"],
            inner["labels"],
            inner["fold_index"],
            config.shrinkage_lambda,
        )
        final_weights = inner_consensus["final_weights"]
        primary = outer_actions @ final_weights
        equal_mean = outer_actions.mean(axis=1)
        attribute = attribute_weighted_prediction(
            outer_actions, outer_app, config.attribute_anchor_prior
        )
        cheating_weights = fit_fixed_convex(
            outer_actions, outer_labels, config.shrinkage_lambda
        )
        cheating_prediction = outer_actions @ cheating_weights
        errors = np.abs(outer_actions - outer_labels[:, None])
        oracle_index = errors.argmin(axis=1)
        sample_oracle = outer_actions[
            np.arange(len(outer_actions)), oracle_index
        ]

        predictions = {
            "anchor": outer_actions[:, 0],
            **{
                name: outer_actions[:, index]
                for index, name in enumerate(EXPERT_NAMES, start=1)
            },
            "equal_mean": equal_mean,
            "attribute_weighted": attribute["prediction"],
            "fixed_convex_shrinkage": primary,
            "old_v921_convex_shrinkage": old_v921,
            "fold_label_cheating_convex": cheating_prediction,
            "sample_oracle": sample_oracle,
        }
        for strategy, prediction in predictions.items():
            metrics = prediction_metrics(prediction, outer_labels, old_v921)
            fold_rows.append(
                {
                    "outer_fold": outer_fold,
                    "strategy": strategy,
                    "deployable": strategy
                    not in {
                        "fold_label_cheating_convex",
                        "sample_oracle",
                    },
                    "label_cheating": strategy
                    in {
                        "fold_label_cheating_convex",
                        "sample_oracle",
                    },
                    **metrics,
                }
            )
        for action, weight in zip(ACTION_NAMES_V930, final_weights):
            weight_rows.append(
                {
                    "outer_fold": outer_fold,
                    "weight_type": "inner_oof_fixed_convex",
                    "action": action,
                    "weight": float(weight),
                }
            )
        for action, weight in zip(ACTION_NAMES_V930, cheating_weights):
            weight_rows.append(
                {
                    "outer_fold": outer_fold,
                    "weight_type": "outer_label_cheating_convex",
                    "action": action,
                    "weight": float(weight),
                }
            )
        for row in inner_consensus["fold_rows"]:
            for action, weight in zip(ACTION_NAMES_V930, row["weights"]):
                weight_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "inner_fold": row["inner_fold"],
                        "weight_type": "inner_crossfit_development",
                        "action": action,
                        "weight": float(weight),
                    }
                )

        corr = residual_correlation(outer_actions, outer_labels)
        for left, left_name in enumerate(ACTION_NAMES_V930):
            for right, right_name in enumerate(ACTION_NAMES_V930):
                correlation_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "left_action": left_name,
                        "right_action": right_name,
                        "residual_correlation": float(corr[left, right]),
                    }
                )
        for index, expert_name in enumerate(EXPERT_NAMES):
            applicability_rows.append(
                {
                    "outer_fold": outer_fold,
                    "expert": expert_name,
                    "mean": float(outer_app[:, index].mean()),
                    "std": float(outer_app[:, index].std()),
                    "p10": float(np.quantile(outer_app[:, index], 0.10)),
                    "p50": float(np.quantile(outer_app[:, index], 0.50)),
                    "p90": float(np.quantile(outer_app[:, index], 0.90)),
                }
            )

        frame = pd.DataFrame(
            {
                "outer_fold": outer_fold,
                "sample_id": outer_payload["sample_ids"],
                "group_id": outer_payload["group_ids"],
                "label": outer_labels,
                "sample_oracle_action": [
                    ACTION_NAMES_V930[index] for index in oracle_index
                ],
            }
        )
        for index, action in enumerate(ACTION_NAMES_V930):
            frame[f"action_prediction_{action}"] = outer_actions[:, index]
        for index, expert_name in enumerate(EXPERT_NAMES):
            frame[f"applicability_{expert_name}"] = outer_app[:, index]
            frame[f"attribute_weight_{expert_name}"] = attribute["weights"][
                :, index + 1
            ]
        frame["attribute_weight_anchor"] = attribute["weights"][:, 0]
        for strategy, prediction in predictions.items():
            frame[f"prediction_{strategy}"] = prediction
        prediction_frames.append(frame)

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["outer_fold", "mae", "strategy"]
    )
    weights = pd.DataFrame(weight_rows)
    predictions = pd.concat(prediction_frames, ignore_index=True)
    if predictions["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("outer prediction aggregation duplicates sample IDs")
    labels = predictions["label"].to_numpy(dtype=np.float64)
    old_v921 = predictions[
        "prediction_old_v921_convex_shrinkage"
    ].to_numpy(dtype=np.float64)
    strategies = sorted(
        column.removeprefix("prediction_")
        for column in predictions.columns
        if column.startswith("prediction_")
    )
    aggregate_rows = []
    bootstrap_rows = []
    for strategy in strategies:
        prediction = predictions[f"prediction_{strategy}"].to_numpy(
            dtype=np.float64
        )
        local_fold = fold_metrics[fold_metrics["strategy"] == strategy]
        metrics = prediction_metrics(prediction, labels, old_v921)
        aggregate_rows.append(
            {
                "strategy": strategy,
                "deployable": bool(local_fold["deployable"].all()),
                "label_cheating": bool(local_fold["label_cheating"].any()),
                **metrics,
            }
        )
        gain = np.abs(old_v921 - labels) - np.abs(prediction - labels)
        interval = group_bootstrap_gain_interval(
            gain,
            predictions["group_id"].tolist(),
            repetitions=2000,
            seed=cli.seed + 7919 * (len(bootstrap_rows) + 1),
        )
        bootstrap_rows.append({"strategy": strategy, **interval})
    aggregate = pd.DataFrame(aggregate_rows).sort_values("mae")
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values("strategy")
    primary_mae = float(
        aggregate.loc[
            aggregate["strategy"] == "fixed_convex_shrinkage", "mae"
        ].iloc[0]
    )
    v921_mae = float(
        aggregate.loc[
            aggregate["strategy"] == "old_v921_convex_shrinkage", "mae"
        ].iloc[0]
    )
    fold_pivot = fold_metrics.pivot(
        index="outer_fold", columns="strategy", values="mae"
    )
    fold_gain = (
        fold_pivot["old_v921_convex_shrinkage"]
        - fold_pivot["fixed_convex_shrinkage"]
    )
    gate = success_gate(
        fold_gain.to_numpy(), v921_mae - primary_mae, config
    )

    summary = {
        "version": AUDIT_VERSION,
        "method": "observable_attributes_before_expert_definition",
        "dataset": cli.dataset,
        "root": str(root),
        "output_dir": str(output),
        "config": asdict(config),
        "sample_count": int(len(predictions)),
        "outer_fold_count": int(cli.outer_folds),
        "action_names": list(ACTION_NAMES_V930),
        "expert_modes": {
            "text_stable": "L",
            "audio_informative": "LA",
            "vision_informative": "LV",
            "cross_modal_conflict": "LAV",
        },
        "primary_strategy": "fixed_convex_shrinkage",
        "primary_mae": primary_mae,
        "old_v921_mae": v921_mae,
        "gain_vs_old_v921": v921_mae - primary_mae,
        "success_gate": gate,
        "provenance": {
            "attributes_computed_from_mode_predictions_only": True,
            "attributes_use_true_labels": False,
            "expert_training_uses_supervised_labels": True,
            "outer_labels_used_for_training_or_calibration": False,
            "primary_weights_fit_on_inner_oof_only": True,
            "winner_router_present": False,
            "oracle_winner_supervision_present": False,
            "knn_confidence_or_stability_gate_present": False,
            "old_v919_pools_used_only_for_splits_checkpoints_and_baseline": True,
        },
    }

    fold_metrics.to_csv(output / "v930_metrics_by_fold.csv", index=False)
    aggregate.to_csv(output / "v930_aggregate_metrics.csv", index=False)
    weights.to_csv(output / "v930_weight_inventory.csv", index=False)
    predictions.to_csv(output / "v930_outer_predictions.csv", index=False)
    pd.DataFrame(source_rows).to_csv(
        output / "v930_source_manifest.csv", index=False
    )
    pd.DataFrame(correlation_rows).to_csv(
        output / "v930_residual_correlations.csv", index=False
    )
    pd.DataFrame(applicability_rows).to_csv(
        output / "v930_applicability_summary.csv", index=False
    )
    bootstrap.to_csv(
        output / "v930_group_bootstrap_gain_ci.csv", index=False
    )
    (output / "v930_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = [
        "# V9.30 Observable-Attribute Experts",
        "",
        "Expert roles and applicability use only deterministic LAV/L/LA/LV predictions available at deployment. No winner router is trained.",
        "",
        "## Aggregate metrics",
        "",
        markdown_table(
            aggregate,
            [
                "strategy",
                "deployable",
                "label_cheating",
                "mae",
                "gain_vs_baseline",
                "win_rate_vs_baseline",
                "large_harm_rate_010",
            ],
        ),
        "",
        "## Primary by outer fold",
        "",
        markdown_table(
            fold_metrics[
                fold_metrics["strategy"].isin(
                    [
                        "fixed_convex_shrinkage",
                        "old_v921_convex_shrinkage",
                        "attribute_weighted",
                        "anchor",
                        "fold_label_cheating_convex",
                        "sample_oracle",
                    ]
                )
            ],
            ["outer_fold", "strategy", "mae", "gain_vs_baseline"],
        ),
        "",
        "## Decision",
        "",
        f"- Old V9.21 MAE: `{v921_mae:.6f}`",
        f"- V9.30 fixed convex MAE: `{primary_mae:.6f}`",
        f"- Gain versus V9.21: `{v921_mae - primary_mae:+.6f}`",
        f"- Pre-registered gate passed: `{gate['passed']}`",
        "",
        "If the gate fails, do not add a learned winner router. Inspect expert diversity and move to MOSEI only if the observable attributes create stable fixed-fusion capacity.",
    ]
    (output / "v930_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.30 OBSERVABLE-ATTRIBUTE EXPERTS COMPLETE")
    print("attributes use labels: False")
    print("winner router trained: False")
    print(
        "old V9.21 / V9.30 primary MAE:",
        f"{v921_mae:.6f}",
        f"{primary_mae:.6f}",
    )
    print("gain vs old V9.21:", f"{v921_mae - primary_mae:+.6f}")
    print("success gate passed:", gate["passed"])
    print("report:", output / "v930_report.md")


if __name__ == "__main__":
    main()
