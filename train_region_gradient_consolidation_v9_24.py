"""Train one outer fold of V9.24 region-gradient consolidation.

All three strategies start from the same frozen V9.19 CFCompat deployment
anchor. Only the V9.23 ``fusion_tail`` parameter scope is trainable.
Checkpoint selection uses the fold's inner-valid partition only. The outer
holdout is evaluated exactly once per frozen selected checkpoint.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path

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
from trains.singleTask.oof_group_splits_v92 import (
    build_subset_loader,
    canonical_sample_id,
    conversation_group_id,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.region_gradient_consolidation_v924 import (
    PRIMARY_STRATEGY,
    STRATEGIES,
    TRAINING_VERSION,
    RegionGradientTrainingConfigV924,
    evaluate_model,
    regression_metrics,
    train_one_strategy,
)
from trains.singleTask.same_stack_expert_factory_v919 import sha256
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train one strict V9.24 outer fold."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--minimum-mgda-direction-norm", type=float, default=0.02
    )
    parser.add_argument("--output-root", default="")
    parser.add_argument("--resume", action="store_true")
    cli = parser.parse_args()
    if not 0 <= cli.outer_fold < cli.outer_folds:
        parser.error("--outer-fold must be in [0, outer-folds)")
    return cli


def _build_args(cli):
    args = get_config_regression(
        "DLF", cli.dataset, Path(cli.config)
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


def _load_split_indices(fold_dir: Path):
    path = fold_dir / "outer_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(path)
    manifest = pd.read_csv(path)
    required = {
        "sample_index",
        "sample_id",
        "group_id",
        "partition",
        "label",
    }
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(
            f"outer manifest missing columns: {sorted(missing)}"
        )
    observed = set(manifest["partition"].astype(str))
    expected = {"inner_train", "inner_valid", "outer_holdout"}
    if observed != expected:
        raise RuntimeError(
            f"unexpected manifest partitions: {sorted(observed)}"
        )

    result = {}
    for partition in sorted(expected):
        local = manifest[
            manifest["partition"].astype(str) == partition
        ].copy()
        indices = local["sample_index"].astype(int).tolist()
        if len(indices) != len(set(indices)) or not indices:
            raise RuntimeError(f"invalid {partition} indices")
        result[partition] = sorted(indices)
    if set(result["inner_train"]) & set(result["inner_valid"]):
        raise RuntimeError("inner train/valid overlap")
    development = set(result["inner_train"]) | set(
        result["inner_valid"]
    )
    if development & set(result["outer_holdout"]):
        raise RuntimeError("outer holdout entered development split")
    return result, manifest


def _make_model(args, checkpoint_path: Path):
    model = MissingModalityWrapper(
        DLF(args).to(args.device),
        int(args.feature_dims[1]),
        int(args.feature_dims[2]),
    ).to(args.device)
    model.load_state_dict(
        _load_state(checkpoint_path, args.device), strict=True
    )
    model.eval()
    return model


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


def _metadata(batch):
    ids = [canonical_sample_id(value) for value in list(batch["id"])]
    return {
        "sample_id": ids,
        "group_id": [conversation_group_id(value) for value in ids],
        "sample_index": [
            int(value) for value in batch["index"].view(-1).tolist()
        ],
    }


def _metric_scalars(metrics):
    result = {
        key: value
        for key, value in metrics.items()
        if key != "region_mae"
    }
    for name, value in metrics["region_mae"].items():
        result[f"region_mae_{name}"] = float(value)
    return result


def _prediction_frame(evaluation, column_name: str) -> pd.DataFrame:
    metadata = evaluation["metadata"]
    return pd.DataFrame(
        {
            "sample_index": metadata["sample_index"],
            "sample_id": metadata["sample_id"],
            "group_id": metadata["group_id"],
            "label": evaluation["labels"].numpy(),
            "true_region": evaluation["regions"],
            column_name: evaluation["prediction"].numpy(),
        }
    )


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    setup_seed(cli.seed)
    args = _build_args(cli)
    dataset = MMDataset(args, mode="train")
    if "seq_lens" in args:
        args["seq_lens"] = dataset.get_seq_len()

    root = Path(cli.v919_root)
    fold_dir = root / f"outer_fold_{cli.outer_fold}"
    source_checkpoint = (
        fold_dir
        / "outer_deployment_stack"
        / "cfcompat_student_best_inner_valid.pth"
    )
    if not source_checkpoint.is_file():
        raise FileNotFoundError(source_checkpoint)
    split_indices, manifest = _load_split_indices(fold_dir)

    output_root = (
        Path(cli.output_root)
        if cli.output_root
        else root / "v924_region_gradient_consolidation"
    )
    output_dir = output_root / f"outer_fold_{cli.outer_fold}"
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "v924_fold_summary.json"
    if cli.resume and summary_path.is_file():
        print("V9.24 OUTER FOLD ALREADY COMPLETE")
        print("outer_fold:", cli.outer_fold)
        print("summary:", summary_path)
        return

    config = RegionGradientTrainingConfigV924(
        parameter_scope="fusion_tail",
        learning_rate=float(cli.learning_rate),
        max_epochs=int(cli.max_epochs),
        early_stop=int(cli.early_stop),
        gradient_clip_norm=float(cli.gradient_clip_norm),
        minimum_mgda_direction_norm=float(
            cli.minimum_mgda_direction_norm
        ),
    )
    config.validate()
    forward_batch = _forward_batch(args.device)

    loaders = {
        name: build_subset_loader(
            dataset,
            split_indices[name],
            int(cli.batch_size),
            int(cli.num_workers),
            False,
            int(cli.seed) + 190001 * (cli.outer_fold + 1),
        )
        for name in (
            "inner_train",
            "inner_valid",
            "outer_holdout",
        )
    }

    source_hash_before = sha256(source_checkpoint)
    anchor_model = _make_model(args, source_checkpoint)
    anchor_valid = evaluate_model(
        anchor_model,
        loaders["inner_valid"],
        forward_batch,
        sample_metadata=_metadata,
    )
    anchor_outer = evaluate_model(
        anchor_model,
        loaders["outer_holdout"],
        forward_batch,
        sample_metadata=_metadata,
    )
    anchor_valid_metrics = regression_metrics(
        anchor_valid["prediction"], anchor_valid["labels"]
    )
    anchor_outer_metrics = regression_metrics(
        anchor_outer["prediction"], anchor_outer["labels"]
    )
    valid_frame = _prediction_frame(
        anchor_valid, "prediction_anchor"
    )
    outer_frame = _prediction_frame(
        anchor_outer, "prediction_anchor"
    )
    del anchor_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    strategy_summaries = {}
    history_frames = []
    weight_frames = []
    initial_fingerprints = []

    for strategy_index, strategy in enumerate(STRATEGIES):
        setup_seed(
            int(cli.seed)
            + 210011 * (cli.outer_fold + 1)
            + 101 * (strategy_index + 1)
        )
        model = _make_model(args, source_checkpoint)
        trained = train_one_strategy(
            model,
            strategy,
            loaders["inner_train"],
            loaders["inner_valid"],
            forward_batch,
            config,
        )
        initial_fingerprints.append(
            trained["initial_parameter_fingerprint"]
        )
        best_state = trained.pop("best_state_dict")
        valid_eval = evaluate_model(
            model,
            loaders["inner_valid"],
            forward_batch,
            sample_metadata=_metadata,
        )
        outer_eval = evaluate_model(
            model,
            loaders["outer_holdout"],
            forward_batch,
            sample_metadata=_metadata,
        )
        valid_metrics = regression_metrics(
            valid_eval["prediction"],
            valid_eval["labels"],
            anchor=anchor_valid["prediction"],
        )
        outer_metrics = regression_metrics(
            outer_eval["prediction"],
            outer_eval["labels"],
            anchor=anchor_outer["prediction"],
        )

        checkpoint_path = (
            output_dir / f"{strategy}_best_inner_valid_v924.pth"
        )
        torch.save(
            {
                "version": TRAINING_VERSION,
                "strategy": strategy,
                "outer_fold": int(cli.outer_fold),
                "config": asdict(config),
                "source_checkpoint": str(source_checkpoint),
                "source_checkpoint_sha256": source_hash_before,
                "best_epoch": int(trained["best_epoch"]),
                "best_validation_objective": float(
                    trained["best_validation_objective"]
                ),
                "selected_parameter_names": trained[
                    "selected_parameter_names"
                ],
                "selected_parameter_count": int(
                    trained["selected_parameter_count"]
                ),
                "model_state_dict": best_state,
                "provenance": {
                    "initialized_from_same_v919_cfcompat_anchor": True,
                    "fusion_tail_only_trainable": True,
                    "plain_sgd_without_momentum_or_weight_decay": True,
                    "inner_train_only_for_updates": True,
                    "inner_valid_only_for_checkpoint_selection": True,
                    "outer_holdout_not_used_for_training_or_selection": True,
                    "router_or_fusion_head_used": False,
                    "expert_predictions_used": False,
                    "official_validation_or_test_used": False,
                },
            },
            checkpoint_path,
        )

        history = pd.DataFrame(trained["history"])
        history.insert(0, "outer_fold", int(cli.outer_fold))
        history_frames.append(history)
        weights = pd.DataFrame(trained["weight_history"])
        if not weights.empty:
            weights.insert(0, "outer_fold", int(cli.outer_fold))
            weight_frames.append(weights)

        valid_local = _prediction_frame(
            valid_eval, f"prediction_{strategy}"
        )
        outer_local = _prediction_frame(
            outer_eval, f"prediction_{strategy}"
        )
        valid_frame = valid_frame.merge(
            valid_local[
                [
                    "sample_index",
                    "sample_id",
                    f"prediction_{strategy}",
                ]
            ],
            on=["sample_index", "sample_id"],
            validate="one_to_one",
        )
        outer_frame = outer_frame.merge(
            outer_local[
                [
                    "sample_index",
                    "sample_id",
                    f"prediction_{strategy}",
                ]
            ],
            on=["sample_index", "sample_id"],
            validate="one_to_one",
        )

        compact = {
            key: value
            for key, value in trained.items()
            if key not in {"history", "weight_history"}
        }
        compact.update(
            {
                "checkpoint": str(checkpoint_path),
                "validation_metrics": _metric_scalars(valid_metrics),
                "outer_holdout_metrics": _metric_scalars(
                    outer_metrics
                ),
            }
        )
        strategy_summaries[strategy] = compact
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if len(set(initial_fingerprints)) != 1:
        raise RuntimeError(
            "strategies did not start from identical parameters"
        )
    source_hash_after = sha256(source_checkpoint)
    if source_hash_before != source_hash_after:
        raise RuntimeError(
            "V9.19 source checkpoint changed during V9.24"
        )

    histories = pd.concat(history_frames, ignore_index=True)
    histories.to_csv(
        output_dir / "v924_training_history.csv", index=False
    )
    if weight_frames:
        pd.concat(weight_frames, ignore_index=True).to_csv(
            output_dir / "v924_region_weights_by_epoch.csv",
            index=False,
        )
    else:
        pd.DataFrame(
            columns=[
                "outer_fold",
                "epoch",
                "strategy",
                "region",
                "weight",
                "direction_cosine",
                "direction_dot",
            ]
        ).to_csv(
            output_dir / "v924_region_weights_by_epoch.csv",
            index=False,
        )
    valid_frame.to_csv(
        output_dir / "v924_inner_valid_predictions.csv",
        index=False,
    )
    outer_frame.to_csv(
        output_dir / "v924_outer_holdout_predictions.csv",
        index=False,
    )

    split_counts = {
        name: len(indices)
        for name, indices in split_indices.items()
    }
    group_counts = {
        name: int(
            manifest[
                manifest["partition"].astype(str) == name
            ]["group_id"].astype(str).nunique()
        )
        for name in split_counts
    }
    summary = {
        "version": TRAINING_VERSION,
        "method": (
            "single_model_fusion_tail_region_gradient_consolidation"
        ),
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "outer_fold": int(cli.outer_fold),
        "outer_folds": int(cli.outer_folds),
        "primary_strategy": PRIMARY_STRATEGY,
        "strategies": list(STRATEGIES),
        "config": asdict(config),
        "split_counts": split_counts,
        "group_counts": group_counts,
        "source_checkpoint": str(source_checkpoint),
        "source_checkpoint_sha256_before": source_hash_before,
        "source_checkpoint_sha256_after": source_hash_after,
        "initial_selected_parameter_fingerprint": (
            initial_fingerprints[0]
        ),
        "anchor_validation_metrics": _metric_scalars(
            anchor_valid_metrics
        ),
        "anchor_outer_holdout_metrics": _metric_scalars(
            anchor_outer_metrics
        ),
        "strategy_summaries": strategy_summaries,
        "provenance": {
            "one_model_one_scalar_output": True,
            "same_initial_anchor_for_all_strategies": True,
            "fusion_tail_only_trainable": True,
            "full_batch_exact_mean_gradients_per_epoch": True,
            "plain_sgd_preserves_analyzed_direction": True,
            "inner_train_only_for_parameter_updates": True,
            "inner_valid_only_for_checkpoint_selection": True,
            "outer_holdout_evaluated_once_after_selection": True,
            "outer_holdout_not_used_for_hyperparameters": True,
            "router_used": False,
            "consensus_or_residual_head_added": False,
            "expert_predictions_or_distillation_used": False,
            "official_validation_used": False,
            "official_test_used": False,
        },
        "outputs": {
            "history": str(
                output_dir / "v924_training_history.csv"
            ),
            "weights": str(
                output_dir / "v924_region_weights_by_epoch.csv"
            ),
            "inner_valid_predictions": str(
                output_dir / "v924_inner_valid_predictions.csv"
            ),
            "outer_holdout_predictions": str(
                output_dir / "v924_outer_holdout_predictions.csv"
            ),
        },
    }
    summary_path.write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("V9.24 OUTER FOLD COMPLETE")
    print("outer_fold:", cli.outer_fold)
    print("models trained: True")
    print("trainable scope: fusion_tail")
    for strategy in STRATEGIES:
        metrics = strategy_summaries[strategy][
            "outer_holdout_metrics"
        ]
        print(
            strategy,
            "best_epoch=",
            strategy_summaries[strategy]["best_epoch"],
            "outer_mae=",
            f"{float(metrics['mae']):.6f}",
            "gain_vs_anchor=",
            f"{float(metrics['gain_vs_anchor']):+.6f}",
        )
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
