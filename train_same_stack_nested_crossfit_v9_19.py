"""Run one outer fold of the fully same-stack V9.19 nested cross-fit protocol."""

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
from trains.singleTask.frontier_expert_pool_v97 import (
    RoleCrossfitConfigV97,
    TailCrossfitConfigV97,
)
from trains.singleTask.oof_group_splits_v92 import (
    StageLimits,
    build_nested_group_folds,
    canonical_sample_id,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable, sample_rows
from trains.singleTask.region_probability_model_training_v918 import (
    RegionModelConfigV918,
)
from trains.singleTask.same_stack_expert_factory_v919 import (
    STACK_VERSION,
    SameStackConfigV919,
    train_and_collect_same_stack,
)
from trains.singleTask.same_stack_nested_router_v919 import (
    PROTOCOL_VERSION,
    FixedPolicyV919,
    InnerGateV919,
    crossfit_router_diagnostic,
    evaluate_outer_pool,
    merge_pools,
    train_calibrated_outer_router,
)
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "One outer fold of fully same-stack nested expert routing."
        )
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--save-root",
        default="result/full_same_stack_nested_crossfit_v919",
    )
    parser.add_argument("--outer-fold", type=int, required=True)
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-valid-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--clean-max-epochs", type=int, default=40)
    parser.add_argument("--moddrop-max-epochs", type=int, default=30)
    parser.add_argument("--cfcompat-max-epochs", type=int, default=30)
    parser.add_argument("--base-early-stop", type=int, default=7)
    parser.add_argument("--role-head-epochs", type=int, default=4)
    parser.add_argument("--role-tail-epochs", type=int, default=12)
    parser.add_argument("--tail-epochs", type=int, default=12)
    parser.add_argument("--router-hidden-dim", type=int, default=64)
    parser.add_argument("--router-max-epochs", type=int, default=80)
    parser.add_argument("--router-early-stop", type=int, default=10)
    parser.add_argument("--gain-margin", type=float, default=0.03)
    parser.add_argument(
        "--min-region-confidence", type=float, default=0.55
    )
    parser.add_argument("--max-coverage", type=float, default=0.25)
    parser.add_argument("--min-inner-gain", type=float, default=0.005)
    parser.add_argument("--max-inner-harm", type=float, default=0.05)
    parser.add_argument(
        "--min-positive-fold-fraction",
        type=float,
        default=2.0 / 3.0,
    )
    parser.add_argument(
        "--min-trigger-precision", type=float, default=0.55
    )
    parser.add_argument("--no-resume", action="store_true")
    cli = parser.parse_args()
    if not 0 <= cli.outer_fold < cli.outer_folds:
        parser.error("--outer-fold must be in [0, outer-folds)")
    if cli.inner_folds < 3:
        parser.error("--inner-folds must be at least 3")
    return cli


def local_to_global(local_indices, parent_indices):
    return [int(parent_indices[int(index)]) for index in local_indices]


def write_result_rows(path: Path, split: str, result):
    pd.DataFrame(
        sample_rows(split, result["pool"], result)
    ).to_csv(path, index=False)


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
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
    if "seq_lens" in args:
        args["seq_lens"] = dataset.get_seq_len()
    sample_ids = [
        canonical_sample_id(value) for value in list(dataset.ids)
    ]
    labels = torch.tensor(
        dataset.labels["M"], dtype=torch.float32
    ).view(-1)
    outer_specs, outer_manifest = build_nested_group_folds(
        sample_ids,
        labels.tolist(),
        outer_folds=cli.outer_folds,
        inner_valid_fraction=cli.inner_valid_fraction,
        seed=cli.seed,
    )
    outer_spec = outer_specs[cli.outer_fold]
    root = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    fold_dir = root / f"outer_fold_{cli.outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    outer_manifest[
        outer_manifest["outer_fold"] == cli.outer_fold
    ].to_csv(fold_dir / "outer_manifest.csv", index=False)

    stage_limits = StageLimits(
        clean_max_epochs=cli.clean_max_epochs,
        moddrop_max_epochs=cli.moddrop_max_epochs,
        cfcompat_max_epochs=cli.cfcompat_max_epochs,
        early_stop=cli.base_early_stop,
    )
    role_config = RoleCrossfitConfigV97(
        head_epochs=cli.role_head_epochs,
        tail_epochs=cli.role_tail_epochs,
        batch_size=cli.batch_size,
    )
    tail_config = TailCrossfitConfigV97(
        epochs=cli.tail_epochs,
        batch_size=max(cli.batch_size, 64),
    )
    stack_config = SameStackConfigV919(
        feature_batch_size=cli.batch_size,
        num_workers=cli.num_workers,
    )
    router_config = RegionModelConfigV918(
        hidden_dim=cli.router_hidden_dim,
        max_epochs=cli.router_max_epochs,
        early_stop=cli.router_early_stop,
    )
    policy = FixedPolicyV919(
        gain_margin=cli.gain_margin,
        min_region_confidence=cli.min_region_confidence,
        max_coverage=cli.max_coverage,
    )
    gate = InnerGateV919(
        min_gain=cli.min_inner_gain,
        max_harm_over_010_rate=cli.max_inner_harm,
        min_positive_fold_fraction=cli.min_positive_fold_fraction,
        min_trigger_precision=cli.min_trigger_precision,
    )

    development_indices = sorted(
        [*outer_spec.inner_train_indices, *outer_spec.inner_valid_indices]
    )
    development_ids = [
        sample_ids[index] for index in development_indices
    ]
    development_labels = [
        float(labels[index].item()) for index in development_indices
    ]
    inner_specs, inner_manifest = build_nested_group_folds(
        development_ids,
        development_labels,
        outer_folds=cli.inner_folds,
        inner_valid_fraction=cli.inner_valid_fraction,
        seed=cli.seed + 50021 * (cli.outer_fold + 1),
    )
    inner_manifest.to_csv(
        fold_dir / "inner_manifest_local.csv", index=False
    )

    inner_pools = []
    for inner_spec in inner_specs:
        inner_dir = fold_dir / f"inner_stack_{inner_spec.outer_fold}"
        pool = train_and_collect_same_stack(
            args,
            dataset,
            local_to_global(
                inner_spec.inner_train_indices, development_indices
            ),
            local_to_global(
                inner_spec.inner_valid_indices, development_indices
            ),
            local_to_global(
                inner_spec.outer_holdout_indices, development_indices
            ),
            inner_dir,
            stage_limits,
            role_config,
            tail_config,
            stack_config,
            seed=(
                cli.seed
                + 70001 * (cli.outer_fold + 1)
                + 1009 * (inner_spec.outer_fold + 1)
            ),
            resume=not cli.no_resume,
        )
        pool["inner_fold"] = int(inner_spec.outer_fold)
        torch.save(
            pool, inner_dir / "same_stack_target_pool_v919.pth"
        )
        inner_pools.append(pool)

    merged = merge_pools(inner_pools, development_indices)
    merged["fold_index"] = torch.full(
        (len(development_indices),), -1, dtype=torch.long
    )
    position = {
        index: offset
        for offset, index in enumerate(development_indices)
    }
    for inner_spec in inner_specs:
        for local_index in inner_spec.outer_holdout_indices:
            global_index = development_indices[int(local_index)]
            merged["fold_index"][position[global_index]] = int(
                inner_spec.outer_fold
            )
    if bool((merged["fold_index"] < 0).any()):
        raise RuntimeError("inner OOF fold assignment incomplete")
    torch.save(
        merged, fold_dir / "inner_oof_same_stack_pool_v919.pth"
    )

    diagnostic = crossfit_router_diagnostic(
        merged,
        args.device,
        router_config,
        policy,
        gate,
        seed=cli.seed + 90001 * (cli.outer_fold + 1),
    )
    pd.DataFrame(diagnostic["fold_rows"]).to_csv(
        fold_dir / "inner_router_fold_metrics.csv", index=False
    )
    pd.DataFrame(diagnostic["history_rows"]).to_csv(
        fold_dir / "inner_router_training_history.csv", index=False
    )
    inner_result = {
        "pool": diagnostic["pool"],
        "region_probabilities": diagnostic["probabilities"],
        "expected_costs": diagnostic["expected_costs"],
        **diagnostic["overall"],
    }
    write_result_rows(
        fold_dir / "inner_oof_router_predictions.csv",
        "inner_oof",
        inner_result,
    )

    outer_pool = train_and_collect_same_stack(
        args,
        dataset,
        outer_spec.inner_train_indices,
        outer_spec.inner_valid_indices,
        outer_spec.outer_holdout_indices,
        fold_dir / "outer_deployment_stack",
        stage_limits,
        role_config,
        tail_config,
        stack_config,
        seed=cli.seed + 110003 * (cli.outer_fold + 1),
        resume=not cli.no_resume,
    )
    trained_router = train_calibrated_outer_router(
        merged,
        args.device,
        router_config,
        policy,
        seed=cli.seed + 130003 * (cli.outer_fold + 1),
        epochs=diagnostic["median_best_epoch"],
    )
    outer_result = evaluate_outer_pool(
        trained_router,
        outer_pool,
        args.device,
        router_config,
        policy,
        accepted=diagnostic["accepted"],
    )
    write_result_rows(
        fold_dir / "outer_holdout_predictions.csv",
        "outer_holdout",
        outer_result,
    )
    torch.save(
        {
            "version": PROTOCOL_VERSION,
            "model_state_dict": {
                key: value.detach().cpu()
                for key, value in trained_router[
                    "model"
                ].state_dict().items()
            },
            "model_config": asdict(router_config),
            "temperature": trained_router["temperature"],
            "cost_matrix": trained_router["cost_matrix"],
            "cost_region_counts": trained_router[
                "cost_region_counts"
            ],
            "epochs": diagnostic["median_best_epoch"],
        },
        fold_dir / "outer_router_v919.pth",
    )

    summary = {
        "version": PROTOCOL_VERSION,
        "stack_version": STACK_VERSION,
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "outer_fold": int(cli.outer_fold),
        "outer_folds": int(cli.outer_folds),
        "inner_folds": int(cli.inner_folds),
        "development_count": len(development_indices),
        "outer_holdout_count": len(
            outer_spec.outer_holdout_indices
        ),
        "development_group_count": len(
            set(
                [
                    *outer_spec.inner_train_groups,
                    *outer_spec.inner_valid_groups,
                ]
            )
        ),
        "outer_holdout_group_count": len(
            outer_spec.outer_holdout_groups
        ),
        "fixed_policy": asdict(policy),
        "inner_gate": asdict(gate),
        "inner_router_accepted": bool(diagnostic["accepted"]),
        "inner_positive_fold_fraction": diagnostic[
            "positive_fold_fraction"
        ],
        "inner_oof_metrics": {
            key: value
            for key, value in diagnostic["overall"].items()
            if key
            not in {
                "selected_action",
                "selected_prediction",
                "predicted_gain",
                "region_confidence",
                "trigger",
            }
        },
        "outer_holdout_metrics": {
            key: value
            for key, value in outer_result.items()
            if key
            not in {
                "pool",
                "logits",
                "region_probabilities",
                "expected_costs",
                "selected_action",
                "selected_prediction",
                "predicted_gain",
                "region_confidence",
                "trigger",
            }
        },
        "router_calibration": {
            "temperature": trained_router["temperature"],
            "training_count": len(
                trained_router["training_indices"]
            ),
            "calibration_count": len(
                trained_router["calibration_indices"]
            ),
            "cost_region_counts": trained_router[
                "cost_region_counts"
            ].tolist(),
        },
        "configs": {
            "stage_limits": asdict(stage_limits),
            "role": asdict(role_config),
            "tail": asdict(tail_config),
            "stack": asdict(stack_config),
            "router": asdict(router_config),
        },
        "provenance": {
            "same_stack_recipe_used_for_inner_and_outer": True,
            "every_inner_oof_row_unseen_by_its_complete_expert_stack": True,
            "outer_holdout_unseen_by_anchor_experts_router_and_policy": True,
            "policy_hyperparameters_pre_registered": True,
            "no_policy_grid_search": True,
            "official_validation_used": False,
            "official_test_used": False,
            "outer_fold_result_not_used_to_retrain_that_fold": True,
        },
        "outputs": {
            "inner_pool": str(
                fold_dir / "inner_oof_same_stack_pool_v919.pth"
            ),
            "inner_fold_metrics": str(
                fold_dir / "inner_router_fold_metrics.csv"
            ),
            "inner_predictions": str(
                fold_dir / "inner_oof_router_predictions.csv"
            ),
            "outer_predictions": str(
                fold_dir / "outer_holdout_predictions.csv"
            ),
            "outer_router": str(fold_dir / "outer_router_v919.pth"),
        },
    }
    (
        fold_dir / "same_stack_nested_crossfit_v919_fold_summary.json"
    ).write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("V9.19 OUTER FOLD COMPLETE")
    print("outer_fold:", cli.outer_fold)
    print("inner_router_accepted:", diagnostic["accepted"])
    print(
        "inner_oof_gain:",
        f"{diagnostic['overall']['gain_vs_anchor']:+.6f}",
    )
    print(
        "outer_holdout_gain:",
        f"{outer_result['gain_vs_anchor']:+.6f}",
    )
    print("outer_holdout_mae:", f"{outer_result['mae']:.6f}")


if __name__ == "__main__":
    main()
