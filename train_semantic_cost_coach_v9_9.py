"""Build the V9.9 strict semantic pool and train the per-action cost coach."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.frontier_expert_pool_v97 import (
    RoleCrossfitConfigV97,
    TailCrossfitConfigV97,
)
from trains.singleTask.semantic_cost_coach_system_v99 import (
    SemanticCostCoachConfigV99,
    SemanticCostTrainerV99,
)
from trains.singleTask.semantic_expert_pool_v99 import (
    FrozenSemanticExpertPoolV99,
    anchor_checkpoint_from_summary,
    default_expert_paths,
)
from trains.singleTask.strict_semantic_expert_pool_v99 import (
    build_strict_semantic_expert_pool_v99,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="V9.9 semantic-signature per-action cost coach"
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v9-root",
        default="./result/role_conditioned_experts_v9/mosi/seed_1111",
    )
    parser.add_argument(
        "--v92-root",
        default="./result/oof_tail_residual_experts_v92/mosi/seed_1111",
    )
    parser.add_argument(
        "--save-root", default="./result/semantic_cost_coach_v99"
    )
    parser.add_argument("--top-oof-cache", default="")
    parser.add_argument("--semantic-pool", default="")
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--boundary-checkpoint", default="")
    parser.add_argument("--positive-checkpoint", default="")
    parser.add_argument("--strong-negative-checkpoint", default="")
    parser.add_argument("--strong-positive-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--role-hidden-dim", type=int, default=192)
    parser.add_argument("--role-dropout", type=float, default=0.15)
    parser.add_argument("--role-residual-max", type=float, default=0.45)
    parser.add_argument("--role-head-epochs", type=int, default=4)
    parser.add_argument("--role-tail-epochs", type=int, default=12)
    parser.add_argument("--role-head-lr", type=float, default=3e-4)
    parser.add_argument("--role-tail-lr", type=float, default=2e-5)
    parser.add_argument("--role-backbone-lr", type=float, default=5e-6)

    parser.add_argument("--tail-hidden-dim", type=int, default=96)
    parser.add_argument("--tail-dropout", type=float, default=0.15)
    parser.add_argument("--tail-residual-max", type=float, default=1.50)
    parser.add_argument("--tail-epochs", type=int, default=12)
    parser.add_argument("--tail-learning-rate", type=float, default=3e-4)

    parser.add_argument("--teacher-fit-steps", type=int, default=300)
    parser.add_argument("--min-region-samples", type=int, default=10)

    parser.add_argument("--coach-hidden-dim", type=int, default=128)
    parser.add_argument("--coach-action-embedding-dim", type=int, default=12)
    parser.add_argument("--coach-dropout", type=float, default=0.15)
    parser.add_argument("--coach-minimum-scale", type=float, default=0.01)
    parser.add_argument("--coach-folds", type=int, default=5)
    parser.add_argument("--coach-max-epochs", type=int, default=50)
    parser.add_argument("--coach-early-stop", type=int, default=8)
    parser.add_argument("--coach-learning-rate", type=float, default=3e-4)
    parser.add_argument("--rank-margin", type=float, default=0.02)
    parser.add_argument("--rank-temperature", type=float, default=0.05)
    parser.add_argument("--selection-temperature", type=float, default=0.08)

    parser.add_argument("--minimum-oof-gain", type=float, default=0.002)
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0015)
    parser.add_argument("--bootstrap-quantile", type=float, default=0.20)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--maximum-harm-rate", type=float, default=0.03)
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def build_loader(dataset, batch_size, num_workers):
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        num_workers=int(num_workers),
        shuffle=False,
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


def resolve_paths(cli):
    v9_root = Path(
        str(cli.v9_root).replace("seed_1111", f"seed_{cli.seed}")
    )
    v92_root = Path(
        str(cli.v92_root).replace("seed_1111", f"seed_{cli.seed}")
    )
    top_cache = (
        Path(cli.top_oof_cache)
        if cli.top_oof_cache
        else v92_root
        / "oof_cfcompat"
        / "nested_grouped_oof_cfcompat_cache_v92.pth"
    )
    if not top_cache.is_file():
        raise FileNotFoundError(top_cache)
    experts = default_expert_paths(v9_root, v92_root)
    overrides = {
        "boundary": cli.boundary_checkpoint,
        "positive": cli.positive_checkpoint,
        "strong_negative": cli.strong_negative_checkpoint,
        "strong_positive": cli.strong_positive_checkpoint,
    }
    for name, value in overrides.items():
        if value:
            experts[name] = Path(value)
    anchor = (
        Path(cli.anchor_checkpoint)
        if cli.anchor_checkpoint
        else anchor_checkpoint_from_summary(v92_root)
    )
    return v9_root, v92_root, top_cache, anchor, experts


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
    args["batch_size"] = int(cli.feature_batch_size)

    v9_root, v92_root, top_cache, anchor_checkpoint, expert_paths = (
        resolve_paths(cli)
    )
    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["train"].get_seq_len()

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    pool_dir = save_dir / "strict_semantic_expert_pool"
    semantic_pool_path = (
        Path(cli.semantic_pool)
        if cli.semantic_pool
        else pool_dir / "strict_semantic_expert_pool_v99.pth"
    )
    if cli.semantic_pool:
        if not semantic_pool_path.is_file():
            raise FileNotFoundError(semantic_pool_path)
    else:
        role_config = RoleCrossfitConfigV97(
            hidden_dim=cli.role_hidden_dim,
            dropout=cli.role_dropout,
            residual_max=cli.role_residual_max,
            head_epochs=cli.role_head_epochs,
            tail_epochs=cli.role_tail_epochs,
            head_lr=cli.role_head_lr,
            tail_lr=cli.role_tail_lr,
            backbone_lr=cli.role_backbone_lr,
            batch_size=cli.feature_batch_size,
        )
        tail_config = TailCrossfitConfigV97(
            hidden_dim=cli.tail_hidden_dim,
            dropout=cli.tail_dropout,
            residual_max=cli.tail_residual_max,
            epochs=cli.tail_epochs,
            learning_rate=cli.tail_learning_rate,
        )
        build_strict_semantic_expert_pool_v99(
            args=args,
            train_dataset=datasets["train"],
            top_oof_cache_path=top_cache,
            output_dir=pool_dir,
            role_config=role_config,
            tail_config=tail_config,
            num_workers=cli.num_workers,
            teacher_fit_steps=cli.teacher_fit_steps,
            min_region_samples=cli.min_region_samples,
            resume=not cli.no_resume,
        )

    frozen_pool = FrozenSemanticExpertPoolV99(
        args,
        anchor_checkpoint=anchor_checkpoint,
        expert_paths=expert_paths,
        role_residual_max=cli.role_residual_max,
        tail_residual_max=cli.tail_residual_max,
    )
    valid_pool = frozen_pool.collect(
        build_loader(
            datasets["valid"],
            cli.feature_batch_size,
            cli.num_workers,
        )
    )
    test_pool = frozen_pool.collect(
        build_loader(
            datasets["test"],
            cli.feature_batch_size,
            cli.num_workers,
        )
    )
    coach_config = SemanticCostCoachConfigV99(
        hidden_dim=cli.coach_hidden_dim,
        action_embedding_dim=cli.coach_action_embedding_dim,
        dropout=cli.coach_dropout,
        minimum_scale=cli.coach_minimum_scale,
        folds=cli.coach_folds,
        max_epochs=cli.coach_max_epochs,
        early_stop=cli.coach_early_stop,
        learning_rate=cli.coach_learning_rate,
        rank_margin=cli.rank_margin,
        rank_temperature=cli.rank_temperature,
        selection_temperature=cli.selection_temperature,
    )
    trainer = SemanticCostTrainerV99(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        semantic_pool_path=semantic_pool_path,
        valid_pool=valid_pool,
        test_pool=test_pool,
        coach_config=coach_config,
        minimum_oof_gain=cli.minimum_oof_gain,
        minimum_validation_gain=cli.minimum_validation_gain,
        bootstrap_quantile=cli.bootstrap_quantile,
        bootstrap_repeats=cli.bootstrap_repeats,
        maximum_harm_rate=cli.maximum_harm_rate,
    )
    trainer.train_all()


if __name__ == "__main__":
    main()
