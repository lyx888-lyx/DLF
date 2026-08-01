"""Train the V9.11 grouped predicted-region expert router."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.predicted_region_router_crossfit_v911 import (
    PredictedRegionRouterConfigV911,
)
from trains.singleTask.predicted_region_router_system_v911 import (
    PredictedRegionRouterTrainerV911,
)
from trains.singleTask.semantic_expert_pool_v99 import (
    FrozenSemanticExpertPoolV99,
    anchor_checkpoint_from_summary,
    default_expert_paths,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="V9.11 grouped five-region expert router"
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
        "--v99-root",
        default="./result/semantic_cost_coach_v99/mosi/seed_1111",
    )
    parser.add_argument(
        "--save-root", default="./result/predicted_region_router_v911"
    )
    parser.add_argument("--semantic-pool", default="")
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--boundary-checkpoint", default="")
    parser.add_argument("--positive-checkpoint", default="")
    parser.add_argument("--strong-negative-checkpoint", default="")
    parser.add_argument("--strong-positive-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--role-residual-max", type=float, default=0.45)
    parser.add_argument("--tail-residual-max", type=float, default=1.50)

    parser.add_argument("--router-hidden-dim", type=int, default=128)
    parser.add_argument("--router-dropout", type=float, default=0.20)
    parser.add_argument("--router-residual-max", type=float, default=2.0)
    parser.add_argument("--ordinal-temperature", type=float, default=0.45)
    parser.add_argument("--ordinal-blend", type=float, default=0.50)
    parser.add_argument("--router-folds", type=int, default=5)
    parser.add_argument("--router-max-epochs", type=int, default=60)
    parser.add_argument("--router-early-stop", type=int, default=10)
    parser.add_argument("--router-learning-rate", type=float, default=3e-4)
    parser.add_argument("--router-weight-decay", type=float, default=1e-3)
    parser.add_argument("--router-batch-size", type=int, default=64)
    parser.add_argument("--empirical-prior-strength", type=float, default=20.0)

    parser.add_argument("--minimum-oof-gain", type=float, default=0.002)
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0015)
    parser.add_argument("--bootstrap-quantile", type=float, default=0.20)
    parser.add_argument("--bootstrap-repeats", type=int, default=500)
    parser.add_argument("--maximum-harm-rate", type=float, default=0.03)
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
    v9_root = Path(str(cli.v9_root).replace("seed_1111", f"seed_{cli.seed}"))
    v92_root = Path(str(cli.v92_root).replace("seed_1111", f"seed_{cli.seed}"))
    v99_root = Path(str(cli.v99_root).replace("seed_1111", f"seed_{cli.seed}"))
    semantic_pool = (
        Path(cli.semantic_pool)
        if cli.semantic_pool
        else v99_root
        / "strict_semantic_expert_pool"
        / "strict_semantic_expert_pool_v99.pth"
    )
    if not semantic_pool.is_file():
        raise FileNotFoundError(
            "V9.11 requires the completed V9.9 strict semantic pool: "
            f"{semantic_pool}. Run V9.9 first or pass --semantic-pool."
        )
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
    return semantic_pool, anchor, experts


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

    semantic_pool_path, anchor_checkpoint, expert_paths = resolve_paths(cli)
    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["train"].get_seq_len()

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    frozen_pool = FrozenSemanticExpertPoolV99(
        args,
        anchor_checkpoint=anchor_checkpoint,
        expert_paths=expert_paths,
        role_residual_max=cli.role_residual_max,
        tail_residual_max=cli.tail_residual_max,
    )
    valid_pool = frozen_pool.collect(
        build_loader(datasets["valid"], cli.feature_batch_size, cli.num_workers)
    )
    test_pool = frozen_pool.collect(
        build_loader(datasets["test"], cli.feature_batch_size, cli.num_workers)
    )
    router_config = PredictedRegionRouterConfigV911(
        hidden_dim=cli.router_hidden_dim,
        dropout=cli.router_dropout,
        residual_max=cli.router_residual_max,
        ordinal_temperature=cli.ordinal_temperature,
        ordinal_blend=cli.ordinal_blend,
        folds=cli.router_folds,
        max_epochs=cli.router_max_epochs,
        early_stop=cli.router_early_stop,
        learning_rate=cli.router_learning_rate,
        weight_decay=cli.router_weight_decay,
        batch_size=cli.router_batch_size,
        empirical_prior_strength=cli.empirical_prior_strength,
    )
    trainer = PredictedRegionRouterTrainerV911(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        semantic_pool_path=semantic_pool_path,
        valid_pool=valid_pool,
        test_pool=test_pool,
        router_config=router_config,
        minimum_oof_gain=cli.minimum_oof_gain,
        minimum_validation_gain=cli.minimum_validation_gain,
        bootstrap_quantile=cli.bootstrap_quantile,
        bootstrap_repeats=cli.bootstrap_repeats,
        maximum_harm_rate=cli.maximum_harm_rate,
    )
    trainer.train_all()


if __name__ == "__main__":
    main()
