"""Train and evaluate the V9.3 ordinal-strength plus expert-advantage coach."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.expert_pool_v93 import (
    FrozenExpertPoolV93,
    anchor_checkpoint_from_summary,
    default_expert_paths,
)
from trains.singleTask.ordinal_advantage_coach_system_v93 import (
    OrdinalAdvantageCoachTrainerV93,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an OOF ordinal sentiment-strength coach and grouped advantage router."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument("--v9-root", default="./result/role_conditioned_experts_v9/mosi/seed_1111")
    parser.add_argument("--v92-root", default="./result/oof_tail_residual_experts_v92/mosi/seed_1111")
    parser.add_argument("--save-root", default="./result/ordinal_advantage_coach_v93")
    parser.add_argument("--oof-cache", default="")
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--boundary-checkpoint", default="")
    parser.add_argument("--positive-checkpoint", default="")
    parser.add_argument("--strong-negative-checkpoint", default="")
    parser.add_argument("--strong-positive-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--region-folds", type=int, default=5)
    parser.add_argument("--advantage-folds", type=int, default=5)
    parser.add_argument("--region-max-epochs", type=int, default=50)
    parser.add_argument("--advantage-max-epochs", type=int, default=60)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--region-hidden-dim", type=int, default=64)
    parser.add_argument("--advantage-hidden-dim", type=int, default=48)
    parser.add_argument("--win-margin", type=float, default=0.02)
    parser.add_argument("--role-residual-max", type=float, default=0.45)
    parser.add_argument("--tail-residual-max", type=float, default=1.50)
    args = parser.parse_args()
    for name in ("region_folds", "advantage_folds"):
        if getattr(args, name) < 2:
            parser.error(f"--{name.replace('_', '-')} must be at least 2")
    return args


def build_loaders(args, batch_size, num_workers):
    datasets = {split: MMDataset(args, mode=split) for split in ("valid", "test")}
    if "seq_lens" in args:
        args["seq_lens"] = datasets["valid"].get_seq_len()
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "shuffle": False,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
    }
    return {split: DataLoader(dataset, **common) for split, dataset in datasets.items()}


def resolve_paths(cli):
    v9_root = Path(cli.v9_root)
    v92_root = Path(cli.v92_root)
    expert_paths = default_expert_paths(v9_root, v92_root)
    overrides = {
        "boundary": cli.boundary_checkpoint,
        "positive": cli.positive_checkpoint,
        "strong_negative": cli.strong_negative_checkpoint,
        "strong_positive": cli.strong_positive_checkpoint,
    }
    for name, value in overrides.items():
        if value:
            expert_paths[name] = Path(value)
    anchor = Path(cli.anchor_checkpoint) if cli.anchor_checkpoint else anchor_checkpoint_from_summary(v92_root)
    oof_cache = (
        Path(cli.oof_cache)
        if cli.oof_cache
        else v92_root / "oof_cfcompat" / "nested_grouped_oof_cfcompat_cache_v92.pth"
    )
    if not oof_cache.is_file():
        raise FileNotFoundError(f"OOF cache missing: {oof_cache}")
    return anchor, expert_paths, oof_cache


def save_pool_csv(pool, path):
    frame = {
        "sample_id": pool["sample_ids"],
        "label": pool["labels"].view(-1).tolist(),
        "anchor": pool["anchor"].view(-1).tolist(),
    }
    for index, key in enumerate(pool["feature_keys"]):
        frame["anchor_" + key] = pool["function_space"][:, index].tolist()
    for name, values in pool["experts"].items():
        frame[name + "_prediction"] = values["prediction"].view(-1).tolist()
        frame[name + "_confidence"] = values["confidence"].view(-1).tolist()
        frame[name + "_correction"] = values["correction"].view(-1).tolist()
    pd.DataFrame(frame).to_csv(path, index=False)


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    setup_seed(cli.seed)
    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = assign_gpu([cli.gpu])
    args["mode"] = "test"
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = int(cli.seed)
    args["cur_seed"] = int(cli.seed)
    args["batch_size"] = int(cli.feature_batch_size)

    anchor_checkpoint, expert_paths, oof_cache = resolve_paths(cli)
    loaders = build_loaders(args, cli.feature_batch_size, cli.num_workers)
    pool = FrozenExpertPoolV93(
        args,
        anchor_checkpoint=anchor_checkpoint,
        expert_paths=expert_paths,
        role_residual_max=cli.role_residual_max,
        tail_residual_max=cli.tail_residual_max,
    )
    valid_pool = pool.collect(loaders["valid"])
    test_pool = pool.collect(loaders["test"])

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_pool_csv(valid_pool, save_dir / "v93_valid_expert_pool_predictions.csv")
    save_pool_csv(test_pool, save_dir / "v93_test_expert_pool_predictions.csv")

    trainer = OrdinalAdvantageCoachTrainerV93(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        oof_cache_path=oof_cache,
        valid_pool=valid_pool,
        test_pool=test_pool,
        region_hidden_dim=cli.region_hidden_dim,
        region_folds=cli.region_folds,
        region_max_epochs=cli.region_max_epochs,
        advantage_hidden_dim=cli.advantage_hidden_dim,
        advantage_folds=cli.advantage_folds,
        advantage_max_epochs=cli.advantage_max_epochs,
        early_stop=cli.early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        batch_size=cli.batch_size,
        win_margin=cli.win_margin,
    )
    trainer.train_all()


if __name__ == "__main__":
    main()
