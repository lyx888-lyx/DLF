"""Train and evaluate the conservative V9.5 selective category coach."""

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
from trains.singleTask.selective_category_coach_system_v95 import (
    SelectiveCategoryCoachTrainerV95,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="V9.5 selective five-region category coach")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument("--v9-root", default="./result/role_conditioned_experts_v9/mosi/seed_1111")
    parser.add_argument("--v92-root", default="./result/oof_tail_residual_experts_v92/mosi/seed_1111")
    parser.add_argument("--save-root", default="./result/selective_category_coach_v95")
    parser.add_argument("--oof-cache", default="")
    parser.add_argument("--anchor-checkpoint", default="")
    parser.add_argument("--boundary-checkpoint", default="")
    parser.add_argument("--positive-checkpoint", default="")
    parser.add_argument("--strong-negative-checkpoint", default="")
    parser.add_argument("--strong-positive-checkpoint", default="")
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--early-stop", type=int, default=7)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--residual-max", type=float, default=0.50)
    parser.add_argument("--ordinal-temperature", type=float, default=0.45)
    parser.add_argument("--minimum-validation-gain", type=float, default=0.0015)
    parser.add_argument("--bootstrap-quantile", type=float, default=0.20)
    parser.add_argument("--bootstrap-repeats", type=int, default=400)
    parser.add_argument("--maximum-activation-rate", type=float, default=0.35)
    parser.add_argument("--role-residual-max", type=float, default=0.45)
    parser.add_argument("--tail-residual-max", type=float, default=1.50)
    return parser.parse_args()


def build_loaders(args, batch_size, num_workers):
    datasets = {split: MMDataset(args, mode=split) for split in ("valid", "test")}
    if "seq_lens" in args:
        args["seq_lens"] = datasets["valid"].get_seq_len()
    common = dict(
        batch_size=int(batch_size), num_workers=int(num_workers), shuffle=False,
        drop_last=False, pin_memory=torch.cuda.is_available(),
    )
    return {split: DataLoader(dataset, **common) for split, dataset in datasets.items()}


def resolve_paths(cli):
    v9_root = Path(cli.v9_root)
    v92_root = Path(cli.v92_root)
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
    anchor = Path(cli.anchor_checkpoint) if cli.anchor_checkpoint else anchor_checkpoint_from_summary(v92_root)
    cache = Path(cli.oof_cache) if cli.oof_cache else (
        v92_root / "oof_cfcompat" / "nested_grouped_oof_cfcompat_cache_v92.pth"
    )
    if not cache.is_file():
        raise FileNotFoundError(cache)
    return anchor, experts, cache


def save_pool_csv(pool, path):
    frame = {
        "sample_id": pool["sample_ids"],
        "label": pool["labels"].view(-1).tolist(),
        "anchor": pool["anchor"].view(-1).tolist(),
    }
    for index, key in enumerate(pool["feature_keys"]):
        frame["anchor_" + key] = pool["function_space"][:, index].tolist()
    for name, values in pool["experts"].items():
        for key in ("prediction", "confidence", "correction"):
            frame[f"{name}_{key}"] = values[key].view(-1).tolist()
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

    anchor, expert_paths, cache = resolve_paths(cli)
    loaders = build_loaders(args, cli.feature_batch_size, cli.num_workers)
    frozen_pool = FrozenExpertPoolV93(
        args,
        anchor_checkpoint=anchor,
        expert_paths=expert_paths,
        role_residual_max=cli.role_residual_max,
        tail_residual_max=cli.tail_residual_max,
    )
    valid_pool = frozen_pool.collect(loaders["valid"])
    test_pool = frozen_pool.collect(loaders["test"])
    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    save_pool_csv(valid_pool, save_dir / "v95_valid_expert_pool_predictions.csv")
    save_pool_csv(test_pool, save_dir / "v95_test_expert_pool_predictions.csv")

    trainer = SelectiveCategoryCoachTrainerV95(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        oof_cache_path=cache,
        valid_pool=valid_pool,
        test_pool=test_pool,
        hidden_dim=cli.hidden_dim,
        residual_max=cli.residual_max,
        ordinal_temperature=cli.ordinal_temperature,
        folds=cli.folds,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        batch_size=cli.batch_size,
        minimum_validation_gain=cli.minimum_validation_gain,
        bootstrap_quantile=cli.bootstrap_quantile,
        bootstrap_repeats=cli.bootstrap_repeats,
        maximum_activation_rate=cli.maximum_activation_rate,
    )
    trainer.train_all()


if __name__ == "__main__":
    main()
