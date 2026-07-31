"""Train V9.2 tail residual heads from nested grouped OOF CFCompatKD caches."""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.oof_tail_residual_system_v92 import (
    OOFTailResidualTrainerV92,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train low-capacity tail experts on grouped OOF CFCompatKD representations."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--save-root", default="./result/oof_tail_residual_experts_v92"
    )
    parser.add_argument("--oof-cache", default="")
    parser.add_argument("--teacher-checkpoint", action="append", default=[])
    parser.add_argument("--teacher-glob", action="append", default=[])
    parser.add_argument("--max-teachers", type=int, default=9)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--residual-max", type=float, default=1.50)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--membership-temperature", type=float, default=0.25)
    parser.add_argument("--gain-margin", type=float, default=0.08)
    parser.add_argument("--gain-fraction", type=float, default=0.20)
    parser.add_argument("--global-tolerance", type=float, default=0.012)
    parser.add_argument("--max-harm-rate", type=float, default=0.15)
    return parser.parse_args()


def discover_teacher_paths(cli) -> List[Path]:
    paths: List[Path] = []
    for value in cli.teacher_checkpoint:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                paths.append(Path(token))
    for pattern in cli.teacher_glob:
        paths.extend(Path(value) for value in glob.glob(pattern, recursive=True))
    excluded = (
        "diagnostic",
        "smoke",
        "oracle",
        "shuffled",
        "failure",
        "student",
        "unimodal",
        "role_conditioned",
        "tail_residual",
        "outer_fold",
    )
    unique = []
    seen = set()
    for path in sorted(paths):
        if not path.is_file():
            continue
        if any(token in str(path).lower() for token in excluded):
            continue
        key = str(path.resolve())
        if key not in seen:
            seen.add(key)
            unique.append(path)
    unique = unique[: max(1, int(cli.max_teachers))]
    if len(unique) < 3:
        raise RuntimeError(
            f"V9.2 requires at least three strong CFCompatKD checkpoints; found {len(unique)}"
        )
    return unique


def build_eval_loaders(args, batch_size, num_workers):
    datasets = {
        split: MMDataset(args, mode=split) for split in ("valid", "test")
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["valid"].get_seq_len()
    common = {
        "batch_size": int(batch_size),
        "num_workers": int(num_workers),
        "shuffle": False,
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
    }
    return {
        split: DataLoader(dataset, **common) for split, dataset in datasets.items()
    }


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

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    oof_cache = (
        Path(cli.oof_cache)
        if cli.oof_cache
        else save_dir
        / "oof_cfcompat"
        / "nested_grouped_oof_cfcompat_cache_v92.pth"
    )
    if not oof_cache.is_file():
        raise FileNotFoundError(
            f"OOF cache not found: {oof_cache}. Run build_grouped_oof_cfcompat_v9_2.py first."
        )
    teachers = discover_teacher_paths(cli)
    loaders = build_eval_loaders(args, cli.feature_batch_size, cli.num_workers)
    trainer = OOFTailResidualTrainerV92(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        oof_cache_path=oof_cache,
        teacher_paths=teachers,
        valid_loader=loaders["valid"],
        test_loader=loaders["test"],
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        membership_temperature=cli.membership_temperature,
        gain_margin=cli.gain_margin,
        gain_fraction=cli.gain_fraction,
        global_tolerance=cli.global_tolerance,
        max_harm_rate=cli.max_harm_rate,
        batch_size=cli.batch_size,
    )
    trainer.train_all()


if __name__ == "__main__":
    main()
