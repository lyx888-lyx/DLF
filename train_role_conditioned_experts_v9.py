"""Train role-conditioned complementary multimodal experts (V9).

This entry point deliberately creates its own deterministic data loaders.  The
legacy MMDataLoader shuffles validation and test splits, which is unsafe for
cross-model sample alignment even though aggregate metrics are unaffected.
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import random
from pathlib import Path
from typing import Iterator, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Sampler

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.function_consensus_system_v7 import build_teacher_cache
from trains.singleTask.role_conditioned_expert_system_v9 import (
    RoleConditionedExpertTrainerV9,
)
from trains.singleTask.role_conditioned_experts_v9 import (
    REGION_NAMES,
    fit_role_teacher_weights,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


class EpochSeededRandomSampler(Sampler[int]):
    """Reproducible shuffle whose order depends only on seed and epoch."""

    def __init__(self, data_source, seed: int):
        self.data_source = data_source
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[int]:
        generator = torch.Generator()
        generator.manual_seed(self.seed + 104729 * self.epoch)
        return iter(torch.randperm(len(self.data_source), generator=generator).tolist())

    def __len__(self) -> int:
        return len(self.data_source)


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_dataloaders(args, num_workers: int, seed: int):
    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["train"].get_seq_len()
    train_sampler = EpochSeededRandomSampler(datasets["train"], seed)
    common = {
        "batch_size": int(args["batch_size"]),
        "num_workers": int(num_workers),
        "drop_last": False,
        "pin_memory": torch.cuda.is_available(),
        "worker_init_fn": _seed_worker,
    }
    return {
        "train": DataLoader(
            datasets["train"], sampler=train_sampler, shuffle=False, **common
        ),
        "valid": DataLoader(
            datasets["valid"], shuffle=False, **common
        ),
        "test": DataLoader(
            datasets["test"], shuffle=False, **common
        ),
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Construct five strong multimodal DLF experts with explicit "
            "sentiment-region roles, validation-fitted role teacher targets, "
            "global competence protection, and deployable category outputs."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--save-root", type=str, default="./result/role_conditioned_experts_v9"
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-checkpoint", action="append", default=[])
    parser.add_argument("--teacher-glob", action="append", default=[])
    parser.add_argument("--max-teachers", type=int, default=9)
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--rebuild-teacher-cache", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--residual-max", type=float, default=0.45)
    parser.add_argument("--head-epochs", type=int, default=4)
    parser.add_argument("--tail-epochs", type=int, default=12)
    parser.add_argument("--non-bert-epochs", type=int, default=0)
    parser.add_argument("--early-stop", type=int, default=5)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--tail-lr", type=float, default=2e-5)
    parser.add_argument("--backbone-lr", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--membership-floor", type=float, default=0.25)
    parser.add_argument("--membership-sigma-scale", type=float, default=1.0)
    parser.add_argument("--gain-margin", type=float, default=0.08)
    parser.add_argument("--gain-fraction", type=float, default=0.20)
    parser.add_argument("--global-tolerance", type=float, default=0.012)
    parser.add_argument("--teacher-fit-steps", type=int, default=450)
    parser.add_argument("--min-region-samples", type=int, default=12)
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
    )
    unique: List[Path] = []
    seen = set()
    for path in sorted(paths):
        if not path.is_file():
            continue
        if any(token in str(path).lower() for token in excluded):
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    unique = unique[: max(1, int(cli.max_teachers))]
    if len(unique) < 3:
        raise RuntimeError(
            "V9 requires at least three strong multimodal teacher checkpoints. "
            "Discovered %d: %s" % (len(unique), [str(value) for value in unique])
        )
    return unique


def _serialize_teacher_fit(teacher_fit):
    return {
        "global_weights": teacher_fit["global_weights"].tolist(),
        "role_weights": teacher_fit["role_weights"].tolist(),
        "selected_regularizations": teacher_fit["selected_regularizations"],
        "region_counts": teacher_fit["region_counts"],
    }


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("MMSA")
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])

    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = device
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = cli.seed
    args["cur_seed"] = 1
    args["batch_size"] = cli.batch_size

    save_dir = Path(cli.save_root) / cli.dataset / ("seed_%d" % cli.seed)
    save_dir.mkdir(parents=True, exist_ok=True)
    teacher_paths = discover_teacher_paths(cli)
    logger.info("V9 teachers (%d): %s", len(teacher_paths), teacher_paths)

    dataloaders = build_dataloaders(args, cli.num_workers, cli.seed)
    cache_path = (
        Path(cli.teacher_cache)
        if cli.teacher_cache
        else save_dir / "role_conditioned_experts_v9_teacher_cache.pth"
    )
    cache = build_teacher_cache(
        args,
        dataloaders,
        teacher_paths,
        device,
        cache_path,
        rebuild=cli.rebuild_teacher_cache,
    )

    valid = cache["splits"]["valid"]
    valid_teacher_mae = torch.abs(
        valid["predictions"] - valid["labels"].unsqueeze(1)
    ).mean(dim=(0, 2))
    anchor_index = int(torch.argmin(valid_teacher_mae).item())
    logger.info(
        "V9 anchor teacher=%d valid_MAE=%.6f path=%s",
        anchor_index,
        float(valid_teacher_mae[anchor_index].item()),
        teacher_paths[anchor_index],
    )

    teacher_fit = fit_role_teacher_weights(
        valid["predictions"],
        valid["labels"],
        valid["sample_ids"],
        steps=cli.teacher_fit_steps,
        min_region_samples=cli.min_region_samples,
    )
    pd.DataFrame(teacher_fit["cv_rows"]).to_csv(
        save_dir / "v9_role_teacher_cv.csv", index=False
    )
    (save_dir / "v9_role_teacher_weights.json").write_text(
        json.dumps(_serialize_teacher_fit(teacher_fit), indent=2),
        encoding="utf-8",
    )

    trainer = RoleConditionedExpertTrainerV9(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        teacher_cache=cache,
        teacher_paths=teacher_paths,
        anchor_index=anchor_index,
        teacher_fit=teacher_fit,
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
        head_epochs=cli.head_epochs,
        tail_epochs=cli.tail_epochs,
        non_bert_epochs=cli.non_bert_epochs,
        early_stop=cli.early_stop,
        head_lr=cli.head_lr,
        tail_lr=cli.tail_lr,
        backbone_lr=cli.backbone_lr,
        weight_decay=cli.weight_decay,
        membership_floor=cli.membership_floor,
        membership_sigma_scale=cli.membership_sigma_scale,
        gain_margin=cli.gain_margin,
        gain_fraction=cli.gain_fraction,
        global_tolerance=cli.global_tolerance,
    )
    summary = trainer.train_all(dataloaders)
    wins = summary["valid_specialization_wins"]
    logger.info(
        "V9 validation designated-role wins: %d/%d",
        sum(int(value["designated_is_best"]) for value in wins.values()),
        len(REGION_NAMES),
    )
    logger.info(
        "V9 TEST anchor MAE=%.6f coach MAE=%.6f true-region oracle MAE=%.6f",
        summary["test_results"]["anchor"]["MAE"],
        summary["test_results"]["category_coach_valid_selected"]["MAE"],
        summary["test_results"]["true_region_designated_oracle"]["MAE"],
    )


if __name__ == "__main__":
    main()
