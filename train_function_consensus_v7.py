"""Train stable function-space committee distillation on DLF."""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.function_consensus_system_v7 import (
    FunctionConsensusTrainerV7,
    build_teacher_cache,
)
from trains.singleTask.model.FSC_DLF import FunctionConsensusStudent
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Distill the stable common function of multiple independently trained "
            "DLF checkpoints using robust prediction consensus, pairwise function "
            "relations and coordinate-free feature kernels."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument("--save-root", type=str, default="./result/function_consensus_v7")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-checkpoint", action="append", default=[])
    parser.add_argument(
        "--teacher-glob",
        action="append",
        default=[],
        help="Glob pattern for teacher checkpoints; may be specified repeatedly.",
    )
    parser.add_argument("--max-teachers", type=int, default=9)
    parser.add_argument("--rebuild-teacher-cache", action="store_true")
    parser.add_argument("--student-init", type=str, default="best-valid-teacher")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.20)
    parser.add_argument("--residual-max", type=float, default=0.50)
    parser.add_argument("--dispersion-temperature", type=float, default=0.12)
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--early-stop", type=int, default=8)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--tail-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    return parser.parse_args()


def discover_teacher_paths(cli):
    paths = []
    for value in cli.teacher_checkpoint:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                paths.append(Path(token))
    patterns = list(cli.teacher_glob)
    if not paths and not patterns:
        patterns = [
            f"./pt/*{cli.dataset}*seed*.pth",
            f"./pt/**/*{cli.dataset}*seed*.pth",
        ]
    for pattern in patterns:
        paths.extend(Path(value) for value in glob.glob(pattern, recursive=True))
    unique = []
    seen = set()
    excluded = ("radiant", "oracle", "residual", "router", "v4", "v5", "v6", "v7")
    for path in sorted(paths):
        if not path.is_file():
            continue
        key = str(path.resolve())
        if key in seen:
            continue
        if any(token in path.name.lower() for token in excluded):
            continue
        seen.add(key)
        unique.append(path)
    unique = unique[: max(1, int(cli.max_teachers))]
    if len(unique) < 3:
        raise RuntimeError(
            "V7 requires at least three independent teacher checkpoints. "
            "Pass them with repeated --teacher-checkpoint or --teacher-glob. "
            f"Discovered only {len(unique)}: {[str(path) for path in unique]}"
        )
    return unique


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

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    teacher_paths = discover_teacher_paths(cli)
    logger.info("V7 teachers (%d): %s", len(teacher_paths), teacher_paths)
    dataloaders = MMDataLoader(args, cli.num_workers)
    cache = build_teacher_cache(
        args,
        dataloaders,
        teacher_paths,
        device,
        save_dir / "function_consensus_v7_teacher_cache.pth",
        rebuild=cli.rebuild_teacher_cache,
    )

    if cli.student_init == "best-valid-teacher":
        valid_predictions = cache["splits"]["valid"]["predictions"]
        valid_labels = cache["splits"]["valid"]["labels"]
        maes = torch.abs(valid_predictions - valid_labels.unsqueeze(1)).mean(dim=(0, 2))
        init_index = int(torch.argmin(maes).item())
        init_path = teacher_paths[init_index]
    elif cli.student_init == "first-teacher":
        init_path = teacher_paths[0]
    else:
        init_path = Path(cli.student_init)
        if not init_path.is_file():
            raise FileNotFoundError(f"Student initialization checkpoint not found: {init_path}")
    logger.info("Initializing V7 student from: %s", init_path)

    model = FunctionConsensusStudent(
        args,
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
    ).to(device)
    incompatible = model.load_backbone_checkpoint(init_path, map_location=device)
    logger.info(
        "Loaded V7 student backbone (missing=%d unexpected=%d)",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )
    trainer = FunctionConsensusTrainerV7(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        teacher_cache=cache,
        teacher_paths=teacher_paths,
        dispersion_temperature=cli.dispersion_temperature,
        head_epochs=cli.head_epochs,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        head_lr=cli.head_lr,
        tail_lr=cli.tail_lr,
        weight_decay=cli.weight_decay,
    )
    best = trainer.train(model, dataloaders)
    logger.info("V7 selected epoch=%d valid_MAE=%.6f", best["epoch"], best["value"])
    trainer.evaluate_and_save(model, dataloaders, best)


if __name__ == "__main__":
    main()
