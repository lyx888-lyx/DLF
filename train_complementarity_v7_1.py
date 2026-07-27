"""Train complementarity-preserving function distillation V7.1."""

from __future__ import annotations

import argparse
import glob
import logging
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.complementarity_system_v71 import (
    ComplementarityTrainerV71,
)
from trains.singleTask.function_consensus_system_v7 import build_teacher_cache
from trains.singleTask.model.CPFD_DLF import ComplementarityResidualStudent
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit globally and softly region-conditioned robust teacher "
            "committees, then distill their residual complementarity into a "
            "frozen DLF student with aligned evaluation."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--save-root",
        type=str,
        default="./result/complementarity_v71",
    )
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--teacher-checkpoint", action="append", default=[])
    parser.add_argument("--teacher-glob", action="append", default=[])
    parser.add_argument("--max-teachers", type=int, default=9)
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--rebuild-teacher-cache", action="store_true")
    parser.add_argument("--student-init", type=str, default="best-valid-teacher")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--residual-max", type=float, default=0.35)
    parser.add_argument("--region-temperature", type=float, default=0.55)
    parser.add_argument("--committee-steps", type=int, default=600)
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--early-stop", type=int, default=6)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--confidence-temperature", type=float, default=0.12)
    return parser.parse_args()


def discover_teacher_paths(cli):
    paths = []
    for value in cli.teacher_checkpoint:
        for token in str(value).split(","):
            token = token.strip()
            if token:
                paths.append(Path(token))
    for pattern in cli.teacher_glob:
        paths.extend(
            Path(value) for value in glob.glob(pattern, recursive=True)
        )

    unique = []
    seen = set()
    excluded = (
        "diagnostic",
        "smoke",
        "oracle",
        "shuffled",
        "failure",
    )
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
            "V7.1 requires at least three independent teacher checkpoints. "
            f"Discovered only {len(unique)}: {[str(value) for value in unique]}"
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

    args = get_config_regression(
        "DLF", cli.dataset, Path(cli.config)
    )
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
    logger.info("V7.1 teachers (%d): %s", len(teacher_paths), teacher_paths)

    dataloaders = MMDataLoader(args, cli.num_workers)
    cache_path = (
        Path(cli.teacher_cache)
        if cli.teacher_cache
        else save_dir / "complementarity_v71_teacher_cache.pth"
    )
    cache = build_teacher_cache(
        args,
        dataloaders,
        teacher_paths,
        device,
        cache_path,
        rebuild=cli.rebuild_teacher_cache,
    )

    valid_predictions = cache["splits"]["valid"]["predictions"]
    valid_labels = cache["splits"]["valid"]["labels"]
    valid_maes = torch.abs(
        valid_predictions - valid_labels.unsqueeze(1)
    ).mean(dim=(0, 2))

    if cli.student_init == "best-valid-teacher":
        init_index = int(torch.argmin(valid_maes).item())
        init_path = teacher_paths[init_index]
    elif cli.student_init == "first-teacher":
        init_index = 0
        init_path = teacher_paths[0]
    else:
        init_path = Path(cli.student_init)
        if not init_path.is_file():
            raise FileNotFoundError(
                f"Student initialization checkpoint not found: {init_path}"
            )
        resolved = str(init_path.resolve())
        matches = [
            index for index, path in enumerate(teacher_paths)
            if str(path.resolve()) == resolved
        ]
        if not matches:
            raise ValueError(
                "A custom student-init must also be included in the teacher pool "
                "so its cached anchor is available."
            )
        init_index = matches[0]

    logger.info(
        "Initializing V7.1 student from teacher %d: %s "
        "(valid MAE=%.6f)",
        init_index,
        init_path,
        float(valid_maes[init_index].item()),
    )

    model = ComplementarityResidualStudent(
        args,
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
        region_temperature=cli.region_temperature,
    ).to(device)
    incompatible = model.load_backbone_checkpoint(
        init_path, map_location=device
    )
    logger.info(
        "Loaded V7.1 backbone (missing=%d unexpected=%d)",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
    )

    trainer = ComplementarityTrainerV71(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        teacher_cache=cache,
        teacher_paths=teacher_paths,
        init_index=init_index,
        region_temperature=cli.region_temperature,
        committee_steps=cli.committee_steps,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        head_lr=cli.head_lr,
        weight_decay=cli.weight_decay,
        residual_clip=cli.residual_max,
        confidence_temperature=cli.confidence_temperature,
    )
    best = trainer.train(model, dataloaders)
    logger.info(
        "V7.1 selected epoch=%d valid_objective=%.6f alpha=%.2f",
        best["epoch"],
        best["value"],
        best["alpha"],
    )
    trainer.evaluate_and_save(model, dataloaders, best)


if __name__ == "__main__":
    main()
