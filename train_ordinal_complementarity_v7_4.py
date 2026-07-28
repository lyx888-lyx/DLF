"""Train ordinal-consistent complementarity V7.4 from the frozen V7.1 run."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.OrdinalCPFD_DLF import (
    OrdinalComplementarityStudent,
)
from trains.singleTask.ordinal_consistency_system_v74 import (
    OrdinalConsistencyTrainerV74,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train monotonic Acc-7/Acc-5 ordinal heads on top of the frozen "
            "V7.1 complementarity Student, with Valid-only constrained policy "
            "selection."
        )
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument(
        "--source-run-dir",
        type=str,
        default="./result/complementarity_v71/mosi/seed_1111",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default="./result/ordinal_consistency_v74",
    )
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--source-summary", type=str, default="")
    parser.add_argument("--source-checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--ordinal-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--residual-max", type=float, default=0.35)
    parser.add_argument("--region-temperature", type=float, default=0.55)
    parser.add_argument("--max-ordinal-blend", type=float, default=0.30)
    parser.add_argument("--initial-ordinal-blend", type=float, default=0.02)
    parser.add_argument("--max-epochs", type=int, default=12)
    parser.add_argument("--early-stop", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--valid-mae-tolerance", type=float, default=0.001)
    parser.add_argument("--fold-mae-tolerance", type=float, default=0.010)
    parser.add_argument("--min-accuracy-gain", type=float, default=0.002)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--boundary-temperature", type=float, default=4.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    return parser.parse_args()


def _load_torch(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def main():
    cli = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger("MMSA")
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])

    source_dir = Path(cli.source_run_dir)
    summary_path = (
        Path(cli.source_summary)
        if cli.source_summary
        else source_dir / "complementarity_v71_summary.json"
    )
    checkpoint_path = (
        Path(cli.source_checkpoint)
        if cli.source_checkpoint
        else source_dir / "complementarity_v71_best.pth"
    )
    cache_path = (
        Path(cli.teacher_cache)
        if cli.teacher_cache
        else source_dir / "complementarity_v71_teacher_cache.pth"
    )
    for path in (summary_path, checkpoint_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    source_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    teacher_paths = [Path(value) for value in source_summary["teacher_paths"]]
    init_index = int(source_summary["base_teacher_index"])
    missing_teachers = [path for path in teacher_paths if not path.is_file()]
    if missing_teachers:
        raise FileNotFoundError(
            "Missing source Teacher checkpoints:\n"
            + "\n".join(f"  - {path}" for path in missing_teachers)
        )

    args = get_config_regression("DLF", cli.dataset, Path(cli.config))
    args["device"] = device
    args["train_mode"] = "regression"
    args["feature_T"] = ""
    args["feature_A"] = ""
    args["feature_V"] = ""
    args["seed"] = cli.seed
    args["cur_seed"] = 1
    args["batch_size"] = cli.batch_size

    dataloaders = MMDataLoader(args, cli.num_workers)
    cache = _load_torch(cache_path)
    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)

    model = OrdinalComplementarityStudent(
        args,
        hidden_dim=cli.hidden_dim,
        ordinal_hidden_dim=cli.ordinal_hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
        region_temperature=cli.region_temperature,
        max_ordinal_blend=cli.max_ordinal_blend,
        initial_ordinal_blend=cli.initial_ordinal_blend,
    ).to(device)
    incompatible = model.load_v71_checkpoint(checkpoint_path, map_location=device)
    logger.info(
        "Loaded frozen V7.1 Student (missing=%d unexpected=%d): %s",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
        checkpoint_path,
    )
    expected_new_prefixes = (
        "ordinal_adapter.",
        "ordinal7_head.",
        "ordinal5_head.",
        "ordinal_blend_logit",
    )
    unexpected_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(expected_new_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "V7.1 checkpoint is incompatible with the V7.4 legacy path: "
            f"missing={unexpected_missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )

    trainer = OrdinalConsistencyTrainerV74(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        teacher_cache=cache,
        teacher_paths=teacher_paths,
        init_index=init_index,
        source_summary=source_summary,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        valid_mae_tolerance=cli.valid_mae_tolerance,
        fold_mae_tolerance=cli.fold_mae_tolerance,
        min_accuracy_gain=cli.min_accuracy_gain,
        folds=cli.folds,
        boundary_temperature=cli.boundary_temperature,
    )
    best = trainer.train(model, dataloaders)
    logger.info(
        "V7.4 selected epoch=%d source=%s Valid(MAE=%.4f Acc7=%.4f Acc5=%.4f)",
        best["epoch"],
        best["policy"]["source"],
        best["policy"]["mae"],
        best["policy"]["acc_7"],
        best["policy"]["acc_5"],
    )
    trainer.evaluate_and_save(
        model,
        dataloaders,
        best,
        bootstrap_samples=cli.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
