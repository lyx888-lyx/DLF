"""Build leakage-controlled grouped OOF CFCompatKD features and predictions."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.grouped_oof_function_space_v92 import (
    StageLimits,
    run_nested_oof_cfcompat,
)
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train nested group-cross-fitted clean DLF, ModDrop evaluator, and "
            "CFCompatKD student models, then cache outer-holdout LAV features."
        )
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--save-root", default="./result/oof_tail_residual_experts_v92"
    )
    parser.add_argument("--outer-folds", type=int, default=3)
    parser.add_argument("--inner-valid-fraction", type=float, default=0.15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--clean-max-epochs", type=int, default=40)
    parser.add_argument("--moddrop-max-epochs", type=int, default=30)
    parser.add_argument("--cfcompat-max-epochs", type=int, default=30)
    parser.add_argument("--early-stop", type=int, default=7)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.outer_folds < 2:
        parser.error("--outer-folds must be at least 2")
    for name in ("clean_max_epochs", "moddrop_max_epochs", "cfcompat_max_epochs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


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
    output_dir = (
        Path(cli.save_root)
        / cli.dataset
        / f"seed_{cli.seed}"
        / "oof_cfcompat"
    )
    limits = StageLimits(
        clean_max_epochs=cli.clean_max_epochs,
        moddrop_max_epochs=cli.moddrop_max_epochs,
        cfcompat_max_epochs=cli.cfcompat_max_epochs,
        early_stop=cli.early_stop,
    )
    payload = run_nested_oof_cfcompat(
        args=args,
        train_dataset=dataset,
        output_dir=output_dir,
        seed=cli.seed,
        outer_folds=cli.outer_folds,
        inner_valid_fraction=cli.inner_valid_fraction,
        num_workers=cli.num_workers,
        limits=limits,
        resume=not cli.no_resume,
    )
    logging.getLogger("MMSA").info(
        "V9.2 OOF complete samples=%d folds=%d feature_dim=%d feature_space=%s",
        len(payload["sample_ids"]),
        payload["outer_folds"],
        payload["oof_feature"].size(1),
        payload["feature_space"],
    )


if __name__ == "__main__":
    main()
