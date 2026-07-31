"""Build fully nested OOF specialist actions for the V9.4 cost-sensitive coach."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.oof_expert_pool_v94 import (
    CoachConfigV94,
    SpecialistConfigV94,
    build_fully_nested_expert_pool,
)
from trains.singleTask.oof_group_splits_v92 import StageLimits
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--save-root", default="./result/cost_sensitive_oof_coach_v94"
    )
    parser.add_argument(
        "--top-oof-cache",
        default=(
            "./result/oof_tail_residual_experts_v92/mosi/seed_1111/"
            "oof_cfcompat/nested_grouped_oof_cfcompat_cache_v92.pth"
        ),
    )
    parser.add_argument("--inner-folds", type=int, default=3)
    parser.add_argument("--inner-valid-fraction", type=float, default=0.20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--clean-max-epochs", type=int, default=40)
    parser.add_argument("--moddrop-max-epochs", type=int, default=30)
    parser.add_argument("--cfcompat-max-epochs", type=int, default=30)
    parser.add_argument("--backbone-early-stop", type=int, default=7)
    parser.add_argument("--specialist-hidden-dim", type=int, default=64)
    parser.add_argument("--specialist-epochs", type=int, default=4)
    parser.add_argument("--specialist-learning-rate", type=float, default=3e-4)
    parser.add_argument("--specialist-residual-max", type=float, default=1.25)
    parser.add_argument("--coach-hidden-dim", type=int, default=96)
    parser.add_argument("--coach-max-epochs", type=int, default=30)
    parser.add_argument("--coach-early-stop", type=int, default=6)
    parser.add_argument("--coach-learning-rate", type=float, default=3e-4)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()
    if args.inner_folds < 2:
        parser.error("--inner-folds must be at least 2")
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
    top_oof_cache = Path(
        str(cli.top_oof_cache).replace("seed_1111", f"seed_{cli.seed}")
    )
    if not top_oof_cache.is_file():
        raise FileNotFoundError(
            f"V9.2 top OOF cache missing: {top_oof_cache}. Run V9.2 first."
        )
    output_dir = (
        Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}" / "oof_expert_pool"
    )
    limits = StageLimits(
        clean_max_epochs=cli.clean_max_epochs,
        moddrop_max_epochs=cli.moddrop_max_epochs,
        cfcompat_max_epochs=cli.cfcompat_max_epochs,
        early_stop=cli.backbone_early_stop,
    )
    specialist_config = SpecialistConfigV94(
        hidden_dim=cli.specialist_hidden_dim,
        residual_max=cli.specialist_residual_max,
        epochs=cli.specialist_epochs,
        learning_rate=cli.specialist_learning_rate,
    )
    coach_config = CoachConfigV94(
        hidden_dim=cli.coach_hidden_dim,
        max_epochs=cli.coach_max_epochs,
        early_stop=cli.coach_early_stop,
        learning_rate=cli.coach_learning_rate,
    )
    payload = build_fully_nested_expert_pool(
        args=args,
        train_dataset=dataset,
        top_oof_cache_path=top_oof_cache,
        output_dir=output_dir,
        inner_folds=cli.inner_folds,
        inner_valid_fraction=cli.inner_valid_fraction,
        num_workers=cli.num_workers,
        limits=limits,
        specialist_config=specialist_config,
        coach_config=coach_config,
        resume=not cli.no_resume,
    )
    logging.getLogger("MMSA").info(
        "V9.4 nested action-cost pool complete samples=%d feature_dim=%d",
        len(payload["sample_ids"]),
        payload["function_space"].size(1),
    )


if __name__ == "__main__":
    main()
