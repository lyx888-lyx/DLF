"""Refit the V9.4 specialist pool and cost-sensitive coach, then evaluate."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.cost_sensitive_coach_system_v94 import (
    CostSensitiveOOFCoachTrainerV94,
)
from trains.singleTask.oof_expert_pool_v94 import (
    CoachConfigV94,
    SpecialistConfigV94,
)
from trains.utils import MetricsTop
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
    parser.add_argument("--nested-pool", default="")
    parser.add_argument(
        "--v92-root", default="./result/oof_tail_residual_experts_v92"
    )
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--specialist-hidden-dim", type=int, default=64)
    parser.add_argument("--specialist-epochs", type=int, default=4)
    parser.add_argument("--specialist-learning-rate", type=float, default=3e-4)
    parser.add_argument("--specialist-residual-max", type=float, default=1.25)
    parser.add_argument("--coach-hidden-dim", type=int, default=96)
    parser.add_argument("--coach-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-oof-role-gain", type=float, default=0.0)
    return parser.parse_args()


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


def anchor_checkpoint(v92_seed_root: Path):
    path = v92_seed_root / "oof_tail_residual_experts_v92_summary.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    checkpoint = Path(str(data.get("anchor_checkpoint", "")))
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint


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
    nested_pool = (
        Path(cli.nested_pool)
        if cli.nested_pool
        else save_dir / "oof_expert_pool" / "fully_nested_oof_expert_pool_v94.pth"
    )
    if not nested_pool.is_file():
        raise FileNotFoundError(
            f"nested V9.4 pool missing: {nested_pool}. Run the builder first."
        )
    v92_seed_root = Path(cli.v92_root) / cli.dataset / f"seed_{cli.seed}"
    loaders = build_eval_loaders(args, cli.feature_batch_size, cli.num_workers)
    trainer = CostSensitiveOOFCoachTrainerV94(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        nested_pool_path=nested_pool,
        anchor_checkpoint=anchor_checkpoint(v92_seed_root),
        valid_loader=loaders["valid"],
        test_loader=loaders["test"],
        specialist_config=SpecialistConfigV94(
            hidden_dim=cli.specialist_hidden_dim,
            residual_max=cli.specialist_residual_max,
            epochs=cli.specialist_epochs,
            learning_rate=cli.specialist_learning_rate,
        ),
        coach_config=CoachConfigV94(
            hidden_dim=cli.coach_hidden_dim,
            learning_rate=cli.coach_learning_rate,
        ),
        min_oof_role_gain=cli.min_oof_role_gain,
    )
    trainer.run()


if __name__ == "__main__":
    main()
