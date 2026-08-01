"""Train the V9.12 single-holdout expert stack and convex soft gate."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.frontier_expert_pool_v97 import (
    RoleCrossfitConfigV97,
    TailCrossfitConfigV97,
)
from trains.singleTask.holdout_expert_pool_v912 import (
    build_holdout_expert_pool_v912,
)
from trains.singleTask.holdout_soft_gate_system_v912 import (
    HoldoutSoftGateConfigV912,
    HoldoutSoftGateTrainerV912,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "V9.12 one fixed Router-Train holdout, one unchanged expert stack, "
            "and a low-capacity convex soft gate"
        )
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", default="./config/config.json")
    parser.add_argument(
        "--v92-root",
        default="./result/oof_tail_residual_experts_v92/mosi/seed_1111",
    )
    parser.add_argument("--top-oof-cache", default="")
    parser.add_argument(
        "--save-root",
        default="./result/holdout_soft_gate_v912",
    )
    parser.add_argument("--outer-fold", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--feature-batch-size", type=int, default=32)
    parser.add_argument("--no-resume", action="store_true")

    parser.add_argument("--role-hidden-dim", type=int, default=192)
    parser.add_argument("--role-dropout", type=float, default=0.15)
    parser.add_argument("--role-residual-max", type=float, default=0.45)
    parser.add_argument("--role-head-epochs", type=int, default=4)
    parser.add_argument("--role-tail-epochs", type=int, default=12)
    parser.add_argument("--role-head-lr", type=float, default=3e-4)
    parser.add_argument("--role-tail-lr", type=float, default=2e-5)
    parser.add_argument("--role-backbone-lr", type=float, default=5e-6)
    parser.add_argument("--role-weight-decay", type=float, default=1e-3)

    parser.add_argument("--tail-hidden-dim", type=int, default=96)
    parser.add_argument("--tail-dropout", type=float, default=0.15)
    parser.add_argument("--tail-residual-max", type=float, default=1.50)
    parser.add_argument("--tail-epochs", type=int, default=12)
    parser.add_argument("--tail-learning-rate", type=float, default=3e-4)
    parser.add_argument("--tail-weight-decay", type=float, default=1e-3)
    parser.add_argument("--tail-batch-size", type=int, default=64)

    parser.add_argument("--teacher-fit-steps", type=int, default=300)
    parser.add_argument("--min-region-samples", type=int, default=10)

    parser.add_argument("--gate-hidden-dim", type=int, default=48)
    parser.add_argument("--gate-action-embedding-dim", type=int, default=8)
    parser.add_argument("--gate-dropout", type=float, default=0.10)
    parser.add_argument("--gate-temperature", type=float, default=1.0)
    parser.add_argument("--gate-anchor-bias", type=float, default=1.50)
    parser.add_argument("--gate-ensemble-members", type=int, default=3)
    parser.add_argument("--gate-max-epochs", type=int, default=80)
    parser.add_argument("--gate-early-stop", type=int, default=10)
    parser.add_argument("--gate-learning-rate", type=float, default=3e-4)
    parser.add_argument("--gate-weight-decay", type=float, default=1e-3)
    parser.add_argument("--gate-batch-size", type=int, default=64)
    parser.add_argument("--gate-harm-margin", type=float, default=0.05)
    parser.add_argument("--gate-expected-cost-weight", type=float, default=0.10)
    parser.add_argument("--gate-harm-weight", type=float, default=0.20)
    parser.add_argument(
        "--gate-specialist-mass-weight", type=float, default=0.005
    )
    parser.add_argument(
        "--validation-harm-penalty", type=float, default=0.05
    )
    parser.add_argument(
        "--minimum-validation-gain", type=float, default=0.0005
    )
    parser.add_argument(
        "--maximum-validation-harm", type=float, default=0.05
    )
    return parser.parse_args()


def resolve_paths(cli):
    v92_root = Path(
        str(cli.v92_root).replace("seed_1111", f"seed_{cli.seed}")
    )
    top_cache = (
        Path(cli.top_oof_cache)
        if cli.top_oof_cache
        else v92_root
        / "oof_cfcompat"
        / "nested_grouped_oof_cfcompat_cache_v92.pth"
    )
    if not top_cache.is_file():
        raise FileNotFoundError(
            "V9.12 requires the completed V9.2 grouped OOF cache and its "
            f"outer-fold checkpoints: {top_cache}"
        )
    return top_cache


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
    args["batch_size"] = int(cli.feature_batch_size)

    datasets = {
        split: MMDataset(args, mode=split)
        for split in ("train", "valid", "test")
    }
    if "seq_lens" in args:
        args["seq_lens"] = datasets["train"].get_seq_len()

    top_cache = resolve_paths(cli)
    save_dir = (
        Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    expert_dir = save_dir / "holdout_expert_stack"

    role_config = RoleCrossfitConfigV97(
        hidden_dim=cli.role_hidden_dim,
        dropout=cli.role_dropout,
        residual_max=cli.role_residual_max,
        head_epochs=cli.role_head_epochs,
        tail_epochs=cli.role_tail_epochs,
        head_lr=cli.role_head_lr,
        tail_lr=cli.role_tail_lr,
        backbone_lr=cli.role_backbone_lr,
        weight_decay=cli.role_weight_decay,
        batch_size=cli.feature_batch_size,
    )
    tail_config = TailCrossfitConfigV97(
        hidden_dim=cli.tail_hidden_dim,
        dropout=cli.tail_dropout,
        residual_max=cli.tail_residual_max,
        epochs=cli.tail_epochs,
        learning_rate=cli.tail_learning_rate,
        weight_decay=cli.tail_weight_decay,
        batch_size=cli.tail_batch_size,
    )
    build_holdout_expert_pool_v912(
        args=args,
        datasets=datasets,
        top_oof_cache_path=top_cache,
        output_dir=expert_dir,
        outer_fold=cli.outer_fold,
        role_config=role_config,
        tail_config=tail_config,
        num_workers=cli.num_workers,
        teacher_fit_steps=cli.teacher_fit_steps,
        min_region_samples=cli.min_region_samples,
        resume=not cli.no_resume,
    )
    expert_pool_path = expert_dir / "holdout_expert_pool_v912.pth"

    gate_config = HoldoutSoftGateConfigV912(
        hidden_dim=cli.gate_hidden_dim,
        action_embedding_dim=cli.gate_action_embedding_dim,
        dropout=cli.gate_dropout,
        temperature=cli.gate_temperature,
        anchor_bias=cli.gate_anchor_bias,
        ensemble_members=cli.gate_ensemble_members,
        max_epochs=cli.gate_max_epochs,
        early_stop=cli.gate_early_stop,
        learning_rate=cli.gate_learning_rate,
        weight_decay=cli.gate_weight_decay,
        batch_size=cli.gate_batch_size,
        harm_margin=cli.gate_harm_margin,
        expected_cost_weight=cli.gate_expected_cost_weight,
        harm_weight=cli.gate_harm_weight,
        specialist_mass_weight=cli.gate_specialist_mass_weight,
        validation_harm_penalty=cli.validation_harm_penalty,
    )
    trainer = HoldoutSoftGateTrainerV912(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        expert_pool_path=expert_pool_path,
        config=gate_config,
        minimum_validation_gain=cli.minimum_validation_gain,
        maximum_validation_harm=cli.maximum_validation_harm,
    )
    summary = trainer.train_all()
    logging.getLogger("MMSA").info(
        "V9.12 TEST anchor=%.6f selected=%.6f gain=%+.6f beta=%.2f",
        summary["test_anchor_mae"],
        summary["test_selected_mae"],
        summary["test_selected_gain"],
        summary["selected_beta"],
    )


if __name__ == "__main__":
    main()
