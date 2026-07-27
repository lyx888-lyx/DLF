"""Train RADIANT-DLF V6 on MOSI/MOSEI.

RADIANT decomposes complete-modality knowledge into a recoverable latent state
and a conditional innovation posterior, then learns value-of-information for
active audio/visual acquisition under missing-modality inputs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.RADIANT_DLF import RADIANTDLF
from trains.singleTask.radiant_system_v6 import RadiantTrainerV6
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recoverability-aware posterior learning and active modality acquisition."
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--config", type=str, default="./config/config.json")
    parser.add_argument("--save-root", type=str, default="./result/radiant_v6")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--head-epochs", type=int, default=8)
    parser.add_argument("--max-epochs", type=int, default=40)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--head-lr", type=float, default=3e-4)
    parser.add_argument("--tail-lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--latent-dim", type=int, default=128)
    parser.add_argument("--posterior-components", type=int, default=5)
    parser.add_argument("--posterior-dropout", type=float, default=0.20)
    parser.add_argument("--residual-max", type=float, default=1.5)
    parser.add_argument("--latent-scale", type=float, default=1.0)
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default="auto",
        help=(
            'DLF checkpoint. "auto" uses ./pt/DLF<dataset>.pth; '
            '"none" starts from a randomly initialized DLF backbone.'
        ),
    )
    return parser.parse_args()


def resolve_checkpoint(value, dataset):
    if value.lower() == "none":
        return None
    if value.lower() == "auto":
        path = Path("./pt") / f"DLF{dataset}.pth"
        return path if path.is_file() else None
    path = Path(value)
    if not path.is_file():
        raise FileNotFoundError(f"Initial DLF checkpoint not found: {path}")
    return path


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
    logger.info("RADIANT V6 outputs: %s", save_dir)

    dataloaders = MMDataLoader(args, cli.num_workers)
    model = RADIANTDLF(
        args,
        latent_dim=cli.latent_dim,
        posterior_components=cli.posterior_components,
        dropout=cli.posterior_dropout,
        residual_max=cli.residual_max,
        latent_scale=cli.latent_scale,
    ).to(device)
    checkpoint = resolve_checkpoint(cli.init_checkpoint, cli.dataset)
    if checkpoint is not None:
        incompatible = model.load_backbone_checkpoint(checkpoint, map_location=device)
        logger.info(
            "Initialized RADIANT DLF backbone from %s (missing=%d unexpected=%d)",
            checkpoint,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
    else:
        logger.warning("No DLF checkpoint found; RADIANT will train from random initialization.")

    trainer = RadiantTrainerV6(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        head_epochs=cli.head_epochs,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        head_lr=cli.head_lr,
        tail_lr=cli.tail_lr,
        weight_decay=cli.weight_decay,
    )
    best_lav, best_joint = trainer.train(model, dataloaders)
    summary = trainer.evaluate_and_save(
        model, dataloaders, best_lav=best_lav, best_j=best_joint
    )
    lav = summary["best_lav"]["modes"]["lav"]
    joint_lav = summary["best_joint"]["modes"]["lav"]
    logger.info(
        "RADIANT V6 TEST best-LAV epoch=%d anchor_MAE=%.4f final_MAE=%.4f "
        "support_MAE=%.4f residual_corr=%.4f policy=%s",
        summary["best_lav"]["selected_epoch"],
        lav["anchor_metrics"]["MAE"],
        lav["metrics"]["MAE"],
        lav["posterior"]["posterior_support_mae"],
        lav["posterior"]["residual_corr"],
        lav["policy"],
    )
    logger.info(
        "RADIANT V6 TEST best-joint epoch=%d LAV_MAE=%.4f LA_MAE=%.4f "
        "LV_MAE=%.4f L_MAE=%.4f",
        summary["best_joint"]["selected_epoch"],
        joint_lav["metrics"]["MAE"],
        summary["best_joint"]["modes"]["la"]["metrics"]["MAE"],
        summary["best_joint"]["modes"]["lv"]["metrics"]["MAE"],
        summary["best_joint"]["modes"]["l"]["metrics"]["MAE"],
    )


if __name__ == "__main__":
    main()
