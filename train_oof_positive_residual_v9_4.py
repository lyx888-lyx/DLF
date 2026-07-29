"""Train cross-fitted calibrated staged positive residual V9.4."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.OOFPositiveResidual_DLF import (
    OOFPositiveResidualSpecialist,
)
from trains.singleTask.oof_positive_residual_system_v94 import (
    OOFPositiveResidualTrainerV94,
)
from trains.utils import MetricsTop
from utils import assign_gpu, setup_seed


def _float_grid(value: str):
    result = tuple(float(token.strip()) for token in str(value).split(",") if token.strip())
    if not result:
        raise argparse.ArgumentTypeError("Grid must contain at least one number.")
    return result


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Cross-fit a grouped calibration layer, train positive residual magnitude "
            "and activation in separate stages, and select deployment only on Validation."
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
        default="./result/oof_positive_residual_v94",
    )
    parser.add_argument("--teacher-cache", type=str, default="")
    parser.add_argument("--source-summary", type=str, default="")
    parser.add_argument("--source-checkpoint", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=1)

    parser.add_argument("--legacy-hidden-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=160)
    parser.add_argument("--dropout", type=float, default=0.18)
    parser.add_argument("--residual-max", type=float, default=0.35)
    parser.add_argument("--region-temperature", type=float, default=0.55)
    parser.add_argument("--max-correction", type=float, default=0.50)
    parser.add_argument("--initial-gate-probability", type=float, default=0.35)
    parser.add_argument("--initial-magnitude-fraction", type=float, default=0.02)

    parser.add_argument("--strong-threshold", type=float, default=1.0)
    parser.add_argument("--neutral-radius", type=float, default=1e-6)
    parser.add_argument(
        "--region-weight-mode",
        choices=["none", "inverse_sqrt", "effective_number"],
        default="inverse_sqrt",
    )
    parser.add_argument("--region-weight-min", type=float, default=0.50)
    parser.add_argument("--region-weight-max", type=float, default=2.00)
    parser.add_argument("--effective-beta", type=float, default=0.999)
    parser.add_argument("--ordinary-positive-boost", type=float, default=1.50)
    parser.add_argument("--huber-delta", type=float, default=0.30)
    parser.add_argument("--residual-margin", type=float, default=0.05)

    parser.add_argument("--crossfit-folds", type=int, default=5)
    parser.add_argument(
        "--calibrator-l2-grid",
        type=_float_grid,
        default=(0.1, 1.0, 10.0, 100.0),
    )
    parser.add_argument(
        "--calibrator-blend-grid",
        type=_float_grid,
        default=(0.0, 0.25, 0.50, 0.75, 1.0),
    )
    parser.add_argument("--max-calibration-shift", type=float, default=0.75)

    parser.add_argument("--magnitude-epochs", type=int, default=20)
    parser.add_argument("--magnitude-early-stop", type=int, default=6)
    parser.add_argument("--gate-epochs", type=int, default=15)
    parser.add_argument("--gate-early-stop", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--magnitude-active-weight", type=float, default=1.50)
    parser.add_argument("--magnitude-zero-weight", type=float, default=0.05)
    parser.add_argument("--magnitude-over-weight", type=float, default=0.05)
    parser.add_argument("--gate-focal-gamma", type=float, default=1.50)
    parser.add_argument("--gate-pos-weight-cap", type=float, default=8.0)

    parser.add_argument("--valid-mae-tolerance", type=float, default=0.001)
    parser.add_argument("--fold-mae-tolerance", type=float, default=0.010)
    parser.add_argument("--worst-region-tolerance", type=float, default=0.10)
    parser.add_argument("--nonpositive-mae-tolerance", type=float, default=0.010)
    parser.add_argument("--robust-selection-weight", type=float, default=0.05)
    parser.add_argument("--stability-selection-weight", type=float, default=0.05)
    parser.add_argument("--nonpositive-selection-weight", type=float, default=0.02)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--beta-grid", type=_float_grid, default=(0.40, 0.50, 0.60))
    parser.add_argument(
        "--gamma-grid",
        type=_float_grid,
        default=(0.10, 0.25, 0.50, 0.75, 1.00),
    )
    parser.add_argument(
        "--zero-shrinkage-grid",
        type=_float_grid,
        default=(0.02, 0.05, 0.10),
    )
    parser.add_argument(
        "--gate-threshold-grid",
        type=_float_grid,
        default=(0.30, 0.40, 0.50, 0.60, 0.70),
    )
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

    model = OOFPositiveResidualSpecialist(
        args,
        hidden_dim=cli.hidden_dim,
        legacy_hidden_dim=cli.legacy_hidden_dim,
        dropout=cli.dropout,
        residual_max=cli.residual_max,
        region_temperature=cli.region_temperature,
        max_correction=cli.max_correction,
        initial_gate_probability=cli.initial_gate_probability,
        initial_magnitude_fraction=cli.initial_magnitude_fraction,
    ).to(device)
    incompatible = model.load_v71_checkpoint(checkpoint_path, map_location=device)
    logger.info(
        "Loaded frozen V7.1 Student (missing=%d unexpected=%d): %s",
        len(incompatible.missing_keys),
        len(incompatible.unexpected_keys),
        checkpoint_path,
    )
    expected_new_prefixes = (
        "specialist_adapter.",
        "gate_head.",
        "magnitude_head.",
    )
    unexpected_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(expected_new_prefixes)
    ]
    if unexpected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            "V7.1 checkpoint is incompatible with the V9.4 legacy path: "
            f"missing={unexpected_missing}, unexpected={incompatible.unexpected_keys}"
        )

    trainer = OOFPositiveResidualTrainerV94(
        args=args,
        metrics_fn=MetricsTop("regression").getMetics(cli.dataset),
        save_dir=save_dir,
        teacher_cache=cache,
        teacher_paths=teacher_paths,
        init_index=init_index,
        source_summary=source_summary,
        max_epochs=cli.magnitude_epochs,
        early_stop=cli.magnitude_early_stop,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        strong_threshold=cli.strong_threshold,
        neutral_radius=cli.neutral_radius,
        region_weight_mode=cli.region_weight_mode,
        region_weight_min=cli.region_weight_min,
        region_weight_max=cli.region_weight_max,
        effective_beta=cli.effective_beta,
        ordinary_positive_boost=cli.ordinary_positive_boost,
        huber_delta=cli.huber_delta,
        residual_margin=cli.residual_margin,
        valid_mae_tolerance=cli.valid_mae_tolerance,
        fold_mae_tolerance=cli.fold_mae_tolerance,
        worst_region_tolerance=cli.worst_region_tolerance,
        nonpositive_mae_tolerance=cli.nonpositive_mae_tolerance,
        robust_selection_weight=cli.robust_selection_weight,
        stability_selection_weight=cli.stability_selection_weight,
        nonpositive_selection_weight=cli.nonpositive_selection_weight,
        folds=cli.folds,
        beta_grid=cli.beta_grid,
        gamma_grid=cli.gamma_grid,
        zero_shrinkage_grid=cli.zero_shrinkage_grid,
        crossfit_folds=cli.crossfit_folds,
        calibrator_l2_grid=cli.calibrator_l2_grid,
        calibrator_blend_grid=cli.calibrator_blend_grid,
        max_calibration_shift=cli.max_calibration_shift,
        magnitude_epochs=cli.magnitude_epochs,
        magnitude_early_stop=cli.magnitude_early_stop,
        gate_epochs=cli.gate_epochs,
        gate_early_stop=cli.gate_early_stop,
        gate_focal_gamma=cli.gate_focal_gamma,
        gate_pos_weight_cap=cli.gate_pos_weight_cap,
        magnitude_active_weight=cli.magnitude_active_weight,
        magnitude_zero_weight=cli.magnitude_zero_weight,
        magnitude_over_weight=cli.magnitude_over_weight,
        gate_threshold_grid=cli.gate_threshold_grid,
    )
    best = trainer.train(model, dataloaders)

    # The inherited V9.2 result writer expects ``expert_metrics``. V9.4 stores
    # the same checkpoint-level diagnostic under the clearer
    # ``magnitude_metrics`` name, so expose a compatibility alias before saving
    # and evaluation. This does not change model selection or predictions.
    best.setdefault("expert_metrics", dict(best.get("magnitude_metrics", {})))
    torch.save(best, save_dir / "oof_positive_residual_v94_best.pth")

    policy = best["policy"]
    logger.info(
        "V9.4 selected magnitude=%d gate=%d source=%s gate_mode=%s "
        "gamma=%.2f Valid(MAE=%.4f op=%.4f)",
        best["epoch"],
        best["gate_epoch"],
        policy["source"],
        policy.get("gate_mode", "none"),
        policy["gamma"],
        policy["mae"],
        policy["ordinary_positive_mae"],
    )
    trainer.evaluate_and_save(
        model, dataloaders, best, bootstrap_samples=cli.bootstrap_samples
    )


if __name__ == "__main__":
    main()
