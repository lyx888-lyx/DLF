"""Validation-only sensitivity analysis for the quantile interpolation coefficient alpha.

For each target condition m, let L_i^m be the number of train discrepancies
strictly smaller than delta_i^m and T_i^m the size of its tied group. The
generalized empirical calibration is

    q_i^m(alpha) = (L_i^m + alpha * T_i^m) / N_m
    C_i^m(alpha) = 1 - q_i^m(alpha),   alpha in [0, 1].

The paper uses alpha=0.5, the midpoint of each tied empirical interval. This
runner tests the symmetric set {0.00, 0.25, 0.50, 0.75, 1.00} while keeping
all other training settings fixed.

IMPORTANT: TRAIN and VALID only. The MOSI test split is never constructed.
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import MMDataLoader
from train_cf_compat_kd import (
    batch_to_device,
    build_config,
    initialize_teacher_student,
)
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    MULTISEED_CACHE_VERSION,
    gated_kd_loss,
    load_counterfactual_cache,
    modes_from_masks,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    compute_full_dlf_loss,
    compute_task_loss,
    evaluate_all_modes,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


DEFAULT_ALPHAS = (0.00, 0.25, 0.50, 0.75, 1.00)


def parse_args():
    p = argparse.ArgumentParser(
        description="Validation-only quantile-interpolation sensitivity for CFCompat."
    )
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--alphas", nargs="+", type=float, default=list(DEFAULT_ALPHAS))
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--result-root", default="result")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    if args.max_epochs is not None and args.max_epochs < 1:
        p.error("--max-epochs must be positive.")
    if not args.alphas:
        p.error("--alphas cannot be empty.")
    for alpha in args.alphas:
        if not (0.0 <= alpha <= 1.0):
            p.error("Each alpha must lie in the closed interval [0,1].")
    if len(set(args.alphas)) != len(args.alphas):
        p.error("--alphas must be unique.")
    return args


def build_cli(args):
    return argparse.Namespace(
        dataset=args.dataset,
        seeds=[args.seed],
        gate_mode="compat",
        eta=1.0,
        lambda_kd=1.0,
        build_gate_cache_only=False,
        smoke_test=False,
        max_epochs=args.max_epochs,
        num_workers=args.num_workers,
        gpu_ids=args.gpu_ids,
        model_save_dir=args.model_save_dir,
        result_root=args.result_root,
        log_dir="log/missing_baseline",
        config_file=args.config_file,
        multiseed_replication=True,
    )


def output_root(args):
    if args.output_dir:
        return Path(args.output_dir)
    return (
        Path(args.result_root)
        / "analysis"
        / "quantile_interpolation_sensitivity_v2"
        / args.dataset
        / f"seed{args.seed}"
    )


def make_logger(output):
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("quantile_interpolation_sensitivity")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(output / "quantile_interpolation_sensitivity.log"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _interpolated_compatibility(deltas, alpha):
    """Compute q=(L+alpha*T)/N and compatibility=1-q for one condition."""
    values = np.asarray(deltas, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Interpolation requires a non-empty finite discrepancy vector.")

    _, inverse, counts = np.unique(
        values,
        return_inverse=True,
        return_counts=True,
    )
    lower_by_group = np.concatenate(
        ([0], np.cumsum(counts[:-1], dtype=np.int64))
    )
    lower = lower_by_group[inverse].astype(np.float64)
    ties = counts[inverse].astype(np.float64)

    q = (lower + float(alpha) * ties) / float(len(values))
    compat = 1.0 - q

    tol = 1e-12
    if np.any(q < -tol) or np.any(q > 1.0 + tol):
        raise ValueError(f"alpha={alpha} produces q outside [0,1].")
    if np.any(compat < -tol) or np.any(compat > 1.0 + tol):
        raise ValueError(f"alpha={alpha} produces compatibility outside [0,1].")

    q = np.clip(q, 0.0, 1.0)
    compat = np.clip(compat, 0.0, 1.0)
    return q, compat, ties


def cache_with_alpha(cache_frame, alpha):
    """Recompute mode-wise compatibility using tied-interval interpolation."""
    frame = cache_frame.copy()

    for mode in MISSING_MODES:
        deltas = frame[f"delta_{mode}"].to_numpy(dtype=np.float64)
        q, compat, ties = _interpolated_compatibility(deltas, alpha)
        frame[f"q_{mode}"] = q
        frame[f"compat_{mode}"] = compat
        frame[f"tie_size_{mode}"] = ties

    lookup = {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }
    return frame, lookup


def compatibility_for_modes_inclusive(cache_by_index, indices, modes, device, dtype):
    """Sensitivity-only lookup that permits the endpoint scores 0 and 1."""
    if len(indices) != len(modes):
        raise ValueError("Indices and modes differ in length.")

    values = []
    for index, mode in zip(indices, modes):
        if mode not in MISSING_MODES or int(index) not in cache_by_index:
            raise KeyError(f"Invalid cache binding index={index} mode={mode}")
        values.append(float(cache_by_index[int(index)][f"compat_{mode}"]))

    result = torch.as_tensor(values, device=device, dtype=dtype)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Compatibility must be finite.")
    if torch.any(result < 0) or torch.any(result > 1):
        raise FloatingPointError("Endpoint sensitivity requires compatibility in [0,1].")
    return result


def macro_metrics(metrics):
    keys = ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE")
    return {
        key: float(np.mean([metrics[mode][key] for mode in MISSING_MODES]))
        for key in keys
    }


def alpha_tag(alpha):
    return f"{alpha:.2f}".replace(".", "p")


def train_one(args, alpha, cache_frame, logger):
    setup_seed(args.seed)
    cli = build_cli(args)
    cfg = build_config(cli, args.seed)

    loaders = MMDataLoader(cfg, args.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Quantile-interpolation sensitivity must expose exactly train/valid loaders.")

    alpha_frame, cache_by_index = cache_with_alpha(cache_frame, alpha)

    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        cfg, cli, args.seed, loaders
    )
    optimizer = optim.Adam(student.parameters(), lr=cfg.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=cfg.patience
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(args.seed) + 104729)

    checkpoint_dir = (
        Path(args.model_save_dir)
        / "analysis"
        / "quantile_interpolation_sensitivity_v2"
        / args.dataset
        / f"seed{args.seed}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"alpha_{alpha_tag(alpha)}_best_valid.pth"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"{checkpoint} already exists. Use --overwrite only for an intentional rerun."
        )

    compatibility_means = {
        mode: float(alpha_frame[f"compat_{mode}"].mean())
        for mode in MISSING_MODES
    }
    compatibility_min = min(
        float(alpha_frame[f"compat_{mode}"].min()) for mode in MISSING_MODES
    )
    compatibility_max = max(
        float(alpha_frame[f"compat_{mode}"].max()) for mode in MISSING_MODES
    )

    best_j = float("inf")
    best_epoch = 0
    epoch_rows = []

    logger.info(
        "start alpha=%.2f seed=%s teacher=%s teacher_sha=%s compat_mean=%s range=[%.8f,%.8f] split=train+valid-only",
        alpha, args.seed, teacher_checkpoint, teacher_sha,
        compatibility_means, compatibility_min, compatibility_max
    )

    for epoch in range(1, (args.max_epochs or 1000) + 1):
        student.train()
        teacher.eval()
        optimizer.zero_grad()
        kd_losses = []

        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, cfg.device)

            full_mask = mode_to_mask("LAV", labels.size(0), cfg.device, audio.dtype)
            full_output = student(text, audio, vision, full_mask)
            full_loss, _ = compute_full_dlf_loss(
                full_output, labels, criterion, cosine, hinge
            )

            missing_mask = sample_missing_masks(
                labels.size(0), missing_generator, cfg.device, audio.dtype
            )
            modes = modes_from_masks(missing_mask)
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)

            teacher_prediction = teacher_lav_prediction(
                teacher, text, audio, vision
            )
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            compatibility = compatibility_for_modes_inclusive(
                cache_by_index, indices, modes, cfg.device, labels.dtype
            )
            # For the endpoint sensitivity settings alpha=0 or 1, exact
            # empirical-boundary samples may receive compatibility 1 or 0.
            # Zero weight is valid in the normalized KD objective.
            gate = compatibility.detach()
            kd_loss, _ = gated_kd_loss(
                missing_output["output_logit"], teacher_prediction, gate
            )
            kd_losses.append(float(kd_loss.detach()))

            total_loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite quantile-interpolation sensitivity loss.")
            total_loss.backward()

            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen teacher received gradients.")

            if step % cfg.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()

        valid = evaluate_all_modes(
            student, loaders["valid"], cfg.device, "moddrop", criterion
        )
        j_valid = float(validation_objective(valid))
        scheduler.step(j_valid)
        macro = macro_metrics(valid)

        epoch_rows.append({
            "Alpha": float(alpha),
            "Seed": args.seed,
            "Epoch": epoch,
            "J_valid": j_valid,
            "KD_loss": float(np.mean(kd_losses)),
            **{f"TargetMacro_{k}": v for k, v in macro.items()},
        })
        logger.info(
            "alpha=%.2f epoch=%s J_valid=%.6f target_macro_MAE=%.6f KD=%.6f",
            alpha, epoch, j_valid, macro["MAE"], float(np.mean(kd_losses))
        )

        if j_valid <= best_j - 1e-6:
            best_j = j_valid
            best_epoch = epoch
            torch.save(student.state_dict(), checkpoint)

        if epoch - best_epoch >= cfg.early_stop:
            break

    if not checkpoint.is_file():
        raise RuntimeError(f"No validation-best checkpoint saved for alpha={alpha}.")

    student.load_state_dict(torch.load(checkpoint, map_location=cfg.device), strict=True)
    final_valid = evaluate_all_modes(
        student, loaders["valid"], cfg.device, "moddrop", criterion
    )
    final_macro = macro_metrics(final_valid)

    midpoint_frame, _ = cache_with_alpha(cache_frame, 0.5)
    midpoint_mean = float(np.mean([
        midpoint_frame[f"compat_{mode}"].mean()
        for mode in MISSING_MODES
    ]))
    mean_compatibility = float(np.mean(list(compatibility_means.values())))
    mean_shift_from_midpoint = mean_compatibility - midpoint_mean

    result = {
        "Alpha": float(alpha),
        "Seed": args.seed,
        "BestValidEpoch": best_epoch,
        "J_valid": float(validation_objective(final_valid)),
        "TargetMacro_Acc7": final_macro["acc_7"],
        "TargetMacro_Acc5": final_macro["acc_5"],
        "TargetMacro_Acc2": final_macro["acc_2"],
        "TargetMacro_F1": final_macro["F1_score"],
        "TargetMacro_Corr": final_macro["Corr"],
        "TargetMacro_MAE": final_macro["MAE"],
        "MeanCompatibility": mean_compatibility,
        "MinCompatibility": compatibility_min,
        "MaxCompatibility": compatibility_max,
        "MeanCompatShiftVsAlpha0p5": mean_shift_from_midpoint,
        "Checkpoint": str(checkpoint),
    }

    condition_rows = []
    for mode in MISSING_MODES:
        m = final_valid[mode]
        condition_rows.append({
            "Alpha": float(alpha),
            "Condition": mode,
            "Acc7": float(m["acc_7"]),
            "Acc5": float(m["acc_5"]),
            "Acc2": float(m["acc_2"]),
            "F1": float(m["F1_score"]),
            "Corr": float(m["Corr"]),
            "MAE": float(m["MAE"]),
        })

    return result, condition_rows, epoch_rows


def latex_table(results):
    frame = pd.DataFrame(results).sort_values("Alpha")
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Sensitivity to the quantile interpolation coefficient $\alpha$ on the CMU-MOSI validation set.}",
        r"\label{tab:quantile_interpolation_sensitivity}",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"$\alpha$ & Acc-7 $\uparrow$ & Acc-5 $\uparrow$ & Acc-2 $\uparrow$ & F1 $\uparrow$ & Corr $\uparrow$ & MAE $\downarrow$ \\",
        r"\midrule",
    ]
    for _, row in frame.iterrows():
        label = f"{row['Alpha']:.2f}"
        if abs(float(row["Alpha"]) - 0.5) < 1e-12:
            label = r"\textbf{0.50}"
        lines.append(
            "{} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.3f} & {:.4f} \\".format(
                label,
                row["TargetMacro_Acc7"],
                row["TargetMacro_Acc5"],
                row["TargetMacro_Acc2"],
                row["TargetMacro_F1"],
                row["TargetMacro_Corr"],
                row["TargetMacro_MAE"],
            )
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    return "\n".join(lines)


def main():
    args = parse_args()
    output = output_root(args)
    logger = make_logger(output)

    cache_frame, _ = load_counterfactual_cache(
        args.result_root,
        args.dataset,
        version=MULTISEED_CACHE_VERSION,
        seed=args.seed,
    )
    if len(cache_frame) != 1284:
        raise RuntimeError("Expected the audited 1284-sample MOSI train cache.")

    required = [f"delta_{mode}" for mode in MISSING_MODES]
    missing = [name for name in required if name not in cache_frame.columns]
    if missing:
        raise ValueError(
            f"Counterfactual cache is missing discrepancy columns: {missing}"
        )

    results = []
    conditions = []
    epochs = []

    for alpha in args.alphas:
        result, local_conditions, local_epochs = train_one(
            args, float(alpha), cache_frame, logger
        )
        results.append(result)
        conditions.extend(local_conditions)
        epochs.extend(local_epochs)

    result_frame = pd.DataFrame(results).sort_values("Alpha").reset_index(drop=True)
    condition_frame = pd.DataFrame(conditions).sort_values(["Alpha", "Condition"]).reset_index(drop=True)
    epoch_frame = pd.DataFrame(epochs).sort_values(["Alpha", "Epoch"]).reset_index(drop=True)

    result_path = output / f"{args.dataset}_seed{args.seed}_quantile_interpolation_sensitivity.csv"
    condition_path = output / f"{args.dataset}_seed{args.seed}_quantile_interpolation_sensitivity_conditions.csv"
    epoch_path = output / f"{args.dataset}_seed{args.seed}_quantile_interpolation_sensitivity_epochs.csv"
    latex_path = output / f"{args.dataset}_seed{args.seed}_quantile_interpolation_sensitivity_table.tex"

    result_frame.to_csv(result_path, index=False)
    condition_frame.to_csv(condition_path, index=False)
    epoch_frame.to_csv(epoch_path, index=False)
    latex_path.write_text(latex_table(results), encoding="utf-8")

    display_cols = [
        "Alpha", "BestValidEpoch", "J_valid",
        "TargetMacro_Acc7", "TargetMacro_Acc5", "TargetMacro_Acc2",
        "TargetMacro_F1", "TargetMacro_Corr", "TargetMacro_MAE",
        "MeanCompatibility", "MeanCompatShiftVsAlpha0p5"
    ]

    print("Validation-only quantile-interpolation sensitivity complete.")
    print(f"results={result_path}")
    print(f"conditions={condition_path}")
    print(f"latex={latex_path}")
    print()
    print(result_frame[display_cols].to_string(index=False))
    print()
    print("Condition-wise validation metrics:")
    print(condition_frame.to_string(index=False))


if __name__ == "__main__":
    main()
