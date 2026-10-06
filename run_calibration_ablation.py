"""Calibration-design ablation for CFCompat.

Compare six pre-declared strategies formed by:
  - calibration scope: pooled/global vs. mode-wise;
  - mapping: Min-Max vs. Gaussian-CDF vs. empirical-CDF.

Training, initialization, missing-mask RNG, optimizer, KD loss, and checkpoint
selection are identical across variants. Checkpoints are selected exclusively
on validation data. When --evaluate-test is supplied, the MOSI test split is
constructed only after the validation-best checkpoint has been fixed, and is
used once per pre-declared variant for final aggregate reporting only.
"""

import argparse
import logging
import math
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
    compatibility_for_modes,
    compatibility_from_deltas,
    gate_weights,
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
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    evaluate_all_modes,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


SCOPES = ("pooled", "modewise")
MAPPINGS = ("minmax", "gaussian", "empirical")
CALIBRATIONS = tuple(
    f"{scope}_{mapping}"
    for scope in SCOPES
    for mapping in MAPPINGS
)
COMPAT_EPS = 1e-6


def parse_args():
    p = argparse.ArgumentParser(
        description="Pooled/mode-wise x Min-Max/Gaussian/empirical calibration ablation."
    )
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument(
        "--calibrations",
        nargs="+",
        choices=CALIBRATIONS,
        default=list(CALIBRATIONS),
    )
    p.add_argument("--max-epochs", type=int)
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--result-root", default="result")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--output-dir")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--evaluate-test",
        action="store_true",
        help=(
            "After validation-only checkpoint selection, evaluate the fixed "
            "checkpoint once on MOSI test and report aggregate metrics."
        ),
    )
    args = p.parse_args()
    if args.max_epochs is not None and args.max_epochs < 1:
        p.error("--max-epochs must be positive.")
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
        / "calibration_ablation_v2"
        / args.dataset
        / f"seed{args.seed}"
    )


def make_logger(output):
    output.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("calibration_ablation")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(output / "calibration_ablation.log"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def _clip_compatibility(values):
    values = np.asarray(values, dtype=np.float64)
    if not np.isfinite(values).all():
        raise FloatingPointError("Compatibility contains non-finite values.")
    return np.clip(values, COMPAT_EPS, 1.0 - COMPAT_EPS)


def minmax_compatibility(reference, values):
    """Map discrepancy to compatibility using train-reference Min-Max scaling."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if ref.size == 0 or not np.isfinite(ref).all() or not np.isfinite(vals).all():
        raise ValueError("Min-Max calibration requires finite non-empty values.")
    lo = float(ref.min())
    hi = float(ref.max())
    if hi <= lo:
        return np.full_like(vals, 0.5, dtype=np.float64)
    q = (vals - lo) / (hi - lo)
    return _clip_compatibility(1.0 - np.clip(q, 0.0, 1.0))


def gaussian_compatibility(reference, values):
    """Map discrepancy to compatibility using a Gaussian CDF fitted on train."""
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if ref.size == 0 or not np.isfinite(ref).all() or not np.isfinite(vals).all():
        raise ValueError("Gaussian calibration requires finite non-empty values.")
    mu = float(ref.mean())
    sigma = float(ref.std(ddof=0))
    if sigma <= 0:
        return np.full_like(vals, 0.5, dtype=np.float64)
    z = (vals - mu) / sigma
    cdf = np.asarray(
        [0.5 * (1.0 + math.erf(float(v) / math.sqrt(2.0))) for v in z],
        dtype=np.float64,
    )
    return _clip_compatibility(1.0 - cdf)


def empirical_compatibility(reference, values):
    """Compatibility = 1 - empirical midpoint CDF under the train reference."""
    ref = np.sort(np.asarray(reference, dtype=np.float64).reshape(-1))
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if ref.size == 0 or not np.isfinite(ref).all() or not np.isfinite(vals).all():
        raise ValueError("Empirical calibration requires finite non-empty values.")
    left = np.searchsorted(ref, vals, side="left").astype(np.float64)
    right = np.searchsorted(ref, vals, side="right").astype(np.float64)
    q = (left + 0.5 * (right - left)) / float(len(ref))
    lo = 0.5 / float(len(ref))
    hi = 1.0 - lo
    q = np.clip(q, lo, hi)
    return _clip_compatibility(1.0 - q)


def calibrated_cache(cache_frame, calibration):
    """Return a train-cache lookup for one pre-declared calibration strategy."""
    frame = cache_frame.copy()
    scope, mapping = calibration.split("_", 1)
    if scope not in SCOPES or mapping not in MAPPINGS:
        raise ValueError(f"Unknown calibration: {calibration}")

    pooled_reference = None
    if scope == "pooled":
        pooled_reference = np.concatenate(
            [frame[f"delta_{mode}"].to_numpy(dtype=np.float64) for mode in MISSING_MODES]
        )

    for mode in MISSING_MODES:
        values = frame[f"delta_{mode}"].to_numpy(dtype=np.float64)
        reference = pooled_reference if scope == "pooled" else values

        if mapping == "minmax":
            compat = minmax_compatibility(reference, values)
        elif mapping == "gaussian":
            compat = gaussian_compatibility(reference, values)
        elif mapping == "empirical":
            compat = empirical_compatibility(reference, values)
            if scope == "modewise":
                _, _, audited = compatibility_from_deltas(values)
                if not np.allclose(compat, audited, atol=1e-12, rtol=0):
                    raise RuntimeError(
                        f"{calibration}/{mode} disagrees with audited empirical compatibility."
                    )
        else:
            raise ValueError(mapping)

        frame[f"compat_{mode}"] = compat

    for mode in MISSING_MODES:
        values = frame[f"compat_{mode}"].to_numpy(dtype=np.float64)
        if not np.all((values > 0) & (values < 1)):
            raise ValueError(f"{calibration}/{mode} compatibility outside (0,1).")

    return frame, {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }

def macro_metrics(metrics):
    """Arithmetic macro over target conditions LA/LV/L."""
    keys = ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE")
    return {
        key: float(np.mean([metrics[mode][key] for mode in MISSING_MODES]))
        for key in keys
    }


def train_one(args, calibration, cache_frame, logger):
    """Train one fixed variant; select on valid, optionally report test once."""
    scope, mapping = calibration.split("_", 1)
    setup_seed(args.seed)
    cli = build_cli(args)
    cfg = build_config(cli, args.seed)

    loaders = MMDataLoader(cfg, args.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Calibration ablation must expose exactly train/valid loaders.")

    _, cache_by_index = calibrated_cache(cache_frame, calibration)

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

    output = output_root(args)
    checkpoint_dir = (
        Path(args.model_save_dir)
        / "analysis"
        / "calibration_ablation_v2"
        / args.dataset
        / f"seed{args.seed}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / f"{calibration}_best_valid.pth"
    if checkpoint.exists() and not args.overwrite:
        raise FileExistsError(
            f"{checkpoint} already exists. Use --overwrite only for an intentional rerun."
        )

    best_j = float("inf")
    best_epoch = 0
    epoch_rows = []

    logger.info(
        "start calibration=%s scope=%s mapping=%s seed=%s teacher=%s "
        "teacher_sha=%s selection_split=valid test_requested=%s",
        calibration, scope, mapping, args.seed, teacher_checkpoint, teacher_sha,
        bool(args.evaluate_test),
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
            compatibility = compatibility_for_modes(
                cache_by_index, indices, modes, cfg.device, labels.dtype
            )
            gate, _ = gate_weights(compatibility, gate_mode="compat")
            kd_loss, _ = gated_kd_loss(
                missing_output["output_logit"], teacher_prediction, gate
            )
            kd_losses.append(float(kd_loss.detach()))

            total_loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite calibration-ablation loss.")
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
            "calibration": calibration,
            "seed": args.seed,
            "epoch": epoch,
            "J_valid": j_valid,
            "KD_loss": float(np.mean(kd_losses)),
            **{f"target_macro_{k}": v for k, v in macro.items()},
        })
        logger.info(
            "calibration=%s epoch=%s J_valid=%.6f target_macro_MAE=%.6f KD=%.6f",
            calibration, epoch, j_valid, macro["MAE"], float(np.mean(kd_losses))
        )

        if j_valid <= best_j - 1e-6:
            best_j = j_valid
            best_epoch = epoch
            torch.save(student.state_dict(), checkpoint)

        if epoch - best_epoch >= cfg.early_stop:
            break

    if not checkpoint.is_file():
        raise RuntimeError("No validation-best checkpoint was saved.")

    student.load_state_dict(torch.load(checkpoint, map_location=cfg.device), strict=True)
    final_valid = evaluate_all_modes(
        student, loaders["valid"], cfg.device, "moddrop", criterion
    )
    valid_macro = macro_metrics(final_valid)

    final_test = None
    test_macro = None
    if args.evaluate_test:
        logger.info(
            "validation selection frozen: calibration=%s best_epoch=%s J_valid=%.6f; "
            "constructing test loader for aggregate final reporting",
            calibration, best_epoch, float(validation_objective(final_valid)),
        )
        test_loader = build_single_split_loader(
            cfg, split="test", num_workers=args.num_workers
        )
        final_test = evaluate_all_modes(
            student, test_loader, cfg.device, "moddrop", criterion
        )
        test_macro = macro_metrics(final_test)

    result = {
        "Scope": scope,
        "Mapping": mapping,
        "Calibration": calibration,
        "Seed": args.seed,
        "BestValidEpoch": best_epoch,
        "J_valid": float(validation_objective(final_valid)),
        "ValidMacro_Acc7": valid_macro["acc_7"],
        "ValidMacro_Acc5": valid_macro["acc_5"],
        "ValidMacro_Acc2": valid_macro["acc_2"],
        "ValidMacro_F1": valid_macro["F1_score"],
        "ValidMacro_Corr": valid_macro["Corr"],
        "ValidMacro_MAE": valid_macro["MAE"],
        "ValidLAV_Acc7": float(final_valid["LAV"]["acc_7"]),
        "ValidLAV_Acc5": float(final_valid["LAV"]["acc_5"]),
        "ValidLAV_Acc2": float(final_valid["LAV"]["acc_2"]),
        "ValidLAV_F1": float(final_valid["LAV"]["F1_score"]),
        "ValidLAV_Corr": float(final_valid["LAV"]["Corr"]),
        "ValidLAV_MAE": float(final_valid["LAV"]["MAE"]),
        "TestEvaluated": bool(args.evaluate_test),
        "Checkpoint": str(checkpoint),
    }

    if test_macro is not None:
        result.update({
            "TestMacro_Acc7": test_macro["acc_7"],
            "TestMacro_Acc5": test_macro["acc_5"],
            "TestMacro_Acc2": test_macro["acc_2"],
            "TestMacro_F1": test_macro["F1_score"],
            "TestMacro_Corr": test_macro["Corr"],
            "TestMacro_MAE": test_macro["MAE"],
            "TestLAV_Acc7": float(final_test["LAV"]["acc_7"]),
            "TestLAV_Acc5": float(final_test["LAV"]["acc_5"]),
            "TestLAV_Acc2": float(final_test["LAV"]["acc_2"]),
            "TestLAV_F1": float(final_test["LAV"]["F1_score"]),
            "TestLAV_Corr": float(final_test["LAV"]["Corr"]),
            "TestLAV_MAE": float(final_test["LAV"]["MAE"]),
        })

    condition_rows = []
    for split_name, metrics in (("valid", final_valid), ("test", final_test)):
        if metrics is None:
            continue
        for mode in MISSING_MODES:
            m = metrics[mode]
            condition_rows.append({
                "Scope": scope,
                "Mapping": mapping,
                "Calibration": calibration,
                "Split": split_name,
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
    frame = pd.DataFrame(results).copy()
    use_test = "TestMacro_Acc7" in frame.columns and frame["TestMacro_Acc7"].notna().all()
    prefix = "TestMacro" if use_test else "ValidMacro"
    split_label = "test" if use_test else "validation"

    scope_names = {"pooled": "Pooled", "modewise": "Mode-wise"}
    mapping_names = {
        "minmax": "Min-Max",
        "gaussian": "Gaussian-CDF",
        "empirical": "Empirical-CDF",
    }

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Calibration-design ablation on CMU-MOSI {split_label} data. Metrics are macro-averaged over LA/LV/L.}}",
        r"\label{tab:calibration_ablation}",
        r"\resizebox{\columnwidth}{!}{",
        r"\begin{tabular}{llcccccc}",
        r"\toprule",
        r"Scope & Mapping & Acc-7 $\uparrow$ & Acc-5 $\uparrow$ & Acc-2 $\uparrow$ & F1 $\uparrow$ & Corr $\uparrow$ & MAE $\downarrow$ \\",
        r"\midrule",
    ]

    for _, row in frame.iterrows():
        scope = scope_names[row["Scope"]]
        mapping = mapping_names[row["Mapping"]]
        if row["Scope"] == "modewise" and row["Mapping"] == "empirical":
            mapping = r"\textbf{Empirical-CDF (Ours)}"
            scope = r"\textbf{Mode-wise}"
        lines.append(
            "{} & {} & {:.2f} & {:.2f} & {:.2f} & {:.2f} & {:.4f} & {:.4f} \\".format(
                scope,
                mapping,
                100.0 * row[f"{prefix}_Acc7"],
                100.0 * row[f"{prefix}_Acc5"],
                100.0 * row[f"{prefix}_Acc2"],
                100.0 * row[f"{prefix}_F1"],
                row[f"{prefix}_Corr"],
                row[f"{prefix}_MAE"],
            )
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"}",
        r"\end{table}",
        "",
    ])
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

    results = []
    conditions = []
    epochs = []

    for calibration in args.calibrations:
        result, local_conditions, local_epochs = train_one(
            args, calibration, cache_frame, logger
        )
        results.append(result)
        conditions.extend(local_conditions)
        epochs.extend(local_epochs)

    result_frame = pd.DataFrame(results)
    condition_frame = pd.DataFrame(conditions)
    epoch_frame = pd.DataFrame(epochs)

    result_path = output / f"{args.dataset}_seed{args.seed}_calibration_ablation.csv"
    condition_path = output / f"{args.dataset}_seed{args.seed}_calibration_ablation_conditions.csv"
    epoch_path = output / f"{args.dataset}_seed{args.seed}_calibration_ablation_epochs.csv"
    latex_path = output / f"{args.dataset}_seed{args.seed}_calibration_ablation_table.tex"

    result_frame.to_csv(result_path, index=False)
    condition_frame.to_csv(condition_path, index=False)
    epoch_frame.to_csv(epoch_path, index=False)
    latex_path.write_text(latex_table(results), encoding="utf-8")

    print("Calibration-design ablation complete.")
    print(f"results={result_path}")
    print(f"conditions={condition_path}")
    print(f"latex={latex_path}")
    print()
    if args.evaluate_test:
        display_cols = [
            "Scope", "Mapping", "BestValidEpoch", "J_valid",
            "TestMacro_Acc7", "TestMacro_Acc5", "TestMacro_Acc2",
            "TestMacro_F1", "TestMacro_Corr", "TestMacro_MAE",
        ]
    else:
        display_cols = [
            "Scope", "Mapping", "BestValidEpoch", "J_valid",
            "ValidMacro_Acc7", "ValidMacro_Acc5", "ValidMacro_Acc2",
            "ValidMacro_F1", "ValidMacro_Corr", "ValidMacro_MAE",
        ]
    print(result_frame[display_cols].to_string(index=False))
    print()
    print("Condition-wise metrics:")
    print(condition_frame.to_string(index=False))


if __name__ == "__main__":
    main()
