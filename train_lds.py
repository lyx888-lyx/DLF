"""Stage 2.6 train-only LDS weighting for the fixed DLF-ModDrop protocol.

This entrypoint constructs only train and validation loaders.  It never creates
or reads a held-out loader.
"""

import argparse
import hashlib
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import get_config_regression
from data_loader import MMDataset, MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.lds_utils import (
    LDS_METHOD_NAME,
    assert_fixed_lds_config,
    compute_full_dlf_loss_lds,
    compute_missing_task_loss_lds,
    config_sha256,
    evaluate_lds_validation,
    lds_checkpoint_path,
    lds_v1_config,
    prepare_train_label_weights,
    sample_weights_for_indices,
    write_density_audit,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    clean_checkpoint_path,
    count_missing_modes,
    flatten_mode_metrics,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train DLF-LDS-ModDrop-v1 with fixed train-only LDS-v1.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-dir", default="result/missing_baseline/lds_moddrop_v1/train")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--audit-density-only", action="store_true")
    parsed = parser.parse_args()
    if parsed.eta != 1.0:
        parser.error("Stage 2.6 fixes eta at 1.0; tuning eta is not permitted.")
    if parsed.max_epochs is not None and parsed.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if parsed.audit_density_only and parsed.smoke_test:
        parser.error("Density audit does not train and cannot be a smoke run.")
    if parsed.smoke_test:
        parsed.max_epochs = 2 if parsed.max_epochs is None else min(parsed.max_epochs, 2)
    assert_fixed_lds_config(lds_v1_config())
    return parsed


def create_logger(log_dir, dataset_name, smoke_test):
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    suffix = "smoke" if smoke_test else "train"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = root / "DLF-{}-ldsmoddrop-{}-{}.log".format(dataset_name, suffix, stamp)
    logger = logging.getLogger("lds_moddrop")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def build_config(cli_args, seed):
    args = get_config_regression("DLF", cli_args.dataset, cli_args.config_file)
    args.mode = "train"
    args.feature_T = ""
    args.feature_A = ""
    args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = int(seed)
    args.cur_seed = int(seed)
    args.device = assign_gpu(list(cli_args.gpu_ids))
    return args


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_clean_backbone(args, model_save_dir, seed):
    checkpoint = clean_checkpoint_path(model_save_dir, args.dataset_name, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError("Gate 3 validation-best checkpoint not found: {}".format(checkpoint))
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    return backbone, checkpoint, _sha256_file(checkpoint)


def checkpoint_for_run(cli_args, dataset_name, seed):
    checkpoint = lds_checkpoint_path(cli_args.model_save_dir, dataset_name, seed)
    if cli_args.smoke_test:
        checkpoint = checkpoint.parent / "smoke" / checkpoint.name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint


def _finite_metrics(metrics_by_mode):
    keys = ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE", "Loss")
    return all(math.isfinite(float(metrics[key])) for metrics in metrics_by_mode.values() for key in keys)


def _describe_metrics(metrics_by_mode):
    return " | ".join(
        "{} acc_7={:.4f} acc_5={:.4f} acc_2={:.4f} F1={:.4f} Corr={:.4f} MAE={:.4f} Loss={:.4f} MacroBinMAE={:.4f}".format(
            mode, values["acc_7"], values["acc_5"], values["acc_2"], values["F1_score"], values["Corr"], values["MAE"], values["Loss"], values["MacroBinMAE"]
        )
        for mode, values in metrics_by_mode.items()
    )


def _audit_output_dir(dataset):
    return Path("result") / "label_density" / "lds_v1" / dataset


def run_density_audit(cli_args):
    """Audit only training labels: no network, update state, batches, or validation."""
    seed = int(cli_args.seeds[0])
    setup_seed(seed)
    args = build_config(cli_args, seed)
    train_dataset = MMDataset(args, mode="train")
    artifacts = prepare_train_label_weights(np.asarray(train_dataset.labels["M"], dtype=np.float64).reshape(-1))
    summary = write_density_audit(artifacts, _audit_output_dir(cli_args.dataset))
    print("LDS train-only density audit complete: output={}".format(_audit_output_dir(cli_args.dataset)))
    print("samples={train_sample_count} nonempty_bins={nonempty_bins} weight_mean={weight_mean:.8f} weight_range=[{lo:.6f},{hi:.6f}] clipped_max={fraction_clipped_max:.6f}".format(
        lo=summary["weight_percentiles"]["min"], hi=summary["weight_percentiles"]["max"], **summary
    ))
    return summary


def train_one_seed(cli_args, seed, logger):
    assert_fixed_lds_config(lds_v1_config())
    setup_seed(seed)
    args = build_config(cli_args, seed)
    dataloader = MMDataLoader(args, cli_args.num_workers)
    if set(dataloader) != {"train", "valid"}:
        raise RuntimeError("Stage 2.6 may expose exactly train and validation loaders.")
    train_dataset = dataloader["train"].dataset
    artifacts = prepare_train_label_weights(np.asarray(train_dataset.labels["M"], dtype=np.float64).reshape(-1))
    if len(artifacts.sample_weights) != len(train_dataset):
        raise AssertionError("LDS train-weight count does not match the train dataset.")

    backbone, clean_checkpoint, clean_sha = load_clean_backbone(args, cli_args.model_save_dir, seed)
    model = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    criterion, cosine, sim_loss = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)
    checkpoint = checkpoint_for_run(cli_args, args.dataset_name, seed)

    logger.info("method=%s seed=%s clean_checkpoint=%s clean_sha256=%s lds_config_sha256=%s device=%s", LDS_METHOD_NAME, seed, clean_checkpoint, clean_sha, config_sha256(), args.device)
    logger.info("LDS source=train labels only; loaders=train/valid only; sampling remains Stage 1 shuffle and seed+104729 missing RNG")
    best_j, best_epoch, best_metrics, best_bin_rows, best_group_rows = float("inf"), 0, None, None, None
    epoch, token_grad_max = 0, {"audio": 0.0, "vision": 0.0}

    while True:
        epoch += 1
        model.train()
        optimizer.zero_grad()
        sums = Counter()
        sampled_counts = Counter({"LA": 0, "LV": 0, "L": 0})
        batch_weight_means, batch_weight_mins, batch_weight_maxs, seen = [], [], [], []
        grad_sums, grad_steps = Counter(), 0
        for step, batch in enumerate(dataloader["train"], start=1):
            indices = batch["index"].view(-1)
            index_values = indices.detach().cpu().numpy().astype(np.int64)
            seen.extend(index_values.tolist())
            weights = sample_weights_for_indices(artifacts, indices, args.device)
            if not np.allclose(weights.detach().cpu().numpy(), artifacts.sample_weights[index_values], rtol=1e-6, atol=1e-6):
                raise AssertionError("A shuffled sample did not retain its fixed LDS weight.")
            batch_weight_means.append(float(weights.mean().item()))
            batch_weight_mins.append(float(weights.min().item()))
            batch_weight_maxs.append(float(weights.max().item()))
            text, audio, vision = batch["text"].to(args.device), batch["audio"].to(args.device), batch["vision"].to(args.device)
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)

            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            full_output = model(text, audio, vision, full_mask)
            full_loss, full_details = compute_full_dlf_loss_lds(full_output, labels, criterion, cosine, sim_loss, weights)
            missing_mask = sample_missing_masks(labels.size(0), missing_generator, args.device, audio.dtype)
            sampled_counts.update(count_missing_modes(missing_mask))
            missing_output = model(text, audio, vision, missing_mask)
            missing_loss, missing_unweighted = compute_missing_task_loss_lds(missing_output, labels, weights, criterion)
            total_loss = full_loss + cli_args.eta * missing_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN or Inf in LDS training loss.")
            total_loss.backward()
            for name, token in (("audio", model.missing_audio_token), ("vision", model.missing_vision_token)):
                value = float(token.grad.norm().item()) if token.grad is not None else 0.0
                token_grad_max[name] = max(token_grad_max[name], value)
                grad_sums[name] += value
            grad_steps += 1
            if step % args.update_epochs == 0 or step == len(dataloader["train"]):
                if args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            sums["full_weighted"] += float(full_details["task_loss"].item())
            sums["missing_weighted"] += float(missing_loss.item())
            sums["full_unweighted"] += float(full_details["task_loss_unweighted"].item())
            sums["missing_unweighted"] += float(missing_unweighted.item())
            sums["reconstruction"] += float(full_details["reconstruction_loss"].item())
            sums["specific"] += float(full_details["specific_loss"].item())
            sums["orthogonality"] += float(full_details["orthogonality_loss"].item())
            sums["similarity"] += float(full_details["similarity_loss"].item())
            sums["total"] += float(total_loss.item())
        if len(seen) != len(train_dataset) or len(set(seen)) != len(train_dataset) or set(seen) != set(range(len(train_dataset))):
            raise AssertionError("LDS epoch did not cover each immutable train index exactly once.")
        divisor = float(len(dataloader["train"]))
        validation_metrics, bin_rows, group_rows = evaluate_lds_validation(model, dataloader["valid"], args.device, criterion, artifacts, seed, epoch)
        if not _finite_metrics(validation_metrics):
            raise FloatingPointError("NaN or Inf in LDS validation metrics.")
        j_val = float(validation_objective(validation_metrics))
        scheduler.step(j_val)
        logger.info(
            "seed=%s epoch=%s samples LA=%s LV=%s L=%s full_task_weighted=%.6f missing_task_weighted=%.6f full_task_unweighted=%.6f missing_task_unweighted=%.6f reconstruction=%.6f specific=%.6f orthogonality=%.6f similarity=%.6f total_loss=%.6f batch_weight_avg=%.6f batch_weight_min=%.6f batch_weight_max=%.6f token_grad_audio=%.6g token_grad_vision=%.6g J_val=%.6f",
            seed, epoch, sampled_counts["LA"], sampled_counts["LV"], sampled_counts["L"], sums["full_weighted"] / divisor, sums["missing_weighted"] / divisor, sums["full_unweighted"] / divisor, sums["missing_unweighted"] / divisor, sums["reconstruction"] / divisor, sums["specific"] / divisor, sums["orthogonality"] / divisor, sums["similarity"] / divisor, sums["total"] / divisor, float(np.mean(batch_weight_means)), float(np.min(batch_weight_mins)), float(np.max(batch_weight_maxs)), grad_sums["audio"] / grad_steps, grad_sums["vision"] / grad_steps, j_val,
        )
        logger.info("seed=%s epoch=%s validation %s", seed, epoch, _describe_metrics(validation_metrics))
        if j_val <= best_j - 1e-6:
            best_j, best_epoch, best_metrics = j_val, epoch, validation_metrics
            best_bin_rows, best_group_rows = bin_rows, group_rows
            torch.save(model.state_dict(), checkpoint)
            logger.info("seed=%s epoch=%s saved validation-best checkpoint=%s", seed, epoch, checkpoint)
        if cli_args.max_epochs is not None and epoch >= cli_args.max_epochs:
            break
        if epoch - best_epoch >= args.early_stop:
            break
    if best_metrics is None or min(token_grad_max.values()) <= 0.0:
        raise RuntimeError("No checkpoint or missing-token gradients were produced.")
    row = {"Seed": int(seed), "BestEpoch": int(best_epoch), "TotalEpochs": int(epoch), "J_val": float(best_j), "Checkpoint": str(checkpoint), "CleanCheckpoint": str(clean_checkpoint), "CleanCheckpointSHA256": clean_sha, "LDSConfigSHA256": config_sha256()}
    row.update(flatten_mode_metrics(best_metrics))
    logger.info("seed=%s complete best_epoch=%s total_epochs=%s J_val=%.6f checkpoint=%s", seed, best_epoch, epoch, best_j, checkpoint)
    return row, best_bin_rows, best_group_rows


def main():
    cli_args = parse_args()
    if cli_args.audit_density_only:
        run_density_audit(cli_args)
        return
    logger, log_path = create_logger(cli_args.log_dir, cli_args.dataset, cli_args.smoke_test)
    logger.info("%s starting; eta=1.0 smoke_test=%s max_epochs=%s", LDS_METHOD_NAME, cli_args.smoke_test, cli_args.max_epochs)
    rows, bin_rows, group_rows = [], [], []
    for seed in cli_args.seeds:
        row, bins, groups = train_one_seed(cli_args, seed, logger)
        rows.append(row)
        bin_rows.extend(bins)
        group_rows.extend(groups)
    output_dir = Path(cli_args.result_dir) / ("smoke" if cli_args.smoke_test else "")
    write_result_csvs(rows, output_dir, cli_args.dataset)
    pd.DataFrame(bin_rows).to_csv(output_dir / "{}_bin_metrics.csv".format(cli_args.dataset), index=False)
    pd.DataFrame(group_rows).to_csv(output_dir / "{}_density_group_metrics.csv".format(cli_args.dataset), index=False)
    logger.info("results=%s log_path=%s", output_dir, log_path)


if __name__ == "__main__":
    main()
