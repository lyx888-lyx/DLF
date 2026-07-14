"""Stage 1 ModDrop training entrypoint.

This program intentionally constructs only train and validation loaders. It never
constructs, reads, or reports the test split during training.
"""

import argparse
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    clean_checkpoint_path,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    flatten_mode_metrics,
    missing_checkpoint_path,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Stage 1 DLF-ModDrop baseline.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--mode", choices=("moddrop",), default="moddrop")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-dir", default="result/missing_baseline/moddrop/train")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parsed = parser.parse_args()
    if parsed.eta != 1.0:
        parser.error("Stage 1 fixes eta at 1.0; tuning eta is not permitted.")
    if parsed.max_epochs is not None and parsed.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if parsed.smoke_test:
        parsed.max_epochs = 2 if parsed.max_epochs is None else min(parsed.max_epochs, 2)
    return parsed


def create_logger(log_dir, dataset_name, smoke_test):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    suffix = "smoke" if smoke_test else "train"
    log_path = log_dir / "DLF-{}-moddrop-{}-{}.log".format(dataset_name, suffix, timestamp)
    logger = logging.getLogger("missing_baseline")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger, log_path


def describe_metrics(metrics_by_mode):
    descriptions = []
    for mode, metrics in metrics_by_mode.items():
        descriptions.append(
            "{} acc_7={:.4f} acc_5={:.4f} acc_2={:.4f} F1={:.4f} Corr={:.4f} MAE={:.4f} Loss={:.4f}".format(
                mode,
                metrics["acc_7"],
                metrics["acc_5"],
                metrics["acc_2"],
                metrics["F1_score"],
                metrics["Corr"],
                metrics["MAE"],
                metrics["Loss"],
            )
        )
    return " | ".join(descriptions)


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


def load_clean_backbone(args, model_save_dir, seed):
    checkpoint = clean_checkpoint_path(model_save_dir, args.dataset_name, seed)
    if not checkpoint.is_file():
        raise FileNotFoundError("Gate 3 validation-best checkpoint not found: {}".format(checkpoint))
    backbone = DLF(args).to(args.device)
    state_dict = torch.load(checkpoint, map_location=args.device)
    backbone.load_state_dict(state_dict, strict=True)
    return backbone, checkpoint


def checkpoint_for_run(cli_args, dataset_name, seed):
    checkpoint = missing_checkpoint_path(cli_args.model_save_dir, dataset_name, seed)
    if cli_args.smoke_test:
        checkpoint = checkpoint.parent / "smoke" / checkpoint.name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint


def _finite_metrics(metrics_by_mode):
    return all(math.isfinite(float(value)) for metrics in metrics_by_mode.values() for value in metrics.values())


def train_one_seed(cli_args, seed, logger):
    setup_seed(seed)
    args = build_config(cli_args, seed)
    dataloader = MMDataLoader(args, cli_args.num_workers)
    if set(dataloader) != {"train", "valid"}:
        raise RuntimeError("Stage 1 training must expose exactly train and valid loaders.")

    backbone, clean_checkpoint = load_clean_backbone(args, cli_args.model_save_dir, seed)
    model = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    sim_loss = HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)

    checkpoint = checkpoint_for_run(cli_args, args.dataset_name, seed)
    best_j_val = float("inf")
    best_epoch = 0
    best_metrics = None
    epoch = 0
    last_token_gradients = {"audio": 0.0, "vision": 0.0}

    logger.info("seed=%s clean_checkpoint=%s device=%s", seed, clean_checkpoint, args.device)
    logger.info("training splits are train/valid only; no test loader is constructed")

    while True:
        epoch += 1
        model.train()
        optimizer.zero_grad()
        totals = {"full": 0.0, "missing": 0.0, "total": 0.0}
        sampled_counts = Counter({"LA": 0, "LV": 0, "L": 0})
        last_token_gradients = {"audio": 0.0, "vision": 0.0}

        for step, batch_data in enumerate(dataloader["train"], start=1):
            text = batch_data["text"].to(args.device)
            audio = batch_data["audio"].to(args.device)
            vision = batch_data["vision"].to(args.device)
            labels = batch_data["labels"]["M"].to(args.device).view(-1, 1)

            full_mask = mode_to_mask("LAV", batch_size=labels.size(0), device=args.device, dtype=audio.dtype)
            full_output = model(text, audio, vision, full_mask)
            full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, sim_loss)

            missing_mask = sample_missing_masks(
                labels.size(0), missing_generator, device=args.device, dtype=audio.dtype
            )
            sampled_counts.update(count_missing_modes(missing_mask))
            missing_output = model(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)

            total_loss = full_loss + cli_args.eta * missing_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN or Inf in Stage 1 training loss.")
            total_loss.backward()
            if model.missing_audio_token.grad is not None:
                last_token_gradients["audio"] = float(model.missing_audio_token.grad.norm().item())
            if model.missing_vision_token.grad is not None:
                last_token_gradients["vision"] = float(model.missing_vision_token.grad.norm().item())

            if step % args.update_epochs == 0 or step == len(dataloader["train"]):
                if args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            totals["full"] += float(full_loss.item())
            totals["missing"] += float(missing_loss.item())
            totals["total"] += float(total_loss.item())

        train_batches = len(dataloader["train"])
        averaged = {name: value / train_batches for name, value in totals.items()}
        validation_metrics = evaluate_all_modes(model, dataloader["valid"], args.device, "moddrop", criterion)
        if not _finite_metrics(validation_metrics):
            raise FloatingPointError("NaN or Inf in Stage 1 validation metrics.")
        j_val = float(validation_objective(validation_metrics))
        scheduler.step(j_val)

        logger.info(
            "seed=%s epoch=%s samples LA=%s LV=%s L=%s full_loss=%.6f missing_loss=%.6f total_loss=%.6f "
            "token_grad_audio=%.6g token_grad_vision=%.6g J_val=%.6f",
            seed,
            epoch,
            sampled_counts["LA"],
            sampled_counts["LV"],
            sampled_counts["L"],
            averaged["full"],
            averaged["missing"],
            averaged["total"],
            last_token_gradients["audio"],
            last_token_gradients["vision"],
            j_val,
        )
        logger.info("seed=%s epoch=%s validation %s", seed, epoch, describe_metrics(validation_metrics))

        if j_val <= best_j_val - 1e-6:
            best_j_val = j_val
            best_epoch = epoch
            best_metrics = validation_metrics
            torch.save(model.state_dict(), checkpoint)
            logger.info("seed=%s epoch=%s saved validation-best checkpoint=%s", seed, epoch, checkpoint)

        if cli_args.max_epochs is not None and epoch >= cli_args.max_epochs:
            break
        if epoch - best_epoch >= args.early_stop:
            break

    if best_metrics is None:
        raise RuntimeError("No validation-best checkpoint was saved.")
    if last_token_gradients["audio"] <= 0.0 or last_token_gradients["vision"] <= 0.0:
        raise RuntimeError("Missing tokens did not receive non-zero gradients.")

    row = {
        "Seed": int(seed),
        "BestEpoch": int(best_epoch),
        "J_val": float(best_j_val),
        "Checkpoint": str(checkpoint),
    }
    row.update(flatten_mode_metrics(best_metrics))
    logger.info(
        "seed=%s complete best_epoch=%s J_val=%.6f checkpoint=%s",
        seed,
        best_epoch,
        best_j_val,
        checkpoint,
    )
    return row


def main():
    cli_args = parse_args()
    logger, log_path = create_logger(cli_args.log_dir, cli_args.dataset, cli_args.smoke_test)
    logger.info("Stage 1 DLF-ModDrop starting; eta=1.0 smoke_test=%s max_epochs=%s", cli_args.smoke_test, cli_args.max_epochs)
    rows = [train_one_seed(cli_args, seed, logger) for seed in cli_args.seeds]
    result_dir = Path(cli_args.result_dir)
    if cli_args.smoke_test:
        result_dir = result_dir / "smoke"
    write_result_csvs(rows, result_dir, cli_args.dataset)
    logger.info("results=%s", result_dir)
    logger.info("log_path=%s", log_path)


if __name__ == "__main__":
    main()
