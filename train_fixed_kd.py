"""Stage 2 fixed-weight prediction knowledge-distillation training entrypoint.

Training constructs only train and validation loaders.  The frozen teacher is
used for LAV prediction targets during training and validation-gap diagnostics;
student-only inference remains the evaluation path.
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
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence,
    assert_student_state_dict_has_no_teacher,
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    checkpoint_sha256,
    compute_validation_gaps,
    fixed_kd_checkpoint_path,
    prediction_kd_loss,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    clean_checkpoint_path,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    flatten_mode_metrics,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
    write_result_csvs,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train the Stage 2 DLF-FixedKD baseline.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1111])
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--lambda-kd", type=float, default=1.0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-dir", default="result/missing_baseline/fixed_kd/train")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parsed = parser.parse_args()
    if parsed.eta != 1.0:
        parser.error("Stage 2 fixes eta at 1.0; tuning is not permitted.")
    if parsed.lambda_kd != 1.0:
        parser.error("Stage 2 fixes lambda_kd at 1.0; tuning is not permitted.")
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
    log_path = log_dir / "DLF-{}-fixedkd-{}-{}.log".format(dataset_name, suffix, timestamp)
    logger = logging.getLogger("fixed_kd")
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


def checkpoint_for_run(cli_args, dataset_name, seed):
    checkpoint = fixed_kd_checkpoint_path(cli_args.model_save_dir, dataset_name, seed)
    if cli_args.smoke_test:
        checkpoint = checkpoint.parent / "smoke" / checkpoint.name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint


def _finite_metrics(metrics_by_mode):
    return all(math.isfinite(float(value)) for metrics in metrics_by_mode.values() for value in metrics.values())


def _grad_norm(parameters):
    squared_norm = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            squared_norm += float(parameter.grad.detach().pow(2).sum().item())
    return squared_norm ** 0.5


def _tensor_norm(tensors):
    squared_norm = 0.0
    for tensor in tensors:
        squared_norm += float(tensor.detach().pow(2).sum().item())
    return squared_norm ** 0.5


def _batch_to_device(batch_data, device):
    return (
        batch_data["text"].to(device),
        batch_data["audio"].to(device),
        batch_data["vision"].to(device),
        batch_data["labels"]["M"].to(device).view(-1, 1),
    )


def initialize_models(args, model_save_dir, seed, dataloader, logger):
    clean_checkpoint = clean_checkpoint_path(model_save_dir, args.dataset_name, seed)
    if not clean_checkpoint.is_file():
        raise FileNotFoundError("Gate 3 validation-best checkpoint not found: {}".format(clean_checkpoint))
    checkpoint_hash = checkpoint_sha256(clean_checkpoint)

    teacher = build_frozen_teacher(DLF, args, clean_checkpoint)
    student_backbone = DLF(args).to(args.device)
    student_backbone.load_state_dict(torch.load(clean_checkpoint, map_location=args.device), strict=True)
    student = MissingModalityWrapper(
        student_backbone,
        args.feature_dims[1],
        args.feature_dims[2],
    ).to(args.device)
    student.eval()

    first_valid_batch = next(iter(dataloader["valid"]))
    text, audio, vision, _ = _batch_to_device(first_valid_batch, args.device)
    assert_initial_lav_equivalence(teacher, student, text, audio, vision)
    logger.info(
        "seed=%s teacher_checkpoint=%s teacher_sha256=%s student_init_checkpoint=%s student_init_sha256=%s",
        seed,
        clean_checkpoint,
        checkpoint_hash,
        clean_checkpoint,
        checkpoint_hash,
    )
    logger.info("seed=%s teacher/student LAV initialization outputs are close", seed)
    return teacher, student, clean_checkpoint, checkpoint_hash


def train_one_seed(cli_args, seed, logger):
    setup_seed(seed)
    args = build_config(cli_args, seed)
    dataloader = MMDataLoader(args, cli_args.num_workers)
    if set(dataloader) != {"train", "valid"}:
        raise RuntimeError("Stage 2 training must expose exactly train and valid loaders.")

    teacher, student, clean_checkpoint, checkpoint_hash = initialize_models(
        args, cli_args.model_save_dir, seed, dataloader, logger
    )
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    criterion = nn.L1Loss()
    kd_criterion = nn.SmoothL1Loss()
    cosine = nn.CosineEmbeddingLoss()
    sim_loss = HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(seed) + 104729)

    checkpoint = checkpoint_for_run(cli_args, args.dataset_name, seed)
    best_j_val = float("inf")
    best_epoch = 0
    best_metrics = None
    best_gaps = None
    epoch = 0
    last_token_gradients = {"audio": 0.0, "vision": 0.0}

    logger.info("seed=%s device=%s lambda_kd=1.0 eta=1.0", seed, args.device)
    logger.info("training splits are train/valid only; no test loader is constructed")

    while True:
        epoch += 1
        student.train()
        teacher.eval()
        optimizer.zero_grad()
        totals = {"full": 0.0, "missing": 0.0, "kd": 0.0, "total": 0.0}
        sampled_counts = Counter({"LA": 0, "LV": 0, "L": 0})
        train_gap_sum = 0.0
        train_gap_samples = 0
        last_token_gradients = {"audio": 0.0, "vision": 0.0}
        last_regular_grad_norm = 0.0
        last_kd_student_grad_norm = 0.0
        last_teacher_grad_count = 0

        for step, batch_data in enumerate(dataloader["train"], start=1):
            text, audio, vision, labels = _batch_to_device(batch_data, args.device)

            full_mask = mode_to_mask("LAV", batch_size=labels.size(0), device=args.device, dtype=audio.dtype)
            full_output = student(text, audio, vision, full_mask)
            full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, sim_loss)

            missing_mask = sample_missing_masks(
                labels.size(0), missing_generator, device=args.device, dtype=audio.dtype
            )
            sampled_counts.update(count_missing_modes(missing_mask))
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)

            teacher_full_output = {"output_logit": teacher_lav_prediction(teacher, text, audio, vision)}
            kd_loss = prediction_kd_loss(kd_criterion, missing_output, teacher_full_output)
            if not torch.isfinite(kd_loss) or kd_loss.item() <= 0.0:
                raise FloatingPointError("Stage 2 prediction KD loss must be finite and positive.")

            if step == 1:
                kd_gradients = torch.autograd.grad(
                    kd_loss,
                    [parameter for parameter in student.parameters() if parameter.requires_grad],
                    retain_graph=True,
                    allow_unused=True,
                )
                last_kd_student_grad_norm = _tensor_norm(
                    [gradient for gradient in kd_gradients if gradient is not None]
                )
                if last_kd_student_grad_norm <= 0.0:
                    raise RuntimeError("Prediction KD did not produce a student gradient.")

            total_loss = full_loss + cli_args.eta * missing_loss + cli_args.lambda_kd * kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN or Inf in Stage 2 training loss.")
            total_loss.backward()

            last_teacher_grad_count = teacher_grad_count(teacher)
            if last_teacher_grad_count != 0:
                raise RuntimeError("Frozen teacher received gradients.")
            last_regular_grad_norm = _grad_norm(
                [
                    parameter
                    for name, parameter in student.named_parameters()
                    if not name.startswith("missing_")
                ]
            )
            if last_regular_grad_norm <= 0.0:
                raise RuntimeError("Student regular parameters did not receive gradients.")
            if student.missing_audio_token.grad is not None:
                last_token_gradients["audio"] = float(student.missing_audio_token.grad.norm().item())
            if student.missing_vision_token.grad is not None:
                last_token_gradients["vision"] = float(student.missing_vision_token.grad.norm().item())

            if step % args.update_epochs == 0 or step == len(dataloader["train"]):
                if args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(student.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_gap_sum += float(
                torch.abs(
                    teacher_full_output["output_logit"] - missing_output["output_logit"]
                ).sum().item()
            )
            train_gap_samples += int(labels.size(0))
            totals["full"] += float(full_loss.item())
            totals["missing"] += float(missing_loss.item())
            totals["kd"] += float(kd_loss.item())
            totals["total"] += float(total_loss.item())

        train_batches = len(dataloader["train"])
        averaged = {name: value / train_batches for name, value in totals.items()}
        train_gap = train_gap_sum / train_gap_samples
        validation_metrics = evaluate_all_modes(student, dataloader["valid"], args.device, "moddrop", criterion)
        validation_gaps = compute_validation_gaps(teacher, student, dataloader["valid"], args.device)
        if not _finite_metrics(validation_metrics) or not all(
            math.isfinite(float(value)) for value in validation_gaps.values()
        ):
            raise FloatingPointError("NaN or Inf in Stage 2 validation diagnostics.")
        j_val = float(validation_objective(validation_metrics))
        scheduler.step(j_val)

        logger.info(
            "seed=%s epoch=%s samples LA=%s LV=%s L=%s full_loss=%.6f missing_loss=%.6f kd_loss=%.6f "
            "total_loss=%.6f train_teacher_student_abs_gap=%.6f token_grad_audio=%.6g "
            "token_grad_vision=%.6g student_regular_grad_norm=%.6g kd_student_grad_norm=%.6g "
            "teacher_grad_count=%s J_val=%.6f",
            seed,
            epoch,
            sampled_counts["LA"],
            sampled_counts["LV"],
            sampled_counts["L"],
            averaged["full"],
            averaged["missing"],
            averaged["kd"],
            averaged["total"],
            train_gap,
            last_token_gradients["audio"],
            last_token_gradients["vision"],
            last_regular_grad_norm,
            last_kd_student_grad_norm,
            last_teacher_grad_count,
            j_val,
        )
        logger.info("seed=%s epoch=%s validation %s", seed, epoch, describe_metrics(validation_metrics))
        logger.info(
            "seed=%s epoch=%s diagnostic Gap_LA=%.6f Gap_LV=%.6f Gap_L=%.6f",
            seed,
            epoch,
            validation_gaps["Gap_LA"],
            validation_gaps["Gap_LV"],
            validation_gaps["Gap_L"],
        )

        if j_val <= best_j_val - 1e-6:
            best_j_val = j_val
            best_epoch = epoch
            best_metrics = validation_metrics
            best_gaps = validation_gaps
            student_state_dict = student.state_dict()
            assert_student_state_dict_has_no_teacher(student_state_dict)
            torch.save(student_state_dict, checkpoint)
            logger.info("seed=%s epoch=%s saved validation-best checkpoint=%s", seed, epoch, checkpoint)

        if cli_args.max_epochs is not None and epoch >= cli_args.max_epochs:
            break
        if epoch - best_epoch >= args.early_stop:
            break

    if best_metrics is None or best_gaps is None:
        raise RuntimeError("No validation-best checkpoint was saved.")
    if last_token_gradients["audio"] <= 0.0 or last_token_gradients["vision"] <= 0.0:
        raise RuntimeError("Missing tokens did not receive non-zero gradients.")

    row = {
        "Seed": int(seed),
        "BestEpoch": int(best_epoch),
        "J_val": float(best_j_val),
        "Checkpoint": str(checkpoint),
        "TeacherCheckpoint": str(clean_checkpoint),
        "TeacherCheckpointSHA256": checkpoint_hash,
        "StudentInitCheckpoint": str(clean_checkpoint),
        "StudentInitCheckpointSHA256": checkpoint_hash,
    }
    row.update(flatten_mode_metrics(best_metrics))
    row.update(best_gaps)
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
    logger.info(
        "Stage 2 DLF-FixedKD starting; eta=1.0 lambda_kd=1.0 smoke_test=%s max_epochs=%s",
        cli_args.smoke_test,
        cli_args.max_epochs,
    )
    rows = [train_one_seed(cli_args, seed, logger) for seed in cli_args.seeds]
    result_dir = Path(cli_args.result_dir)
    if cli_args.smoke_test:
        result_dir = result_dir / "smoke"
    write_result_csvs(rows, result_dir, cli_args.dataset)
    logger.info("results=%s", result_dir)
    logger.info("log_path=%s", log_path)


if __name__ == "__main__":
    main()
