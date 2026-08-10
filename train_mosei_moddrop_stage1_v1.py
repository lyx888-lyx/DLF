"""MOSEI Stage-1 DLF-ModDrop training, Valid-only and Windows-safe.

This is a cross-dataset port of the frozen Stage-1 ModDrop mathematics in
``train_missing.py``. It deliberately does NOT construct Test.

The only execution adaptation is sequential-equivalent backward within each
minibatch:

    backward(L_full); backward(L_missing); optimizer.step() only at the same
    gradient-accumulation boundary as the original trainer.

Thus the accumulated parameter gradient is still grad(L_full + L_missing), while
only one forward graph needs to be resident at a time. This materially reduces
activation memory on an 8 GiB Windows laptop GPU without changing the objective,
checkpoint criterion, scheduler metric, or early-stop rule.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import pandas as pd
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
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


VERSION = "mosei_moddrop_stage1_v1"
FORMAL_SEEDS = (1111, 1112, 1113, 1114, 1115)
DATASET = "mosei"
ETA = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one MOSEI Stage-1 ModDrop seed on Valid only.")
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline/mosei_moddrop_stage1_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_workers != 0:
        parser.error("This Windows MOSEI runner fixes --num-workers=0 to avoid spawn/copy overhead.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(int(args.max_epochs), 2)
    elif args.max_epochs is not None:
        parser.error("Formal Stage-1 forbids --max-epochs; validation early-stop remains the frozen rule.")
    return args


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_logger(cli: argparse.Namespace) -> tuple[logging.Logger, Path]:
    root = Path(cli.log_dir)
    root.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = root / f"DLF-mosei-moddrop-seed{cli.seed}-{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger = logging.getLogger("mosei_moddrop_stage1_v1")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def build_config(cli: argparse.Namespace) -> Any:
    args = get_config_regression("DLF", DATASET, cli.config_file)
    args.mode = "train"
    args.feature_T = ""
    args.feature_A = ""
    args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = int(cli.seed)
    args.cur_seed = int(cli.seed)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def stage0_manifest_path(result_root: str, seed: int) -> Path:
    return Path(result_root) / "missing_baseline" / "mosei_clean_stage0_v1" / f"seed{seed}" / "run_manifest.json"


def verify_clean_source(cli: argparse.Namespace, args: Any) -> tuple[Path, str, Dict[str, Any]]:
    checkpoint = clean_checkpoint_path(cli.model_save_dir, DATASET, cli.seed)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Required clean MOSEI checkpoint is absent: {checkpoint}")
    manifest_path = stage0_manifest_path(cli.result_root, cli.seed)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Required Stage-0 manifest is absent: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual_sha = sha256(checkpoint)
    recorded_sha = str(manifest.get("checkpoint_sha256", ""))
    if actual_sha != recorded_sha:
        raise RuntimeError(
            f"Clean checkpoint SHA mismatch for seed {cli.seed}: actual={actual_sha}, manifest={recorded_sha}"
        )
    if manifest.get("dataset") != DATASET or int(manifest.get("seed", -1)) != int(cli.seed):
        raise RuntimeError("Stage-0 manifest dataset/seed binding mismatch.")
    if bool(manifest.get("test_constructed", True)):
        raise RuntimeError("Stage-0 training manifest unexpectedly says Test was constructed.")
    if int(args.batch_size) != int(manifest.get("batch_size")):
        raise RuntimeError("Stage-1 config batch size differs from its clean source training manifest.")
    if int(args.update_epochs) != int(manifest.get("update_epochs")):
        raise RuntimeError("Stage-1 update_epochs differs from its clean source training manifest.")
    return checkpoint, actual_sha, manifest


def checkpoint_for_run(cli: argparse.Namespace) -> Path:
    checkpoint = missing_checkpoint_path(cli.model_save_dir, DATASET, cli.seed)
    if cli.smoke_test:
        checkpoint = checkpoint.parent / "smoke" / checkpoint.name
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    return checkpoint


def output_dir(cli: argparse.Namespace) -> Path:
    root = Path(cli.result_root) / "missing_baseline" / VERSION
    if cli.smoke_test:
        root = root / "smoke"
    return root / f"seed{cli.seed}"


def canonical_result_dir(cli: argparse.Namespace) -> Path:
    return Path(cli.result_root) / "missing_baseline" / "moddrop" / "train"


def prepare_outputs(cli: argparse.Namespace, checkpoint: Path) -> None:
    out = output_dir(cli)
    canonical_csv = canonical_result_dir(cli) / f"{DATASET}_per_seed.csv"
    existing_seed_row = False
    if canonical_csv.is_file() and not cli.smoke_test:
        frame = pd.read_csv(canonical_csv)
        if "Seed" in frame.columns:
            existing_seed_row = bool((frame.Seed.astype(int) == int(cli.seed)).any())
    if (checkpoint.exists() or out.exists() or existing_seed_row) and not cli.overwrite:
        raise FileExistsError(
            "Stage-1 output already exists for this seed. Inspect it, or pass --overwrite intentionally. "
            f"checkpoint={checkpoint} output={out} canonical_row={existing_seed_row}"
        )
    if cli.overwrite:
        if checkpoint.exists():
            checkpoint.unlink()
        if out.exists():
            import shutil
            shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def finite_metrics(metrics_by_mode: Dict[str, Dict[str, float]]) -> bool:
    return all(math.isfinite(float(value)) for metrics in metrics_by_mode.values() for value in metrics.values())


def train_one_seed(cli: argparse.Namespace, logger: logging.Logger) -> tuple[Dict[str, Any], Dict[str, Any]]:
    setup_seed(int(cli.seed))
    args = build_config(cli)
    clean_checkpoint, clean_sha, clean_manifest = verify_clean_source(cli, args)
    dataloader = MMDataLoader(args, int(cli.num_workers))
    if set(dataloader) != {"train", "valid"}:
        raise RuntimeError(f"Stage-1 must construct exactly train/valid, got {sorted(dataloader)}")

    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(clean_checkpoint, map_location="cpu"), strict=True)
    model = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    sim_loss = HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(cli.seed) + 104729)

    checkpoint = checkpoint_for_run(cli)
    prepare_outputs(cli, checkpoint)

    train_batches = len(dataloader["train"])
    updates_per_epoch = int(math.ceil(train_batches / float(args.update_epochs)))
    logger.info(
        "start version=%s dataset=mosei seed=%s clean=%s clean_sha=%s device=%s test_constructed=false",
        VERSION, cli.seed, clean_checkpoint, clean_sha, args.device,
    )
    logger.info(
        "config train_samples=%s valid_samples=%s batches_per_epoch=%s batch_size=%s update_epochs=%s "
        "optimizer_updates_per_epoch=%s lr=%s patience=%s early_stop=%s eta=1.0 memory_execution=sequential_equivalent_backward",
        len(dataloader["train"].dataset), len(dataloader["valid"].dataset), train_batches,
        args.batch_size, args.update_epochs, updates_per_epoch, args.learning_rate, args.patience, args.early_stop,
    )

    best_j_val = float("inf")
    best_epoch = 0
    best_metrics = None
    epoch = 0
    optimizer_steps_total = 0
    last_token_gradients = {"audio": 0.0, "vision": 0.0}

    while True:
        epoch += 1
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = {"full": 0.0, "missing": 0.0, "total": 0.0}
        sampled_counts = Counter({"LA": 0, "LV": 0, "L": 0})
        epoch_optimizer_steps = 0

        for step, batch_data in enumerate(dataloader["train"], start=1):
            text = batch_data["text"].to(args.device)
            audio = batch_data["audio"].to(args.device)
            vision = batch_data["vision"].to(args.device)
            labels = batch_data["labels"]["M"].to(args.device).view(-1, 1)

            full_mask = mode_to_mask("LAV", batch_size=labels.size(0), device=args.device, dtype=audio.dtype)
            full_output = model(text, audio, vision, full_mask)
            full_loss, full_details = compute_full_dlf_loss(full_output, labels, criterion, cosine, sim_loss)
            if not torch.isfinite(full_loss):
                raise FloatingPointError("NaN or Inf in Stage-1 full-view loss.")
            full_value = float(full_loss.detach().item())
            full_loss.backward()
            del full_output, full_details, full_loss

            missing_mask = sample_missing_masks(
                labels.size(0), missing_generator, device=args.device, dtype=audio.dtype
            )
            sampled_counts.update(count_missing_modes(missing_mask))
            missing_output = model(text, audio, vision, missing_mask)
            missing_loss, missing_details = compute_task_loss(missing_output, labels, criterion)
            if not torch.isfinite(missing_loss):
                raise FloatingPointError("NaN or Inf in Stage-1 missing-view loss.")
            missing_value = float(missing_loss.detach().item())
            (ETA * missing_loss).backward()
            del missing_output, missing_details, missing_loss

            if model.missing_audio_token.grad is not None:
                last_token_gradients["audio"] = float(model.missing_audio_token.grad.norm().item())
            if model.missing_vision_token.grad is not None:
                last_token_gradients["vision"] = float(model.missing_vision_token.grad.norm().item())

            if step % args.update_epochs == 0 or step == train_batches:
                if args.grad_clip != -1.0:
                    nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                epoch_optimizer_steps += 1
                optimizer_steps_total += 1

            totals["full"] += full_value
            totals["missing"] += missing_value
            totals["total"] += full_value + ETA * missing_value

        averaged = {name: value / train_batches for name, value in totals.items()}
        validation_metrics = evaluate_all_modes(model, dataloader["valid"], args.device, "moddrop", criterion)
        if not finite_metrics(validation_metrics):
            raise FloatingPointError("NaN or Inf in Stage-1 validation metrics.")
        j_val = float(validation_objective(validation_metrics))
        scheduler.step(j_val)
        current_lr = float(optimizer.param_groups[0]["lr"])

        logger.info(
            "seed=%s epoch=%s optimizer_steps_epoch=%s optimizer_steps_total=%s samples LA=%s LV=%s L=%s "
            "full_loss=%.6f missing_loss=%.6f total_loss=%.6f token_grad_audio=%.6g token_grad_vision=%.6g "
            "J_val=%.6f lr=%.8g",
            cli.seed, epoch, epoch_optimizer_steps, optimizer_steps_total,
            sampled_counts["LA"], sampled_counts["LV"], sampled_counts["L"],
            averaged["full"], averaged["missing"], averaged["total"],
            last_token_gradients["audio"], last_token_gradients["vision"], j_val, current_lr,
        )
        for mode, metrics in validation_metrics.items():
            logger.info(
                "seed=%s epoch=%s valid_%s acc7=%.4f acc5=%.4f acc2=%.4f F1=%.4f Corr=%.4f MAE=%.4f Loss=%.4f",
                cli.seed, epoch, mode, metrics["acc_7"], metrics["acc_5"], metrics["acc_2"],
                metrics["F1_score"], metrics["Corr"], metrics["MAE"], metrics["Loss"],
            )

        if j_val <= best_j_val - 1e-6:
            best_j_val = j_val
            best_epoch = epoch
            best_metrics = validation_metrics
            torch.save(model.state_dict(), checkpoint)
            logger.info("seed=%s epoch=%s saved validation-best checkpoint=%s", cli.seed, epoch, checkpoint)

        if cli.max_epochs is not None and epoch >= int(cli.max_epochs):
            break
        if epoch - best_epoch >= int(args.early_stop):
            break

    if best_metrics is None:
        raise RuntimeError("No validation-best Stage-1 checkpoint was saved.")
    if last_token_gradients["audio"] <= 0.0 or last_token_gradients["vision"] <= 0.0:
        raise RuntimeError("Missing tokens did not receive non-zero gradients.")

    # Release every GPU owner from training before constructing the strict audit copy.
    # Adam keeps parameter/state references even after ``del model``; ``backbone`` is
    # another explicit alias to the wrapped DLF.  Releasing all three prevents a
    # late audit-only OOM on the 8 GiB laptop GPU.
    del optimizer, scheduler
    del cosine, sim_loss
    del model, backbone
    torch.cuda.empty_cache()

    audit_backbone = DLF(args).to(args.device)
    audit_model = MissingModalityWrapper(
        audit_backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    audit_model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    audit_metrics = evaluate_all_modes(audit_model, dataloader["valid"], args.device, "moddrop", criterion)
    audit_j = float(validation_objective(audit_metrics))
    if abs(audit_j - best_j_val) > 1e-8:
        raise RuntimeError(f"Fresh reload audit failed: saved J={best_j_val}, reload J={audit_j}")

    row: Dict[str, Any] = {
        "Seed": int(cli.seed),
        "BestEpoch": int(best_epoch),
        "J_val": float(audit_j),
        "Checkpoint": str(checkpoint),
        "Stage1Version": VERSION,
        "MemoryExecution": "sequential_equivalent_backward",
        "OptimizerUpdatesPerEpoch": int(updates_per_epoch),
        "BatchSize": int(args.batch_size),
        "UpdateEpochs": int(args.update_epochs),
        "LearningRate": float(args.learning_rate),
        "Patience": int(args.patience),
        "EarlyStop": int(args.early_stop),
        "CleanCheckpointSHA256": clean_sha,
        "Stage1CheckpointSHA256": sha256(checkpoint),
        "TestConstructed": False,
    }
    row.update(flatten_mode_metrics(audit_metrics))

    context = {
        "args": args,
        "clean_checkpoint": clean_checkpoint,
        "clean_sha": clean_sha,
        "clean_manifest": clean_manifest,
        "checkpoint": checkpoint,
        "checkpoint_sha": row["Stage1CheckpointSHA256"],
        "best_epoch": int(best_epoch),
        "best_j": float(audit_j),
        "metrics": audit_metrics,
        "updates_per_epoch": int(updates_per_epoch),
        "optimizer_steps_total_at_stop": int(optimizer_steps_total),
    }
    return row, context


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def update_canonical_results(cli: argparse.Namespace, row: Dict[str, Any]) -> tuple[Path, Path]:
    result_dir = canonical_result_dir(cli)
    per_seed = result_dir / f"{DATASET}_per_seed.csv"
    summary = result_dir / f"{DATASET}_summary.csv"
    new = pd.DataFrame([row])
    if per_seed.is_file():
        old = pd.read_csv(per_seed)
        if "Seed" not in old.columns:
            raise RuntimeError(f"Existing canonical CSV has no Seed column: {per_seed}")
        old = old.loc[old.Seed.astype(int) != int(cli.seed)].copy()
        frame = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        frame = new
    frame = frame.sort_values("Seed", kind="mergesort").reset_index(drop=True)
    atomic_write_csv(frame, per_seed)
    numeric = frame.select_dtypes(include=["number"])
    summary_frame = pd.DataFrame({
        "Metric": numeric.columns,
        "Mean": numeric.mean().values,
        "Std": numeric.std(ddof=0).values,
    })
    atomic_write_csv(summary_frame, summary)
    return per_seed, summary


def write_manifest(
    cli: argparse.Namespace,
    row: Dict[str, Any],
    context: Dict[str, Any],
    log_path: Path,
    canonical_paths: tuple[Path, Path] | None,
) -> Path:
    out = output_dir(cli)
    per_seed, summary = canonical_paths if canonical_paths is not None else (None, None)
    args = context["args"]
    manifest = {
        "version": VERSION,
        "dataset": DATASET,
        "seed": int(cli.seed),
        "smoke_test": bool(cli.smoke_test),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_used_for_tuning": False,
        "eta": ETA,
        "checkpoint_selection": "min J_val = 0.5*MAE_LAV + 0.5*mean(MAE_LA,MAE_LV,MAE_L)",
        "memory_execution": "sequential_equivalent_backward",
        "memory_equivalence": "backward(L_full) + backward(L_missing) before any optimizer.step",
        "missing_mask_generator_seed": int(cli.seed) + 104729,
        "training": {
            "batch_size": int(args.batch_size),
            "update_epochs": int(args.update_epochs),
            "learning_rate": float(args.learning_rate),
            "patience": int(args.patience),
            "early_stop": int(args.early_stop),
            "max_epochs": cli.max_epochs,
            "num_workers": int(cli.num_workers),
            "matmul_precision": cli.matmul_precision,
            "optimizer_updates_per_epoch": int(context["updates_per_epoch"]),
            "optimizer_steps_total_at_stop": int(context["optimizer_steps_total_at_stop"]),
            "best_epoch": int(context["best_epoch"]),
            "best_J_val": float(context["best_j"]),
        },
        "source_clean_checkpoint": {
            "path": str(context["clean_checkpoint"]),
            "sha256": context["clean_sha"],
        },
        "stage1_checkpoint": {
            "path": str(context["checkpoint"]),
            "sha256": context["checkpoint_sha"],
        },
        "best_valid_metrics": context["metrics"],
        "canonical_results": None if cli.smoke_test else {
            "per_seed_csv": str(per_seed),
            "summary_csv": str(summary),
        },
        "environment": {
            "python_platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device),
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "completed_utc": utc_now(),
            "log": str(log_path),
        },
        "canonical_row": row,
    }
    path = out / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal MOSEI Stage-1 training.")
    torch.cuda.set_device(int(cli.gpu_id))
    torch.set_float32_matmul_precision(cli.matmul_precision)
    torch.backends.cudnn.allow_tf32 = cli.matmul_precision != "highest"

    logger, log_path = create_logger(cli)
    row, context = train_one_seed(cli, logger)
    canonical_paths = None if cli.smoke_test else update_canonical_results(cli, row)
    manifest_path = write_manifest(cli, row, context, log_path, canonical_paths)

    print("================ MOSEI ModDrop Stage-1 complete ====================")
    print("seed:", cli.seed)
    print("best_epoch:", row["BestEpoch"])
    print("J_val:", f"{row['J_val']:.9f}")
    print("checkpoint:", row["Checkpoint"])
    print("sha256:", row["Stage1CheckpointSHA256"])
    print("TEST CONSTRUCTED:", False)
    print("manifest:", manifest_path)
    if canonical_paths is not None:
        print("canonical per-seed CSV:", canonical_paths[0])
        print("canonical summary CSV:", canonical_paths[1])
    for mode in ("LAV", "LA", "LV", "L"):
        print(
            f"{mode}: MAE={row[f'{mode}_MAE']:.6f} Corr={row[f'{mode}_Corr']:.6f} "
            f"Acc2={row[f'{mode}_acc_2']:.6f} F1={row[f'{mode}_F1_score']:.6f} "
            f"Acc7={row[f'{mode}_acc_7']:.6f} Acc5={row[f'{mode}_acc_5']:.6f}"
        )


if __name__ == "__main__":
    main()
