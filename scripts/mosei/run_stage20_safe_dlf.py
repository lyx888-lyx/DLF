"""Unified train/official-valid-only runner for Stage 20 SAFE-DLF.

There is intentionally no test split, no test flag, and no import of a
test-evaluation entry point.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataset
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.mgrd_utils import (
    MISSING_MODES,
    VectorizedHingeLoss,
    canonical_ids,
    multigranular_kd_loss,
    ordered_id_sha,
    sha256_file,
)
from trains.singleTask.missing_utils import (
    MODALITY_MASKS,
    MissingModalityWrapper,
    compute_full_dlf_loss,
    flatten_mode_metrics,
    mode_to_mask,
    regression_metrics,
    sample_missing_masks,
    validation_objective,
)
from trains.singleTask.safe_dlf_utils import (
    METHODS,
    SafeMissingModalityWrapper,
    support_aligned_task_loss,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


ALLOWED_SPLITS = ("train", "valid")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=METHODS)
    parser.add_argument("--seed", required=True, type=int, choices=(1111, 1114))
    parser.add_argument("--resume-from")
    parser.add_argument("--screen-only", action="store_true")
    parser.add_argument("--promote-run", action="store_true")
    parser.add_argument("--amp-mode", required=True, choices=("off", "bf16", "fp16"))
    parser.add_argument("--max-epochs", required=True, type=int)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--artifact-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--frozen-protocol", required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parsed = parser.parse_args()
    if parsed.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    if parsed.max_train_batches < 0 or parsed.max_train_batches > 300:
        parser.error("--max-train-batches must be in [0, 300].")
    if parsed.screen_only and parsed.promote_run:
        parser.error("--screen-only and --promote-run are mutually exclusive.")
    if parsed.promote_run and not parsed.resume_from:
        parser.error("--promote-run requires --resume-from.")
    return parsed


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True).strip()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def build_args(cli):
    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(cli.seed)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def locked_dataset(args, split):
    if split not in ALLOWED_SPLITS:
        raise RuntimeError("Stage 20 runner permits train/official-valid only.")
    return MMDataset(args, mode=split)


def build_loaders(args, num_workers, sampler_generator):
    train = locked_dataset(args, "train")
    valid = locked_dataset(args, "valid")
    args.seq_lens = train.get_seq_len()
    return {
        "train": DataLoader(
            train,
            batch_size=args.batch_size,
            shuffle=True,
            generator=sampler_generator,
            drop_last=False,
            num_workers=num_workers,
        ),
        "valid": DataLoader(
            valid,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
        ),
    }


class PredictionCache:
    def __init__(self, cache_root, seed):
        root = Path(cache_root) / "seed{}".format(seed)
        self.aggregate_path = root / "cache_manifest.json"
        self.aggregate = json.loads(self.aggregate_path.read_text())
        if self.aggregate.get("locked_test_access_count") != 0:
            raise RuntimeError("Cache manifest test-lock mismatch.")
        self.manifest_sha = sha256_file(self.aggregate_path)
        self.entries = {}
        for split in ALLOWED_SPLITS:
            entry = self.aggregate["entries"][split]
            path = Path(entry["path"])
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError("Cache file SHA mismatch.")
            with np.load(path, allow_pickle=False) as archive:
                values = {key: archive[key].copy() for key in archive.files}
            ids = canonical_ids(values.pop("sample_id").tolist())
            if ordered_id_sha(ids) != entry["ordered_sample_id_sha256"]:
                raise RuntimeError("Cache ordered sample-ID SHA mismatch.")
            if len(ids) != int(entry["sample_count"]):
                raise RuntimeError("Cache sample count mismatch.")
            self.entries[split] = {
                "manifest": entry,
                "ids": ids,
                "by_id": {
                    sample_id: {
                        key: float(values[key][position])
                        for key in values
                    }
                    for position, sample_id in enumerate(ids)
                },
            }

    def validate_dataset(self, split, dataset):
        ids = canonical_ids(dataset.ids)
        entry = self.entries[split]
        if ordered_id_sha(ids) != entry["manifest"]["ordered_sample_id_sha256"]:
            raise RuntimeError("Dataset/cache sample-order SHA mismatch.")
        if ids != entry["ids"]:
            raise RuntimeError("Dataset/cache sample IDs differ.")

    def lookup(self, split, sample_ids, modes, device):
        sample_ids = canonical_ids(sample_ids)
        if len(sample_ids) != len(modes):
            raise ValueError("Cache lookup IDs and modes differ.")
        table = self.entries[split]["by_id"]
        missing = [sample_id for sample_id in sample_ids if sample_id not in table]
        if missing:
            raise RuntimeError("Missing cache sample IDs: {}".format(missing[:3]))
        teacher = torch.tensor(
            [table[sample_id]["teacher_lav"] for sample_id in sample_ids],
            dtype=torch.float32,
            device=device,
        )
        reference = torch.tensor(
            [table[sample_id]["moddrop_{}".format(mode)] for sample_id, mode in zip(sample_ids, modes)],
            dtype=torch.float32,
            device=device,
        )
        return teacher, reference


def mask_modes(mask):
    mapping = {(1, 1, 0): "LA", (1, 0, 1): "LV", (1, 0, 0): "L"}
    return [mapping[tuple(row)] for row in mask.detach().cpu().to(torch.int64).tolist()]


def float_outputs(output):
    return {key: value.float() if torch.is_tensor(value) and value.is_floating_point() else value for key, value in output.items()}


class FP32HingeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = VectorizedHingeLoss()

    def forward(self, ids, feats, margin=0.1):
        return self.base(ids.float(), feats.float(), margin)


class FP32CosineLoss(nn.Module):
    def forward(self, left, right, target):
        return nn.functional.cosine_embedding_loss(left.float(), right.float(), target.float())


def finite_metrics(metrics):
    return all(math.isfinite(float(value)) for mode in metrics.values() for value in mode.values())


def evaluate(model, loader, args, criterion, amp_mode):
    model.eval()
    collected = {mode: {"pred": [], "label": [], "loss": []} for mode in MODALITY_MASKS}
    enabled = amp_mode != "off"
    dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            for mode in MODALITY_MASKS:
                mask = mode_to_mask(mode, labels.size(0), args.device, audio.dtype)
                with torch.autocast(device_type="cuda", dtype=dtype, enabled=enabled):
                    prediction = model(text, audio, vision, mask)["output_logit"]
                prediction = prediction.float()
                collected[mode]["pred"].append(prediction.cpu())
                collected[mode]["label"].append(labels.cpu())
                collected[mode]["loss"].append(float(criterion(prediction, labels)))
    result = {}
    for mode, values in collected.items():
        prediction = torch.cat(values["pred"])
        labels = torch.cat(values["label"])
        result[mode] = regression_metrics(prediction, labels)
        result[mode]["Loss"] = float(np.mean(values["loss"]))
        result[mode]["PredictionStd"] = float(prediction.std(unbiased=False))
    return result


def tensor_grad_norm(tensors):
    total = sum(float(tensor.detach().float().pow(2).sum().cpu()) for tensor in tensors if tensor is not None)
    return total ** 0.5


def parameter_grad_norm(parameters):
    return tensor_grad_norm([parameter.grad for parameter in parameters if parameter.grad is not None])


def gradient_diagnostics(supervised, kd, parameters):
    supervised_grads = torch.autograd.grad(supervised, parameters, retain_graph=True, allow_unused=True)
    kd_grads = torch.autograd.grad(kd, parameters, retain_graph=True, allow_unused=True)
    dot = 0.0
    sup_sq = 0.0
    kd_sq = 0.0
    for left, right in zip(supervised_grads, kd_grads):
        if left is None or right is None:
            continue
        left = left.detach().float()
        right = right.detach().float()
        dot += float((left * right).sum().cpu())
        sup_sq += float(left.square().sum().cpu())
        kd_sq += float(right.square().sum().cpu())
    denominator = math.sqrt(sup_sq * kd_sq)
    return {
        "supervised_gradient_norm": math.sqrt(sup_sq),
        "kd_gradient_norm": math.sqrt(kd_sq),
        "gradient_cosine": dot / denominator if denominator else 0.0,
    }


def capture_rng(sampler_generator, missing_generator, shuffle_generator):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all(),
        "sampler_generator": sampler_generator.get_state(),
        "missing_generator": missing_generator.get_state(),
        "shuffle_generator": shuffle_generator.get_state(),
    }


def restore_rng(state, sampler_generator, missing_generator, shuffle_generator):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    torch.cuda.set_rng_state_all(state["torch_cuda"])
    sampler_generator.set_state(state["sampler_generator"])
    missing_generator.set_state(state["missing_generator"])
    shuffle_generator.set_state(state["shuffle_generator"])


def checkpoint_payload(
    cli,
    args,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch,
    global_optimizer_step,
    best,
    best_epoch,
    patience_counter,
    sampler_generator,
    missing_generator,
    shuffle_generator,
    cache,
    protocol,
):
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "epoch": int(epoch),
        "global_optimizer_step": int(global_optimizer_step),
        "microbatch_step": 0,
        "best_valid_metric": float(best),
        "best_epoch": int(best_epoch),
        "patience_counter": int(patience_counter),
        "rng_state": capture_rng(sampler_generator, missing_generator, shuffle_generator),
        "missing_schedule_seed": int(cli.seed) + 104729,
        "amp_mode": cli.amp_mode,
        "method_config": {
            "method": cli.method,
            "support_aligned_objective": True,
            "hard_availability_projection": cli.method == "safe_full",
            "lambda_kd": 1.0,
            "active_term_normalization": False,
        },
        "teacher_cache_manifest_sha": cache.manifest_sha,
        "moddrop_cache_manifest_sha": cache.manifest_sha,
        "dataset_sample_order_sha": cache.entries["train"]["manifest"]["ordered_sample_id_sha256"],
        "code_commit_sha": git_head(),
        "frozen_protocol_sha": sha256_file(cli.frozen_protocol),
        "batch_size": int(args.batch_size),
        "update_epochs": int(args.update_epochs),
        "locked_test_access_count": 0,
        "protocol": protocol,
    }


def save_resumable(payload, output, epoch, screen_epochs):
    checkpoint_dir = Path(output) / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    latest = checkpoint_dir / "latest_resumable.pt"
    previous = checkpoint_dir / "previous_resumable.pt"
    temporary = checkpoint_dir / "next_resumable.pt"
    atomic_torch_save(payload, temporary)
    if latest.exists():
        os.replace(str(latest), str(previous))
    os.replace(str(temporary), str(latest))
    with latest.open("rb") as handle:
        os.fsync(handle.fileno())
    if epoch in screen_epochs:
        screen = checkpoint_dir / "screen_epoch{:03d}.pt".format(epoch)
        if not screen.exists():
            shutil.copy2(latest, screen)
    atomic_json(
        checkpoint_dir / "latest_checkpoint.json",
        {"path": str(latest), "sha256": sha256_file(latest), "epoch": epoch, "created_at": utc_now()},
    )
    return latest


def load_checkpoint(path, cli, model, optimizer, scheduler, scaler, cache, protocol_sha):
    state = torch.load(path, map_location="cpu")
    expected = {
        "amp_mode": cli.amp_mode,
        "teacher_cache_manifest_sha": cache.manifest_sha,
        "moddrop_cache_manifest_sha": cache.manifest_sha,
        "frozen_protocol_sha": protocol_sha,
        "code_commit_sha": git_head(),
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise RuntimeError("Resume binding mismatch for {}.".format(key))
    if state["method_config"]["method"] != cli.method:
        raise RuntimeError("Resume method mismatch.")
    model.load_state_dict(state["model_state_dict"], strict=True)
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None:
        if state["grad_scaler_state_dict"] is None:
            raise RuntimeError("Resume scaler state absent.")
        scaler.load_state_dict(state["grad_scaler_state_dict"])
    elif state["grad_scaler_state_dict"] is not None:
        raise RuntimeError("Unexpected scaler state.")
    return state


def main():
    cli = parse_args()
    output = Path(cli.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(Path(cli.frozen_protocol).read_text())
    if cli.amp_mode != "off":
        raise RuntimeError("Stage 20 is frozen to FP32 (amp-mode=off).")
    if protocol["batch_size"] != 16 or protocol["update_epochs"] != 10:
        raise RuntimeError("Frozen batch/accumulation protocol mismatch.")
    if cli.screen_only and cli.max_epochs != int(protocol["screen_epoch_2"]):
        raise RuntimeError("Screen run must stop at frozen screen_epoch_2.")
    if cli.amp_mode == "bf16" and not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 autocast is unsupported on this GPU/runtime.")

    setup_seed(cli.seed)
    args = build_args(cli)
    if int(args.batch_size) != 16 or int(args.update_epochs) != 10:
        raise RuntimeError("Runtime config changed frozen batch/accumulation.")
    sampler_generator = torch.Generator().manual_seed(cli.seed)
    missing_generator = torch.Generator().manual_seed(cli.seed + 104729)
    shuffle_generator = torch.Generator().manual_seed(cli.seed + 314159)
    loaders = build_loaders(args, cli.num_workers, sampler_generator)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Stage 20 exposed a forbidden split.")
    cache = PredictionCache(cli.cache_root, cli.seed)
    cache.validate_dataset("train", loaders["train"].dataset)
    cache.validate_dataset("valid", loaders["valid"].dataset)

    clean_manifest_path = (
        Path(cli.artifact_root) / "clean_dlf" / "seed{}".format(cli.seed) / "stage_manifest.json"
    )
    clean_manifest = json.loads(clean_manifest_path.read_text())
    clean_checkpoint = Path(clean_manifest["Checkpoint"])
    if sha256_file(clean_checkpoint) != clean_manifest["CheckpointSHA256"]:
        raise RuntimeError("Student initialization checkpoint SHA mismatch.")
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(clean_checkpoint, map_location=args.device), strict=True)
    wrapper = (
        SafeMissingModalityWrapper
        if cli.method == "safe_full"
        else MissingModalityWrapper
    )
    student = wrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    scaler = torch.cuda.amp.GradScaler(enabled=cli.amp_mode == "fp16")
    scaler_object = scaler if cli.amp_mode == "fp16" else None
    criterion = nn.L1Loss()
    cosine = FP32CosineLoss()
    hinge = FP32HingeLoss()
    enabled = cli.amp_mode != "off"
    amp_dtype = torch.bfloat16 if cli.amp_mode == "bf16" else torch.float16

    epoch = 0
    global_optimizer_step = 0
    best = float("inf")
    best_epoch = 0
    best_metrics = None
    patience_counter = 0
    rows = []
    metrics_path = output / "epoch_metrics.csv"
    if metrics_path.exists():
        rows = pd.read_csv(metrics_path).to_dict("records")

    if cli.resume_from:
        resume = load_checkpoint(
            cli.resume_from,
            cli,
            student,
            optimizer,
            scheduler,
            scaler_object,
            cache,
            sha256_file(cli.frozen_protocol),
        )
        epoch = int(resume["epoch"])
        global_optimizer_step = int(resume["global_optimizer_step"])
        best = float(resume["best_valid_metric"])
        best_epoch = int(resume["best_epoch"])
        patience_counter = int(resume["patience_counter"])
        restore_rng(
            resume["rng_state"], sampler_generator, missing_generator, shuffle_generator
        )

    run_started = time.perf_counter()
    manifest = {
        "method": cli.method,
        "seed": cli.seed,
        "amp_mode": cli.amp_mode,
        "batch_size": int(args.batch_size),
        "update_epochs": int(args.update_epochs),
        "accumulation_semantics": "sum_not_mean",
        "learning_rate": float(args.learning_rate),
        "teacher_online_forward_count": 0,
        "reference_online_forward_count": 0,
        "cache_manifest_sha": cache.manifest_sha,
        "code_commit": git_head(),
        "locked_test_access_count": 0,
        "started_at": utc_now(),
        "resume_from": cli.resume_from,
    }
    atomic_json(output / "run_manifest.json", manifest)

    while epoch < cli.max_epochs:
        epoch += 1
        epoch_start = time.perf_counter()
        student.train()
        optimizer.zero_grad()
        totals = defaultdict(float)
        batches = 0
        samples = 0
        sampled = Counter({mode: 0 for mode in MISSING_MODES})
        epoch_gate = defaultdict(list)
        diagnostic = None
        last_grad_norm = 0.0

        for step, batch in enumerate(loaders["train"], 1):
            if cli.max_train_batches and step > cli.max_train_batches:
                break
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            missing_mask = sample_missing_masks(
                labels.size(0), missing_generator, args.device, audio.dtype
            )
            modes = mask_modes(missing_mask)
            sampled.update(modes)
            teacher_target, reference_target = cache.lookup(
                "train", batch["id"], modes, args.device
            )

            with torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=enabled
            ):
                full_output = student(text, audio, vision, full_mask)
                missing_output = student(text, audio, vision, missing_mask)
            full_loss, _ = compute_full_dlf_loss(
                float_outputs(full_output), labels.float(), criterion, cosine, hinge
            )
            missing_loss, support_details = support_aligned_task_loss(
                float_outputs(missing_output), labels.float(), missing_mask
            )
            kd_loss, kd_details = multigranular_kd_loss(
                "uniform_kd",
                missing_output["output_logit"],
                teacher_target,
                modes,
            )
            supervised_loss = full_loss + missing_loss
            total_loss = supervised_loss + kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in Stage 20 loss.")
            if step == 1:
                parameters = [parameter for parameter in student.parameters() if parameter.requires_grad]
                diagnostic = gradient_diagnostics(supervised_loss, kd_loss, parameters)

            if scaler_object is not None:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()
            should_step = step % args.update_epochs == 0 or step == len(loaders["train"]) or (
                cli.max_train_batches and step == cli.max_train_batches
            )
            if should_step:
                if scaler_object is not None:
                    scaler.unscale_(optimizer)
                last_grad_norm = parameter_grad_norm(student.parameters())
                nn.utils.clip_grad_value_(student.parameters(), args.grad_clip)
                if scaler_object is not None:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                global_optimizer_step += 1

            totals["full_loss"] += float(full_loss.detach().cpu())
            totals["missing_loss"] += float(missing_loss.detach().cpu())
            totals["kd_loss"] += float(kd_loss.detach().cpu())
            totals["total_loss"] += float(total_loss.detach().cpu())
            for key, value in support_details.items():
                totals["support_{}".format(key)] += float(value.detach().cpu())
            for key in ("L_reg", "L_Acc2", "L_Acc5", "L_Acc7", "weighted_reg", "weighted_ordinal"):
                totals[key] += float(kd_details[key])
            for key, stats in kd_details["gate"].items():
                epoch_gate[key].append(stats)
            batches += 1
            samples += int(labels.size(0))

        if batches == 0:
            raise RuntimeError("Empty train epoch.")
        valid = evaluate(student, loaders["valid"], args, criterion, cli.amp_mode)
        if not finite_metrics(valid):
            raise FloatingPointError("NaN/Inf in Stage 20 validation.")
        if any(valid[mode]["PredictionStd"] <= 1e-8 for mode in valid):
            raise RuntimeError("Prediction collapse detected.")
        j_valid = float(validation_objective(valid))
        scheduler.step(j_valid)
        improved = j_valid <= best - 1e-6
        if improved:
            best = j_valid
            best_epoch = epoch
            best_metrics = valid
            patience_counter = 0
            best_path = output / "checkpoints" / "best_epoch{:03d}.pth".format(epoch)
            atomic_torch_save(student.state_dict(), best_path)
            atomic_json(
                output / "checkpoints" / "best_checkpoint.json",
                {"path": str(best_path), "sha256": sha256_file(best_path), "epoch": epoch, "J": best},
            )
        else:
            patience_counter += 1

        averages = {key: value / batches for key, value in totals.items()}
        gate_summary = {}
        for key, values in epoch_gate.items():
            for statistic in ("mean", "std", "nonzero_rate", "ess"):
                gate_summary["gate_{}_{}".format(key, statistic)] = float(
                    np.mean([value[statistic] for value in values])
                )
        row = {
            "Epoch": epoch,
            "GlobalOptimizerStep": global_optimizer_step,
            "JValid": j_valid,
            "LearningRate": float(optimizer.param_groups[0]["lr"]),
            "TrainBatches": batches,
            "TrainSamples": samples,
            "SamplesPerSecond": samples / (time.perf_counter() - epoch_start),
            "EpochWallSeconds": time.perf_counter() - epoch_start,
            "PeakGPUMemoryBytes": int(torch.cuda.max_memory_allocated()),
            "GradientNormBeforeClip": last_grad_norm,
            "ScalerScale": float(scaler.get_scale()) if scaler_object is not None else 1.0,
            "SampledLA": sampled["LA"],
            "SampledLV": sampled["LV"],
            "SampledL": sampled["L"],
            **averages,
            **(diagnostic or {}),
            **flatten_mode_metrics(valid),
            **gate_summary,
        }
        rows.append(row)
        atomic_csv(pd.DataFrame(rows), metrics_path)

        payload = checkpoint_payload(
            cli,
            args,
            student,
            optimizer,
            scheduler,
            scaler_object,
            epoch,
            global_optimizer_step,
            best,
            best_epoch,
            patience_counter,
            sampler_generator,
            missing_generator,
            shuffle_generator,
            cache,
            protocol,
        )
        screen_epochs = tuple(
            int(protocol[key])
            for key in ("screen_epoch_1", "screen_epoch_2")
            if key in protocol
        )
        latest = save_resumable(
            payload,
            output,
            epoch,
            screen_epochs,
        )
        print(
            "stage20 method={} seed={} epoch={} J={:.6f} best={:.6f} wall={:.1f}s latest={}".format(
                cli.method, cli.seed, epoch, j_valid, best, row["EpochWallSeconds"], latest
            ),
            flush=True,
        )
        if patience_counter >= int(args.early_stop):
            break

    if best_metrics is None:
        pointer = output / "checkpoints" / "best_checkpoint.json"
        if pointer.exists():
            best_state = json.loads(pointer.read_text())
            best_epoch = int(best_state["epoch"])
            best = float(best_state["J"])
        else:
            raise RuntimeError("No validation-best model is available.")
    manifest.update(
        {
            "ended_at": utc_now(),
            "elapsed_seconds_this_invocation": time.perf_counter() - run_started,
            "completed_epoch": epoch,
            "global_optimizer_step": global_optimizer_step,
            "best_epoch": best_epoch,
            "best_J_valid": best,
            "patience_counter": patience_counter,
            "locked_test_access_count": 0,
            "status": "COMPLETED_TO_REQUESTED_BUDGET_OR_EARLY_STOP",
        }
    )
    atomic_json(output / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
