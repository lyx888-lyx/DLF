"""Bounded train-only Stage 19A microbatch profiler.

This script intentionally cannot construct validation or test loaders, save a
checkpoint, or update any Stage 10 artifact.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import sha256, stage_directory, utc_now
from scripts.mosei.stage10_train import build_args
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_for_modes,
    gated_kd_loss,
    modes_from_masks,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
    sample_missing_masks,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--warmup-batches", type=int, default=20)
    parser.add_argument("--max-train-batches", type=int, default=100)
    args = parser.parse_args()
    total = args.warmup_batches + args.max_train_batches
    if args.warmup_batches < 0 or not 1 <= args.max_train_batches <= 300:
        parser.error("Invalid bounded batch counts.")
    if total > 300:
        parser.error("Warmup plus timed batches must not exceed 300.")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible != str(args.physical_gpu):
        parser.error(
            "CUDA_VISIBLE_DEVICES must contain exactly --physical-gpu; got {!r}."
            .format(visible)
        )
    return args


def assert_runtime_safe(physical_gpu):
    process_text = subprocess.check_output(
        ["ps", "-eo", "pid=,args="], text=True
    )
    forbidden = (
        "stage10_worker.py",
        "stage10_supervisor.py",
        "stage10_train.py",
        "stage18",
    )
    offenders = [
        line.strip()
        for line in process_text.splitlines()
        if any(token in line for token in forbidden)
        and str(os.getpid()) not in line
    ]
    if offenders:
        raise RuntimeError("Active protected training process: " + repr(offenders))
    query = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            str(physical_gpu),
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    if query:
        raise RuntimeError("Selected physical GPU is occupied: " + query)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one CUDA device after masking.")


def timed(cuda, fn):
    cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000.0


def describe(values):
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean_ms": float(statistics.mean(values)),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
        "min_ms": float(min(values)),
        "max_ms": float(max(values)),
        "sum_ms": float(sum(values)),
    }


def main():
    args = parse_args()
    assert_runtime_safe(args.physical_gpu)
    setup_seed(args.seed)

    class Cli:
        seed = args.seed
        config_file = args.config_file
        gpu_id = 0

    runtime_args = build_args(Cli)
    if runtime_args.device.type != "cuda":
        raise RuntimeError("Profiler requires CUDA.")

    result_root = Path(args.result_root)
    clean_manifest = json.loads(
        (
            stage_directory(result_root, "clean", args.seed)
            / "stage_manifest.json"
        ).read_text()
    )
    cache_manifest = json.loads(
        (
            stage_directory(result_root, "compatibility", args.seed)
            / "stage_manifest.json"
        ).read_text()
    )
    clean = Path(clean_manifest["Checkpoint"])
    cache_path = Path(cache_manifest["Cache"])
    if sha256(clean) != clean_manifest["CheckpointSHA256"]:
        raise RuntimeError("Clean checkpoint SHA mismatch.")
    if sha256(cache_path) != cache_manifest["CacheSHA256"]:
        raise RuntimeError("Compatibility cache SHA mismatch.")
    cache = pd.read_csv(cache_path)
    if cache.sample_index.duplicated().any():
        raise RuntimeError("Compatibility cache contains duplicate indices.")
    cache_by_index = cache.set_index("sample_index").to_dict("index")

    # This is the only loader the script can construct.
    train_loader = build_single_split_loader(
        runtime_args, "train", args.num_workers
    )
    teacher = build_frozen_teacher(DLF, runtime_args, clean)
    backbone = DLF(runtime_args).to(runtime_args.device)
    backbone.load_state_dict(
        torch.load(clean, map_location=runtime_args.device), strict=True
    )
    student = MissingModalityWrapper(
        backbone, runtime_args.feature_dims[1], runtime_args.feature_dims[2]
    ).to(runtime_args.device)
    student.train()
    optimizer = optim.Adam(student.parameters(), lr=runtime_args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    class TimedHingeLoss(HingeLoss):
        def __init__(self):
            super().__init__()
            self.active = False
            self.measurements = []

        def forward(self, ids, feats, margin=0.1):
            if not self.active:
                return super().forward(ids, feats, margin)
            torch.cuda.synchronize()
            start = time.perf_counter()
            result = super().forward(ids, feats, margin)
            torch.cuda.synchronize()
            self.measurements.append((time.perf_counter() - start) * 1000.0)
            return result

    hinge = TimedHingeLoss()
    generator = torch.Generator().manual_seed(int(args.seed) + 104729)

    bert_counts = {"student": 0, "teacher": 0}

    def count_student(_module, _inputs, _output):
        bert_counts["student"] += 1

    def count_teacher(_module, _inputs, _output):
        bert_counts["teacher"] += 1

    handles = [
        student.backbone.text_model.register_forward_hook(count_student),
        teacher.text_model.register_forward_hook(count_teacher),
    ]
    optimizer.zero_grad()
    iterator = iter(train_loader)
    timings = defaultdict(list)
    total_target = args.warmup_batches + args.max_train_batches
    timed_start = None
    timed_samples = 0
    timed_steps = 0
    torch.cuda.reset_peak_memory_stats()

    for batch_number in range(1, total_target + 1):
        data_start = time.perf_counter()
        batch = next(iterator)
        data_wait_ms = (time.perf_counter() - data_start) * 1000.0
        measured = batch_number > args.warmup_batches
        if batch_number == args.warmup_batches + 1:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            bert_counts = {"student": 0, "teacher": 0}
            hinge.active = True
            timed_start = time.perf_counter()

        def move():
            return (
                batch["text"].to(runtime_args.device),
                batch["audio"].to(runtime_args.device),
                batch["vision"].to(runtime_args.device),
                batch["labels"]["M"].to(runtime_args.device).view(-1, 1),
            )

        (text, audio, vision, labels), h2d_ms = timed(torch.cuda, move)
        full_mask = mode_to_mask(
            "LAV", labels.size(0), runtime_args.device, audio.dtype
        )
        full_output, student_lav_ms = timed(
            torch.cuda, lambda: student(text, audio, vision, full_mask)
        )
        (full_loss, _), full_loss_ms = timed(
            torch.cuda,
            lambda: compute_full_dlf_loss(
                full_output, labels, criterion, cosine, hinge
            ),
        )
        missing_mask = sample_missing_masks(
            labels.size(0), generator, runtime_args.device, audio.dtype
        )
        modes = modes_from_masks(missing_mask)
        missing_output, missing_forward_ms = timed(
            torch.cuda, lambda: student(text, audio, vision, missing_mask)
        )
        (missing_loss, _), missing_loss_ms = timed(
            torch.cuda,
            lambda: compute_task_loss(missing_output, labels, criterion),
        )
        teacher_prediction, teacher_ms = timed(
            torch.cuda,
            lambda: teacher_lav_prediction(teacher, text, audio, vision),
        )

        def make_kd():
            indices = batch["index"].view(-1).numpy().astype(int).tolist()
            compatibility = compatibility_for_modes(
                cache_by_index,
                indices,
                modes,
                runtime_args.device,
                labels.dtype,
            )
            return gated_kd_loss(
                missing_output["output_logit"],
                teacher_prediction,
                compatibility,
            )[0]

        kd_loss, kd_ms = timed(torch.cuda, make_kd)
        loss = full_loss + missing_loss + kd_loss
        _, backward_ms = timed(torch.cuda, loss.backward)
        if teacher_grad_count(teacher):
            raise RuntimeError("Frozen teacher received gradients.")
        step_ms = 0.0
        if batch_number % runtime_args.update_epochs == 0:
            def step_optimizer():
                nn.utils.clip_grad_value_(
                    student.parameters(), runtime_args.grad_clip
                )
                optimizer.step()
                optimizer.zero_grad()

            _, step_ms = timed(torch.cuda, step_optimizer)
            if measured:
                timed_steps += 1

        if measured:
            timed_samples += int(labels.size(0))
            for name, value in {
                "data_wait": data_wait_ms,
                "host_to_device": h2d_ms,
                "student_lav_forward": student_lav_ms,
                "full_loss": full_loss_ms,
                "student_missing_forward": missing_forward_ms,
                "missing_loss": missing_loss_ms,
                "teacher_lav_forward": teacher_ms,
                "compatibility_and_kd": kd_ms,
                "backward": backward_ms,
                "optimizer_step": step_ms,
            }.items():
                timings[name].append(value)

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - timed_start
    for handle in handles:
        handle.remove()
    expected = {
        "student": 2 * args.max_train_batches,
        "teacher": args.max_train_batches,
    }
    if bert_counts != expected:
        raise RuntimeError(
            "Unexpected timed BERT forward counts: {} != {}".format(
                bert_counts, expected
            )
        )
    if len(hinge.measurements) != args.max_train_batches:
        raise RuntimeError("HingeLoss timing count differs from timed batches.")
    timings["hinge_similarity_loss"] = hinge.measurements
    output = {
        "Audit": "Stage 19A MOSEI Training Runtime Forensic Audit v1",
        "Status": "COMPLETED_BOUNDED_TRAIN_ONLY_PROFILE",
        "CreatedAt": utc_now(),
        "NoTestAccess": True,
        "NoValidationAccess": True,
        "NoCheckpointWrite": True,
        "PhysicalGPU": args.physical_gpu,
        "VisibleCUDADevices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "InternalCUDADevice": 0,
        "GPUName": torch.cuda.get_device_name(0),
        "Seed": args.seed,
        "WarmupBatches": args.warmup_batches,
        "TimedTrainBatches": args.max_train_batches,
        "TimedSamples": timed_samples,
        "OptimizerSteps": timed_steps,
        "UpdateEpochs": int(runtime_args.update_epochs),
        "BatchSize": int(runtime_args.batch_size),
        "ElapsedSeconds": elapsed,
        "BatchesPerSecond": args.max_train_batches / elapsed,
        "SamplesPerSecond": timed_samples / elapsed,
        "BERTForwardCounts": bert_counts,
        "ExpectedBERTForwardCounts": expected,
        "PeakAllocatedBytes": int(torch.cuda.max_memory_allocated()),
        "PeakReservedBytes": int(torch.cuda.max_memory_reserved()),
        "Timing": {
            name: describe(values) for name, values in sorted(timings.items())
        },
        "Inputs": {
            "CleanCheckpoint": str(clean),
            "CleanCheckpointSHA256": clean_manifest["CheckpointSHA256"],
            "CompatibilityCache": str(cache_path),
            "CompatibilityCacheSHA256": cache_manifest["CacheSHA256"],
            "ConfigFile": str(Path(args.config_file).resolve()),
        },
        "MethodSemantics": {
            "StudentForwardsPerMicrobatch": 2,
            "FrozenTeacherForwardsPerMicrobatch": 1,
            "EvaluatorForwardsPerMicrobatch": 0,
            "BackwardsPerMicrobatch": 1,
            "OneRandomMissingViewPerMicrobatch": True,
            "LossDividedByAccumulationFactor": False,
        },
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
