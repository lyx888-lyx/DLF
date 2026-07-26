#!/usr/bin/env python3
"""Resumable single-GPU extraction queue for the ten frozen Expert audits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from stage23d_self_risk_common import (
    EXPERTS,
    FOLDS,
    OUT,
    RUNTIME,
    atomic_json,
    utc_now,
)


ROOT = Path(__file__).resolve().parents[2]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, required=True, choices=(0, 1, 2, 3))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def manifest_path(fold, expert):
    return (
        OUT
        / "features"
        / f"checkpoint_fold{fold}"
        / expert
        / "extraction_manifest.json"
    )


def is_complete(path):
    try:
        payload = json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    return (
        payload.get("status") == "COMPLETED"
        and payload.get("duplicates") == 0
        and payload.get("missing") == 0
        and payload.get("nan_inf_count") == 0
        and payload.get("inactive_head_leakage") == 0
        and payload.get("official_valid_access_count") == 0
        and payload.get("locked_test_access_count") == 0
    )


def gpu_processes(gpu_id):
    query = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-compute-apps=pid,process_name",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in query.stdout.splitlines() if line.strip()]


def write_state(status, current=None, error=None):
    completed = []
    pending = []
    for fold in FOLDS:
        for expert in EXPERTS:
            key = f"fold{fold}/{expert}"
            (completed if is_complete(manifest_path(fold, expert)) else pending).append(
                key
            )
    atomic_json(
        RUNTIME / "extraction_queue_state.json",
        {
            "stage": "Stage23D-A frozen Expert feature extraction",
            "status": status,
            "current": current,
            "completed": completed,
            "pending": pending,
            "error": error,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "updated_at": utc_now(),
        },
    )


def main():
    cli = parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    write_state("RUNNING")
    for fold in FOLDS:
        for expert in EXPERTS:
            target = manifest_path(fold, expert)
            key = f"fold{fold}/{expert}"
            if is_complete(target):
                print(f"[{utc_now()}] SKIP complete {key}", flush=True)
                continue
            # The queue owns no GPU process while between checkpoints. Refuse to
            # enter an occupied GPU instead of competing with an unrelated job.
            occupied = gpu_processes(cli.gpu_id)
            if occupied:
                error = f"GPU {cli.gpu_id} occupied before {key}: {occupied}"
                write_state("PAUSED_GPU_OCCUPIED", key, error)
                raise RuntimeError(error)
            write_state("RUNNING", key)
            print(f"[{utc_now()}] START {key} on GPU {cli.gpu_id}", flush=True)
            environment = os.environ.copy()
            environment.update(
                {
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "NUMEXPR_NUM_THREADS": "1",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            command = [
                sys.executable,
                "-u",
                str(ROOT / "scripts" / "mosei" / "stage23d_extract.py"),
                "--checkpoint-fold",
                str(fold),
                "--expert-id",
                expert,
                "--gpu-id",
                str(cli.gpu_id),
                "--batch-size",
                str(cli.batch_size),
                "--num-workers",
                str(cli.num_workers),
            ]
            started = time.monotonic()
            result = subprocess.run(command, cwd=ROOT, env=environment, check=False)
            if result.returncode != 0 or not is_complete(target):
                error = (
                    f"{key} extraction failed with return code "
                    f"{result.returncode}"
                )
                write_state("FAILED", key, error)
                raise RuntimeError(error)
            print(
                f"[{utc_now()}] DONE {key} elapsed={time.monotonic()-started:.1f}s",
                flush=True,
            )
    write_state("COMPLETED")
    print(f"[{utc_now()}] ALL 10 EXTRACTIONS COMPLETED", flush=True)


if __name__ == "__main__":
    main()
