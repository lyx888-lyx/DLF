#!/usr/bin/env python3
"""Resume v2 extraction, then hand off to the Stage23C-safe Phase-1 queue."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

from stage23d_self_risk_common import ROOT, RUNTIME, utc_now


def active_extract_queues():
    result = subprocess.run(
        ["pgrep", "-af", "stage23d_extract_queue.py"],
        capture_output=True,
        text=True,
        check=False,
    )
    return [
        row
        for row in result.stdout.splitlines()
        if "stage23d_extract_queue.py" in row and "pgrep" not in row
    ]


def run_logged(command, log_name):
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    path = RUNTIME / log_name
    with path.open("ab", buffering=0) as handle:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=environment,
            stdout=handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"Queue failed ({result.returncode}): {command}")


def main():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    while True:
        active = active_extract_queues()
        if not active:
            break
        print(
            f"[{utc_now()}] waiting for prior extraction queue: {len(active)}",
            flush=True,
        )
        time.sleep(30)
    print(f"[{utc_now()}] starting schema-v2 resume queue", flush=True)
    run_logged(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "mosei" / "stage23d_extract_queue.py"),
            "--gpu-id",
            "3",
            "--batch-size",
            "16",
            "--num-workers",
            "0",
        ],
        "extraction_v2_resume.log",
    )
    print(f"[{utc_now()}] schema-v2 extraction complete", flush=True)
    run_logged(
        [
            sys.executable,
            "-u",
            str(
                ROOT
                / "scripts"
                / "mosei"
                / "stage23d_validate_features.py"
            ),
        ],
        "feature_validation.log",
    )
    print(f"[{utc_now()}] schema-v2 integrity validation complete", flush=True)
    run_logged(
        [
            sys.executable,
            "-u",
            str(ROOT / "scripts" / "mosei" / "stage23d_phase1_queue.py"),
        ],
        "phase1_queue.log",
    )
    print(f"[{utc_now()}] Phase-1 queue complete", flush=True)


if __name__ == "__main__":
    main()
