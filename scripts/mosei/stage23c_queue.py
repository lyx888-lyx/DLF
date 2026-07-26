#!/usr/bin/env python3
"""Resumable fixed Stage23C GPU queues.

The queue only orchestrates already-authorized ``stage23c_train.py`` calls.
Selection, data access, and model behavior remain in that audited entry point.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result" / "oracle_distillation_v1" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23c"
PYTHON = Path(sys.executable)
TRAIN = ROOT / "scripts" / "mosei" / "stage23c_train.py"

METHODS = {
    "S0": "S0_supervised_moddrop",
    "S1": "S1_uniform_single_teacher_kd",
    "S2": "S2_equal_ensemble_final_kd",
    "S3": "S3_fixed_stacking_final_kd",
    "S4": "S4_soft_oracle_final_kd",
    "S5": "S5_soft_oracle_hierarchical_kd",
}

QUEUES = {
    "pre0": [
        ("A", "S0", None),
        ("A", "S3", None),
        ("A", "S5", 0.25),
        ("B", "S4", None),
        ("B", "S5", 0.25),
    ],
    "pre1": [
        ("B", "S3", None),
        ("B", "S5", 0.5),
        ("A", "S2", None),
        ("A", "S5", 1.0),
    ],
    "pre2": [
        ("B", "S2", None),
        ("B", "S5", 1.0),
        ("A", "S1", None),
        ("A", "S4", None),
        ("A", "S5", 0.5),
    ],
}

PREREQUISITES = {
    "pre0": OUT / "training" / "direction_A" / "screen" / "clean"
    / "run_manifest.json",
    "pre1": OUT / "training" / "direction_B" / "screen" / "S0"
    / "run_manifest.json",
    "pre2": OUT / "training" / "direction_B" / "screen" / "S1"
    / "run_manifest.json",
}


def run_id(short_name, ratio):
    if short_name == "S5":
        return f"S5_hier_ratio{str(ratio).replace('.', 'p')}"
    return short_name


def completed(path):
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8"))["status"] == "COMPLETED"
    except (KeyError, json.JSONDecodeError):
        return False


def wait_for(path):
    while not completed(path):
        print(f"waiting for {path}", flush=True)
        time.sleep(30)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=sorted(QUEUES), required=True)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2), required=True)
    cli = parser.parse_args()
    RUNTIME.mkdir(parents=True, exist_ok=True)
    wait_for(PREREQUISITES[cli.worker])
    for direction, short_name, ratio in QUEUES[cli.worker]:
        identifier = run_id(short_name, ratio)
        manifest = (
            OUT
            / "training"
            / f"direction_{direction}"
            / "screen"
            / identifier
            / "run_manifest.json"
        )
        if completed(manifest):
            print(f"skip completed direction={direction} run={identifier}", flush=True)
            continue
        command = [
            str(PYTHON),
            "-u",
            str(TRAIN),
            "--phase",
            "student-screen",
            "--direction",
            direction,
            "--method",
            METHODS[short_name],
            "--gpu-id",
            str(cli.gpu_id),
            "--num-workers",
            "1",
        ]
        if ratio is not None:
            command.extend(["--hier-ratio", str(ratio)])
        log_path = RUNTIME / f"screen_{direction}_{identifier}.log"
        print(f"start direction={direction} run={identifier}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode != 0 or not completed(manifest):
            raise RuntimeError(
                f"direction={direction} run={identifier} failed; see {log_path}"
            )
        print(f"complete direction={direction} run={identifier}", flush=True)
    print(f"worker {cli.worker} complete", flush=True)


if __name__ == "__main__":
    main()
