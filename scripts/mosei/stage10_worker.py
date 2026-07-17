"""Sequential GPU worker for validation-only Stage 10 training."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.mosei.stage10_common import (
    STAGES,
    append_jsonl,
    atomic_json,
    git_head,
    sentinel_path,
    stage_manifest_path,
    utc_now,
    validate_sentinel,
    validate_stage_manifest,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--physical-gpu", required=True, type=int)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--commit", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    result_root = Path(args.result_root).resolve()
    if git_head(ROOT) != args.commit:
        raise RuntimeError("Worker commit differs from launch commit.")
    completed_path = run_dir / "completed_stages.json"
    completed = (
        json.loads(completed_path.read_text()) if completed_path.is_file() else {}
    )
    for seed in args.seeds:
        for stage in STAGES:
            done = sentinel_path(run_dir, seed, stage)
            existing = validate_sentinel(
                done, result_root, seed, stage, args.commit
            )
            if existing:
                continue
            manifest_path = stage_manifest_path(result_root, stage, seed)
            if manifest_path.parent.exists() and any(manifest_path.parent.iterdir()):
                raise RuntimeError(
                    "Incomplete output blocks safe resume: {}".format(
                        manifest_path.parent
                    )
                )
            command = [
                sys.executable,
                str(ROOT / "scripts/mosei/stage10_train.py"),
                "--stage",
                stage,
                "--dataset",
                "mosei",
                "--seed",
                str(seed),
                "--result-root",
                str(result_root),
                "--config-file",
                str(Path(args.config_file).resolve()),
                "--gpu-id",
                "0",
                "--num-workers",
                "1",
            ]
            start = utc_now()
            record = {
                "Type": "stage",
                "Seed": seed,
                "Stage": stage,
                "PhysicalGPU": args.physical_gpu,
                "Command": command,
                "StartedAt": start,
            }
            append_jsonl(run_dir / "commands.jsonl", record)
            print(
                "START seed={} stage={} physical_gpu={}".format(
                    seed, stage, args.physical_gpu
                ),
                flush=True,
            )
            environment = os.environ.copy()
            environment["CUDA_VISIBLE_DEVICES"] = str(args.physical_gpu)
            completed_process = subprocess.run(
                command,
                cwd=str(ROOT),
                env=environment,
                stdin=subprocess.DEVNULL,
            )
            end = utc_now()
            if completed_process.returncode:
                raise RuntimeError(
                    "seed={} stage={} exited {}.".format(
                        seed, stage, completed_process.returncode
                    )
                )
            manifest = validate_stage_manifest(
                manifest_path, seed, stage, args.commit
            )
            sentinel = {
                "Seed": seed,
                "Stage": stage,
                "Command": command,
                "ExitCode": 0,
                "StartedAt": start,
                "EndedAt": end,
                "Commit": args.commit,
                "PhysicalGPU": args.physical_gpu,
                "Outputs": manifest["Outputs"],
                "ValidationMetrics": manifest["ValidationMetrics"],
                "Manifest": str(manifest_path),
            }
            atomic_json(done, sentinel)
            completed["seed_{}".format(seed)] = stage
            atomic_json(completed_path, completed)
            print("DONE seed={} stage={}".format(seed, stage), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        arguments = parse_args()
        append_jsonl(
            Path(arguments.run_dir) / "failures.jsonl",
            {
                "Component": "worker",
                "PhysicalGPU": arguments.physical_gpu,
                "Error": repr(error),
                "Traceback": traceback.format_exc(),
                "At": utc_now(),
            },
        )
        raise
