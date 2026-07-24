"""Run the five MOSI folds sequentially with exactly one training worker."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from stage23a_mosi_common import N_FOLDS, RESULT_ROOT, atomic_json, git_head


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime" / "stage23a_mosi"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--num-workers", type=int, default=2)
    cli = parser.parse_args()
    if cli.num_workers != 2:
        raise RuntimeError("Frozen initial worker count is two.")
    RUNTIME.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    fold_records = []
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "2"
    for fold in range(N_FOLDS):
        fold_manifest = (
            RESULT_ROOT
            / "expert_oof"
            / "outer_fold{}".format(fold)
            / "fold_manifest.json"
        )
        if fold_manifest.is_file():
            fold_records.append(
                {"fold": fold, "status": "REUSED_COMPLETE", "returncode": 0}
            )
            continue
        command = [
            sys.executable,
            str(ROOT / "scripts" / "mosei" / "stage23a_mosi_train_fold.py"),
            "--outer-fold",
            str(fold),
            "--gpu-id",
            str(cli.gpu_id),
            "--max-epochs",
            "30",
            "--patience",
            "6",
            "--num-workers",
            str(cli.num_workers),
        ]
        print("Starting MOSI fold {}: {}".format(fold, " ".join(command)), flush=True)
        completed = subprocess.run(
            command,
            cwd=str(ROOT),
            env=environment,
            check=False,
        )
        fold_records.append(
            {
                "fold": fold,
                "status": "COMPLETED" if completed.returncode == 0 else "FAILED",
                "returncode": completed.returncode,
            }
        )
        if completed.returncode:
            break
    ended = datetime.now(timezone.utc)
    manifest = {
        "stage": "Stage23A-MOSI sequential fold coordinator",
        "status": (
            "COMPLETED"
            if len(fold_records) == N_FOLDS
            and all(value["returncode"] == 0 for value in fold_records)
            else "FAILED"
        ),
        "gpu_id": int(cli.gpu_id),
        "maximum_simultaneous_training_workers": 1,
        "OMP_NUM_THREADS": 2,
        "dataloader_workers": 2,
        "folds": fold_records,
        "started_at": started.isoformat(),
        "ended_at": ended.isoformat(),
        "wall_seconds": (ended - started).total_seconds(),
        "code_commit": git_head(),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
    }
    atomic_json(RESULT_ROOT / "runtime_manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    if manifest["status"] != "COMPLETED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
