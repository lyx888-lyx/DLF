"""Combined read-only MOSI transfer and primary MOSEI progress display."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from stage23a_mosi_common import EXPERTS, N_FOLDS, RESULT_ROOT


ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "runtime" / "stage23a_mosi"
COMPONENTS = (
    ("clean_seed1111", "components/clean_seed1111"),
    ("clean_seed1114", "components/clean_seed1114"),
    *[(value, "experts/{}".format(value)) for value in EXPERTS],
)


def duration(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        "{}h {:02d}m".format(hours, minutes)
        if hours
        else "{}m {:02d}s".format(minutes, seconds)
    )


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def main():
    pid_path = RUNTIME / "coordinator.pid"
    pid = int(pid_path.read_text()) if pid_path.is_file() else None
    completed = 0
    current = "not_started"
    current_epoch = 0
    epoch_times = []
    found_current = False
    for fold in range(N_FOLDS):
        fold_root = RESULT_ROOT / "expert_oof" / "outer_fold{}".format(fold)
        for name, relative in COMPONENTS:
            directory = fold_root / relative
            manifest = directory / "run_manifest.json"
            metrics = directory / "epoch_metrics.csv"
            if manifest.is_file():
                completed += 1
                frame = pd.read_csv(metrics)
                if len(frame):
                    epoch_times.append(float(frame.wall_seconds.iloc[-1]) / len(frame))
            elif metrics.is_file():
                frame = pd.read_csv(metrics)
                current = "fold{}:{}".format(fold, name)
                current_epoch = len(frame)
                if len(frame):
                    epoch_times.append(float(frame.wall_seconds.iloc[-1]) / len(frame))
                found_current = True
                break
            else:
                current = "fold{}:{}".format(fold, name)
                found_current = True
                break
        if found_current and not (fold_root / "fold_manifest.json").is_file():
            break
    mean_epoch = float(np.median(epoch_times)) if epoch_times else None
    expected_epochs = 12
    remaining = max(0, 35 - completed - 1) * expected_epochs + max(
        0, expected_epochs - current_epoch
    )
    eta = remaining * mean_epoch if mean_epoch is not None else None
    print("Stage23A-MOSI: {} pid={} | components={}/35 | {} epoch={} | ETA={}".format(
        "RUNNING" if alive(pid) else "STOPPED/COMPLETE",
        pid,
        completed,
        current,
        current_epoch,
        duration(eta),
    ))
    speed = RESULT_ROOT / "speed_probe" / "speed_probe_manifest.json"
    if speed.is_file():
        value = json.loads(speed.read_text())
        print(
            "MOSI speed probe: {:.2f}s/epoch, peak GPU {:.2f}GiB".format(
                value["observed_seconds_per_epoch"],
                value["peak_gpu_memory_bytes"] / (1024 ** 3),
            )
        )
    monitor = RESULT_ROOT / "parallel_safety_monitor.json"
    if monitor.is_file():
        value = json.loads(monitor.read_text())
        waiver = RESULT_ROOT / "parallel_resource_user_waiver.json"
        if waiver.is_file():
            print(
                "Parallel safety: USER_WAIVED_SLOWDOWN_STOP "
                "(original: {})".format(value.get("verdict", value.get("status")))
            )
        else:
            print("Parallel safety: {}".format(value.get("verdict", value.get("status"))))
    try:
        output = subprocess.check_output(
            [
                "/usr/miniconda3/envs/DLF/bin/python",
                "/code/DLF-mosei-arbiter-audit-v1/scripts/mosei/stage23a_status.py",
            ],
            text=True,
        )
        print("\nPrimary MOSEI:\n" + output)
    except Exception as error:
        print("MOSEI status unavailable: {}".format(error))


if __name__ == "__main__":
    main()
