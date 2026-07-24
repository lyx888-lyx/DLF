"""Ten-minute automatic safety monitor; it can stop only the MOSI process group."""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from stage23a_mosi_common import RESULT_ROOT, atomic_json


MOSEI_ROOT = Path(
    "/code/DLF-mosei-arbiter-audit-v1/result/arbiter_audit_v1/mosei/expert_oof"
)


def metric_files(fold):
    return sorted(
        (MOSEI_ROOT / "outer_fold{}".format(fold)).glob(
            "**/epoch_metrics.csv"
        )
    )


def epoch_durations(path):
    frame = pd.read_csv(path)
    if not len(frame) or "wall_seconds" not in frame:
        return []
    values = frame.wall_seconds.astype(float).to_numpy()
    return np.diff(np.r_[0.0, values]).tolist()


def capture():
    result = {}
    for fold in (0, 1):
        files = metric_files(fold)
        if not files:
            raise RuntimeError("No MOSEI metrics for fold{}".format(fold))
        latest = max(files, key=lambda path: path.stat().st_mtime)
        durations = epoch_durations(latest)
        if not durations:
            raise RuntimeError("No MOSEI epoch durations for fold{}".format(fold))
        result[str(fold)] = {
            "baseline_file": str(latest),
            "baseline_seconds_per_epoch": float(np.median(durations[-3:])),
            "row_counts": {
                str(path): len(pd.read_csv(path)) for path in files
            },
        }
    return result


def new_durations(baseline, fold):
    values = []
    old_counts = baseline[str(fold)]["row_counts"]
    for path in metric_files(fold):
        durations = epoch_durations(path)
        start = int(old_counts.get(str(path), 0))
        values.extend(durations[start:])
    return values


def alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mosi-pid", type=int, required=True)
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--interval-seconds", type=int, default=60)
    cli = parser.parse_args()
    baseline = capture()
    snapshots = []
    unsafe = False
    for sample in range(cli.samples):
        time.sleep(cli.interval_seconds)
        snapshot = {
            "sample": sample + 1,
            "at": datetime.now(timezone.utc).isoformat(),
            "mosi_alive": alive(cli.mosi_pid),
            "folds": {},
        }
        for fold in (0, 1):
            values = new_durations(baseline, fold)
            reference = baseline[str(fold)]["baseline_seconds_per_epoch"]
            recent = float(np.mean(values[-2:])) if len(values) >= 2 else None
            ratio = recent / reference if recent is not None else None
            snapshot["folds"][str(fold)] = {
                "baseline_seconds_per_epoch": reference,
                "new_epoch_seconds": values,
                "recent_two_ratio": ratio,
            }
            if ratio is not None and ratio > 1.15:
                unsafe = True
        snapshots.append(snapshot)
        atomic_json(
            RESULT_ROOT / "parallel_safety_monitor_live.json",
            {
                "status": "UNSAFE_STOPPING_MOSI" if unsafe else "MONITORING",
                "baseline": baseline,
                "snapshots": snapshots,
                "locked_test_access_count": 0,
            },
        )
        if unsafe:
            # The launcher is started with setsid, so this process group contains
            # only the MOSI transfer launcher/coordinator/worker chain.
            try:
                os.killpg(cli.mosi_pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            break
    verdict = (
        "MOSI stopped: sustained primary MOSEI epoch slowdown exceeded 15%"
        if unsafe
        else "PASSED_10_MIN_NO_SUSTAINED_MOSEI_SLOWDOWN_GT_15_PERCENT"
    )
    result = {
        "status": "COMPLETED",
        "verdict": verdict,
        "unsafe": unsafe,
        "baseline": baseline,
        "snapshots": snapshots,
        "monitor_minutes": len(snapshots) * cli.interval_seconds / 60.0,
        "mosi_pid": cli.mosi_pid,
        "stopped_only_mosi_process_group": unsafe,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(RESULT_ROOT / "parallel_safety_monitor.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
