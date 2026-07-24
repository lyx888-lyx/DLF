"""Read-only progress and ETA display for the asynchronous Stage 23A folds."""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "result" / "arbiter_audit_v1" / "mosei" / "expert_oof"
RUNTIME = ROOT / "runtime" / "stage23a"
MAX_EPOCHS = 30
PATIENCE = 6
COMPONENTS = (
    ("clean_seed1111", "components/clean_seed1111"),
    ("clean_seed1114", "components/clean_seed1114"),
    ("moddrop_seed1111", "experts/moddrop_seed1111"),
    ("moddrop_seed1114", "experts/moddrop_seed1114"),
    ("uniform_kd_seed1111", "experts/uniform_kd_seed1111"),
    ("cfcompat_seed1111", "experts/cfcompat_seed1111"),
    ("cfcompat_seed1114", "experts/cfcompat_seed1114"),
)


def duration(seconds):
    if seconds is None or not np.isfinite(seconds):
        return "unknown"
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return "{}h {:02d}m".format(hours, minutes)
    return "{}m {:02d}s".format(minutes, seconds)


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def component_state(fold_root, relative):
    directory = fold_root / relative
    metrics_path = directory / "epoch_metrics.csv"
    manifest_path = directory / "run_manifest.json"
    epochs = 0
    best_epoch = 0
    seconds_per_epoch = None
    last_j = None
    if metrics_path.is_file():
        frame = pd.read_csv(metrics_path)
        epochs = len(frame)
        if epochs:
            best_epoch = int(frame.inner_J.astype(float).idxmin()) + 1
            last_j = float(frame.iloc[-1].inner_J)
            if "wall_seconds" in frame:
                seconds_per_epoch = float(frame.iloc[-1].wall_seconds) / epochs
    completed = False
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        completed = manifest.get("status") == "COMPLETED"
        best_epoch = int(manifest.get("best_epoch", best_epoch))
    return {
        "completed": completed,
        "epochs": epochs,
        "best_epoch": best_epoch,
        "seconds_per_epoch": seconds_per_epoch,
        "last_j": last_j,
    }


def fold_status(fold):
    root = RESULT / "outer_fold{}".format(fold)
    states = [
        (name, relative, component_state(root, relative))
        for name, relative in COMPONENTS
    ]
    completed = sum(int(state["completed"]) for _, _, state in states)
    current = next(
        ((name, state) for name, _, state in states if not state["completed"]),
        ("complete", {"epochs": 0, "best_epoch": 0, "last_j": None}),
    )
    observed_epoch_times = [
        state["seconds_per_epoch"]
        for _, _, state in states
        if state["seconds_per_epoch"] is not None
    ]
    seconds_per_epoch = (
        float(np.median(observed_epoch_times)) if observed_epoch_times else None
    )
    completed_epochs = [
        state["epochs"] for _, _, state in states if state["completed"]
    ]
    # Before a component completes, 12 epochs is a neutral estimate: at least
    # patience=6 and below the 30-epoch hard cap.  Once evidence accumulates,
    # use the median completed-component length.
    expected_epochs = (
        int(round(float(np.median(completed_epochs))))
        if completed_epochs
        else 12
    )
    expected_epochs = min(MAX_EPOCHS, max(PATIENCE, expected_epochs))
    current_epochs = int(current[1]["epochs"])
    remaining_components = len(COMPONENTS) - completed - 1
    estimated_remaining_epochs = max(0, expected_epochs - current_epochs) + max(
        0, remaining_components
    ) * expected_epochs
    maximum_remaining_epochs = max(0, MAX_EPOCHS - current_epochs) + max(
        0, remaining_components
    ) * MAX_EPOCHS
    eta = (
        estimated_remaining_epochs * seconds_per_epoch
        if seconds_per_epoch is not None
        else None
    )
    max_eta = (
        maximum_remaining_epochs * seconds_per_epoch
        if seconds_per_epoch is not None
        else None
    )
    progress = (
        completed + min(current_epochs / max(expected_epochs, 1), 0.99)
    ) / len(COMPONENTS)
    pid_path = RUNTIME / "fold{}.pid".format(fold)
    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    manifest_complete = (root / "fold_manifest.json").is_file()
    if manifest_complete:
        progress, eta, max_eta = 1.0, 0.0, 0.0
    return {
        "fold": fold,
        "pid": pid,
        "alive": process_alive(pid),
        "completed_components": completed,
        "current_component": current[0],
        "current_epochs": current_epochs,
        "current_best_epoch": int(current[1]["best_epoch"]),
        "current_last_j": current[1]["last_j"],
        "seconds_per_epoch": seconds_per_epoch,
        "expected_component_epochs": expected_epochs,
        "progress": progress,
        "eta": eta,
        "max_eta": max_eta,
        "manifest_complete": manifest_complete,
    }


def gpu_lines():
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.used,utilization.gpu",
                "--format=csv,noheader",
            ],
            text=True,
        )
        return [line.strip() for line in output.splitlines()]
    except Exception as error:
        return ["nvidia-smi unavailable: {}".format(error)]


def main():
    statuses = [fold_status(fold) for fold in (0, 1)]
    print("Stage 23A Expert OOF progress (ETA is empirical; early stopping may shorten it)")
    print("=" * 86)
    for status in statuses:
        j_text = (
            "{:.6f}".format(status["current_last_j"])
            if status["current_last_j"] is not None
            else "-"
        )
        print(
            "fold{fold}: {state} pid={pid} | {done}/7 components | "
            "{component} epoch={epoch} best_epoch={best} J={j} | "
            "progress={progress:.1%}".format(
                fold=status["fold"],
                state=(
                    "COMPLETE"
                    if status["manifest_complete"]
                    else ("RUNNING" if status["alive"] else "STOPPED")
                ),
                pid=status["pid"],
                done=status["completed_components"],
                component=status["current_component"],
                epoch=status["current_epochs"],
                best=status["current_best_epoch"],
                j=j_text,
                progress=status["progress"],
            )
        )
        print(
            "       observed/epoch={} | estimated ETA={} | conservative 30-epoch cap={}".format(
                duration(status["seconds_per_epoch"]),
                duration(status["eta"]),
                duration(status["max_eta"]),
            )
        )
    overall_eta = max(
        [value["eta"] for value in statuses if value["eta"] is not None],
        default=None,
    )
    overall_cap = max(
        [value["max_eta"] for value in statuses if value["max_eta"] is not None],
        default=None,
    )
    print("-" * 86)
    print(
        "Parallel job ETA: {}  | conservative cap: {}".format(
            duration(overall_eta), duration(overall_cap)
        )
    )
    print("GPU index, memory.used, utilization:")
    for line in gpu_lines():
        print("  " + line)


if __name__ == "__main__":
    main()
