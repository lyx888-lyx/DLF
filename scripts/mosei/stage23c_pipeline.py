#!/usr/bin/env python3
"""Resumable post-screen Stage23C pipeline workers."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "result" / "oracle_distillation_v1" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23c"
PYTHON = str(Path(sys.executable))
TRAIN = str(ROOT / "scripts" / "mosei" / "stage23c_train.py")
FINALIZER = str(ROOT / "scripts" / "mosei" / "stage23c_finalize.py")
PRE_S6 = OUT / "protocol" / "frozen_pre_s6_selection.json"
SELECTION = OUT / "protocol" / "frozen_student_selection.json"
SHUFFLE_SEEDS = (23611, 23612, 23613)

METHODS = {
    "S0": "S0_supervised_moddrop",
    "S1": "S1_uniform_single_teacher_kd",
    "S2": "S2_equal_ensemble_final_kd",
    "S3": "S3_fixed_stacking_final_kd",
    "S4": "S4_soft_oracle_final_kd",
    "S5": "S5_soft_oracle_hierarchical_kd",
    "S6": "S6_shuffled_soft_oracle_hierarchical_kd",
}

S6_QUEUES = {
    "s6_0": (("A", 23611), ("B", 23611)),
    "s6_1": (("A", 23612), ("B", 23612)),
    "s6_2": (("A", 23613), ("B", 23613)),
}

FINAL_QUEUES = {
    "final_0": (
        ("A", "S0", None),
        ("A", "S3", None),
        ("A", "S5", None),
        ("A", "S6", 23611),
        ("B", "S2", None),
        ("B", "S6", 23611),
    ),
    "final_1": (
        ("B", "S0", None),
        ("B", "S3", None),
        ("B", "S5", None),
        ("B", "S6", 23612),
        ("A", "S2", None),
        ("A", "S6", 23612),
    ),
    "final_2": (
        ("A", "S1", None),
        ("A", "S4", None),
        ("A", "S6", 23613),
        ("B", "S1", None),
        ("B", "S4", None),
        ("B", "S6", 23613),
    ),
}


def valid_json(path):
    if not path.exists():
        return False
    try:
        json.loads(path.read_text(encoding="utf-8"))
        return True
    except json.JSONDecodeError:
        return False


def completed(path):
    if not valid_json(path):
        return False
    return json.loads(path.read_text(encoding="utf-8")).get("status") == "COMPLETED"


def wait_until(predicate, description):
    while not predicate():
        print(f"waiting for {description}", flush=True)
        time.sleep(30)


def call(command, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("start " + " ".join(command), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        result = subprocess.run(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise RuntimeError(f"command failed; see {log_path}")
    print(f"complete log={log_path}", flush=True)


def selection_value():
    return json.loads(SELECTION.read_text(encoding="utf-8"))


def identifier(direction, short_name, seed=None):
    if short_name == "S5":
        ratio = selection_value()["directions"][direction][
            "selected_hier_ratio"
        ]
        return f"S5_hier_ratio{str(ratio).replace('.', 'p')}"
    if short_name == "S6":
        return f"S6_shuffle_seed{seed}"
    return short_name


def screen_manifest(direction, seed):
    return (
        OUT
        / "training"
        / f"direction_{direction}"
        / "screen"
        / f"S6_shuffle_seed{seed}"
        / "run_manifest.json"
    )


def pre_s6_screens_complete():
    for direction in ("A", "B"):
        for identifier_value in (
            "S0",
            "S1",
            "S2",
            "S3",
            "S4",
            "S5_hier_ratio0p25",
            "S5_hier_ratio0p5",
            "S5_hier_ratio1p0",
        ):
            manifest = (
                OUT
                / "training"
                / f"direction_{direction}"
                / "screen"
                / identifier_value
                / "run_manifest.json"
            )
            if not completed(manifest):
                return False
    return True


def freeze_pre_s6():
    wait_until(pre_s6_screens_complete, "all 16 pre-S6 screen manifests")
    if valid_json(PRE_S6):
        print(f"skip existing {PRE_S6}", flush=True)
        return
    call(
        [PYTHON, "-u", TRAIN, "--phase", "freeze-pre-s6"],
        RUNTIME / "freeze_pre_s6.log",
    )
    if not valid_json(PRE_S6):
        raise RuntimeError("Pre-S6 ratio selection was not frozen")


def final_directory(direction, short_name, seed=None):
    return (
        OUT
        / "training"
        / f"direction_{direction}"
        / "final"
        / identifier(direction, short_name, seed)
    )


def run_s6(worker, gpu_id):
    wait_until(lambda: valid_json(PRE_S6), PRE_S6)
    for direction, seed in S6_QUEUES[worker]:
        manifest = screen_manifest(direction, seed)
        if completed(manifest):
            print(f"skip completed {manifest}", flush=True)
            continue
        command = [
            PYTHON,
            "-u",
            TRAIN,
            "--phase",
            "student-screen",
            "--direction",
            direction,
            "--method",
            METHODS["S6"],
            "--shuffle-seed",
            str(seed),
            "--gpu-id",
            str(gpu_id),
            "--num-workers",
            "1",
        ]
        call(command, RUNTIME / f"screen_{direction}_S6_seed{seed}.log")
        if not completed(manifest):
            raise RuntimeError(f"S6 screen did not complete: {manifest}")


def freeze_selection():
    expected = [
        screen_manifest(direction, seed)
        for direction in ("A", "B")
        for seed in SHUFFLE_SEEDS
    ]
    wait_until(
        lambda: all(completed(path) for path in expected),
        "all six S6 screen manifests",
    )
    if valid_json(SELECTION):
        print(f"skip existing {SELECTION}", flush=True)
        return
    call(
        [PYTHON, "-u", TRAIN, "--phase", "freeze-selection"],
        RUNTIME / "freeze_selection.log",
    )
    if not valid_json(SELECTION):
        raise RuntimeError("Student selection was not frozen")


def run_clean_final(direction, gpu_id):
    directory = (
        OUT / "training" / f"direction_{direction}" / "final" / "clean"
    )
    manifest = directory / "run_manifest.json"
    if completed(manifest):
        return
    call(
        [
            PYTHON,
            "-u",
            TRAIN,
            "--phase",
            "clean-final",
            "--direction",
            direction,
            "--gpu-id",
            str(gpu_id),
            "--num-workers",
            "1",
        ],
        RUNTIME / f"final_clean_{direction}.log",
    )
    if not completed(manifest):
        raise RuntimeError(f"Final clean did not complete: {manifest}")


def run_final(worker, gpu_id):
    wait_until(lambda: valid_json(SELECTION), SELECTION)
    if worker == "final_0":
        run_clean_final("A", gpu_id)
    elif worker == "final_1":
        run_clean_final("B", gpu_id)
    else:
        clean_a = (
            OUT
            / "training"
            / "direction_A"
            / "final"
            / "clean"
            / "run_manifest.json"
        )
        clean_b = (
            OUT
            / "training"
            / "direction_B"
            / "final"
            / "clean"
            / "run_manifest.json"
        )
        wait_until(
            lambda: completed(clean_a) and completed(clean_b),
            "both final clean checkpoints",
        )
    for direction, short_name, seed in FINAL_QUEUES[worker]:
        directory = final_directory(direction, short_name, seed)
        manifest = directory / "run_manifest.json"
        prediction = directory / "outer_predictions_label_free.csv.gz"
        if completed(manifest) and prediction.exists():
            print(f"skip completed {directory}", flush=True)
            continue
        command = [
            PYTHON,
            "-u",
            TRAIN,
            "--phase",
            "student-final",
            "--direction",
            direction,
            "--method",
            METHODS[short_name],
            "--gpu-id",
            str(gpu_id),
            "--num-workers",
            "1",
        ]
        if seed is not None:
            command.extend(["--shuffle-seed", str(seed)])
        call(
            command,
            RUNTIME
            / f"final_{direction}_{identifier(direction, short_name, seed)}.log",
        )
        if not completed(manifest) or not prediction.exists():
            raise RuntimeError(f"Final Student output incomplete: {directory}")


def final_outputs_complete():
    if not valid_json(SELECTION):
        return False
    for direction in ("A", "B"):
        for short_name in ("S0", "S1", "S2", "S3", "S4", "S5"):
            directory = final_directory(direction, short_name)
            if not completed(directory / "run_manifest.json"):
                return False
            if not (directory / "outer_predictions_label_free.csv.gz").exists():
                return False
        for seed in SHUFFLE_SEEDS:
            directory = final_directory(direction, "S6", seed)
            if not completed(directory / "run_manifest.json"):
                return False
            if not (directory / "outer_predictions_label_free.csv.gz").exists():
                return False
    return True


def finalize():
    wait_until(final_outputs_complete, "all 18 frozen outer prediction ledgers")
    call(
        [PYTHON, "-u", FINALIZER],
        RUNTIME / "finalize.log",
    )


def main():
    choices = (
        ("pre_freeze",)
        + tuple(S6_QUEUES)
        + ("freeze",)
        + tuple(FINAL_QUEUES)
        + ("finalize",)
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", required=True, choices=choices)
    parser.add_argument("--gpu-id", type=int, choices=(0, 1, 2))
    cli = parser.parse_args()
    if cli.worker == "pre_freeze":
        freeze_pre_s6()
    elif cli.worker.startswith("s6_"):
        if cli.gpu_id is None:
            raise ValueError("S6 worker requires --gpu-id")
        run_s6(cli.worker, cli.gpu_id)
    elif cli.worker == "freeze":
        freeze_selection()
    elif cli.worker.startswith("final_"):
        if cli.gpu_id is None:
            raise ValueError("final worker requires --gpu-id")
        run_final(cli.worker, cli.gpu_id)
    else:
        finalize()
    print(f"pipeline worker {cli.worker} complete", flush=True)


if __name__ == "__main__":
    main()
