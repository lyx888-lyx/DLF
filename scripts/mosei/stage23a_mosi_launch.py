"""Run the speed probe and then the single-worker five-fold coordinator."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from stage23a_mosi_common import RESULT_ROOT


ROOT = Path(__file__).resolve().parents[2]
LEGACY_ASSET_CWD = Path("/code/DLF")


def run(command, environment):
    print("Launching: {}".format(" ".join(command)), flush=True)
    completed = subprocess.run(
        command, cwd=str(LEGACY_ASSET_CWD), env=environment, check=False
    )
    if completed.returncode:
        raise SystemExit(completed.returncode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu-id", type=int, required=True)
    cli = parser.parse_args()
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = "2"
    speed = RESULT_ROOT / "speed_probe" / "speed_probe_manifest.json"
    if not speed.is_file():
        run(
            [
                sys.executable,
                str(
                    ROOT
                    / "scripts"
                    / "mosei"
                    / "stage23a_mosi_train_fold.py"
                ),
                "--outer-fold",
                "0",
                "--gpu-id",
                str(cli.gpu_id),
                "--max-epochs",
                "3",
                "--patience",
                "3",
                "--num-workers",
                "2",
                "--benchmark-clean-only",
            ],
            environment,
        )
    run(
        [
            sys.executable,
            str(
                ROOT
                / "scripts"
                / "mosei"
                / "stage23a_mosi_coordinator.py"
            ),
            "--gpu-id",
            str(cli.gpu_id),
            "--num-workers",
            "2",
        ],
        environment,
    )


if __name__ == "__main__":
    main()
