"""Windows-native real-chain smoke run for Safe-CFCompatKD.

This is deliberately different from the formal replay gate.  It executes seed
1112 for each of the three frozen runs for two epochs, using the formal clean
DLF, ModDrop evaluator, and train-only compatibility cache.  It never
constructs or traverses the official Test split.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

import smoke_test_cfcompat_safe_projection as synthetic_smoke
import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_safe_projection_valid_screen_v3 as memory_safe
from trains.singleTask.cfcompat_safe_projection_utils import RUNS
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


SEED = 1112


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows real-chain Safe-CFCompatKD smoke test."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/windows_valid_only_prereq_v1")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("The Windows smoke protocol fixes num_workers=1.")
    args.seeds = [SEED]
    args.smoke_test = True
    args.max_epochs = 2
    return args


def git_value(*arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def main() -> None:
    cli = parse_args()

    # Unit-level projection/gate checks first.
    synthetic_smoke.main()

    # Bind the memory-safe v3 implementations into the frozen v1 trajectory.
    base.load_assets = memory_safe.load_assets
    base.forward_objective = memory_safe.forward_objective
    base.reference_prediction_rows = memory_safe.reference_prediction_rows
    base.train_trajectory = memory_safe.train_trajectory

    output_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_safe_projection_v1"
        / cli.dataset
        / "valid_screen"
        / "smoke"
    )
    model_root = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cfcompat_safe_projection_v1"
        / cli.dataset
        / "valid_screen"
        / "smoke"
    )
    if output_root.exists() or model_root.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "Safe-CFCompat smoke output already exists; inspect it or use "
                "--overwrite: {} / {}".format(output_root, model_root)
            )
        if output_root.exists():
            shutil.rmtree(output_root)
        if model_root.exists():
            shutil.rmtree(model_root)
    output_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)

    logger, log_path = base.create_logger(cli)
    rows = []
    for run in RUNS:
        result, epoch_rows, raw_events = memory_safe.train_trajectory(
            cli, logger, output_root, model_root, SEED, run
        )
        if bool(result.get("TestConstructed", True)):
            raise RuntimeError("The smoke trajectory reported Test construction.")
        if int(result["TrainEpochCount"]) != 2:
            raise RuntimeError("Every smoke trajectory must execute exactly two epochs.")
        if len(epoch_rows) != 2:
            raise RuntimeError("Smoke epoch metrics are incomplete.")
        if len(raw_events) != 229 * 4:
            raise RuntimeError("Smoke Valid prediction grid is incomplete.")
        if not math.isfinite(float(result["J_valid"])):
            raise FloatingPointError("Smoke Valid J is non-finite.")
        rows.append(result)

    frame = pd.DataFrame(rows)
    if set(frame.Run.astype(str)) != set(RUNS) or len(frame) != len(RUNS):
        raise RuntimeError("The three-run smoke grid is incomplete.")
    summary_path = output_root / "windows_real_chain_smoke_grid.csv"
    frame.to_csv(summary_path, index=False)

    manifest = {
        "purpose": "Windows-native real-chain Safe-CFCompatKD smoke",
        "seed": SEED,
        "runs": list(RUNS),
        "epochs_per_run": 2,
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "grid_csv": str(summary_path.resolve()),
        "grid_csv_sha256": checkpoint_sha256(summary_path),
        "log_path": str(Path(log_path).resolve()),
    }
    manifest_path = output_root / "windows_real_chain_smoke_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    columns = [
        "Seed",
        "Run",
        "BestValidEpoch",
        "TrainEpochCount",
        "J_valid",
        "valid_LAV_MAE",
        "valid_LA_MAE",
        "valid_LV_MAE",
        "valid_L_MAE",
        "TestConstructed",
    ]
    print("\nWindows Safe-CFCompat real-chain smoke passed")
    print(frame[columns].to_string(index=False))
    print("manifest:", manifest_path)


if __name__ == "__main__":
    main()
