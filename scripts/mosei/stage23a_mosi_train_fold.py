"""Thin MOSI adapter around the SHA-bound MOSEI Stage23A fold trainer."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

import stage23a_train_fold as frozen
from stage23a_mosi_common import (
    DATASET_PATH,
    EXPERTS,
    N_FOLDS,
    RESULT_ROOT,
    atomic_json,
    load_preregistered,
    load_split_manifest,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[2]


class BenchmarkComplete(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outer-fold", required=True, type=int, choices=range(N_FOLDS))
    parser.add_argument("--gpu-id", required=True, type=int)
    parser.add_argument("--config-file", default=str(ROOT / "config" / "config.json"))
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--benchmark-clean-only", action="store_true")
    return parser.parse_args()


def configure_frozen_module(cli):
    frozen.RESULT_ROOT = (
        RESULT_ROOT / "speed_probe" if cli.benchmark_clean_only else RESULT_ROOT
    )
    frozen.EXPERTS = EXPERTS
    frozen.load_preregistered = load_preregistered
    frozen.load_split_manifest = load_split_manifest
    frozen.parse_args = lambda: cli

    def build_args(adapted_cli):
        args = frozen.get_config_regression(
            "DLF", "mosi", adapted_cli.config_file
        )
        args.mode = "train"
        args.feature_T = args.feature_A = args.feature_V = ""
        args.featurePath = str(DATASET_PATH)
        args.is_training = True
        args.train_mode = "regression"
        args.seed = args.cur_seed = 23000 + int(adapted_cli.outer_fold)
        args.device = frozen.assign_gpu([int(adapted_cli.gpu_id)])
        return args

    frozen.build_args = build_args
    original_train = frozen.train_with_inner_selection

    def monitored_train(
        name,
        model,
        train_loader,
        valid_loader,
        args,
        adapted_cli,
        train_step,
        evaluate,
        output_dir,
    ):
        torch.cuda.reset_peak_memory_stats(args.device)
        started = time.perf_counter()
        model, manifest = original_train(
            name,
            model,
            train_loader,
            valid_loader,
            args,
            adapted_cli,
            train_step,
            evaluate,
            output_dir,
        )
        manifest.update(
            {
                "dataset": "mosi",
                "wall_seconds": time.perf_counter() - started,
                "peak_gpu_memory_bytes": int(
                    torch.cuda.max_memory_allocated(args.device)
                ),
                "OMP_NUM_THREADS": int(os.environ.get("OMP_NUM_THREADS", "0")),
                "dataloader_workers": int(adapted_cli.num_workers),
                "official_valid_access_count": 0,
                "locked_test_access_count": 0,
            }
        )
        atomic_json(Path(output_dir) / "run_manifest.json", manifest)
        if adapted_cli.benchmark_clean_only and name == "clean_seed1111":
            raise BenchmarkComplete()
        return model, manifest

    frozen.train_with_inner_selection = monitored_train


def write_benchmark(cli):
    directory = (
        RESULT_ROOT
        / "speed_probe"
        / "expert_oof"
        / "outer_fold{}".format(cli.outer_fold)
        / "components"
        / "clean_seed1111"
    )
    metrics = pd.read_csv(directory / "epoch_metrics.csv")
    manifest = json.loads((directory / "run_manifest.json").read_text())
    if len(metrics) != 3:
        raise RuntimeError("Speed probe must contain exactly three epochs.")
    epoch_seconds = metrics.wall_seconds.diff()
    epoch_seconds.iloc[0] = metrics.wall_seconds.iloc[0]
    report = {
        "stage": "Stage23A-MOSI 3-epoch speed probe",
        "fold": int(cli.outer_fold),
        "component": "clean_seed1111",
        "epochs": 3,
        "epoch_seconds": [float(value) for value in epoch_seconds],
        "observed_seconds_per_epoch": float(epoch_seconds.mean()),
        "estimated_seconds_per_component_12_epochs": float(
            epoch_seconds.mean() * 12
        ),
        "estimated_seconds_all_35_components_12_epochs": float(
            epoch_seconds.mean() * 12 * 35
        ),
        "conservative_seconds_all_35_components_30_epochs": float(
            epoch_seconds.mean() * 30 * 35
        ),
        "peak_gpu_memory_bytes": int(manifest["peak_gpu_memory_bytes"]),
        "gpu_id": int(cli.gpu_id),
        "OMP_NUM_THREADS": int(os.environ.get("OMP_NUM_THREADS", "0")),
        "dataloader_workers": int(cli.num_workers),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    path = RESULT_ROOT / "speed_probe" / "speed_probe_manifest.json"
    atomic_json(path, report)
    print(json.dumps(report, indent=2, sort_keys=True))


def main():
    cli = parse_args()
    if cli.benchmark_clean_only:
        if cli.max_epochs != 3:
            raise RuntimeError("Benchmark must use exactly three epochs.")
        cli.patience = 3
    if int(os.environ.get("OMP_NUM_THREADS", "0")) != 2:
        raise RuntimeError("Stage23A-MOSI requires OMP_NUM_THREADS=2.")
    if cli.num_workers != 2:
        raise RuntimeError("Stage23A-MOSI initially requires two DataLoader workers.")
    configure_frozen_module(cli)
    try:
        frozen.main()
    except BenchmarkComplete:
        write_benchmark(cli)


if __name__ == "__main__":
    main()

