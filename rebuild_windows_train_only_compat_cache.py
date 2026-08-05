"""Build one Windows-native, train-only counterfactual compatibility cache.

The runner consumes the validation-best ModDrop evaluator for one formal seed.
A runtime guard rejects any attempt to construct a validation or Test loader.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

import train_cf_compat_kd
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_COLUMNS,
    MULTISEED_CACHE_VERSION,
    cache_paths,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


VERSION = "windows_valid_only_prereq_v1"
FORMAL_SEEDS = (1112, 1113, 1115)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows-native train-only compatibility-cache reconstruction."
    )
    parser.add_argument("--seed", required=True, type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/windows_valid_only_prereq_v1")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.num_workers not in (0, 1):
        parser.error("Windows reconstruction permits only num_workers 0 or 1.")
    if args.num_workers != 1:
        parser.error("Formal cache reconstruction fixes num_workers=1.")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_logger(cli: argparse.Namespace) -> tuple[logging.Logger, Path]:
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "compat-cache-seed{}-formal-{}.log".format(
        cli.seed, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("windows_train_only_compat_cache")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (
        logging.FileHandler(path, encoding="utf-8"),
        logging.StreamHandler(),
    ):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def build_cli(cli: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=cli.dataset,
        seeds=[int(cli.seed)],
        gate_mode="compat",
        eta=1.0,
        lambda_kd=1.0,
        build_gate_cache_only=True,
        smoke_test=False,
        max_epochs=None,
        num_workers=int(cli.num_workers),
        gpu_ids=list(cli.gpu_ids),
        model_save_dir=cli.model_save_dir,
        result_root=cli.result_root,
        log_dir=cli.log_dir,
        config_file=cli.config_file,
        multiseed_replication=True,
    )


def locate_and_audit_evaluator(
    cli: argparse.Namespace,
) -> tuple[Path, int, Path, str]:
    evaluator, best_epoch, source_csv = train_cf_compat_kd.locate_stage1_evaluator(
        cli.result_root,
        cli.dataset,
        int(cli.seed),
        multiseed=True,
        smoke=False,
    )
    source_csv = Path(source_csv)
    manifest = source_csv.parent / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError("ModDrop manifest is absent: {}".format(manifest))

    rows = pd.read_csv(source_csv)
    selected = rows.loc[rows["Seed"].astype(int) == int(cli.seed)]
    if len(selected) != 1:
        raise ValueError("ModDrop CSV does not contain one unique seed row.")
    row = selected.iloc[0]
    actual_sha = checkpoint_sha256(evaluator)
    recorded_sha = str(row.get("MainCheckpointSHA256", "")).lower()
    if not recorded_sha or recorded_sha != actual_sha.lower():
        raise ValueError("ModDrop evaluator SHA does not match its per-seed CSV.")
    if str(row.get("TestConstructed", False)).lower() not in ("false", "0"):
        raise ValueError("ModDrop input was not recorded as Test-free.")
    return Path(evaluator), int(best_epoch), source_csv, actual_sha


def output_directory(cli: argparse.Namespace) -> Path:
    return cache_paths(
        cli.result_root,
        cli.dataset,
        version=MULTISEED_CACHE_VERSION,
        seed=int(cli.seed),
    )["directory"]


def audit_cache(
    cli: argparse.Namespace,
    paths: Dict[str, Path],
    evaluator: Path,
    evaluator_sha: str,
    loader_calls: List[str],
) -> tuple[pd.DataFrame, Dict[str, Any]]:
    required = ("csv", "config", "summary", "bins")
    for key in required:
        if not Path(paths[key]).is_file():
            raise FileNotFoundError("Cache artifact is absent: {}".format(paths[key]))

    frame = pd.read_csv(paths["csv"])
    config = json.loads(Path(paths["config"]).read_text(encoding="utf-8"))
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Cache columns do not match the registered schema.")
    if len(frame) != 1284 or frame["sample_index"].nunique() != 1284:
        raise RuntimeError("Cache must contain 1284 unique MOSI train samples.")
    indices = frame["sample_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(indices, np.arange(1284, dtype=np.int64)):
        raise RuntimeError("Cache sample indices are not exactly 0..1283.")
    for mode in ("LA", "LV", "L"):
        values = frame["compat_{}".format(mode)].to_numpy(dtype=np.float64)
        if not np.isfinite(values).all() or not np.all((values > 0) & (values < 1)):
            raise FloatingPointError("Cached compatibility is invalid for {}.".format(mode))

    expected = {
        "version": MULTISEED_CACHE_VERSION,
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "train_sample_count": 1284,
        "source": "train_only",
        "created_from_train_only": True,
        "rng_state_preserved": True,
        "evaluator_sha256": evaluator_sha,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError("Cache config binding mismatch for {}.".format(key))
    if Path(str(config["evaluator_checkpoint"])).resolve() != evaluator.resolve():
        raise ValueError("Cache config evaluator path binding is invalid.")
    if config.get("cache_sha256") != checkpoint_sha256(paths["csv"]):
        raise ValueError("Cache CSV SHA binding is invalid.")
    if loader_calls != ["train"]:
        raise RuntimeError(
            "Cache loader audit expected exactly one train loader; got {}.".format(
                loader_calls
            )
        )
    return frame, config


def run(cli: argparse.Namespace) -> Dict[str, Any]:
    evaluator, best_epoch, source_csv, evaluator_sha = locate_and_audit_evaluator(cli)
    destination = output_directory(cli)
    if destination.exists() and not cli.overwrite:
        raise FileExistsError(
            "Cache output already exists. Inspect it or rerun with --overwrite: {}".format(
                destination
            )
        )
    if destination.exists():
        shutil.rmtree(destination)

    logger, log_path = create_logger(cli)
    logger.info(
        "start version=%s seed=%s evaluator=%s evaluator_sha=%s "
        "best_epoch=%s source_csv=%s train_only=true test_constructed=false",
        VERSION,
        cli.seed,
        evaluator,
        evaluator_sha,
        best_epoch,
        source_csv,
    )

    local = build_cli(cli)
    loader_calls: List[str] = []
    original_builder = train_cf_compat_kd.build_single_split_loader

    def guarded_builder(args, split, num_workers):
        split = str(split)
        loader_calls.append(split)
        if split != "train":
            raise RuntimeError(
                "Compatibility-cache stage forbids non-train loader: {}".format(split)
            )
        return original_builder(args, split, num_workers)

    train_cf_compat_kd.build_single_split_loader = guarded_builder
    try:
        paths = train_cf_compat_kd.build_gate_cache_only(
            local, int(cli.seed), logger
        )
    finally:
        train_cf_compat_kd.build_single_split_loader = original_builder

    paths = {key: Path(value) for key, value in paths.items()}
    frame, config = audit_cache(
        cli, paths, evaluator, evaluator_sha, loader_calls
    )

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    row: Dict[str, Any] = {
        "Seed": int(cli.seed),
        "Method": "CFCompat-Train-Cache-Windows-v1",
        "CacheVersion": MULTISEED_CACHE_VERSION,
        "TrainSampleCount": int(len(frame)),
        "UniqueSampleCount": int(frame["sample_index"].nunique()),
        "EvaluatorCheckpoint": str(evaluator.resolve()),
        "EvaluatorSHA256": evaluator_sha,
        "EvaluatorBestValidEpoch": int(best_epoch),
        "EvaluatorSourceCSV": str(source_csv.resolve()),
        "CacheCSV": str(paths["csv"].resolve()),
        "CacheSHA256": config["cache_sha256"],
        "ConfigSHA256": config["config_sha256"],
        "LoaderCalls": ",".join(loader_calls),
        "OfficialTestConstructed": False,
        "TestLoaderConstructionCount": 0,
        "NumWorkers": int(cli.num_workers),
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "Torch": torch.__version__,
        "TorchCUDARuntime": torch.version.cuda,
        "GPU": torch.cuda.get_device_name(0),
        "GPUCapability": "{}.{}".format(*torch.cuda.get_device_capability(0)),
        "GitBranch": git_value("branch", "--show-current"),
        "GitCommit": git_value("rev-parse", "HEAD"),
        "CompletedUTC": utc_now(),
        "LogPath": str(log_path.resolve()),
    }
    for mode in ("LA", "LV", "L"):
        row["{}_CompatMin".format(mode)] = float(summary[mode]["compat"]["min"])
        row["{}_CompatMax".format(mode)] = float(summary[mode]["compat"]["max"])
        row["{}_DeltaCompatSpearman".format(mode)] = float(
            summary[mode]["corr_delta_compat_spearman"]
        )

    logger.info(
        "complete seed=%s samples=%s cache=%s cache_sha=%s loader_calls=%s",
        cli.seed,
        len(frame),
        paths["csv"],
        config["cache_sha256"],
        loader_calls,
    )
    for handler in logger.handlers:
        handler.flush()

    manifest = {
        "version": VERSION,
        "action": "train-only-compatibility-cache",
        "seed": int(cli.seed),
        "source_split": "train_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "loader_calls": loader_calls,
        "sample_count": int(len(frame)),
        "unique_sample_count": int(frame["sample_index"].nunique()),
        "evaluator": {
            "path": str(evaluator.resolve()),
            "sha256": evaluator_sha,
            "best_valid_epoch": int(best_epoch),
            "source_csv": str(source_csv.resolve()),
        },
        "environment": {
            "python": row["Python"],
            "platform": row["Platform"],
            "torch": row["Torch"],
            "torch_cuda_runtime": row["TorchCUDARuntime"],
            "gpu": row["GPU"],
            "gpu_capability": row["GPUCapability"],
            "git_branch": row["GitBranch"],
            "git_commit": row["GitCommit"],
        },
        "artifacts": {
            key: {"path": str(paths[key].resolve()), "sha256": sha256(paths[key])}
            for key in ("csv", "config", "summary", "bins")
        },
        "log": {"path": str(log_path.resolve()), "sha256": sha256(log_path)},
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    pd.DataFrame([row]).to_csv(destination / "cache_run.csv", index=False)
    print(pd.DataFrame([row]).to_string(index=False))
    print("manifest:", manifest_path)
    return row


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
