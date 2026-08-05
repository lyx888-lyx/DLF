"""Build one Windows-native Valid-only original CFCompatKD reference.

The resulting per-seed CSV is the Stage-3 reference consumed by the formal
Safe-CFCompatKD replay gate.  This entry point reuses the exact
``cfcompat_replay`` trajectory from the memory-safe Safe v3 implementation and
never constructs, reads, or reports the official Test split.
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
from typing import Any, Dict

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch

import train_cfcompat_safe_projection_valid_screen as safe_base
import train_cfcompat_safe_projection_valid_screen_v3 as safe_v3
from trains.singleTask.cf_compat_kd_utils import (
    MULTISEED_CACHE_VERSION,
    cache_paths,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256


VERSION = "windows_valid_only_cfcompat_reference_v1"
FORMAL_SEEDS = (1112, 1113, 1115)
RUN = "cfcompat_replay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows-native Valid-only original CFCompatKD reference."
    )
    parser.add_argument("--seed", required=True, type=int, choices=FORMAL_SEEDS)
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/windows_valid_only_prereq_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if int(args.num_workers) != 1:
        parser.error("This reconstruction fixes num_workers=1.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    elif args.max_epochs is not None:
        parser.error("Formal reconstruction uses the frozen early-stop rule.")
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


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def create_logger(cli: argparse.Namespace) -> tuple[logging.Logger, Path]:
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / "cfcompat-reference-seed{}-{}-{}.log".format(
        cli.seed, kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("windows_valid_only_cfcompat_reference")
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


def roots(cli: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    result_base = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cf_compat_kd_v1"
        / "benchmark_multiseed"
    )
    model_base = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cf_compat_kd_v1"
        / "benchmark_multiseed"
    )
    if cli.smoke_test:
        result_base = result_base / "smoke"
        model_base = model_base / "smoke"
    seed_result = result_base / "seed{}".format(cli.seed)
    seed_model = model_base / "seed{}".format(cli.seed)
    checkpoint = seed_model / "DLF_{}_seed{}_best_valid.pth".format(
        cli.dataset, cli.seed
    )
    return result_base, model_base, seed_result, checkpoint


def required_assets(cli: argparse.Namespace) -> Dict[str, Path]:
    seed = int(cli.seed)
    moddrop_dir = (
        Path(cli.result_root)
        / "missing_baseline"
        / "moddrop_benchmark_multiseed_v1"
        / "seed{}".format(seed)
    )
    moddrop_checkpoint = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "moddrop_benchmark_multiseed_v1"
        / "seed{}".format(seed)
        / "DLF_{}_seed{}_best_valid.pth".format(cli.dataset, seed)
    )
    cache = cache_paths(
        cli.result_root,
        cli.dataset,
        version=MULTISEED_CACHE_VERSION,
        seed=seed,
    )
    return {
        "clean_checkpoint": Path(cli.model_save_dir)
        / "DLF_{}_seed{}_best.pth".format(cli.dataset, seed),
        "moddrop_checkpoint": moddrop_checkpoint,
        "moddrop_csv": moddrop_dir / "{}_per_seed.csv".format(cli.dataset),
        "moddrop_manifest": moddrop_dir / "manifest.json",
        "cache_csv": cache["csv"],
        "cache_config": cache["config"],
        "cache_summary": cache["summary"],
    }


def audit_assets(cli: argparse.Namespace) -> Dict[str, Any]:
    assets = required_assets(cli)
    missing = [str(path) for path in assets.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required CFCompat reference assets are absent: {}".format(missing)
        )

    config = json.loads(assets["cache_config"].read_text(encoding="utf-8"))
    moddrop_sha = checkpoint_sha256(assets["moddrop_checkpoint"])
    cache_sha = checkpoint_sha256(assets["cache_csv"])
    if int(config.get("seed", -1)) != int(cli.seed):
        raise ValueError("Compatibility cache seed binding is invalid.")
    if config.get("source") != "train_only":
        raise ValueError("Compatibility cache is not train-only.")
    if int(config.get("train_sample_count", -1)) != 1284:
        raise ValueError("Compatibility cache does not contain 1284 train samples.")
    if str(config.get("evaluator_sha256")) != moddrop_sha:
        raise ValueError("Cache evaluator SHA does not match the ModDrop checkpoint.")
    if str(config.get("cache_sha256")) != cache_sha:
        raise ValueError("Compatibility cache SHA binding is invalid.")

    return {
        "paths": assets,
        "clean_sha": checkpoint_sha256(assets["clean_checkpoint"]),
        "moddrop_sha": moddrop_sha,
        "cache_sha": cache_sha,
        "cache_config_sha": checkpoint_sha256(assets["cache_config"]),
    }


def configure_memory_safe_replay() -> None:
    # Bind the formal base trajectory to the v3 memory-safe asset handling.
    safe_base.load_assets = safe_v3.load_assets
    safe_base.forward_objective = safe_v3.forward_objective
    safe_base.reference_prediction_rows = safe_v3.reference_prediction_rows

    # The replay path must never request a single split loader.  Safe v3 only
    # uses this helper for the two Safe candidates, not for cfcompat_replay.
    original = safe_v3.build_single_split_loader

    def forbidden_single_split_loader(args, split, num_workers):
        raise RuntimeError(
            "CFCompat reference replay forbids single-split loader construction: "
            "{}".format(split)
        )

    safe_v3.build_single_split_loader = forbidden_single_split_loader
    safe_v3._WINDOWS_ORIGINAL_SINGLE_SPLIT_LOADER = original


def restore_memory_safe_replay() -> None:
    original = getattr(safe_v3, "_WINDOWS_ORIGINAL_SINGLE_SPLIT_LOADER", None)
    if original is not None:
        safe_v3.build_single_split_loader = original
        delattr(safe_v3, "_WINDOWS_ORIGINAL_SINGLE_SPLIT_LOADER")


def prediction_frame(raw_events: list[dict]) -> pd.DataFrame:
    raw = pd.DataFrame(raw_events)
    if raw.empty:
        raise RuntimeError("CFCompat reference produced no Valid prediction events.")
    required = {"sample_index", "sample_id", "label", "Mode", "candidate_prediction"}
    if not required.issubset(raw.columns):
        raise RuntimeError("CFCompat Valid events lack required prediction fields.")
    pivot = raw.pivot_table(
        index=["sample_index", "sample_id", "label"],
        columns="Mode",
        values="candidate_prediction",
        aggfunc="first",
    ).reset_index()
    pivot.columns.name = None
    for mode in ("LAV", "LA", "LV", "L"):
        if mode not in pivot.columns:
            raise RuntimeError("CFCompat Valid predictions lack mode {}.".format(mode))
        pivot = pivot.rename(columns={mode: "{}_pred".format(mode)})
    if len(pivot) != 229 or pivot.sample_index.nunique() != 229:
        raise RuntimeError("CFCompat Valid prediction audit expected 229 unique samples.")
    return pivot.sort_values("sample_index", kind="mergesort")


def run(cli: argparse.Namespace) -> Dict[str, Any]:
    audit = audit_assets(cli)
    result_base, model_base, output_dir, checkpoint = roots(cli)

    if (output_dir.exists() or checkpoint.exists()) and not cli.overwrite:
        raise FileExistsError(
            "Output already exists. Re-run with --overwrite after checking it: "
            "{} or {}".format(output_dir, checkpoint)
        )
    if cli.overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        model_seed = checkpoint.parent
        if model_seed.exists():
            shutil.rmtree(model_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logger, log_path = create_logger(cli)

    logger.info(
        "start version=%s seed=%s smoke=%s clean_sha=%s moddrop_sha=%s "
        "cache_sha=%s test_constructed=false",
        VERSION,
        cli.seed,
        cli.smoke_test,
        audit["clean_sha"],
        audit["moddrop_sha"],
        audit["cache_sha"],
    )

    configure_memory_safe_replay()
    try:
        result, epoch_rows, raw_events = safe_v3.train_trajectory(
            cli,
            logger,
            result_base,
            model_base,
            int(cli.seed),
            RUN,
        )
    finally:
        restore_memory_safe_replay()

    generated_checkpoint = Path(result["MainCheckpoint"])
    if not generated_checkpoint.is_file():
        raise FileNotFoundError(
            "Generated CFCompat replay checkpoint is absent: {}".format(
                generated_checkpoint
            )
        )
    shutil.copy2(generated_checkpoint, checkpoint)
    checkpoint_hash = checkpoint_sha256(checkpoint)

    result.update(
        {
            "Method": "DLF-CFCompatKD-v1",
            "Run": RUN,
            "MainCheckpoint": str(checkpoint.resolve()),
            "MainCheckpointRelative": str(checkpoint),
            "MainCheckpointSHA256": checkpoint_hash,
            "OfficialTestConstructed": False,
            "TestConstructed": False,
            "TestLoaderConstructionCount": 0,
            "TestLoaderTraversalCount": 0,
            "SmokeTest": bool(cli.smoke_test),
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
    )

    per_seed = output_dir / "{}_per_seed.csv".format(cli.dataset)
    epoch_path = output_dir / "{}_epoch_metrics.csv".format(cli.dataset)
    raw_path = output_dir / "{}_best_valid_raw_events.csv".format(cli.dataset)
    prediction_path = output_dir / "{}_best_valid_predictions.csv".format(cli.dataset)
    summary_path = output_dir / "{}_summary.csv".format(cli.dataset)

    pd.DataFrame([result]).to_csv(per_seed, index=False)
    pd.DataFrame(epoch_rows).to_csv(epoch_path, index=False)
    pd.DataFrame(raw_events).to_csv(raw_path, index=False)
    prediction_frame(raw_events).to_csv(prediction_path, index=False)
    numeric = pd.DataFrame([result]).select_dtypes(include="number")
    pd.DataFrame(
        {
            "Metric": numeric.columns,
            "Mean": numeric.iloc[0].values,
            "Std": [0.0] * len(numeric.columns),
        }
    ).to_csv(summary_path, index=False)

    for handler in logger.handlers:
        handler.flush()

    manifest = {
        "version": VERSION,
        "action": "original_cfcompat_reference",
        "seed": int(cli.seed),
        "run": RUN,
        "decision_split": "official_valid_only",
        "checkpoint_selection": "minimum_official_valid_J",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "formal_replay_tolerance": 1e-4,
        "assets": {
            "clean_checkpoint": {
                "path": str(audit["paths"]["clean_checkpoint"].resolve()),
                "sha256": audit["clean_sha"],
            },
            "moddrop_checkpoint": {
                "path": str(audit["paths"]["moddrop_checkpoint"].resolve()),
                "sha256": audit["moddrop_sha"],
            },
            "compatibility_cache": {
                "path": str(audit["paths"]["cache_csv"].resolve()),
                "sha256": audit["cache_sha"],
            },
            "compatibility_config": {
                "path": str(audit["paths"]["cache_config"].resolve()),
                "sha256": audit["cache_config_sha"],
            },
        },
        "environment": {
            "python": result["Python"],
            "platform": result["Platform"],
            "torch": result["Torch"],
            "torch_cuda_runtime": result["TorchCUDARuntime"],
            "gpu": result["GPU"],
            "gpu_capability": result["GPUCapability"],
            "git_branch": result["GitBranch"],
            "git_commit": result["GitCommit"],
        },
        "artifacts": {},
    }
    for name, path in {
        "checkpoint": checkpoint,
        "per_seed_csv": per_seed,
        "epoch_metrics_csv": epoch_path,
        "raw_valid_events_csv": raw_path,
        "valid_predictions_csv": prediction_path,
        "summary_csv": summary_path,
        "log": log_path,
    }.items():
        manifest["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": sha256(path),
        }

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.info(
        "complete seed=%s best_epoch=%s J_valid=%.6f checkpoint=%s sha=%s",
        cli.seed,
        result["BestValidEpoch"],
        result["J_valid"],
        checkpoint,
        checkpoint_hash,
    )
    print(pd.DataFrame([result]).to_string(index=False))
    print("manifest:", manifest_path)
    return result


def main() -> None:
    cli = parse_args()
    run(cli)


if __name__ == "__main__":
    main()
