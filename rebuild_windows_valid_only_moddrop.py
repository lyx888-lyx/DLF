"""Rebuild one Windows-native, validation-only ModDrop evaluator.

This wrapper reuses the frozen Stage-1 ModDrop mathematics in ``train_missing``
but writes the exact multiseed paths consumed by the later CFCompat cache. It
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
import torch.nn as nn

import train_cf_compat_kd
import train_missing
from data_loader import MMDataLoader
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    evaluate_all_modes,
    flatten_mode_metrics,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


VERSION = "windows_valid_only_prereq_v1"
FORMAL_SEEDS = (1112, 1113, 1115)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows-native Valid-only ModDrop reconstruction."
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
    if args.num_workers not in (0, 1):
        parser.error("Windows reconstruction permits only num_workers 0 or 1.")
    if not args.smoke_test and args.num_workers != 1:
        parser.error("Formal reconstruction fixes num_workers=1.")
    if args.smoke_test:
        args.max_epochs = 2 if args.max_epochs is None else min(2, args.max_epochs)
    elif args.max_epochs is not None:
        parser.error("--max-epochs is allowed only with --smoke-test.")
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


def resolve_paths(cli: argparse.Namespace) -> Dict[str, Path]:
    result_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "moddrop_benchmark_multiseed_v1"
    )
    model_root = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "moddrop_benchmark_multiseed_v1"
    )
    if cli.smoke_test:
        result_root = result_root / "smoke"
        model_root = model_root / "smoke"
    return {
        "output": result_root / "seed{}".format(cli.seed),
        "checkpoint": (
            model_root
            / "seed{}".format(cli.seed)
            / "DLF_{}_seed{}_best_valid.pth".format(cli.dataset, cli.seed)
        ),
        "clean": (
            Path(cli.model_save_dir)
            / "DLF_{}_seed{}_best.pth".format(cli.dataset, cli.seed)
        ),
    }


def create_logger(cli: argparse.Namespace) -> tuple[logging.Logger, Path]:
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / "moddrop-seed{}-{}-{}.log".format(
        cli.seed, kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("windows_valid_only_moddrop")
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


def prepare_outputs(output: Path, checkpoint: Path, overwrite: bool) -> None:
    if (output.exists() or checkpoint.exists()) and not overwrite:
        raise FileExistsError(
            "Output already exists. Review it or rerun with --overwrite: "
            "{} or {}".format(output, checkpoint)
        )
    if overwrite:
        if output.exists():
            shutil.rmtree(output)
        if checkpoint.exists():
            checkpoint.unlink()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)


def make_train_cli(cli: argparse.Namespaace) -> argparse.Namespace:
    return argparse.Namespace(
        dataset=cli.dataset,
        seeds=[cli.seed],
        eta=1.0,
        mode="moddrop",
        smoke_test=cli.smoke_test,
        max_epochs=cli.max_epochs,
        num_workers=cli.num_workers,
        gpu_ids=cli.gpu_ids,
        model_save_dir=cli.model_save_dir,
        result_dir=str(resolve_paths(cli)["output"]),
        log_dir=cli.log_dir,
        config_file=cli.config_file,
    )


def run(cli: argparse.Namespace) -> Dict[str, Any]:
    resolved = resolve_paths(cli)
    output_dir = resolved["output"]
    checkpoint = resolved["checkpoint"]
    clean_checkpoint = resolved["clean"]
    prepare_outputs(output_dir, checkpoint, cli.overwrite)
    if not clean_checkpoint.is_file():
        raise FileNotFoundError(
            "Required clean DLF checkpoint is absent: {}".format(clean_checkpoint)
        )

    clean_hash = sha256(clean_checkpoint)
    logger, log_path = create_logger(cli)
    local = make_train_cli(cli)
    logger.info(
        "start version=%s seed=%s smoke=%s clean=%s clean_sha=%s "
        "test_constructed=false",
        VERSION,
        cli.seed,
        cli.smoke_test,
        clean_checkpoint,
        clean_hash,
    )

    # Reuse the frozen Stage-1 trainer while overriding only its destination.
    original_checkpoint_for_run = train_missing.checkpoint_for_run

    def checkpoint_override(_cli, _dataset, _seed):
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint

    train_missing.checkpoint_for_run = checkpoint_override
    try:
        source_row = train_missing.train_one_seed(local, int(cli.seed), logger)
    finally:
        train_missing.checkpoint_for_run = original_checkpoint_for_run

    args = train_missing.build_config(local, int(cli.seed))
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("ModDrop unexpectedly constructed a non-Valid split.")

    # Fresh-model strict reload audit.
    del source_row["Checkpoint"]
    torch.cuda.empty_cache()
    setup_seed(int(cli.seed))
    audit_model = MissingModalityWrapper(
        DLF(args).to(args.device),
        args.feature_dims[1],
        args.feature_dims[2],
    ).to(args.device)
    audit_model.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    criterion = nn.L1Loss()
    final_valid = evaluate_all_modes(
        audit_model, loaders["valid"], args.device, "moddrop", criterion
    )
    final_j = float(validation_objective(final_valid))
    if abs(final_j - float(source_row["J_val"])) > 1e-10:
        raise RuntimeError("Fresh-model ModDrop checkpoint reload audit failed.")

    predictions = train_cf_compat_kd.prediction_rows(
        audit_model, loaders["valid"], args.device
    )
    checkpoint_hash = sha256(checkpoint)
    row: Dict[str, Any] = {
        "Seed": int(cli.seed),
        "Method": "DLF-ModDrop-Paired-Windows-ValidOnly-v1",
        "BestValidEpoch": int(source_row.pop("BestEpoch")),
        "J_valid": final_j,
        "MainCheckpoint": str(checkpoint.resolve()),
        "MainCheckpointRelative": str(checkpoint),
        "MainCheckpointSHA256": checkpoint_hash,
        "EvaluatorCheckpoint": str(checkpoint.resolve()),
        "EvaluatorSHA256": checkpoint_hash,
        "Gate3Checkpoint": str(clean_checkpoint.resolve()),
        "Gate3SHA256": clean_hash,
        "TestConstructed": False,
        "TestLoaderConstructionCount": 0,
        "TrainSamples": int(len(loaders["train"].dataset)),
        "ValidSamples": int(len(loaders["valid"].dataset)),
        "BatchSize": int(args.batch_size),
        "UpdateEpochs": int(args.update_epochs),
        "LearningRate": float(args.learning_rate),
        "EarlyStop": int(args.early_stop),
        "NumWorkers": int(cli.num_workers),
        "SmokeTest": bool(cli.smoke_test),
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "Torch": torch.__version__,
        "TorchCUDARuntime": torch.version.cuda,
        "GPU": torch.cuda.get_device_name(args.device),
        "GPUCapability": "{}.{}".format(
            *torch.cuda.get_device_capability(args.device)
        ),
        "GitBranch": git_value("branch", "--show-current"),
        "GitCommit": git_value("rev-parse", "HEAD"),
        "StartedOrCompletedUTC": utc_now(),
        "LogPath": str(log_path.resolve()),
        **{
            "valid_{}".format(key): value
            for key, value in flatten_mode_metrics(final_valid).items()
        },
    }

    per_seed = output_dir / "{}_per_seed.csv".format(cli.dataset)
    summary = output_dir / "{}_summary.csv".format(cli.dataset)
    predictions_path = (
        output_dir / "{}_best_valid_predictions.csv".format(cli.dataset)
    )
    pd.DataFrame([row]).to_csv(per_seed, index=False)
    numeric = pd.DataFrame([row]).select_dtypes(include="number")
    pd.DataFrame(
        {
            "Metric": numeric.columns,
            "Mean": numeric.iloc[0].values,
            "Std": [0.0] * len(numeric.columns),
        }
    ).to_csv(summary, index=False)
    predictions.to_csv(predictions_path, index=False)

    logger.info(
        "complete seed=%s best_epoch=%s J_valid=%.6f checkpoint=%s sha=%s",
        cli.seed,
        row["BestValidEpoch"],
        final_j,
        checkpoint,
        checkpoint_hash,
    )
    for handler in logger.handlers:
        handler.flush()

    manifest = {
        "version": VERSION,
        "action": "moddrop",
        "seed": int(cli.seed),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "checkpoint_selection": (
            "minimum J_valid = 0.5*MAE_LAV + "
            "0.5*mean(MAE_LA,MAE_LV,MAE_L)"
        ),
        "missing_mask_generator_seed": int(cli.seed) + 104729,
        "source_clean_checkpoint": {
            "path": str(clean_checkpoint.resolve()),
            "sha256": clean_hash,
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
            "checkpoint": {
                "path": str(checkpoint.resolve()),
                "sha256": checkpoint_hash,
            },
            "per_seed_csv": {
                "path": str(per_seed.resolve()),
                "sha256": sha256(per_seed),
            },
            "summary_csv": {
                "path": str(summary.resolve()),
                "sha256": sha256(summary),
            },
            "best_valid_predictions_csv": {
                "path": str(predictions_path.resolve()),
                "sha256": sha256(predictions_path),
            },
            "log": {
                "path": str(log_path.resolve()),
                "sha256": sha256(log_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame([row]).to_string(index=False))
    print("manifest:", manifest_path)
    return row


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
