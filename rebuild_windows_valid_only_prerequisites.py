"""Rebuild Windows-native, validation-only prerequisites for Safe-CFCompatKD.

Current implementation:
  * preflight
  * clean-dlf

The remaining ModDrop, train-only compatibility-cache, and CFCompatKD actions
will be added after the clean DLF Seed-1112 smoke run passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

# Set deterministic CUDA workspace configuration before importing torch.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pandas as pd
import torch
from transformers import BertModel, BertTokenizer

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.model.DLF import DLF
from trains.singleTask.windows_valid_only_clean_dlf import (
    CleanDLFValidOnlyTrainer,
)
from utils.functions import assign_gpu, setup_seed


VERSION = "windows_valid_only_prereq_v1"
FORMAL_SEEDS = (1112, 1113, 1115)
ACTIONS = ("preflight", "clean-dlf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Windows-native Valid-only prerequisite reconstruction."
    )
    parser.add_argument("--action", required=True, choices=ACTIONS)
    parser.add_argument("--seed", type=int, default=1112, choices=FORMAL_SEEDS)
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
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
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


def create_logger(args: argparse.Namespace) -> tuple[logging.Logger, Path]:
    directory = Path(args.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if args.smoke_test else "formal"
    path = directory / "clean-dlf-seed{}-{}-{}.log".format(
        args.seed,
        kind,
        datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    logger = logging.getLogger("windows_valid_only_prereq")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def build_config(cli: argparse.Namespace):
    args = get_config_regression("DLF", cli.dataset, cli.config_file)
    args.is_training = True
    args.mode = "train"
    args.train_mode = "regression"
    args.feature_T = ""
    args.feature_A = ""
    args.feature_V = ""
    args.seed = int(cli.seed)
    args.cur_seed = int(cli.seed)
    args.device = assign_gpu(list(cli.gpu_ids))
    return args


def output_paths(cli: argparse.Namespace) -> tuple[Path, Path]:
    if cli.smoke_test:
        output = (
            Path(cli.result_root)
            / VERSION
            / "smoke"
            / "clean_dlf"
            / "seed{}".format(cli.seed)
        )
        checkpoint = (
            Path(cli.model_save_dir)
            / VERSION
            / "smoke"
            / "clean_dlf"
            / "seed{}".format(cli.seed)
            / "DLF_{}_seed{}_best.pth".format(cli.dataset, cli.seed)
        )
    else:
        output = (
            Path(cli.result_root)
            / VERSION
            / "clean_dlf"
            / "seed{}".format(cli.seed)
        )
        checkpoint = (
            Path(cli.model_save_dir)
            / "DLF_{}_seed{}_best.pth".format(cli.dataset, cli.seed)
        )
    return output, checkpoint


def inspect_dataset(config) -> Dict[str, int]:
    path = Path(config.featurePath)
    if not path.is_file():
        raise FileNotFoundError("MOSI feature file is absent: {}".format(path))
    with path.open("rb") as handle:
        data = pickle.load(handle)
    for split in ("train", "valid"):
        if split not in data or "regression_labels" not in data[split]:
            raise ValueError("MOSI feature file lacks {} labels.".format(split))
    return {
        "train": int(len(data["train"]["regression_labels"])),
        "valid": int(len(data["valid"]["regression_labels"])),
    }


def preflight(cli: argparse.Namespace) -> Dict[str, Any]:
    config = build_config(cli)
    counts = inspect_dataset(config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the active Python environment.")
    if config.device.type != "cuda":
        raise RuntimeError("DLF preflight did not select a CUDA device.")

    # Force one real CUDA operation, not only device discovery.
    probe = torch.randn(64, 64, device=config.device)
    probe = probe @ probe
    torch.cuda.synchronize(config.device)
    if not torch.isfinite(probe).all():
        raise FloatingPointError("CUDA matrix-multiplication preflight was non-finite.")

    local_bert = Path(config.pretrained)
    if not local_bert.is_dir():
        raise FileNotFoundError(
            "Local pretrained BERT directory is absent: {}".format(local_bert)
        )
    BertTokenizer.from_pretrained(str(local_bert), local_files_only=True)
    bert = BertModel.from_pretrained(str(local_bert), local_files_only=True)
    del bert

    information = {
        "version": VERSION,
        "action": "preflight",
        "seed": int(cli.seed),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(config.device),
        "gpu": torch.cuda.get_device_name(config.device),
        "gpu_capability": list(torch.cuda.get_device_capability(config.device)),
        "dataset_path": str(Path(config.featurePath).resolve()),
        "train_samples": counts["train"],
        "valid_samples": counts["valid"],
        "bert_path": str(local_bert.resolve()),
        "git_branch": git_value("branch", "--show-current"),
        "git_commit": git_value("rev-parse", "HEAD"),
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
    }
    print(json.dumps(information, indent=2, ensure_ascii=False))
    return information


def run_clean_dlf(cli: argparse.Namespace) -> Dict[str, Any]:
    output_dir, checkpoint = output_paths(cli)
    if (output_dir.exists() or checkpoint.exists()) and not cli.overwrite:
        raise FileExistsError(
            "Output already exists. Re-run with --overwrite after checking it: "
            "{} or {}".format(output_dir, checkpoint)
        )
    if cli.overwrite:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if checkpoint.exists():
            checkpoint.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    logger, log_path = create_logger(cli)

    setup_seed(int(cli.seed))
    config = build_config(cli)
    counts = inspect_dataset(config)
    loaders = MMDataLoader(config, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Clean DLF unexpectedly constructed a non-Valid split.")

    logger.info(
        "start version=%s seed=%s smoke=%s device=%s torch=%s cuda=%s "
        "test_constructed=false",
        VERSION,
        cli.seed,
        cli.smoke_test,
        config.device,
        torch.__version__,
        torch.version.cuda,
    )

    model = DLF(config).to(config.device)
    trainer = CleanDLFValidOnlyTrainer(config, logger)
    result, epoch_rows = trainer.train(
        model,
        loaders,
        checkpoint,
        max_epochs=cli.max_epochs,
    )

    # Strict reload audit in a fresh model instance.
    del model
    torch.cuda.empty_cache()
    setup_seed(int(cli.seed))
    audit_model = DLF(config).to(config.device)
    audit_state = torch.load(checkpoint, map_location=config.device)
    audit_model.load_state_dict(audit_state, strict=True)
    audit_valid = trainer.evaluate(audit_model, loaders["valid"])
    if abs(float(audit_valid["Loss"]) - float(result["BestValidLoss"])) > 1e-8:
        raise RuntimeError("Fresh-model checkpoint reload audit failed.")

    checkpoint_hash = sha256(checkpoint)
    row: Dict[str, Any] = {
        "Seed": int(cli.seed),
        "Method": "DLF-Clean-Windows-ValidOnly-v1",
        **result,
        "Checkpoint": str(checkpoint.resolve()),
        "CheckpointSHA256": checkpoint_hash,
        "CheckpointSizeBytes": int(checkpoint.stat().st_size),
        "TrainSamples": counts["train"],
        "ValidSamples": counts["valid"],
        "BatchSize": int(config.batch_size),
        "UpdateEpochs": int(config.update_epochs),
        "LearningRate": float(config.learning_rate),
        "EarlyStop": int(config.early_stop),
        "NumWorkers": int(cli.num_workers),
        "SmokeTest": bool(cli.smoke_test),
        "Python": sys.version.split()[0],
        "Platform": platform.platform(),
        "Torch": torch.__version__,
        "TorchCUDARuntime": torch.version.cuda,
        "GPU": torch.cuda.get_device_name(config.device),
        "GPUCapability": "{}.{}".format(
            *torch.cuda.get_device_capability(config.device)
        ),
        "GitBranch": git_value("branch", "--show-current"),
        "GitCommit": git_value("rev-parse", "HEAD"),
        "StartedOrCompletedUTC": utc_now(),
        "LogPath": str(log_path.resolve()),
    }

    per_seed = output_dir / "{}_per_seed.csv".format(cli.dataset)
    epochs = output_dir / "{}_epoch_metrics.csv".format(cli.dataset)
    pd.DataFrame([row]).to_csv(per_seed, index=False)
    pd.DataFrame(epoch_rows).to_csv(epochs, index=False)

    logger.info(
        "complete seed=%s best_epoch=%s best_valid_loss=%.4f checkpoint=%s sha=%s",
        cli.seed,
        row["BestValidEpoch"],
        row["BestValidLoss"],
        checkpoint,
        checkpoint_hash,
    )
    for handler in logger.handlers:
        handler.flush()

    manifest = {
        "version": VERSION,
        "action": "clean-dlf",
        "seed": int(cli.seed),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "checkpoint_selection": "minimum rounded validation output-logit L1 loss",
        "preserved_original_partial_accumulation_drop": True,
        "source_branch": "feature/cfcompat-safe-projection-valid-screen-v1",
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
            "epoch_metrics_csv": {
                "path": str(epochs.resolve()),
                "sha256": sha256(epochs),
            },
            "log": {
                "path": str(log_path.resolve()),
                "sha256": sha256(log_path),
            },
        },
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8",
    )

    print(pd.DataFrame([row]).to_string(index=False))
    print("manifest:", manifest_path)
    return row


def main() -> None:
    cli = parse_args()
    if cli.action == "preflight":
        preflight(cli)
    elif cli.action == "clean-dlf":
        run_clean_dlf(cli)
    else:
        raise AssertionError("Unhandled action: {}".format(cli.action))


if __name__ == "__main__":
    main()
