"""MOSEI port of the frozen DLF-CFCompatKD-v1 member trainer.

This file produces the five validation-selected CFCompat-v1 student members that
feed Raw5. It deliberately does not construct or read the official MOSEI Test
split. The mathematical method is the frozen Stage-3B CFCompat-v1 method:

    L = L_full + L_missing + L_KD
    L_KD = sum(compat_i * SmoothL1(student_missing_i, teacher_LAV_i))
           / (sum(compat_i) + 1e-8)

Compatibility is the train-only per-mode empirical-rank transform of the frozen
Stage-1 ModDrop evaluator's |LAV - missing| counterfactual delta.

For the 8 GiB Windows GPU, the full-view term is backpropagated before the
missing/KD graph is constructed, with no optimizer step between the two
backward calls. Thus the accumulated gradient remains grad(L_full) +
grad(L_missing + L_KD), while peak activation memory is lower. The forward/RNG
order and optimizer-step boundaries are unchanged from the frozen method.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import platform
import shutil
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import get_config_regression
from data_loader import MMDataLoader
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    build_counterfactual_cache,
    build_frozen_evaluator,
    compatibility_for_modes,
    distribution_stats,
    effective_sample_size,
    gated_kd_loss,
    load_counterfactual_cache,
    locate_stage1_evaluator,
    modes_from_masks,
    write_counterfactual_cache,
)
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence,
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    build_single_split_loader,
    clean_checkpoint_path,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    flatten_mode_metrics,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


DATASET = "mosei"
FORMAL_SEEDS = (1111, 1112, 1113, 1114, 1115)
EXPECTED_TRAIN_N = 16326
METHOD = "DLF-CFCompatKD-v1"
VERSION = "mosei_cfcompat_v1"
CACHE_VERSION = "mosei_cf_compat_v1"
SMOKE_CACHE_VERSION = "mosei_cf_compat_v1_smoke"
MISSING_STREAM_OFFSET = 104729


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSEI CFCompat-v1 Valid-only port")
    parser.add_argument("--action", choices=("cache", "train"), required=True)
    parser.add_argument("--seed", type=int, choices=FORMAL_SEEDS, required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline/mosei_cfcompat_v1")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 0:
        parser.error("Windows MOSEI CFCompat fixes --num-workers=0.")
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_config(cli: argparse.Namespace) -> Any:
    args = get_config_regression("DLF", DATASET, cli.config_file)
    args.mode = "train"
    args.feature_T = ""
    args.feature_A = ""
    args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = int(cli.seed)
    args.cur_seed = int(cli.seed)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def batch_to_device(batch: Dict[str, Any], device: torch.device):
    return (
        batch["text"].to(device),
        batch["audio"].to(device),
        batch["vision"].to(device),
        batch["labels"]["M"].to(device).view(-1, 1),
    )


def stage0_manifest_path(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / "mosei_clean_stage0_v1"
        / f"seed{cli.seed}"
        / "run_manifest.json"
    )


def stage1_manifest_path(cli: argparse.Namespace) -> Path:
    return (
        Path(cli.result_root)
        / "missing_baseline"
        / "mosei_moddrop_stage1_v1"
        / f"seed{cli.seed}"
        / "run_manifest.json"
    )


def verify_clean_source(cli: argparse.Namespace, args: Any) -> tuple[Path, str, Dict[str, Any]]:
    checkpoint = clean_checkpoint_path(cli.model_save_dir, DATASET, cli.seed)
    manifest_path = stage0_manifest_path(cli)
    if not checkpoint.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(f"Missing clean Stage-0 source: {checkpoint} / {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    actual = sha256(checkpoint)
    recorded = str(manifest.get("checkpoint_sha256", "")).lower()
    if actual != recorded:
        raise RuntimeError(f"Stage-0 checkpoint SHA mismatch: actual={actual} manifest={recorded}")
    if manifest.get("dataset") != DATASET or int(manifest.get("seed", -1)) != int(cli.seed):
        raise RuntimeError("Stage-0 manifest dataset/seed binding failed.")
    if bool(manifest.get("test_constructed", True)):
        raise RuntimeError("Stage-0 training manifest unexpectedly constructed Test.")
    if int(manifest.get("batch_size", -1)) != int(args.batch_size):
        raise RuntimeError("Stage-0 batch size differs from current MOSEI config.")
    if int(manifest.get("update_epochs", -1)) != int(args.update_epochs):
        raise RuntimeError("Stage-0 update_epochs differs from current MOSEI config.")
    return checkpoint, actual, manifest


def verify_stage1_source(cli: argparse.Namespace) -> tuple[Path, int, str, Dict[str, Any]]:
    checkpoint, best_epoch, source_csv = locate_stage1_evaluator(
        cli.result_root, DATASET, cli.seed, multiseed=False, smoke=False
    )
    manifest_path = stage1_manifest_path(cli)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing Stage-1 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    binding = manifest.get("stage1_checkpoint", {})
    recorded_sha = str(binding.get("sha256", "")).lower()
    actual_sha = sha256(checkpoint)
    if actual_sha != recorded_sha:
        raise RuntimeError(
            f"Stage-1 checkpoint SHA mismatch seed={cli.seed}: actual={actual_sha} manifest={recorded_sha}"
        )
    if manifest.get("dataset") != DATASET or int(manifest.get("seed", -1)) != int(cli.seed):
        raise RuntimeError("Stage-1 manifest dataset/seed binding failed.")
    if bool(manifest.get("official_test_constructed", True)):
        raise RuntimeError("Stage-1 formal training unexpectedly constructed Test.")
    if int(manifest.get("training", {}).get("best_epoch", -1)) != int(best_epoch):
        raise RuntimeError("Stage-1 CSV/manifest best-epoch binding failed.")
    return checkpoint, int(best_epoch), actual_sha, manifest


def cache_version(cli: argparse.Namespace) -> str:
    return SMOKE_CACHE_VERSION if cli.smoke_test else CACHE_VERSION


def cache_manifest_path(cli: argparse.Namespace) -> Path:
    root = Path(cli.result_root) / "missing_baseline" / VERSION / "cache"
    if cli.smoke_test:
        root = root / "smoke"
    return root / f"seed{cli.seed}" / "cache_manifest.json"


def train_output_dir(cli: argparse.Namespace) -> Path:
    root = Path(cli.result_root) / "missing_baseline" / VERSION
    if cli.smoke_test:
        root = root / "smoke"
    return root / f"seed{cli.seed}"


def checkpoint_path(cli: argparse.Namespace) -> Path:
    root = Path(cli.model_save_dir) / "missing_baseline" / "cf_compat_kd_v1" / DATASET
    if cli.smoke_test:
        root = root / "smoke"
    return root / f"seed{cli.seed}" / f"DLF_{DATASET}_seed{cli.seed}_best_valid.pth"


def canonical_result_paths(cli: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(cli.result_root) / "missing_baseline" / "cf_compat_kd_v1"
    return root / f"{DATASET}_per_seed.csv", root / f"{DATASET}_summary.csv"


def raw_member_prediction_path(cli: argparse.Namespace) -> Path:
    root = Path(cli.result_root) / "missing_baseline" / "cfcompat_prediction_ensemble_v1" / DATASET
    return root / f"online_seed{cli.seed}_valid_predictions.csv"


def create_logger(cli: argparse.Namespace) -> tuple[logging.Logger, Path]:
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / f"DLF-mosei-cfcompat-v1-seed{cli.seed}-{cli.action}-{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger = logging.getLogger("mosei_cfcompat_v1")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def atomic_write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    os.replace(tmp, path)


def _json_write(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _grad_norm(gradients: Iterable[torch.Tensor | None]) -> float:
    return math.sqrt(sum(float(g.detach().pow(2).sum().item()) for g in gradients if g is not None))


def _finite_metrics(metrics: Dict[str, Dict[str, float]]) -> bool:
    return all(math.isfinite(float(v)) for by_mode in metrics.values() for v in by_mode.values())


def prediction_rows(model: nn.Module, loader, device: torch.device) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.inference_mode():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            predictions = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                predictions[mode] = model(text, audio, vision, mask)["output_logit"].view(-1).cpu().numpy()
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            identifiers = list(batch["id"])
            for offset, index in enumerate(indices):
                rows.append(
                    {
                        "sample_id": str(identifiers[offset]),
                        "sample_index": int(index),
                        "label": float(labels[offset].item()),
                        **{f"{mode}_pred": float(predictions[mode][offset]) for mode in predictions},
                    }
                )
    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if frame.sample_index.duplicated().any():
        raise RuntimeError("Duplicate sample_index in prediction replay.")
    return frame


def gate_epoch_summary(records: list[Dict[str, float]], seed: int, epoch: int) -> Dict[str, float]:
    frame = pd.DataFrame(records)
    if frame.empty:
        raise RuntimeError("No CFCompat gate records were collected.")
    stats = distribution_stats(frame.gate.to_numpy(float))
    ess = effective_sample_size(frame.gate.to_numpy(float))
    return {
        "Seed": int(seed),
        "Epoch": int(epoch),
        "SampleCount": int(len(frame)),
        **{f"gate_{k}": float(v) for k, v in stats.items()},
        "ESS": float(ess),
        "ESS_fraction": float(ess / len(frame)),
        "mean_compat": float(frame.compat.mean()),
        "mean_KD": float(frame.kd.mean()),
        "mean_student_missing_abs_label_error": float(frame.student_error.mean()),
        "mean_teacher_abs_label_error": float(frame.teacher_error.mean()),
        "mean_teacher_student_abs_gap": float(frame.teacher_student_gap.mean()),
    }


def prepare_cache_output(cli: argparse.Namespace) -> None:
    manifest = cache_manifest_path(cli)
    version = cache_version(cli)
    from trains.singleTask.cf_compat_kd_utils import cache_paths

    paths = cache_paths(cli.result_root, DATASET, version=version, seed=cli.seed)
    existing = any(Path(p).exists() for k, p in paths.items() if k != "directory") or manifest.exists()
    if existing and not cli.overwrite:
        raise FileExistsError(
            f"CFCompat cache output already exists for seed{cli.seed}; inspect it or pass --overwrite intentionally."
        )
    if cli.overwrite:
        if paths["directory"].exists():
            shutil.rmtree(paths["directory"])
        if manifest.parent.exists():
            shutil.rmtree(manifest.parent)


def build_cache(cli: argparse.Namespace, logger: logging.Logger, log_path: Path) -> None:
    setup_seed(int(cli.seed))
    args = build_config(cli)
    clean_checkpoint, clean_sha, _ = verify_clean_source(cli, args)
    evaluator_checkpoint, evaluator_best_epoch, evaluator_sha, _ = verify_stage1_source(cli)
    prepare_cache_output(cli)

    train_loader = build_single_split_loader(args, "train", cli.num_workers)
    observed_n = int(len(train_loader.dataset))
    if observed_n != EXPECTED_TRAIN_N:
        raise RuntimeError(f"MOSEI Train size mismatch: expected={EXPECTED_TRAIN_N} observed={observed_n}")

    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    missing_generator = torch.Generator().manual_seed(int(cli.seed) + MISSING_STREAM_OFFSET)
    frame = build_counterfactual_cache(evaluator, train_loader, args.device, missing_generator)
    if len(frame) != observed_n or frame.sample_index.nunique() != observed_n:
        raise RuntimeError("MOSEI CFCompat cache does not bind every Train sample exactly once.")
    if not np.array_equal(frame.sample_index.to_numpy(np.int64), np.arange(observed_n, dtype=np.int64)):
        raise RuntimeError("MOSEI CFCompat cache sample_index is not exactly 0..N-1.")

    version = cache_version(cli)
    paths, config = write_counterfactual_cache(
        frame,
        cli.result_root,
        DATASET,
        evaluator_checkpoint,
        evaluator_best_epoch,
        version=version,
        seed=cli.seed,
        rng_state_preserved=True,
    )
    if config["evaluator_sha256"] != evaluator_sha:
        raise RuntimeError("Written cache evaluator SHA differs from frozen Stage-1 source.")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    payload = {
        "version": VERSION,
        "action": "cache",
        "dataset": DATASET,
        "seed": int(cli.seed),
        "smoke_test": bool(cli.smoke_test),
        "source_split": "train_only",
        "official_test_constructed": False,
        "test_used_for_tuning": False,
        "train_sample_count": int(len(frame)),
        "cache_version": version,
        "cache_csv": str(paths["csv"]),
        "cache_sha256": sha256(paths["csv"]),
        "clean_teacher": {"path": str(clean_checkpoint), "sha256": clean_sha},
        "stage1_evaluator": {
            "path": str(evaluator_checkpoint),
            "sha256": evaluator_sha,
            "best_epoch": int(evaluator_best_epoch),
        },
        "compatibility_summary": summary,
        "environment": {
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device),
            "matmul_precision": cli.matmul_precision,
            "num_workers": int(cli.num_workers),
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "completed_utc": utc_now(),
            "log": str(log_path),
        },
    }
    manifest = cache_manifest_path(cli)
    _json_write(manifest, payload)

    logger.info(
        "cache complete seed=%s N=%s evaluator=%s evaluator_sha=%s cache=%s",
        cli.seed,
        len(frame),
        evaluator_checkpoint,
        evaluator_sha,
        paths["csv"],
    )
    for mode in MISSING_MODES:
        values = summary[mode]
        logger.info(
            "cache mode=%s compat=[%.6f, %.6f] delta_mean=%.6f spearman=%.9f",
            mode,
            values["compat"]["min"],
            values["compat"]["max"],
            values["delta"]["mean"],
            values["corr_delta_compat_spearman"],
        )

    print("================ MOSEI CFCompat-v1 cache complete =================")
    print("seed:", cli.seed)
    print("samples:", len(frame))
    print("cache:", paths["csv"])
    print("cache sha256:", sha256(paths["csv"]))
    print("stage1 evaluator sha256:", evaluator_sha)
    print("TEST CONSTRUCTED:", False)
    print("manifest:", manifest)


def initialize_teacher_student(args: Any, cli: argparse.Namespace, loaders):
    clean_checkpoint, clean_sha, _ = verify_clean_source(cli, args)
    teacher = build_frozen_teacher(DLF, args, clean_checkpoint)
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(clean_checkpoint, map_location="cpu"), strict=True)
    student = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    student.eval()
    first = next(iter(loaders["valid"]))
    text, audio, vision, _ = batch_to_device(first, args.device)
    assert_initial_lav_equivalence(teacher, student, text, audio, vision)
    return teacher, student, backbone, clean_checkpoint, clean_sha


def prepare_train_output(cli: argparse.Namespace) -> None:
    checkpoint = checkpoint_path(cli)
    output = train_output_dir(cli)
    raw_prediction = raw_member_prediction_path(cli) if not cli.smoke_test else None
    canonical, _ = canonical_result_paths(cli)
    canonical_row_exists = False
    if canonical.is_file() and not cli.smoke_test:
        frame = pd.read_csv(canonical)
        if "Seed" in frame.columns:
            canonical_row_exists = bool((frame.Seed.astype(int) == int(cli.seed)).any())
    exists = checkpoint.exists() or output.exists() or canonical_row_exists or (
        raw_prediction is not None and raw_prediction.exists()
    )
    if exists and not cli.overwrite:
        raise FileExistsError(
            f"Formal CFCompat output already exists for seed{cli.seed}; inspect it or pass --overwrite intentionally."
        )
    if cli.overwrite:
        if checkpoint.exists():
            checkpoint.unlink()
        if output.exists():
            shutil.rmtree(output)
        if raw_prediction is not None and raw_prediction.exists():
            raw_prediction.unlink()
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)


def update_canonical_results(cli: argparse.Namespace, row: Dict[str, Any]) -> tuple[Path, Path]:
    per_seed, summary = canonical_result_paths(cli)
    new = pd.DataFrame([row])
    if per_seed.is_file():
        old = pd.read_csv(per_seed)
        if "Seed" not in old.columns:
            raise RuntimeError(f"Existing canonical CFCompat CSV has no Seed column: {per_seed}")
        old = old.loc[old.Seed.astype(int) != int(cli.seed)].copy()
        frame = pd.concat([old, new], ignore_index=True, sort=False)
    else:
        frame = new
    frame = frame.sort_values("Seed", kind="mergesort").reset_index(drop=True)
    atomic_write_csv(frame, per_seed)
    numeric = frame.select_dtypes(include=["number"])
    summary_frame = pd.DataFrame(
        {"Metric": numeric.columns, "Mean": numeric.mean().values, "Std": numeric.std(ddof=0).values}
    )
    atomic_write_csv(summary_frame, summary)
    return per_seed, summary


def train(cli: argparse.Namespace, logger: logging.Logger, log_path: Path) -> None:
    setup_seed(int(cli.seed))
    args = build_config(cli)
    clean_checkpoint, clean_sha, _ = verify_clean_source(cli, args)
    evaluator_checkpoint, evaluator_best_epoch, evaluator_sha, _ = verify_stage1_source(cli)
    cache_frame, cache_by_index = load_counterfactual_cache(
        cli.result_root,
        DATASET,
        version=cache_version(cli),
        seed=cli.seed,
        expected_evaluator_sha=evaluator_sha,
    )
    if len(cache_frame) != EXPECTED_TRAIN_N or cache_frame.sample_index.nunique() != EXPECTED_TRAIN_N:
        raise RuntimeError(
            f"MOSEI CFCompat requires the audited {EXPECTED_TRAIN_N}-sample Train cache for seed{cli.seed}."
        )
    if not np.array_equal(
        np.sort(cache_frame.sample_index.to_numpy(np.int64)), np.arange(EXPECTED_TRAIN_N, dtype=np.int64)
    ):
        raise RuntimeError("CFCompat cache sample_index binding failed.")

    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError(f"MOSEI CFCompat must construct exactly train/valid, got {sorted(loaders)}")
    if len(loaders["train"].dataset) != EXPECTED_TRAIN_N:
        raise RuntimeError("Training loader/cache sample count mismatch.")

    prepare_train_output(cli)
    teacher, student, backbone, teacher_checkpoint, teacher_sha = initialize_teacher_student(args, cli, loaders)
    if teacher_sha != clean_sha or teacher_checkpoint != clean_checkpoint:
        raise RuntimeError("Clean teacher binding changed during initialization.")

    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=args.patience)
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(int(cli.seed) + MISSING_STREAM_OFFSET)

    checkpoint = checkpoint_path(cli)
    output = train_output_dir(cli)
    train_batches = len(loaders["train"])
    updates_per_epoch = int(math.ceil(train_batches / float(args.update_epochs)))
    max_epochs = 2 if cli.smoke_test else 1000

    best_j = float("inf")
    best_epoch = 0
    best_metrics = None
    epoch_rows = []
    gate_rows = []
    optimizer_steps_total = 0

    logger.info(
        "start method=%s dataset=%s seed=%s train_N=%s valid_N=%s test_constructed=false",
        METHOD,
        DATASET,
        cli.seed,
        len(loaders["train"].dataset),
        len(loaders["valid"].dataset),
    )
    logger.info(
        "teacher=%s teacher_sha=%s evaluator=%s evaluator_sha=%s evaluator_best_epoch=%s cache_version=%s",
        teacher_checkpoint,
        teacher_sha,
        evaluator_checkpoint,
        evaluator_sha,
        evaluator_best_epoch,
        cache_version(cli),
    )
    logger.info(
        "config batch=%s update_epochs=%s updates_per_epoch=%s lr=%s patience=%s early_stop=%s "
        "memory_execution=sequential_equivalent_backward",
        args.batch_size,
        args.update_epochs,
        updates_per_epoch,
        args.learning_rate,
        args.patience,
        args.early_stop,
    )

    for epoch in range(1, max_epochs + 1):
        student.train()
        optimizer.zero_grad(set_to_none=True)
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        records: list[Dict[str, float]] = []
        batch_kd_losses = []
        full_losses = []
        missing_losses = []
        optimizer_steps_epoch = 0

        for step, batch in enumerate(loaders["train"], start=1):
            text, audio, vision, labels = batch_to_device(batch, args.device)

            # Preserve the original forward/RNG order, but release the full-view
            # graph before constructing the missing/KD graph.
            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            full_output = student(text, audio, vision, full_mask)
            full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
            if not torch.isfinite(full_loss):
                raise FloatingPointError("Non-finite CFCompat full-view loss.")
            full_value = float(full_loss.detach().item())
            full_loss.backward()
            del full_output, full_loss

            missing_mask = sample_missing_masks(labels.size(0), missing_generator, args.device, audio.dtype)
            modes = modes_from_masks(missing_mask)
            counts.update(count_missing_modes(missing_mask))
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)

            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            compatibility = compatibility_for_modes(
                cache_by_index, indices, modes, args.device, labels.dtype
            )
            kd_loss, each_kd = gated_kd_loss(
                missing_output["output_logit"], teacher_prediction, compatibility
            )
            if step == 1 and epoch == 1:
                gradients = torch.autograd.grad(
                    kd_loss,
                    [p for p in student.parameters() if p.requires_grad],
                    retain_graph=True,
                    allow_unused=True,
                )
                if _grad_norm(gradients) <= 0:
                    raise RuntimeError("Compatibility-gated KD did not reach student parameters.")
                del gradients

            missing_kd = missing_loss + kd_loss
            if not torch.isfinite(missing_kd):
                raise FloatingPointError("Non-finite CFCompat missing/KD loss.")
            missing_value = float(missing_loss.detach().item())
            kd_value = float(kd_loss.detach().item())
            missing_kd.backward()

            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen full teacher received gradients.")

            if step % args.update_epochs == 0 or step == train_batches:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps_epoch += 1
                optimizer_steps_total += 1

            denominator = float(compatibility.sum().detach().cpu()) + 1e-8
            student_error = torch.abs(missing_output["output_logit"].detach().view(-1) - labels.view(-1))
            teacher_error = torch.abs(teacher_prediction.detach().view(-1) - labels.view(-1))
            teacher_student_gap = torch.abs(
                missing_output["output_logit"].detach().view(-1) - teacher_prediction.detach().view(-1)
            )
            for offset, index in enumerate(indices):
                records.append(
                    {
                        "sample_index": int(index),
                        "mode": modes[offset],
                        "compat": float(compatibility[offset].detach()),
                        "kd": float(each_kd[offset].detach()),
                        "weighted_kd_contribution": float(
                            compatibility[offset].detach() * each_kd[offset].detach() / denominator
                        ),
                        "student_error": float(student_error[offset]),
                        "teacher_error": float(teacher_error[offset]),
                        "teacher_student_gap": float(teacher_student_gap[offset]),
                    }
                )

            full_losses.append(full_value)
            missing_losses.append(missing_value)
            batch_kd_losses.append(kd_value)
            del missing_output, missing_loss, kd_loss, missing_kd, teacher_prediction, each_kd

        if optimizer_steps_epoch != updates_per_epoch:
            raise RuntimeError(
                f"Unexpected optimizer-step count epoch={epoch}: {optimizer_steps_epoch} != {updates_per_epoch}"
            )

        valid = evaluate_all_modes(student, loaders["valid"], args.device, "moddrop", criterion)
        if not _finite_metrics(valid):
            raise FloatingPointError("Non-finite CFCompat Valid metrics.")
        j_valid = float(validation_objective(valid))
        scheduler.step(j_valid)
        current_lr = float(optimizer.param_groups[0]["lr"])
        is_best = j_valid <= best_j - 1e-6
        if is_best:
            best_j = j_valid
            best_epoch = epoch
            best_metrics = valid
            torch.save(student.state_dict(), checkpoint)

        gate_summary = gate_epoch_summary(records, cli.seed, epoch)
        gate_rows.append(gate_summary)
        row = {
            "Seed": int(cli.seed),
            "Epoch": int(epoch),
            "J_valid": j_valid,
            "IsBestValid": bool(is_best),
            "FullLoss": float(np.mean(full_losses)),
            "MissingLoss": float(np.mean(missing_losses)),
            "KD_loss": float(np.mean(batch_kd_losses)),
            "LR": current_lr,
            "OptimizerStepsEpoch": int(optimizer_steps_epoch),
            "OptimizerStepsTotal": int(optimizer_steps_total),
            "LA_count": int(counts["LA"]),
            "LV_count": int(counts["LV"]),
            "L_count": int(counts["L"]),
            "ESS": gate_summary["ESS"],
            "ESS_fraction": gate_summary["ESS_fraction"],
            "gate_mean": gate_summary["gate_mean"],
            **{f"valid_{k}": v for k, v in flatten_mode_metrics(valid).items()},
        }
        epoch_rows.append(row)

        logger.info(
            "seed=%s epoch=%s steps=%s/%s LA=%s LV=%s L=%s full=%.6f missing=%.6f KD=%.6f "
            "J_valid=%.6f gate=%.6f ESS_fraction=%.6f lr=%.8g best=%s",
            cli.seed,
            epoch,
            optimizer_steps_epoch,
            optimizer_steps_total,
            counts["LA"],
            counts["LV"],
            counts["L"],
            row["FullLoss"],
            row["MissingLoss"],
            row["KD_loss"],
            j_valid,
            row["gate_mean"],
            row["ESS_fraction"],
            current_lr,
            is_best,
        )
        for mode, metrics in valid.items():
            logger.info(
                "seed=%s epoch=%s valid_%s MAE=%.6f Corr=%.6f Acc2=%.6f F1=%.6f Acc7=%.6f Acc5=%.6f",
                cli.seed,
                epoch,
                mode,
                metrics["MAE"],
                metrics["Corr"],
                metrics["acc_2"],
                metrics["F1_score"],
                metrics["acc_7"],
                metrics["acc_5"],
            )

        if cli.smoke_test and epoch >= 2:
            break
        if epoch - best_epoch >= int(args.early_stop):
            break

    if best_metrics is None or not checkpoint.is_file():
        raise RuntimeError("No validation-best CFCompat checkpoint was saved.")

    # Release all training-time owners before strict reload/audit.
    del optimizer, scheduler, cosine, hinge
    del teacher, student, backbone
    torch.cuda.empty_cache()

    audit_backbone = DLF(args).to(args.device)
    audit_student = MissingModalityWrapper(
        audit_backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    audit_student.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    audit_metrics = evaluate_all_modes(audit_student, loaders["valid"], args.device, "moddrop", criterion)
    audit_j = float(validation_objective(audit_metrics))
    if abs(audit_j - best_j) > 1e-8:
        raise RuntimeError(f"Fresh validation replay differs: saved={best_j} replay={audit_j}")

    valid_predictions = prediction_rows(audit_student, loaders["valid"], args.device)
    if len(valid_predictions) != len(loaders["valid"].dataset):
        raise RuntimeError("Valid prediction replay length mismatch.")
    valid_predictions["Seed"] = int(cli.seed)
    valid_predictions["Method"] = "Online"
    valid_predictions["Split"] = "valid"
    valid_predictions["SelectedBy"] = "validation_J"

    epoch_csv = output / "epoch_metrics.csv"
    gate_csv = output / "gate_summary.csv"
    local_prediction_csv = output / "best_valid_predictions.csv"
    atomic_write_csv(pd.DataFrame(epoch_rows), epoch_csv)
    atomic_write_csv(pd.DataFrame(gate_rows), gate_csv)
    atomic_write_csv(valid_predictions, local_prediction_csv)

    raw_prediction = None
    if not cli.smoke_test:
        raw_prediction = raw_member_prediction_path(cli)
        atomic_write_csv(valid_predictions, raw_prediction)

    missing_macro = float(np.mean([audit_metrics[m]["MAE"] for m in MISSING_MODES]))
    row: Dict[str, Any] = {
        "Seed": int(cli.seed),
        "Method": METHOD,
        "BestValidEpoch": int(best_epoch),
        "J_valid": float(audit_j),
        "LAV_MAE": float(audit_metrics["LAV"]["MAE"]),
        "MissingMacro_MAE": missing_macro,
        "Checkpoint": str(checkpoint),
        "CheckpointSHA256": sha256(checkpoint),
        "TeacherCheckpoint": str(teacher_checkpoint),
        "TeacherSHA256": teacher_sha,
        "EvaluatorCheckpoint": str(evaluator_checkpoint),
        "EvaluatorSHA256": evaluator_sha,
        "EvaluatorBestEpoch": int(evaluator_best_epoch),
        "CacheVersion": cache_version(cli),
        "CacheSHA256": checkpoint_sha256(
            Path(cli.result_root)
            / "counterfactual_compatibility"
            / cache_version(cli)
            / DATASET
            / f"seed{cli.seed}"
            / "train_counterfactual_compatibility.csv"
        ),
        "BatchSize": int(args.batch_size),
        "UpdateEpochs": int(args.update_epochs),
        "OptimizerUpdatesPerEpoch": int(updates_per_epoch),
        "LearningRate": float(args.learning_rate),
        "Patience": int(args.patience),
        "EarlyStop": int(args.early_stop),
        "MemoryExecution": "sequential_equivalent_backward",
        "TestConstructed": False,
        **flatten_mode_metrics(audit_metrics),
    }

    canonical_paths = None
    if not cli.smoke_test:
        canonical_paths = update_canonical_results(cli, row)

    manifest = {
        "version": VERSION,
        "method": METHOD,
        "dataset": DATASET,
        "seed": int(cli.seed),
        "smoke_test": bool(cli.smoke_test),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_used_for_tuning": False,
        "objective": "L_full + L_missing + compatibility_gated_SmoothL1_prediction_KD",
        "checkpoint_selection": "min J_valid = 0.5*MAE_LAV + 0.5*mean(MAE_LA,MAE_LV,MAE_L)",
        "memory_execution": "sequential_equivalent_backward",
        "memory_equivalence": "backward(L_full) + backward(L_missing + L_KD) before any optimizer.step",
        "missing_mask_generator_seed": int(cli.seed) + MISSING_STREAM_OFFSET,
        "training": {
            "batch_size": int(args.batch_size),
            "update_epochs": int(args.update_epochs),
            "optimizer_updates_per_epoch": int(updates_per_epoch),
            "optimizer_steps_total_at_stop": int(optimizer_steps_total),
            "learning_rate": float(args.learning_rate),
            "patience": int(args.patience),
            "early_stop": int(args.early_stop),
            "best_epoch": int(best_epoch),
            "best_J_valid": float(audit_j),
            "num_workers": int(cli.num_workers),
            "matmul_precision": cli.matmul_precision,
        },
        "clean_teacher": {"path": str(teacher_checkpoint), "sha256": teacher_sha},
        "stage1_evaluator": {
            "path": str(evaluator_checkpoint),
            "sha256": evaluator_sha,
            "best_epoch": int(evaluator_best_epoch),
        },
        "cache": {
            "version": cache_version(cli),
            "sha256": row["CacheSHA256"],
            "train_sample_count": EXPECTED_TRAIN_N,
        },
        "checkpoint": {"path": str(checkpoint), "sha256": row["CheckpointSHA256"]},
        "best_valid_metrics": audit_metrics,
        "valid_prediction_csv": str(local_prediction_csv),
        "raw5_member_valid_prediction_csv": None if raw_prediction is None else str(raw_prediction),
        "canonical_results": None if canonical_paths is None else {
            "per_seed_csv": str(canonical_paths[0]),
            "summary_csv": str(canonical_paths[1]),
        },
        "environment": {
            "python_platform": platform.platform(),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(args.device),
            "git_branch": git_value("branch", "--show-current"),
            "git_commit": git_value("rev-parse", "HEAD"),
            "completed_utc": utc_now(),
            "log": str(log_path),
        },
        "canonical_row": row,
    }
    manifest_path = output / "run_manifest.json"
    _json_write(manifest_path, manifest)

    del audit_student, audit_backbone
    torch.cuda.empty_cache()

    print("================ MOSEI CFCompat-v1 training complete ==============")
    print("seed:", cli.seed)
    print("best_epoch:", best_epoch)
    print("J_valid:", f"{audit_j:.9f}")
    print("LAV_MAE:", f"{audit_metrics['LAV']['MAE']:.9f}")
    print("MissingMacro_MAE:", f"{missing_macro:.9f}")
    print("checkpoint:", checkpoint)
    print("sha256:", row["CheckpointSHA256"])
    print("TEST CONSTRUCTED:", False)
    print("manifest:", manifest_path)
    if raw_prediction is not None:
        print("Raw5 valid member:", raw_prediction)


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MOSEI CFCompat.")
    torch.cuda.set_device(int(cli.gpu_id))
    torch.set_float32_matmul_precision(cli.matmul_precision)
    torch.backends.cudnn.allow_tf32 = cli.matmul_precision != "highest"

    logger, log_path = create_logger(cli)
    if cli.action == "cache":
        build_cache(cli, logger, log_path)
    else:
        train(cli, logger, log_path)


if __name__ == "__main__":
    main()
