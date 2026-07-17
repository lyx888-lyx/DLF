"""Train one validation-only Stage 10 asset; this module cannot access test."""
import argparse
import itertools
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import get_config_regression
from data_loader import MMDataLoader
from scripts.mosei.stage10_common import (
    SEEDS,
    atomic_json,
    git_head,
    sha256,
    stage_directory,
    utc_now,
)
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    build_counterfactual_cache,
    build_frozen_evaluator,
    compatibility_for_modes,
    gated_kd_loss,
    modes_from_masks,
)
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence,
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    flatten_mode_metrics,
    mode_to_mask,
    regression_metrics,
    sample_missing_masks,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import assign_gpu, setup_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage", required=True, choices=("clean", "moddrop", "compatibility", "cfcompat")
    )
    parser.add_argument("--dataset", choices=("mosei",), default="mosei")
    parser.add_argument("--seed", required=True, type=int, choices=SEEDS)
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--config-file", required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--smoke-batches", type=int, default=0)
    parser.add_argument("--max-epochs", type=int)
    args = parser.parse_args()
    if args.smoke_batches < 0:
        parser.error("--smoke-batches must be nonnegative.")
    if args.max_epochs is not None and args.max_epochs < 1:
        parser.error("--max-epochs must be positive.")
    return args


def build_args(cli):
    args = get_config_regression("DLF", "mosei", cli.config_file)
    args.mode = "train"
    args.feature_T = args.feature_A = args.feature_V = ""
    args.is_training = True
    args.train_mode = "regression"
    args.seed = args.cur_seed = int(cli.seed)
    args.device = assign_gpu([int(cli.gpu_id)])
    return args


def _iter_limited(loader, limit):
    for index, batch in enumerate(loader, 1):
        if limit and index > limit:
            break
        yield index, batch


def _output_record(path):
    path = Path(path)
    return {"Path": str(path), "SHA256": sha256(path), "Bytes": path.stat().st_size}


def _write_manifest(cli, output_dir, outputs, validation_metrics, best_epoch, extra=None):
    manifest = {
        "Dataset": "mosei",
        "Seed": int(cli.seed),
        "Stage": cli.stage,
        "Commit": git_head(ROOT),
        "ValidationSelected": True,
        "SelectionSplit": "valid",
        "SelectionMetric": "J" if cli.stage in ("moddrop", "cfcompat") else "Loss",
        "BestValidEpoch": int(best_epoch),
        "ValidationMetrics": validation_metrics,
        "NoTestAccess": True,
        "TestLoaderConstructed": False,
        "SmokeBatches": int(cli.smoke_batches),
        "Outputs": [_output_record(path) for path in outputs],
        "CompletedAt": utc_now(),
    }
    manifest.update(extra or {})
    atomic_json(output_dir / "stage_manifest.json", manifest)
    return manifest


def _finite_metrics(metrics):
    return all(
        math.isfinite(float(value))
        for mode in metrics.values()
        for value in mode.values()
    )


def _clean_eval(model, loader, device, limit=0):
    model.eval()
    predictions, labels, losses = [], [], []
    criterion = nn.L1Loss()
    with torch.no_grad():
        for _, batch in _iter_limited(loader, limit):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            target = batch["labels"]["M"].to(device).view(-1, 1)
            prediction = model(text, audio, vision)["output_logit"]
            predictions.append(prediction.cpu())
            labels.append(target.cpu())
            losses.append(float(criterion(prediction, target)))
    result = regression_metrics(torch.cat(predictions), torch.cat(labels))
    result["Loss"] = float(np.mean(losses))
    return result


def _missing_eval(model, loader, device, limit=0):
    model.eval()
    criterion = nn.L1Loss()
    collected = {
        mode: {"pred": [], "label": [], "loss": []}
        for mode in ("LAV",) + MISSING_MODES
    }
    with torch.no_grad():
        for _, batch in _iter_limited(loader, limit):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["labels"]["M"].to(device).view(-1, 1)
            for mode in collected:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                prediction = model(text, audio, vision, mask)["output_logit"]
                collected[mode]["pred"].append(prediction.cpu())
                collected[mode]["label"].append(labels.cpu())
                collected[mode]["loss"].append(float(criterion(prediction, labels)))
    result = {}
    for mode, values in collected.items():
        result[mode] = regression_metrics(
            torch.cat(values["pred"]), torch.cat(values["label"])
        )
        result[mode]["Loss"] = float(np.mean(values["loss"]))
    return result


def train_clean(cli, args, output_dir):
    setup_seed(cli.seed)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Clean training exposed a non-train/valid split.")
    model = DLF(args).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    checkpoint = output_dir / "DLF_mosei_seed{}_best_valid.pth".format(cli.seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, best_metrics = float("inf"), 0, None
    epoch_rows = []
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        model.train()
        optimizer.zero_grad()
        total = 0.0
        batches = 0
        for step, batch in _iter_limited(loaders["train"], cli.smoke_batches):
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            loss, _ = compute_full_dlf_loss(
                model(text, audio, vision), labels, criterion, cosine, hinge
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite clean loss.")
            loss.backward()
            if step % args.update_epochs == 0 or (
                cli.smoke_batches and step == cli.smoke_batches
            ):
                nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            total += float(loss.detach())
            batches += 1
        if batches == 0:
            raise RuntimeError("Empty clean train loader.")
        if batches % args.update_epochs and not cli.smoke_batches:
            nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        valid = _clean_eval(model, loaders["valid"], args.device, cli.smoke_batches)
        score = float(valid["Loss"])
        scheduler.step(score)
        epoch_rows.append({"Epoch": epoch, "TrainLoss": total / batches, **valid})
        print("clean seed={} epoch={} valid_loss={:.6f}".format(cli.seed, epoch, score), flush=True)
        if score <= best - 1e-6:
            best, best_epoch, best_metrics = score, epoch, valid
            torch.save(model.state_dict(), checkpoint)
        if cli.max_epochs and epoch >= cli.max_epochs:
            break
        if epoch - best_epoch >= args.early_stop:
            break
    metrics_path = output_dir / "epoch_metrics.csv"
    pd.DataFrame(epoch_rows).to_csv(metrics_path, index=False)
    return _write_manifest(
        cli,
        output_dir,
        (checkpoint, metrics_path),
        {"LAV": best_metrics},
        best_epoch,
        {"Checkpoint": str(checkpoint), "CheckpointSHA256": sha256(checkpoint)},
    )


def train_moddrop(cli, args, output_dir):
    setup_seed(cli.seed)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("ModDrop training exposed a non-train/valid split.")
    clean_manifest = json.loads(
        (stage_directory(cli.result_root, "clean", cli.seed) / "stage_manifest.json").read_text()
    )
    clean = Path(clean_manifest["Checkpoint"])
    if sha256(clean) != clean_manifest["CheckpointSHA256"]:
        raise RuntimeError("Clean teacher SHA mismatch.")
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(clean, map_location=args.device), strict=True)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    generator = torch.Generator().manual_seed(int(cli.seed) + 104729)
    checkpoint = output_dir / "DLF_mosei_seed{}_best_valid.pth".format(cli.seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, best_metrics = float("inf"), 0, None
    epoch_rows = []
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        model.train()
        optimizer.zero_grad()
        total, batches = 0.0, 0
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        for step, batch in _iter_limited(loaders["train"], cli.smoke_batches):
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            full_loss, _ = compute_full_dlf_loss(
                model(text, audio, vision, full_mask),
                labels,
                criterion,
                cosine,
                hinge,
            )
            missing_mask = sample_missing_masks(
                labels.size(0), generator, args.device, audio.dtype
            )
            counts.update(count_missing_modes(missing_mask))
            missing_loss, _ = compute_task_loss(
                model(text, audio, vision, missing_mask), labels, criterion
            )
            loss = full_loss + missing_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite ModDrop loss.")
            loss.backward()
            if step % args.update_epochs == 0 or (
                cli.smoke_batches and step == cli.smoke_batches
            ):
                nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            total += float(loss.detach())
            batches += 1
        if batches % args.update_epochs and not cli.smoke_batches:
            nn.utils.clip_grad_value_(model.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        valid = _missing_eval(model, loaders["valid"], args.device, cli.smoke_batches)
        if not _finite_metrics(valid):
            raise FloatingPointError("Non-finite ModDrop validation metrics.")
        score = float(validation_objective(valid))
        scheduler.step(score)
        epoch_rows.append(
            {"Epoch": epoch, "TrainLoss": total / batches, "JValid": score, **flatten_mode_metrics(valid)}
        )
        print("moddrop seed={} epoch={} J_valid={:.6f}".format(cli.seed, epoch, score), flush=True)
        if score <= best - 1e-6:
            best, best_epoch, best_metrics = score, epoch, valid
            torch.save(model.state_dict(), checkpoint)
        if cli.max_epochs and epoch >= cli.max_epochs:
            break
        if epoch - best_epoch >= args.early_stop:
            break
    metrics_path = output_dir / "epoch_metrics.csv"
    pd.DataFrame(epoch_rows).to_csv(metrics_path, index=False)
    return _write_manifest(
        cli,
        output_dir,
        (checkpoint, metrics_path),
        best_metrics,
        best_epoch,
        {
            "JValid": best,
            "Checkpoint": str(checkpoint),
            "CheckpointSHA256": sha256(checkpoint),
            "CleanCheckpoint": str(clean),
            "CleanCheckpointSHA256": sha256(clean),
        },
    )


def build_compatibility(cli, args, output_dir):
    setup_seed(cli.seed)
    moddrop_manifest = json.loads(
        (stage_directory(cli.result_root, "moddrop", cli.seed) / "stage_manifest.json").read_text()
    )
    evaluator_checkpoint = Path(moddrop_manifest["Checkpoint"])
    if sha256(evaluator_checkpoint) != moddrop_manifest["CheckpointSHA256"]:
        raise RuntimeError("ModDrop evaluator SHA mismatch.")
    generator = torch.Generator().manual_seed(int(cli.seed) + 104729)
    train_loader = build_single_split_loader(args, "train", cli.num_workers)
    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    if any(parameter.requires_grad for parameter in evaluator.parameters()):
        raise RuntimeError("Evaluator is not frozen.")
    cache_loader = (
        list(itertools.islice(train_loader, cli.smoke_batches))
        if cli.smoke_batches
        else train_loader
    )
    frame = build_counterfactual_cache(evaluator, cache_loader, args.device, generator)
    expected_rows = (
        len(frame) if cli.smoke_batches else int(args.train_samples)
    )
    if len(frame) != expected_rows or not len(frame):
        raise RuntimeError("Compatibility cache is not train-only complete.")
    cache = output_dir / "train_counterfactual_compatibility.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(cache, index=False, float_format="%.17g")
    summary = output_dir / "cache_summary.json"
    atomic_json(
        summary,
        {
            "Rows": len(frame),
            "UniqueSampleIndices": int(frame.sample_index.nunique()),
            "Split": "train",
            "ContainsValid": False,
            "ContainsTest": False,
            "EvaluatorCheckpoint": str(evaluator_checkpoint),
            "EvaluatorSHA256": sha256(evaluator_checkpoint),
            "Formula": "compatibility=1-(average_rank-0.5)/N, independently per mode",
        },
    )
    return _write_manifest(
        cli,
        output_dir,
        (cache, summary),
        {"CacheRows": len(frame)},
        0,
        {
            "SelectionMetric": "not_applicable_train_only_cache",
            "Cache": str(cache),
            "CacheSHA256": sha256(cache),
            "EvaluatorCheckpoint": str(evaluator_checkpoint),
            "EvaluatorCheckpointSHA256": sha256(evaluator_checkpoint),
            "TrainOnly": True,
        },
    )


def _prediction_rows(model, loader, device, method, seed, limit=0):
    model.eval()
    rows = []
    with torch.no_grad():
        for _, batch in _iter_limited(loader, limit):
            text = batch["text"].to(device)
            audio = batch["audio"].to(device)
            vision = batch["vision"].to(device)
            labels = batch["labels"]["M"].to(device).view(-1)
            values = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                values[mode] = (
                    model(text, audio, vision, mask)["output_logit"].view(-1).cpu().numpy()
                )
            for offset, index in enumerate(batch["index"].view(-1).numpy()):
                rows.append(
                    {
                        "sample_index": int(index),
                        "sample_id": str(list(batch["id"])[offset]),
                        "label": float(labels[offset].cpu()),
                        **{
                            "{}_pred".format(mode): float(values[mode][offset])
                            for mode in values
                        },
                        "Split": "valid",
                        "Method": method,
                        "Seed": int(seed),
                    }
                )
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort")


def train_cfcompat(cli, args, output_dir):
    setup_seed(cli.seed)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("CFCompat training exposed a non-train/valid split.")
    clean_manifest = json.loads(
        (stage_directory(cli.result_root, "clean", cli.seed) / "stage_manifest.json").read_text()
    )
    moddrop_manifest = json.loads(
        (stage_directory(cli.result_root, "moddrop", cli.seed) / "stage_manifest.json").read_text()
    )
    cache_manifest = json.loads(
        (stage_directory(cli.result_root, "compatibility", cli.seed) / "stage_manifest.json").read_text()
    )
    clean = Path(clean_manifest["Checkpoint"])
    evaluator = Path(moddrop_manifest["Checkpoint"])
    cache_path = Path(cache_manifest["Cache"])
    for path, expected in (
        (clean, clean_manifest["CheckpointSHA256"]),
        (evaluator, moddrop_manifest["CheckpointSHA256"]),
        (cache_path, cache_manifest["CacheSHA256"]),
    ):
        if sha256(path) != expected:
            raise RuntimeError("Seed-specific CFCompat input SHA mismatch.")
    cache = pd.read_csv(cache_path)
    expected_cache_rows = len(cache) if cli.smoke_batches else int(args.train_samples)
    if (
        len(cache) != expected_cache_rows
        or not len(cache)
        or cache.sample_index.duplicated().any()
    ):
        raise RuntimeError("CFCompat cache binding is incomplete.")
    cache_by_index = cache.set_index("sample_index").to_dict("index")
    teacher = build_frozen_teacher(DLF, args, clean)
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(clean, map_location=args.device), strict=True)
    student = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    student.eval()
    first = next(iter(loaders["valid"]))
    assert_initial_lav_equivalence(
        teacher,
        student,
        first["text"].to(args.device),
        first["audio"].to(args.device),
        first["vision"].to(args.device),
    )
    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion, cosine, hinge = nn.L1Loss(), nn.CosineEmbeddingLoss(), HingeLoss()
    generator = torch.Generator().manual_seed(int(cli.seed) + 104729)
    checkpoint = output_dir / "DLF_mosei_seed{}_best_valid.pth".format(cli.seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best, best_epoch, best_metrics = float("inf"), 0, None
    epoch_rows = []
    train_loader = (
        build_single_split_loader(args, "train", cli.num_workers)
        if cli.smoke_batches
        else loaders["train"]
    )
    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        student.train()
        optimizer.zero_grad()
        total, kd_total, batches = 0.0, 0.0, 0
        for step, batch in _iter_limited(train_loader, cli.smoke_batches):
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels = batch["labels"]["M"].to(args.device).view(-1, 1)
            full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
            full_loss, _ = compute_full_dlf_loss(
                student(text, audio, vision, full_mask),
                labels,
                criterion,
                cosine,
                hinge,
            )
            missing_mask = sample_missing_masks(
                labels.size(0), generator, args.device, audio.dtype
            )
            modes = modes_from_masks(missing_mask)
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision)
            indices = batch["index"].view(-1).numpy().astype(int).tolist()
            compatibility = compatibility_for_modes(
                cache_by_index, indices, modes, args.device, labels.dtype
            )
            kd_loss, _ = gated_kd_loss(
                missing_output["output_logit"], teacher_prediction, compatibility
            )
            loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite CFCompat loss.")
            loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen teacher received gradients.")
            if step % args.update_epochs == 0 or (
                cli.smoke_batches and step == cli.smoke_batches
            ):
                nn.utils.clip_grad_value_(student.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
            total += float(loss.detach())
            kd_total += float(kd_loss.detach())
            batches += 1
        if batches % args.update_epochs and not cli.smoke_batches:
            nn.utils.clip_grad_value_(student.parameters(), args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
        valid = _missing_eval(
            student, loaders["valid"], args.device, cli.smoke_batches
        )
        if not _finite_metrics(valid):
            raise FloatingPointError("Non-finite CFCompat validation metrics.")
        score = float(validation_objective(valid))
        scheduler.step(score)
        epoch_rows.append(
            {
                "Epoch": epoch,
                "TrainLoss": total / batches,
                "KDLoss": kd_total / batches,
                "JValid": score,
                **flatten_mode_metrics(valid),
            }
        )
        print("cfcompat seed={} epoch={} J_valid={:.6f}".format(cli.seed, epoch, score), flush=True)
        if score <= best - 1e-6:
            best, best_epoch, best_metrics = score, epoch, valid
            torch.save(student.state_dict(), checkpoint)
        if cli.max_epochs and epoch >= cli.max_epochs:
            break
        if epoch - best_epoch >= args.early_stop:
            break
    student.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    predictions = _prediction_rows(
        student,
        loaders["valid"],
        args.device,
        "CFCompatKD",
        cli.seed,
        cli.smoke_batches,
    )
    metrics_path = output_dir / "epoch_metrics.csv"
    predictions_path = output_dir / "valid_predictions.csv"
    pd.DataFrame(epoch_rows).to_csv(metrics_path, index=False)
    predictions.to_csv(predictions_path, index=False)
    return _write_manifest(
        cli,
        output_dir,
        (checkpoint, metrics_path, predictions_path),
        best_metrics,
        best_epoch,
        {
            "JValid": best,
            "Checkpoint": str(checkpoint),
            "CheckpointSHA256": sha256(checkpoint),
            "TeacherCheckpoint": str(clean),
            "TeacherCheckpointSHA256": sha256(clean),
            "EvaluatorCheckpoint": str(evaluator),
            "EvaluatorCheckpointSHA256": sha256(evaluator),
            "CompatibilityCache": str(cache_path),
            "CompatibilityCacheSHA256": sha256(cache_path),
            "StudentInitCheckpointSHA256": sha256(clean),
            "MissingSequenceSeed": int(cli.seed) + 104729,
        },
    )


def main():
    cli = parse_args()
    output_dir = stage_directory(cli.result_root, cli.stage, cli.seed)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("Stage output already exists without validated resume.")
    output_dir.mkdir(parents=True, exist_ok=True)
    args = build_args(cli)
    if cli.stage == "clean":
        manifest = train_clean(cli, args, output_dir)
    elif cli.stage == "moddrop":
        manifest = train_moddrop(cli, args, output_dir)
    elif cli.stage == "compatibility":
        manifest = build_compatibility(cli, args, output_dir)
    else:
        manifest = train_cfcompat(cli, args, output_dir)
    print(json.dumps(manifest, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
