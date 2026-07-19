"""Fixed-missing-mode, label-only specialists for Stage 18B."""

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import MMDataLoader
from train_cf_compat_kd import batch_to_device, build_config, prediction_rows
from trains.singleTask.cfcompat_fair_trainer import (
    TEST_ISOLATION,
    _asset_manifest,
    canonical_sha,
    metrics_with_missing_macro,
    state_dict_sha,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    compute_task_loss,
    evaluate_all_modes,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


def specialist_gap(unified_error, specialist_error, teacher_error):
    g_opt = float(unified_error) - float(specialist_error)
    g_info = float(specialist_error) - float(teacher_error)
    g_total = float(unified_error) - float(teacher_error)
    if abs(g_total - (g_opt + g_info)) > 1e-8:
        raise AssertionError("Specialist gap identity failed.")
    return g_opt, g_info, g_total


def train_specialist(
    cli,
    mode,
    output_dir,
    checkpoint_dir,
    asset_result_root="/code/DLF/result",
    asset_model_root="/code/DLF/pt",
):
    if mode not in MISSING_MODES:
        raise ValueError("Specialist mode must be LA, LV, or L.")
    seed = int(cli.seed)
    setup_seed(seed)
    args = build_config(cli, seed)
    assets = _asset_manifest(seed, asset_result_root, asset_model_root)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Specialist may construct only train/valid loaders.")

    clean_checkpoint = Path(assets["teacher_checkpoint"])
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(
        torch.load(clean_checkpoint, map_location=args.device), strict=True
    )
    student = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    initial_student_sha = state_dict_sha(student.state_dict())
    criterion = nn.L1Loss()
    initial_metrics = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    teacher_lav = initial_metrics["LAV"]

    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    optimizer_config = {
        "name": "Adam",
        "learning_rate": float(args.learning_rate),
        "weight_decay": 0.0,
        "scheduler": "ReduceLROnPlateau",
        "scheduler_mode": "min",
        "scheduler_factor": 0.5,
        "scheduler_patience": int(args.patience),
        "update_epochs": int(args.update_epochs),
        "gradient_clipping": "none_historical_cfcompat_online",
        "early_stop": int(args.early_stop),
        "max_epochs": int(cli.max_epochs or 1000),
        "selection": "{}_MAE".format(mode),
        "supervision": "fixed_mode_label_only",
    }
    output_dir, checkpoint_dir = Path(output_dir), Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "best_valid.pth"
    if checkpoint.exists():
        raise FileExistsError("Refusing to overwrite specialist checkpoint.")

    best_score, best_epoch, best_metrics = float("inf"), 0, None
    batch_hasher = hashlib.sha256()
    epoch_rows = []
    start = time.time()
    last_epoch = 0
    for epoch in range(1, int(cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        losses = []
        sample_count = 0
        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            batch_hasher.update(
                np.asarray([epoch, step] + indices, dtype=np.int64).tobytes()
            )
            mask = mode_to_mask(mode, labels.size(0), args.device, audio.dtype)
            output = student(text, audio, vision, mask)
            loss, _ = compute_task_loss(output, labels, criterion)
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite specialist loss.")
            loss.backward()
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()
            losses.append(float(loss.detach()))
            sample_count += int(labels.size(0))
        if sample_count != 1284:
            raise RuntimeError("Specialist epoch must traverse 1284 samples once.")
        metrics = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        score = float(metrics[mode]["MAE"])
        if not math.isfinite(score):
            raise FloatingPointError("Non-finite specialist validation score.")
        scheduler.step(score)
        is_best = score <= best_score - 1e-6
        if is_best:
            best_score, best_epoch, best_metrics = score, epoch, metrics
            torch.save(student.state_dict(), checkpoint)
        epoch_rows.append(
            {
                "Seed": seed,
                "Mode": mode,
                "Epoch": epoch,
                "SelectionScore": score,
                "IsBestValid": is_best,
                "TrainSamples": sample_count,
                "TrainLoss": float(np.mean(losses)),
                **{
                    "valid_{}".format(key): value
                    for key, value in metrics_with_missing_macro(metrics).items()
                },
            }
        )
        print(
            "stage18b specialist={} seed={} epoch={} valid_MAE={:.9f}".format(
                mode, seed, epoch, score
            ),
            flush=True,
        )
        if epoch - best_epoch >= args.early_stop:
            break
    if best_metrics is None or not checkpoint.is_file():
        raise RuntimeError("No specialist validation-best checkpoint.")
    elapsed = time.time() - start
    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    final_metrics = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    if abs(float(final_metrics[mode]["MAE"]) - best_score) > 1e-12:
        raise RuntimeError("Reloaded specialist checkpoint changed selection score.")
    predictions = prediction_rows(student, loaders["valid"], args.device)
    predictions = predictions[
        ["sample_id", "sample_index", "label", "{}_pred".format(mode)]
    ]
    predictions["Seed"] = seed
    predictions["Mode"] = mode
    predictions["Split"] = "valid"
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(
        output_dir / "epoch_metrics.csv", index=False
    )
    row = {
        "Seed": seed,
        "Method": "specialist",
        "Mode": mode,
        "BestValidEpoch": int(best_epoch),
        "LastEpoch": int(last_epoch),
        "SelectionMetric": "{}_MAE".format(mode),
        "SelectionScore": float(best_score),
        "TrainingSeconds": float(elapsed),
        "Checkpoint": str(checkpoint),
        "CheckpointSHA256": checkpoint_sha256(checkpoint),
        "InitialStudentStateSHA256": initial_student_sha,
        "CleanInitializationCheckpoint": str(clean_checkpoint),
        "CleanInitializationSHA256": assets["teacher_sha"],
        "TeacherUsedForTraining": False,
        "KDUsed": False,
        "FixedMode": mode,
        "TrainSamplesPerEpoch": 1284,
        "BatchOrderSHA256": batch_hasher.hexdigest(),
        "OptimizerConfigSHA256": canonical_sha(optimizer_config),
        "TrainerSHA256": checkpoint_sha256(Path(__file__)),
        "PhysicalGPU": int(getattr(cli, "physical_gpu", 2)),
        "InternalGPU": 0,
        **{
            "valid_{}".format(key): value
            for key, value in metrics_with_missing_macro(final_metrics).items()
        },
        **{
            "teacher_LAV_{}".format(key): float(value)
            for key, value in teacher_lav.items()
        },
        **TEST_ISOLATION,
    }
    pd.DataFrame([row]).to_csv(output_dir / "run_metrics.csv", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "assets": assets,
                "optimizer_config": optimizer_config,
                "row": row,
                "test_isolation": TEST_ISOLATION,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return row
