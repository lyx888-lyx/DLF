"""Frozen no-Test trainer used by the Stage 18 CFCompatKD evidence study.

The implementation is intentionally derived from
``run_cfcompat_stability_multiseed.py::train_one_seed``. It constructs only
the train and official validation loaders and preserves the historical Online
optimization, scheduler, checkpoint-selection, early-stop, missing-mask, and
batch-order semantics. No function in this module can construct a Test loader.
"""

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import MMDataLoader
from train_cf_compat_kd import (
    batch_to_device,
    build_config,
    initialize_teacher_student,
    prediction_rows,
)
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_COLUMNS,
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    cache_paths,
    compatibility_for_modes,
    gated_kd_loss,
    load_counterfactual_cache,
    modes_from_masks,
)
from trains.singleTask.cfcompat_stability_utils import MissingSequenceHasher
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    flatten_mode_metrics,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


STAGE18_VERSION = "cfcompat_evidence_v1"
ALLOWED_STAGE18A_METHODS = ("moddrop", "cfcompat")
TEST_ISOLATION = {
    "test_loader_constructed": False,
    "test_features_read": False,
    "test_labels_read": False,
    "test_predictions_read": False,
    "test_evaluation_performed": False,
    "locked_test_access_count": 0,
}


def canonical_sha(payload):
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def state_dict_sha(state):
    """Hash tensor content without relying on serialization metadata."""
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def metrics_with_missing_macro(metrics):
    row = flatten_mode_metrics(metrics)
    for metric in tuple(metrics["LAV"]):
        row["MissingMacro_{}".format(metric)] = float(
            np.mean([metrics[mode][metric] for mode in MISSING_MODES])
        )
    return row


def _finite(metrics):
    return all(
        math.isfinite(float(value))
        for mode_metrics in metrics.values()
        for value in mode_metrics.values()
    )


def _asset_manifest(seed, asset_result_root, asset_model_root):
    multiseed = int(seed) != 1111
    if multiseed:
        evaluator = (
            Path(asset_model_root)
            / "missing_baseline/moddrop_benchmark_multiseed_v1"
            / "seed{}".format(seed)
            / "DLF_mosi_seed{}_best_valid.pth".format(seed)
        )
        evaluator_source = (
            Path(asset_result_root)
            / "missing_baseline/moddrop_benchmark_multiseed_v1"
            / "seed{}".format(seed)
            / "mosi_per_seed.csv"
        )
    else:
        evaluator = (
            Path(asset_model_root)
            / "missing_baseline/moddrop"
            / "DLF_mosi_seed1111_best.pth"
        )
        evaluator_source = (
            Path(asset_result_root)
            / "missing_baseline/moddrop/train/mosi_per_seed.csv"
        )
    if not evaluator.is_file() or not evaluator_source.is_file():
        raise FileNotFoundError("Frozen Stage 1 evaluator assets are incomplete.")
    source_frame = pd.read_csv(evaluator_source)
    source_row = source_frame.loc[source_frame.Seed.astype(int) == int(seed)]
    if len(source_row) != 1:
        raise RuntimeError("Stage 1 evaluator source does not uniquely bind seed.")
    epoch_column = (
        "BestValidEpoch"
        if "BestValidEpoch" in source_row.columns
        else "BestEpoch"
    )
    evaluator_epoch = int(source_row.iloc[0][epoch_column])
    run_manifest = (
        Path(asset_result_root)
        / "missing_baseline/cf_compat_kd_v1/benchmark_multiseed/RUN_MANIFEST.json"
    )
    records = json.loads(run_manifest.read_text())["Seeds"]
    record = next(row for row in records if int(row["Seed"]) == int(seed))
    evaluator_sha = checkpoint_sha256(evaluator)
    if evaluator_sha != record["EvaluatorSHA256"]:
        raise RuntimeError("Evaluator SHA differs from the frozen Stage 3 binding.")
    teacher = Path(asset_model_root) / "DLF_mosi_seed{}_best.pth".format(seed)
    teacher_sha = checkpoint_sha256(teacher)
    if teacher_sha != record["Gate3SHA256"]:
        raise RuntimeError("Teacher SHA differs from the frozen Stage 3 binding.")
    return {
        "teacher_checkpoint": str(teacher),
        "teacher_sha": teacher_sha,
        "evaluator_checkpoint": str(evaluator),
        "evaluator_sha": evaluator_sha,
        "evaluator_best_epoch": int(evaluator_epoch),
        "evaluator_source": str(evaluator_source),
        "stage3_manifest": str(run_manifest),
    }


def _load_frozen_cache(
    seed, asset_result_root, evaluator_sha, cache_version
):
    """Load the immutable cache, including the audited legacy seed1111 schema."""
    if int(seed) != 1111:
        return load_counterfactual_cache(
            asset_result_root,
            "mosi",
            version=cache_version,
            seed=int(seed),
            expected_evaluator_sha=evaluator_sha,
        )
    paths = cache_paths(
        asset_result_root, "mosi", version=CACHE_VERSION, seed=None
    )
    frame = pd.read_csv(paths["csv"])
    config = json.loads(paths["config"].read_text())
    if (
        config.get("version") != CACHE_VERSION
        or config.get("seed") is not None
        or config.get("source") != "train_only"
        or config.get("evaluator_sha256") != evaluator_sha
        or list(frame.columns) != list(CACHE_COLUMNS)
        or frame.sample_index.duplicated().any()
    ):
        raise RuntimeError("Legacy seed1111 train-only cache audit failed.")
    for mode in MISSING_MODES:
        values = frame["compat_{}".format(mode)].to_numpy(dtype=np.float64)
        if not np.all((values > 0) & (values < 1)):
            raise RuntimeError("Legacy seed1111 compatibility is outside (0,1).")
    by_index = {
        int(row.sample_index): row._asdict()
        for row in frame.itertuples(index=False)
    }
    return frame, by_index


def train_no_test(
    cli,
    method,
    run_label,
    output_dir,
    checkpoint_dir,
    asset_result_root="/code/DLF/result",
    asset_model_root="/code/DLF/pt",
):
    """Train one Stage18A method and return its validation-only result row."""
    if method not in ALLOWED_STAGE18A_METHODS:
        raise ValueError("Stage18A method is not registered: {}".format(method))
    seed = int(cli.seed)
    setup_seed(seed)
    args = build_config(cli, seed)
    assets = _asset_manifest(seed, asset_result_root, asset_model_root)
    multiseed = seed != 1111
    cache_version = MULTISEED_CACHE_VERSION if multiseed else CACHE_VERSION
    cache_frame, cache_by_index = _load_frozen_cache(
        seed,
        asset_result_root,
        assets["evaluator_sha"],
        cache_version,
    )
    if len(cache_frame) != 1284 or cache_frame.sample_index.nunique() != 1284:
        raise RuntimeError(
            "Frozen train-only compatibility cache is not 1284 unique rows."
        )

    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Fair trainer must construct exactly train and valid loaders.")
    teacher_cli = type("TeacherCLI", (), {"model_save_dir": asset_model_root})()
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, teacher_cli, seed, loaders
    )
    if teacher_sha != assets["teacher_sha"]:
        raise RuntimeError("Runtime Teacher binding changed.")
    initial_student_sha = state_dict_sha(student.state_dict())

    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion = nn.L1Loss()
    cosine = nn.CosineEmbeddingLoss()
    hinge = HingeLoss()
    missing_generator = torch.Generator().manual_seed(seed + 104729)
    missing_hasher = MissingSequenceHasher()
    batch_hasher = hashlib.sha256()
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
    }

    output_dir = Path(output_dir)
    checkpoint_dir = Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "best_valid.pth"
    if checkpoint.exists():
        raise FileExistsError(
            "Refusing to overwrite Stage18 checkpoint: {}".format(checkpoint)
        )

    best_j = float("inf")
    best_epoch = 0
    best_metrics = None
    epoch_rows = []
    batch_sizes = None
    start = time.time()
    last_epoch = 0
    for epoch in range(1, int(cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        kd_values = []
        epoch_batch_sizes = []
        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            epoch_batch_sizes.append(int(labels.size(0)))
            batch_hasher.update(
                np.asarray([epoch, step] + indices, dtype=np.int64).tobytes()
            )
            full_mask = mode_to_mask(
                "LAV", labels.size(0), args.device, audio.dtype
            )
            full_loss, _ = compute_full_dlf_loss(
                student(text, audio, vision, full_mask),
                labels,
                criterion,
                cosine,
                hinge,
            )
            missing_mask = sample_missing_masks(
                labels.size(0), missing_generator, args.device, audio.dtype
            )
            modes = modes_from_masks(missing_mask)
            missing_hasher.update(modes)
            counts.update(count_missing_modes(missing_mask))
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
            if method == "cfcompat":
                teacher_prediction = teacher_lav_prediction(
                    teacher, text, audio, vision
                )
                gate = compatibility_for_modes(
                    cache_by_index, indices, modes, args.device, labels.dtype
                )
                kd_loss, _ = gated_kd_loss(
                    missing_output["output_logit"], teacher_prediction, gate
                )
            else:
                kd_loss = missing_output["output_logit"].sum() * 0.0
            kd_values.append(float(kd_loss.detach()))
            total_loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("Non-finite Stage18A training loss.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                # Historical CFCompat Online semantics do not clip gradients.
                optimizer.step()
                optimizer.zero_grad()
        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError("Train batch-size sequence changed across epochs.")
        valid = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        if not _finite(valid):
            raise FloatingPointError("Non-finite official validation metric.")
        j_valid = float(validation_objective(valid))
        scheduler.step(j_valid)
        is_best = j_valid <= best_j - 1e-6
        if is_best:
            best_j, best_epoch, best_metrics = j_valid, epoch, valid
            torch.save(student.state_dict(), checkpoint)
        epoch_rows.append(
            {
                "Seed": seed,
                "Method": method,
                "RunLabel": run_label,
                "Epoch": epoch,
                "J_valid": j_valid,
                "IsBestValid": is_best,
                "LA_count": counts["LA"],
                "LV_count": counts["LV"],
                "L_count": counts["L"],
                "KD_loss": float(np.mean(kd_values)),
                **{
                    "valid_{}".format(k): v
                    for k, v in metrics_with_missing_macro(valid).items()
                },
            }
        )
        print(
            "stage18a method={} run={} seed={} epoch={} J_valid={:.9f}".format(
                method, run_label, seed, epoch, j_valid
            ),
            flush=True,
        )
        if epoch - best_epoch >= args.early_stop:
            break
    elapsed = time.time() - start
    if best_metrics is None or not checkpoint.is_file():
        raise RuntimeError("No official-validation checkpoint was saved.")
    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    final_valid = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    final_j = float(validation_objective(final_valid))
    if abs(final_j - best_j) > 1e-12:
        raise RuntimeError("Reloaded validation-best checkpoint changed J.")
    predictions = prediction_rows(student, loaders["valid"], args.device)
    predictions["Seed"] = seed
    predictions["Method"] = method
    predictions["RunLabel"] = run_label
    predictions["Split"] = "valid"
    predictions.to_csv(output_dir / "valid_predictions.csv", index=False)
    pd.DataFrame(epoch_rows).to_csv(
        output_dir / "epoch_metrics.csv", index=False
    )

    cache_path = (
        Path(asset_result_root)
        / "counterfactual_compatibility"
        / cache_version
        / "mosi"
    )
    if multiseed:
        cache_path /= "seed{}".format(seed)
    cache_path /= "train_counterfactual_compatibility.csv"
    trainer_path = Path(__file__)
    row = {
        "Seed": seed,
        "Method": method,
        "RunLabel": run_label,
        "BestValidEpoch": int(best_epoch),
        "LastEpoch": int(last_epoch),
        "J_valid": final_j,
        "TrainingSeconds": float(elapsed),
        "Checkpoint": str(checkpoint),
        "CheckpointSHA256": checkpoint_sha256(checkpoint),
        "InitialStudentStateSHA256": initial_student_sha,
        "TeacherCheckpoint": str(teacher_checkpoint),
        "TeacherSHA256": teacher_sha,
        "EvaluatorCheckpoint": assets["evaluator_checkpoint"],
        "EvaluatorSHA256": assets["evaluator_sha"],
        "CompatibilityCacheSHA256": checkpoint_sha256(cache_path),
        "MissingScheduleSHA256": missing_hasher.hexdigest(),
        "MissingScheduleCount": int(missing_hasher.count),
        "BatchOrderSHA256": batch_hasher.hexdigest(),
        "OptimizerConfigSHA256": canonical_sha(optimizer_config),
        "TrainerSHA256": checkpoint_sha256(trainer_path),
        "NumWorkers": int(cli.num_workers),
        "PhysicalGPU": int(getattr(cli, "physical_gpu", 3)),
        "InternalGPU": 0,
        "SelectedBy": "official_valid_J",
        "GradientClipping": "none_historical_cfcompat_online",
        **{
            "valid_{}".format(k): v
            for k, v in metrics_with_missing_macro(final_valid).items()
        },
        **TEST_ISOLATION,
    }
    pd.DataFrame([row]).to_csv(output_dir / "run_metrics.csv", index=False)
    manifest = {
        "assets": assets,
        "optimizer_config": optimizer_config,
        "row": row,
        "test_isolation": TEST_ISOLATION,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return row
