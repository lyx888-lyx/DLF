"""Fair no-Test trainer for the preregistered Stage 18 KD controls."""

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
    CACHE_VERSION,
    MULTISEED_CACHE_VERSION,
    modes_from_masks,
)
from trains.singleTask.cfcompat_fair_trainer import (
    TEST_ISOLATION,
    _asset_manifest,
    _load_frozen_cache,
    canonical_sha,
    metrics_with_missing_macro,
    state_dict_sha,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.kd_control_gates import (
    CONTROL_METHODS,
    build_epoch_bindings,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    build_single_split_loader,
    compute_full_dlf_loss,
    compute_task_loss,
    count_missing_modes,
    evaluate_all_modes,
    mode_to_mask,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


def _teacher_train_cache(teacher, loader, device):
    teacher_by_index, label_by_index, id_by_index = {}, {}, {}
    teacher.eval()
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            prediction = teacher_lav_prediction(
                teacher, text, audio, vision
            ).view(-1).cpu().numpy()
            indices = batch["index"].view(-1).cpu().numpy().astype(int)
            ids = list(batch["id"])
            for offset, index in enumerate(indices):
                teacher_by_index[int(index)] = float(prediction[offset])
                label_by_index[int(index)] = float(labels[offset].item())
                id_by_index[int(index)] = str(ids[offset])
    if len(teacher_by_index) != 1284:
        raise RuntimeError("Teacher train cache must bind 1284 samples.")
    return teacher_by_index, label_by_index, id_by_index


def _schedule_epoch(loader, generator):
    indices, modes, batch_records = [], [], []
    for batch in loader:
        local_indices = (
            batch["index"].view(-1).cpu().numpy().astype(int).tolist()
        )
        mask = sample_missing_masks(len(local_indices), generator)
        local_modes = modes_from_masks(mask)
        indices.extend(local_indices)
        modes.extend(local_modes)
        batch_records.append((local_indices, local_modes))
    if len(indices) != 1284 or len(set(indices)) != 1284:
        raise RuntimeError("Canonical epoch schedule is not 1284 unique samples.")
    return indices, modes, batch_records


def _finite(metrics):
    return all(
        math.isfinite(float(value))
        for row in metrics.values()
        for value in row.values()
    )


def _write_train_analysis(
    student,
    loader,
    device,
    seed,
    method,
    teacher_by_index,
    cache_by_index,
    output,
):
    rows = []
    student.eval()
    with torch.no_grad():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            ids = list(batch["id"])
            for mode in MISSING_MODES:
                mask = mode_to_mask(
                    mode, labels.size(0), device, audio.dtype
                )
                prediction = (
                    student(text, audio, vision, mask)["output_logit"]
                    .view(-1)
                    .cpu()
                    .numpy()
                )
                for offset, index in enumerate(indices):
                    evaluator = float(
                        cache_by_index[index][
                            "evaluator_{}_pred".format(mode)
                        ]
                    )
                    teacher = float(teacher_by_index[index])
                    label = float(labels[offset].item())
                    rows.append(
                        {
                            "Seed": seed,
                            "Method": method,
                            "Split": "train",
                            "sample_id": str(ids[offset]),
                            "sample_index": index,
                            "mode": mode,
                            "prediction": float(prediction[offset]),
                            "teacher_prediction": teacher,
                            "label": label,
                            "compatibility": float(
                                cache_by_index[index][
                                    "compat_{}".format(mode)
                                ]
                            ),
                            "evaluator_prediction": evaluator,
                            "oracle_gate": int(
                                (teacher - evaluator)
                                * (label - evaluator)
                                > 0
                            ),
                        }
                    )
    pd.DataFrame(rows).to_csv(output, index=False)


def train_control(
    cli,
    method,
    output_dir,
    checkpoint_dir,
    asset_result_root="/code/DLF/result",
    asset_model_root="/code/DLF/pt",
):
    if method not in CONTROL_METHODS:
        raise ValueError("Unknown Stage18 control.")
    seed = int(cli.seed)
    setup_seed(seed)
    args = build_config(cli, seed)
    assets = _asset_manifest(seed, asset_result_root, asset_model_root)
    cache_version = MULTISEED_CACHE_VERSION if seed != 1111 else CACHE_VERSION
    cache_frame, cache_by_index = _load_frozen_cache(
        seed, asset_result_root, assets["evaluator_sha"], cache_version
    )
    loaders = MMDataLoader(args, cli.num_workers)
    schedule_loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"} or set(schedule_loaders) != {
        "train",
        "valid",
    }:
        raise RuntimeError("Control trainer may construct only train/valid.")
    teacher_cli = type("TeacherCLI", (), {"model_save_dir": asset_model_root})()
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, teacher_cli, seed, loaders
    )
    initial_student_sha = state_dict_sha(student.state_dict())
    analysis_loader = build_single_split_loader(args, "train", cli.num_workers)
    teacher_by_index, label_by_index, _ = _teacher_train_cache(
        teacher, analysis_loader, args.device
    )

    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.patience
    )
    criterion, cosine, hinge = (
        nn.L1Loss(),
        nn.CosineEmbeddingLoss(),
        HingeLoss(),
    )
    actual_missing_generator = torch.Generator().manual_seed(seed + 104729)
    schedule_missing_generator = torch.Generator().manual_seed(seed + 104729)
    optimizer_config = {
        "name": "Adam",
        "learning_rate": float(args.learning_rate),
        "weight_decay": 0.0,
        "scheduler": "ReduceLROnPlateau",
        "factor": 0.5,
        "patience": int(args.patience),
        "early_stop": int(args.early_stop),
        "update_epochs": int(args.update_epochs),
        "gradient_clipping": "none_historical_cfcompat_online",
        "lambda_kd": 1.0,
        "method": method,
    }
    output_dir, checkpoint_dir = Path(output_dir), Path(checkpoint_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = checkpoint_dir / "best_valid.pth"
    if checkpoint.exists():
        raise FileExistsError("Refusing to overwrite Stage18 control checkpoint.")

    best_j, best_epoch, last_epoch = float("inf"), 0, 0
    best_metrics = None
    epoch_rows, mass_rows = [], []
    batch_hasher = hashlib.sha256()
    start = time.time()
    for epoch in range(1, int(cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        schedule_indices, schedule_modes, scheduled_batches = _schedule_epoch(
            schedule_loaders["train"], schedule_missing_generator
        )
        binding = build_epoch_bindings(
            method,
            seed,
            epoch,
            schedule_indices,
            schedule_modes,
            cache_by_index,
            teacher_by_index,
            label_by_index,
        )
        mass_rows.append(
            {
                "Seed": seed,
                "Method": method,
                "Epoch": epoch,
                "TotalKDMass": binding["mass"]["total"],
                "KDMass_LA": binding["mass"]["LA"],
                "KDMass_LV": binding["mass"]["LV"],
                "KDMass_L": binding["mass"]["L"],
                "ReferenceCFMass": binding["reference_mass"]["total"],
                "ReferenceCFMass_LA": binding["reference_mass"]["LA"],
                "ReferenceCFMass_LV": binding["reference_mass"]["LV"],
                "ReferenceCFMass_L": binding["reference_mass"]["L"],
                "GateBindingSHA256": binding["gate_sha256"],
                "TeacherBindingSHA256": binding["teacher_binding_sha256"],
            }
        )
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        kd_losses = []
        for step, (batch, expected) in enumerate(
            zip(loaders["train"], scheduled_batches), 1
        ):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            mask = sample_missing_masks(
                labels.size(0),
                actual_missing_generator,
                args.device,
                audio.dtype,
            )
            modes = modes_from_masks(mask)
            if indices != expected[0] or modes != expected[1]:
                raise RuntimeError("Actual and preregistered schedules diverged.")
            batch_hasher.update(
                np.asarray([epoch, step] + indices, dtype=np.int64).tobytes()
            )
            counts.update(count_missing_modes(mask))
            full = mode_to_mask(
                "LAV", labels.size(0), args.device, audio.dtype
            )
            full_loss, _ = compute_full_dlf_loss(
                student(text, audio, vision, full),
                labels,
                criterion,
                cosine,
                hinge,
            )
            missing_output = student(text, audio, vision, mask)
            missing_loss, _ = compute_task_loss(
                missing_output, labels, criterion
            )
            if method == "moddrop":
                kd_loss = missing_output["output_logit"].sum() * 0.0
            else:
                gate = torch.as_tensor(
                    [binding["gate"][index] for index in indices],
                    device=args.device,
                    dtype=labels.dtype,
                )
                target = torch.as_tensor(
                    [
                        teacher_by_index[
                            binding["teacher_source"][index]
                        ]
                        for index in indices
                    ],
                    device=args.device,
                    dtype=labels.dtype,
                )
                each = torch.nn.functional.smooth_l1_loss(
                    missing_output["output_logit"].view(-1),
                    target.detach().view(-1),
                    reduction="none",
                )
                kd_loss = torch.sum(gate.detach() * each) / (
                    torch.sum(gate.detach()) + 1e-8
                )
            kd_losses.append(float(kd_loss.detach()))
            total = full_loss + missing_loss + kd_loss
            if not torch.isfinite(total):
                raise FloatingPointError("Non-finite Stage18 control loss.")
            total.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer.step()
                optimizer.zero_grad()
        valid = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        if not _finite(valid):
            raise FloatingPointError("Non-finite control validation metrics.")
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
                "Epoch": epoch,
                "J_valid": j_valid,
                "IsBestValid": is_best,
                "LA_count": counts["LA"],
                "LV_count": counts["LV"],
                "L_count": counts["L"],
                "KD_loss": float(np.mean(kd_losses)),
                **{
                    "valid_{}".format(key): value
                    for key, value in metrics_with_missing_macro(valid).items()
                },
            }
        )
        print(
            "stage18-control method={} seed={} epoch={} J_valid={:.9f}".format(
                method, seed, epoch, j_valid
            ),
            flush=True,
        )
        if epoch - best_epoch >= args.early_stop:
            break
    if best_metrics is None:
        raise RuntimeError("No validation-best control checkpoint.")
    student.load_state_dict(
        torch.load(checkpoint, map_location=args.device), strict=True
    )
    final_valid = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    if abs(validation_objective(final_valid) - best_j) > 1e-12:
        raise RuntimeError("Reloaded control checkpoint changed J.")
    valid_predictions = prediction_rows(
        student, loaders["valid"], args.device
    )
    valid_predictions["Seed"] = seed
    valid_predictions["Method"] = method
    valid_predictions["Split"] = "valid"
    valid_predictions.to_csv(
        output_dir / "valid_predictions.csv", index=False
    )
    _write_train_analysis(
        student,
        analysis_loader,
        args.device,
        seed,
        method,
        teacher_by_index,
        cache_by_index,
        output_dir / "train_analysis_predictions.csv",
    )
    pd.DataFrame(epoch_rows).to_csv(
        output_dir / "epoch_metrics.csv", index=False
    )
    pd.DataFrame(mass_rows).to_csv(
        output_dir / "kd_mass_by_epoch.csv", index=False
    )
    row = {
        "Seed": seed,
        "Method": method,
        "BestValidEpoch": int(best_epoch),
        "LastEpoch": int(last_epoch),
        "J_valid": float(best_j),
        "TrainingSeconds": float(time.time() - start),
        "Checkpoint": str(checkpoint),
        "CheckpointSHA256": checkpoint_sha256(checkpoint),
        "InitialStudentStateSHA256": initial_student_sha,
        "TeacherCheckpoint": str(teacher_checkpoint),
        "TeacherSHA256": teacher_sha,
        "EvaluatorCheckpoint": assets["evaluator_checkpoint"],
        "EvaluatorSHA256": assets["evaluator_sha"],
        "BatchOrderSHA256": batch_hasher.hexdigest(),
        "OptimizerConfigSHA256": canonical_sha(optimizer_config),
        "TrainerSHA256": checkpoint_sha256(Path(__file__)),
        "TotalKDMass": float(sum(row["TotalKDMass"] for row in mass_rows)),
        "KDMass_LA": float(sum(row["KDMass_LA"] for row in mass_rows)),
        "KDMass_LV": float(sum(row["KDMass_LV"] for row in mass_rows)),
        "KDMass_L": float(sum(row["KDMass_L"] for row in mass_rows)),
        "PhysicalGPU": int(cli.physical_gpu),
        "InternalGPU": 0,
        **{
            "valid_{}".format(key): value
            for key, value in metrics_with_missing_macro(final_valid).items()
        },
        **TEST_ISOLATION,
    }
    pd.DataFrame([row]).to_csv(output_dir / "run_metrics.csv", index=False)
    (output_dir / "run_manifest.json").write_text(
        json.dumps(
            {
                "row": row,
                "assets": assets,
                "optimizer_config": optimizer_config,
                "test_isolation": TEST_ISOLATION,
                "teacher_train_cache_source": "train_only_frozen_teacher_inference",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return row
