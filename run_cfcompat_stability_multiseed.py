"""Run one frozen Stage 3 CFCompatKD trajectory with EMA and trajectory soups."""
import argparse
import json
import logging
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_loader import MMDataLoader
from train_cf_compat_kd import (
    _diagnostic_rows,
    _flatten,
    _grad_norm,
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
    gate_weights,
    gated_kd_loss,
    load_counterfactual_cache,
    locate_stage1_evaluator,
    modes_from_masks,
)
from trains.singleTask.cfcompat_stability_utils import (
    EMA_DECAY,
    MissingSequenceHasher,
    STAGE8_SEEDS,
    capture_rng_state,
    clone_state_cpu,
    consider_trajectory_checkpoint,
    expected_missing_sequence_sha,
    initialize_ema,
    optimizer_step_and_update_ema,
    preserve_rng_state,
    rank_trajectory,
    rng_states_equal,
    state_distance,
    uniform_soup,
    update_ema,
    verify_online_replay,
    load_stage3_reference,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    checkpoint_sha256,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    build_single_split_loader,
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


RESULT_VERSION = "cfcompat_stability_v1"
METHODS = ("Online", "EMA", "Soup-3", "Soup-5")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage 8 EMA/Trajectory-Soup CFCompatKD."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--seed", type=int, choices=STAGE8_SEEDS, required=True)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--max-epochs", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if args.num_workers != 0:
        parser.error("Stage 8 fixes num_workers=0 to replay the formal runs.")
    if args.smoke_test:
        if args.seed != 1111:
            parser.error("Stage 8 smoke is locked to seed1111.")
        args.max_epochs = 5
    elif args.max_epochs is not None:
        parser.error("Formal Stage 8 may not override the frozen maximum epoch.")
    return args


def stage8_paths(cli):
    result = (
        Path(cli.result_root) / "missing_baseline" / RESULT_VERSION
        / cli.dataset
    )
    model = (
        Path(cli.model_save_dir) / "missing_baseline" / RESULT_VERSION
        / cli.dataset
    )
    if cli.smoke_test:
        result, model = result / "smoke", model / "smoke"
    return result, result / "seed{}".format(cli.seed), model / "seed{}".format(cli.seed)


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "train"
    path = directory / "DLF-{}-cfcompat-stability-seed{}-{}-{}.log".format(
        cli.dataset, cli.seed, kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_stability")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def stage3_manifest_record(cli):
    path = (
        Path(cli.result_root) / "missing_baseline" / "cf_compat_kd_v1"
        / "benchmark_multiseed" / "RUN_MANIFEST.json"
    )
    payload = json.loads(path.read_text())
    records = [
        row for row in payload["Seeds"] if int(row["Seed"]) == int(cli.seed)
    ]
    if len(records) != 1:
        raise RuntimeError("Stage 3 manifest seed binding is invalid.")
    return records[0], path


def load_locked_cache(cli, evaluator_sha):
    multiseed = cli.seed != 1111
    version = MULTISEED_CACHE_VERSION if multiseed else CACHE_VERSION
    seed_arg = cli.seed if multiseed else None
    paths = cache_paths(
        cli.result_root, cli.dataset, version=version, seed=seed_arg
    )
    if multiseed:
        frame, by_index = load_counterfactual_cache(
            cli.result_root,
            cli.dataset,
            version=version,
            seed=seed_arg,
            expected_evaluator_sha=evaluator_sha,
        )
    else:
        # The locked seed1111 artifact predates the redundant
        # created_from_train_only manifest field.  Validate its full schema,
        # source, evaluator binding, and Stage3-recorded SHA without rewriting it.
        frame = pd.read_csv(paths["csv"])
        config = json.loads(paths["config"].read_text())
        if (
            config.get("version") != version
            or config.get("seed") is not None
            or config.get("source") != "train_only"
            or config.get("evaluator_sha256") != evaluator_sha
            or list(frame.columns) != list(CACHE_COLUMNS)
            or frame.sample_index.duplicated().any()
        ):
            raise RuntimeError("Locked seed1111 compatibility cache is invalid.")
        for mode in MISSING_MODES:
            values = frame["compat_{}".format(mode)].to_numpy()
            if not np.all((values > 0) & (values < 1)):
                raise RuntimeError("Locked compatibility lies outside (0,1).")
        by_index = {
            int(row.sample_index): row._asdict()
            for row in frame.itertuples(index=False)
        }
    manifest, manifest_path = stage3_manifest_record(cli)
    cache_sha = checkpoint_sha256(paths["csv"])
    if cache_sha != manifest["CacheSHA256"]:
        raise RuntimeError("Compatibility cache SHA differs from Stage3 manifest.")
    return frame, by_index, paths["csv"], cache_sha, manifest, manifest_path


def audit_model_state(student):
    batch_norm = [
        name
        for name, module in student.named_modules()
        if isinstance(module, nn.modules.batchnorm._BatchNorm)
    ]
    if batch_norm:
        raise RuntimeError(
            "BatchNorm requires an unregistered policy; stopping: {}".format(batch_norm)
        )
    state = student.state_dict()
    return {
        "BatchNormModules": batch_norm,
        "StateTensorCount": len(state),
        "FloatingStateTensorCount": sum(
            int(value.is_floating_point()) for value in state.values()
        ),
        "NonFloatingStateTensorCount": sum(
            int(not value.is_floating_point()) for value in state.values()
        ),
        "NamedBuffers": [
            {
                "Name": name,
                "DType": str(value.dtype),
                "Shape": list(value.shape),
                "Floating": bool(value.is_floating_point()),
            }
            for name, value in student.named_buffers()
        ],
        "Policy": (
            "No BatchNorm/running-stat recalibration. Floating state follows EMA "
            "or CPU-FP64 soup; non-floating state is copied/required identical."
        ),
    }


def metrics_with_missing_macro(metrics):
    result = flatten_mode_metrics(metrics)
    metric_names = tuple(metrics["LAV"])
    for metric in metric_names:
        result["MissingMacro_{}".format(metric)] = float(
            np.mean([metrics[mode][metric] for mode in MISSING_MODES])
        )
    return result


def method_row(
    cli,
    method,
    best_epoch,
    valid,
    test,
    checkpoint,
    teacher_checkpoint,
    teacher_sha,
    evaluator_checkpoint,
    evaluator_sha,
    cache_path,
    cache_sha,
    missing_sha,
    student_init_sha,
    extra=None,
):
    row = {
        "Seed": int(cli.seed),
        "Method": method,
        "BestValidEpoch": int(best_epoch),
        "J_valid": validation_objective(valid),
        "J_test_at_valid_best": validation_objective(test),
        "Checkpoint": str(checkpoint),
        "CheckpointSHA256": checkpoint_sha256(checkpoint),
        "StudentInitCheckpoint": str(teacher_checkpoint),
        "StudentInitSHA256": student_init_sha,
        "TeacherCheckpoint": str(teacher_checkpoint),
        "TeacherSHA256": teacher_sha,
        "EvaluatorCheckpoint": str(evaluator_checkpoint),
        "EvaluatorSHA256": evaluator_sha,
        "CompatibilityCache": str(cache_path),
        "CompatibilityCacheSHA256": cache_sha,
        "MissingSequenceSHA256": missing_sha,
        "SelectedBy": "validation_J",
        "StudentOnlyEval": True,
        "BestObservedTestEpoch": 0,
        "BestObservedTestJ": 0.0,
        "EMADecay": 0.0,
        "EMAUpdateCount": 0,
        "EMAOnlineRMSDistanceAtBest": 0.0,
        "SourceEpochs": "",
        "SourceCount": 0,
        "Averaging": "none",
        **{
            "valid_{}".format(key): value
            for key, value in metrics_with_missing_macro(valid).items()
        },
        **{
            "test_at_valid_best_{}".format(key): value
            for key, value in metrics_with_missing_macro(test).items()
        },
    }
    if extra:
        row.update(extra)
    return row


def write_predictions(model, loaders, device, seed_dir, method):
    slug = method.lower().replace("-", "")
    for split, loader in loaders.items():
        with preserve_rng_state():
            frame = prediction_rows(model, loader, device)
        frame["Seed"] = int(seed_dir.name.replace("seed", ""))
        frame["Method"] = method
        frame["Split"] = split
        frame["SelectedBy"] = "validation_J"
        frame.to_csv(
            seed_dir / "{}_{}_predictions.csv".format(slug, split), index=False
        )


def train_one_seed(cli, logger):
    setup_seed(cli.seed)
    args = build_config(cli, cli.seed)
    multiseed = cli.seed != 1111
    evaluator_checkpoint, evaluator_best_epoch, evaluator_source = (
        locate_stage1_evaluator(
            cli.result_root,
            cli.dataset,
            cli.seed,
            multiseed=multiseed,
            smoke=False,
        )
    )
    evaluator_sha = checkpoint_sha256(evaluator_checkpoint)
    cache_frame, cache_by_index, cache_path, cache_sha, stage3_manifest, manifest_path = (
        load_locked_cache(cli, evaluator_sha)
    )
    if len(cache_frame) != 1284:
        raise RuntimeError("Stage 8 requires the locked 1284-sample cache.")

    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Training must construct train/valid loaders.")
    test_loader = build_single_split_loader(
        args, "test", cli.num_workers
    )
    teacher, student, teacher_checkpoint, teacher_sha = initialize_teacher_student(
        args, cli, cli.seed, loaders
    )
    if teacher_sha != stage3_manifest["Gate3SHA256"]:
        raise RuntimeError("Teacher SHA differs from Stage3 manifest.")
    state_audit = audit_model_state(student)

    optimizer = optim.Adam(student.parameters(), lr=args.learning_rate)
    assert_teacher_not_in_optimizer(teacher, optimizer)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=.5, patience=args.patience
    )
    criterion, cosine, hinge = (
        nn.L1Loss(),
        nn.CosineEmbeddingLoss(),
        HingeLoss(),
    )
    missing_generator = torch.Generator().manual_seed(cli.seed + 104729)
    actual_missing = MissingSequenceHasher()
    root_dir, seed_dir, model_dir = stage8_paths(cli)
    seed_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)
    online_checkpoint = model_dir / "online_best_valid.pth"
    ema_checkpoint = model_dir / "ema_best_valid.pth"

    rng_before_ema = capture_rng_state()
    ema = initialize_ema(student)
    rng_after_ema = capture_rng_state()
    if not rng_states_equal(rng_before_ema, rng_after_ema):
        raise RuntimeError("EMA initialization affected online RNG.")
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if any(id(parameter) in optimizer_ids for parameter in ema.parameters()):
        raise RuntimeError("EMA parameters entered the online optimizer.")

    best_online_j, best_online_epoch = float("inf"), 0
    best_ema_j, best_ema_epoch = float("inf"), 0
    best_test_j, best_test_epoch = float("inf"), 0
    ema_updates = 0
    epoch_rows, trajectory = [], []
    batch_sizes = None
    last_epoch = 0
    logger.info(
        "seed=%s frozen Stage3 trajectory; EMA=%.3f; Soup=Top3/Top5; "
        "all main checkpoints selected by validation J",
        cli.seed,
        EMA_DECAY,
    )
    logger.info(
        "teacher=%s sha=%s evaluator=%s sha=%s cache=%s sha=%s manifest=%s",
        teacher_checkpoint,
        teacher_sha,
        evaluator_checkpoint,
        evaluator_sha,
        cache_path,
        cache_sha,
        manifest_path,
    )

    for epoch in range(1, (cli.max_epochs or 1000) + 1):
        last_epoch = epoch
        student.train()
        optimizer.zero_grad()
        counts = Counter({"LA": 0, "LV": 0, "L": 0})
        gate_records, batch_kd_losses, epoch_batch_sizes = [], [], []
        for step, batch in enumerate(loaders["train"], 1):
            text, audio, vision, labels = batch_to_device(batch, args.device)
            epoch_batch_sizes.append(int(labels.size(0)))
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
            actual_missing.update(modes)
            counts.update(count_missing_modes(missing_mask))
            missing_output = student(text, audio, vision, missing_mask)
            missing_loss, _ = compute_task_loss(
                missing_output, labels, criterion
            )
            teacher_prediction = teacher_lav_prediction(
                teacher, text, audio, vision
            )
            indices = (
                batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            )
            compatibility = compatibility_for_modes(
                cache_by_index,
                indices,
                modes,
                args.device,
                labels.dtype,
            )
            gate, reliability = gate_weights(
                compatibility, teacher_prediction, labels, "compat"
            )
            kd_loss, each_kd = gated_kd_loss(
                missing_output["output_logit"], teacher_prediction, gate
            )
            batch_kd_losses.append(float(kd_loss.detach()))
            if step == 1:
                gradients = torch.autograd.grad(
                    kd_loss,
                    [
                        parameter
                        for parameter in student.parameters()
                        if parameter.requires_grad
                    ],
                    retain_graph=True,
                    allow_unused=True,
                )
                if _grad_norm(gradients) <= 0:
                    raise RuntimeError("Gated KD did not reach online Student.")
            total_loss = full_loss + missing_loss + kd_loss
            if not torch.isfinite(total_loss):
                raise FloatingPointError("NaN/Inf in frozen CFCompatKD loss.")
            total_loss.backward()
            if teacher_grad_count(teacher):
                raise RuntimeError("Frozen Teacher received gradients.")
            if step % args.update_epochs == 0 or step == len(loaders["train"]):
                optimizer_step_and_update_ema(optimizer, ema, student)
                ema_updates += 1
                optimizer.zero_grad()

            denominator = float(gate.sum().detach().cpu()) + 1e-8
            student_error = torch.abs(
                missing_output["output_logit"].detach().view(-1)
                - labels.view(-1)
            )
            teacher_error = torch.abs(
                teacher_prediction.detach().view(-1) - labels.view(-1)
            )
            teacher_student_gap = torch.abs(
                missing_output["output_logit"].detach().view(-1)
                - teacher_prediction.detach().view(-1)
            )
            for offset, index in enumerate(indices):
                mode = modes[offset]
                gate_records.append(
                    {
                        "sample_index": index,
                        "mode": mode,
                        "delta": cache_by_index[index][
                            "delta_{}".format(mode)
                        ],
                        "compat": float(compatibility[offset]),
                        "reliability": float(reliability[offset]),
                        "gate": float(gate[offset]),
                        "kd": float(each_kd[offset].detach()),
                        "weighted_kd_contribution": float(
                            gate[offset].detach()
                            * each_kd[offset].detach()
                            / denominator
                        ),
                        "student_missing_abs_label_error": float(
                            student_error[offset]
                        ),
                        "teacher_error": float(teacher_error[offset]),
                        "teacher_student_abs_gap": float(
                            teacher_student_gap[offset]
                        ),
                    }
                )

        if batch_sizes is None:
            batch_sizes = epoch_batch_sizes
        elif batch_sizes != epoch_batch_sizes:
            raise RuntimeError("Train batch-size sequence changed across epochs.")
        if (
            epoch == 1
            and cli.seed == 1111
            and counts != Counter({"LA": 435, "LV": 430, "L": 419})
        ):
            raise RuntimeError(
                "Epoch1 missing counts must be LA=435 LV=430 L=419."
            )

        # Keep the original online evaluation/scheduler/checkpoint order intact.
        valid = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        test = evaluate_all_modes(
            student, test_loader, args.device, "moddrop", criterion
        )
        j_valid, j_test = validation_objective(valid), validation_objective(test)
        if not (math.isfinite(j_valid) and math.isfinite(j_test)):
            raise FloatingPointError("Non-finite online metric.")
        scheduler.step(j_valid)
        is_best_online = j_valid <= best_online_j - 1e-6
        if is_best_online:
            best_online_j, best_online_epoch = j_valid, epoch
            torch.save(student.state_dict(), online_checkpoint)
        if j_test <= best_test_j - 1e-6:
            best_test_j, best_test_epoch = j_test, epoch
        trajectory = consider_trajectory_checkpoint(
            trajectory, student, cli.seed, epoch, j_valid
        )

        # EMA evaluation is auxiliary and restores every global RNG afterward.
        rng_before_eval = capture_rng_state()
        with preserve_rng_state():
            ema_valid = evaluate_all_modes(
                ema, loaders["valid"], args.device, "moddrop", criterion
            )
            ema_test = evaluate_all_modes(
                ema, test_loader, args.device, "moddrop", criterion
            )
        rng_after_eval = capture_rng_state()
        if not rng_states_equal(rng_before_eval, rng_after_eval):
            raise RuntimeError("EMA evaluation affected online RNG.")
        ema_j_valid = validation_objective(ema_valid)
        ema_j_test = validation_objective(ema_test)
        if not (math.isfinite(ema_j_valid) and math.isfinite(ema_j_test)):
            raise FloatingPointError("Non-finite EMA metric.")
        if ema_j_valid <= best_ema_j - 1e-6:
            best_ema_j, best_ema_epoch = ema_j_valid, epoch
            torch.save(ema.state_dict(), ema_checkpoint)

        gate_summary, _ = _diagnostic_rows(
            gate_records, cli.seed, epoch, "compat"
        )
        online_state = student.state_dict()
        ema_state = ema.state_dict()
        distance = state_distance(ema_state, online_state)
        epoch_rows.append(
            {
                "Seed": cli.seed,
                "Epoch": epoch,
                "Online_J_valid": j_valid,
                "Online_J_test": j_test,
                "EMA_J_valid": ema_j_valid,
                "EMA_J_test": ema_j_test,
                "EMAOnlineRMSDistance": distance,
                "EMAUpdateCount": ema_updates,
                "EMADecay": EMA_DECAY,
                "LA": counts["LA"],
                "LV": counts["LV"],
                "L": counts["L"],
                "KD_loss": float(np.mean(batch_kd_losses)),
                "GateMean": gate_summary["gate_mean"],
                "OnlineIsBestValid": is_best_online,
                "EMAIsBestValid": ema_j_valid <= best_ema_j + 1e-12,
            }
        )
        logger.info(
            "epoch=%s LA=%s LV=%s L=%s online_valid=%.6f online_test=%.6f "
            "ema_valid=%.6f ema_test=%.6f updates=%s distance=%.9g",
            epoch,
            counts["LA"],
            counts["LV"],
            counts["L"],
            j_valid,
            j_test,
            ema_j_valid,
            ema_j_test,
            ema_updates,
            distance,
        )
        if epoch - best_online_epoch >= args.early_stop:
            break

    if not online_checkpoint.is_file() or not ema_checkpoint.is_file():
        raise RuntimeError("Online and EMA validation-best checkpoints are required.")
    expected_sha, expected_count = expected_missing_sequence_sha(
        cli.seed, last_epoch, batch_sizes
    )
    actual_sha = actual_missing.hexdigest()
    if actual_sha != expected_sha or actual_missing.count != expected_count:
        raise RuntimeError("Missing-mode sequence differs from Stage3 algorithm.")

    student.load_state_dict(
        torch.load(online_checkpoint, map_location=args.device), strict=True
    )
    online_valid = evaluate_all_modes(
        student, loaders["valid"], args.device, "moddrop", criterion
    )
    online_test = evaluate_all_modes(
        student, test_loader, args.device, "moddrop", criterion
    )
    online_row = method_row(
        cli,
        "Online",
        best_online_epoch,
        online_valid,
        online_test,
        online_checkpoint,
        teacher_checkpoint,
        teacher_sha,
        evaluator_checkpoint,
        evaluator_sha,
        cache_path,
        cache_sha,
        actual_sha,
        teacher_sha,
        {
            "BestObservedTestEpoch": best_test_epoch,
            "BestObservedTestJ": best_test_j,
            "ReplayReference": str(load_stage3_reference(cli.result_root, cli.seed)[1]),
        },
    )
    reference, reference_path = load_stage3_reference(
        cli.result_root, cli.seed
    )
    replay = verify_online_replay(online_row, reference)
    replay.update(
        {
            "Seed": cli.seed,
            "ReferencePath": str(reference_path),
            "ActualMissingSequenceSHA256": actual_sha,
            "ExpectedMissingSequenceSHA256": expected_sha,
            "MissingSequenceMatch": actual_sha == expected_sha,
        }
    )
    (seed_dir / "online_replay_audit.json").write_text(
        json.dumps(replay, indent=2, sort_keys=True) + "\n"
    )
    if not cli.smoke_test and not replay["Passed"]:
        pd.DataFrame([online_row]).to_csv(
            seed_dir / "per_seed_all_methods.csv", index=False
        )
        raise RuntimeError("ONLINE REPLAY FAILURE seed{}".format(cli.seed))

    # Formal smoke only verifies mechanics; it intentionally cannot reach the
    # historical validation-best epoch and therefore does not apply replay.
    if cli.smoke_test:
        replay["FormalReplayApplied"] = False
        (seed_dir / "online_replay_audit.json").write_text(
            json.dumps(replay, indent=2, sort_keys=True) + "\n"
        )

    write_predictions(
        student,
        {"valid": loaders["valid"], "test": test_loader},
        args.device,
        seed_dir,
        "Online",
    )

    ema.load_state_dict(
        torch.load(ema_checkpoint, map_location=args.device), strict=True
    )
    with preserve_rng_state():
        ema_valid = evaluate_all_modes(
            ema, loaders["valid"], args.device, "moddrop", criterion
        )
        ema_test = evaluate_all_modes(
            ema, test_loader, args.device, "moddrop", criterion
        )
    best_ema_epoch_row = next(
        row for row in epoch_rows if int(row["Epoch"]) == int(best_ema_epoch)
    )
    ema_row = method_row(
        cli,
        "EMA",
        best_ema_epoch,
        ema_valid,
        ema_test,
        ema_checkpoint,
        teacher_checkpoint,
        teacher_sha,
        evaluator_checkpoint,
        evaluator_sha,
        cache_path,
        cache_sha,
        actual_sha,
        teacher_sha,
        {
            "EMADecay": EMA_DECAY,
            "EMAUpdateCount": ema_updates,
            "EMAOnlineRMSDistanceAtBest": best_ema_epoch_row[
                "EMAOnlineRMSDistance"
            ],
        },
    )
    write_predictions(
        ema,
        {"valid": loaders["valid"], "test": test_loader},
        args.device,
        seed_dir,
        "EMA",
    )

    trajectory = rank_trajectory(trajectory)
    source_dir = model_dir / "trajectory_sources"
    source_dir.mkdir(parents=True, exist_ok=True)
    for rank, entry in enumerate(trajectory, 1):
        path = source_dir / "rank{}_epoch{}.pth".format(rank, entry["Epoch"])
        torch.save(entry["state"], path)
        entry["Checkpoint"] = str(path)
        entry["CheckpointSHA256"] = checkpoint_sha256(path)
        entry["Rank"] = rank

    method_rows = [online_row, ema_row]
    soup_source_rows = []
    for top_k in (3, 5):
        soup_state, selected = uniform_soup(
            trajectory, top_k, expected_seed=cli.seed
        )
        selected_epochs = {int(entry["Epoch"]) for entry in selected}
        method = "Soup-{}".format(top_k)
        checkpoint = model_dir / "soup{}.pth".format(top_k)
        torch.save(soup_state, checkpoint)
        student.load_state_dict(soup_state, strict=True)
        with preserve_rng_state():
            valid = evaluate_all_modes(
                student, loaders["valid"], args.device, "moddrop", criterion
            )
            test = evaluate_all_modes(
                student, test_loader, args.device, "moddrop", criterion
            )
        row = method_row(
            cli,
            method,
            selected[0]["Epoch"],
            valid,
            test,
            checkpoint,
            teacher_checkpoint,
            teacher_sha,
            evaluator_checkpoint,
            evaluator_sha,
            cache_path,
            cache_sha,
            actual_sha,
            teacher_sha,
            {
                "SourceEpochs": ",".join(
                    str(entry["Epoch"]) for entry in selected
                ),
                "SourceCount": top_k,
                "Averaging": "uniform_CPU_FP64",
            },
        )
        method_rows.append(row)
        write_predictions(
            student,
            {"valid": loaders["valid"], "test": test_loader},
            args.device,
            seed_dir,
            method,
        )
        for entry in trajectory:
            soup_source_rows.append(
                {
                    "Seed": cli.seed,
                    "SoupMethod": method,
                    "SourceRank": entry["Rank"],
                    "SourceEpoch": entry["Epoch"],
                    "SourceJValid": entry["J_valid"],
                    "SourceCheckpoint": entry["Checkpoint"],
                    "SourceCheckpointSHA256": entry["CheckpointSHA256"],
                    "Included": int(entry["Epoch"]) in selected_epochs,
                    "SoupCheckpoint": str(checkpoint),
                    "SoupCheckpointSHA256": checkpoint_sha256(checkpoint),
                    "SoupSourceRMSDistance": state_distance(
                        soup_state, entry["state"]
                    ),
                    "KeysIdentical": True,
                    "ShapesIdentical": True,
                    "NonFloatingIdentical": True,
                    "Averaging": "uniform_CPU_FP64",
                }
            )

    pd.DataFrame(method_rows).to_csv(
        seed_dir / "per_seed_all_methods.csv", index=False
    )
    pd.DataFrame(epoch_rows).to_csv(
        seed_dir / "ema_epoch_metrics.csv", index=False
    )
    pd.DataFrame(soup_source_rows).to_csv(
        seed_dir / "soup_source_checkpoints.csv", index=False
    )
    manifest = {
        "Seed": cli.seed,
        "Methods": {
            row["Method"]: {
                "Checkpoint": row["Checkpoint"],
                "CheckpointSHA256": row["CheckpointSHA256"],
                "J_valid": row["J_valid"],
                "J_test_at_valid_best": row["J_test_at_valid_best"],
                "BestValidEpoch": row["BestValidEpoch"],
            }
            for row in method_rows
        },
        "Replay": replay,
        "EMA": {
            "Decay": EMA_DECAY,
            "UpdateCount": ema_updates,
            "BestValidEpoch": best_ema_epoch,
            "NeverInOptimizer": True,
            "NeverInBackward": all(
                parameter.grad is None for parameter in ema.parameters()
            ),
            "RNGPreserved": True,
        },
        "Soup": {
            "Top5Epochs": [entry["Epoch"] for entry in trajectory],
            "Top5JValid": [entry["J_valid"] for entry in trajectory],
            "OnlyOnlineSameSeedSources": True,
            "CPUFP64Averaging": True,
            "TestUsedForSourceSelection": False,
        },
        "StateAudit": state_audit,
        "TeacherSHA256": teacher_sha,
        "EvaluatorSHA256": evaluator_sha,
        "CompatibilityCacheSHA256": cache_sha,
        "MissingSequenceSHA256": actual_sha,
        "OnlineTrainingProtocol": "Stage3 CFCompatKD unchanged",
        "NoCrossSeedWeightAveraging": True,
    }
    (seed_dir / "seed_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    logger.info(
        "complete seed=%s replay=%s online=(%s,%.6f,%.6f) "
        "ema=(%s,%.6f,%.6f) soup3=(%.6f,%.6f) soup5=(%.6f,%.6f)",
        cli.seed,
        replay["Passed"] if not cli.smoke_test else "smoke-not-applied",
        best_online_epoch,
        online_row["J_valid"],
        online_row["J_test_at_valid_best"],
        best_ema_epoch,
        ema_row["J_valid"],
        ema_row["J_test_at_valid_best"],
        method_rows[2]["J_valid"],
        method_rows[2]["J_test_at_valid_best"],
        method_rows[3]["J_valid"],
        method_rows[3]["J_test_at_valid_best"],
    )
    return method_rows


def main():
    cli = parse_args()
    logger, log_path = create_logger(cli)
    train_one_seed(cli, logger)
    logger.info("log=%s", log_path)


if __name__ == "__main__":
    main()
