"""CFCompatKD v8: frozen-S0 sample-conditioned residual heads, Seed1113 Valid-only.

v8 keeps the entire historical S0 path immutable and trains only three
mode-specific bounded residual MLPs over detached frozen S0 features.  The v4
DISTILL/PRESERVE/ABSTAIN objective is reused unchanged.  Official Test is never
constructed or accessed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
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
from train_cf_compat_kd import _flatten, build_config
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_frozen_backbone_adapter_isolation_valid_screen_v7 as v7
from trains.singleTask.HingeLoss import HingeLoss
from trains.singleTask.cf_compat_kd_utils import modes_from_masks
from trains.singleTask.cfcompat_adapter_isolation_utils import (
    build_epoch_event_trajectory,
    clip_failure_epoch_summary,
    epoch_transfer_summary,
    failure_onset_table,
    mechanism_transfer_summary,
)
from trains.singleTask.cfcompat_regret_preserve_utils import (
    DISTILL_MARGIN,
    LAMBDA_PRESERVE,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    PRESERVE_MARGIN,
    regret_projection_summary,
)
from trains.singleTask.cfcompat_sample_residual_utils import (
    BENEFICIAL_NTR_MAX_DEGRADATION_VS_V7,
    DEV_SEED,
    J_MAX_DEGRADATION_VS_V7,
    MAX_ABS_RESIDUAL,
    METHOD,
    NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V7,
    OUTPUT_TAG,
    OVERALL_NTR_MAX_DEGRADATION_VS_V4,
    RESIDUAL_HIDDEN_DIM,
    RUN,
    VERSION,
    FrozenS0SampleResidual,
    assert_s0_no_gradients,
    development_signal_gate,
    jsonable,
    module_state_sha256,
    residual_diagnostic_frame,
    residual_parameter_summary,
)
from trains.singleTask.cfcompat_stability_utils import (
    MissingSequenceHasher,
    expected_missing_sequence_sha,
    preserve_rng_state,
)
from trains.singleTask.fixed_kd_utils import (
    assert_teacher_not_in_optimizer,
    checkpoint_sha256,
    teacher_grad_count,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    count_missing_modes,
    evaluate_all_modes,
    sample_missing_masks,
    validation_objective,
)
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen-S0 sample-conditioned residual CFCompatKD v8"
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("v8 fixes num_workers=1.")
    args.seeds = [DEV_SEED]
    args.max_epochs = 2 if args.smoke_test else None
    return args


def result_paths(cli):
    output = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    if cli.smoke_test:
        output, model = output / "smoke", model / "smoke"
    if output.exists() or model.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "v8 output exists; inspect it or use --overwrite: {} / {}".format(
                    output, model
                )
            )
        if output.exists():
            shutil.rmtree(output)
        if model.exists():
            shutil.rmtree(model)
    output.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return output, model


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / "DLF-mosi-sample-residual-v8-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("sample_residual_v8")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def load_v7_reference(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_frozen_backbone_adapter_isolation_v7"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    grid_path = root / "frozen_backbone_v7_candidate_grid.csv"
    raw_path = root / "frozen_backbone_v7_candidate_raw_valid_events.csv"
    if not grid_path.is_file() or not raw_path.is_file():
        raise FileNotFoundError("Missing frozen v7 artifacts under {}".format(root))
    grid = pd.read_csv(grid_path)
    raw = pd.read_csv(raw_path)
    if len(grid) != 1 or int(grid.iloc[0].Seed) != DEV_SEED:
        raise RuntimeError("Frozen v7 grid is not unique Seed1113.")
    if set(raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("Frozen v7 raw events contain non-Valid split.")
    return grid.iloc[0].to_dict(), raw


def train_trajectory(cli, logger, output_root, model_root):
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    if int(args.update_epochs) != 10:
        raise RuntimeError("v8 fixes original update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v8 may construct only train/valid loaders.")

    v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    decision_records = []
    try:
        teacher, s0_student, evaluator_bundle, assets = v4.load_assets(
            cli, args, loaders, DEV_SEED
        )
        train_baseline = v4._ACTIVE_TRAIN_BASELINE_FRAME.copy()

        # Residual head construction consumes initialization RNG; restore it so
        # the historical loader/update trajectory remains observationally stable.
        with preserve_rng_state():
            student = FrozenS0SampleResidual(
                s0_student,
                hidden_dim=RESIDUAL_HIDDEN_DIM,
                max_abs_residual=MAX_ABS_RESIDUAL,
            ).to(args.device)

        isolation = residual_parameter_summary(student)
        s0_sha_before = module_state_sha256(student.s0)
        trainable_before = v7.capture_trainable_state(student)
        trainable_parameters = [
            parameter for parameter in student.parameters() if parameter.requires_grad
        ]
        optimizer = optim.Adam(trainable_parameters, lr=args.learning_rate)
        assert_teacher_not_in_optimizer(teacher, optimizer)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if any(id(parameter) in optimizer_ids for parameter in student.s0.parameters()):
            raise RuntimeError("Frozen S0 entered the v8 optimizer.")
        if any(id(parameter) in optimizer_ids for parameter in evaluator_bundle.parameters()):
            raise RuntimeError("Frozen baseline bundle entered the optimizer.")

        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=args.patience
        )
        criterion = nn.L1Loss()
        cosine = nn.CosineEmbeddingLoss()
        hinge = HingeLoss()
        missing_generator = torch.Generator().manual_seed(DEV_SEED + 104729)
        missing_hasher = MissingSequenceHasher()

        run_dir = output_root / "seed1113" / RUN
        checkpoint = model_root / "seed1113" / RUN / "best_valid.pth"
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)

        student.eval()
        s0_valid_metrics = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        s0_j = validation_objective(s0_valid_metrics)
        s0_prediction_frame = v7.snapshot_predictions(
            student, loaders["valid"], args.device
        )
        epoch_prediction_frames = [
            v7.prediction_frame_to_long(s0_prediction_frame, 0, False)
        ]

        best_j = float("inf")
        best_epoch = 0
        best_metrics = None
        epoch_rows = []
        batch_sizes = None
        last_epoch = 0
        all_projection_records = []

        logger.info(
            "seed=%s run=%s architecture=frozen_s0_sample_residual "
            "feature_dim=%s hidden_dim=%s max_residual=%s trainables=%s test=forbidden",
            DEV_SEED,
            RUN,
            isolation["feature_dim"],
            isolation["hidden_dim"],
            isolation["max_abs_residual"],
            isolation["trainable_parameter_count"],
        )

        for epoch in range(1, (cli.max_epochs or 1000) + 1):
            last_epoch = epoch
            student.train()
            optimizer.zero_grad()
            counts = Counter({"LA": 0, "LV": 0, "L": 0})
            epoch_batch_sizes = []
            objective_rows = []
            epoch_projection_records = []

            for step, batch in enumerate(loaders["train"], 1):
                labels_cpu = batch["labels"]["M"].view(-1, 1)
                batch_size = int(labels_cpu.size(0))
                epoch_batch_sizes.append(batch_size)
                missing_mask = sample_missing_masks(
                    batch_size,
                    missing_generator,
                    torch.device("cpu"),
                    torch.float32,
                )
                modes = tuple(modes_from_masks(missing_mask))
                missing_hasher.update(modes)
                counts.update(count_missing_modes(missing_mask))

                loss, diagnostics, projection_records = v7.forward_objective(
                    batch,
                    missing_mask,
                    modes,
                    args,
                    teacher,
                    evaluator_bundle,
                    student,
                    assets["cache_by_index"],
                    criterion,
                    cosine,
                    hinge,
                    decision_records,
                )
                loss.backward()
                if teacher_grad_count(teacher):
                    raise RuntimeError("Frozen Teacher received gradients.")
                assert_s0_no_gradients(student)
                if step % int(args.update_epochs) == 0 or step == len(loaders["train"]):
                    optimizer.step()
                    optimizer.zero_grad()
                objective_rows.append(diagnostics)
                epoch_projection_records.extend(projection_records)

            if batch_sizes is None:
                batch_sizes = epoch_batch_sizes
            elif batch_sizes != epoch_batch_sizes:
                raise RuntimeError("Batch-size sequence changed across epochs.")

            s0_sha_epoch = module_state_sha256(student.s0)
            if s0_sha_epoch != s0_sha_before:
                raise RuntimeError("Frozen S0 state changed during epoch {}.".format(epoch))

            valid = evaluate_all_modes(
                student, loaders["valid"], args.device, "moddrop", criterion
            )
            j_valid = validation_objective(valid)
            if not math.isfinite(j_valid):
                raise FloatingPointError("Non-finite Valid J.")
            scheduler.step(j_valid)
            is_best = j_valid <= best_j - 1e-6
            if is_best:
                best_j = j_valid
                best_epoch = epoch
                best_metrics = valid
                torch.save(student.state_dict(), checkpoint)

            epoch_prediction_frames.append(
                v7.prediction_frame_to_long(
                    v7.snapshot_predictions(student, loaders["valid"], args.device),
                    epoch,
                    is_best,
                )
            )

            local_projection = regret_projection_summary(epoch_projection_records)
            all_projection_records.extend(epoch_projection_records)
            row = {
                "Seed": DEV_SEED,
                "Run": RUN,
                "Epoch": int(epoch),
                "J_valid": float(j_valid),
                "IsBestValid": bool(is_best),
                "s0_state_sha256": s0_sha_epoch,
                "full_loss": float(np.mean([x["full_loss"] for x in objective_rows])),
                "missing_loss": float(np.mean([x["missing_loss"] for x in objective_rows])),
                "KD_loss": float(np.mean([x["kd_loss"] for x in objective_rows])),
                "mean_gate": float(np.mean([x["mean_gate"] for x in objective_rows])),
                "baseline_missing_MAE": float(
                    np.mean([x["baseline_missing_MAE"] for x in objective_rows])
                ),
                "safe_target_MAE": float(
                    np.mean([x["safe_target_MAE"] for x in objective_rows])
                ),
                **local_projection,
                **_flatten(valid, "valid"),
            }
            epoch_rows.append(row)
            logger.info(
                "epoch=%s J=%.6f MissingMacro=%.6f residual_params=%s s0_unchanged=True",
                epoch,
                j_valid,
                np.mean([valid[mode]["MAE"] for mode in MISSING_MODES]),
                isolation["trainable_parameter_count"],
            )
            if epoch - best_epoch >= args.early_stop:
                break

        if not checkpoint.is_file() or best_metrics is None:
            raise RuntimeError("Validation-best checkpoint is absent.")
        expected_sha, expected_count = expected_missing_sequence_sha(
            DEV_SEED, last_epoch, batch_sizes
        )
        if missing_hasher.hexdigest() != expected_sha or missing_hasher.count != expected_count:
            raise RuntimeError("Missing-mode sequence hash differs from frozen Stage 3 sequence.")

        if module_state_sha256(student.s0) != s0_sha_before:
            raise RuntimeError("S0 changed before checkpoint reload.")
        student.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
        s0_sha_after = module_state_sha256(student.s0)
        if s0_sha_after != s0_sha_before:
            raise RuntimeError("S0 changed after loading validation-best checkpoint.")

        final_valid = evaluate_all_modes(
            student, loaders["valid"], args.device, "moddrop", criterion
        )
        final_predictions = v7.snapshot_predictions(
            student, loaders["valid"], args.device
        )
        reference_predictions = evaluator_bundle.valid_reference.copy()
        raw_events = base.raw_events_for_run(
            DEV_SEED, RUN, final_predictions, reference_predictions
        )
        decisions = pd.DataFrame(decision_records)
        expected_decisions = 1284 * int(last_epoch)
        if len(decisions) != expected_decisions:
            raise RuntimeError(
                "v8 Train decision count mismatch: {} != {}".format(
                    len(decisions), expected_decisions
                )
            )
        decisions["Epoch"] = (
            (decisions.event_ordinal.astype(int) - 1) // 1284
        ) + 1

        epoch_predictions = pd.concat(epoch_prediction_frames, ignore_index=True)
        epoch_predictions["SelectedBestValid"] = (
            epoch_predictions.Epoch.astype(int).eq(best_epoch)
        )
        trajectory = build_epoch_event_trajectory(
            epoch_predictions, reference_predictions, best_epoch
        )
        transfer_by_epoch = epoch_transfer_summary(trajectory)
        failure_onset = failure_onset_table(trajectory)
        clip_epoch_summary = clip_failure_epoch_summary(trajectory)
        trainable_delta = v7.trainable_delta_summary(trainable_before, student)
        residual_valid = residual_diagnostic_frame(
            pd.DataFrame(raw_events), s0_prediction_frame
        )
        overall_projection = regret_projection_summary(all_projection_records)

        result = {
            "Seed": DEV_SEED,
            "Run": RUN,
            "Method": METHOD,
            "BestValidEpoch": int(best_epoch),
            "TrainEpochCount": int(last_epoch),
            "J_valid": float(validation_objective(final_valid)),
            "S0_J_valid": float(s0_j),
            "MainCheckpoint": str(checkpoint.resolve()),
            "MainCheckpointSHA256": checkpoint_sha256(checkpoint),
            "MissingSequenceSHA256": missing_hasher.hexdigest(),
            "MissingSequenceCount": int(missing_hasher.count),
            "TeacherCheckpoint": str(Path(assets["teacher_checkpoint"]).resolve()),
            "TeacherSHA256": assets["teacher_sha"],
            "EvaluatorCheckpoint": str(Path(assets["evaluator_checkpoint"]).resolve()),
            "EvaluatorSHA256": assets["evaluator_sha"],
            "S0StateSHA256Before": s0_sha_before,
            "S0StateSHA256After": s0_sha_after,
            "S0StateUnchanged": True,
            "ResidualFeatureDim": isolation["feature_dim"],
            "ResidualHiddenDim": isolation["hidden_dim"],
            "MaxAbsResidual": isolation["max_abs_residual"],
            "TrainableParameterCount": isolation["trainable_parameter_count"],
            "TotalParameterCount": isolation["total_parameter_count"],
            "TrainableParameterFraction": isolation["trainable_parameter_fraction"],
            "TestConstructed": False,
            **{"projection_{}".format(k): v for k, v in overall_projection.items()},
            **_flatten(final_valid, "valid"),
        }
        return {
            "result": result,
            "epoch_rows": pd.DataFrame(epoch_rows),
            "raw_events": pd.DataFrame(raw_events),
            "train_baseline": train_baseline,
            "train_decisions": decisions,
            "s0_valid_predictions": s0_prediction_frame,
            "valid_reference": reference_predictions,
            "epoch_predictions": epoch_predictions,
            "trajectory": trajectory,
            "transfer_by_epoch": transfer_by_epoch,
            "failure_onset": failure_onset,
            "clip_epoch_summary": clip_epoch_summary,
            "trainable_delta": trainable_delta,
            "residual_valid": residual_valid,
            "isolation": isolation,
            "s0_sha_before": s0_sha_before,
            "s0_sha_after": s0_sha_after,
        }
    finally:
        v4._ACTIVE_TRAIN_BASELINE_FRAME = None


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    v7_grid, v7_raw = load_v7_reference(cli)
    v4_grid, v4_raw = v7.load_v4_reference(cli)

    bundle = train_trajectory(cli, logger, output_root, model_root)
    result = bundle["result"]
    raw_events = bundle["raw_events"]
    events = base.derive_valid_events(raw_events)
    v7_events = base.derive_valid_events(v7_raw)
    v4_events = base.derive_valid_events(v4_raw)

    candidate_transfer = mechanism_transfer_summary(events, RUN)
    v7_run = str(v7_raw.Run.iloc[0])
    v4_run = str(v4_raw.Run.iloc[0])
    v7_transfer = mechanism_transfer_summary(v7_events, v7_run)
    v4_transfer = mechanism_transfer_summary(v4_events, v4_run)

    gate = None if cli.smoke_test else development_signal_gate(
        float(result["J_valid"]),
        float(v7_grid["J_valid"]),
        candidate_transfer,
        v7_transfer,
        v4_transfer,
    )
    verdict = (
        "SMOKE_ONLY_NO_MECHANISM_DECISION"
        if gate is None
        else (
            "MECHANISM_SIGNAL_POSITIVE_SAMPLE_CONDITIONED_RESIDUAL"
            if gate["passed"]
            else "MECHANISM_SIGNAL_NEGATIVE_OR_MIXED_SAMPLE_CONDITIONED_RESIDUAL"
        )
    )

    transfer_rows = []
    for source, summary in (
        ("v8_candidate", candidate_transfer),
        ("v7_frozen", v7_transfer),
        ("v4_frozen", v4_transfer),
    ):
        for group in ("all_missing", "teacher_beneficial", "teacher_nonbeneficial"):
            transfer_rows.append({"source": source, "group": group, **summary[group]})

    artifacts = {
        "sample_residual_v8_candidate_grid.csv": pd.DataFrame([result]),
        "sample_residual_v8_epoch_metrics.csv": bundle["epoch_rows"],
        "sample_residual_v8_candidate_raw_valid_events.csv": raw_events,
        "sample_residual_v8_candidate_valid_events.csv": events,
        "sample_residual_v8_transfer_summary.csv": pd.DataFrame(transfer_rows),
        "sample_residual_v8_train_baseline_cache.csv": bundle["train_baseline"],
        "sample_residual_v8_train_decisions.csv": bundle["train_decisions"],
        "sample_residual_v8_s0_valid_predictions.csv": bundle["s0_valid_predictions"],
        "sample_residual_v8_valid_reference_cache.csv": bundle["valid_reference"],
        "sample_residual_v8_valid_epoch_predictions.csv": bundle["epoch_predictions"],
        "sample_residual_v8_valid_epoch_event_trajectory.csv": bundle["trajectory"],
        "sample_residual_v8_valid_epoch_transfer_summary.csv": bundle["transfer_by_epoch"],
        "sample_residual_v8_valid_failure_onset.csv": bundle["failure_onset"],
        "sample_residual_v8_valid_clip_failure_epoch_summary.csv": bundle["clip_epoch_summary"],
        "sample_residual_v8_residual_valid_events.csv": bundle["residual_valid"],
        "sample_residual_v8_trainable_parameter_delta.csv": bundle["trainable_delta"],
        "sample_residual_v8_v7_reference_raw_valid_events.csv": v7_raw,
        "sample_residual_v8_v4_reference_raw_valid_events.csv": v4_raw,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_root / name, index=False)

    best_epoch_rows = bundle["transfer_by_epoch"].loc[
        bundle["transfer_by_epoch"].Epoch.astype(int).eq(int(result["BestValidEpoch"]))
    ].to_dict("records")
    residual = bundle["residual_valid"]
    residual_summary = {
        "mean_abs_residual": float(residual.abs_residual_delta.mean()),
        "p95_abs_residual": float(residual.abs_residual_delta.quantile(0.95)),
        "max_abs_residual": float(residual.abs_residual_delta.max()),
        "saturation_fraction_abs_ge_0p95": float(
            (residual.abs_residual_delta >= 0.95 * MAX_ABS_RESIDUAL).mean()
        ),
    }
    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "mechanism_signal_gate": jsonable(gate) if gate is not None else None,
        "candidate_transfer": jsonable(candidate_transfer),
        "frozen_v7_transfer": jsonable(v7_transfer),
        "frozen_v4_transfer": jsonable(v4_transfer),
        "selected_best_epoch_transfer_and_drift": jsonable(best_epoch_rows),
        "residual_summary": jsonable(residual_summary),
        "parameter_isolation": {
            **jsonable(bundle["isolation"]),
            "s0_state_sha256_before": bundle["s0_sha_before"],
            "s0_state_sha256_after": bundle["s0_sha_after"],
            "s0_state_unchanged": bool(
                bundle["s0_sha_before"] == bundle["s0_sha_after"]
            ),
            "s0_forced_eval_during_train": True,
            "residual_input_uses_labels": False,
            "residual_input_uses_fitted_split_statistics": False,
            "residual_normalization": "parameter_free_per_sample_layer_norm",
        },
        "protocol": {
            "development_seed": DEV_SEED,
            "new_trajectories_trained": 1,
            "base_objective": "v4_distill_preserve_abstain_unchanged",
            "entire_s0_path_trainable": False,
            "trainable_scope": "three_mode_specific_sample_conditioned_residual_heads_only",
            "residual_hidden_dim": RESIDUAL_HIDDEN_DIM,
            "max_abs_residual": MAX_ABS_RESIDUAL,
            "per_epoch_valid_sample_predictions_saved": True,
            "checkpoint_selection": "minimum_official_valid_J",
            "official_test_constructed": False,
            "official_test_accessed": False,
            "distill_margin": DISTILL_MARGIN,
            "preserve_margin": PRESERVE_MARGIN,
            "lambda_preserve": LAMBDA_PRESERVE,
            "mild_cfcompat": "{:.2f}+{:.2f}*compatibility".format(
                MILD_CFCOMPAT_BASE, MILD_CFCOMPAT_SCALE
            ),
        },
        "frozen_signal_thresholds": {
            "J_max_degradation_vs_v7": J_MAX_DEGRADATION_VS_V7,
            "beneficial_NTR_max_degradation_vs_v7": BENEFICIAL_NTR_MAX_DEGRADATION_VS_V7,
            "nonbeneficial_NTR_reduction_required_vs_v7": NONBENEFICIAL_NTR_REDUCTION_REQUIRED_VS_V7,
            "overall_NTR_max_degradation_vs_v4": OVERALL_NTR_MAX_DEGRADATION_VS_V4,
        },
        "cross_split_rationale": [
            "No Train-label-derived inference gate is used.",
            "The entire S0 function is immutable, preventing shared-backbone drift.",
            "Residual inputs are sample-conditioned frozen features rather than split-level calibration statistics.",
            "Parameter-free per-sample normalization avoids fitted Train mean/variance dependence.",
            "Bounded residuals limit extreme corrections under out-of-distribution feature values.",
            "These choices target robustness but do not guarantee Official Test improvement.",
        ],
    }
    summary_path = output_root / "sample_residual_v8_valid_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logger.info("complete verdict=%s output=%s log=%s", verdict, output_root, log_path)
    print("Frozen-S0 Sample-Conditioned Residual CFCompatKD v8 complete")
    print("candidate J:", result["J_valid"])
    print("S0 J:", result["S0_J_valid"])
    print("frozen v7 J:", v7_grid["J_valid"])
    print("frozen v4 J:", v4_grid["J_valid"])
    print("S0 state unchanged:", result["S0StateUnchanged"])
    print("residual trainable parameter count:", result["TrainableParameterCount"])
    if gate is not None:
        print(
            "nonbeneficial-Teacher NTR reduction vs v7:",
            gate["nonbeneficial_teacher_NTR_reduction_vs_v7"],
        )
        print(
            "beneficial-Teacher NTR degradation vs v7:",
            gate["beneficial_teacher_NTR_degradation_vs_v7"],
        )
    print("verdict:", verdict)
    print("official Test was not constructed or accessed")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
