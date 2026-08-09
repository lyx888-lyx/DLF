"""CFCompatKD v12.3: exact-replay window-level actual Adam-step audit.

No new candidate is defined.  The formal v12 numerical-hotfix fold trainer is
replayed exactly.  During replay, read-only hooks capture every optimizer
window in the predeclared epoch interval 4..12:

* accumulated supervised gradient;
* accumulated selective gradient;
* post-surgery update gradient;
* residual parameters immediately before and after the real Adam step.

No OOF diagnostic pass occurs until all five replay folds finish and the
formal v12 selected states/epochs/missing-mode hashes are verified exactly.
After verification, each captured finite step is evaluated on the fold's held-
out Train videos.  This separates raw window direction, Adam-transformed
parameter displacement, and realized finite-step OOF loss change.

Official Valid is not used for any v12.3 statistic or decision.  The legacy v4
asset loader may materialize its immutable Valid reference as a prerequisite.
Official Test is never constructed or accessed.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import types
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import MMDataLoader
from train_cf_compat_kd import build_config
import audit_cfcompat_objective_train_oof_v11 as v11audit
import audit_cfcompat_trajectory_train_oof_v12p2 as v12p2
import train_cfcompat_gradient_surgery_valid_screen_v12 as v12
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_crossfit_residual_consensus_valid_screen_v9 as v9
from trains.singleTask.cfcompat_gradient_surgery_numerical_hotfix import (
    asymmetric_project_supervised_stable,
)
from trains.singleTask.cfcompat_window_actual_step_audit_utils import (
    AUDIT_EPOCH_END,
    AUDIT_EPOCH_START,
    AUDIT_EPOCHS,
    DEV_SEED,
    KEY_OOF_GROUPS,
    METHOD,
    N_FOLDS,
    OUTPUT_TAG,
    VERSION,
    aggregate_mechanism,
    assign_parameter_tuple,
    clone_tensor_tuple,
    flatten_tensor_tuple,
    jsonable,
    overall_mechanism_counts,
    step_mechanism_row,
    vector_cosine,
)
from utils.functions import setup_seed


ORIGINAL_ADAM = torch.optim.Adam


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact-replay v12 window-level actual Adam-step Train-OOF audit v12.3"
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("v12.3 fixes num_workers=1 to match formal v12.")
    args.seeds = [DEV_SEED]
    args.max_epochs = None
    args.smoke_test = False
    return args


def output_paths(cli):
    result = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "train_oof"
        / "seed1113_dev"
    )
    replay_model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "train_oof"
        / "seed1113_dev"
    )
    if result.exists() or replay_model.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "v12.3 output exists; inspect it or use --overwrite: {} / {}".format(
                    result, replay_model
                )
            )
        if result.exists():
            shutil.rmtree(result)
        if replay_model.exists():
            shutil.rmtree(replay_model)
    result.mkdir(parents=True, exist_ok=True)
    replay_model.mkdir(parents=True, exist_ok=True)
    return result, replay_model


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "DLF-mosi-window-actual-step-audit-v12p3-{}.log".format(
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_window_actual_step_audit_v12p3")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


class WindowTracer:
    """Read-only capture of selected formal-v12 optimizer windows."""

    def __init__(self, fold: int, windows_per_epoch: int):
        self.fold = int(fold)
        self.windows_per_epoch = int(windows_per_epoch)
        if self.windows_per_epoch <= 0:
            raise ValueError("windows_per_epoch must be positive.")
        self.projection_call_count = 0
        self.current_record = None
        self.records = []

    def _meta_for_call(self):
        self.projection_call_count += 1
        zero = self.projection_call_count - 1
        epoch = zero // self.windows_per_epoch + 1
        window = zero % self.windows_per_epoch + 1
        return int(epoch), int(window)

    def projection(self, supervised, selective):
        projected, update, diagnostics = asymmetric_project_supervised_stable(
            supervised, selective
        )
        epoch, window = self._meta_for_call()
        if epoch in AUDIT_EPOCHS:
            if self.current_record is not None:
                raise RuntimeError("A captured projection was not followed by Adam.step().")
            self.current_record = {
                "Fold": self.fold,
                "Epoch": epoch,
                "UpdateWindow": window,
                "supervised_gradient": flatten_tensor_tuple(supervised),
                "selective_gradient": flatten_tensor_tuple(selective),
                "projected_supervised_gradient": flatten_tensor_tuple(projected),
                "surgery_update_gradient": flatten_tensor_tuple(update),
                "surgery_diagnostics": dict(diagnostics),
            }
        return projected, update, diagnostics

    def adam_step(self, parameters, step_callable, closure=None):
        record = self.current_record
        if record is None:
            return step_callable(closure=closure)
        before = clone_tensor_tuple(parameters)
        result = step_callable(closure=closure)
        after = clone_tensor_tuple(parameters)
        record["theta_before"] = before
        record["theta_after"] = after
        record["actual_delta"] = tuple(b - a for a, b in zip(before, after))
        record["actual_delta_vector"] = flatten_tensor_tuple(record["actual_delta"])
        record["adam_effective_direction_vector"] = -record["actual_delta_vector"]
        record["raw_to_adam_effective_cosine"] = vector_cosine(
            record["surgery_update_gradient"],
            record["adam_effective_direction_vector"],
        )
        self.records.append(record)
        self.current_record = None
        return result



def replay_fold_with_window_trace(
    cli,
    logger,
    args,
    fold,
    train_loader,
    holdout_loader,
    teacher,
    s0_student,
    evaluator_bundle,
    assets,
    replay_model_root,
    s0_sha,
):
    windows_per_epoch = int(math.ceil(len(train_loader) / float(args.update_epochs)))
    tracer = WindowTracer(fold, windows_per_epoch)

    class TracingAdam(ORIGINAL_ADAM):
        def step(self, closure=None):
            parameters = [
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            ]
            return tracer.adam_step(
                parameters,
                lambda closure=None: super(TracingAdam, self).step(closure=closure),
                closure=closure,
            )

    original_projection = v12.asymmetric_project_supervised
    original_optim = v12.optim
    v12.asymmetric_project_supervised = tracer.projection
    v12.optim = types.SimpleNamespace(Adam=TracingAdam)
    try:
        replay = v12p2.capture_exact_fold_replay(
            cli,
            logger,
            args,
            fold,
            train_loader,
            holdout_loader,
            teacher,
            s0_student,
            evaluator_bundle,
            assets,
            replay_model_root,
            s0_sha,
        )
    finally:
        v12.asymmetric_project_supervised = original_projection
        v12.optim = original_optim

    if tracer.current_record is not None:
        raise RuntimeError("Final captured projection was not followed by Adam.step().")
    states, epoch_frame, decisions, surgery_frame, fold_result, selected_state = replay
    expected = surgery_frame.loc[
        surgery_frame.Epoch.astype(int).between(AUDIT_EPOCH_START, AUDIT_EPOCH_END)
    ].sort_values(["Epoch", "UpdateWindow"], kind="mergesort")
    observed_keys = [(int(x["Epoch"]), int(x["UpdateWindow"])) for x in tracer.records]
    expected_keys = list(
        zip(expected.Epoch.astype(int).tolist(), expected.UpdateWindow.astype(int).tolist())
    )
    if observed_keys != expected_keys:
        raise RuntimeError(
            "Window trace keys differ from formal v12 surgery rows in fold {}.".format(fold)
        )
    for record, row in zip(tracer.records, expected.itertuples(index=False)):
        record["MicrobatchCount"] = int(row.MicrobatchCount)
        record["LearningRate"] = float(row.LearningRate)
        record["formal_conflict"] = bool(row.conflict)
        record["formal_pre_surgery_cosine"] = float(row.gradient_cosine_before)
        record["formal_removed_fraction"] = float(row.supervised_l2_removed_fraction)
    return replay, tracer.records


def group_losses(events: pd.DataFrame) -> dict:
    result = {}
    teacher = events.teacher_beneficial.astype(bool)
    s0 = events.s0_beneficial.astype(bool)
    masks = {
        "OOF_TEACHER_BENEFICIAL": teacher,
        "OOF_TEACHER_NONBENEFICIAL": ~teacher,
        "OOF_S0_BENEFICIAL": s0,
        "OOF_S0_NONBENEFICIAL": ~s0,
    }
    for group, mask in masks.items():
        local = events.loc[mask]
        if local.empty:
            raise RuntimeError("Empty OOF group: {}".format(group))
        result[group] = {
            "N": int(len(local)),
            "loss": float(local.current_error.astype(float).mean()),
        }
    return result


def trace_geometry_row(record):
    sup = record["supervised_gradient"]
    sel = record["selective_gradient"]
    update = record["surgery_update_gradient"]
    delta = record["actual_delta_vector"]
    return {
        "Fold": int(record["Fold"]),
        "Epoch": int(record["Epoch"]),
        "UpdateWindow": int(record["UpdateWindow"]),
        "MicrobatchCount": int(record["MicrobatchCount"]),
        "LearningRate": float(record["LearningRate"]),
        "formal_conflict": bool(record["formal_conflict"]),
        "formal_pre_surgery_cosine": float(record["formal_pre_surgery_cosine"]),
        "formal_removed_fraction": float(record["formal_removed_fraction"]),
        "supervised_gradient_l2": float(torch.linalg.vector_norm(sup)),
        "selective_gradient_l2": float(torch.linalg.vector_norm(sel)),
        "surgery_update_gradient_l2": float(torch.linalg.vector_norm(update)),
        "actual_delta_l2": float(torch.linalg.vector_norm(delta)),
        "raw_to_adam_effective_cosine": float(record["raw_to_adam_effective_cosine"]),
    }


def main():
    cli = parse_args()
    output_root, replay_model_root = output_paths(cli)
    logger, log_path = create_logger(cli)
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    if int(args.update_epochs) != 10:
        raise RuntimeError("v12.3 requires formal v12 update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v12.3 may construct only train/valid loaders; Test is forbidden.")

    formal_manifest, assignment, formal_checkpoints = v12p2.formal_v12_inputs(cli)
    expected_assignment = v9.deterministic_video_group_folds(
        list(loaders["train"].dataset.ids), N_FOLDS
    )
    columns = ["sample_index", "sample_id", "video_id", "fold"]
    if not assignment[columns].sort_values("sample_index").reset_index(drop=True).equals(
        expected_assignment[columns].sort_values("sample_index").reset_index(drop=True)
    ):
        raise RuntimeError("Frozen v12 fold assignment differs from deterministic reconstruction.")

    v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    try:
        teacher, s0_student, evaluator_bundle, assets = v4.load_assets(
            cli, args, loaders, DEV_SEED
        )
    finally:
        v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    s0_sha = v9.module_state_sha256(s0_student)

    # Phase 1: exact replay plus read-only window capture.  No OOF pass here.
    traces = {}
    replay_rows = []
    for fold in range(N_FOLDS):
        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset, assignment, fold, args, cli.num_workers
        )
        logger.info(
            "fold=%s exact v12 replay with read-only window trace epochs=%s..%s",
            fold,
            AUDIT_EPOCH_START,
            AUDIT_EPOCH_END,
        )
        replay, records = replay_fold_with_window_trace(
            cli,
            logger,
            args,
            fold,
            train_loader,
            holdout_loader,
            teacher,
            s0_student,
            evaluator_bundle,
            assets,
            replay_model_root,
            s0_sha,
        )
        states, epoch_frame, decisions, surgery_frame, fold_result, selected_state = replay
        check = v12p2.verify_replay_fold(
            fold,
            fold_result,
            selected_state,
            formal_manifest,
            formal_checkpoints[fold],
        )
        check["TrainN"] = int(len(train_indices))
        check["OOFN"] = int(len(holdout_indices))
        check["CapturedWindowN"] = int(len(records))
        replay_rows.append(check)
        traces[fold] = records
        logger.info(
            "fold=%s replay exact; captured_windows=%s selected=%s best=%s",
            fold,
            len(records),
            fold_result["ConservativeSelectedEpoch"],
            fold_result["AbsoluteBestTrainHoldoutEpoch"],
        )

    replay_manifest = pd.DataFrame(replay_rows)
    exact_columns = [
        "selected_epoch_exact",
        "absolute_best_epoch_exact",
        "missing_sequence_hash_exact",
        "selected_state_content_hash_exact",
        "selected_state_tensor_exact",
    ]
    if not bool(replay_manifest[exact_columns].all().all()):
        raise RuntimeError("Five-fold replay was not exact; no v12.3 interpretation allowed.")
    if v9.module_state_sha256(s0_student) != s0_sha:
        raise RuntimeError("Frozen S0 changed during v12.3 replay.")

    # Phase 2: all training is over.  Replay captured windows offline against
    # their fold's held-out Train videos.
    mechanism_rows = []
    geometry_rows = []
    for fold in range(N_FOLDS):
        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset, assignment, fold, args, cli.num_workers
        )
        audit_model = v9.fresh_fold_student(s0_student, args.device)
        parameters = list(audit_model.bank.parameters())
        logger.info("fold=%s offline actual-step audit windows=%s", fold, len(traces[fold]))
        for index, record in enumerate(traces[fold], 1):
            assign_parameter_tuple(parameters, record["theta_before"])
            before_events, oof_gradients, oof_counts = (
                v11audit.audit_oof_events_and_loss_gradients(
                    fold,
                    audit_model,
                    holdout_loader,
                    teacher,
                    evaluator_bundle,
                    assets["cache_by_index"],
                    args,
                    assignment,
                )
            )
            before_events = v11audit.enrich_oof_event_frame(before_events)
            before_losses = group_losses(before_events)

            assign_parameter_tuple(parameters, record["theta_after"])
            after_events = v12p2.snapshot_oof_events(
                fold,
                int(record["Epoch"]),
                audit_model,
                holdout_loader,
                teacher,
                evaluator_bundle,
                assets["cache_by_index"],
                args,
                assignment,
            )
            after_losses = group_losses(after_events)

            for group in KEY_OOF_GROUPS:
                oof_gradient = oof_gradients["ALL"][group]
                count = int(oof_counts.get(("ALL", group), 0))
                if count <= 0:
                    raise RuntimeError("Missing OOF gradient count for {}.".format(group))
                row = step_mechanism_row(
                    fold=fold,
                    epoch=int(record["Epoch"]),
                    window=int(record["UpdateWindow"]),
                    oof_group=group,
                    oof_count=count,
                    surgery_gradient=record["surgery_update_gradient"],
                    actual_delta=record["actual_delta_vector"],
                    oof_gradient_before=oof_gradient,
                    loss_before=before_losses[group]["loss"],
                    loss_after=after_losses[group]["loss"],
                )
                row["MicrobatchCount"] = int(record["MicrobatchCount"])
                row["LearningRate"] = float(record["LearningRate"])
                row["formal_conflict"] = bool(record["formal_conflict"])
                row["formal_pre_surgery_cosine"] = float(record["formal_pre_surgery_cosine"])
                row["formal_removed_fraction"] = float(record["formal_removed_fraction"])
                mechanism_rows.append(row)
            geometry_rows.append(trace_geometry_row(record))
            if index % 8 == 0 or index == len(traces[fold]):
                logger.info(
                    "fold=%s audited_window=%s/%s epoch=%s window=%s",
                    fold,
                    index,
                    len(traces[fold]),
                    record["Epoch"],
                    record["UpdateWindow"],
                )
        del audit_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mechanism = pd.DataFrame(mechanism_rows)
    geometry = pd.DataFrame(geometry_rows)
    if mechanism.empty or geometry.empty:
        raise RuntimeError("v12.3 produced no window diagnostics.")
    if geometry[["Fold", "Epoch", "UpdateWindow"]].duplicated().any():
        raise RuntimeError("Duplicate captured optimizer window detected.")
    aggregate = aggregate_mechanism(mechanism)

    replay_manifest.to_csv(output_root / "window_v12p3_exact_replay_manifest.csv", index=False)
    geometry.to_csv(output_root / "window_v12p3_optimizer_geometry.csv", index=False)
    mechanism.to_csv(output_root / "window_v12p3_mechanism_by_window_group.csv", index=False)
    aggregate.to_csv(output_root / "window_v12p3_mechanism_epoch_summary.csv", index=False)

    headline_groups = (
        "OOF_TEACHER_BENEFICIAL",
        "OOF_S0_BENEFICIAL",
        "OOF_TEACHER_NONBENEFICIAL",
    )
    headline = {group: overall_mechanism_counts(mechanism, group) for group in headline_groups}
    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": "DIAGNOSTIC_ONLY_NO_MODEL_PROMOTION_DECISION",
        "protocol": {
            "development_seed": DEV_SEED,
            "new_candidate_model_defined": False,
            "formal_v12_training_function_reused": True,
            "formal_v12_numerical_hotfix_reused": True,
            "exact_replay_required_before_oof_window_analysis": True,
            "audit_epoch_start": AUDIT_EPOCH_START,
            "audit_epoch_end": AUDIT_EPOCH_END,
            "audit_epochs": list(AUDIT_EPOCHS),
            "all_optimizer_windows_in_audit_epochs_included": True,
            "oof_diagnostics_deferred_until_all_five_folds_finish_training": True,
            "fold_grouping": "whole_video_id_same_as_v12",
            "official_valid_used_for_audit_statistics": False,
            "official_valid_reference_materialized_by_legacy_v4_asset_loader": True,
            "official_test_constructed": False,
            "official_test_accessed": False,
        },
        "exact_replay": {
            "all_folds_exact": True,
            "fold_manifest": jsonable(replay_manifest.to_dict("records")),
        },
        "mechanism_interpretation": {
            "raw_surgery_direction_harm": "raw post-surgery gradient itself is first-order harmful to the OOF group",
            "adam_transform_harm": "raw surgery gradient is first-order safe but actual Adam parameter displacement is first-order harmful",
            "nonlinear_finite_step_harm": "raw and actual Adam displacement are first-order safe but realized finite OOF loss increases",
            "safe_or_improving": "none of the three harm conditions is present",
            "sign_only_no_tuned_magnitude_thresholds": True,
        },
        "headline": jsonable(headline),
        "captured_optimizer_window_count": int(len(geometry)),
        "window_group_row_count": int(len(mechanism)),
        "mean_raw_to_adam_effective_cosine": float(
            geometry.raw_to_adam_effective_cosine.mean()
        ),
        "frozen_s0_state_sha256": s0_sha,
    }
    summary_path = output_root / "window_v12p3_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("CFCompatKD v12.3 window-level actual-step audit complete")
    print("formal v12 exact replay folds: 5 / 5")
    print("new candidate model defined: False")
    print("audited epochs:", list(AUDIT_EPOCHS))
    print("captured optimizer windows:", len(geometry))
    for group in headline_groups:
        local = headline[group]
        print(
            group,
            "raw_harm={}/{} adam_transform_harm={}/{} nonlinear_finite_harm={}/{} safe={}/{}".format(
                local["raw_direction_harm_count"], local["window_count"],
                local["adam_transform_harm_count"], local["window_count"],
                local["nonlinear_finite_step_harm_count"], local["window_count"],
                local["safe_or_improving_count"], local["window_count"],
            ),
        )
    print("Official Test was not constructed or accessed")
    print("summary:", summary_path)
    logger.info(
        "complete windows=%s rows=%s output=%s log=%s",
        len(geometry),
        len(mechanism),
        output_root,
        log_path,
    )


if __name__ == "__main__":
    main()
