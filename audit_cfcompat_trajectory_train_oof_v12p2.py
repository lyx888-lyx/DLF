"""CFCompatKD v12.2: exact-replay, trajectory-resolved Train-OOF audit.

This script defines no new candidate model.  It reruns the formal v12 training
function with the numerical hotfix and captures the residual-bank state at the
same epoch snapshot call already present in v12.  No extra forward/backward
pass is performed until *all five folds have finished training*, so diagnostic
work cannot perturb the replay RNG path.

The replay is accepted only if each fold reproduces the frozen v12 selected
checkpoint state exactly, the selected/best epochs, and the missing-mode
sequence hash.  If any check fails, the trajectory audit stops without an
interpretation.

After replay is verified, every captured epoch is evaluated on that fold's
held-out Train videos.  At fixed predeclared milestones plus the frozen
conservative-selected and absolute-best epochs, the script computes the same
post-surgery first-order gradient geometry as v12.1.

Official Valid is not used for any trajectory statistic or decision.  The
legacy v4 asset loader may materialize its immutable Valid reference as an
implementation prerequisite.  Official Test is never constructed or accessed.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import MMDataLoader
from train_cf_compat_kd import batch_to_device, build_config
import audit_cfcompat_objective_train_oof_v11 as v11audit
import train_cfcompat_gradient_surgery_valid_screen_v12 as v12
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_crossfit_residual_consensus_valid_screen_v9 as v9
from trains.singleTask.cf_compat_kd_utils import compatibility_for_modes
from trains.singleTask.cfcompat_gradient_surgery_numerical_hotfix import (
    asymmetric_project_supervised_stable,
)
from trains.singleTask.cfcompat_objective_audit_utils import (
    AUDIT_MODES,
    OOF_GROUPS,
    enrich_oof_event_frame,
    gradient_influence_row,
)
from trains.singleTask.cfcompat_post_surgery_audit_utils import (
    POST_SURGERY_COMPONENTS,
    replay_asymmetric_projection,
)
from trains.singleTask.cfcompat_regret_preserve_utils import regret_preserve_decision
from trains.singleTask.cfcompat_trajectory_audit_utils import (
    DEV_SEED,
    FIXED_GRADIENT_MILESTONES,
    METHOD,
    N_FOLDS,
    OUTPUT_TAG,
    VERSION,
    aggregate_epoch_groups,
    clip_sync_summary,
    clone_state_dict,
    enrich_transfer_flags,
    epoch_group_summary,
    failure_onset_table,
    jsonable,
    milestone_roles,
    state_dict_sha256,
    states_exactly_equal,
)
from trains.singleTask.fixed_kd_utils import teacher_lav_prediction
from trains.singleTask.missing_utils import MISSING_MODES, mode_to_mask
from utils.functions import setup_seed


# Use the exact numerical realization that produced the formal v12 result.
v12.asymmetric_project_supervised = asymmetric_project_supervised_stable


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exact-replay trajectory-resolved Train-OOF audit v12.2"
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
        parser.error("v12.2 fixes num_workers=1 to match formal v12.")
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
                "v12.2 output exists; inspect it or use --overwrite: {} / {}".format(
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
    path = directory / "DLF-mosi-trajectory-audit-v12p2-{}.log".format(
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_trajectory_audit_v12p2")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def formal_v12_inputs(cli):
    result_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_gradient_surgery_v12"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    model_root = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cfcompat_gradient_surgery_v12"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    manifest_path = result_root / "gradient_surgery_v12_fold_manifest.csv"
    assignment_path = result_root / "gradient_surgery_v12_fold_assignment.csv"
    if not manifest_path.is_file() or not assignment_path.is_file():
        raise FileNotFoundError("Frozen v12 manifest/assignment missing under {}".format(result_root))
    manifest = pd.read_csv(manifest_path)
    assignment = pd.read_csv(assignment_path)
    if len(manifest) != N_FOLDS or set(manifest.Fold.astype(int)) != set(range(N_FOLDS)):
        raise RuntimeError("Frozen v12 manifest must contain folds 0..4 exactly.")
    if len(assignment) != 1284 or assignment.sample_index.nunique() != 1284:
        raise RuntimeError("Frozen v12 assignment must contain 1284 unique Train samples.")
    if assignment.groupby("video_id").fold.nunique().max() != 1:
        raise RuntimeError("Frozen v12 assignment leaks a video across folds.")
    checkpoints = {}
    for fold in range(N_FOLDS):
        path = model_root / "seed1113" / "fold{}".format(fold) / "conservative_train_holdout_bank.pth"
        if not path.is_file():
            raise FileNotFoundError("Frozen formal v12 checkpoint missing: {}".format(path))
        checkpoints[fold] = path
    return manifest, assignment, checkpoints


def capture_exact_fold_replay(
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
    """Run the unmodified v12 fold trainer while observing its snapshot calls."""
    original_fresh = v12.v9.fresh_fold_student
    original_bank_state = v12.bank_state_cpu
    initial_holder = {}
    captured_epoch_states = []

    def fresh_wrapper(s0, device):
        student = original_fresh(s0, device)
        if initial_holder:
            raise RuntimeError("v12 fold trainer created more than one fresh residual student.")
        initial_holder["state"] = clone_state_dict(original_bank_state(student))
        return student

    def snapshot_wrapper(student):
        state = original_bank_state(student)
        captured_epoch_states.append(clone_state_dict(state))
        return state

    v12.v9.fresh_fold_student = fresh_wrapper
    v12.bank_state_cpu = snapshot_wrapper
    try:
        result = v12.train_one_fold_surgery(
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
        v12.v9.fresh_fold_student = original_fresh
        v12.bank_state_cpu = original_bank_state

    selected_state, epoch_frame, decisions, surgery_frame, fold_result = result
    if "state" not in initial_holder:
        raise RuntimeError("Failed to capture v12 fold initial state.")
    if len(captured_epoch_states) != len(epoch_frame):
        raise RuntimeError(
            "Epoch snapshot hook mismatch fold {}: {} states vs {} rows".format(
                fold, len(captured_epoch_states), len(epoch_frame)
            )
        )
    epochs = epoch_frame.Epoch.astype(int).tolist()
    if epochs != list(range(1, len(epochs) + 1)):
        raise RuntimeError("v12 replay epoch rows are not consecutive from 1.")
    states = {0: clone_state_dict(initial_holder["state"])}
    for epoch, state in zip(epochs, captured_epoch_states):
        states[int(epoch)] = clone_state_dict(state)
    selected_epoch = int(fold_result["ConservativeSelectedEpoch"])
    if not states_exactly_equal(states[selected_epoch], selected_state):
        raise RuntimeError("Captured selected epoch state differs from v12 return state.")
    return states, epoch_frame, decisions, surgery_frame, fold_result, selected_state


def verify_replay_fold(
    fold,
    fold_result,
    selected_state,
    formal_manifest,
    formal_checkpoint,
):
    formal_row = formal_manifest.loc[formal_manifest.Fold.astype(int).eq(int(fold))]
    if len(formal_row) != 1:
        raise RuntimeError("Frozen v12 manifest has no unique row for fold {}.".format(fold))
    formal_row = formal_row.iloc[0]
    formal_state = torch.load(formal_checkpoint, map_location="cpu")
    replay_hash = state_dict_sha256(selected_state)
    formal_hash = state_dict_sha256(formal_state)
    checks = {
        "selected_epoch_exact": int(fold_result["ConservativeSelectedEpoch"]) == int(formal_row.ConservativeSelectedEpoch),
        "absolute_best_epoch_exact": int(fold_result["AbsoluteBestTrainHoldoutEpoch"]) == int(formal_row.AbsoluteBestTrainHoldoutEpoch),
        "missing_sequence_hash_exact": str(fold_result["MissingSequenceSHA256"]) == str(formal_row.MissingSequenceSHA256),
        "selected_state_content_hash_exact": replay_hash == formal_hash,
        "selected_state_tensor_exact": states_exactly_equal(selected_state, formal_state),
    }
    if not all(checks.values()):
        raise RuntimeError(
            "v12 exact replay failed for fold {}: {}. No trajectory interpretation is allowed.".format(
                fold, checks
            )
        )
    return {
        "Fold": int(fold),
        **checks,
        "ReplaySelectedEpoch": int(fold_result["ConservativeSelectedEpoch"]),
        "FormalSelectedEpoch": int(formal_row.ConservativeSelectedEpoch),
        "ReplayAbsoluteBestEpoch": int(fold_result["AbsoluteBestTrainHoldoutEpoch"]),
        "FormalAbsoluteBestEpoch": int(formal_row.AbsoluteBestTrainHoldoutEpoch),
        "ReplayMissingSequenceSHA256": str(fold_result["MissingSequenceSHA256"]),
        "FormalMissingSequenceSHA256": str(formal_row.MissingSequenceSHA256),
        "ReplaySelectedStateSHA256": replay_hash,
        "FormalSelectedStateSHA256": formal_hash,
    }


def snapshot_oof_events(
    fold,
    epoch,
    model,
    holdout_loader,
    teacher,
    evaluator_bundle,
    cache_by_index,
    args,
    assignment,
):
    """Forward-only exhaustive LA/LV/L OOF snapshot; no gradients and no RNG use."""
    model.eval()
    assignment_by_index = assignment.set_index("sample_index", drop=False)
    rows = []
    with torch.no_grad():
        for batch in holdout_loader:
            text, audio, vision, labels = batch_to_device(batch, args.device)
            indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
            identifiers = list(batch["id"])
            teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision).view(-1, 1)
            label_flat = labels.view(-1)
            batch_size = int(labels.size(0))
            for mode in MISSING_MODES:
                mode_list = [mode] * batch_size
                mask = mode_to_mask(mode, batch_size, args.device, audio.dtype)
                output = model(text, audio, vision, mask)
                prediction = output["output_logit"].view(-1)
                s0_prediction = output["s0_output_logit"].view(-1)
                residual_delta = output["residual_delta"].view(-1)
                compatibility = compatibility_for_modes(
                    cache_by_index, indices, mode_list, args.device, labels.dtype
                )
                baseline = v4.baseline_for_modes(
                    evaluator_bundle,
                    indices,
                    mode_list,
                    labels,
                    args.device,
                    labels.dtype,
                )
                decision = regret_preserve_decision(
                    prediction.view(-1, 1),
                    teacher_prediction,
                    baseline,
                    labels,
                    compatibility,
                )
                for i, sample_index in enumerate(indices):
                    assigned = assignment_by_index.loc[int(sample_index)]
                    if int(assigned.fold) != int(fold):
                        raise RuntimeError("OOF sample bound to wrong fold during trajectory snapshot.")
                    rows.append(
                        {
                            "Fold": int(fold),
                            "Epoch": int(epoch),
                            "fold": int(fold),
                            "sample_index": int(sample_index),
                            "sample_id": str(identifiers[i]),
                            "video_id": str(assigned.video_id),
                            "mode": str(mode),
                            "label": float(label_flat[i].cpu()),
                            "baseline_prediction": float(baseline.view(-1)[i].cpu()),
                            "s0_prediction": float(s0_prediction[i].cpu()),
                            "student_prediction": float(prediction[i].cpu()),
                            "residual_delta": float(residual_delta[i].cpu()),
                            "teacher_prediction": float(teacher_prediction.view(-1)[i].cpu()),
                            "teacher_safe_target": float(decision["teacher_safe_target"].view(-1)[i].cpu()),
                            "preserve_safe_target": float(decision["preserve_safe_target"].view(-1)[i].cpu()),
                            "distill": bool(decision["distill"].view(-1)[i].cpu()),
                            "preserve": bool(decision["preserve"].view(-1)[i].cpu()),
                            "decision_abstain": bool(decision["abstain"].view(-1)[i].cpu()),
                            "teacher_beneficial": bool(decision["teacher_beneficial"].view(-1)[i].cpu()),
                            "current_regressed": bool(decision["current_regressed"].view(-1)[i].cpu()),
                            "teacher_advantage_vs_baseline": float(decision["teacher_advantage_vs_baseline"].view(-1)[i].cpu()),
                            "current_regret_vs_baseline": float(decision["current_regret_vs_baseline"].view(-1)[i].cpu()),
                            "compatibility": float(compatibility.view(-1)[i].cpu()),
                            "distill_gate": float(decision["distill_gate"].view(-1)[i].cpu()),
                            "preserve_gate": float(decision["preserve_gate"].view(-1)[i].cpu()),
                        }
                    )
    frame = enrich_oof_event_frame(pd.DataFrame(rows))
    frame["Fold"] = int(fold)
    frame["Epoch"] = int(epoch)
    return enrich_transfer_flags(frame)


def aggregate_milestone_influence(frame: pd.DataFrame, keys):
    rows = []
    for values, local in frame.groupby(list(keys), sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        row = {key: value for key, value in zip(keys, values)}
        row.update(
            {
                "FoldCount": int(local.Fold.nunique()),
                "mean_gradient_dot": float(local.gradient_dot.mean()),
                "median_gradient_dot": float(local.gradient_dot.median()),
                "mean_gradient_cosine": float(local.gradient_cosine.mean()),
                "median_gradient_cosine": float(local.gradient_cosine.median()),
                "harm_fold_count": int(local.predicted_effect_of_gradient_descent.eq("HARM_OOF").sum()),
                "improve_fold_count": int(local.predicted_effect_of_gradient_descent.eq("IMPROVE_OOF").sum()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    cli = parse_args()
    output_root, replay_model_root = output_paths(cli)
    logger, log_path = create_logger(cli)
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    if int(args.update_epochs) != 10:
        raise RuntimeError("v12.2 requires formal v12 update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v12.2 may construct only train/valid loaders; Test is forbidden.")

    formal_manifest, assignment, formal_checkpoints = formal_v12_inputs(cli)
    expected_assignment = v9.deterministic_video_group_folds(
        list(loaders["train"].dataset.ids), N_FOLDS
    )
    compare_columns = ["sample_index", "sample_id", "video_id", "fold"]
    left = assignment[compare_columns].sort_values("sample_index").reset_index(drop=True)
    right = expected_assignment[compare_columns].sort_values("sample_index").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("Frozen v12 fold assignment differs from deterministic reconstruction.")

    v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    try:
        teacher, s0_student, evaluator_bundle, assets = v4.load_assets(
            cli, args, loaders, DEV_SEED
        )
    finally:
        v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    s0_sha = v9.module_state_sha256(s0_student)

    # Phase 1: exact five-fold replay.  Do not run any additional diagnostic
    # model/data pass until this phase is completely finished.
    captured = {}
    replay_rows = []
    replay_epoch_frames = []
    replay_surgery_frames = []
    fold_results = {}
    for fold in range(N_FOLDS):
        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset, assignment, fold, args, cli.num_workers
        )
        logger.info("fold=%s exact v12 replay starting; diagnostics deferred", fold)
        states, epoch_frame, decisions, surgery_frame, fold_result, selected_state = capture_exact_fold_replay(
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
        replay_check = verify_replay_fold(
            fold,
            fold_result,
            selected_state,
            formal_manifest,
            formal_checkpoints[fold],
        )
        replay_check["TrainN"] = int(len(train_indices))
        replay_check["OOFN"] = int(len(holdout_indices))
        replay_rows.append(replay_check)
        epoch_frame = epoch_frame.copy()
        epoch_frame["Fold"] = int(fold)
        surgery_frame = surgery_frame.copy()
        surgery_frame["Fold"] = int(fold)
        replay_epoch_frames.append(epoch_frame)
        replay_surgery_frames.append(surgery_frame)
        captured[fold] = states
        fold_results[fold] = fold_result
        logger.info(
            "fold=%s exact replay verified selected=%s best=%s epochs=%s",
            fold,
            fold_result["ConservativeSelectedEpoch"],
            fold_result["AbsoluteBestTrainHoldoutEpoch"],
            len(epoch_frame),
        )

    replay_manifest = pd.DataFrame(replay_rows)
    if not bool(
        replay_manifest[
            [
                "selected_epoch_exact",
                "absolute_best_epoch_exact",
                "missing_sequence_hash_exact",
                "selected_state_content_hash_exact",
                "selected_state_tensor_exact",
            ]
        ].all().all()
    ):
        raise RuntimeError("Five-fold replay did not exactly reproduce formal v12.")
    if v9.module_state_sha256(s0_student) != s0_sha:
        raise RuntimeError("Frozen S0 changed during exact v12 replay.")

    # Phase 2: post-hoc trajectory observation.  Training is over, so these
    # passes cannot change the replayed optimizer/RNG trajectory.
    all_events = []
    gradient_rows = []
    projection_rows = []
    for fold in range(N_FOLDS):
        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset, assignment, fold, args, cli.num_workers
        )
        audit_model = v9.fresh_fold_student(s0_student, args.device)
        available_epochs = sorted(captured[fold])
        roles = milestone_roles(
            available_epochs,
            int(fold_results[fold]["ConservativeSelectedEpoch"]),
            int(fold_results[fold]["AbsoluteBestTrainHoldoutEpoch"]),
        )
        logger.info(
            "fold=%s post-hoc trajectory audit epochs=%s milestones=%s",
            fold,
            len(available_epochs),
            sorted(roles),
        )
        for epoch in available_epochs:
            audit_model.bank.load_state_dict(captured[fold][epoch], strict=True)
            if v9.module_state_sha256(audit_model.s0) != s0_sha:
                raise RuntimeError("S0 drifted during post-hoc fold {} epoch {} audit.".format(fold, epoch))
            events = snapshot_oof_events(
                fold,
                epoch,
                audit_model,
                holdout_loader,
                teacher,
                evaluator_bundle,
                assets["cache_by_index"],
                args,
                assignment,
            )
            if len(events) != len(holdout_indices) * len(MISSING_MODES):
                raise RuntimeError("Unexpected OOF event count fold {} epoch {}.".format(fold, epoch))
            all_events.append(events)

            if epoch not in roles:
                continue
            # Gradient audit only at frozen milestones/endpoints.
            _, oof_gradients, oof_counts = v11audit.audit_oof_events_and_loss_gradients(
                fold,
                audit_model,
                holdout_loader,
                teacher,
                evaluator_bundle,
                assets["cache_by_index"],
                args,
                assignment,
            )
            train_gradients = v11audit.audit_train_component_gradients(
                audit_model,
                train_loader,
                teacher,
                evaluator_bundle,
                assets["cache_by_index"],
                args,
            )
            for mode in AUDIT_MODES:
                surgery = replay_asymmetric_projection(
                    train_gradients[mode]["SUPERVISED_ALL"],
                    train_gradients[mode]["SELECTIVE_ONLY"],
                )
                diagnostic = dict(surgery["diagnostics"])
                diagnostic.update(
                    {
                        "Fold": int(fold),
                        "Epoch": int(epoch),
                        "EpochRole": str(roles[epoch]),
                        "Mode": str(mode),
                    }
                )
                projection_rows.append(diagnostic)
                for component in POST_SURGERY_COMPONENTS:
                    for group in OOF_GROUPS:
                        count = int(oof_counts.get((mode, group), 0))
                        if count <= 0:
                            continue
                        row = gradient_influence_row(
                            fold,
                            mode,
                            component,
                            group,
                            surgery[component],
                            oof_gradients[mode][group],
                            count,
                        )
                        row["Epoch"] = int(epoch)
                        row["EpochRole"] = str(roles[epoch])
                        gradient_rows.append(row)
        del audit_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    events = pd.concat(all_events, ignore_index=True)
    if events[["Fold", "Epoch", "sample_index", "mode"]].duplicated().any():
        raise RuntimeError("Duplicate trajectory OOF event detected.")
    per_fold_groups = epoch_group_summary(events)
    aggregate_groups = aggregate_epoch_groups(per_fold_groups)
    onset = failure_onset_table(events)
    clip_sync = clip_sync_summary(events)
    gradient_by_fold = pd.DataFrame(gradient_rows)
    projection = pd.DataFrame(projection_rows)
    fixed_gradient_summary = aggregate_milestone_influence(
        gradient_by_fold,
        ("Epoch", "Mode", "TrainComponent", "OOFGroup"),
    )

    selected_rows = gradient_by_fold.loc[
        gradient_by_fold.EpochRole.astype(str).str.contains("CONSERVATIVE_SELECTED")
    ].copy()
    best_rows = gradient_by_fold.loc[
        gradient_by_fold.EpochRole.astype(str).str.contains("ABSOLUTE_BEST")
    ].copy()
    selected_summary = aggregate_milestone_influence(
        selected_rows,
        ("Mode", "TrainComponent", "OOFGroup"),
    )
    best_summary = aggregate_milestone_influence(
        best_rows,
        ("Mode", "TrainComponent", "OOFGroup"),
    )

    replay_epoch_metrics = pd.concat(replay_epoch_frames, ignore_index=True)
    replay_surgery_windows = pd.concat(replay_surgery_frames, ignore_index=True)

    events.to_csv(output_root / "trajectory_v12p2_train_oof_epoch_events.csv", index=False)
    per_fold_groups.to_csv(output_root / "trajectory_v12p2_epoch_group_summary_by_fold.csv", index=False)
    aggregate_groups.to_csv(output_root / "trajectory_v12p2_epoch_group_summary.csv", index=False)
    onset.to_csv(output_root / "trajectory_v12p2_failure_onset.csv", index=False)
    clip_sync.to_csv(output_root / "trajectory_v12p2_clip_sync_summary.csv", index=False)
    gradient_by_fold.to_csv(output_root / "trajectory_v12p2_gradient_influence_by_fold.csv", index=False)
    fixed_gradient_summary.to_csv(output_root / "trajectory_v12p2_fixed_milestone_gradient_summary.csv", index=False)
    selected_summary.to_csv(output_root / "trajectory_v12p2_selected_gradient_summary.csv", index=False)
    best_summary.to_csv(output_root / "trajectory_v12p2_absolute_best_gradient_summary.csv", index=False)
    projection.to_csv(output_root / "trajectory_v12p2_milestone_projection_geometry.csv", index=False)
    replay_manifest.to_csv(output_root / "trajectory_v12p2_exact_replay_manifest.csv", index=False)
    replay_epoch_metrics.to_csv(output_root / "trajectory_v12p2_replay_fold_epoch_metrics.csv", index=False)
    replay_surgery_windows.to_csv(output_root / "trajectory_v12p2_replay_update_windows.csv", index=False)

    def headline(component, group, source):
        local = source.loc[
            source.Mode.astype(str).eq("ALL")
            & source.TrainComponent.astype(str).eq(component)
            & source.OOFGroup.astype(str).eq(group)
        ]
        if len(local) != 1:
            return {}
        row = local.iloc[0]
        return {
            "mean_gradient_cosine": float(row.mean_gradient_cosine),
            "mean_gradient_dot": float(row.mean_gradient_dot),
            "harm_fold_count": int(row.harm_fold_count),
            "improve_fold_count": int(row.improve_fold_count),
            "fold_count": int(row.FoldCount),
        }

    fixed_teacher_update = fixed_gradient_summary.loc[
        fixed_gradient_summary.Mode.astype(str).eq("ALL")
        & fixed_gradient_summary.TrainComponent.astype(str).eq("SURGERY_UPDATE")
        & fixed_gradient_summary.OOFGroup.astype(str).eq("OOF_TEACHER_BENEFICIAL")
    ].sort_values("Epoch")
    fixed_s0_update = fixed_gradient_summary.loc[
        fixed_gradient_summary.Mode.astype(str).eq("ALL")
        & fixed_gradient_summary.TrainComponent.astype(str).eq("SURGERY_UPDATE")
        & fixed_gradient_summary.OOFGroup.astype(str).eq("OOF_S0_BENEFICIAL")
    ].sort_values("Epoch")

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": "DIAGNOSTIC_ONLY_NO_MODEL_PROMOTION_DECISION",
        "protocol": {
            "development_seed": DEV_SEED,
            "new_candidate_model_defined": False,
            "formal_v12_training_function_reused": True,
            "formal_v12_numerical_hotfix_reused": True,
            "diagnostics_deferred_until_all_five_folds_finish_training": True,
            "exact_replay_required_before_interpretation": True,
            "fixed_gradient_milestones": list(FIXED_GRADIENT_MILESTONES),
            "selected_and_absolute_best_epochs_audited": True,
            "fold_grouping": "whole_video_id_same_as_v12",
            "official_valid_used_for_trajectory_statistics": False,
            "official_valid_reference_materialized_by_legacy_v4_asset_loader": True,
            "official_test_constructed": False,
            "official_test_accessed": False,
        },
        "exact_replay": {
            "all_folds_exact": True,
            "fold_manifest": jsonable(replay_manifest.to_dict("records")),
        },
        "gradient_interpretation": {
            "positive_dot": "gradient descent on audited direction locally decreases OOF MAE",
            "negative_dot": "gradient descent on audited direction locally increases OOF MAE",
            "scope": "post_hoc_first_order_geometry_at_captured_historical_v12_epoch_states",
            "not_claimed": "does not equate expected full-Train gradient with each finite Adam update window",
        },
        "selected_endpoint": {
            "surgery_update_teacher_beneficial": headline(
                "SURGERY_UPDATE", "OOF_TEACHER_BENEFICIAL", selected_summary
            ),
            "surgery_update_s0_beneficial": headline(
                "SURGERY_UPDATE", "OOF_S0_BENEFICIAL", selected_summary
            ),
            "surgery_update_teacher_nonbeneficial": headline(
                "SURGERY_UPDATE", "OOF_TEACHER_NONBENEFICIAL", selected_summary
            ),
        },
        "fixed_milestone_teacher_beneficial_surgery_update": jsonable(
            fixed_teacher_update[
                [
                    "Epoch",
                    "FoldCount",
                    "mean_gradient_dot",
                    "mean_gradient_cosine",
                    "harm_fold_count",
                    "improve_fold_count",
                ]
            ].to_dict("records")
        ),
        "fixed_milestone_s0_beneficial_surgery_update": jsonable(
            fixed_s0_update[
                [
                    "Epoch",
                    "FoldCount",
                    "mean_gradient_dot",
                    "mean_gradient_cosine",
                    "harm_fold_count",
                    "improve_fold_count",
                ]
            ].to_dict("records")
        ),
        "trajectory_event_count": int(len(events)),
        "frozen_s0_state_sha256": s0_sha,
    }
    summary_path = output_root / "trajectory_v12p2_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    logger.info(
        "complete exact_replay=5/5 events=%s selected_teacher_beneficial_harm=%s selected_s0_beneficial_harm=%s output=%s log=%s",
        len(events),
        summary["selected_endpoint"]["surgery_update_teacher_beneficial"].get("harm_fold_count"),
        summary["selected_endpoint"]["surgery_update_s0_beneficial"].get("harm_fold_count"),
        output_root,
        log_path,
    )
    print("CFCompatKD v12.2 trajectory Train-OOF audit complete")
    print("formal v12 exact replay folds: 5 / 5")
    print("new candidate model defined: False")
    print("trajectory OOF events:", len(events))
    print("fixed gradient milestones:", list(FIXED_GRADIENT_MILESTONES))
    print("Official Test was not constructed or accessed")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
