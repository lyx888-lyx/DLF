"""CFCompatKD v12.1: pure post-surgery Train-OOF gradient audit.

No model is trained and no checkpoint is selected.  The script loads the five
frozen v12 conservative residual banks.  Each MOSI Train sample is evaluated
only by the fold that held out its whole video.  At each frozen checkpoint it
uses the same exhaustive LA/LV/L expected-moddrop decomposition as v11,
replays the analytic v12 asymmetric projection in float64 gradient-vector
space, and measures first-order alignment of raw/projected/update directions
with held-out Train subgroup MAE gradients.

This is a local frozen-checkpoint diagnostic, not an exact replay of the
historical Adam trajectory.  Official Test is never constructed or accessed.
Official Valid is not used in any audit statistic or decision; the legacy v4
asset loader may materialize its immutable reference as a prerequisite.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from data_loader import MMDataLoader
from train_cf_compat_kd import build_config
import audit_cfcompat_objective_train_oof_v11 as v11audit
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_crossfit_residual_consensus_valid_screen_v9 as v9
from trains.singleTask.cfcompat_objective_audit_utils import (
    AUDIT_MODES,
    OOF_GROUPS,
    aggregate_influence,
    enrich_oof_event_frame,
    event_summary,
    gradient_influence_row,
)
from trains.singleTask.cfcompat_post_surgery_audit_utils import (
    DEV_SEED,
    METHOD,
    N_FOLDS,
    OUTPUT_TAG,
    POST_SURGERY_COMPONENTS,
    VERSION,
    headline_findings,
    jsonable,
    replay_asymmetric_projection,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frozen-v12 post-surgery Train-OOF gradient audit v12.1"
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
        parser.error("v12.1 fixes num_workers=1 for deterministic audit batches.")
    args.seeds = [DEV_SEED]
    return args


def result_path(cli):
    output = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "train_oof"
        / "seed1113_dev"
    )
    if output.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "v12.1 audit output exists; inspect it or use --overwrite: {}".format(
                    output
                )
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "DLF-mosi-post-surgery-oof-audit-v12p1-{}.log".format(
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_post_surgery_oof_audit_v12p1")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def frozen_roots(cli):
    v12_result = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_gradient_surgery_v12"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    v12_model = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cfcompat_gradient_surgery_v12"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    v10_result = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_conservative_crossfit_residual_v10"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    return v12_result, v12_model, v10_result


def load_v12_audit_inputs(cli):
    v12_result, v12_model, v10_result = frozen_roots(cli)
    manifest_path = v12_result / "gradient_surgery_v12_fold_manifest.csv"
    completion_path = v12_result / "gradient_surgery_v12_valid_screen_summary.json"
    assignment_path = v10_result / "conservative_crossfit_v10_fold_assignment.csv"
    for path in (manifest_path, completion_path, assignment_path):
        if not path.is_file():
            raise FileNotFoundError("Required frozen audit artifact missing: {}".format(path))

    manifest = pd.read_csv(manifest_path)
    assignment = pd.read_csv(assignment_path)
    if len(manifest) != N_FOLDS or set(manifest.Fold.astype(int)) != set(range(N_FOLDS)):
        raise RuntimeError("v12 fold manifest must contain exactly folds 0..4.")
    if len(assignment) != 1284 or assignment.sample_index.nunique() != 1284:
        raise RuntimeError("Frozen fold assignment must contain 1284 unique Train samples.")
    if assignment.groupby("video_id").fold.nunique().max() != 1:
        raise RuntimeError("Frozen assignment leaks a video across folds.")

    checkpoints = {}
    for row in manifest.itertuples(index=False):
        fold = int(row.Fold)
        checkpoint = (
            v12_model
            / "seed1113"
            / "fold{}".format(fold)
            / "conservative_train_holdout_bank.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError("Frozen v12 checkpoint missing: {}".format(checkpoint))
        actual_sha = checkpoint_sha256(checkpoint)
        recorded_sha = getattr(row, "ConservativeCheckpointSHA256", None)
        if recorded_sha is not None and str(recorded_sha) != str(actual_sha):
            raise RuntimeError("v12 fold {} checkpoint SHA mismatch.".format(fold))
        checkpoints[fold] = {
            "path": checkpoint,
            "sha256": actual_sha,
            "selected_epoch": int(row.ConservativeSelectedEpoch),
        }
    return manifest, assignment, checkpoints, completion_path


def main():
    cli = parse_args()
    output_root = result_path(cli)
    logger, log_path = create_logger(cli)
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v12.1 may construct only train/valid loaders; Test is forbidden.")

    v12_manifest, assignment, checkpoints, completion_path = load_v12_audit_inputs(cli)
    expected_assignment = v9.deterministic_video_group_folds(
        list(loaders["train"].dataset.ids), N_FOLDS
    )
    compare_columns = ["sample_index", "sample_id", "video_id", "fold"]
    left = assignment[compare_columns].sort_values("sample_index").reset_index(drop=True)
    right = expected_assignment[compare_columns].sort_values("sample_index").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("v12.1 reconstructed fold assignment differs from frozen v10/v12 folds.")

    v4._ACTIVE_TRAIN_BASELINE_FRAME = None
    try:
        teacher, s0_student, evaluator_bundle, assets = v4.load_assets(
            cli, args, loaders, DEV_SEED
        )
    finally:
        v4._ACTIVE_TRAIN_BASELINE_FRAME = None

    s0_sha = v9.module_state_sha256(s0_student)
    all_events = []
    influence_rows = []
    projection_rows = []
    fold_rows = []

    for fold in range(N_FOLDS):
        logger.info(
            "fold=%s loading frozen v12 conservative checkpoint epoch=%s audit=train_oof_post_surgery",
            fold,
            checkpoints[fold]["selected_epoch"],
        )
        model = v9.fresh_fold_student(s0_student, args.device)
        state = torch.load(checkpoints[fold]["path"], map_location="cpu")
        model.bank.load_state_dict(state, strict=True)
        model.to(args.device)
        model.eval()
        if v9.module_state_sha256(model.s0) != s0_sha:
            raise RuntimeError("Frozen S0 changed while loading v12 fold {}.".format(fold))

        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset,
            assignment,
            fold,
            args,
            cli.num_workers,
        )
        if set(train_indices).intersection(holdout_indices):
            raise RuntimeError("Train/OOF index leakage in fold {}.".format(fold))

        oof_frame, oof_gradients, oof_counts = v11audit.audit_oof_events_and_loss_gradients(
            fold,
            model,
            holdout_loader,
            teacher,
            evaluator_bundle,
            assets["cache_by_index"],
            args,
            assignment,
        )
        logger.info("fold=%s OOF event/gradient audit complete events=%s", fold, len(oof_frame))

        train_gradients = v11audit.audit_train_component_gradients(
            model,
            train_loader,
            teacher,
            evaluator_bundle,
            assets["cache_by_index"],
            args,
        )
        logger.info("fold=%s frozen-v12 Train objective gradients complete", fold)
        all_events.append(oof_frame)

        for mode in AUDIT_MODES:
            surgery = replay_asymmetric_projection(
                train_gradients[mode]["SUPERVISED_ALL"],
                train_gradients[mode]["SELECTIVE_ONLY"],
            )
            diagnostics = dict(surgery["diagnostics"])
            diagnostics.update({"Fold": int(fold), "Mode": str(mode)})
            projection_rows.append(diagnostics)

            for component in POST_SURGERY_COMPONENTS:
                train_gradient = surgery[component]
                for group in OOF_GROUPS:
                    count = int(oof_counts.get((mode, group), 0))
                    if count <= 0:
                        continue
                    influence_rows.append(
                        gradient_influence_row(
                            fold,
                            mode,
                            component,
                            group,
                            train_gradient,
                            oof_gradients[mode][group],
                            count,
                        )
                    )

        fold_rows.append(
            {
                "Fold": int(fold),
                "SelectedV12Epoch": int(checkpoints[fold]["selected_epoch"]),
                "CheckpointSHA256": checkpoints[fold]["sha256"],
                "TrainN": int(len(train_indices)),
                "OOFN": int(len(holdout_indices)),
                "TrainVideoN": int(
                    assignment.loc[
                        assignment.sample_index.isin(train_indices), "video_id"
                    ].nunique()
                ),
                "OOFVideoN": int(
                    assignment.loc[
                        assignment.sample_index.isin(holdout_indices), "video_id"
                    ].nunique()
                ),
            }
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    events = pd.concat(all_events, ignore_index=True)
    if len(events) != 1284 * len(MISSING_MODES):
        raise RuntimeError("Expected exactly 3852 Train-OOF missing-mode events.")
    if events[["sample_index", "mode"]].duplicated().any():
        raise RuntimeError("A Train sample/mode was audited by more than one fold.")
    events = enrich_oof_event_frame(events)
    event_summary_frame = event_summary(events)
    influence = pd.DataFrame(influence_rows)
    aggregate = aggregate_influence(influence)
    projection_frame = pd.DataFrame(projection_rows)
    fold_manifest = pd.DataFrame(fold_rows)
    findings = headline_findings(aggregate, projection_frame)

    events.to_csv(
        output_root / "post_surgery_audit_v12p1_train_oof_events.csv", index=False
    )
    event_summary_frame.to_csv(
        output_root / "post_surgery_audit_v12p1_event_summary.csv", index=False
    )
    influence.to_csv(
        output_root / "post_surgery_audit_v12p1_gradient_influence_by_fold.csv",
        index=False,
    )
    aggregate.to_csv(
        output_root / "post_surgery_audit_v12p1_gradient_influence_summary.csv",
        index=False,
    )
    projection_frame.to_csv(
        output_root / "post_surgery_audit_v12p1_projection_geometry.csv", index=False
    )
    fold_manifest.to_csv(
        output_root / "post_surgery_audit_v12p1_fold_manifest.csv", index=False
    )

    headline = aggregate.loc[
        aggregate.Mode.astype(str).eq("ALL")
        & aggregate.OOFGroup.astype(str).isin(
            [
                "OOF_TEACHER_BENEFICIAL",
                "OOF_TEACHER_NONBENEFICIAL",
                "OOF_S0_BENEFICIAL",
                "OOF_S0_NONBENEFICIAL",
            ]
        )
        & aggregate.TrainComponent.astype(str).isin(
            [
                "SUPERVISED_ALL_RAW",
                "SELECTIVE_ONLY",
                "SUPERVISED_PROJECTED",
                "SURGERY_UPDATE",
            ]
        )
    ].copy()
    headline.to_csv(
        output_root / "post_surgery_audit_v12p1_headline_gradient_matrix.csv",
        index=False,
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": "DIAGNOSTIC_ONLY_NO_MODEL_PROMOTION_DECISION",
        "headline_findings": jsonable(findings),
        "protocol": {
            "development_seed": DEV_SEED,
            "new_models_trained": 0,
            "checkpoint_source": "frozen_v12_conservative_gradient_surgery_banks",
            "v12_completion_artifact_required_but_not_used_for_audit_statistics": str(
                completion_path
            ),
            "train_oof_event_count": int(len(events)),
            "fold_count": N_FOLDS,
            "group_key": "video_id",
            "each_train_sample_audited_only_by_its_heldout_fold": True,
            "audit_views": "same_exhaustive_LA_LV_L_equal_weight_expected_moddrop_as_v11",
            "projection_replay": "analytic_v12_projection_on_frozen_checkpoint_expected_gradients",
            "historical_adam_trajectory_replayed": False,
            "official_valid_used_for_audit_statistics": False,
            "official_valid_reference_materialized_by_legacy_v4_asset_loader": True,
            "official_test_constructed": False,
            "official_test_accessed": False,
            "no_optimizer_steps": True,
            "no_checkpoint_selection": True,
        },
        "gradient_interpretation": {
            "positive_gradient_dot": "gradient descent on audited Train direction locally decreases OOF MAE",
            "negative_gradient_dot": "gradient descent on audited Train direction locally increases OOF MAE",
            "central_question": "after enforcing projected_supervised orthogonality to selective, does projected_supervised or surgery_update still harm OOF beneficial groups",
            "scope": "first_order_local_diagnostic_at_frozen_v12_conservative_checkpoint",
            "not_claimed": "not an exact replay of historical v12 Adam updates and not a Test-generalization claim",
        },
        "frozen_s0_state_sha256": s0_sha,
        "fold_manifest": jsonable(fold_manifest.to_dict("records")),
        "frozen_v12_manifest_rows": int(len(v12_manifest)),
    }
    summary_path = output_root / "post_surgery_audit_v12p1_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    logger.info(
        "complete events=%s projected_harm_teacher_beneficial_folds=%s update_harm_teacher_beneficial_folds=%s output=%s log=%s",
        len(events),
        findings["projected_supervised_harm_teacher_beneficial_folds"],
        findings["surgery_update_harm_teacher_beneficial_folds"],
        output_root,
        log_path,
    )
    print("CFCompatKD v12.1 post-surgery Train-OOF audit complete")
    print("new models trained: 0")
    print("Train-OOF missing events:", len(events))
    print(
        "projected supervised harm Teacher-beneficial folds:",
        findings["projected_supervised_harm_teacher_beneficial_folds"],
        "/ 5",
    )
    print(
        "surgery update harm Teacher-beneficial folds:",
        findings["surgery_update_harm_teacher_beneficial_folds"],
        "/ 5",
    )
    print(
        "projected supervised harm S0-beneficial folds:",
        findings["projected_supervised_harm_s0_beneficial_folds"],
        "/ 5",
    )
    print(
        "surgery update harm S0-beneficial folds:",
        findings["surgery_update_harm_s0_beneficial_folds"],
        "/ 5",
    )
    print("Official Test was not constructed or accessed")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
