"""CFCompatKD v11: pure Train-OOF objective/gradient audit.

No new model is trained and no checkpoint is selected. The script loads the
five frozen v10 conservative residual banks. Each MOSI Train sample is audited
only by the fold for which its complete video was held out during residual
training. Official Test is never constructed or accessed. Official Valid is not
used in any audit statistic or decision; the legacy v4 frozen-asset loader
materializes its immutable Valid reference as an implementation prerequisite.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_loader import MMDataLoader
from train_cf_compat_kd import batch_to_device, build_config
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_crossfit_residual_consensus_valid_screen_v9 as v9
from trains.singleTask.cf_compat_kd_utils import compatibility_for_modes, gated_kd_loss
from trains.singleTask.cfcompat_objective_audit_utils import (
    AUDIT_MODES,
    BASE_TRAIN_COMPONENTS,
    DEV_SEED,
    METHOD,
    N_FOLDS,
    OOF_GROUPS,
    OUTPUT_TAG,
    TRAIN_COMPONENTS,
    VERSION,
    aggregate_influence,
    derive_train_component_vectors,
    enrich_oof_event_frame,
    event_summary,
    gradient_influence_row,
    gradient_vector,
    headline_findings,
    jsonable,
    zero_vector_like_parameters,
)
from trains.singleTask.cfcompat_regret_preserve_utils import (
    DISTILL_MARGIN,
    LAMBDA_PRESERVE,
    regret_preserve_decision,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256, teacher_lav_prediction
from trains.singleTask.missing_utils import MISSING_MODES, mode_to_mask
from utils.functions import setup_seed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train-OOF CFCompatKD objective/gradient audit v11"
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
        parser.error("v11 fixes num_workers=1 for deterministic audit batches.")
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
                "v11 audit output already exists; inspect it or use --overwrite: {}".format(
                    output
                )
            )
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    return output


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "DLF-mosi-objective-audit-v11-{}.log".format(
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_objective_audit_v11")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def v10_roots(cli):
    result_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_conservative_crossfit_residual_v10"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    model_root = (
        Path(cli.model_save_dir)
        / "missing_baseline"
        / "cfcompat_conservative_crossfit_residual_v10"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    return result_root, model_root


def load_v10_audit_manifest(cli):
    result_root, model_root = v10_roots(cli)
    manifest_path = result_root / "conservative_crossfit_v10_fold_manifest.csv"
    assignment_path = result_root / "conservative_crossfit_v10_fold_assignment.csv"
    summary_path = result_root / "conservative_crossfit_v10_valid_screen_summary.json"
    for path in (manifest_path, assignment_path, summary_path):
        if not path.is_file():
            raise FileNotFoundError("Required frozen v10 artifact missing: {}".format(path))
    manifest = pd.read_csv(manifest_path)
    assignment = pd.read_csv(assignment_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if set(manifest.Fold.astype(int)) != set(range(N_FOLDS)) or len(manifest) != N_FOLDS:
        raise RuntimeError("v10 fold manifest must contain exactly folds 0..4.")
    if len(assignment) != 1284 or assignment.sample_index.nunique() != 1284:
        raise RuntimeError("v10 fold assignment must contain 1284 unique Train samples.")
    if assignment.groupby("video_id").fold.nunique().max() != 1:
        raise RuntimeError("v10 assignment leaks a video across folds.")
    checkpoints = {}
    for row in manifest.itertuples(index=False):
        fold = int(row.Fold)
        checkpoint = (
            model_root
            / "seed1113"
            / "fold{}".format(fold)
            / "conservative_train_holdout_bank.pth"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError("Frozen v10 conservative checkpoint missing: {}".format(checkpoint))
        recorded_sha = getattr(row, "ConservativeCheckpointSHA256", None)
        actual_sha = checkpoint_sha256(checkpoint)
        if recorded_sha is not None and str(recorded_sha) != str(actual_sha):
            raise RuntimeError("v10 fold {} checkpoint SHA mismatch.".format(fold))
        checkpoints[fold] = {
            "path": checkpoint,
            "sha256": actual_sha,
            "selected_epoch": int(row.ConservativeSelectedEpoch),
        }
    return manifest, assignment, summary, checkpoints


def _record_vector(accumulator, counts, key, vector):
    if key not in accumulator:
        accumulator[key] = torch.zeros_like(vector)
        counts[key] = 0
    accumulator[key].add_(vector)
    counts[key] += 1


def _mean_vectors(accumulator, counts, component_names):
    result = {}
    for mode in AUDIT_MODES:
        base = {}
        for component in component_names:
            key = (mode, component)
            if key not in accumulator or counts[key] <= 0:
                raise RuntimeError("No gradient observations for {} / {}.".format(mode, component))
            base[component] = accumulator[key] / float(counts[key])
        result[mode] = derive_train_component_vectors(base)
    return result


def audit_train_component_gradients(
    model,
    train_loader,
    teacher,
    evaluator_bundle,
    cache_by_index,
    args,
):
    """Expected residual-head force using exhaustive LA/LV/L audit views."""
    model.eval()
    parameters = list(model.bank.parameters())
    if not parameters or any(not p.requires_grad for p in parameters):
        raise RuntimeError("Residual bank parameters are not audit-trainable tensors.")
    accumulator = {}
    counts = {}

    for batch in train_loader:
        text, audio, vision, labels = batch_to_device(batch, args.device)
        indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
        teacher_prediction = teacher_lav_prediction(
            teacher, text, audio, vision
        ).view(-1, 1)
        batch_size = int(labels.size(0))
        label_flat = labels.view(-1)

        for mode in MISSING_MODES:
            mode_list = [mode] * batch_size
            mask = mode_to_mask(mode, batch_size, args.device, audio.dtype)
            output = model(text, audio, vision, mask)
            prediction = output["output_logit"].view(-1)
            current_detached = prediction.detach().view(-1, 1)
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
                current_detached,
                teacher_prediction,
                baseline,
                labels,
                compatibility,
            )

            each_supervised = F.l1_loss(prediction, label_flat, reduction="none")
            distill_mask = decision["distill"].to(each_supervised)
            preserve_mask = decision["preserve"].to(each_supervised)
            abstain_mask = decision["abstain"].to(each_supervised)
            beneficial_mask = decision["teacher_beneficial"].to(each_supervised)
            nonbeneficial_mask = 1.0 - beneficial_mask
            denom = float(batch_size)

            supervised_distill = torch.sum(each_supervised * distill_mask) / denom
            supervised_preserve = torch.sum(each_supervised * preserve_mask) / denom
            supervised_abstain = torch.sum(each_supervised * abstain_mask) / denom
            supervised_beneficial = torch.sum(each_supervised * beneficial_mask) / denom
            supervised_nonbeneficial = torch.sum(each_supervised * nonbeneficial_mask) / denom

            kd_loss, _ = gated_kd_loss(
                prediction,
                decision["teacher_safe_target"],
                decision["distill_gate"],
            )
            each_preserve = F.smooth_l1_loss(
                prediction,
                decision["preserve_safe_target"].view(-1),
                reduction="none",
            )
            preserve_weight = decision["preserve_gate"].to(each_preserve)
            preserve_loss = torch.sum(preserve_weight * each_preserve) / (
                torch.sum(preserve_weight) + 1e-8
            )
            preserve_scaled = LAMBDA_PRESERVE * preserve_loss

            losses = [
                ("SUPERVISED_BRANCH_DISTILL", supervised_distill),
                ("SUPERVISED_BRANCH_PRESERVE", supervised_preserve),
                ("SUPERVISED_BRANCH_ABSTAIN", supervised_abstain),
                ("SUPERVISED_TEACHER_BENEFICIAL", supervised_beneficial),
                ("SUPERVISED_TEACHER_NONBENEFICIAL", supervised_nonbeneficial),
                ("DISTILL_KD", kd_loss),
                ("PRESERVE_SCALED", preserve_scaled),
            ]
            for offset, (component, loss) in enumerate(losses):
                vector = gradient_vector(
                    loss,
                    parameters,
                    retain_graph=(offset != len(losses) - 1),
                )
                _record_vector(accumulator, counts, (mode, component), vector)
                _record_vector(accumulator, counts, ("ALL", component), vector)

    return _mean_vectors(accumulator, counts, BASE_TRAIN_COMPONENTS)


def audit_oof_events_and_loss_gradients(
    fold,
    model,
    holdout_loader,
    teacher,
    evaluator_bundle,
    cache_by_index,
    args,
    assignment,
):
    """Build held-out Train events and group MAE gradients for one fold."""
    model.eval()
    parameters = list(model.bank.parameters())
    gradient_sums = {}
    group_counts = defaultdict(int)
    rows = []
    assignment_by_index = assignment.set_index("sample_index", drop=False)

    for batch in holdout_loader:
        text, audio, vision, labels = batch_to_device(batch, args.device)
        indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
        identifiers = list(batch["id"])
        teacher_prediction = teacher_lav_prediction(
            teacher, text, audio, vision
        ).view(-1, 1)
        batch_size = int(labels.size(0))
        label_flat = labels.view(-1)

        for mode in MISSING_MODES:
            mode_list = [mode] * batch_size
            mask = mode_to_mask(mode, batch_size, args.device, audio.dtype)
            output = model(text, audio, vision, mask)
            prediction = output["output_logit"].view(-1)
            s0_prediction = output["s0_output_logit"].view(-1).detach()
            residual_delta = output["residual_delta"].view(-1).detach()
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
                prediction.detach().view(-1, 1),
                teacher_prediction,
                baseline,
                labels,
                compatibility,
            )

            current_error_each = torch.abs(prediction - label_flat)
            teacher_beneficial = decision["teacher_beneficial"].view(-1)
            baseline_error = torch.abs(baseline.view(-1) - label_flat).detach()
            s0_error = torch.abs(s0_prediction - label_flat.detach())
            s0_beneficial = (baseline_error - s0_error) >= DISTILL_MARGIN
            group_masks = {
                "OOF_ALL": torch.ones_like(current_error_each, dtype=torch.bool),
                "OOF_TEACHER_BENEFICIAL": teacher_beneficial,
                "OOF_TEACHER_NONBENEFICIAL": ~teacher_beneficial,
                "OOF_S0_BENEFICIAL": s0_beneficial,
                "OOF_S0_NONBENEFICIAL": ~s0_beneficial,
            }
            active_groups = []
            for group in OOF_GROUPS:
                group_mask = group_masks[group].to(current_error_each)
                count = int(group_masks[group].sum().detach().cpu())
                if count <= 0:
                    continue
                loss_sum = torch.sum(current_error_each * group_mask)
                active_groups.append((group, count, loss_sum))
            for offset, (group, count, loss_sum) in enumerate(active_groups):
                vector = gradient_vector(
                    loss_sum,
                    parameters,
                    retain_graph=(offset != len(active_groups) - 1),
                )
                for audit_mode in (mode, "ALL"):
                    key = (audit_mode, group)
                    if key not in gradient_sums:
                        gradient_sums[key] = torch.zeros_like(vector)
                    gradient_sums[key].add_(vector)
                    group_counts[key] += count

            current_cpu = prediction.detach().cpu().numpy()
            s0_cpu = s0_prediction.cpu().numpy()
            delta_cpu = residual_delta.cpu().numpy()
            labels_cpu = label_flat.detach().cpu().numpy()
            baseline_cpu = baseline.view(-1).detach().cpu().numpy()
            teacher_cpu = teacher_prediction.view(-1).detach().cpu().numpy()
            teacher_target_cpu = decision["teacher_safe_target"].view(-1).cpu().numpy()
            preserve_target_cpu = decision["preserve_safe_target"].view(-1).cpu().numpy()
            for i, sample_index in enumerate(indices):
                assigned = assignment_by_index.loc[int(sample_index)]
                if int(assigned.fold) != int(fold):
                    raise RuntimeError("OOF sample is bound to the wrong fold.")
                rows.append(
                    {
                        "fold": int(fold),
                        "sample_index": int(sample_index),
                        "sample_id": str(identifiers[i]),
                        "video_id": str(assigned.video_id),
                        "mode": str(mode),
                        "label": float(labels_cpu[i]),
                        "baseline_prediction": float(baseline_cpu[i]),
                        "s0_prediction": float(s0_cpu[i]),
                        "student_prediction": float(current_cpu[i]),
                        "residual_delta": float(delta_cpu[i]),
                        "teacher_prediction": float(teacher_cpu[i]),
                        "teacher_safe_target": float(teacher_target_cpu[i]),
                        "preserve_safe_target": float(preserve_target_cpu[i]),
                        "distill": bool(decision["distill"][i].cpu()),
                        "preserve": bool(decision["preserve"][i].cpu()),
                        "decision_abstain": bool(decision["abstain"][i].cpu()),
                        "teacher_beneficial": bool(decision["teacher_beneficial"][i].cpu()),
                        "current_regressed": bool(decision["current_regressed"][i].cpu()),
                        "teacher_advantage_vs_baseline": float(
                            decision["teacher_advantage_vs_baseline"][i].cpu()
                        ),
                        "current_regret_vs_baseline": float(
                            decision["current_regret_vs_baseline"][i].cpu()
                        ),
                        "compatibility": float(compatibility[i].detach().cpu()),
                        "distill_gate": float(decision["distill_gate"][i].cpu()),
                        "preserve_gate": float(decision["preserve_gate"][i].cpu()),
                    }
                )

    oof_gradients = {}
    for mode in AUDIT_MODES:
        oof_gradients[mode] = {}
        for group in OOF_GROUPS:
            key = (mode, group)
            count = int(group_counts.get(key, 0))
            if count <= 0:
                oof_gradients[mode][group] = zero_vector_like_parameters(parameters)
            else:
                oof_gradients[mode][group] = gradient_sums[key] / float(count)
    return pd.DataFrame(rows), oof_gradients, dict(group_counts)


def main():
    cli = parse_args()
    output_root = result_path(cli)
    logger, log_path = create_logger(cli)
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v11 audit may construct only train/valid loaders; Test is forbidden.")

    v10_manifest, v10_assignment, v10_summary, checkpoints = load_v10_audit_manifest(cli)
    expected_assignment = v9.deterministic_video_group_folds(
        list(loaders["train"].dataset.ids), N_FOLDS
    )
    compare_columns = ["sample_index", "sample_id", "video_id", "fold"]
    left = v10_assignment[compare_columns].sort_values("sample_index").reset_index(drop=True)
    right = expected_assignment[compare_columns].sort_values("sample_index").reset_index(drop=True)
    if not left.equals(right):
        raise RuntimeError("v11 reconstructed fold assignment differs from frozen v10.")

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
    fold_rows = []

    for fold in range(N_FOLDS):
        logger.info(
            "fold=%s loading frozen v10 conservative checkpoint epoch=%s audit=train_oof_only",
            fold,
            checkpoints[fold]["selected_epoch"],
        )
        model = v9.fresh_fold_student(s0_student, args.device)
        state = torch.load(checkpoints[fold]["path"], map_location="cpu")
        model.bank.load_state_dict(state, strict=True)
        model.to(args.device)
        model.eval()
        if v9.module_state_sha256(model.s0) != s0_sha:
            raise RuntimeError("Frozen S0 changed while loading fold {}.".format(fold))

        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset,
            v10_assignment,
            fold,
            args,
            cli.num_workers,
        )
        if set(train_indices).intersection(holdout_indices):
            raise RuntimeError("Train/OOF index leakage in fold {}.".format(fold))

        oof_frame, oof_gradients, oof_counts = audit_oof_events_and_loss_gradients(
            fold,
            model,
            holdout_loader,
            teacher,
            evaluator_bundle,
            assets["cache_by_index"],
            args,
            v10_assignment,
        )
        logger.info("fold=%s OOF event/gradient audit complete events=%s", fold, len(oof_frame))
        train_gradients = audit_train_component_gradients(
            model,
            train_loader,
            teacher,
            evaluator_bundle,
            assets["cache_by_index"],
            args,
        )
        logger.info("fold=%s Train objective gradient decomposition complete", fold)
        all_events.append(oof_frame)

        for mode in AUDIT_MODES:
            for component in TRAIN_COMPONENTS:
                train_gradient = train_gradients[mode][component]
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
                "SelectedV10Epoch": int(checkpoints[fold]["selected_epoch"]),
                "CheckpointSHA256": checkpoints[fold]["sha256"],
                "TrainN": int(len(train_indices)),
                "OOFN": int(len(holdout_indices)),
                "TrainVideoN": int(
                    v10_assignment.loc[
                        v10_assignment.sample_index.isin(train_indices), "video_id"
                    ].nunique()
                ),
                "OOFVideoN": int(
                    v10_assignment.loc[
                        v10_assignment.sample_index.isin(holdout_indices), "video_id"
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
    findings = headline_findings(events, aggregate)
    fold_manifest = pd.DataFrame(fold_rows)

    abstain = events.loc[events.branch.astype(str).eq("ABSTAIN")]
    if len(abstain) and not bool(abstain.supervised_missing_active.astype(bool).all()):
        raise RuntimeError("ABSTAIN unexpectedly disabled supervised missing loss.")
    active_branch = events.loc[events.branch_target_active.astype(bool)]
    if len(active_branch) and float(active_branch.branch_target_locally_safe.mean()) < 1.0 - 1e-9:
        raise RuntimeError("A v4 safe branch target is not locally safe in OOF audit.")

    events.to_csv(output_root / "objective_audit_v11_train_oof_events.csv", index=False)
    event_summary_frame.to_csv(
        output_root / "objective_audit_v11_event_summary.csv", index=False
    )
    influence.to_csv(
        output_root / "objective_audit_v11_gradient_influence_by_fold.csv", index=False
    )
    aggregate.to_csv(
        output_root / "objective_audit_v11_gradient_influence_summary.csv", index=False
    )
    fold_manifest.to_csv(
        output_root / "objective_audit_v11_fold_manifest.csv", index=False
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": "DIAGNOSTIC_ONLY_NO_MODEL_PROMOTION_DECISION",
        "headline_findings": jsonable(findings),
        "protocol": {
            "development_seed": DEV_SEED,
            "new_models_trained": 0,
            "checkpoint_source": "frozen_v10_conservative_crossfit_residual_banks",
            "train_oof_event_count": int(len(events)),
            "fold_count": N_FOLDS,
            "group_key": "video_id",
            "each_train_sample_audited_only_by_its_heldout_fold": True,
            "audit_views": "exhaustive_LA_LV_L_equal_weight_expected_moddrop",
            "official_valid_used_for_audit_statistics": False,
            "official_valid_reference_materialized_by_legacy_v4_asset_loader": True,
            "official_test_constructed": False,
            "official_test_accessed": False,
            "no_optimizer_steps": True,
            "no_checkpoint_selection": True,
        },
        "objective_decomposition": {
            "residual_head_receives_full_LAV_gradient": False,
            "supervised_missing_active_on_every_missing_event": True,
            "distill_kd_active_only_on_DISTILL": True,
            "preserve_scaled_active_only_on_PRESERVE": True,
            "abstain_meaning": "no_KD_or_preserve_but_supervised_missing_remains_active",
            "lambda_preserve": LAMBDA_PRESERVE,
        },
        "gradient_interpretation": {
            "positive_gradient_dot": "gradient descent on Train component locally decreases OOF MAE",
            "negative_gradient_dot": "gradient descent on Train component locally increases OOF MAE",
            "scope": "first_order_local_diagnostic_at_frozen_v10_conservative_checkpoint",
            "not_claimed": "not an exact replay of Adam trajectory or causal proof by itself",
        },
        "frozen_v10_verdict": v10_summary.get("verdict"),
        "frozen_s0_state_sha256": s0_sha,
        "fold_manifest": jsonable(fold_manifest.to_dict("records")),
    }
    summary_path = output_root / "objective_audit_v11_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    headline_table = aggregate.loc[
        aggregate.Mode.astype(str).eq("ALL")
        & aggregate.OOFGroup.astype(str).isin(
            ["OOF_TEACHER_BENEFICIAL", "OOF_TEACHER_NONBENEFICIAL"]
        )
        & aggregate.TrainComponent.astype(str).isin(
            [
                "SUPERVISED_ALL",
                "SUPERVISED_BRANCH_ABSTAIN",
                "SUPERVISED_TEACHER_NONBENEFICIAL",
                "DISTILL_KD",
                "PRESERVE_SCALED",
                "SELECTIVE_ONLY",
                "TOTAL_RESIDUAL_OBJECTIVE",
            ]
        )
    ].copy()
    headline_table.to_csv(
        output_root / "objective_audit_v11_headline_gradient_matrix.csv", index=False
    )

    logger.info(
        "complete events=%s active_branch_safe=%.4f abstain_fraction=%.4f output=%s log=%s",
        len(events),
        findings["active_branch_target_locally_safe_fraction"],
        findings["abstain_event_fraction"],
        output_root,
        log_path,
    )
    print("CFCompatKD v11 Train-OOF objective audit complete")
    print("new models trained: 0")
    print("Train-OOF missing events:", len(events))
    print("active branch target locally safe fraction:", findings["active_branch_target_locally_safe_fraction"])
    print("ABSTAIN event fraction:", findings["abstain_event_fraction"])
    print("ABSTAIN still receives supervised missing loss: True")
    print("Official Test was not constructed or accessed")
    print("summary:", summary_path)


if __name__ == "__main__":
    main()
