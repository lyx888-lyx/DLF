"""MOSEI-native Valid-only port of the frozen MOSI CFCompatKD v13 expert.

This entrypoint preserves the v13 algorithmic mechanism while removing MOSI-only
historical-reference gates. It uses only frozen MOSEI upstream assets: clean DLF
seed1113, validation-best Stage1 ModDrop seed1113, and the formal train-only
MOSEI CFCompat compatibility cache for seed1113.

Five deterministic video-grouped Train folds train v8-compatible residual banks
using the v12 asymmetric supervised/selective gradient surgery plus the v13
actual-Adam-step two-halfspace functional-safety projection. Fold checkpoints
are selected only by held-out Train-video J. Official Valid is evaluated only
after all five banks are frozen. Official Test is never constructed or read.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from data_loader import MMDataLoader
import train_cfcompat_adam_step_safety_valid_screen_v13 as legacy_v13
import train_cfcompat_crossfit_residual_consensus_valid_screen_v9 as v9
import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_safe_projection_valid_screen as base
from train_mosei_cfcompat_v1 import (
    CACHE_VERSION as MOSEI_CFCOMPAT_CACHE_VERSION,
    EXPECTED_TRAIN_N,
    batch_to_device,
    build_config as build_mosei_config,
    prediction_rows,
    verify_clean_source,
    verify_stage1_source,
)
from trains.singleTask.cf_compat_kd_utils import (
    build_frozen_evaluator,
    evaluator_prediction,
    load_counterfactual_cache,
)
from trains.singleTask.cfcompat_adapter_isolation_utils import mechanism_transfer_summary
from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import (
    CONSENSUS_MIN_AGREE,
    N_FOLDS,
    FrozenS0CrossfitConsensus,
    consensus_summary,
    deterministic_video_group_folds,
    module_state_sha256,
)
from trains.singleTask.cfcompat_sample_residual_utils import MAX_ABS_RESIDUAL, RESIDUAL_HIDDEN_DIM
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence,
    build_frozen_teacher,
    checkpoint_sha256,
)
from trains.singleTask.missing_utils import MISSING_MODES, MissingModalityWrapper, evaluate_all_modes, validation_objective
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed

DATASET = "mosei"
DEV_SEED = 1113
EXPECTED_VALID_N = 1871
OUTPUT_TAG = "cfcompat_adam_step_safety_v13"
VERSION = "mosei_cfcompat_adam_step_safety_v13"
METHOD = "DLF-CFCompatKD-v13-MOSEI-FrozenPort"
RUN = "mosei_frozen_v13"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MOSEI frozen v13 Valid-only expert port")
    parser.add_argument("--seed", type=int, choices=(DEV_SEED,), default=DEV_SEED)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--matmul-precision", choices=("highest", "high", "medium"), default="high")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/missing_baseline/mosei_v13")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 0:
        parser.error("Windows MOSEI v13 fixes --num-workers=0.")
    args.dataset = DATASET
    args.gpu_ids = [int(args.gpu_id)]
    args.seeds = [DEV_SEED]
    args.max_epochs = 2 if args.smoke_test else None
    return args


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def output_paths(cli: argparse.Namespace) -> tuple[Path, Path]:
    output = Path(cli.result_root) / "missing_baseline" / OUTPUT_TAG / DATASET / "valid_screen" / "seed1113_dev"
    model = Path(cli.model_save_dir) / "missing_baseline" / OUTPUT_TAG / DATASET / "valid_screen" / "seed1113_dev"
    if cli.smoke_test:
        output = output / "smoke"
        model = model / "smoke"
    return output, model


def prepare_output(cli: argparse.Namespace) -> tuple[Path, Path]:
    output, model = output_paths(cli)
    if output.exists() or model.exists():
        if not cli.overwrite:
            raise FileExistsError(f"MOSEI v13 output exists; inspect it or pass --overwrite: {output} / {model}")
        if output.exists():
            shutil.rmtree(output)
        if model.exists():
            shutil.rmtree(model)
    output.mkdir(parents=True, exist_ok=True)
    model.mkdir(parents=True, exist_ok=True)
    return output, model


def create_logger(cli: argparse.Namespace) -> tuple[logging.Logger, Path]:
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = directory / f"DLF-mosei-v13-{kind}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
    logger = logging.getLogger("mosei_v13")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path, encoding="utf-8"), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def _loader_generator_state(loader):
    generator = getattr(loader, "generator", None)
    return generator, (generator.get_state().clone() if generator is not None else None)


def _restore_loader_generator(generator, state) -> None:
    if generator is not None and state is not None:
        generator.set_state(state)
        if not torch.equal(generator.get_state(), state):
            raise RuntimeError("DataLoader generator restoration failed.")


def train_baseline_prediction_rows(evaluator, loader, device) -> pd.DataFrame:
    evaluator.eval()
    rows = []
    generator, generator_state = _loader_generator_state(loader)
    try:
        with preserve_rng_state():
            for batch in loader:
                text, audio, vision, labels = batch_to_device(batch, device)
                baseline = {
                    mode: evaluator_prediction(evaluator, text, audio, vision, mode).view(-1)
                    for mode in MISSING_MODES
                }
                indices = batch["index"].view(-1).cpu().numpy().astype(int)
                identifiers = list(batch["id"])
                for offset, index in enumerate(indices):
                    row = {
                        "sample_index": int(index),
                        "sample_id": str(identifiers[offset]),
                        "label": float(labels[offset].item()),
                    }
                    for mode in MISSING_MODES:
                        row[f"baseline_{mode}_pred"] = float(baseline[mode][offset].detach().cpu())
                    rows.append(row)
    finally:
        _restore_loader_generator(generator, generator_state)
    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_TRAIN_N or frame.sample_index.nunique() != EXPECTED_TRAIN_N:
        raise RuntimeError(f"MOSEI v13 Train baseline must contain exactly {EXPECTED_TRAIN_N} unique samples.")
    if not np.array_equal(frame.sample_index.to_numpy(np.int64), np.arange(EXPECTED_TRAIN_N)):
        raise RuntimeError("MOSEI v13 Train baseline sample_index must be exactly 0..N-1.")
    numeric = ["label"] + [f"baseline_{mode}_pred" for mode in MISSING_MODES]
    if not np.isfinite(frame[numeric].to_numpy(np.float64)).all():
        raise FloatingPointError("MOSEI v13 Train baseline contains NaN/Inf.")
    return frame


def snapshot_valid_predictions(model, loader, device) -> pd.DataFrame:
    generator, state = _loader_generator_state(loader)
    try:
        with preserve_rng_state():
            frame = prediction_rows(model, loader, device)
    finally:
        _restore_loader_generator(generator, state)
    frame = frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_VALID_N or frame.sample_index.nunique() != EXPECTED_VALID_N:
        raise RuntimeError(f"MOSEI v13 Valid prediction snapshot expected {EXPECTED_VALID_N} rows, got {len(frame)}.")
    return frame


def build_s0_teacher(cli: argparse.Namespace, args, loaders):
    clean_checkpoint, clean_sha, clean_manifest = verify_clean_source(cli, args)
    teacher = build_frozen_teacher(DLF, args, clean_checkpoint)
    backbone = DLF(args).to(args.device)
    backbone.load_state_dict(torch.load(clean_checkpoint, map_location="cpu"), strict=True)
    s0_student = MissingModalityWrapper(backbone, args.feature_dims[1], args.feature_dims[2]).to(args.device)
    generator, state = _loader_generator_state(loaders["train"])
    try:
        with preserve_rng_state():
            first = next(iter(loaders["train"]))
            text, audio, vision, _ = batch_to_device(first, args.device)
            assert_initial_lav_equivalence(teacher, s0_student, text, audio, vision)
    finally:
        _restore_loader_generator(generator, state)
    return teacher, s0_student, backbone, clean_checkpoint, clean_sha, clean_manifest


def load_train_only_assets(cli: argparse.Namespace, args, loaders):
    clean = build_s0_teacher(cli, args, loaders)
    teacher, s0_student, backbone, clean_checkpoint, clean_sha, clean_manifest = clean
    evaluator_checkpoint, evaluator_best_epoch, evaluator_sha, stage1_manifest = verify_stage1_source(cli)
    cache_frame, cache_by_index = load_counterfactual_cache(
        cli.result_root,
        DATASET,
        version=MOSEI_CFCOMPAT_CACHE_VERSION,
        seed=DEV_SEED,
        expected_evaluator_sha=evaluator_sha,
    )
    if len(cache_frame) != EXPECTED_TRAIN_N or cache_frame.sample_index.nunique() != EXPECTED_TRAIN_N:
        raise RuntimeError("MOSEI v13 requires the frozen 16326-row formal compatibility cache.")
    if not np.array_equal(np.sort(cache_frame.sample_index.to_numpy(np.int64)), np.arange(EXPECTED_TRAIN_N)):
        raise RuntimeError("MOSEI v13 compatibility cache binding failed.")
    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    train_baseline = train_baseline_prediction_rows(evaluator, loaders["train"], args.device)
    evaluator_bundle = v4.FrozenRegretReference(pd.DataFrame(), train_baseline)
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assets = {
        "cache_by_index": cache_by_index,
        "cache_frame": cache_frame,
        "clean_checkpoint": clean_checkpoint,
        "clean_sha": clean_sha,
        "clean_manifest": clean_manifest,
        "stage1_checkpoint": evaluator_checkpoint,
        "stage1_sha": evaluator_sha,
        "stage1_best_epoch": int(evaluator_best_epoch),
        "stage1_manifest": stage1_manifest,
        "cache_version": MOSEI_CFCOMPAT_CACHE_VERSION,
    }
    return teacher, s0_student, backbone, evaluator_bundle, assets, train_baseline


def valid_reference_after_folds(cli, args, teacher, loaders, evaluator_checkpoint) -> pd.DataFrame:
    evaluator = build_frozen_evaluator(DLF, args, evaluator_checkpoint)
    generator, state = _loader_generator_state(loaders["valid"])
    try:
        with preserve_rng_state():
            frame = base.reference_prediction_rows(evaluator, teacher, loaders["valid"], args.device)
    finally:
        _restore_loader_generator(generator, state)
        del evaluator
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    frame = frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_VALID_N or frame.sample_index.nunique() != EXPECTED_VALID_N:
        raise RuntimeError("MOSEI v13 Valid reference binding failed.")
    return frame


def preflight(cli: argparse.Namespace) -> None:
    setup_seed(DEV_SEED)
    args = build_mosei_config(cli)
    if int(args.update_epochs) != 10:
        raise RuntimeError(f"Frozen v13 requires update_epochs=10, got {args.update_epochs}.")
    clean_checkpoint, clean_sha, _ = verify_clean_source(cli, args)
    stage1_checkpoint, stage1_epoch, stage1_sha, _ = verify_stage1_source(cli)
    cache_frame, _ = load_counterfactual_cache(
        cli.result_root,
        DATASET,
        version=MOSEI_CFCOMPAT_CACHE_VERSION,
        seed=DEV_SEED,
        expected_evaluator_sha=stage1_sha,
    )
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError(f"v13 preflight constructed unexpected splits: {sorted(loaders)}")
    if len(loaders["train"].dataset) != EXPECTED_TRAIN_N or len(loaders["valid"].dataset) != EXPECTED_VALID_N:
        raise RuntimeError("MOSEI Train/Valid count changed.")
    if len(cache_frame) != EXPECTED_TRAIN_N:
        raise RuntimeError("Formal MOSEI compatibility cache count changed.")
    raw5_root = Path(cli.result_root) / "missing_baseline" / "cfcompat_prediction_ensemble_v1" / DATASET
    raw5_members = [raw5_root / f"online_seed{seed}_valid_predictions.csv" for seed in (1111,1112,1113,1114,1115)]
    missing_raw5 = [str(path) for path in raw5_members if not path.is_file()]
    print("================ MOSEI v13 preflight ==============================")
    print("seed:", DEV_SEED)
    print("clean checkpoint:", clean_checkpoint)
    print("clean sha256:", clean_sha)
    print("stage1 evaluator:", stage1_checkpoint)
    print("stage1 sha256:", stage1_sha)
    print("stage1 best epoch:", stage1_epoch)
    print("compatibility cache version:", MOSEI_CFCOMPAT_CACHE_VERSION)
    print("compatibility cache rows:", len(cache_frame))
    print("Train rows:", len(loaders["train"].dataset))
    print("Valid rows:", len(loaders["valid"].dataset))
    print("Raw5 Valid members ready:", not missing_raw5)
    if missing_raw5:
        print("missing Raw5 members:", missing_raw5)
    print("folds:", N_FOLDS)
    print("consensus min agree:", CONSENSUS_MIN_AGREE)
    print("Official Test constructed:", False)
    print("STATUS: MOSEI_V13_UPSTREAM_READY")


def main_train(cli: argparse.Namespace) -> None:
    output_root, model_root = prepare_output(cli)
    logger, log_path = create_logger(cli)
    setup_seed(DEV_SEED)
    args = build_mosei_config(cli)
    if int(args.update_epochs) != 10:
        raise RuntimeError("Frozen v13 requires update_epochs=10.")
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("MOSEI v13 may construct only Train/Valid loaders; Test is forbidden.")
    if len(loaders["train"].dataset) != EXPECTED_TRAIN_N or len(loaders["valid"].dataset) != EXPECTED_VALID_N:
        raise RuntimeError("MOSEI split sizes changed from the frozen port audit.")

    teacher, s0_student, backbone, evaluator_bundle, assets, train_baseline = load_train_only_assets(cli, args, loaders)
    s0_sha_before = module_state_sha256(s0_student)
    assignment = deterministic_video_group_folds(list(loaders["train"].dataset.ids), N_FOLDS)
    assignment.to_csv(output_root / "adam_step_safety_v13_fold_assignment.csv", index=False)
    logger.info("start method=%s dataset=%s seed=%s train_N=%s valid_N=%s test_constructed=false", METHOD, DATASET, DEV_SEED, EXPECTED_TRAIN_N, EXPECTED_VALID_N)
    logger.info("clean=%s clean_sha=%s stage1=%s stage1_sha=%s stage1_best_epoch=%s cache_version=%s", assets["clean_checkpoint"], assets["clean_sha"], assets["stage1_checkpoint"], assets["stage1_sha"], assets["stage1_best_epoch"], assets["cache_version"])
    logger.info("config batch=%s update_epochs=%s lr=%s patience=%s early_stop=%s folds=%s consensus=%s smoke=%s", args.batch_size, args.update_epochs, args.learning_rate, args.patience, args.early_stop, N_FOLDS, CONSENSUS_MIN_AGREE, cli.smoke_test)

    bank_states, fold_epoch_frames, fold_decision_frames = [], [], []
    fold_surgery_frames, fold_safety_frames, sentinel_frames, fold_results = [], [], [], []
    for fold in range(N_FOLDS):
        train_loader, holdout_loader, train_indices, holdout_indices = v9.make_fold_loaders(
            loaders["train"].dataset, assignment, fold, args, cli.num_workers
        )
        safety_loader = legacy_v13.make_safety_loader(loaders["train"].dataset, train_indices, args.batch_size)
        state, epoch_frame, decisions, surgery_frame, safety_frame, sentinel_manifest, fold_result = legacy_v13.train_one_fold_v13(
            cli, logger, args, fold, train_loader, holdout_loader, safety_loader,
            teacher, s0_student, evaluator_bundle, assets, model_root, s0_sha_before,
        )
        bank_states.append(state)
        fold_epoch_frames.append(epoch_frame)
        fold_decision_frames.append(decisions)
        fold_surgery_frames.append(surgery_frame)
        fold_safety_frames.append(safety_frame)
        sentinel_frames.append(sentinel_manifest)
        fold_result["TrainN"] = int(len(train_indices))
        fold_result["HoldoutN"] = int(len(holdout_indices))
        fold_result["TrainVideoCount"] = int(assignment.loc[assignment.sample_index.isin(train_indices), "video_id"].nunique())
        fold_result["HoldoutVideoCount"] = int(assignment.loc[assignment.sample_index.isin(holdout_indices), "video_id"].nunique())
        fold_results.append(fold_result)

    if module_state_sha256(s0_student) != s0_sha_before:
        raise RuntimeError("Frozen S0 changed across MOSEI v13 fold training.")
    candidate = FrozenS0CrossfitConsensus(
        s0_student, bank_states, hidden_dim=RESIDUAL_HIDDEN_DIM,
        max_abs_residual=MAX_ABS_RESIDUAL, min_agree=CONSENSUS_MIN_AGREE,
    ).to(args.device)
    if any(parameter.requires_grad for parameter in candidate.parameters()):
        raise RuntimeError("Final MOSEI v13 consensus model must be inference-only.")

    criterion = nn.L1Loss()
    final_valid = evaluate_all_modes(candidate, loaders["valid"], args.device, "moddrop", criterion)
    candidate_j = float(validation_objective(final_valid))
    final_predictions = snapshot_valid_predictions(candidate, loaders["valid"], args.device)
    final_predictions["Seed"] = DEV_SEED
    final_predictions["Method"] = METHOD
    final_predictions["Split"] = "valid"
    final_predictions["SelectedBy"] = "frozen_mosi_v13_algorithm_train_crossfit_only"
    valid_reference = valid_reference_after_folds(cli, args, teacher, loaders, assets["stage1_checkpoint"])
    raw_events = base.raw_events_for_run(DEV_SEED, RUN, final_predictions, valid_reference)
    events = base.derive_valid_events(pd.DataFrame(raw_events))
    candidate_transfer = mechanism_transfer_summary(events, RUN)
    consensus_events = v9.consensus_diagnostic_rows(candidate, loaders["valid"], args.device)
    consensus_stats = consensus_summary(consensus_events)
    check = events.loc[events.Mode.astype(str).isin(MISSING_MODES), ["sample_index", "Mode", "candidate_prediction"]].merge(
        consensus_events[["sample_index", "Mode", "candidate_prediction"]],
        on=["sample_index", "Mode"], suffixes=("_official", "_diagnostic"), validate="one_to_one",
    )
    max_diff = float(np.max(np.abs(check.candidate_prediction_official.to_numpy(float) - check.candidate_prediction_diagnostic.to_numpy(float))))
    if max_diff > 1e-6:
        raise RuntimeError(f"MOSEI v13 consensus prediction path drifted: {max_diff}")

    fold_metrics = pd.concat(fold_epoch_frames, ignore_index=True)
    fold_decisions = pd.concat(fold_decision_frames, ignore_index=True)
    surgery_windows = pd.concat(fold_surgery_frames, ignore_index=True)
    step_safety_windows = pd.concat(fold_safety_frames, ignore_index=True)
    sentinel_manifest = pd.concat(sentinel_frames, ignore_index=True)
    fold_manifest = pd.DataFrame(fold_results)
    surgery_summary = legacy_v13.summarize_surgery_windows(surgery_windows.to_dict("records"))
    step_safety_summary = legacy_v13.summarize_step_safety(step_safety_windows)

    prediction_path = output_root / "v13_valid_predictions.csv"
    final_predictions.to_csv(prediction_path, index=False)
    pd.DataFrame(raw_events).to_csv(output_root / "adam_step_safety_v13_candidate_raw_valid_events.csv", index=False)
    events.to_csv(output_root / "adam_step_safety_v13_candidate_valid_events.csv", index=False)
    consensus_events.to_csv(output_root / "adam_step_safety_v13_valid_consensus_events.csv", index=False)
    train_baseline.to_csv(output_root / "adam_step_safety_v13_train_baseline_cache.csv", index=False)
    fold_manifest.to_csv(output_root / "adam_step_safety_v13_fold_manifest.csv", index=False)
    fold_metrics.to_csv(output_root / "adam_step_safety_v13_fold_epoch_metrics.csv", index=False)
    fold_decisions.to_csv(output_root / "adam_step_safety_v13_fold_train_decisions.csv", index=False)
    surgery_windows.to_csv(output_root / "adam_step_safety_v13_raw_surgery_windows.csv", index=False)
    step_safety_windows.to_csv(output_root / "adam_step_safety_v13_actual_step_safety_windows.csv", index=False)
    sentinel_manifest.to_csv(output_root / "adam_step_safety_v13_sentinel_manifest.csv", index=False)

    consensus_checkpoint = model_root / "frozen_consensus_valid_ready.pth"
    torch.save(candidate.state_dict(), consensus_checkpoint)
    consensus_sha = checkpoint_sha256(consensus_checkpoint)
    grid_row: Dict[str, Any] = {
        "Seed": DEV_SEED, "Run": RUN, "Method": METHOD, "J_valid": candidate_j,
        "S0StateSHA256": s0_sha_before,
        "S0StateUnchanged": bool(module_state_sha256(s0_student) == s0_sha_before),
        "CrossfitFolds": N_FOLDS, "ConsensusMinAgree": CONSENSUS_MIN_AGREE,
        "NearOptimalRelativeTolerance": legacy_v13.NEAR_OPTIMAL_REL_TOL,
        "ResidualHiddenDim": RESIDUAL_HIDDEN_DIM, "MaxAbsResidual": MAX_ABS_RESIDUAL,
        "RawGradientSurgery": "v12_exact_asymmetric_projection",
        "ActualStepSafety": "per_mode_two_halfspace_projection_teacher_and_s0_beneficial",
        "OfficialValidUsedForFoldSelection": False, "TestConstructed": False,
    }
    for mode, metrics in final_valid.items():
        for key, value in metrics.items():
            grid_row[f"valid_{mode}_{key}"] = float(value)
    pd.DataFrame([grid_row]).to_csv(output_root / "adam_step_safety_v13_candidate_grid.csv", index=False)

    verdict = "SMOKE_ONLY" if cli.smoke_test else "FROZEN_MOSI_V13_PORT_READY_FOR_FIXED_0P5_BLEND"
    summary = {
        "version": VERSION,
        "method": METHOD,
        "dataset": DATASET,
        "seed": DEV_SEED,
        "verdict": verdict,
        "candidate_J_valid": candidate_j,
        "candidate_valid_metrics": final_valid,
        "candidate_transfer": candidate_transfer,
        "consensus_summary": consensus_stats,
        "raw_gradient_surgery_summary": surgery_summary,
        "actual_step_safety_summary": step_safety_summary,
        "fold_manifest": fold_manifest.to_dict("records"),
        "sentinel_manifest": sentinel_manifest.to_dict("records"),
        "valid_prediction_path_max_abs_difference": max_diff,
        "upstream": {
            "clean_checkpoint": str(assets["clean_checkpoint"]), "clean_sha256": assets["clean_sha"],
            "stage1_evaluator": str(assets["stage1_checkpoint"]), "stage1_sha256": assets["stage1_sha"],
            "stage1_best_epoch": assets["stage1_best_epoch"],
            "compatibility_cache_version": assets["cache_version"], "compatibility_cache_rows": EXPECTED_TRAIN_N,
        },
        "frozen_consensus_checkpoint": {"path": str(consensus_checkpoint), "sha256": consensus_sha},
        "valid_predictions": str(prediction_path),
        "protocol": {
            "source_method_frozen_on_mosi": True,
            "development_seed": DEV_SEED,
            "base_objective": "v4 supervised missing + distill + preserve separated by v12 gradient surgery",
            "raw_gradient_surgery": "v12 asymmetric projection of supervised gradient against selective gradient",
            "actual_step_safety": "v13 per-mode Teacher-beneficial and S0-beneficial first-order MAE halfspaces",
            "sentinel_source": "current_fold_Train_4of5_videos_only",
            "teacher_beneficial_margin": float(legacy_v13.DISTILL_MARGIN),
            "s0_beneficial_margin": float(legacy_v13.S0_PROTECT_MARGIN),
            "fold_assignment": "deterministic video-grouped 5-fold",
            "fold_checkpoint_selector": "earliest epoch within frozen 1% of absolute-best Train-video-holdout J",
            "consensus_rule": f"{CONSENSUS_MIN_AGREE}-of-{N_FOLDS} same-sign then median residual else zero",
            "official_valid_used_for_fold_training": False,
            "official_valid_used_for_fold_checkpoint_selection": False,
            "official_valid_first_evaluated_after_all_fold_models_frozen": True,
            "official_test_constructed": False,
            "official_test_accessed": False,
            "mosei_blend_weight_search_allowed": False,
            "downstream_blend": "fixed Raw5 0.5 + v13 0.5 from MOSI",
        },
        "environment": {
            "python_platform": platform.platform(), "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device),
            "matmul_precision": cli.matmul_precision, "num_workers": cli.num_workers,
            "git_branch": git_value("branch", "--show-current"), "git_commit": git_value("rev-parse", "HEAD"),
            "completed_utc": utc_now(), "log": str(log_path),
        },
    }
    summary_path = output_root / "adam_step_safety_v13_mosei_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    logger.info("complete seed=%s J_valid=%.6f consensus_apply=%.4f projected_windows=%.4f output=%s", DEV_SEED, candidate_j, consensus_stats["consensus_applied_rate"], step_safety_summary["projected_window_fraction"], output_root)
    print("================ MOSEI frozen v13 complete ========================")
    print("seed:", DEV_SEED)
    print("candidate J_valid:", f"{candidate_j:.9f}")
    print("LAV MAE:", f"{final_valid['LAV']['MAE']:.9f}")
    missing_macro = float(np.mean([final_valid[m]["MAE"] for m in MISSING_MODES]))
    print("MissingMacro MAE:", f"{missing_macro:.9f}")
    print("consensus applied rate:", consensus_stats["consensus_applied_rate"])
    print("actual-step projected window fraction:", step_safety_summary["projected_window_fraction"])
    print("frozen consensus checkpoint:", consensus_checkpoint)
    print("checkpoint sha256:", consensus_sha)
    print("v13 Valid predictions:", prediction_path)
    print("TEST CONSTRUCTED:", False)
    print("summary:", summary_path)


def main() -> None:
    cli = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for MOSEI v13.")
    torch.cuda.set_device(int(cli.gpu_id))
    torch.set_float32_matmul_precision(cli.matmul_precision)
    torch.backends.cudnn.allow_tf32 = cli.matmul_precision != "highest"
    if cli.preflight_only:
        preflight(cli)
    else:
        main_train(cli)


if __name__ == "__main__":
    main()
