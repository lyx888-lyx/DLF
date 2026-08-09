"""Exploratory aggregate-only MOSI Test viability probe for CFCompatKD through v13.

IMPORTANT: this is NOT a pristine final Test evaluation.  MOSI Test was already
accessed historically for the Original-CFCompat-vs-v4 viability probe.  This
script performs a second, explicitly exploratory access only to decide whether
the current mechanism line has enough cross-split signal to justify more work.

Frozen methods compared in one invocation:
  * Original CFCompatKD v1 (the historical first innovation; validation-best),
  * Regret-Preserve v4,
  * Sample-conditioned residual v8,
  * Gradient-surgery v12,
  * Adam-step functional-safety v13.

No training, checkpoint selection, gate fitting, calibration, or Test-driven
model modification occurs.  No sample-level Test predictions/events/IDs are
written to disk.  Only aggregate metrics, transfer summaries, checkpoint
bindings, and the pre-frozen sign-only route decision are persisted.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import MMDataLoader
from train_cf_compat_kd import build_config
from trains.singleTask.cf_compat_kd_utils import build_frozen_evaluator, evaluator_prediction
from trains.singleTask.cfcompat_crossfit_residual_consensus_utils import (
    CONSENSUS_MIN_AGREE,
    FrozenS0CrossfitConsensus,
    module_state_sha256,
)
from trains.singleTask.cfcompat_exploratory_test_utils import (
    METHOD_ORDER,
    PRIMARY_REFERENCE,
    VERSION,
    add_reference_deltas,
    build_missing_events,
    comparison_row,
    jsonable,
    transfer_rows,
    transfer_summary,
    v13_route_decision,
)
from trains.singleTask.cfcompat_sample_residual_utils import (
    MAX_ABS_RESIDUAL,
    RESIDUAL_HIDDEN_DIM,
    FrozenS0SampleResidual,
)
from trains.singleTask.fixed_kd_utils import (
    build_frozen_teacher,
    checkpoint_sha256,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    mode_to_mask,
    regression_metrics,
    validation_objective,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


DEV_SEED = 1113
EXPECTED_TEST_N = 686
OUTPUT_TAG = "cfcompat_exploratory_test_viability_v13"
RUN_LABEL = "EXPLORATORY_TEST_VIABILITY_ONLY"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate-only exploratory MOSI Test viability probe through CFCompatKD v13"
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
        parser.error("Exploratory Test probe fixes num_workers=1.")
    return args


def output_path(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / OUTPUT_TAG
        / cli.dataset
        / "exploratory_test"
        / "seed1113"
    )
    if root.exists():
        if not cli.overwrite:
            raise FileExistsError(
                "Exploratory Test output already exists. Inspect it instead of rerunning, "
                "or explicitly use --overwrite if a technical rerun is consciously intended: {}".format(root)
            )
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def create_logger(cli):
    directory = Path(cli.log_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "DLF-mosi-exploratory-test-viability-v13-{}.log".format(
        datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("cfcompat_exploratory_test_viability_v13")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def _read_unique_grid(path: Path, seed: int = DEV_SEED) -> dict:
    if not path.is_file():
        raise FileNotFoundError("Required frozen grid missing: {}".format(path))
    frame = pd.read_csv(path)
    if "Seed" not in frame.columns:
        raise RuntimeError("Frozen grid lacks Seed: {}".format(path))
    local = frame.loc[frame.Seed.astype(int).eq(int(seed))]
    if len(local) != 1:
        raise RuntimeError("Frozen grid is not unique for Seed {}: {}".format(seed, path))
    return local.iloc[0].to_dict()


def _recorded_checkpoint(recorded, expected_sha=None, forbidden_tokens=("best_test", "diagnostic")) -> Path:
    path = Path(str(recorded))
    if not path.is_file():
        raise FileNotFoundError("Recorded frozen checkpoint is absent: {}".format(path))
    lower = str(path).lower()
    if any(token in lower for token in forbidden_tokens):
        raise RuntimeError("Test-selected/diagnostic checkpoint is forbidden: {}".format(path))
    actual = checkpoint_sha256(path)
    if expected_sha is not None and str(expected_sha) not in ("", "nan", "None"):
        if actual != str(expected_sha):
            raise RuntimeError(
                "Frozen checkpoint SHA mismatch {} != {} for {}".format(
                    actual, expected_sha, path
                )
            )
    return path


def load_original_source(cli) -> dict:
    path = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_student_safe_abstain_v2"
        / cli.dataset
        / "valid_screen"
        / "student_safe_abstain_source_manifest.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            "Original CFCompat frozen source manifest is required: {}".format(path)
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    records = [
        record
        for record in manifest.get("source_records", [])
        if int(record.get("seed", -1)) == DEV_SEED
        and str(record.get("run")) == "cfcompat_replay"
    ]
    if len(records) != 1:
        raise RuntimeError("Could not bind unique Seed1113 Original CFCompat replay source.")
    record = records[0]
    return {
        "source_manifest": path,
        "checkpoint": _recorded_checkpoint(
            record["checkpoint"], record.get("checkpoint_sha256")
        ),
        "checkpoint_expected_sha": str(record.get("checkpoint_sha256", "")),
        "teacher_checkpoint": _recorded_checkpoint(
            record["teacher_checkpoint"], record.get("teacher_sha256"), forbidden_tokens=()
        ),
        "teacher_expected_sha": str(record.get("teacher_sha256", "")),
        "evaluator_checkpoint": _recorded_checkpoint(
            record["evaluator_checkpoint"], record.get("evaluator_sha256"), forbidden_tokens=("best_test", "diagnostic")
        ),
        "evaluator_expected_sha": str(record.get("evaluator_sha256", "")),
    }


def load_frozen_paths(cli):
    original = load_original_source(cli)

    v4_grid_path = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_regret_preserve_v4"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
        / "regret_preserve_v4_candidate_grid.csv"
    )
    v4 = _read_unique_grid(v4_grid_path)
    v4_checkpoint = _recorded_checkpoint(
        v4["MainCheckpoint"], v4.get("MainCheckpointSHA256")
    )

    v8_grid_path = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_sample_conditioned_residual_v8"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
        / "sample_residual_v8_candidate_grid.csv"
    )
    v8 = _read_unique_grid(v8_grid_path)
    v8_checkpoint = _recorded_checkpoint(
        v8["MainCheckpoint"], v8.get("MainCheckpointSHA256")
    )

    bank_bindings = {}
    for method, tag, manifest_name in (
        (
            "gradient_surgery_v12",
            "cfcompat_gradient_surgery_v12",
            "gradient_surgery_v12_fold_manifest.csv",
        ),
        (
            "adam_step_safety_v13",
            "cfcompat_adam_step_safety_v13",
            "adam_step_safety_v13_fold_manifest.csv",
        ),
    ):
        manifest_path = (
            Path(cli.result_root)
            / "missing_baseline"
            / tag
            / cli.dataset
            / "valid_screen"
            / "seed1113_dev"
            / manifest_name
        )
        if not manifest_path.is_file():
            raise FileNotFoundError("Frozen fold manifest missing: {}".format(manifest_path))
        manifest = pd.read_csv(manifest_path).sort_values("Fold", kind="mergesort")
        if len(manifest) != 5 or list(manifest.Fold.astype(int)) != list(range(5)):
            raise RuntimeError("{} fold manifest must contain folds 0..4.".format(method))
        checkpoints = []
        for row in manifest.itertuples(index=False):
            expected = getattr(row, "ConservativeCheckpointSHA256", None)
            checkpoints.append(
                _recorded_checkpoint(
                    getattr(row, "ConservativeCheckpoint"), expected
                )
            )
        bank_bindings[method] = {
            "manifest": manifest_path,
            "checkpoints": checkpoints,
        }

    v13_summary_path = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_adam_step_safety_v13"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
        / "adam_step_safety_v13_valid_screen_summary.json"
    )
    if not v13_summary_path.is_file():
        raise FileNotFoundError("v13 frozen summary missing: {}".format(v13_summary_path))
    v13_summary = json.loads(v13_summary_path.read_text(encoding="utf-8"))
    if bool(v13_summary.get("protocol", {}).get("official_test_accessed", True)):
        raise RuntimeError("Frozen v13 summary does not certify pre-probe Test isolation.")

    return {
        "original_cfcompat_v1": original,
        "regret_preserve_v4": {
            "grid": v4_grid_path,
            "checkpoint": v4_checkpoint,
        },
        "sample_residual_v8": {
            "grid": v8_grid_path,
            "checkpoint": v8_checkpoint,
        },
        "gradient_surgery_v12": bank_bindings["gradient_surgery_v12"],
        "adam_step_safety_v13": bank_bindings["adam_step_safety_v13"],
        "v13_summary_path": v13_summary_path,
        "v13_summary": v13_summary,
    }


def fresh_missing_student(args):
    backbone = DLF(args).to(args.device)
    return MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)


def load_missing_student(args, checkpoint: Path):
    student = fresh_missing_student(args)
    student.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    student.eval()
    return student


def load_v8_model(args, checkpoint: Path):
    model = FrozenS0SampleResidual(
        fresh_missing_student(args),
        hidden_dim=RESIDUAL_HIDDEN_DIM,
        max_abs_residual=MAX_ABS_RESIDUAL,
    ).to(args.device)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device), strict=True)
    model.eval()
    return model


def load_consensus_model(args, s0_state, checkpoint_paths):
    s0 = fresh_missing_student(args)
    s0.load_state_dict(copy.deepcopy(s0_state), strict=True)
    banks = [
        torch.load(path, map_location="cpu")
        for path in checkpoint_paths
    ]
    model = FrozenS0CrossfitConsensus(
        s0,
        banks,
        hidden_dim=RESIDUAL_HIDDEN_DIM,
        max_abs_residual=MAX_ABS_RESIDUAL,
        min_agree=CONSENSUS_MIN_AGREE,
    ).to(args.device)
    model.eval()
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise RuntimeError("Exploratory Test consensus model unexpectedly trainable.")
    return model


def collect_reference(evaluator, teacher, loader, args):
    evaluator.eval()
    teacher.eval()
    rows = []
    for batch in loader:
        text = batch["text"].to(args.device)
        audio = batch["audio"].to(args.device)
        vision = batch["vision"].to(args.device)
        labels = batch["labels"]["M"].view(-1).cpu().numpy().astype(np.float64)
        indices = batch["index"].view(-1).cpu().numpy().astype(np.int64)
        teacher_pred = (
            teacher_lav_prediction(teacher, text, audio, vision)
            .view(-1)
            .detach()
            .cpu()
            .numpy()
        )
        baseline = {
            mode: (
                evaluator_prediction(evaluator, text, audio, vision, mode)
                .view(-1)
                .detach()
                .cpu()
                .numpy()
            )
            for mode in ("LAV",) + MISSING_MODES
        }
        for offset, index in enumerate(indices):
            rows.append(
                {
                    "sample_index": int(index),
                    "label": float(labels[offset]),
                    "teacher_prediction": float(teacher_pred[offset]),
                    **{
                        "baseline_{}_pred".format(mode): float(baseline[mode][offset])
                        for mode in ("LAV",) + MISSING_MODES
                    },
                }
            )
    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_TEST_N or frame.sample_index.nunique() != EXPECTED_TEST_N:
        raise RuntimeError(
            "MOSI Test reference expected {} unique samples, got {} / {}".format(
                EXPECTED_TEST_N, len(frame), frame.sample_index.nunique()
            )
        )
    return frame


def collect_model_once(model, loader, args):
    """One Test-loader pass for one frozen model; returns only in-memory rows/metrics."""
    model.eval()
    collected = {
        mode: {"pred": [], "label": []}
        for mode in ("LAV",) + MISSING_MODES
    }
    rows = []
    with torch.no_grad():
        for batch in loader:
            text = batch["text"].to(args.device)
            audio = batch["audio"].to(args.device)
            vision = batch["vision"].to(args.device)
            labels_device = batch["labels"]["M"].to(args.device).view(-1, 1)
            labels_cpu = labels_device.detach().cpu()
            indices = batch["index"].view(-1).cpu().numpy().astype(np.int64)
            predictions = {}
            for mode in ("LAV",) + MISSING_MODES:
                mask = mode_to_mask(
                    mode,
                    batch_size=labels_device.size(0),
                    device=args.device,
                    dtype=audio.dtype,
                )
                prediction = model(text, audio, vision, mask)["output_logit"].view(-1, 1)
                normal = prediction.detach().cpu()
                predictions[mode] = normal.view(-1).numpy()
                collected[mode]["pred"].append(normal)
                collected[mode]["label"].append(labels_cpu)
            label_values = labels_cpu.view(-1).numpy()
            for offset, index in enumerate(indices):
                rows.append(
                    {
                        "sample_index": int(index),
                        "label": float(label_values[offset]),
                        **{
                            "{}_pred".format(mode): float(predictions[mode][offset])
                            for mode in ("LAV",) + MISSING_MODES
                        },
                    }
                )
    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != EXPECTED_TEST_N or frame.sample_index.nunique() != EXPECTED_TEST_N:
        raise RuntimeError("Frozen model Test pass lost or duplicated samples.")
    metrics_by_mode = {}
    for mode in ("LAV",) + MISSING_MODES:
        predictions = torch.cat(collected[mode]["pred"], dim=0)
        labels = torch.cat(collected[mode]["label"], dim=0)
        metrics_by_mode[mode] = regression_metrics(predictions, labels)
    metrics = {
        "TestJ": float(validation_objective(metrics_by_mode)),
        "MissingMacroMAE": float(
            np.mean([metrics_by_mode[mode]["MAE"] for mode in MISSING_MODES])
        ),
        **{
            "{}_MAE".format(mode): float(metrics_by_mode[mode]["MAE"])
            for mode in ("LAV",) + MISSING_MODES
        },
        **{
            "{}_Corr".format(mode): float(metrics_by_mode[mode]["Corr"])
            for mode in ("LAV",) + MISSING_MODES
        },
        **{
            "{}_acc_2".format(mode): float(metrics_by_mode[mode]["acc_2"])
            for mode in ("LAV",) + MISSING_MODES
        },
        **{
            "{}_F1".format(mode): float(metrics_by_mode[mode]["F1_score"])
            for mode in ("LAV",) + MISSING_MODES
        },
    }
    return frame, metrics


def baseline_metrics(reference: pd.DataFrame):
    metrics_by_mode = {}
    labels = torch.as_tensor(reference.label.to_numpy(np.float32)).view(-1, 1)
    for mode in ("LAV",) + MISSING_MODES:
        pred = torch.as_tensor(
            reference["baseline_{}_pred".format(mode)].to_numpy(np.float32)
        ).view(-1, 1)
        metrics_by_mode[mode] = regression_metrics(pred, labels)
    return {
        "TestJ": float(validation_objective(metrics_by_mode)),
        "MissingMacroMAE": float(
            np.mean([metrics_by_mode[mode]["MAE"] for mode in MISSING_MODES])
        ),
        **{
            "{}_MAE".format(mode): float(metrics_by_mode[mode]["MAE"])
            for mode in ("LAV",) + MISSING_MODES
        },
    }


def checkpoint_manifest(bindings):
    result = {}
    for method in METHOD_ORDER:
        binding = bindings[method]
        if "checkpoint" in binding:
            path = Path(binding["checkpoint"])
            result[method] = {
                "checkpoint": str(path.resolve()),
                "checkpoint_sha256": checkpoint_sha256(path),
                "selection": "validation_best_frozen_before_this_probe",
            }
        else:
            result[method] = {
                "fold_checkpoints": [
                    {
                        "fold": int(fold),
                        "checkpoint": str(Path(path).resolve()),
                        "checkpoint_sha256": checkpoint_sha256(path),
                    }
                    for fold, path in enumerate(binding["checkpoints"])
                ],
                "selection": "Train_video_holdout_conservative_frozen_before_this_probe",
                "consensus": "4_of_5_same_sign_then_median",
            }
    result["reference_teacher"] = {
        "checkpoint": str(bindings[PRIMARY_REFERENCE]["teacher_checkpoint"].resolve()),
        "checkpoint_sha256": checkpoint_sha256(
            bindings[PRIMARY_REFERENCE]["teacher_checkpoint"]
        ),
    }
    result["reference_moddrop"] = {
        "checkpoint": str(bindings[PRIMARY_REFERENCE]["evaluator_checkpoint"].resolve()),
        "checkpoint_sha256": checkpoint_sha256(
            bindings[PRIMARY_REFERENCE]["evaluator_checkpoint"]
        ),
        "selection": "validation_best_frozen_before_this_probe",
    }
    return result


def main():
    cli = parse_args()
    output_root = output_path(cli)
    logger, log_path = create_logger(cli)

    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    # Strictly construct Test only.  No Train/Valid loader exists in this process.
    args.mode = "test"
    args.is_training = False
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"test"}:
        raise RuntimeError("Exploratory probe must construct exactly one Test split loader.")
    test_loader = loaders["test"]
    if len(test_loader.dataset) != EXPECTED_TEST_N:
        raise RuntimeError(
            "Expected MOSI Test N={}, got {}".format(
                EXPECTED_TEST_N, len(test_loader.dataset)
            )
        )

    bindings = load_frozen_paths(cli)
    manifest = checkpoint_manifest(bindings)
    manifest_path = output_root / "exploratory_test_viability_v13_checkpoint_manifest.json"
    manifest_path.write_text(
        json.dumps(jsonable(manifest), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    logger.warning("EXPLORATORY_TEST_VIABILITY_ONLY")
    logger.warning("TEST_ALREADY_HISTORICALLY_ACCESSED")
    logger.warning("NO_TEST_DRIVEN_TUNING_ALLOWED")
    logger.info("frozen methods=%s TestN=%s", list(METHOD_ORDER), EXPECTED_TEST_N)

    # Build the common frozen Teacher + validation-best ModDrop reference once.
    teacher = build_frozen_teacher(
        DLF, args, bindings[PRIMARY_REFERENCE]["teacher_checkpoint"]
    )
    evaluator = build_frozen_evaluator(
        DLF, args, bindings[PRIMARY_REFERENCE]["evaluator_checkpoint"]
    )
    reference = collect_reference(evaluator, teacher, test_loader, args)
    baseline = baseline_metrics(reference)
    del teacher, evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    comparison_rows = []
    all_transfer_rows = []
    transfer_by_method = {}
    metrics_by_method = {}

    # Original CFCompat v1.
    model = load_missing_student(args, bindings["original_cfcompat_v1"]["checkpoint"])
    predictions, metrics = collect_model_once(model, test_loader, args)
    events = build_missing_events(predictions, reference, "original_cfcompat_v1")
    transfer = transfer_summary(events)
    comparison_rows.append(comparison_row("original_cfcompat_v1", metrics, transfer))
    all_transfer_rows.extend(transfer_rows("original_cfcompat_v1", transfer))
    transfer_by_method["original_cfcompat_v1"] = transfer
    metrics_by_method["original_cfcompat_v1"] = metrics
    logger.info(
        "method=original_cfcompat_v1 TestJ=%.6f overallNTR=%.4f beneficialNTR=%.4f severeNTR=%.4f",
        metrics["TestJ"],
        transfer["all_missing"]["negative_transfer_rate"],
        transfer["teacher_beneficial"]["negative_transfer_rate"],
        transfer["all_missing"]["severe_negative_transfer_rate"],
    )
    del model, predictions, events
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # v4 Regret-Preserve.
    model = load_missing_student(args, bindings["regret_preserve_v4"]["checkpoint"])
    predictions, metrics = collect_model_once(model, test_loader, args)
    events = build_missing_events(predictions, reference, "regret_preserve_v4")
    transfer = transfer_summary(events)
    comparison_rows.append(comparison_row("regret_preserve_v4", metrics, transfer))
    all_transfer_rows.extend(transfer_rows("regret_preserve_v4", transfer))
    transfer_by_method["regret_preserve_v4"] = transfer
    metrics_by_method["regret_preserve_v4"] = metrics
    logger.info(
        "method=regret_preserve_v4 TestJ=%.6f overallNTR=%.4f beneficialNTR=%.4f severeNTR=%.4f",
        metrics["TestJ"], transfer["all_missing"]["negative_transfer_rate"],
        transfer["teacher_beneficial"]["negative_transfer_rate"],
        transfer["all_missing"]["severe_negative_transfer_rate"],
    )
    del model, predictions, events
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # v8 also supplies the exact frozen S0 state used by the later residual banks.
    v8_model = load_v8_model(args, bindings["sample_residual_v8"]["checkpoint"])
    expected_s0_sha = str(
        bindings["v13_summary"].get("parameter_isolation", {}).get(
            "s0_state_sha256_before", ""
        )
    )
    actual_s0_sha = module_state_sha256(v8_model.s0)
    if expected_s0_sha and actual_s0_sha != expected_s0_sha:
        raise RuntimeError(
            "v8-carried S0 hash differs from frozen v13 S0: {} != {}".format(
                actual_s0_sha, expected_s0_sha
            )
        )
    s0_state = {
        key: value.detach().cpu().clone()
        for key, value in v8_model.s0.state_dict().items()
    }
    predictions, metrics = collect_model_once(v8_model, test_loader, args)
    events = build_missing_events(predictions, reference, "sample_residual_v8")
    transfer = transfer_summary(events)
    comparison_rows.append(comparison_row("sample_residual_v8", metrics, transfer))
    all_transfer_rows.extend(transfer_rows("sample_residual_v8", transfer))
    transfer_by_method["sample_residual_v8"] = transfer
    metrics_by_method["sample_residual_v8"] = metrics
    logger.info(
        "method=sample_residual_v8 TestJ=%.6f overallNTR=%.4f beneficialNTR=%.4f severeNTR=%.4f",
        metrics["TestJ"], transfer["all_missing"]["negative_transfer_rate"],
        transfer["teacher_beneficial"]["negative_transfer_rate"],
        transfer["all_missing"]["severe_negative_transfer_rate"],
    )
    del v8_model, predictions, events
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Frozen 5-fold residual consensus methods.
    for method in ("gradient_surgery_v12", "adam_step_safety_v13"):
        model = load_consensus_model(
            args, s0_state, bindings[method]["checkpoints"]
        )
        predictions, metrics = collect_model_once(model, test_loader, args)
        events = build_missing_events(predictions, reference, method)
        transfer = transfer_summary(events)
        comparison_rows.append(comparison_row(method, metrics, transfer))
        all_transfer_rows.extend(transfer_rows(method, transfer))
        transfer_by_method[method] = transfer
        metrics_by_method[method] = metrics
        logger.info(
            "method=%s TestJ=%.6f overallNTR=%.4f beneficialNTR=%.4f severeNTR=%.4f",
            method,
            metrics["TestJ"],
            transfer["all_missing"]["negative_transfer_rate"],
            transfer["teacher_beneficial"]["negative_transfer_rate"],
            transfer["all_missing"]["severe_negative_transfer_rate"],
        )
        del model, predictions, events
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    comparison = add_reference_deltas(pd.DataFrame(comparison_rows))
    transfer_frame = pd.DataFrame(all_transfer_rows)
    route = v13_route_decision(comparison)

    # Extra descriptive v13-v12 deltas; these are not used for tuning/selection.
    indexed = comparison.set_index("Method")
    v12_row = indexed.loc["gradient_surgery_v12"]
    v13_row = indexed.loc["adam_step_safety_v13"]
    v13_vs_v12 = {
        "delta_J_v13_minus_v12": float(v13_row.TestJ) - float(v12_row.TestJ),
        "overall_NTR_reduction_v13_vs_v12": float(v12_row.OverallNTR)
        - float(v13_row.OverallNTR),
        "beneficial_NTR_reduction_v13_vs_v12": float(v12_row.BeneficialNTR)
        - float(v13_row.BeneficialNTR),
        "severe_NTR_reduction_v13_vs_v12": float(v12_row.OverallSevereNTR)
        - float(v13_row.OverallSevereNTR),
        "nonbeneficial_NTR_reduction_v13_vs_v12": float(v12_row.NonbeneficialNTR)
        - float(v13_row.NonbeneficialNTR),
    }

    comparison_path = output_root / "exploratory_test_viability_v13_comparison.csv"
    transfer_path = output_root / "exploratory_test_viability_v13_transfer.csv"
    comparison.to_csv(comparison_path, index=False)
    transfer_frame.to_csv(transfer_path, index=False)

    summary = {
        "version": VERSION,
        "run_label": RUN_LABEL,
        "route_decision": jsonable(route),
        "v13_vs_v12": jsonable(v13_vs_v12),
        "frozen_moddrop_baseline_test_metrics": jsonable(baseline),
        "method_test_metrics": jsonable(metrics_by_method),
        "method_transfer": jsonable(transfer_by_method),
        "protocol": {
            "development_seed": DEV_SEED,
            "exploratory_test_only": True,
            "test_already_historically_accessed": True,
            "this_is_not_final_unbiased_test": True,
            "no_training": True,
            "no_checkpoint_selection": True,
            "no_gate_fitting": True,
            "no_calibration": True,
            "no_test_driven_tuning_allowed": True,
            "sample_level_test_artifacts_written": False,
            "aggregate_artifacts_only": True,
            "test_loader_construction_count": 1,
            "frozen_method_order": list(METHOD_ORDER),
            "primary_reference": PRIMARY_REFERENCE,
            "route_rule": (
                "CLEAR_POSITIVE iff v13 strictly improves Original CFCompat TestJ and "
                "Teacher-beneficial NTR while overall NTR is not worse; "
                "CLEAR_NEGATIVE iff none of those three checks pass; otherwise MIXED"
            ),
            "future_method_changes_must_not_use_sample_level_test_information": True,
        },
        "checkpoint_manifest": str(manifest_path.resolve()),
        "comparison_csv": str(comparison_path.resolve()),
        "transfer_csv": str(transfer_path.resolve()),
    }
    summary_path = output_root / "exploratory_test_viability_v13_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("CFCompatKD exploratory Test viability probe complete")
    print("EXPLORATORY_TEST_VIABILITY_ONLY")
    print("TEST_ALREADY_HISTORICALLY_ACCESSED")
    print("NO_TEST_DRIVEN_TUNING_ALLOWED")
    print("Test sample count:", EXPECTED_TEST_N)
    print("sample-level Test artifacts written: False")
    print("route verdict:", route["verdict"])
    print("v13 delta J vs original:", route["delta_J_v13_minus_original"])
    print(
        "v13 beneficial NTR reduction vs original:",
        route["beneficial_NTR_reduction_v13_vs_original"],
    )
    print(
        "v13 overall NTR reduction vs original:",
        route["overall_NTR_reduction_v13_vs_original"],
    )
    print("summary:", summary_path)
    logger.info(
        "complete route=%s output=%s log=%s",
        route["verdict"], output_root, log_path
    )


if __name__ == "__main__":
    main()
