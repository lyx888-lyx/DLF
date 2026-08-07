"""Cross-Fitted Transfer-Risk CFCompatKD v5, Seed1113 Valid-only development.

Exactly one new Student trajectory is permitted.  Before training, a cheap
frozen-model pre-pass builds label-free transfer-risk descriptors for each
Train sample x missing mode.  Train labels create the benefit target, but the
trajectory is routed only by grouped out-of-fold gate probabilities.

Official Test is never constructed by this script.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_safe_projection_valid_screen_v2 as hardened
import train_cfcompat_regret_preserve_valid_screen as v4
from data_loader import MMDataLoader
from train_cf_compat_kd import batch_to_device, build_config
from trains.singleTask.cf_compat_kd_utils import evaluator_prediction
from trains.singleTask.cfcompat_crossfit_transfer_risk_utils import (
    DEV_SEED,
    METHOD,
    OUTPUT_TAG,
    RUN,
    RUNS,
    VERSION,
    crossfit_projection_summary,
    dev_candidate_gate,
    fit_crossfit_transfer_risk_gate,
    frozen_thresholds,
    negative_transfer_summary,
    transfer_risk_kd_loss,
)
from trains.singleTask.cfcompat_safe_projection_utils import (
    derive_valid_events,
    group_summary,
    overall_from_events,
)
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.cfcompat_student_safe_abstain_utils import student_safe_project_teacher
from trains.singleTask.fixed_kd_utils import checkpoint_sha256, teacher_lav_prediction
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
)
from utils.functions import setup_seed


_ORIGINAL_LOAD_ASSETS = base.load_assets
_ORIGINAL_TRAIN_TRAJECTORY = base.train_trajectory
_PRECOMPUTED_GATE_FRAME = None
_PRECOMPUTED_GATE_SUMMARY = None
_PRECOMPUTED_GATE_PATHS = None
_ACTIVE_TRAIN_DECISIONS = None


class FrozenRiskReference:
    """CPU-only OOF Train risk cache plus frozen Valid reference predictions."""

    def __init__(self, valid_reference: pd.DataFrame, gate_frame: pd.DataFrame):
        self.valid_reference = valid_reference.copy()
        self.gate_frame = gate_frame.copy()
        self.train_by_key = {
            (int(row.sample_index), str(row.mode)): row._asdict()
            for row in self.gate_frame.itertuples(index=False)
        }
        if len(self.train_by_key) != 1284 * len(MISSING_MODES):
            raise RuntimeError("OOF risk cache does not contain 1284 x 3 unique Train events.")

    def parameters(self):
        return iter(())


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-fitted transfer-risk CFCompatKD v5 Valid-only Seed1113 screen.")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/windows_valid_only_prereq_v1")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.smoke_test and args.gate_only:
        parser.error("Choose at most one of --smoke-test / --gate-only.")
    if int(args.num_workers) != 1:
        parser.error("v5 fixes num_workers=1.")
    args.seeds = [DEV_SEED]
    args.max_epochs = 2 if args.smoke_test else None
    return args


def result_paths(cli):
    output = Path(cli.result_root) / "missing_baseline" / OUTPUT_TAG / cli.dataset / "valid_screen" / "seed1113_dev"
    model = Path(cli.model_save_dir) / "missing_baseline" / OUTPUT_TAG / cli.dataset / "valid_screen" / "seed1113_dev"
    if cli.gate_only:
        output, model = output / "gate_only", model / "gate_only"
    elif cli.smoke_test:
        output, model = output / "smoke", model / "smoke"
    if output.exists() or model.exists():
        if not cli.overwrite:
            raise FileExistsError("v5 output exists; inspect it or use --overwrite: {} / {}".format(output, model))
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
    kind = "gate" if cli.gate_only else ("smoke" if cli.smoke_test else "formal")
    path = directory / "DLF-mosi-crossfit-transfer-risk-v5-seed1113-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("crossfit_transfer_risk_valid_screen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def frozen_feature_rows(evaluator, teacher, student, loader, device, split):
    """Build label-free risk descriptors; restore Train loader generator exactly."""
    evaluator.eval()
    teacher.eval()
    student.eval()
    rows = []
    train_generator = getattr(loader, "generator", None)
    generator_state = train_generator.get_state().clone() if train_generator is not None else None
    try:
        with preserve_rng_state():
            with torch.inference_mode():
                for batch in loader:
                    text, audio, vision, labels = batch_to_device(batch, device)
                    teacher_full = teacher_lav_prediction(teacher, text, audio, vision).view(-1)
                    baseline_full = evaluator_prediction(evaluator, text, audio, vision, "LAV").view(-1)
                    baseline_missing = {}
                    initial_missing = {}
                    for mode in MISSING_MODES:
                        baseline_missing[mode] = evaluator_prediction(
                            evaluator, text, audio, vision, mode
                        ).view(-1)
                        mask = mode_to_mask(mode, labels.size(0), device, audio.dtype)
                        initial_missing[mode] = student(
                            text, audio, vision, mask
                        )["output_logit"].detach().view(-1)
                    indices = batch["index"].view(-1).cpu().numpy().astype(int)
                    identifiers = list(batch["id"])
                    for offset, index in enumerate(indices):
                        for mode in MISSING_MODES:
                            rows.append(
                                {
                                    "sample_index": int(index),
                                    "sample_id": str(identifiers[offset]),
                                    "mode": str(mode),
                                    "split": str(split),
                                    "label": float(labels[offset].detach().cpu()),
                                    "baseline_missing_prediction": float(baseline_missing[mode][offset].detach().cpu()),
                                    "baseline_full_prediction": float(baseline_full[offset].detach().cpu()),
                                    "teacher_full_prediction": float(teacher_full[offset].detach().cpu()),
                                    "initial_student_missing_prediction": float(initial_missing[mode][offset].detach().cpu()),
                                }
                            )
    finally:
        if generator_state is not None:
            train_generator.set_state(generator_state)
    if generator_state is not None and not torch.equal(train_generator.get_state(), generator_state):
        raise RuntimeError("Train DataLoader generator changed during v5 frozen feature pre-pass.")
    frame = pd.DataFrame(rows).sort_values(["sample_index", "mode"], kind="mergesort").reset_index(drop=True)
    if frame.empty or frame.duplicated(["sample_index", "mode"]).any():
        raise RuntimeError("Frozen risk feature frame is empty or duplicated.")
    return frame


def prepare_gate_assets(cli, output_root, logger):
    """Cheap cross-fit pre-screen performed before any new Student trajectory."""
    global _PRECOMPUTED_GATE_FRAME, _PRECOMPUTED_GATE_SUMMARY, _PRECOMPUTED_GATE_PATHS
    setup_seed(DEV_SEED)
    args = build_config(cli, DEV_SEED)
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("v5 gate pre-pass may construct only Train/Valid loaders.")
    teacher, student, evaluator, assets = _ORIGINAL_LOAD_ASSETS(cli, args, loaders, DEV_SEED)
    try:
        train_raw = frozen_feature_rows(evaluator, teacher, student, loaders["train"], args.device, "train")
        valid_raw = frozen_feature_rows(evaluator, teacher, student, loaders["valid"], args.device, "valid")
        train_gate, valid_gate, gate_summary = fit_crossfit_transfer_risk_gate(train_raw, valid_raw)
    finally:
        del teacher, student, evaluator, loaders
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    prefix = "crossfit_transfer_risk_v5"
    train_path = output_root / f"{prefix}_train_oof_gate.csv"
    valid_path = output_root / f"{prefix}_valid_gate_diagnostic.csv"
    summary_path = output_root / f"{prefix}_gate_summary.json"
    train_gate.to_csv(train_path, index=False)
    valid_gate.to_csv(valid_path, index=False)
    gate_summary["test_constructed"] = False
    gate_summary["train_labels_used_only_for_gate_supervision"] = True
    gate_summary["train_routing_probability_source"] = "grouped_out_of_fold_only"
    gate_summary["valid_gate_probability_source"] = "full_train_model_diagnostic_only"
    gate_summary["frozen_thresholds"] = frozen_thresholds()
    summary_path.write_text(json.dumps(gate_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    _PRECOMPUTED_GATE_FRAME = train_gate.copy()
    _PRECOMPUTED_GATE_SUMMARY = gate_summary
    _PRECOMPUTED_GATE_PATHS = {
        "train": train_path,
        "valid": valid_path,
        "summary": summary_path,
    }
    logger.info(
        "risk gate: OOF AUC=%.4f Valid AUC=%.4f Valid Brier=%.4f constant=%.4f prescreen=%s",
        gate_summary["train_oof"]["roc_auc"],
        gate_summary["valid_full_train_gate"]["roc_auc"],
        gate_summary["valid_full_train_gate"]["brier"],
        gate_summary["valid_full_train_gate"]["constant_brier"],
        gate_summary["prescreen_passed"],
    )
    return gate_summary


def load_assets(cli, args, loaders, seed):
    global _PRECOMPUTED_GATE_FRAME
    if int(seed) != DEV_SEED or _PRECOMPUTED_GATE_FRAME is None:
        raise RuntimeError("v5 requires the frozen Seed1113 OOF gate pre-pass before training.")
    teacher, student, evaluator, assets = _ORIGINAL_LOAD_ASSETS(cli, args, loaders, seed)
    with preserve_rng_state():
        valid_reference = hardened.reference_prediction_rows(
            evaluator, teacher, loaders["valid"], args.device
        )
    if len(valid_reference) != 229:
        raise RuntimeError("MOSI Valid reference must contain 229 samples.")
    bundle = FrozenRiskReference(valid_reference, _PRECOMPUTED_GATE_FRAME)
    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    assets["risk_gate_source"] = "grouped_5fold_out_of_fold_train_probability"
    assets["risk_gate_train_event_count"] = int(len(_PRECOMPUTED_GATE_FRAME))
    assets["risk_gate_test_constructed"] = False
    return teacher, student, bundle, assets


def risk_for_modes(bundle, indices, modes, labels, device, dtype):
    probabilities, baselines, folds = [], [], []
    for offset, (index, mode) in enumerate(zip(indices, modes)):
        key = (int(index), str(mode))
        if key not in bundle.train_by_key:
            raise KeyError("No OOF risk entry for {}.".format(key))
        record = bundle.train_by_key[key]
        if abs(float(record["label"]) - float(labels[offset].detach().cpu())) > 1e-6:
            raise RuntimeError("OOF risk label binding changed for sample {}.".format(index))
        probabilities.append(float(record["oof_benefit_probability"]))
        baselines.append(float(record["baseline_missing_prediction"]))
        folds.append(int(record["crossfit_fold"]))
    p = torch.as_tensor(probabilities, device=device, dtype=dtype).view(-1)
    b = torch.as_tensor(baselines, device=device, dtype=dtype).view(-1, 1)
    if not torch.isfinite(p).all() or torch.any(p <= 0.0) or torch.any(p >= 1.0):
        raise FloatingPointError("Bound OOF probabilities are invalid.")
    return p, b, folds


def forward_objective(
    run, batch, missing_mask, modes, args, teacher, evaluator_bundle, student,
    cache_by_index, criterion, cosine, hinge,
):
    del cache_by_index
    global _ACTIVE_TRAIN_DECISIONS
    if run != RUN or _ACTIVE_TRAIN_DECISIONS is None:
        raise RuntimeError("Invalid v5 objective state.")

    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = missing_mask.to(device=args.device, dtype=audio.dtype)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    current_student = missing_output["output_logit"].detach().view(-1, 1)
    teacher_prediction = teacher_lav_prediction(teacher, text, audio, vision).view(-1, 1)

    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    probability, baseline_prediction, folds = risk_for_modes(
        evaluator_bundle, indices, list(modes), labels, args.device, labels.dtype
    )
    teacher_safe, projection = student_safe_project_teacher(
        current_student, teacher_prediction, labels
    )
    active = projection["active"].view(-1)
    kd_loss, each_kd, effective_gate, eligible_mass, risk_weight = transfer_risk_kd_loss(
        missing_output["output_logit"], teacher_safe, active, probability
    )
    total_loss = full_loss + missing_loss + kd_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in cross-fitted transfer-risk objective.")

    identifiers = list(batch["id"])
    projection_records = []
    for offset in range(labels.size(0)):
        record = {
            key: (
                bool(value[offset].detach().cpu())
                if value.dtype == torch.bool
                else float(value[offset].detach().cpu())
            )
            for key, value in projection.items()
        }
        record.update(
            {
                "sample_index": int(indices[offset]),
                "sample_id": str(identifiers[offset]),
                "mode": str(modes[offset]),
                "label": float(labels[offset].detach().cpu()),
                "student_prediction": float(current_student[offset].detach().cpu()),
                "baseline_prediction": float(baseline_prediction[offset].detach().cpu()),
                "teacher_prediction": float(teacher_prediction[offset].detach().cpu()),
                "teacher_safe_target": float(teacher_safe[offset].detach().cpu()),
                "crossfit_fold": int(folds[offset]),
                "oof_benefit_probability": float(probability[offset].detach().cpu()),
                "risk_weight": float(risk_weight[offset].detach().cpu()),
                "safe_teacher_active": bool(active[offset].detach().cpu()),
                "eligible_distill_mass": float(eligible_mass[offset].detach().cpu()),
                "effective_gate": float(effective_gate[offset].detach().cpu()),
                "distill_loss_each": float(each_kd[offset].detach().cpu()),
            }
        )
        record["event_ordinal"] = len(_ACTIVE_TRAIN_DECISIONS) + 1
        projection_records.append(record)
        _ACTIVE_TRAIN_DECISIONS.append(dict(record))

    diagnostics = {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "mean_gate": float(effective_gate.mean().detach().cpu()),
        "weighted_kd": float(kd_loss.detach().cpu()),
        "baseline_missing_MAE": float(torch.abs(baseline_prediction.view(-1) - labels.view(-1)).mean().cpu()),
        "safe_target_MAE": float(torch.abs(teacher_safe.view(-1) - labels.view(-1)).mean().cpu()),
    }
    return total_loss, diagnostics, projection_records


def reference_prediction_rows(evaluator_bundle, teacher, loader, device):
    del teacher, loader, device
    return evaluator_bundle.valid_reference.copy()


def train_trajectory(cli, logger, output_root, model_root, seed, run):
    global _ACTIVE_TRAIN_DECISIONS
    if int(seed) != DEV_SEED or run != RUN:
        raise RuntimeError("v5 may train only Seed1113 / crossfit_transfer_risk_cfcompat.")
    _ACTIVE_TRAIN_DECISIONS = []
    try:
        result, epoch_rows, raw_events = _ORIGINAL_TRAIN_TRAJECTORY(
            cli, logger, output_root, model_root, seed, run
        )
        decisions = pd.DataFrame(_ACTIVE_TRAIN_DECISIONS)
    finally:
        _ACTIVE_TRAIN_DECISIONS = None
    expected = 1284 * int(result["TrainEpochCount"])
    if len(decisions) != expected:
        raise RuntimeError("v5 Train decision count mismatch: {} != {}.".format(len(decisions), expected))
    decisions["Epoch"] = ((decisions.event_ordinal.astype(int) - 1) // 1284) + 1
    run_dir = output_root / "seed{}".format(seed) / run
    decisions_path = run_dir / "train_crossfit_transfer_risk_decisions.csv"
    decisions.to_csv(decisions_path, index=False)
    result.update(
        {
            "Method": METHOD,
            "RiskGate": "grouped_5fold_out_of_fold_logistic",
            "RiskGateUsesTrainLabelAsFeature": False,
            "RiskGateTrainRoutingUsesOOFOnly": True,
            "RiskWeightTransform": "max(2*p-1,0)",
            "RiskKDNormalization": "sum(effective_gate*loss)/sum(current_student_safe_active)",
            "TrainDecisionRecords": str(decisions_path.resolve()),
            "TrainDecisionRecordsSHA256": checkpoint_sha256(decisions_path),
            "RiskGateTrainCache": str(_PRECOMPUTED_GATE_PATHS["train"].resolve()),
            "RiskGateTrainCacheSHA256": checkpoint_sha256(_PRECOMPUTED_GATE_PATHS["train"]),
            "RiskGateValidDiagnostic": str(_PRECOMPUTED_GATE_PATHS["valid"].resolve()),
            "RiskGateValidDiagnosticSHA256": checkpoint_sha256(_PRECOMPUTED_GATE_PATHS["valid"]),
            "RiskGateSummary": str(_PRECOMPUTED_GATE_PATHS["summary"].resolve()),
            "RiskGateSummarySHA256": checkpoint_sha256(_PRECOMPUTED_GATE_PATHS["summary"]),
        }
    )
    for row in epoch_rows:
        local = decisions.loc[decisions.Epoch.astype(int).eq(int(row["Epoch"]))]
        row["risk_positive_weight_fraction"] = float((local.risk_weight.to_numpy(float) > 0.0).mean())
        row["risk_effective_distill_fraction"] = float((local.effective_gate.to_numpy(float) > 0.0).mean())
        row["risk_mean_oof_probability"] = float(local.oof_benefit_probability.mean())
    pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    return result, epoch_rows, raw_events, decisions


def load_v4_reference(cli):
    root = Path(cli.result_root) / "missing_baseline" / "cfcompat_regret_preserve_v4" / cli.dataset / "valid_screen" / "seed1113_dev"
    paths = {
        "grid": root / "regret_preserve_v4_candidate_grid.csv",
        "raw": root / "regret_preserve_v4_candidate_raw_valid_events.csv",
        "summary": root / "regret_preserve_v4_valid_screen_summary.json",
        "manifest": root / "regret_preserve_v4_source_manifest.json",
        "audit": root / "regret_preserve_v4_audit_check.json",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing frozen v4 reference artifacts:\n" + "\n".join(missing))
    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if not audit.get("passed", False):
        raise RuntimeError("Frozen v4 audit did not pass.")
    grid = pd.read_csv(paths["grid"])
    raw = pd.read_csv(paths["raw"])
    if len(grid) != 1 or str(grid.iloc[0].Run) != "regret_preserve_cfcompat":
        raise RuntimeError("Frozen v4 grid is not the expected Seed1113 candidate.")
    if len(raw) != 4 * 229 or set(raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("Frozen v4 raw Valid reference is incomplete.")
    sources = {
        key: {"path": str(path.resolve()), "sha256": checkpoint_sha256(path)}
        for key, path in paths.items()
    }
    return grid, raw, sources


def render_report(summary):
    gate = summary.get("candidate_gate")
    risk = summary["risk_gate"]
    lines = [
        "# Cross-Fitted Transfer-Risk CFCompatKD v5", "",
        "- Development seed: `1113`",
        "- New Student trajectories: `0` for gate-only, otherwise exactly `1`",
        "- Official Test constructed: `False`",
        "- Train routing: grouped 5-fold out-of-fold probabilities only",
        "- Label is never a gate feature",
        "- Benefit target: Teacher improves frozen missing ModDrop by >= 0.02 MAE",
        "- Risk weight: `max(2*p-1, 0)`",
        "- KD normalization preserves attenuation; it does not renormalize by gate sum",
        "",
        "## Risk-gate pre-screen", "",
        "- OOF Train AUC: `{:.4f}`".format(risk["train_oof"]["roc_auc"]),
        "- Valid AUC: `{:.4f}`".format(risk["valid_full_train_gate"]["roc_auc"]),
        "- Valid Brier: `{:.4f}`".format(risk["valid_full_train_gate"]["brier"]),
        "- Valid constant Brier: `{:.4f}`".format(risk["valid_full_train_gate"]["constant_brier"]),
        "- Pre-screen passed: `{}`".format(risk["prescreen_passed"]), "",
    ]
    if gate is None:
        lines += ["No formal candidate decision was made in this run.", ""]
    else:
        lines += [
            "## Candidate decision", "",
            "- Verdict: `{}`".format(summary["verdict"]),
            "- Candidate J: `{:.6f}`".format(gate["candidate_J"]),
            "- Replay J: `{:.6f}`".format(gate["replay_J"]),
            "- v4 J: `{:.6f}`".format(gate["v4_J"]),
            "- NTR reduction vs replay: `{:+.4f}`".format(gate["negative_transfer_reduction_vs_replay"]),
            "", "### Checks", "",
        ]
        for name, value in gate["checks"].items():
            lines.append("- {}: `{}`".format(name, value))
        lines.append("")
    return "\n".join(lines) + "\n"


def bind_hooks():
    base.RUNS = RUNS
    base.METHOD = METHOD
    base.load_assets = load_assets
    base.forward_objective = forward_objective
    base.reference_prediction_rows = reference_prediction_rows
    base.projection_summary = crossfit_projection_summary


def main():
    cli = parse_args()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)
    risk_summary = prepare_gate_assets(cli, output_root, logger)

    prefix = "crossfit_transfer_risk_v5"
    if cli.gate_only:
        summary = {
            "version": VERSION,
            "method": METHOD,
            "verdict": "GATE_PRESCREEN_PASS" if risk_summary["prescreen_passed"] else "STOP_CROSSFIT_TRANSFER_RISK_GATE_PRESCREEN_FAILED",
            "risk_gate": risk_summary,
            "candidate_gate": None,
            "protocol": {"development_seed": DEV_SEED, "new_trajectories_trained": 0, "official_test_constructed": False},
        }
        (output_root / f"{prefix}_valid_screen_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / f"{prefix}_valid_screen_report.md").write_text(render_report(summary), encoding="utf-8")
        print("Cross-fitted transfer-risk v5 gate-only pre-screen complete")
        print("OOF Train AUC: {:.6f}".format(risk_summary["train_oof"]["roc_auc"]))
        print("Valid AUC: {:.6f}".format(risk_summary["valid_full_train_gate"]["roc_auc"]))
        print("Valid Brier: {:.6f} constant: {:.6f}".format(
            risk_summary["valid_full_train_gate"]["brier"], risk_summary["valid_full_train_gate"]["constant_brier"]
        ))
        print("prescreen:", summary["verdict"])
        print("official Test was not constructed")
        return

    if not cli.smoke_test and not risk_summary["prescreen_passed"]:
        summary = {
            "version": VERSION,
            "method": METHOD,
            "verdict": "STOP_CROSSFIT_TRANSFER_RISK_GATE_PRESCREEN_FAILED",
            "risk_gate": risk_summary,
            "candidate_gate": None,
            "protocol": {"development_seed": DEV_SEED, "new_trajectories_trained": 0, "official_test_constructed": False},
        }
        (output_root / f"{prefix}_valid_screen_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (output_root / f"{prefix}_valid_screen_report.md").write_text(render_report(summary), encoding="utf-8")
        print("STOP_CROSSFIT_TRANSFER_RISK_GATE_PRESCREEN_FAILED")
        print("No Student trajectory was trained; official Test was not constructed.")
        return

    bind_hooks()
    replay_grid, replay_raw, v2_sources = v4.load_v2_references(cli)
    v4_grid, v4_raw, v4_sources = load_v4_reference(cli)
    result, epoch_rows, candidate_raw, train_decisions = train_trajectory(
        cli, logger, output_root, model_root, DEV_SEED, RUN
    )
    candidate_grid = pd.DataFrame([result])
    candidate_raw_frame = pd.DataFrame(candidate_raw)
    combined_raw = pd.concat([replay_raw, v4_raw, candidate_raw_frame], ignore_index=True, sort=False)
    events = derive_valid_events(combined_raw)
    overall = overall_from_events(events)
    groups = group_summary(events)
    transfer = negative_transfer_summary(events)

    if cli.smoke_test:
        candidate_gate = None
        verdict = "SMOKE_ONLY_NO_DECISION"
    else:
        candidate_gate = dev_candidate_gate(
            result, replay_grid, v4_grid, groups, transfer, epoch_rows, risk_summary
        )
        verdict = (
            "PROMOTE_CROSSFIT_TRANSFER_RISK_V5_TO_3SEED_VALID_SCREEN"
            if candidate_gate["passed"]
            else "STOP_CROSSFIT_TRANSFER_RISK_V5_SINGLE_SEED_DEV_FAILED"
        )

    artifacts = {
        f"{prefix}_candidate_grid.csv": candidate_grid,
        f"{prefix}_all_epoch_metrics.csv": pd.DataFrame(epoch_rows),
        f"{prefix}_candidate_raw_valid_events.csv": candidate_raw_frame,
        f"{prefix}_v2_reference_grid.csv": replay_grid,
        f"{prefix}_v2_reference_raw_valid_events.csv": replay_raw,
        f"{prefix}_v4_reference_grid.csv": v4_grid,
        f"{prefix}_v4_reference_raw_valid_events.csv": v4_raw,
        f"{prefix}_combined_valid_events.csv": events,
        f"{prefix}_overall_metrics.csv": overall,
        f"{prefix}_group_metrics.csv": groups,
        f"{prefix}_transfer_metrics.csv": transfer,
        f"{prefix}_train_decisions.csv": train_decisions,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_root / name, index=False)

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "risk_gate": risk_summary,
        "candidate_gate": candidate_gate,
        "protocol": {
            "development_seed": DEV_SEED,
            "new_trajectories_trained": 1,
            "runs": list(RUNS),
            "decision_split": "official_valid_only",
            "train_routing_probability_source": "grouped_out_of_fold_only",
            "label_used_as_gate_feature": False,
            "official_test_constructed": False,
            "official_test_authorized": False,
            "frozen_thresholds": frozen_thresholds(),
        },
        "log": str(log_path.resolve()),
    }
    summary_path = output_root / f"{prefix}_valid_screen_summary.json"
    report_path = output_root / f"{prefix}_valid_screen_report.md"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path.write_text(render_report(summary), encoding="utf-8")

    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_commit": "c9739a9594867ca345424ff73a2fa69462955da1",
        "implementation_branch": "feature/cfcompat-crossfit-transfer-risk-gate-v5",
        "development_seed": DEV_SEED,
        "runs": list(RUNS),
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "v2_reference_sources": v2_sources,
        "v4_reference_sources": v4_sources,
        "gate_artifacts": {
            key: {"path": str(path.resolve()), "sha256": checkpoint_sha256(path)}
            for key, path in _PRECOMPUTED_GATE_PATHS.items()
        },
        "artifacts": {},
    }
    for name in artifacts:
        path = output_root / name
        manifest["artifacts"][name] = {"path": str(path.resolve()), "sha256": checkpoint_sha256(path)}
    manifest["artifacts"][summary_path.name] = {"path": str(summary_path.resolve()), "sha256": checkpoint_sha256(summary_path)}
    manifest["artifacts"][report_path.name] = {"path": str(report_path.resolve()), "sha256": checkpoint_sha256(report_path)}
    manifest_path = output_root / f"{prefix}_source_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Cross-Fitted Transfer-Risk CFCompatKD v5 screen complete")
    print("risk OOF AUC: {:.6f}".format(risk_summary["train_oof"]["roc_auc"]))
    print("risk Valid AUC: {:.6f}".format(risk_summary["valid_full_train_gate"]["roc_auc"]))
    print("candidate J: {:.6f}".format(float(result["J_valid"])))
    if candidate_gate is not None:
        print("negative-transfer reduction vs replay: {:+.6f}".format(candidate_gate["negative_transfer_reduction_vs_replay"]))
    print("verdict:", verdict)
    print("official Test was not constructed")
    print("report:", report_path)


if __name__ == "__main__":
    main()
