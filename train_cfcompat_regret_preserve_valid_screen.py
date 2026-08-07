"""Regret-Aware Preserve-or-Distill CFCompatKD v4, Valid-only development screen.

The v4 development protocol intentionally trains only one new trajectory on
MOSI Seed 1113.  Existing audited v2 CFCompat replay and student-safe Uniform
artifacts are treated as frozen references and are not retrained.

Official Test is never constructed.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_cfcompat_safe_projection_valid_screen as base
import train_cfcompat_safe_projection_valid_screen_v2 as hardened
from train_cf_compat_kd import batch_to_device
from trains.singleTask.cf_compat_kd_utils import (
    compatibility_for_modes,
    evaluator_prediction,
    gated_kd_loss,
)
from trains.singleTask.cfcompat_regret_preserve_utils import (
    DEV_SEED,
    DISTILL_MARGIN,
    LAMBDA_PRESERVE,
    METHOD,
    MILD_CFCOMPAT_BASE,
    MILD_CFCOMPAT_SCALE,
    OUTPUT_TAG,
    PRESERVE_MARGIN,
    RUN,
    RUNS,
    VERSION,
    dev_candidate_gate,
    frozen_thresholds,
    jsonable,
    negative_transfer_summary,
    regret_preserve_decision,
    regret_projection_summary,
)
from trains.singleTask.cfcompat_safe_projection_utils import (
    derive_valid_events,
    group_summary,
    overall_from_events,
)
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.fixed_kd_utils import checkpoint_sha256, teacher_lav_prediction
from trains.singleTask.missing_utils import (
    MISSING_MODES,
    compute_full_dlf_loss,
    compute_task_loss,
    mode_to_mask,
)


_ORIGINAL_LOAD_ASSETS = base.load_assets
_ORIGINAL_TRAIN_TRAJECTORY = base.train_trajectory
_ACTIVE_TRAIN_BASELINE_FRAME = None
_ACTIVE_TRAIN_DECISIONS = None


class FrozenRegretReference:
    """CPU-only Train baseline cache plus frozen Valid reference predictions."""

    def __init__(self, valid_reference: pd.DataFrame, train_baseline: pd.DataFrame):
        self.valid_reference = valid_reference.copy()
        self.train_baseline = train_baseline.copy()
        self.train_by_index = {
            int(row.sample_index): row._asdict()
            for row in self.train_baseline.itertuples(index=False)
        }

    def parameters(self):
        return iter(())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regret-Aware Preserve-or-Distill CFCompatKD v4 Valid-only Seed-1113 development screen."
    )
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--log-dir", default="log/windows_valid_only_prereq_v1")
    parser.add_argument("--config-file", default="config/config.json")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("The v4 development protocol fixes num_workers=1.")
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
                "v4 output already exists; inspect it or use --overwrite: {} / {}".format(
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
    path = directory / "DLF-mosi-regret-preserve-v4-seed1113-{}-{}.log".format(
        kind, datetime.now().strftime("%Y%m%d-%H%M%S")
    )
    logger = logging.getLogger("regret_preserve_valid_screen")
    logger.handlers.clear()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(path), logging.StreamHandler()):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger, path


def train_baseline_prediction_rows(evaluator, loader, device):
    """Cache the frozen ModDrop baseline for every Train sample and missing mode."""
    evaluator.eval()
    rows = []
    with preserve_rng_state():
        for batch in loader:
            text, audio, vision, labels = batch_to_device(batch, device)
            baseline = {
                mode: evaluator_prediction(
                    evaluator, text, audio, vision, mode
                ).view(-1)
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
                    row["baseline_{}_pred".format(mode)] = float(
                        baseline[mode][offset].detach().cpu()
                    )
                rows.append(row)
    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if len(frame) != 1284 or frame.sample_index.nunique() != 1284:
        raise RuntimeError("v4 Train baseline cache must contain exactly 1284 unique samples.")
    required_numeric = ["label"] + ["baseline_{}_pred".format(mode) for mode in MISSING_MODES]
    if not np.isfinite(frame[required_numeric].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("v4 Train baseline cache contains NaN/Inf.")
    return frame


def load_assets(cli, args, loaders, seed):
    """Build frozen Train/Valid references, then release the ModDrop evaluator."""
    global _ACTIVE_TRAIN_BASELINE_FRAME
    if int(seed) != DEV_SEED:
        raise RuntimeError("v4 development is locked to Seed 1113.")
    teacher, student, evaluator, assets = _ORIGINAL_LOAD_ASSETS(
        cli, args, loaders, seed
    )
    with preserve_rng_state():
        valid_reference = hardened.reference_prediction_rows(
            evaluator, teacher, loaders["valid"], args.device
        )
        train_baseline = train_baseline_prediction_rows(
            evaluator, loaders["train"], args.device
        )
    if len(valid_reference) != 229:
        raise RuntimeError("MOSI Valid reference cache must contain 229 samples.")

    _ACTIVE_TRAIN_BASELINE_FRAME = train_baseline.copy()
    bundle = FrozenRegretReference(valid_reference, train_baseline)

    del evaluator
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    assets["baseline_valid_cache_sample_count"] = int(len(valid_reference))
    assets["baseline_train_cache_sample_count"] = int(len(train_baseline))
    assets["regret_anchor"] = "frozen_validation_best_moddrop_missing_prediction"
    assets["three_way_policy"] = "distill_preserve_abstain"
    return teacher, student, bundle, assets


def baseline_for_modes(bundle, indices, modes, labels, device, dtype):
    values = []
    for offset, (index, mode) in enumerate(zip(indices, modes)):
        if mode not in MISSING_MODES or int(index) not in bundle.train_by_index:
            raise KeyError("Invalid Train baseline binding index={} mode={}.".format(index, mode))
        record = bundle.train_by_index[int(index)]
        if abs(float(record["label"]) - float(labels[offset].detach().cpu())) > 1e-6:
            raise RuntimeError("Train baseline label binding changed at sample {}.".format(index))
        values.append(float(record["baseline_{}_pred".format(mode)]))
    result = torch.as_tensor(values, device=device, dtype=dtype).view(-1, 1)
    if not torch.isfinite(result).all():
        raise FloatingPointError("Bound Train baseline predictions are non-finite.")
    return result


def forward_objective(
    run,
    batch,
    missing_mask,
    modes,
    args,
    teacher,
    evaluator_bundle,
    student,
    cache_by_index,
    criterion,
    cosine,
    hinge,
):
    global _ACTIVE_TRAIN_DECISIONS
    if run != RUN:
        raise ValueError("Unknown v4 run: {}".format(run))
    if _ACTIVE_TRAIN_DECISIONS is None:
        raise RuntimeError("v4 Train decision recorder is not active.")

    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = missing_mask.to(device=args.device, dtype=audio.dtype)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)

    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(
        full_output, labels, criterion, cosine, hinge
    )
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    current_student = missing_output["output_logit"].detach().view(-1, 1)
    teacher_prediction = teacher_lav_prediction(
        teacher, text, audio, vision
    ).view(-1, 1)

    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist()
    compatibility = compatibility_for_modes(
        cache_by_index,
        indices,
        list(modes),
        args.device,
        labels.dtype,
    ).view(-1)
    baseline_prediction = baseline_for_modes(
        evaluator_bundle,
        indices,
        list(modes),
        labels,
        args.device,
        labels.dtype,
    )

    decision = regret_preserve_decision(
        current_student,
        teacher_prediction,
        baseline_prediction,
        labels,
        compatibility,
    )
    kd_loss, each_kd = gated_kd_loss(
        missing_output["output_logit"],
        decision["teacher_safe_target"],
        decision["distill_gate"],
    )
    each_preserve = F.smooth_l1_loss(
        missing_output["output_logit"].view(-1),
        decision["preserve_safe_target"].view(-1),
        reduction="none",
    )
    preserve_weight = decision["preserve_gate"].to(each_preserve)
    preserve_loss = torch.sum(preserve_weight * each_preserve) / (
        torch.sum(preserve_weight) + 1e-8
    )

    total_loss = full_loss + missing_loss + kd_loss + LAMBDA_PRESERVE * preserve_loss
    if not torch.isfinite(total_loss):
        raise FloatingPointError("NaN/Inf in regret-aware preserve-or-distill objective.")

    projection_records = []
    teacher_projection = decision["teacher_projection"]
    identifiers = list(batch["id"])
    for offset in range(labels.size(0)):
        record = {
            key: (
                bool(value[offset].detach().cpu())
                if value.dtype == torch.bool
                else float(value[offset].detach().cpu())
            )
            for key, value in teacher_projection.items()
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
                "teacher_safe_target": float(decision["teacher_safe_target"][offset].detach().cpu()),
                "preserve_safe_target": float(decision["preserve_safe_target"][offset].detach().cpu()),
                "distill": bool(decision["distill"][offset].detach().cpu()),
                "preserve": bool(decision["preserve"][offset].detach().cpu()),
                "decision_abstain": bool(decision["abstain"][offset].detach().cpu()),
                "teacher_beneficial": bool(decision["teacher_beneficial"][offset].detach().cpu()),
                "current_regressed": bool(decision["current_regressed"][offset].detach().cpu()),
                "baseline_error": float(decision["baseline_error"][offset].detach().cpu()),
                "teacher_error": float(decision["teacher_error"][offset].detach().cpu()),
                "current_error": float(decision["current_error"][offset].detach().cpu()),
                "teacher_advantage_vs_baseline": float(decision["teacher_advantage_vs_baseline"][offset].detach().cpu()),
                "current_regret_vs_baseline": float(decision["current_regret_vs_baseline"][offset].detach().cpu()),
                "compatibility": float(compatibility[offset].detach().cpu()),
                "mild_compatibility": float(decision["mild_compatibility"][offset].detach().cpu()),
                "distill_gate": float(decision["distill_gate"][offset].detach().cpu()),
                "preserve_gate": float(decision["preserve_gate"][offset].detach().cpu()),
                "distill_loss_each": float(each_kd[offset].detach().cpu()),
                "preserve_loss_each": float(each_preserve[offset].detach().cpu()),
            }
        )
        record["event_ordinal"] = len(_ACTIVE_TRAIN_DECISIONS) + 1
        projection_records.append(record)
        _ACTIVE_TRAIN_DECISIONS.append(dict(record))

    diagnostics = {
        "full_loss": float(full_loss.detach().cpu()),
        "missing_loss": float(missing_loss.detach().cpu()),
        "kd_loss": float(kd_loss.detach().cpu()),
        "mean_gate": float(decision["distill_gate"].detach().mean().cpu()),
        "weighted_kd": float(kd_loss.detach().cpu()),
        # Frozen trajectory compatibility keys.  Here the baseline key really
        # is the frozen ModDrop anchor, while current Student error is tracked
        # explicitly in the Train decision artifact.
        "baseline_missing_MAE": float(
            torch.abs(baseline_prediction.view(-1) - labels.view(-1)).mean().cpu()
        ),
        "safe_target_MAE": float(
            torch.abs(decision["teacher_safe_target"].view(-1) - labels.view(-1)).mean().cpu()
        ),
    }
    return total_loss, diagnostics, projection_records


def reference_prediction_rows(evaluator_bundle, teacher, loader, device):
    del teacher, loader, device
    return evaluator_bundle.valid_reference.copy()


def train_trajectory(cli, logger, output_root, model_root, seed, run):
    """Run one frozen Stage-3 trajectory with the v4 objective hooks."""
    global _ACTIVE_TRAIN_BASELINE_FRAME, _ACTIVE_TRAIN_DECISIONS
    if int(seed) != DEV_SEED or run != RUN:
        raise RuntimeError("v4 may train only Seed 1113 / regret_preserve_cfcompat.")
    if _ACTIVE_TRAIN_DECISIONS is not None:
        raise RuntimeError("Nested v4 Train decision recording is forbidden.")

    _ACTIVE_TRAIN_BASELINE_FRAME = None
    _ACTIVE_TRAIN_DECISIONS = []
    try:
        result, epoch_rows, raw_events = _ORIGINAL_TRAIN_TRAJECTORY(
            cli, logger, output_root, model_root, seed, run
        )
        train_baseline = _ACTIVE_TRAIN_BASELINE_FRAME.copy()
        decisions = pd.DataFrame(_ACTIVE_TRAIN_DECISIONS)
    finally:
        _ACTIVE_TRAIN_BASELINE_FRAME = None
        _ACTIVE_TRAIN_DECISIONS = None

    if len(train_baseline) != 1284:
        raise RuntimeError("v4 Train baseline cache was not preserved.")
    expected_decisions = 1284 * int(result["TrainEpochCount"])
    if len(decisions) != expected_decisions:
        raise RuntimeError(
            "v4 Train decision count mismatch: {} != {}.".format(len(decisions), expected_decisions)
        )
    decisions["Epoch"] = ((decisions.event_ordinal.astype(int) - 1) // 1284) + 1
    if int(decisions.Epoch.max()) != int(result["TrainEpochCount"]):
        raise RuntimeError("v4 Train decision epoch accounting failed.")

    run_dir = output_root / "seed{}".format(seed) / run
    baseline_path = run_dir / "train_frozen_moddrop_baseline_cache.csv"
    decisions_path = run_dir / "train_regret_decisions.csv"
    train_baseline.to_csv(baseline_path, index=False)
    decisions.to_csv(decisions_path, index=False)

    result.update(
        {
            "Method": METHOD,
            "RegretAnchor": "frozen_validation_best_moddrop_missing_prediction",
            "ThreeWayPolicy": "distill_preserve_abstain",
            "DistillMargin": DISTILL_MARGIN,
            "PreserveMargin": PRESERVE_MARGIN,
            "LambdaPreserve": LAMBDA_PRESERVE,
            "MildCFCompatBase": MILD_CFCOMPAT_BASE,
            "MildCFCompatScale": MILD_CFCOMPAT_SCALE,
            "TrainBaselineCache": str(baseline_path.resolve()),
            "TrainBaselineCacheSHA256": checkpoint_sha256(baseline_path),
            "TrainDecisionRecords": str(decisions_path.resolve()),
            "TrainDecisionRecordsSHA256": checkpoint_sha256(decisions_path),
        }
    )

    for row in epoch_rows:
        epoch = int(row["Epoch"])
        local = decisions.loc[decisions.Epoch.astype(int).eq(epoch)]
        if len(local) != 1284:
            raise RuntimeError("v4 epoch {} has {} decision rows.".format(epoch, len(local)))
        row["preserve_loss_event_mean"] = float(
            (local.preserve_gate * local.preserve_loss_each).sum()
            / (local.preserve_gate.sum() + 1e-8)
        )
        row["distill_loss_event_mean"] = float(
            (local.distill_gate * local.distill_loss_each).sum()
            / (local.distill_gate.sum() + 1e-8)
        )
    pd.DataFrame(epoch_rows).to_csv(run_dir / "epoch_metrics.csv", index=False)
    return result, epoch_rows, raw_events, train_baseline, decisions


def v2_reference_paths(cli):
    root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_student_safe_abstain_v2"
        / cli.dataset
        / "valid_screen"
    )
    return {
        "root": root,
        "grid": root / "student_safe_abstain_valid_grid_summary.csv",
        "raw": root / "student_safe_abstain_raw_valid_events.csv",
        "summary": root / "student_safe_abstain_valid_screen_summary.json",
        "manifest": root / "student_safe_abstain_source_manifest.json",
        "audit": root / "student_safe_abstain_audit_check.json",
    }


def load_v2_references(cli):
    paths = v2_reference_paths(cli)
    missing = [str(path) for key, path in paths.items() if key != "root" and not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing audited v2 reference artifacts:\n" + "\n".join(missing))

    audit = json.loads(paths["audit"].read_text(encoding="utf-8"))
    if not audit.get("passed", False):
        raise RuntimeError("The frozen v2 reference audit did not pass.")

    grid = pd.read_csv(paths["grid"])
    reference_grid = grid.loc[
        grid.Seed.astype(int).eq(DEV_SEED)
        & grid.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    if len(reference_grid) != 2 or set(reference_grid.Run.astype(str)) != {
        "cfcompat_replay",
        "student_safe_uniform",
    }:
        raise RuntimeError("v2 Seed-1113 reference grid is incomplete.")

    raw = pd.read_csv(paths["raw"])
    reference_raw = raw.loc[
        raw.Seed.astype(int).eq(DEV_SEED)
        & raw.Run.astype(str).isin(["cfcompat_replay", "student_safe_uniform"])
    ].copy()
    expected = 2 * 4 * 229
    if len(reference_raw) != expected:
        raise RuntimeError("v2 Seed-1113 reference predictions are incomplete.")
    if set(reference_raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("v2 reference unexpectedly contains a non-Valid split.")

    source_records = {
        key: {
            "path": str(path.resolve()),
            "sha256": checkpoint_sha256(path),
        }
        for key, path in paths.items()
        if key != "root"
    }
    return reference_grid, reference_raw, source_records


def render_report(summary):
    gate = summary.get("candidate_gate")
    lines = [
        "# Regret-Aware Preserve-or-Distill CFCompatKD v4",
        "",
        "## Frozen development protocol",
        "",
        "- Development seed: `1113`",
        "- New trajectories trained: `1` (`regret_preserve_cfcompat`)",
        "- Frozen references: audited v2 `cfcompat_replay` and `student_safe_uniform`",
        "- Official Test constructed: `False`",
        "- Regret anchor: validation-best frozen ModDrop missing-modality prediction",
        "- DISTILL margin: `0.02`",
        "- PRESERVE margin: `0.02`",
        "- Preservation coefficient: `0.25`",
        "- Mild CFCompat prior: `0.75 + 0.25 * compatibility`",
        "",
    ]
    if not gate:
        lines.extend(["Smoke-only run; no scientific promotion decision was made.", ""])
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## Development decision",
            "",
            "- Verdict: `{}`".format(summary["verdict"]),
            "- Candidate Valid-J: `{:.6f}`".format(gate["candidate_J"]),
            "- v2 Uniform Valid-J: `{:.6f}`".format(gate["uniform_J"]),
            "- Original CFCompat replay Valid-J: `{:.6f}`".format(gate["replay_J"]),
            "- Candidate negative-transfer rate: `{:.4f}`".format(gate["candidate_negative_transfer_rate"]),
            "- v2 Uniform negative-transfer rate: `{:.4f}`".format(gate["uniform_negative_transfer_rate"]),
            "- Negative-transfer reduction: `{:+.4f}`".format(gate["negative_transfer_reduction"]),
            "",
            "### Checks",
            "",
        ]
    )
    for name, value in gate["checks"].items():
        lines.append("- {}: `{}`".format(name, value))
    lines.append("")
    return "\n".join(lines) + "\n"


def bind_v4_hooks():
    base.RUNS = RUNS
    base.METHOD = METHOD
    base.load_assets = load_assets
    base.forward_objective = forward_objective
    base.reference_prediction_rows = reference_prediction_rows
    base.projection_summary = regret_projection_summary


def main():
    cli = parse_args()
    bind_v4_hooks()
    output_root, model_root = result_paths(cli)
    logger, log_path = create_logger(cli)

    reference_grid, reference_raw, reference_sources = load_v2_references(cli)
    result, epoch_rows, candidate_raw, train_baseline, train_decisions = train_trajectory(
        cli, logger, output_root, model_root, DEV_SEED, RUN
    )
    candidate_grid = pd.DataFrame([result])
    candidate_raw_frame = pd.DataFrame(candidate_raw)

    combined_raw = pd.concat(
        [reference_raw, candidate_raw_frame], ignore_index=True, sort=False
    )
    events = derive_valid_events(combined_raw)
    overall = overall_from_events(events)
    groups = group_summary(events)
    negative_transfer = negative_transfer_summary(events)

    if cli.smoke_test:
        gate = None
        verdict = "SMOKE_ONLY_NO_DECISION"
    else:
        gate = dev_candidate_gate(
            result,
            reference_grid,
            groups,
            negative_transfer,
            epoch_rows,
        )
        verdict = (
            "PROMOTE_REGRET_PRESERVE_TO_3SEED_VALID_SCREEN"
            if gate["passed"]
            else "STOP_REGRET_PRESERVE_SINGLE_SEED_DEV_FAILED"
        )

    prefix = "regret_preserve_v4"
    artifacts = {
        f"{prefix}_candidate_grid.csv": candidate_grid,
        f"{prefix}_all_epoch_metrics.csv": pd.DataFrame(epoch_rows),
        f"{prefix}_candidate_raw_valid_events.csv": candidate_raw_frame,
        f"{prefix}_reference_grid.csv": reference_grid,
        f"{prefix}_reference_raw_valid_events.csv": reference_raw,
        f"{prefix}_combined_valid_events.csv": events,
        f"{prefix}_overall_metrics.csv": overall,
        f"{prefix}_group_metrics.csv": groups,
        f"{prefix}_negative_transfer_metrics.csv": negative_transfer,
        f"{prefix}_train_baseline_cache.csv": train_baseline,
        f"{prefix}_train_decisions.csv": train_decisions,
    }
    for name, frame in artifacts.items():
        frame.to_csv(output_root / name, index=False)

    source_record = {
        "seed": DEV_SEED,
        "run": RUN,
        "checkpoint": result["MainCheckpoint"],
        "checkpoint_sha256": result["MainCheckpointSHA256"],
        "teacher_checkpoint": result["TeacherCheckpoint"],
        "teacher_sha256": result["TeacherSHA256"],
        "evaluator_checkpoint": result["EvaluatorCheckpoint"],
        "evaluator_sha256": result["EvaluatorSHA256"],
        "evaluator_source": result["EvaluatorSource"],
        "evaluator_source_sha256": result["EvaluatorSourceSHA256"],
        "compatibility_cache": result["CompatibilityCache"],
        "compatibility_cache_sha256": result["CompatibilityCacheSHA256"],
        "compatibility_config": result["CompatibilityConfig"],
        "compatibility_config_sha256": result["CompatibilityConfigSHA256"],
        "train_baseline_cache": result["TrainBaselineCache"],
        "train_baseline_cache_sha256": result["TrainBaselineCacheSHA256"],
        "train_decision_records": result["TrainDecisionRecords"],
        "train_decision_records_sha256": result["TrainDecisionRecordsSHA256"],
    }
    manifest = {
        "version": VERSION,
        "method": METHOD,
        "base_commit": "641f395774363ce6be8e5f694c7e0d11db25feb3",
        "implementation_branch": "feature/cfcompat-regret-preserve-distill-valid-screen-v4",
        "development_seed": DEV_SEED,
        "runs": list(RUNS),
        "decision_split": "official_valid_only",
        "official_test_constructed": False,
        "test_loader_construction_count": 0,
        "test_loader_traversal_count": 0,
        "additional_inference_parameters": 0,
        "reference_sources": reference_sources,
        "source_record": source_record,
        "artifacts": {},
    }
    for name in artifacts:
        path = output_root / name
        manifest["artifacts"][name] = {
            "path": str(path.resolve()),
            "sha256": checkpoint_sha256(path),
        }
    manifest_path = output_root / f"{prefix}_source_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "version": VERSION,
        "method": METHOD,
        "verdict": verdict,
        "candidate_gate": jsonable(gate) if gate is not None else None,
        "protocol": {
            "development_seed": DEV_SEED,
            "new_trajectories_trained": 1,
            "runs": list(RUNS),
            "reference_runs": ["cfcompat_replay", "student_safe_uniform"],
            "regret_anchor": "frozen_validation_best_moddrop_missing_prediction",
            "distill_rule": "teacher_error_plus_margin_le_baseline_error_and_current_student_safe_target_active",
            "preserve_rule": "not_distill_and_current_error_minus_baseline_error_ge_margin",
            "abstain_rule": "not_distill_and_not_preserve",
            "teacher_target": "teacher_clipped_to_current_student_to_train_label_interval",
            "preserve_target": "frozen_moddrop_prediction_clipped_to_current_student_to_train_label_interval",
            "mild_cfcompat": "0.75_plus_0.25_times_compatibility_on_distill_only",
            "lambda_kd": 1.0,
            "lambda_preserve": LAMBDA_PRESERVE,
            "optimizer": "Adam",
            "update_epochs": 10,
            "checkpoint_selection": "minimum_official_valid_J",
            "official_test_constructed": False,
            "official_test_authorized": False,
            "next_stage_on_pass": "frozen_three_seed_1112_1113_1115_valid_screen",
            "frozen_thresholds": frozen_thresholds(),
        },
    }
    summary_path = output_root / f"{prefix}_valid_screen_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report_path = output_root / f"{prefix}_valid_screen_report.md"
    report_path.write_text(render_report(summary), encoding="utf-8")

    logger.info("complete verdict=%s output=%s log=%s", verdict, output_root, log_path)
    print("Regret-Aware Preserve-or-Distill CFCompatKD v4 screen complete")
    if gate is not None:
        print("candidate J:", "{:.6f}".format(gate["candidate_J"]))
        print("v2 uniform J:", "{:.6f}".format(gate["uniform_J"]))
        print(
            "negative-transfer reduction vs v2 uniform:",
            "{:+.6f}".format(gate["negative_transfer_reduction"]),
        )
        print(
            "decision fractions: distill={:.4f} preserve={:.4f} abstain={:.4f}".format(
                gate["decision_fractions"]["distill"],
                gate["decision_fractions"]["preserve"],
                gate["decision_fractions"]["abstain"],
            )
        )
    print("verdict:", verdict)
    print("official Test was not constructed")
    print("report:", report_path)


if __name__ == "__main__":
    main()
