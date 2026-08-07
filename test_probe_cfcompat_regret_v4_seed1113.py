"""One-time MOSI Test viability probe for frozen CFCompatKD vs Regret-Preserve v4.

This is intentionally NOT a training script.  It exists only to answer one
pre-registered development question after the user explicitly authorized a
limited Test viability probe:

    Does frozen Seed1113 Regret-Preserve v4 show cross-split evidence of
    improvement over the frozen original CFCompatKD replay?

Protocol constraints:
* Seed is fixed to 1113.
* Compared checkpoints are selected exclusively by their prior official Valid
  runs and are hash-bound before Test is constructed.
* The frozen validation-best ModDrop checkpoint is the common regret baseline.
* No optimizer, scheduler, training loader, Valid loader, epoch selection, or
  hyperparameter selection is constructed here.
* Test is used only for inference and aggregate evaluation.
* No sample-level Test CSV/prediction artifact is written.
* A one-time marker is written before Test construction so accidental repeated
  Test probing is refused.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_loader import MMDataLoader
from train_cf_compat_kd import build_config, prediction_rows
from trains.singleTask.cfcompat_regret_preserve_utils import (
    NEGATIVE_TRANSFER_MARGIN,
    SEVERE_NEGATIVE_TRANSFER_MARGIN,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import MISSING_MODES, MissingModalityWrapper
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


PROBE_ID = "MOSI_TEST_VIABILITY_CFCompat_vs_RegretPreserveV4_Seed1113_v1"
CONFIRM_TOKEN = "RUN_MOSI_TEST_PROBE_1113_CFCompat_vs_V4_ONCE"
SEED = 1113
CF_RUN = "cfcompat_replay"
V4_RUN = "regret_preserve_cfcompat"
OUTPUT_TAG = "cfcompat_regret_v4_test_viability_probe_v1"

# Frozen before Test access.  PASS is only a viability signal, not a final
# publication claim and not authorization to tune on Test.
NEGATIVE_TRANSFER_MAX_INCREASE = 0.01
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 20260807


def parse_args():
    parser = argparse.ArgumentParser(description="One-time frozen MOSI Test viability probe.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--run-once", action="store_true")
    parser.add_argument("--confirm-token", default="")
    parser.add_argument("--dataset", choices=("mosi",), default="mosi")
    parser.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--model-save-dir", default="pt")
    parser.add_argument("--config-file", default="config/config.json")
    args = parser.parse_args()
    if int(args.num_workers) != 1:
        parser.error("Test probe fixes num_workers=1.")
    if args.run_once and args.confirm_token != CONFIRM_TOKEN:
        parser.error("The exact frozen Test-probe confirmation token is required.")
    return args


def _read_passed_audit(path: Path, label: str):
    if not path.is_file():
        raise FileNotFoundError("{} audit is absent: {}".format(label, path))
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("passed", False):
        raise RuntimeError("{} audit did not pass; Test probe is forbidden.".format(label))
    return payload


def _one_row(frame: pd.DataFrame, run: str, label: str):
    selected = frame.loc[
        frame.Seed.astype(int).eq(SEED) & frame.Run.astype(str).eq(str(run))
    ]
    if len(selected) != 1:
        raise RuntimeError("{} has no unique Seed{} / {} row.".format(label, SEED, run))
    return selected.iloc[0].to_dict()


def _bind_checkpoint(row, path_key="MainCheckpoint", sha_key="MainCheckpointSHA256"):
    if path_key not in row or sha_key not in row:
        raise KeyError("Grid row lacks {} / {}.".format(path_key, sha_key))
    path = Path(str(row[path_key]))
    expected = str(row[sha_key])
    if not path.is_file():
        raise FileNotFoundError("Frozen checkpoint is absent: {}".format(path))
    actual = checkpoint_sha256(path)
    if actual != expected:
        raise RuntimeError("Frozen checkpoint hash changed: {}".format(path))
    return path.resolve(), actual


def frozen_sources(cli):
    v2_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_student_safe_abstain_v2"
        / cli.dataset
        / "valid_screen"
    )
    v4_root = (
        Path(cli.result_root)
        / "missing_baseline"
        / "cfcompat_regret_preserve_v4"
        / cli.dataset
        / "valid_screen"
        / "seed1113_dev"
    )
    v2_grid_path = v2_root / "student_safe_abstain_valid_grid_summary.csv"
    v2_audit_path = v2_root / "student_safe_abstain_audit_check.json"
    v4_grid_path = v4_root / "regret_preserve_v4_candidate_grid.csv"
    v4_audit_path = v4_root / "regret_preserve_v4_audit_check.json"

    _read_passed_audit(v2_audit_path, "v2 frozen reference")
    _read_passed_audit(v4_audit_path, "v4 candidate")
    if not v2_grid_path.is_file() or not v4_grid_path.is_file():
        raise FileNotFoundError("Frozen Valid grid artifact is absent.")

    v2_grid = pd.read_csv(v2_grid_path)
    v4_grid = pd.read_csv(v4_grid_path)
    cf_row = _one_row(v2_grid, CF_RUN, "v2 grid")
    v4_row = _one_row(v4_grid, V4_RUN, "v4 grid")

    cf_checkpoint, cf_sha = _bind_checkpoint(cf_row)
    v4_checkpoint, v4_sha = _bind_checkpoint(v4_row)
    baseline_checkpoint, baseline_sha = _bind_checkpoint(
        v4_row, "EvaluatorCheckpoint", "EvaluatorSHA256"
    )

    # Both compared trajectories must be evaluated relative to the same frozen
    # ModDrop anchor that was already bound during Valid development.
    if str(cf_row.get("EvaluatorSHA256")) != baseline_sha:
        raise RuntimeError("CFCompat and v4 are not bound to the same frozen ModDrop evaluator.")

    return {
        "cfcompat": {
            "run": CF_RUN,
            "checkpoint": str(cf_checkpoint),
            "sha256": cf_sha,
            "best_valid_epoch": int(cf_row["BestValidEpoch"]),
            "J_valid": float(cf_row["J_valid"]),
        },
        "regret_preserve_v4": {
            "run": V4_RUN,
            "checkpoint": str(v4_checkpoint),
            "sha256": v4_sha,
            "best_valid_epoch": int(v4_row["BestValidEpoch"]),
            "J_valid": float(v4_row["J_valid"]),
        },
        "moddrop_baseline": {
            "checkpoint": str(baseline_checkpoint),
            "sha256": baseline_sha,
            "best_valid_epoch": int(v4_row["EvaluatorBestEpoch"]),
        },
        "source_artifacts": {
            "v2_grid": {"path": str(v2_grid_path.resolve()), "sha256": checkpoint_sha256(v2_grid_path)},
            "v2_audit": {"path": str(v2_audit_path.resolve()), "sha256": checkpoint_sha256(v2_audit_path)},
            "v4_grid": {"path": str(v4_grid_path.resolve()), "sha256": checkpoint_sha256(v4_grid_path)},
            "v4_audit": {"path": str(v4_audit_path.resolve()), "sha256": checkpoint_sha256(v4_audit_path)},
        },
    }


def output_paths(cli):
    root = Path(cli.result_root) / "test_viability_probe" / OUTPUT_TAG / "mosi" / "seed1113"
    return {
        "root": root,
        "started": root / "TEST_PROBE_STARTED.json",
        "summary": root / "test_viability_probe_summary.json",
        "report": root / "test_viability_probe_report.md",
        "manifest": root / "test_viability_probe_manifest.json",
    }


def build_inference_model(args, checkpoint: Path):
    backbone = DLF(args).to(args.device)
    model = MissingModalityWrapper(
        backbone, args.feature_dims[1], args.feature_dims[2]
    ).to(args.device)
    state = torch.load(checkpoint, map_location=args.device)
    model.load_state_dict(state, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def infer_rows(args, loader, checkpoint: Path):
    model = build_inference_model(args, checkpoint)
    try:
        frame = prediction_rows(model, loader, args.device).reset_index(drop=True)
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    if frame.empty or frame.sample_index.duplicated().any():
        raise RuntimeError("Test inference returned an empty or duplicate-index frame.")
    numeric = ["label"] + ["{}_pred".format(mode) for mode in ("LAV",) + MISSING_MODES]
    if not np.isfinite(frame[numeric].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("Test inference produced NaN/Inf.")
    return frame


def bind_test_frames(baseline, cfcompat, candidate):
    keys = ["sample_index", "sample_id", "label"]
    reference = baseline[keys].copy().reset_index(drop=True)
    for name, frame in (("cfcompat", cfcompat), ("candidate", candidate)):
        local = frame[keys].copy().reset_index(drop=True)
        if len(local) != len(reference):
            raise RuntimeError("{} Test sample count differs from baseline.".format(name))
        if not np.array_equal(local.sample_index.to_numpy(), reference.sample_index.to_numpy()):
            raise RuntimeError("{} Test sample indices differ from baseline.".format(name))
        if not local.sample_id.astype(str).equals(reference.sample_id.astype(str)):
            raise RuntimeError("{} Test sample IDs differ from baseline.".format(name))
        if not np.allclose(
            local.label.to_numpy(dtype=np.float64),
            reference.label.to_numpy(dtype=np.float64),
            atol=1e-7,
            rtol=0.0,
        ):
            raise RuntimeError("{} Test labels differ from baseline traversal.".format(name))


def aggregate_metrics(frame):
    label = frame.label.to_numpy(dtype=np.float64)
    result = {}
    for mode in ("LAV",) + MISSING_MODES:
        pred = frame["{}_pred".format(mode)].to_numpy(dtype=np.float64)
        result[mode] = {"MAE": float(np.mean(np.abs(pred - label)))}
    result["MissingMacro"] = float(
        np.mean([result[mode]["MAE"] for mode in MISSING_MODES])
    )
    result["J_test"] = float(
        0.5 * result["LAV"]["MAE"] + 0.5 * result["MissingMacro"]
    )
    return result


def transfer_summary(candidate, baseline):
    regrets = []
    label = candidate.label.to_numpy(dtype=np.float64)
    for mode in MISSING_MODES:
        candidate_error = np.abs(
            candidate["{}_pred".format(mode)].to_numpy(dtype=np.float64) - label
        )
        baseline_error = np.abs(
            baseline["{}_pred".format(mode)].to_numpy(dtype=np.float64) - label
        )
        regrets.append(candidate_error - baseline_error)
    regret = np.concatenate(regrets)
    return {
        "N_missing_events": int(regret.size),
        "negative_transfer_margin": NEGATIVE_TRANSFER_MARGIN,
        "negative_transfer_rate": float((regret > NEGATIVE_TRANSFER_MARGIN).mean()),
        "severe_negative_transfer_margin": SEVERE_NEGATIVE_TRANSFER_MARGIN,
        "severe_negative_transfer_rate": float((regret > SEVERE_NEGATIVE_TRANSFER_MARGIN).mean()),
        "positive_transfer_rate": float((regret < -NEGATIVE_TRANSFER_MARGIN).mean()),
        "mean_regret_vs_baseline": float(regret.mean()),
        "median_regret_vs_baseline": float(np.median(regret)),
    }


def per_sample_j(frame):
    label = frame.label.to_numpy(dtype=np.float64)
    lav = np.abs(frame.LAV_pred.to_numpy(dtype=np.float64) - label)
    missing = np.mean(
        np.stack(
            [
                np.abs(frame["{}_pred".format(mode)].to_numpy(dtype=np.float64) - label)
                for mode in MISSING_MODES
            ],
            axis=1,
        ),
        axis=1,
    )
    return 0.5 * lav + 0.5 * missing


def paired_bootstrap_delta(candidate_values, reference_values):
    candidate_values = np.asarray(candidate_values, dtype=np.float64)
    reference_values = np.asarray(reference_values, dtype=np.float64)
    if candidate_values.shape != reference_values.shape or candidate_values.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be same-length vectors.")
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    delta = candidate_values - reference_values
    n = len(delta)
    estimates = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    # Chunked generation avoids a large temporary index matrix.
    chunk = 500
    cursor = 0
    while cursor < BOOTSTRAP_REPLICATES:
        count = min(chunk, BOOTSTRAP_REPLICATES - cursor)
        indices = rng.integers(0, n, size=(count, n), endpoint=False)
        estimates[cursor : cursor + count] = delta[indices].mean(axis=1)
        cursor += count
    low, high = np.quantile(estimates, [0.025, 0.975])
    return {
        "mean_delta": float(delta.mean()),
        "ci95_low": float(low),
        "ci95_high": float(high),
        "replicates": BOOTSTRAP_REPLICATES,
        "seed": BOOTSTRAP_SEED,
    }


def ephemeral_digest(baseline, cfcompat, candidate):
    columns = [baseline.sample_index.to_numpy(dtype=np.float64), baseline.label.to_numpy(dtype=np.float64)]
    for frame in (baseline, cfcompat, candidate):
        for mode in ("LAV",) + MISSING_MODES:
            columns.append(frame["{}_pred".format(mode)].to_numpy(dtype=np.float64))
    matrix = np.ascontiguousarray(np.stack(columns, axis=1), dtype=np.float64)
    return hashlib.sha256(matrix.tobytes()).hexdigest()


def render_report(summary):
    metrics = summary["metrics"]
    checks = summary["viability_checks"]
    lines = [
        "# MOSI Test viability probe: CFCompatKD vs Regret-Preserve v4",
        "",
        "This was a pre-registered one-time development probe, not a final untouched Test claim.",
        "",
        "- Seed: `1113`",
        "- Checkpoints: frozen best-Valid only",
        "- Training / checkpoint selection during probe: `False`",
        "- Sample-level Test artifact written: `False`",
        "- Verdict: `{}`".format(summary["verdict"]),
        "",
        "## Aggregate Test metrics",
        "",
        "| Method | LAV MAE | LA MAE | LV MAE | L MAE | MissingMacro | J_test |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, label in (("cfcompat", "CFCompatKD"), ("regret_preserve_v4", "Regret-Preserve v4")):
        m = metrics[key]
        lines.append(
            "| {} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} | {:.6f} |".format(
                label, m["LAV"]["MAE"], m["LA"]["MAE"], m["LV"]["MAE"],
                m["L"]["MAE"], m["MissingMacro"], m["J_test"]
            )
        )
    lines += [
        "",
        "## Pre-registered viability checks",
        "",
    ]
    for key, value in checks.items():
        lines.append("- {}: `{}`".format(key, value))
    lines += [
        "",
        "Test was used here only as a limited viability probe.  Do not tune further from sample-level Test behavior.",
        "",
    ]
    return "\n".join(lines)


def run_preflight(cli):
    sources = frozen_sources(cli)
    paths = output_paths(cli)
    print("Test viability probe preflight passed")
    print("probe_id:", PROBE_ID)
    print("seed:", SEED)
    print("cfcompat best-valid epoch:", sources["cfcompat"]["best_valid_epoch"])
    print("v4 best-valid epoch:", sources["regret_preserve_v4"]["best_valid_epoch"])
    print("cfcompat checkpoint sha256:", sources["cfcompat"]["sha256"])
    print("v4 checkpoint sha256:", sources["regret_preserve_v4"]["sha256"])
    print("baseline checkpoint sha256:", sources["moddrop_baseline"]["sha256"])
    print("Test constructed: False")
    if paths["started"].exists():
        print("WARNING: one-time Test probe marker already exists:", paths["started"])
    else:
        print("one-time marker absent: ready for explicit --run-once")


def run_once(cli):
    paths = output_paths(cli)
    if paths["started"].exists():
        raise RuntimeError(
            "The one-time MOSI Test probe marker already exists; repeated Test probing is refused: {}".format(
                paths["started"]
            )
        )
    sources = frozen_sources(cli)
    paths["root"].mkdir(parents=True, exist_ok=True)

    started = {
        "probe_id": PROBE_ID,
        "status": "STARTED_BEFORE_TEST_CONSTRUCTION",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "confirmation_token_sha256": hashlib.sha256(CONFIRM_TOKEN.encode("utf-8")).hexdigest(),
        "frozen_sources": sources,
    }
    paths["started"].write_text(
        json.dumps(started, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    setup_seed(SEED)
    args = build_config(cli, SEED)
    args.mode = "test"
    args.is_training = False
    loaders = MMDataLoader(args, cli.num_workers)
    if set(loaders) != {"test"}:
        raise RuntimeError("One-time probe must construct exactly one Test loader and no Train/Valid loader.")
    test_loader = loaders["test"]

    baseline = infer_rows(args, test_loader, Path(sources["moddrop_baseline"]["checkpoint"]))
    cfcompat = infer_rows(args, test_loader, Path(sources["cfcompat"]["checkpoint"]))
    candidate = infer_rows(args, test_loader, Path(sources["regret_preserve_v4"]["checkpoint"]))
    bind_test_frames(baseline, cfcompat, candidate)

    baseline_metrics = aggregate_metrics(baseline)
    cf_metrics = aggregate_metrics(cfcompat)
    v4_metrics = aggregate_metrics(candidate)
    cf_transfer = transfer_summary(cfcompat, baseline)
    v4_transfer = transfer_summary(candidate, baseline)

    mode_delta = {
        mode: float(v4_metrics[mode]["MAE"] - cf_metrics[mode]["MAE"])
        for mode in ("LAV",) + MISSING_MODES
    }
    missing_modes_improved = int(sum(mode_delta[mode] < 0.0 for mode in MISSING_MODES))
    j_delta = float(v4_metrics["J_test"] - cf_metrics["J_test"])
    missing_macro_delta = float(v4_metrics["MissingMacro"] - cf_metrics["MissingMacro"])

    checks = {
        "J_test_improves_vs_cfcompat": bool(j_delta < 0.0),
        "MissingMacro_not_worse_than_cfcompat": bool(missing_macro_delta <= 0.0),
        "at_least_two_missing_modes_improve": bool(missing_modes_improved >= 2),
        "negative_transfer_not_worse_than_cfcompat_by_more_than_0p01": bool(
            v4_transfer["negative_transfer_rate"]
            <= cf_transfer["negative_transfer_rate"] + NEGATIVE_TRANSFER_MAX_INCREASE
        ),
    }
    passed = bool(all(checks.values()))
    verdict = "TEST_VIABILITY_SIGNAL_POSITIVE" if passed else "TEST_VIABILITY_SIGNAL_NEGATIVE"

    bootstrap = paired_bootstrap_delta(per_sample_j(candidate), per_sample_j(cfcompat))
    digest = ephemeral_digest(baseline, cfcompat, candidate)

    summary = {
        "probe_id": PROBE_ID,
        "verdict": verdict,
        "seed": SEED,
        "test_sample_count": int(len(candidate)),
        "selection_basis": "frozen_best_valid_checkpoints_only",
        "training_during_probe": False,
        "checkpoint_selection_during_probe": False,
        "sample_level_test_artifact_written": False,
        "metrics": {
            "moddrop_baseline": baseline_metrics,
            "cfcompat": cf_metrics,
            "regret_preserve_v4": v4_metrics,
        },
        "transfer_vs_frozen_moddrop": {
            "cfcompat": cf_transfer,
            "regret_preserve_v4": v4_transfer,
        },
        "paired_deltas_v4_minus_cfcompat": {
            "mode_MAE": mode_delta,
            "MissingMacro": missing_macro_delta,
            "J_test": j_delta,
            "missing_modes_improved_count": missing_modes_improved,
            "J_test_paired_bootstrap_95ci": bootstrap,
        },
        "viability_checks": checks,
        "frozen_viability_thresholds": {
            "require_J_test_delta_lt_0": True,
            "require_MissingMacro_delta_le_0": True,
            "require_at_least_two_of_LA_LV_L_improve": True,
            "negative_transfer_max_increase": NEGATIVE_TRANSFER_MAX_INCREASE,
        },
        "ephemeral_test_prediction_digest_sha256": digest,
    }
    paths["summary"].write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths["report"].write_text(render_report(summary) + "\n", encoding="utf-8")

    manifest = {
        "probe_id": PROBE_ID,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "frozen_sources": sources,
        "test_loader_construction_count": 1,
        "test_model_inference_passes": 3,
        "train_loader_construction_count": 0,
        "valid_loader_construction_count": 0,
        "optimizer_construction_count": 0,
        "sample_level_test_output_count": 0,
        "artifacts": {
            paths["started"].name: checkpoint_sha256(paths["started"]),
            paths["summary"].name: checkpoint_sha256(paths["summary"]),
            paths["report"].name: checkpoint_sha256(paths["report"]),
        },
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    started["status"] = "COMPLETED"
    started["completed_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    started["verdict"] = verdict
    started["summary_sha256"] = checkpoint_sha256(paths["summary"])
    started["manifest_sha256"] = checkpoint_sha256(paths["manifest"])
    paths["started"].write_text(
        json.dumps(started, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("MOSI one-time Test viability probe complete")
    print("probe_id:", PROBE_ID)
    print("Test sample count:", len(candidate))
    print("CFCompat J_test: {:.6f}".format(cf_metrics["J_test"]))
    print("Regret-Preserve v4 J_test: {:.6f}".format(v4_metrics["J_test"]))
    print("delta J_test (v4-cfcompat): {:+.6f}".format(j_delta))
    print("CFCompat MissingMacro: {:.6f}".format(cf_metrics["MissingMacro"]))
    print("v4 MissingMacro: {:.6f}".format(v4_metrics["MissingMacro"]))
    print("missing modes improved:", missing_modes_improved, "/ 3")
    print("CFCompat negative-transfer rate: {:.6f}".format(cf_transfer["negative_transfer_rate"]))
    print("v4 negative-transfer rate: {:.6f}".format(v4_transfer["negative_transfer_rate"]))
    print("paired bootstrap 95% CI for delta J: [{:+.6f}, {:+.6f}]".format(
        bootstrap["ci95_low"], bootstrap["ci95_high"]
    ))
    print("verdict:", verdict)
    print("sample-level Test artifacts written: False")
    print("report:", paths["report"])


if __name__ == "__main__":
    cli = parse_args()
    if cli.preflight:
        run_preflight(cli)
    else:
        run_once(cli)
