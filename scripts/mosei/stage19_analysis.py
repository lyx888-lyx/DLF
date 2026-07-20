"""Validation-only Stage 19 gate calculations."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("LAV", "LA", "LV", "L")
MISSING = ("LA", "LV", "L")
CLASSIFICATION = ("acc_7", "acc_5", "acc_2", "F1_score")


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def frame(directory):
    return pd.read_csv(Path(directory) / "epoch_metrics.csv")


def row_for_epoch(directory, epoch):
    selected = frame(directory)
    selected = selected[selected.Epoch == int(epoch)]
    if len(selected) != 1:
        raise RuntimeError("Expected exactly one matched epoch {}.".format(epoch))
    return selected.iloc[0]


def best_row(directory):
    values = frame(directory)
    return values.loc[values.JValid.idxmin()]


def metrics(row):
    result = {"J": float(row.JValid)}
    for mode in MODES:
        result[mode] = {
            metric: float(row["{}_{}".format(mode, metric)])
            for metric in CLASSIFICATION + ("Corr", "MAE", "Loss")
        }
    result["MissingMacro"] = {
        metric: float(np.mean([result[mode][metric] for mode in MISSING]))
        for metric in CLASSIFICATION + ("Corr", "MAE", "Loss")
    }
    return result


def delta(left, right):
    result = {"J": left["J"] - right["J"]}
    for mode in MODES + ("MissingMacro",):
        result[mode] = {
            key: left[mode][key] - right[mode][key] for key in left[mode]
        }
    return result


def amp_gate(fp32_dir, fp16_dir, output):
    fp32_frame = frame(fp32_dir)
    fp16_frame = frame(fp16_dir)
    fp32 = metrics(fp32_frame.loc[fp32_frame.JValid.idxmin()])
    fp16 = metrics(fp16_frame.loc[fp16_frame.JValid.idxmin()])
    differences = delta(fp16, fp32)
    checks = {
        "abs_J_le_0_003": abs(differences["J"]) <= 0.003,
        "LAV_MAE_degradation_le_0_003": differences["LAV"]["MAE"] <= 0.003,
        "MissingMacro_MAE_degradation_le_0_003": differences["MissingMacro"]["MAE"] <= 0.003,
        "LAV_Corr_degradation_le_0_002": differences["LAV"]["Corr"] >= -0.002,
        "MissingMacro_Corr_degradation_le_0_002": differences["MissingMacro"]["Corr"] >= -0.002,
    }
    for mode in MODES + ("MissingMacro",):
        for metric in CLASSIFICATION:
            checks["{}_{}_degradation_le_0_003".format(mode, metric)] = (
                differences[mode][metric] >= -0.003
            )
    fp32_wall = float(fp32_frame.EpochWallSeconds.sum())
    fp16_wall = float(fp16_frame.EpochWallSeconds.sum())
    wall_reduction = 1.0 - fp16_wall / fp32_wall
    checks["wall_time_reduction_at_least_20pct"] = wall_reduction >= 0.20
    checks["finite"] = bool(
        np.isfinite(fp32_frame.select_dtypes(include=[np.number]).to_numpy()).all()
        and np.isfinite(fp16_frame.select_dtypes(include=[np.number]).to_numpy()).all()
    )
    accepted = all(checks.values())
    result = {
        "status": "STAGE19B_AMP_PROTOCOL_ACCEPTED"
        if accepted
        else "STAGE19B_AMP_REJECTED_FALLBACK_FP32",
        "accepted": accepted,
        "checks": checks,
        "fp32_best": fp32,
        "fp16_best": fp16,
        "fp16_minus_fp32": differences,
        "fp32_wall_seconds": fp32_wall,
        "fp16_wall_seconds": fp16_wall,
        "wall_time_reduction": wall_reduction,
        "locked_test_access_count": 0,
    }
    atomic_json(output, result)
    return result


def paired_gate(baseline_dir, method_dir, epoch, output):
    baseline = metrics(row_for_epoch(baseline_dir, epoch))
    method = metrics(row_for_epoch(method_dir, epoch))
    differences = delta(method, baseline)
    improved_modes = sum(
        differences[mode]["MAE"] < 0 for mode in MISSING
    )
    result = {
        "epoch": int(epoch),
        "baseline": baseline,
        "method": method,
        "method_minus_baseline": differences,
        "missing_modes_mae_improved": improved_modes,
        "missing_macro_mae_improved": differences["MissingMacro"]["MAE"] < 0,
        "any_major_metric_improved": any(
            differences[mode][metric] < 0
            if metric in ("MAE", "Loss")
            else differences[mode][metric] > 0
            for mode in MODES + ("MissingMacro",)
            for metric in CLASSIFICATION + ("Corr", "MAE")
        ),
        "locked_test_access_count": 0,
    }
    atomic_json(output, result)
    return result


def mgd_full_gate(baseline_dir, method_dir, output):
    baseline = metrics(best_row(baseline_dir))
    method = metrics(best_row(method_dir))
    differences = delta(method, baseline)
    checks = {
        "delta_J_le_minus_0_003": differences["J"] <= -0.003,
        "at_least_2_missing_modes_MAE_improve": sum(
            differences[mode]["MAE"] < 0 for mode in MISSING
        )
        >= 2,
        "MissingMacro_MAE_improves": differences["MissingMacro"]["MAE"] < 0,
        "LAV_MAE_safe": differences["LAV"]["MAE"] <= 0.003,
        "LAV_Corr_safe": differences["LAV"]["Corr"] >= -0.002,
        "MissingMacro_Corr_safe": differences["MissingMacro"]["Corr"] >= -0.002,
    }
    for mode in MODES + ("MissingMacro",):
        for metric in CLASSIFICATION:
            checks["{}_{}_safe".format(mode, metric)] = differences[mode][metric] >= -0.003
    passed = all(checks.values())
    result = {
        "status": "STAGE19D_MGD_FULL_SEED1_PROMOTED"
        if passed
        else "STAGE19D_MGD_FULL_SEED1_FAILED",
        "passed": passed,
        "checks": checks,
        "uniform": baseline,
        "mgd": method,
        "mgd_minus_uniform": differences,
        "locked_test_access_count": 0,
    }
    atomic_json(output, result)
    return result


def parse_args():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    amp = sub.add_parser("amp")
    amp.add_argument("--fp32-dir", required=True)
    amp.add_argument("--fp16-dir", required=True)
    amp.add_argument("--output", required=True)
    paired = sub.add_parser("paired")
    paired.add_argument("--baseline-dir", required=True)
    paired.add_argument("--method-dir", required=True)
    paired.add_argument("--epoch", type=int, required=True)
    paired.add_argument("--output", required=True)
    full = sub.add_parser("mgd-full")
    full.add_argument("--baseline-dir", required=True)
    full.add_argument("--method-dir", required=True)
    full.add_argument("--output", required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "amp":
        result = amp_gate(args.fp32_dir, args.fp16_dir, args.output)
    elif args.command == "paired":
        result = paired_gate(args.baseline_dir, args.method_dir, args.epoch, args.output)
    else:
        result = mgd_full_gate(args.baseline_dir, args.method_dir, args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
