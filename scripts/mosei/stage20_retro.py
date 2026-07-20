"""Validation-only retrospective audit of Stage 19 checkpoints."""

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("LAV", "LA", "LV", "L")
MISSING = ("LA", "LV", "L")
METRICS = ("MAE", "Corr", "acc_7", "acc_5", "acc_2", "F1_score", "Loss")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage19-root", required=True)
    parser.add_argument("--output-root", required=True)
    return parser.parse_args()


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value)
    os.replace(str(temporary), str(path))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def selected(row):
    result = {"J": float(row["JValid"])}
    for mode in MODES:
        result[mode] = {
            metric: float(row["{}_{}".format(mode, metric)])
            for metric in METRICS
        }
    result["MissingMacro"] = {
        metric: float(np.mean([result[mode][metric] for mode in MISSING]))
        for metric in METRICS
    }
    return result


def subtract(method, baseline):
    result = {"J": method["J"] - baseline["J"]}
    for mode in MODES + ("MissingMacro",):
        result[mode] = {
            metric: method[mode][metric] - baseline[mode][metric]
            for metric in METRICS
        }
    return result


def main():
    cli = parse_args()
    root = Path(cli.stage19_root)
    output = Path(cli.output_root) / "retro"
    uniform_frame = pd.read_csv(
        root / "baseline/uniform_seed1111_fp32/epoch_metrics.csv"
    )
    mgd_frame = pd.read_csv(root / "candidates/mgd_seed1111/epoch_metrics.csv")
    uniform_row = uniform_frame.loc[uniform_frame["JValid"].idxmin()]
    mgd_row = mgd_frame.loc[mgd_frame["JValid"].idxmin()]
    uniform = selected(uniform_row)
    mgd = selected(mgd_row)
    difference = subtract(mgd, uniform)
    missing_improved = sum(
        difference[mode]["MAE"] < 0 for mode in MISSING
    )
    checks = {
        "delta_best_le_minus_0_003": difference["J"] <= -0.003,
        "missing_macro_mae_improved": difference["MissingMacro"]["MAE"] < 0,
        "at_least_two_missing_mae_improved": missing_improved >= 2,
        "lav_mae_safe": difference["LAV"]["MAE"] <= 0.003,
        "lav_corr_safe": difference["LAV"]["Corr"] >= -0.002,
        "missing_macro_corr_safe": difference["MissingMacro"]["Corr"] >= -0.002,
        "classification_safe": all(
            difference[mode][metric] >= -0.003
            for mode in ("LAV", "MissingMacro")
            for metric in ("acc_7", "acc_5", "acc_2", "F1_score")
        ),
    }
    retained = all(checks.values())
    status = (
        "STAGE19_RETRO_MGD_EARLY_CHECKPOINT_RETAINED_AUXILIARY"
        if retained
        else "STAGE19_RETRO_MGD_CONFIRMED_CLOSED"
    )
    payload = {
        "status": status,
        "locked_test_access_count": 0,
        "selection": "minimum J over every existing official-valid checkpoint",
        "uniform": {
            "selected_epoch": int(uniform_row["Epoch"]),
            "best_so_far_J": uniform["J"],
            "metrics": uniform,
        },
        "mgd": {
            "selected_epoch": int(mgd_row["Epoch"]),
            "best_so_far_J": mgd["J"],
            "metrics": mgd,
        },
        "mgd_minus_uniform": difference,
        "missing_modes_mae_improved": missing_improved,
        "safety_checks": checks,
        "retrained": False,
        "mgrd_run": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_json(output / "stage19_retro_best_checkpoint_audit.json", payload)
    lines = [
        "# Stage 19 retrospective best-checkpoint audit",
        "",
        "- Status: `{}`".format(status),
        "- Uniform selected epoch: {}; J={:.9f}".format(
            int(uniform_row["Epoch"]), uniform["J"]
        ),
        "- MGD selected epoch: {}; J={:.9f}".format(
            int(mgd_row["Epoch"]), mgd["J"]
        ),
        "- Delta best (MGD - Uniform): `{:+.9f}`".format(difference["J"]),
        "- Missing modes with lower MAE: `{}/3`".format(missing_improved),
        "- Locked Test access count: `0`",
        "",
        "| Mode | Delta MAE | Delta Corr | Delta Acc7 | Delta Acc5 | Delta Acc2 | Delta F1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in MODES + ("MissingMacro",):
        value = difference[mode]
        lines.append(
            "| {} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} | {:+.6f} |".format(
                mode,
                value["MAE"],
                value["Corr"],
                value["acc_7"],
                value["acc_5"],
                value["acc_2"],
                value["F1_score"],
            )
        )
    lines += [
        "",
        "MGD is not retrained, MGRD is not run, and no tau or loss weight is changed.",
    ]
    atomic_text(
        output / "stage19_retro_best_checkpoint_audit.md",
        "\n".join(lines) + "\n",
    )
    print(json.dumps({"status": status, "delta_best": difference["J"]}, indent=2))


if __name__ == "__main__":
    main()
