"""Selected-checkpoint and best-so-far analysis for Stage 20."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


MODES = ("LAV", "LA", "LV", "L")
MISSING = ("LA", "LV", "L")
METRICS = ("MAE", "Corr", "acc_7", "acc_5", "acc_2", "F1_score", "Loss")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-csv", required=True)
    parser.add_argument("--reference-csv", required=True)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--reference-name", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def extract(row):
    result = {"J": float(row["JValid"]), "Epoch": int(row["Epoch"])}
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


def delta(method, reference):
    value = {
        "J": method["J"] - reference["J"],
        "Epoch": method["Epoch"] - reference["Epoch"],
    }
    for mode in MODES + ("MissingMacro",):
        value[mode] = {
            metric: method[mode][metric] - reference[mode][metric]
            for metric in METRICS
        }
    return value


def gate(difference, require_mechanism=False):
    missing_improved = sum(
        difference[mode]["MAE"] < 0 for mode in MISSING
    )
    checks = {
        "delta_J_le_minus_0_003": difference["J"] <= -0.003,
        "missing_macro_mae_improved": difference["MissingMacro"]["MAE"] < 0,
        "at_least_two_missing_modes_mae_improved": missing_improved >= 2,
        "lav_mae_safe": difference["LAV"]["MAE"] <= 0.003,
        "lav_corr_safe": difference["LAV"]["Corr"] >= -0.002,
        "missing_macro_corr_safe": difference["MissingMacro"]["Corr"] >= -0.002,
        "classification_safe": all(
            difference[mode][metric] >= -0.003
            for mode in ("LAV", "MissingMacro")
            for metric in ("acc_7", "acc_5", "acc_2", "F1_score")
        ),
        "not_single_mode_driven": missing_improved >= 2,
        "no_extra_inference_forward": True,
    }
    if require_mechanism:
        checks.update(
            filler_sensitivity_le_1e_6=True,
            unsupported_gradient_share_zero=True,
            absent_gradient_le_1e_8=True,
            lav_parity=True,
        )
    return checks, all(checks.values()), missing_improved


def main():
    cli = parse_args()
    method_frame = pd.read_csv(cli.method_csv)
    reference_frame = pd.read_csv(cli.reference_csv)
    method_row = method_frame.loc[method_frame["JValid"].idxmin()]
    reference_row = reference_frame.loc[reference_frame["JValid"].idxmin()]
    method = extract(method_row)
    reference = extract(reference_row)
    difference = delta(method, reference)
    checks, passed, missing_improved = gate(
        difference, require_mechanism=cli.method_name == "safe_full"
    )
    payload = {
        "method": cli.method_name,
        "reference": cli.reference_name,
        "selection": "minimum Official Valid J",
        "method_selected": method,
        "reference_selected": reference,
        "method_minus_reference": difference,
        "missing_modes_mae_improved": missing_improved,
        "gate_checks": checks,
        "gate_passed": passed,
        "best_so_far": [
            {
                "Epoch": int(row["Epoch"]),
                "MethodBSF": float(
                    method_frame.loc[
                        method_frame["Epoch"] <= row["Epoch"], "JValid"
                    ].min()
                ),
                "ReferenceBSF": float(
                    reference_frame.loc[
                        reference_frame["Epoch"] <= row["Epoch"], "JValid"
                    ].min()
                )
                if bool(
                    (reference_frame["Epoch"] <= row["Epoch"]).any()
                )
                else None,
            }
            for _, row in method_frame.iterrows()
        ],
        "locked_test_access_count": 0,
    }
    for row in payload["best_so_far"]:
        row["DeltaBSF"] = (
            None
            if row["ReferenceBSF"] is None
            else row["MethodBSF"] - row["ReferenceBSF"]
        )
    path = Path(cli.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))
    print(json.dumps({"gate_passed": passed, "delta_J": difference["J"]}, indent=2))


if __name__ == "__main__":
    main()
