#!/usr/bin/env python3
"""Audit V8 stage isolation, uncertainty direction, and final metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="./result/unimodal_experts_v8_safe/mosi/seed_1111",
    )
    return parser.parse_args()


def main():
    root = Path(parse_args().root)
    rows = []
    failures = []
    for modality in ("text", "audio", "vision"):
        summary_path = root / modality / "unimodal_expert_v8_summary.json"
        if not summary_path.is_file():
            failures.append(f"{modality}: missing {summary_path}")
            continue
        data = json.loads(summary_path.read_text(encoding="utf-8"))
        invariant = data["stage_b"]["prediction_invariance"]
        test = data["test"]
        metrics = test["metrics"]
        uncertainty = test["uncertainty"]
        rows.append({
            "modality": modality,
            "selected_stage": data["selected_stage"],
            "orientation": data["score_orientation"],
            "MAE": metrics["MAE"],
            "Corr": metrics["Corr"],
            "Acc7": metrics["acc_7"],
            "Acc5": metrics["acc_5"],
            "Spearman": uncertainty["error_spearman"],
            "AUROC": uncertainty["high_error_auroc"],
            "Q4_Q1": uncertainty["q4_q1_ratio"],
            "target_error_spearman": uncertainty["target_abs_error_spearman"],
            "stage_b_max_prediction_diff": invariant[
                "max_abs_prediction_difference"
            ],
        })
        if not invariant["within_1e_7"]:
            failures.append(f"{modality}: Stage B changed sentiment predictions")
        if uncertainty["target_abs_error_spearman"] <= 0.99:
            failures.append(f"{modality}: target/error alignment is invalid")
        if data["selected_stage"] != "uncertainty_head_only":
            failures.append(
                f"{modality}: diagnostic run unexpectedly selected "
                f"{data['selected_stage']}"
            )

    if rows:
        print(pd.DataFrame(rows).to_string(index=False))
    if failures:
        print("\nAUDIT FAILURES")
        for failure in failures:
            print("-", failure)
        raise SystemExit(1)
    print("\nAUDIT PASSED")


if __name__ == "__main__":
    main()
