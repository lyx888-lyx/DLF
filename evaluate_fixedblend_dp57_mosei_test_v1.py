"""MOSEI Test evaluation for FixedBlend-DP57 with a Valid-frozen anchor seed.

This file exists separately so Test can never select its own anchor.  The caller
must supply --anchor-seed obtained from the completed MOSEI Valid run.
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from evaluate_fixedblend_dp57_v1 import (
    METHOD,
    METRICS,
    MODES,
    SEEDS,
    fixed_blend,
    frame_metrics,
    load_raw5,
    load_v13,
    metric_rows,
    project_dp57,
    raw5_mean,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="MOSEI Test FixedBlend-DP57 with anchor frozen on Valid"
    )
    parser.add_argument("--anchor-seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--raw5-root", required=True)
    parser.add_argument("--v13-predictions", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    output = Path(args.output_root)
    if output.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output exists; inspect it or use --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    frames = load_raw5(Path(args.raw5_root), "test")
    anchor = frames[int(args.anchor_seed)]
    raw5 = raw5_mean(frames)
    v13 = load_v13(Path(args.v13_predictions), frames[SEEDS[0]], "test")
    blend = fixed_blend(raw5, v13)
    projected, diagnostics = project_dp57(anchor, blend, "mosei")

    methods = {
        f"anchor_seed{args.anchor_seed}": anchor,
        "raw5_pe5": raw5,
        "v13": v13,
        "raw5_0p5_v13_0p5_fixed_blend": blend,
        METHOD: projected,
    }
    rows = []
    summary_metrics = {}
    for method, frame in methods.items():
        by_mode, j = frame_metrics(frame)
        rows.extend(metric_rows(method, by_mode, j))
        summary_metrics[method] = {"J": j, **by_mode}

    pd.DataFrame(rows).to_csv(output / "fixedblend_dp57_metrics.csv", index=False)
    diagnostics.to_csv(output / "fixedblend_dp57_projection_diagnostics.csv", index=False)

    anchor_metrics = summary_metrics[f"anchor_seed{args.anchor_seed}"]
    dp_metrics = summary_metrics[METHOD]
    blend_metrics = summary_metrics["raw5_0p5_v13_0p5_fixed_blend"]
    exact = {
        mode: {
            "acc7_equal_anchor": bool(abs(dp_metrics[mode]["acc_7"] - anchor_metrics[mode]["acc_7"]) <= 1e-12),
            "acc5_equal_anchor": bool(abs(dp_metrics[mode]["acc_5"] - anchor_metrics[mode]["acc_5"]) <= 1e-12),
        }
        for mode in MODES + ("MissingMacro",)
    }
    summary = {
        "method": METHOD,
        "dataset": "mosei",
        "split": "test",
        "anchor_seed": int(args.anchor_seed),
        "anchor_source": "frozen_from_mosei_valid_before_test",
        "exact_acc7_acc5_inheritance": exact,
        "fixedblend_J": float(blend_metrics["J"]),
        "dp57_J": float(dp_metrics["J"]),
        "delta_J_dp57_minus_fixedblend": float(dp_metrics["J"] - blend_metrics["J"]),
        "fixedblend_LAV": blend_metrics["LAV"],
        "dp57_LAV": dp_metrics["LAV"],
        "anchor_LAV": anchor_metrics["LAV"],
        "fixedblend_MissingMacro": blend_metrics["MissingMacro"],
        "dp57_MissingMacro": dp_metrics["MissingMacro"],
        "protocol": {
            "anchor_selected_on_test": False,
            "projection_reads_labels": False,
            "raw5_weight": 0.5,
            "v13_weight": 0.5,
            "test_weight_search": False,
            "sample_level_projection_written": False,
        },
    }
    (output / "fixedblend_dp57_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("MOSEI FixedBlend-DP57 Test complete")
    print("Valid-frozen anchor seed:", args.anchor_seed)
    print("FixedBlend Test J: {:.9f}".format(blend_metrics["J"]))
    print("DP57 Test J:       {:.9f}".format(dp_metrics["J"]))
    print("delta J:           {:+.9f}".format(dp_metrics["J"] - blend_metrics["J"]))
    print("Exact Acc7/Acc5 inheritance:", all(
        item["acc7_equal_anchor"] and item["acc5_equal_anchor"]
        for item in exact.values()
    ))
    print("LAV:")
    for key in METRICS:
        print(
            "  {:8s} FixedBlend={:.9f} DP57={:.9f} Anchor={:.9f}".format(
                key,
                float(blend_metrics["LAV"][key]),
                float(dp_metrics["LAV"][key]),
                float(anchor_metrics["LAV"][key]),
            )
        )
    print("sample-level projected predictions written: False")
    print("output:", output)


if __name__ == "__main__":
    main()
