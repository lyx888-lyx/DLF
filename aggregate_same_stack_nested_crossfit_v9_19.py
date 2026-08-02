"""Aggregate all V9.19 outer folds without touching Validation or Test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.same_stack_nested_router_v919 import PROTOCOL_VERSION


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--required-mean-gain", type=float, default=0.005)
    parser.add_argument("--required-positive-folds", type=int, default=4)
    parser.add_argument("--max-harm-over-010-rate", type=float, default=0.05)
    return parser.parse_args()


def main():
    cli = parse_args()
    root = Path(cli.root)
    summaries = []
    predictions = []
    for fold in range(cli.outer_folds):
        fold_dir = root / f"outer_fold_{fold}"
        summary_path = fold_dir / "same_stack_nested_crossfit_v919_fold_summary.json"
        prediction_path = fold_dir / "outer_holdout_predictions.csv"
        if not summary_path.is_file() or not prediction_path.is_file():
            raise FileNotFoundError(f"outer fold {fold} is incomplete")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("version") != PROTOCOL_VERSION:
            raise ValueError(f"outer fold {fold} version mismatch")
        summaries.append(summary)
        frame = pd.read_csv(prediction_path)
        frame["outer_fold"] = fold
        predictions.append(frame)

    combined = pd.concat(predictions, ignore_index=True)
    if combined["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("outer holdout aggregation contains duplicate sample IDs")
    anchor_error = (combined["anchor_prediction"] - combined["label"]).abs()
    selected_error = (combined["selected_prediction"] - combined["label"]).abs()
    gain = anchor_error - selected_error

    fold_rows = []
    for summary in summaries:
        metrics = summary["outer_holdout_metrics"]
        fold_rows.append(
            {
                "outer_fold": summary["outer_fold"],
                "inner_router_accepted": summary["inner_router_accepted"],
                "inner_oof_gain": summary["inner_oof_metrics"]["gain_vs_anchor"],
                "outer_anchor_mae": metrics["anchor_mae"],
                "outer_mae": metrics["mae"],
                "outer_gain": metrics["gain_vs_anchor"],
                "outer_harm_over_010_rate": metrics["harm_over_010_rate"],
                "outer_coverage": metrics["coverage"],
                "outer_trigger_precision": metrics["trigger_precision"],
            }
        )
    fold_frame = pd.DataFrame(fold_rows).sort_values("outer_fold")
    positive_folds = int((fold_frame["outer_gain"] > 0).sum())
    mean_gain = float(gain.mean())
    harm = float((gain < -0.10).mean())
    stop_gate_passed = (
        mean_gain >= cli.required_mean_gain
        and positive_folds >= cli.required_positive_folds
        and harm <= cli.max_harm_over_010_rate
    )
    result = {
        "version": PROTOCOL_VERSION,
        "method": "fully_same_stack_nested_crossfit_only",
        "sample_count": int(len(combined)),
        "outer_fold_count": int(cli.outer_folds),
        "anchor_mae": float(anchor_error.mean()),
        "router_mae": float(selected_error.mean()),
        "gain_vs_anchor": mean_gain,
        "harm_over_010_rate": harm,
        "positive_outer_folds": positive_folds,
        "accepted_inner_router_folds": int(
            fold_frame["inner_router_accepted"].sum()
        ),
        "stop_gate": {
            "required_mean_gain": cli.required_mean_gain,
            "required_positive_folds": cli.required_positive_folds,
            "max_harm_over_010_rate": cli.max_harm_over_010_rate,
            "passed": bool(stop_gate_passed),
            "interpretation": (
                "Only if passed may the fixed protocol be retrained once for a new "
                "locked dataset. Failure means stop dynamic expert routing rather "
                "than tune on MOSI Test."
            ),
        },
        "fold_metrics": fold_rows,
        "provenance": {
            "official_validation_used": False,
            "official_test_used": False,
            "all_metrics_are_outer_group_holdouts": True,
            "same_stack_inner_and_outer": True,
            "no_posthoc_fold_selection": True,
        },
    }
    combined.to_csv(root / "v919_all_outer_holdout_predictions.csv", index=False)
    fold_frame.to_csv(root / "v919_outer_fold_metrics.csv", index=False)
    (root / "full_same_stack_nested_crossfit_v919_summary.json").write_text(
        json.dumps(jsonable(result), indent=2, sort_keys=True), encoding="utf-8"
    )
    print("V9.19 NESTED CROSSFIT AGGREGATION COMPLETE")
    print(f"Anchor MAE: {result['anchor_mae']:.6f}")
    print(f"Router MAE: {result['router_mae']:.6f}")
    print(f"Gain: {result['gain_vs_anchor']:+.6f}")
    print(f"Positive folds: {positive_folds}/{cli.outer_folds}")
    print("Stop gate passed:", stop_gate_passed)


if __name__ == "__main__":
    main()
