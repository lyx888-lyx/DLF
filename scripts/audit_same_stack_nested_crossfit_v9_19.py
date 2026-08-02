"""Engineering audit for V9.19 fold and aggregate artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--outer-fold", type=int)
    parser.add_argument("--aggregate", action="store_true")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def audit_fold(root: Path, fold: int):
    fold_dir = root / f"outer_fold_{fold}"
    summary_path = fold_dir / "same_stack_nested_crossfit_v919_fold_summary.json"
    inner_pool_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
    inner_metrics_path = fold_dir / "inner_router_fold_metrics.csv"
    outer_predictions_path = fold_dir / "outer_holdout_predictions.csv"
    for path in (summary_path, inner_pool_path, inner_metrics_path, outer_predictions_path):
        require(path.is_file(), f"missing V9.19 fold artifact: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    provenance = summary["provenance"]
    require(
        provenance["same_stack_recipe_used_for_inner_and_outer"] is True,
        "stack mismatch",
    )
    require(
        provenance["every_inner_oof_row_unseen_by_its_complete_expert_stack"] is True,
        "inner leakage",
    )
    require(
        provenance["outer_holdout_unseen_by_anchor_experts_router_and_policy"] is True,
        "outer leakage",
    )
    require(provenance["no_policy_grid_search"] is True, "policy search detected")
    require(provenance["official_validation_used"] is False, "Validation used")
    require(provenance["official_test_used"] is False, "Test used")

    pool = torch.load(inner_pool_path, map_location="cpu")
    require(
        pool["provenance"]["all_rows_are_unseen_by_their_expert_stack"] is True,
        "inner pool provenance",
    )
    require(len(pool["sample_ids"]) == len(set(pool["sample_ids"])), "duplicate inner IDs")
    require(bool((pool["fold_index"] >= 0).all()), "inner fold assignment incomplete")
    outer = pd.read_csv(outer_predictions_path)
    require(not outer["sample_id"].astype(str).duplicated().any(), "duplicate outer IDs")
    require(set(pool["sample_ids"]).isdisjoint(set(outer["sample_id"])), "inner/outer overlap")
    require(len(pd.read_csv(inner_metrics_path)) >= 3, "too few inner router folds")
    labels = torch.tensor(outer["label"].to_numpy()).float()
    anchor = torch.tensor(outer["anchor_prediction"].to_numpy()).float()
    selected = torch.tensor(outer["selected_prediction"].to_numpy()).float()
    anchor_mae = float(torch.abs(anchor - labels).mean().item())
    selected_mae = float(torch.abs(selected - labels).mean().item())
    require(
        abs(anchor_mae - summary["outer_holdout_metrics"]["anchor_mae"]) < 2e-6,
        "anchor MAE mismatch",
    )
    require(
        abs(selected_mae - summary["outer_holdout_metrics"]["mae"]) < 2e-6,
        "router MAE mismatch",
    )
    print("V9.19 OUTER FOLD ENGINEERING AUDIT PASSED")
    print("outer_fold:", fold)
    print("inner_router_accepted:", summary["inner_router_accepted"])
    print("inner_oof_gain:", f"{summary['inner_oof_metrics']['gain_vs_anchor']:+.6f}")
    print(
        "outer_holdout_gain:",
        f"{summary['outer_holdout_metrics']['gain_vs_anchor']:+.6f}",
    )


def audit_aggregate(root: Path):
    summary_path = root / "full_same_stack_nested_crossfit_v919_summary.json"
    predictions_path = root / "v919_all_outer_holdout_predictions.csv"
    folds_path = root / "v919_outer_fold_metrics.csv"
    for path in (summary_path, predictions_path, folds_path):
        require(path.is_file(), f"missing aggregate artifact: {path}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(predictions_path)
    folds = pd.read_csv(folds_path)
    require(
        not predictions["sample_id"].astype(str).duplicated().any(),
        "duplicate aggregate IDs",
    )
    require(len(folds) == summary["outer_fold_count"], "outer fold count mismatch")
    require(summary["provenance"]["official_validation_used"] is False, "Validation used")
    require(summary["provenance"]["official_test_used"] is False, "Test used")
    require(
        summary["provenance"]["no_posthoc_fold_selection"] is True,
        "fold selection detected",
    )
    print("V9.19 AGGREGATE ENGINEERING AUDIT PASSED")
    print(
        "anchor/router/gain:",
        f"{summary['anchor_mae']:.6f}",
        f"{summary['router_mae']:.6f}",
        f"{summary['gain_vs_anchor']:+.6f}",
    )
    print("positive_outer_folds:", summary["positive_outer_folds"])
    print("stop_gate_passed:", summary["stop_gate"]["passed"])


def main():
    cli = parse_args()
    root = Path(cli.root)
    if cli.aggregate:
        audit_aggregate(root)
    elif cli.outer_fold is not None:
        audit_fold(root, cli.outer_fold)
    else:
        raise SystemExit("Specify --outer-fold or --aggregate")


if __name__ == "__main__":
    main()
