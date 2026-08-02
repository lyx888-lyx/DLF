"""Decompose V9.19 failures using saved artifacts only; trains no model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from trains.singleTask.no_train_decomposition_reporting_v920 import (
    aggregate_strategies,
    align_frame,
    build_outer_sample_frame,
    diagnose,
    mapping_rows,
    markdown_table,
    weighted_region_consistency,
)
from trains.singleTask.no_train_decomposition_v920 import (
    AUDIT_VERSION,
    PoolView,
    action_metric_rows,
    actual_gated_metrics,
    best_action_by_region,
    counterfactual_router,
    normalize_pool,
    oracle_metrics,
    predicted_gain_bin_rows,
    region_action_rows,
    routing_diagnostics,
    select_by_region,
)
from trains.singleTask.region_cost_reporting_v918 import jsonable


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--gain-bins", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def main():
    cli = parse_args()
    root = Path(cli.root)
    output = Path(cli.output_dir) if cli.output_dir else root / "v920_no_train_decomposition"
    output.mkdir(parents=True, exist_ok=True)

    fold_rows, action_rows, region_rows = [], [], []
    mapping_records, calibration_rows, sample_frames, outer_pools = [], [], [], []

    for fold in range(cli.outer_folds):
        fold_dir = root / f"outer_fold_{fold}"
        summary_path = fold_dir / "same_stack_nested_crossfit_v919_fold_summary.json"
        inner_pool_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        outer_pool_path = (
            fold_dir / "outer_deployment_stack" / "same_stack_target_pool_v919.pth"
        )
        inner_prediction_path = fold_dir / "inner_oof_router_predictions.csv"
        outer_prediction_path = fold_dir / "outer_holdout_predictions.csv"
        for path in (
            summary_path,
            inner_pool_path,
            outer_pool_path,
            inner_prediction_path,
            outer_prediction_path,
        ):
            if not path.is_file():
                raise FileNotFoundError(path)

        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        inner_pool = normalize_pool(torch.load(inner_pool_path, map_location="cpu"))
        outer_pool = normalize_pool(torch.load(outer_pool_path, map_location="cpu"))
        align_frame(pd.read_csv(inner_prediction_path), inner_pool)
        outer_frame = align_frame(pd.read_csv(outer_prediction_path), outer_pool)

        inner_oracle = oracle_metrics(inner_pool)
        outer_oracle = oracle_metrics(outer_pool)
        development_mapping = best_action_by_region(inner_pool)
        dev_locked_region = select_by_region(outer_pool, development_mapping)
        actual = actual_gated_metrics(outer_frame)
        counterfactual = counterfactual_router(outer_frame, summary)
        diagnostics = routing_diagnostics(outer_frame, counterfactual)

        action_rows += action_metric_rows(inner_pool, "inner_oof", fold)
        action_rows += action_metric_rows(outer_pool, "outer_holdout", fold)
        region_rows += region_action_rows(inner_pool, "inner_oof", fold)
        region_rows += region_action_rows(outer_pool, "outer_holdout", fold)
        mapping_records += mapping_rows(
            fold, inner_pool, outer_pool, development_mapping
        )
        calibration_rows += predicted_gain_bin_rows(
            outer_frame, counterfactual, fold, bins=cli.gain_bins
        )
        sample_frames.append(
            build_outer_sample_frame(
                outer_frame,
                outer_pool,
                fold,
                development_mapping,
                outer_oracle["posthoc_region_mapping"],
                counterfactual,
                actual,
            )
        )
        outer_pools.append(outer_pool)
        fold_rows.append(
            {
                "outer_fold": fold,
                "sample_count": len(outer_pool.labels),
                "anchor_mae": actual["anchor_mae"],
                "inner_oof_router_gain": summary["inner_oof_metrics"][
                    "gain_vs_anchor"
                ],
                "inner_router_accepted": bool(summary["inner_router_accepted"]),
                "inner_sample_oracle_gain": inner_oracle["sample_oracle"][
                    "gain_vs_anchor"
                ],
                "outer_sample_oracle_gain": outer_oracle["sample_oracle"][
                    "gain_vs_anchor"
                ],
                "outer_semantic_region_oracle_gain": outer_oracle[
                    "semantic_region_oracle"
                ]["gain_vs_anchor"],
                "outer_dev_locked_region_oracle_gain": dev_locked_region[
                    "gain_vs_anchor"
                ],
                "outer_posthoc_region_oracle_gain": outer_oracle[
                    "posthoc_region_oracle"
                ]["gain_vs_anchor"],
                "outer_counterfactual_router_gain": counterfactual[
                    "frozen_policy"
                ]["gain_vs_anchor"],
                "outer_counterfactual_router_coverage": counterfactual[
                    "frozen_policy"
                ]["coverage"],
                "outer_counterfactual_router_precision": counterfactual[
                    "frozen_policy"
                ]["trigger_precision"],
                "outer_counterfactual_router_harm": counterfactual[
                    "frozen_policy"
                ]["harm_over_010_rate"],
                "outer_expected_cost_argmin_gain": counterfactual[
                    "expected_cost_argmin"
                ]["gain_vs_anchor"],
                "outer_actual_gated_gain": actual["gain_vs_anchor"],
                "gain_cutoff": counterfactual["gain_cutoff"],
                **diagnostics,
            }
        )

    folds = pd.DataFrame(fold_rows).sort_values("outer_fold")
    actions = pd.DataFrame(action_rows)
    regions = pd.DataFrame(region_rows)
    mappings = pd.DataFrame(mapping_records)
    calibration = pd.DataFrame(calibration_rows)
    samples = pd.concat(sample_frames, ignore_index=True)
    if samples["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("outer sample aggregation contains duplicate IDs")

    combined = PoolView(
        sample_ids=[item for pool in outer_pools for item in pool.sample_ids],
        group_ids=[item for pool in outer_pools for item in pool.group_ids],
        labels=torch.cat([pool.labels for pool in outer_pools]),
        actions=torch.cat([pool.actions for pool in outer_pools]),
        regions=torch.cat([pool.regions for pool in outer_pools]),
    )
    aggregate_actions = pd.DataFrame(action_metric_rows(combined, "outer_all", -1))
    aggregate_regions = pd.DataFrame(region_action_rows(combined, "outer_all", -1))
    consistency = weighted_region_consistency(
        regions[regions["split"] == "outer_holdout"]
    )
    strategies = aggregate_strategies(samples)
    decomposition, best_single = diagnose(folds, strategies, aggregate_actions)

    summary = {
        "version": AUDIT_VERSION,
        "method": "saved_artifact_decomposition_without_training",
        "root": str(root),
        "models_trained": False,
        "official_validation_loaded": False,
        "official_test_loaded": False,
        "sample_count": len(samples),
        "outer_fold_count": cli.outer_folds,
        "aggregate_strategies": strategies,
        "best_single_action": best_single,
        "decomposition": decomposition,
        "fold_metrics": fold_rows,
        "interpretation_rules": {
            "counterfactual_stop_gate": {
                "mean_gain_at_least": 0.005,
                "positive_outer_folds_at_least": 4,
                "harm_over_010_rate_at_most": 0.05,
            },
            "note": (
                "True-region and sample-oracle metrics use labels only for "
                "diagnosis and are not deployable scores."
            ),
        },
        "provenance": {
            "uses_only_v919_saved_pools_predictions_and_summaries": True,
            "no_optimizer_or_backward_pass": True,
            "outer_counterfactual_uses_the_frozen_v919_threshold": True,
            "development_region_mapping_is_fitted_on_inner_oof_only": True,
            "outer_labels_are_used_for_diagnostic_reporting_only": True,
        },
    }

    outputs = {
        "v920_fold_decomposition.csv": folds,
        "v920_action_metrics_by_fold.csv": actions,
        "v920_outer_action_metrics_aggregate.csv": aggregate_actions,
        "v920_region_action_metrics_by_fold.csv": regions,
        "v920_outer_region_action_aggregate.csv": aggregate_regions,
        "v920_outer_region_action_consistency.csv": consistency,
        "v920_development_locked_region_mappings.csv": mappings,
        "v920_predicted_gain_calibration_bins.csv": calibration,
        "v920_outer_sample_decomposition.csv": samples,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)
    (output / "v920_decomposition_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    strategy_table = pd.DataFrame(
        [
            {
                "strategy": name,
                "mae": value["mae"],
                "gain": value["gain_vs_anchor"],
                "coverage": value["coverage"],
                "harm": value["harm_over_010_rate"],
            }
            for name, value in strategies.items()
        ]
    )
    report = [
        "# V9.20 No-Training Decomposition Audit",
        "",
        "No model is trained; only saved V9.19 artifacts are read.",
        "",
        "## Aggregate strategy ladder",
        "",
        markdown_table(strategy_table, ["strategy", "mae", "gain", "coverage", "harm"]),
        "",
        "## Fold decomposition",
        "",
        markdown_table(
            folds,
            [
                "outer_fold",
                "anchor_mae",
                "outer_sample_oracle_gain",
                "outer_dev_locked_region_oracle_gain",
                "outer_counterfactual_router_gain",
                "outer_counterfactual_router_harm",
                "predicted_gain_spearman",
            ],
        ),
        "",
        "## Evidence summary",
        "",
        f"- Primary bottleneck: `{decomposition['primary_bottleneck']}`",
        f"- Gate verdict: `{decomposition['gate_verdict']}`",
        f"- Sample-oracle gain: `{decomposition['sample_oracle_capacity_gain']:+.6f}`",
        f"- Development-locked region gain: `{decomposition['development_locked_true_region_gain']:+.6f}`",
        f"- Counterfactual router gain: `{decomposition['counterfactual_frozen_router_gain']:+.6f}`",
        f"- Actual gated gain: `{decomposition['actual_gated_router_gain']:+.6f}`",
        f"- Best single action: `{best_single['action']}` with gain `{best_single['gain_vs_anchor']:+.6f}`",
        "",
        "Oracle rows are diagnostic upper bounds, not deployment results.",
    ]
    (output / "v920_decomposition_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.20 NO-TRAINING DECOMPOSITION COMPLETE")
    print("models trained: False")
    print("primary bottleneck:", decomposition["primary_bottleneck"])
    print("gate verdict:", decomposition["gate_verdict"])
    print(
        "capacity/region/router/actual gains:",
        f"{decomposition['sample_oracle_capacity_gain']:+.6f}",
        f"{decomposition['development_locked_true_region_gain']:+.6f}",
        f"{decomposition['counterfactual_frozen_router_gain']:+.6f}",
        f"{decomposition['actual_gated_router_gain']:+.6f}",
    )
    print("report:", output / "v920_decomposition_report.md")


if __name__ == "__main__":
    main()
