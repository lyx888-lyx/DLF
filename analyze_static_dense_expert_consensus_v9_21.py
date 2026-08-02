"""Strict outer-holdout audit of static dense expert consensus for V9.21."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from trains.singleTask.model.GlobalDenseExpertConsensusV921 import MODEL_VERSION
from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES
from trains.singleTask.no_train_decomposition_v920 import PoolView, normalize_pool
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.static_dense_expert_consensus_v921 import (
    AUDIT_VERSION,
    PRIMARY_STRATEGY,
    ConsensusConfigV921,
    aggregate_prediction_frames,
    group_bootstrap_gain_interval,
    inner_crossfit_consensus,
    residual_correlation_rows,
    strategy_predictions,
    success_gate,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Fit one global convex expert weight vector on each V9.19 inner OOF "
            "pool and evaluate it unchanged on that fold's outer holdout."
        )
    )
    parser.add_argument(
        "--root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--shrinkage-lambda", type=float, default=0.01)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1111)
    return parser.parse_args()


def scalar_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"prediction", "sample_gain"}
    }


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    values = frame.loc[:, columns].copy()
    for column in values.columns:
        if pd.api.types.is_float_dtype(values[column]):
            values[column] = values[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    header = "|" + "|".join(columns) + "|"
    separator = "|" + "|".join(["---"] * len(columns)) + "|"
    rows = [
        "|" + "|".join(str(value) for value in row) + "|"
        for row in values.itertuples(index=False, name=None)
    ]
    return "\n".join([header, separator, *rows])


def combined_pool(pools: list[PoolView]) -> PoolView:
    return PoolView(
        sample_ids=[value for pool in pools for value in pool.sample_ids],
        group_ids=[value for pool in pools for value in pool.group_ids],
        labels=torch.cat([pool.labels for pool in pools], dim=0),
        actions=torch.cat([pool.actions for pool in pools], dim=0),
        regions=torch.cat([pool.regions for pool in pools], dim=0),
    )


def main():
    cli = parse_args()
    root = Path(cli.root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v921_static_dense_expert_consensus"
    )
    output.mkdir(parents=True, exist_ok=True)
    config = ConsensusConfigV921(
        shrinkage_lambda=float(cli.shrinkage_lambda),
        bootstrap_repetitions=int(cli.bootstrap_repetitions),
        bootstrap_seed=int(cli.bootstrap_seed),
    )

    fold_metric_rows = []
    inner_metric_rows = []
    weight_rows = []
    sample_frames = []
    outer_pools: list[PoolView] = []

    for fold in range(int(cli.outer_folds)):
        fold_dir = root / f"outer_fold_{fold}"
        inner_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        outer_path = (
            fold_dir
            / "outer_deployment_stack"
            / "same_stack_target_pool_v919.pth"
        )
        summary_path = fold_dir / "same_stack_nested_crossfit_v919_fold_summary.json"
        for path in (inner_path, outer_path, summary_path):
            if not path.is_file():
                raise FileNotFoundError(path)

        inner_payload = torch.load(inner_path, map_location="cpu")
        outer_payload = torch.load(outer_path, map_location="cpu")
        inner_pool = normalize_pool(inner_payload)
        outer_pool = normalize_pool(outer_payload)
        if set(inner_pool.sample_ids) & set(outer_pool.sample_ids):
            raise RuntimeError(f"outer fold {fold} has inner/outer sample overlap")

        inner_cv = inner_crossfit_consensus(inner_payload, config)
        for row in inner_cv["fold_rows"]:
            inner_metric_rows.append({"outer_fold": fold, **row})
        for strategy, metrics in inner_cv["aggregate"].items():
            inner_metric_rows.append(
                {
                    "outer_fold": fold,
                    "inner_fold": "aggregate",
                    "strategy": strategy,
                    "selected_single_action": "cross_fitted",
                    **scalar_metrics(metrics),
                }
            )

        result = strategy_predictions(inner_pool, outer_pool, config)
        for strategy, metrics in result["metrics"].items():
            fold_metric_rows.append(
                {
                    "outer_fold": fold,
                    "strategy": strategy,
                    "inner_selected_single_action": result[
                        "selected_single_action"
                    ],
                    **scalar_metrics(metrics),
                }
            )

        for strategy, weights in result["weights"].items():
            for action, value in zip(ACTION_NAMES, weights.tolist()):
                weight_rows.append(
                    {
                        "outer_fold": fold,
                        "strategy": strategy,
                        "action": action,
                        "weight": float(value),
                    }
                )

        checkpoint = {
            "version": AUDIT_VERSION,
            "model_version": MODEL_VERSION,
            "outer_fold": fold,
            "action_names": list(ACTION_NAMES),
            "config": asdict(config),
            "inner_selected_single_index": result["selected_single_index"],
            "inner_selected_single_action": result["selected_single_action"],
            "weights": {
                name: torch.tensor(value, dtype=torch.float32)
                for name, value in result["weights"].items()
            },
            "provenance": {
                "weights_fit_on_inner_oof_only": True,
                "outer_holdout_used_for_weight_fitting": False,
                "sample_dependent_weights": False,
                "router_or_gate_present": False,
            },
        }
        torch.save(checkpoint, output / f"outer_fold_{fold}_consensus_v921.pth")

        frame = pd.DataFrame(
            {
                "outer_fold": fold,
                "sample_id": outer_pool.sample_ids,
                "group_id": outer_pool.group_ids,
                "label": outer_pool.labels.numpy(),
                "inner_selected_single_action": result[
                    "selected_single_action"
                ],
            }
        )
        for action_index, action in enumerate(ACTION_NAMES):
            frame[f"action_prediction_{action}"] = outer_pool.actions[
                :, action_index
            ].numpy()
        for strategy, prediction in result["predictions"].items():
            frame[f"prediction_{strategy}"] = prediction.numpy()
        sample_frames.append(frame)
        outer_pools.append(outer_pool)

    fold_metrics = pd.DataFrame(fold_metric_rows).sort_values(
        ["outer_fold", "strategy"]
    )
    inner_metrics = pd.DataFrame(inner_metric_rows)
    weights = pd.DataFrame(weight_rows).sort_values(
        ["strategy", "outer_fold", "action"]
    )
    samples = pd.concat(sample_frames, ignore_index=True)
    if samples["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("outer holdout aggregation contains duplicate sample IDs")

    aggregate = aggregate_prediction_frames(samples)
    aggregate_rows = [
        {"strategy": strategy, **metrics}
        for strategy, metrics in aggregate.items()
    ]
    aggregate_frame = pd.DataFrame(aggregate_rows).sort_values("mae")

    bootstrap_rows = []
    for strategy in aggregate:
        prediction = torch.tensor(
            samples[f"prediction_{strategy}"].to_numpy(), dtype=torch.float32
        )
        labels = torch.tensor(samples["label"].to_numpy(), dtype=torch.float32)
        anchor = torch.tensor(
            samples["prediction_anchor"].to_numpy(), dtype=torch.float32
        )
        gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
        interval = group_bootstrap_gain_interval(
            gain,
            samples["group_id"].tolist(),
            config.bootstrap_repetitions,
            config.bootstrap_seed + 1009 * (len(bootstrap_rows) + 1),
        )
        bootstrap_rows.append({"strategy": strategy, **interval})
    bootstrap = pd.DataFrame(bootstrap_rows).sort_values("strategy")

    pool = combined_pool(outer_pools)
    signed_correlation = pd.DataFrame(
        residual_correlation_rows(pool.actions, pool.labels, "signed_residual")
    )
    absolute_correlation = pd.DataFrame(
        residual_correlation_rows(pool.actions, pool.labels, "absolute_error")
    )

    stability_rows = []
    for strategy in sorted(weights["strategy"].unique()):
        strategy_frame = weights[weights["strategy"] == strategy]
        for action in ACTION_NAMES:
            values = strategy_frame[strategy_frame["action"] == action]["weight"]
            stability_rows.append(
                {
                    "strategy": strategy,
                    "action": action,
                    "mean_weight": float(values.mean()),
                    "std_weight": float(values.std(ddof=0)),
                    "min_weight": float(values.min()),
                    "max_weight": float(values.max()),
                    "range_weight": float(values.max() - values.min()),
                }
            )
    stability = pd.DataFrame(stability_rows)

    gate = success_gate(fold_metrics, aggregate, config)
    primary = aggregate[PRIMARY_STRATEGY]
    selected = aggregate["inner_selected_single"]
    comparisons = {
        "gain_vs_anchor": float(primary["gain_vs_anchor"]),
        "mae_improvement_vs_inner_selected_single": float(
            selected["mae"] - primary["mae"]
        ),
        "mae_improvement_vs_fixed_positive": float(
            aggregate["fixed_positive"]["mae"] - primary["mae"]
        ),
        "mae_improvement_vs_fixed_boundary": float(
            aggregate["fixed_boundary"]["mae"] - primary["mae"]
        ),
        "mae_improvement_vs_boundary_positive_mean": float(
            aggregate["boundary_positive_mean"]["mae"] - primary["mae"]
        ),
    }
    summary = {
        "version": AUDIT_VERSION,
        "method": "strict_nested_static_global_convex_expert_consensus",
        "model_version": MODEL_VERSION,
        "root": str(root),
        "output_dir": str(output),
        "sample_count": len(samples),
        "outer_fold_count": int(cli.outer_folds),
        "primary_strategy": PRIMARY_STRATEGY,
        "config": asdict(config),
        "aggregate_metrics": aggregate,
        "primary_comparisons": comparisons,
        "success_gate": gate,
        "inner_selected_actions": {
            str(fold): str(
                fold_metrics[fold_metrics["outer_fold"] == fold][
                    "inner_selected_single_action"
                ].iloc[0]
            )
            for fold in range(int(cli.outer_folds))
        },
        "provenance": {
            "uses_only_v919_saved_predictions": True,
            "new_anchor_or_expert_models_trained": False,
            "weights_fit_on_inner_oof_only": True,
            "outer_holdouts_used_only_for_evaluation": True,
            "official_validation_loaded": False,
            "official_test_loaded": False,
            "sample_dependent_expert_weights": False,
            "router_region_model_or_gate_used": False,
            "primary_regularization_pre_registered": True,
            "outer_results_not_used_to_select_between_convex_variants": True,
        },
    }

    fold_metrics.to_csv(output / "v921_strategy_metrics_by_fold.csv", index=False)
    inner_metrics.to_csv(output / "v921_inner_crossfit_metrics.csv", index=False)
    weights.to_csv(output / "v921_convex_weights_by_fold.csv", index=False)
    stability.to_csv(output / "v921_weight_stability.csv", index=False)
    samples.to_csv(output / "v921_outer_predictions.csv", index=False)
    aggregate_frame.to_csv(output / "v921_aggregate_strategy_metrics.csv", index=False)
    bootstrap.to_csv(output / "v921_group_bootstrap_gain_ci.csv", index=False)
    signed_correlation.to_csv(output / "v921_signed_residual_correlation.csv", index=False)
    absolute_correlation.to_csv(
        output / "v921_absolute_error_correlation.csv", index=False
    )
    (output / "v921_static_consensus_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    report_columns = [
        "strategy",
        "mae",
        "gain_vs_anchor",
        "win_rate",
        "large_harm_rate_010",
    ]
    fold_primary = fold_metrics[
        fold_metrics["strategy"].isin(
            [
                PRIMARY_STRATEGY,
                "inner_selected_single",
                "boundary_positive_mean",
                "fixed_positive",
                "fixed_boundary",
            ]
        )
    ].copy()
    report = [
        "# V9.21 Static Dense Expert Consensus Audit",
        "",
        "All outer-fold weights are global and frozen; no sample-dependent routing is used.",
        "",
        "## Aggregate strategies",
        "",
        markdown_table(aggregate_frame, report_columns),
        "",
        "## Key strategies by outer fold",
        "",
        markdown_table(
            fold_primary,
            [
                "outer_fold",
                "strategy",
                "inner_selected_single_action",
                "mae",
                "gain_vs_anchor",
                "large_harm_rate_010",
            ],
        ),
        "",
        "## Primary comparison",
        "",
        f"- Primary strategy: `{PRIMARY_STRATEGY}`",
        f"- Gain versus Anchor: `{comparisons['gain_vs_anchor']:+.6f}`",
        f"- MAE improvement versus inner-selected single model: `{comparisons['mae_improvement_vs_inner_selected_single']:+.6f}`",
        f"- MAE improvement versus P+B mean: `{comparisons['mae_improvement_vs_boundary_positive_mean']:+.6f}`",
        f"- Pre-registered gate passed: `{gate['passed']}`",
        "",
        "The unregularized convex result is diagnostic only; outer results do not select it over the pre-registered shrinkage strategy.",
    ]
    (output / "v921_static_consensus_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.21 STATIC DENSE EXPERT CONSENSUS COMPLETE")
    print("new expert models trained: False")
    print("sample-dependent routing: False")
    print("primary strategy:", PRIMARY_STRATEGY)
    print(
        "anchor/selected-single/primary MAE:",
        f"{aggregate['anchor']['mae']:.6f}",
        f"{aggregate['inner_selected_single']['mae']:.6f}",
        f"{primary['mae']:.6f}",
    )
    print(
        "primary gain vs anchor / selected single:",
        f"{primary['gain_vs_anchor']:+.6f}",
        f"{comparisons['mae_improvement_vs_inner_selected_single']:+.6f}",
    )
    print("success gate passed:", gate["passed"])
    print("report:", output / "v921_static_consensus_report.md")


if __name__ == "__main__":
    main()
