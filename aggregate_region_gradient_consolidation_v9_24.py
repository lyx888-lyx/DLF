"""Aggregate strict V9.24 outer-holdout results and apply the fixed gate."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.region_gradient_consolidation_v924 import (
    PRIMARY_STRATEGY,
    STRATEGIES,
    TRAINING_VERSION,
    SuccessGateConfigV924,
    paired_group_bootstrap_difference,
    regression_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1111)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def _scalar_metrics(metrics):
    result = {
        key: value
        for key, value in metrics.items()
        if key != "region_mae"
    }
    for name, value in metrics["region_mae"].items():
        result[f"region_mae_{name}"] = float(value)
    return result


def _metrics_from_frame(frame: pd.DataFrame, strategy: str):
    labels = torch.tensor(
        frame["label"].to_numpy(), dtype=torch.float32
    )
    prediction = torch.tensor(
        frame[f"prediction_{strategy}"].to_numpy(),
        dtype=torch.float32,
    )
    anchor = torch.tensor(
        frame["prediction_anchor"].to_numpy(),
        dtype=torch.float32,
    )
    return regression_metrics(prediction, labels, anchor=anchor)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    local = frame.loc[:, columns].copy()
    for column in local.columns:
        if pd.api.types.is_float_dtype(local[column]):
            local[column] = local[column].map(
                lambda value: (
                    "" if pd.isna(value) else f"{float(value):.6f}"
                )
            )
    return "\n".join(
        [
            "|" + "|".join(columns) + "|",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in local.itertuples(index=False, name=None)
            ],
        ]
    )


def _optional_v921(frame: pd.DataFrame, root: Path):
    path = (
        root
        / "v921_static_dense_expert_consensus"
        / "v921_outer_predictions.csv"
    )
    if not path.is_file():
        return frame, None
    v921 = pd.read_csv(
        path,
        usecols=[
            "outer_fold",
            "sample_id",
            "prediction_convex_shrinkage",
        ],
    )
    v921["sample_id"] = v921["sample_id"].astype(str)
    merged = frame.merge(
        v921,
        on=["outer_fold", "sample_id"],
        how="left",
        validate="one_to_one",
    )
    if merged["prediction_convex_shrinkage"].isna().any():
        raise RuntimeError("V9.21/V9.24 sample alignment mismatch")
    labels = torch.tensor(
        merged["label"].to_numpy(), dtype=torch.float32
    )
    prediction = torch.tensor(
        merged["prediction_convex_shrinkage"].to_numpy(),
        dtype=torch.float32,
    )
    anchor = torch.tensor(
        merged["prediction_anchor"].to_numpy(),
        dtype=torch.float32,
    )
    metrics = regression_metrics(prediction, labels, anchor=anchor)
    return merged, metrics


def main():
    cli = parse_args()
    root = Path(cli.v919_root)
    experiment_root = root / "v924_region_gradient_consolidation"
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else experiment_root
    )
    output.mkdir(parents=True, exist_ok=True)
    gate = SuccessGateConfigV924()

    fold_frames = []
    fold_rows = []
    for fold in range(int(cli.outer_folds)):
        fold_dir = experiment_root / f"outer_fold_{fold}"
        summary_path = fold_dir / "v924_fold_summary.json"
        predictions_path = (
            fold_dir / "v924_outer_holdout_predictions.csv"
        )
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        if not predictions_path.is_file():
            raise FileNotFoundError(predictions_path)
        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        if summary["version"] != TRAINING_VERSION:
            raise RuntimeError("V9.24 fold version mismatch")
        frame = pd.read_csv(predictions_path)
        frame.insert(0, "outer_fold", int(fold))
        frame["sample_id"] = frame["sample_id"].astype(str)
        fold_frames.append(frame)
        for strategy in ("anchor", *STRATEGIES):
            if strategy == "anchor":
                labels = torch.tensor(
                    frame["label"].to_numpy(),
                    dtype=torch.float32,
                )
                anchor = torch.tensor(
                    frame["prediction_anchor"].to_numpy(),
                    dtype=torch.float32,
                )
                metrics = regression_metrics(
                    anchor,
                    labels,
                    anchor=anchor,
                )
            else:
                metrics = _metrics_from_frame(frame, strategy)
            fold_rows.append(
                {
                    "outer_fold": fold,
                    "strategy": strategy,
                    **_scalar_metrics(metrics),
                }
            )

    samples = pd.concat(fold_frames, ignore_index=True)
    if samples["sample_id"].duplicated().any():
        raise RuntimeError(
            "aggregate outer holdouts contain duplicate IDs"
        )
    samples, v921_metrics = _optional_v921(samples, root)

    aggregate_rows = []
    aggregate_metrics = {}
    for strategy in ("anchor", *STRATEGIES):
        if strategy == "anchor":
            labels = torch.tensor(
                samples["label"].to_numpy(),
                dtype=torch.float32,
            )
            anchor = torch.tensor(
                samples["prediction_anchor"].to_numpy(),
                dtype=torch.float32,
            )
            metrics = regression_metrics(
                anchor,
                labels,
                anchor=anchor,
            )
        else:
            metrics = _metrics_from_frame(samples, strategy)
        aggregate_metrics[strategy] = _scalar_metrics(metrics)
        aggregate_rows.append(
            {"strategy": strategy, **_scalar_metrics(metrics)}
        )
    if v921_metrics is not None:
        aggregate_metrics["v921_convex_shrinkage"] = (
            _scalar_metrics(v921_metrics)
        )
        aggregate_rows.append(
            {
                "strategy": "v921_convex_shrinkage",
                **_scalar_metrics(v921_metrics),
            }
        )

    fold_metrics = pd.DataFrame(fold_rows).sort_values(
        ["outer_fold", "strategy"]
    )
    aggregate_frame = pd.DataFrame(aggregate_rows).sort_values(
        "mae"
    )

    comparison_rows = []
    for fold in range(int(cli.outer_folds)):
        local = fold_metrics[
            fold_metrics["outer_fold"] == fold
        ].set_index("strategy")
        mgda = local.loc[PRIMARY_STRATEGY]
        global_control = local.loc["global_mae"]
        comparison_rows.append(
            {
                "outer_fold": fold,
                "mgda_mae": float(mgda["mae"]),
                "global_control_mae": float(
                    global_control["mae"]
                ),
                "gain_vs_global_control": float(
                    global_control["mae"] - mgda["mae"]
                ),
                "mgda_macro_region_mae": float(
                    mgda["macro_region_mae"]
                ),
                "global_control_macro_region_mae": float(
                    global_control["macro_region_mae"]
                ),
                "macro_region_gain_vs_global": float(
                    global_control["macro_region_mae"]
                    - mgda["macro_region_mae"]
                ),
                "strong_negative_gain_vs_global": float(
                    global_control[
                        "region_mae_strong_negative"
                    ]
                    - mgda["region_mae_strong_negative"]
                ),
                "mgda_nondegrading": bool(
                    float(mgda["mae"])
                    <= float(global_control["mae"])
                ),
            }
        )
    comparisons = pd.DataFrame(comparison_rows)

    primary = aggregate_metrics[PRIMARY_STRATEGY]
    global_control = aggregate_metrics["global_mae"]
    gain_vs_global = float(
        global_control["mae"] - primary["mae"]
    )
    macro_gain = float(
        global_control["macro_region_mae"]
        - primary["macro_region_mae"]
    )
    strong_negative_gain = float(
        global_control["region_mae_strong_negative"]
        - primary["region_mae_strong_negative"]
    )
    positive_folds = int(
        (comparisons["gain_vs_global_control"] > 0.0).sum()
    )
    worst_degradation = float(
        max(0.0, -comparisons["gain_vs_global_control"].min())
    )
    passed = bool(
        gain_vs_global >= float(gate.required_gain_vs_global)
        and positive_folds
        >= int(gate.required_positive_outer_folds)
        and worst_degradation
        <= float(gate.maximum_worst_fold_degradation)
        and (
            not gate.require_macro_region_improvement
            or macro_gain > 0.0
        )
        and (
            not gate.require_strong_negative_nondegradation
            or strong_negative_gain >= 0.0
        )
    )

    bootstrap_rows = []
    for baseline in (
        "anchor",
        "global_mae",
        "equal_region_mean",
    ):
        bootstrap_rows.append(
            {
                "candidate": PRIMARY_STRATEGY,
                "baseline": baseline,
                **paired_group_bootstrap_difference(
                    samples,
                    candidate_column=(
                        f"prediction_{PRIMARY_STRATEGY}"
                    ),
                    baseline_column=f"prediction_{baseline}",
                    repetitions=int(cli.bootstrap_repetitions),
                    seed=int(cli.bootstrap_seed)
                    + 1009 * (len(bootstrap_rows) + 1),
                ),
            }
        )
    if "prediction_convex_shrinkage" in samples.columns:
        bootstrap_rows.append(
            {
                "candidate": PRIMARY_STRATEGY,
                "baseline": "v921_convex_shrinkage",
                **paired_group_bootstrap_difference(
                    samples,
                    candidate_column=(
                        f"prediction_{PRIMARY_STRATEGY}"
                    ),
                    baseline_column=(
                        "prediction_convex_shrinkage"
                    ),
                    repetitions=int(cli.bootstrap_repetitions),
                    seed=int(cli.bootstrap_seed) + 7001,
                ),
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)

    summary = {
        "version": TRAINING_VERSION,
        "method": "strict_nested_region_gradient_consolidation",
        "primary_strategy": PRIMARY_STRATEGY,
        "outer_fold_count": int(cli.outer_folds),
        "sample_count": int(len(samples)),
        "aggregate_metrics": aggregate_metrics,
        "success_gate_config": asdict(gate),
        "success_gate": {
            "passed": passed,
            "aggregate_gain_vs_global_control": gain_vs_global,
            "positive_outer_folds": positive_folds,
            "worst_fold_degradation": worst_degradation,
            "macro_region_gain_vs_global_control": macro_gain,
            "strong_negative_gain_vs_global_control": (
                strong_negative_gain
            ),
        },
        "comparison_to_v921": (
            None
            if v921_metrics is None
            else {
                "v921_mae": float(v921_metrics["mae"]),
                "mgda_mae": float(primary["mae"]),
                "mgda_improvement_vs_v921": float(
                    v921_metrics["mae"] - primary["mae"]
                ),
            }
        ),
        "provenance": {
            "outer_results_not_used_for_training_or_checkpoint_selection": True,
            "same_pre_registered_training_config_all_folds": True,
            "primary_strategy_pre_registered": True,
            "router_or_fusion_head_used": False,
            "expert_predictions_or_distillation_used": False,
            "official_validation_or_test_used": False,
        },
    }

    fold_metrics.to_csv(
        output / "v924_strategy_metrics_by_fold.csv",
        index=False,
    )
    aggregate_frame.to_csv(
        output / "v924_aggregate_strategy_metrics.csv",
        index=False,
    )
    comparisons.to_csv(
        output / "v924_fold_primary_comparisons.csv",
        index=False,
    )
    bootstrap.to_csv(
        output / "v924_paired_group_bootstrap.csv",
        index=False,
    )
    samples.to_csv(
        output / "v924_outer_predictions.csv",
        index=False,
    )
    (output / "v924_aggregate_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    report = [
        "# V9.24 Region-Gradient Consolidation",
        "",
        "One scalar-output CFCompat model is trained per strategy and outer "
        "fold. Only the pre-registered fusion tail is updated; no router, "
        "expert prediction, distillation target, or consensus head is used.",
        "",
        "## Aggregate strategies",
        "",
        _markdown_table(
            aggregate_frame,
            [
                "strategy",
                "mae",
                "macro_region_mae",
                "gain_vs_anchor",
                "large_harm_rate_010",
            ],
        ),
        "",
        "## Primary comparison by outer fold",
        "",
        _markdown_table(
            comparisons,
            [
                "outer_fold",
                "mgda_mae",
                "global_control_mae",
                "gain_vs_global_control",
                "macro_region_gain_vs_global",
                "strong_negative_gain_vs_global",
            ],
        ),
        "",
        "## Fixed success gate",
        "",
        f"- Passed: `{passed}`",
        f"- Aggregate gain versus global control: "
        f"`{gain_vs_global:+.6f}`",
        f"- Positive outer folds: "
        f"`{positive_folds}/{int(cli.outer_folds)}`",
        f"- Worst fold degradation: "
        f"`{worst_degradation:.6f}`",
        f"- Macro-region gain: `{macro_gain:+.6f}`",
        f"- Strong-negative gain: "
        f"`{strong_negative_gain:+.6f}`",
    ]
    if v921_metrics is not None:
        report.extend(
            [
                "",
                "## Comparison with V9.21 static consensus",
                "",
                f"- V9.21 MAE: "
                f"`{float(v921_metrics['mae']):.6f}`",
                f"- V9.24 MGDA MAE: "
                f"`{float(primary['mae']):.6f}`",
                f"- V9.24 improvement: "
                f"`{float(v921_metrics['mae'] - primary['mae']):+.6f}`",
            ]
        )
    (output / "v924_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("V9.24 REGION-GRADIENT CONSOLIDATION COMPLETE")
    print("models trained: True")
    print("primary strategy:", PRIMARY_STRATEGY)
    print("primary mae:", f"{float(primary['mae']):.6f}")
    print(
        "gain vs global control:",
        f"{gain_vs_global:+.6f}",
    )
    print(
        "positive folds:",
        f"{positive_folds}/{int(cli.outer_folds)}",
    )
    print("success gate passed:", passed)
    print("report:", output / "v924_report.md")


if __name__ == "__main__":
    main()
