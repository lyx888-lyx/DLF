"""Exploratory static consensus compatibility audit for frozen V9.17 experts."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.static_dense_expert_consensus_v921 import (
    residual_correlation_rows,
)
from trains.singleTask.v917_static_dense_consensus_v922 import (
    AUDIT_VERSION,
    PRIMARY_STRATEGY,
    CompatibilityConfigV922,
    evaluate_predictions,
    fit_full_validation,
    normalize_v917_pool,
    paired_group_bootstrap_difference,
    scalar_metrics,
    strategy_predictions,
    validation_group_crossfit,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v917-root",
        default="result/fixed_expert_region_audit_v917/mosi/seed_1111",
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--shrinkage-lambda", type=float, default=0.01)
    parser.add_argument("--validation-group-folds", type=int, default=5)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1111)
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return "\n".join(
        [
            "|" + "|".join(columns) + "|",
            "|" + "|".join(["---"] * len(columns)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in view.itertuples(index=False, name=None)
            ],
        ]
    )


def main():
    cli = parse_args()
    root = Path(cli.v917_root)
    valid_path = root / "valid_fixed_expert_pool_v917.pth"
    test_path = root / "test_fixed_expert_pool_v917.pth"
    for path in (valid_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v922_static_dense_consensus_compatibility"
    )
    output.mkdir(parents=True, exist_ok=True)

    config = CompatibilityConfigV922(
        shrinkage_lambda=float(cli.shrinkage_lambda),
        validation_group_folds=int(cli.validation_group_folds),
        bootstrap_repetitions=int(cli.bootstrap_repetitions),
        bootstrap_seed=int(cli.bootstrap_seed),
    )
    validation = normalize_v917_pool(
        torch.load(valid_path, map_location="cpu"),
        "v917_frozen_validation",
    )
    test = normalize_v917_pool(
        torch.load(test_path, map_location="cpu"),
        "v917_frozen_test",
    )
    if set(validation.sample_ids) & set(test.sample_ids):
        raise RuntimeError("V9.17 Validation/Test sample overlap")

    validation_cv = validation_group_crossfit(validation, config)
    fit = fit_full_validation(validation, config)
    validation_predictions = strategy_predictions(fit, validation)
    test_predictions = strategy_predictions(fit, test)
    validation_metrics = evaluate_predictions(validation, validation_predictions)
    test_metrics = evaluate_predictions(test, test_predictions)

    selected_name = ACTION_NAMES[int(fit["selected_single_index"])]
    weight_rows = []
    for strategy, values in fit["weights"].items():
        for action, value in zip(ACTION_NAMES, values.tolist()):
            weight_rows.append(
                {"strategy": strategy, "action": action, "weight": float(value)}
            )
    weights = pd.DataFrame(weight_rows)

    valid_cv_rows = []
    for strategy, metrics in validation_cv["aggregate"].items():
        valid_cv_rows.append(
            {"scope": "validation_group_crossfit", "strategy": strategy, **scalar_metrics(metrics)}
        )
    valid_cv_metrics = pd.DataFrame(valid_cv_rows).sort_values("mae")
    valid_fold_metrics = pd.DataFrame(validation_cv["fold_rows"])

    valid_rows = [
        {"split": "validation_in_sample", "strategy": name, **scalar_metrics(metrics)}
        for name, metrics in validation_metrics.items()
    ]
    test_rows = [
        {"split": "test_exploratory", "strategy": name, **scalar_metrics(metrics)}
        for name, metrics in test_metrics.items()
    ]
    all_metrics = pd.DataFrame([*valid_rows, *test_rows]).sort_values(
        ["split", "mae"]
    )
    test_metric_frame = pd.DataFrame(test_rows).sort_values("mae")

    validation_frame = pd.DataFrame(
        {
            "sample_id": validation.sample_ids,
            "group_id": validation.group_ids,
            "label": validation.labels.numpy(),
        }
    )
    for index, action in enumerate(ACTION_NAMES):
        validation_frame[f"action_prediction_{action}"] = validation.actions[:, index].numpy()
    for strategy, prediction in validation_predictions.items():
        validation_frame[f"prediction_{strategy}"] = prediction.numpy()
    for strategy, prediction in validation_cv["predictions"].items():
        validation_frame[f"crossfit_prediction_{strategy}"] = prediction.numpy()

    test_frame = pd.DataFrame(
        {
            "sample_id": test.sample_ids,
            "group_id": test.group_ids,
            "label": test.labels.numpy(),
        }
    )
    for index, action in enumerate(ACTION_NAMES):
        test_frame[f"action_prediction_{action}"] = test.actions[:, index].numpy()
    for strategy, prediction in test_predictions.items():
        test_frame[f"prediction_{strategy}"] = prediction.numpy()

    primary_gain = test_metrics[PRIMARY_STRATEGY]["sample_gain"]
    comparisons = {}
    for baseline in (
        "anchor",
        "validation_selected_single",
        "fixed_strong_negative",
        "fixed_positive",
        "boundary_positive_mean",
    ):
        comparisons[baseline] = paired_group_bootstrap_difference(
            primary_gain,
            test_metrics[baseline]["sample_gain"],
            test.group_ids,
            config.bootstrap_repetitions,
            config.bootstrap_seed + 1009 * (len(comparisons) + 1),
        )
    bootstrap = pd.DataFrame(
        [{"primary": PRIMARY_STRATEGY, "baseline": name, **value} for name, value in comparisons.items()]
    )

    signed_valid = pd.DataFrame(
        residual_correlation_rows(validation.actions, validation.labels, "signed_residual")
    )
    signed_valid.insert(0, "split", "validation")
    signed_test = pd.DataFrame(
        residual_correlation_rows(test.actions, test.labels, "signed_residual")
    )
    signed_test.insert(0, "split", "test")
    absolute_valid = pd.DataFrame(
        residual_correlation_rows(validation.actions, validation.labels, "absolute_error")
    )
    absolute_valid.insert(0, "split", "validation")
    absolute_test = pd.DataFrame(
        residual_correlation_rows(test.actions, test.labels, "absolute_error")
    )
    absolute_test.insert(0, "split", "test")

    summary = {
        "version": AUDIT_VERSION,
        "method": "v917_validation_fitted_static_global_convex_consensus",
        "evaluation_status": "exploratory_compatibility_audit_test_previously_observed",
        "config": asdict(config),
        "validation_count": len(validation.labels),
        "validation_group_count": len(set(validation.group_ids)),
        "test_count": len(test.labels),
        "test_group_count": len(set(test.group_ids)),
        "primary_strategy": PRIMARY_STRATEGY,
        "validation_selected_single_action": selected_name,
        "weights": {
            strategy: {action: float(value) for action, value in zip(ACTION_NAMES, values.tolist())}
            for strategy, values in fit["weights"].items()
        },
        "validation_group_crossfit_metrics": {
            name: scalar_metrics(metrics)
            for name, metrics in validation_cv["aggregate"].items()
        },
        "validation_in_sample_metrics": {
            name: scalar_metrics(metrics) for name, metrics in validation_metrics.items()
        },
        "test_exploratory_metrics": {
            name: scalar_metrics(metrics) for name, metrics in test_metrics.items()
        },
        "primary_paired_group_bootstrap": comparisons,
        "key_differences": {
            "test_mae_improvement_vs_anchor": float(
                test_metrics["anchor"]["mae"] - test_metrics[PRIMARY_STRATEGY]["mae"]
            ),
            "test_mae_improvement_vs_validation_selected_single": float(
                test_metrics["validation_selected_single"]["mae"]
                - test_metrics[PRIMARY_STRATEGY]["mae"]
            ),
            "test_mae_improvement_vs_strong_negative": float(
                test_metrics["fixed_strong_negative"]["mae"]
                - test_metrics[PRIMARY_STRATEGY]["mae"]
            ),
            "test_mae_improvement_vs_fixed_positive": float(
                test_metrics["fixed_positive"]["mae"]
                - test_metrics[PRIMARY_STRATEGY]["mae"]
            ),
        },
        "provenance": {
            "new_anchor_or_expert_models_trained": False,
            "weights_fit_on_v917_validation_only": True,
            "test_used_for_weight_fitting_or_strategy_selection": False,
            "shrinkage_lambda_pre_registered_from_v921": True,
            "sample_dependent_weights": False,
            "router_region_model_or_gate_used": False,
            "official_test_previously_observed_in_prior_experiments": True,
            "result_is_not_an_independent_confirmatory_test": True,
        },
    }

    weights.to_csv(output / "v922_frozen_weights.csv", index=False)
    valid_cv_metrics.to_csv(output / "v922_validation_group_crossfit_metrics.csv", index=False)
    valid_fold_metrics.to_csv(output / "v922_validation_group_crossfit_by_fold.csv", index=False)
    all_metrics.to_csv(output / "v922_validation_and_test_strategy_metrics.csv", index=False)
    test_metric_frame.to_csv(output / "v922_test_strategy_metrics.csv", index=False)
    validation_frame.to_csv(output / "v922_validation_predictions.csv", index=False)
    test_frame.to_csv(output / "v922_test_predictions.csv", index=False)
    bootstrap.to_csv(output / "v922_test_paired_group_bootstrap.csv", index=False)
    pd.concat([signed_valid, signed_test], ignore_index=True).to_csv(
        output / "v922_signed_residual_correlation.csv", index=False
    )
    pd.concat([absolute_valid, absolute_test], ignore_index=True).to_csv(
        output / "v922_absolute_error_correlation.csv", index=False
    )
    checkpoint = {
        "version": AUDIT_VERSION,
        "action_names": list(ACTION_NAMES),
        "primary_strategy": PRIMARY_STRATEGY,
        "config": asdict(config),
        "validation_selected_single_index": int(fit["selected_single_index"]),
        "weights": {
            name: torch.tensor(value, dtype=torch.float32)
            for name, value in fit["weights"].items()
        },
        "provenance": summary["provenance"],
    }
    torch.save(checkpoint, output / "v922_v917_static_consensus.pth")
    (output / "v922_static_consensus_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    report = [
        "# V9.22 V9.17 Frozen-Expert Static Consensus Compatibility Audit",
        "",
        "This is exploratory because MOSI Test was observed in prior experiments.",
        "No model is trained; Validation fits one global frozen weight vector.",
        "",
        "## Validation conversation-group cross-fit",
        "",
        markdown_table(
            valid_cv_metrics,
            ["strategy", "mae", "gain_vs_anchor", "win_rate", "large_harm_rate_010"],
        ),
        "",
        "## Exploratory Test strategies",
        "",
        markdown_table(
            test_metric_frame,
            ["strategy", "mae", "gain_vs_anchor", "win_rate", "large_harm_rate_010"],
        ),
        "",
        "## Frozen weights",
        "",
        markdown_table(weights, ["strategy", "action", "weight"]),
        "",
        f"Validation-selected single action: `{selected_name}`",
        f"Primary Test MAE: `{test_metrics[PRIMARY_STRATEGY]['mae']:.6f}`",
        f"Primary gain versus Anchor: `{test_metrics[PRIMARY_STRATEGY]['gain_vs_anchor']:+.6f}`",
        f"Primary MAE improvement versus Validation-selected single: "
        f"`{summary['key_differences']['test_mae_improvement_vs_validation_selected_single']:+.6f}`",
        "",
        "The unregularized convex result is diagnostic only; Test does not select it.",
    ]
    (output / "v922_static_consensus_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.22 V9.17 STATIC CONSENSUS COMPATIBILITY COMPLETE")
    print("models trained: False")
    print("evaluation status: exploratory; Test previously observed")
    print("validation selected single:", selected_name)
    print("frozen primary weights:", summary["weights"][PRIMARY_STRATEGY])
    print(
        "test anchor/primary/gain:",
        f"{test_metrics['anchor']['mae']:.6f}",
        f"{test_metrics[PRIMARY_STRATEGY]['mae']:.6f}",
        f"{test_metrics[PRIMARY_STRATEGY]['gain_vs_anchor']:+.6f}",
    )
    print("report:", output / "v922_static_consensus_report.md")


if __name__ == "__main__":
    main()
