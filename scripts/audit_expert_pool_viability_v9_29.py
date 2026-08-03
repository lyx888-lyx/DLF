"""Independent output audit for V9.29 expert-pool viability results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.expert_pool_viability_audit_v929 import AUDIT_VERSION, mae
from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default=(
            "result/full_same_stack_nested_crossfit_v919/mosi/seed_1111/"
            "v929_expert_pool_viability_audit"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=2e-6)
    return parser.parse_args()


def assert_close(left: float, right: float, tolerance: float, message: str) -> None:
    if not np.isfinite(left) or not np.isfinite(right):
        raise AssertionError(f"{message}: non-finite {left} / {right}")
    if abs(float(left) - float(right)) > float(tolerance):
        raise AssertionError(f"{message}: {left} != {right}")


def main():
    cli = parse_args()
    result_dir = Path(cli.result_dir)
    required = {
        "summary": result_dir / "v929_viability_summary.json",
        "samples": result_dir / "v929_outer_predictions.csv",
        "aggregate": result_dir / "v929_aggregate_bounds.csv",
        "fold_metrics": result_dir / "v929_viability_metrics_by_fold.csv",
        "actions": result_dir / "v929_action_metrics_aggregate.csv",
        "weights": result_dir / "v929_outer_label_cheating_weights.csv",
        "sources": result_dir / "v929_source_manifest.csv",
        "report": result_dir / "v929_viability_report.md",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V9.29 outputs: {missing}")

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    samples = pd.read_csv(required["samples"])
    aggregate = pd.read_csv(required["aggregate"])
    fold_metrics = pd.read_csv(required["fold_metrics"])
    actions_frame = pd.read_csv(required["actions"])
    weights = pd.read_csv(required["weights"])
    sources = pd.read_csv(required["sources"])

    if summary.get("version") != AUDIT_VERSION:
        raise AssertionError(
            f"unexpected audit version: {summary.get('version')} != {AUDIT_VERSION}"
        )
    provenance = summary.get("provenance", {})
    required_provenance = {
        "uses_only_v919_saved_predictions": True,
        "new_models_trained": False,
        "outer_labels_used_for_diagnostic_upper_bounds": True,
        "outer_label_cheating_methods_are_not_deployable": True,
        "different_protocol_results_are_not_loaded": True,
        "inner_outer_sample_disjointness_checked": True,
        "inner_outer_group_disjointness_checked": True,
        "outer_sample_uniqueness_checked": True,
        "source_sha256_recorded": True,
    }
    for key, expected in required_provenance.items():
        if provenance.get(key) is not expected:
            raise AssertionError(f"invalid provenance flag {key}: {provenance.get(key)}")

    if samples.empty or samples["sample_id"].astype(str).duplicated().any():
        raise AssertionError("outer predictions are empty or contain duplicate sample IDs")
    if int(samples["outer_fold"].nunique()) != int(summary["outer_fold_count"]):
        raise AssertionError("outer fold count mismatch")
    group_fold_count = samples.groupby(samples["group_id"].astype(str))["outer_fold"].nunique()
    if int(group_fold_count.max()) != 1:
        raise AssertionError("a group appears in multiple outer holdouts")
    if len(samples) != int(summary["sample_count"]):
        raise AssertionError("sample count mismatch")

    action_names = tuple(summary.get("action_names", ()))
    if action_names != tuple(ACTION_NAMES):
        raise AssertionError(f"action order changed: {action_names}")
    action_columns = [f"action_prediction_{name}" for name in ACTION_NAMES]
    missing_columns = [column for column in action_columns if column not in samples]
    if missing_columns:
        raise AssertionError(f"missing action columns: {missing_columns}")

    labels = samples["label"].to_numpy(dtype=np.float64)
    actions = samples[action_columns].to_numpy(dtype=np.float64)
    errors = np.abs(actions - labels[:, None])
    global_action_mae = errors.mean(axis=0)
    global_best_index = int(global_action_mae.argmin())
    global_best_name = ACTION_NAMES[global_best_index]
    if global_best_name != summary["best_global_fixed_action_posthoc"]:
        raise AssertionError("best global fixed action does not recompute")
    if not np.allclose(
        samples["prediction_best_global_fixed_action_cheating"].to_numpy(),
        actions[:, global_best_index],
        atol=cli.tolerance,
        rtol=0.0,
    ):
        raise AssertionError("best-global-fixed predictions do not recompute")

    recomputed_oracle_index = errors.argmin(axis=1)
    recomputed_oracle_prediction = actions[
        np.arange(len(samples), dtype=np.int64), recomputed_oracle_index
    ]
    recomputed_oracle_action = np.asarray(
        [ACTION_NAMES[index] for index in recomputed_oracle_index], dtype=object
    )
    if not np.allclose(
        samples["prediction_sample_oracle"].to_numpy(),
        recomputed_oracle_prediction,
        atol=cli.tolerance,
        rtol=0.0,
    ):
        raise AssertionError("sample-oracle predictions do not recompute")
    if not np.array_equal(
        samples["sample_oracle_action"].astype(str).to_numpy(),
        recomputed_oracle_action.astype(str),
    ):
        raise AssertionError("sample-oracle actions do not recompute")

    pooled = weights[weights["fit_scope"] == "pooled_global_outer_label_cheating"]
    if len(pooled) != len(ACTION_NAMES):
        raise AssertionError("pooled cheating weight vector is incomplete")
    pooled = pooled.set_index("action").loc[list(ACTION_NAMES), "weight"].to_numpy()
    if np.any(pooled < -cli.tolerance):
        raise AssertionError("pooled cheating weights contain a negative value")
    assert_close(float(pooled.sum()), 1.0, cli.tolerance, "pooled simplex sum")
    pooled_prediction = actions @ pooled
    if not np.allclose(
        samples["prediction_pooled_global_convex_cheating"].to_numpy(),
        pooled_prediction,
        atol=cli.tolerance,
        rtol=0.0,
    ):
        raise AssertionError("pooled cheating convex predictions do not recompute")

    fold_best_actions = {}
    for fold, local in samples.groupby("outer_fold", sort=True):
        local_index = local.index.to_numpy(dtype=np.int64)
        local_actions = actions[local_index]
        local_labels = labels[local_index]
        local_errors = np.abs(local_actions - local_labels[:, None])
        best_index = int(local_errors.mean(axis=0).argmin())
        best_name = ACTION_NAMES[best_index]
        fold_best_actions[str(int(fold))] = best_name
        if not np.allclose(
            local["prediction_fold_best_single_cheating"].to_numpy(),
            local_actions[:, best_index],
            atol=cli.tolerance,
            rtol=0.0,
        ):
            raise AssertionError(f"fold {fold} best-single predictions do not recompute")
        if local["fold_best_single_action_cheating"].astype(str).nunique() != 1:
            raise AssertionError(f"fold {fold} stores multiple fold-best actions")
        stored_name = str(local["fold_best_single_action_cheating"].iloc[0])
        if stored_name != best_name:
            raise AssertionError(f"fold {fold} best action mismatch")

        local_weights = weights[
            (weights["fit_scope"] == "fold_local_outer_label_cheating")
            & (weights["outer_fold"].astype(str) == str(int(fold)))
        ]
        if len(local_weights) != len(ACTION_NAMES):
            raise AssertionError(f"fold {fold} cheating weight vector is incomplete")
        local_weights = (
            local_weights.set_index("action")
            .loc[list(ACTION_NAMES), "weight"]
            .to_numpy(dtype=np.float64)
        )
        if np.any(local_weights < -cli.tolerance):
            raise AssertionError(f"fold {fold} has negative cheating weight")
        assert_close(
            float(local_weights.sum()), 1.0, cli.tolerance, f"fold {fold} simplex sum"
        )
        local_prediction = local_actions @ local_weights
        if not np.allclose(
            local["prediction_fold_local_convex_cheating"].to_numpy(),
            local_prediction,
            atol=cli.tolerance,
            rtol=0.0,
        ):
            raise AssertionError(f"fold {fold} convex predictions do not recompute")
        if mae(local_prediction, local_labels) > mae(
            local_actions[:, best_index], local_labels
        ) + cli.tolerance:
            raise AssertionError(f"fold {fold} convex upper bound is worse than one-hot")

    if fold_best_actions != summary["fold_posthoc_best_actions"]:
        raise AssertionError("fold posthoc best-action map does not recompute")

    prediction_columns = {
        "anchor": "prediction_anchor",
        "inner_selected_single": "prediction_inner_selected_single",
        "v921_convex_shrinkage": "prediction_v921_convex_shrinkage",
        "best_global_fixed_action_cheating": "prediction_best_global_fixed_action_cheating",
        "fold_best_single_cheating": "prediction_fold_best_single_cheating",
        "pooled_global_convex_cheating": "prediction_pooled_global_convex_cheating",
        "fold_local_convex_cheating": "prediction_fold_local_convex_cheating",
        "sample_oracle": "prediction_sample_oracle",
    }
    aggregate_by_strategy = aggregate.set_index("strategy")
    for strategy, column in prediction_columns.items():
        if strategy not in aggregate_by_strategy.index:
            raise AssertionError(f"aggregate table missing {strategy}")
        recomputed = mae(samples[column].to_numpy(dtype=np.float64), labels)
        stored = float(aggregate_by_strategy.loc[strategy, "mae"])
        assert_close(recomputed, stored, cli.tolerance, f"aggregate MAE {strategy}")
        summary_mae = float(summary["aggregate_metrics"][strategy]["mae"])
        assert_close(recomputed, summary_mae, cli.tolerance, f"summary MAE {strategy}")

    sample_oracle_mae = float(aggregate_by_strategy.loc["sample_oracle", "mae"])
    fold_best_mae = float(
        aggregate_by_strategy.loc["fold_best_single_cheating", "mae"]
    )
    fold_convex_mae = float(
        aggregate_by_strategy.loc["fold_local_convex_cheating", "mae"]
    )
    best_global_mae = float(
        aggregate_by_strategy.loc["best_global_fixed_action_cheating", "mae"]
    )
    pooled_convex_mae = float(
        aggregate_by_strategy.loc["pooled_global_convex_cheating", "mae"]
    )
    if sample_oracle_mae > fold_best_mae + cli.tolerance:
        raise AssertionError("sample oracle is worse than fold-best single")
    if fold_best_mae > best_global_mae + cli.tolerance:
        raise AssertionError("fold-best single is worse than global-best single")
    if fold_convex_mae > fold_best_mae + cli.tolerance:
        raise AssertionError("fold-local convex is worse than fold-best single")
    if pooled_convex_mae > best_global_mae + cli.tolerance:
        raise AssertionError("pooled convex is worse than global-best single")

    if len(sources) != int(summary["outer_fold_count"]) * 3:
        raise AssertionError("source manifest should contain three files per outer fold")
    if not bool(sources["sha256"].astype(str).str.fullmatch(r"[0-9a-f]{64}").all()):
        raise AssertionError("source manifest contains invalid SHA256 values")
    if int(actions_frame["is_posthoc_best_global_action"].astype(str).str.lower().eq("true").sum()) != 1:
        raise AssertionError("action aggregate must mark exactly one global best action")
    required_fold_strategies = {
        "anchor",
        "inner_selected_single",
        "v921_convex_shrinkage",
        "fold_best_single_cheating",
        "fold_local_convex_cheating",
        "sample_oracle",
    }
    for fold, local in fold_metrics.groupby("outer_fold"):
        if set(local["strategy"]) != required_fold_strategies:
            raise AssertionError(f"fold {fold} has incomplete strategy rows")

    audit_result = {
        "version": AUDIT_VERSION,
        "passed": True,
        "result_dir": str(result_dir),
        "sample_count": int(len(samples)),
        "outer_fold_count": int(samples["outer_fold"].nunique()),
        "best_global_fixed_action": global_best_name,
        "aggregate_mae": {
            strategy: float(aggregate_by_strategy.loc[strategy, "mae"])
            for strategy in prediction_columns
        },
        "checks": {
            "predictions_recomputed": True,
            "single_action_bounds_recomputed": True,
            "convex_weights_reapplied": True,
            "sample_oracle_recomputed": True,
            "source_manifest_checked": True,
            "provenance_checked": True,
        },
    }
    (result_dir / "v929_audit_check.json").write_text(
        json.dumps(audit_result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("V9.29 OUTPUT AUDIT PASSED")
    print("samples:", len(samples))
    print("best global fixed action:", global_best_name)
    print("audit check:", result_dir / "v929_audit_check.json")


if __name__ == "__main__":
    main()
