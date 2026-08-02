"""Engineering audit for V9.20 saved-artifact decomposition outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.no_train_decomposition_v920 import (  # noqa: E402
    ACTION_NAMES,
    AUDIT_VERSION,
    REGION_NAMES,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, atol: float = 3e-6) -> bool:
    return abs(float(left) - float(right)) <= atol


def strategy_metrics(frame: pd.DataFrame, prediction: str, action: str):
    anchor_error = (frame["anchor_prediction"] - frame["label"]).abs()
    selected_error = (frame[prediction] - frame["label"]).abs()
    gain = anchor_error - selected_error
    triggered = frame[action].astype(str) != "anchor"
    return {
        "anchor_mae": float(anchor_error.mean()),
        "mae": float(selected_error.mean()),
        "gain_vs_anchor": float(gain.mean()),
        "coverage": float(triggered.mean()),
        "harm_over_010_rate": float((gain < -0.10).mean()),
    }


def main():
    cli = parse_args()
    root = Path(cli.root)
    output = Path(cli.output_dir) if cli.output_dir else root / "v920_no_train_decomposition"
    files = {
        "summary": output / "v920_decomposition_summary.json",
        "report": output / "v920_decomposition_report.md",
        "folds": output / "v920_fold_decomposition.csv",
        "actions": output / "v920_action_metrics_by_fold.csv",
        "aggregate_actions": output / "v920_outer_action_metrics_aggregate.csv",
        "regions": output / "v920_region_action_metrics_by_fold.csv",
        "aggregate_regions": output / "v920_outer_region_action_aggregate.csv",
        "consistency": output / "v920_outer_region_action_consistency.csv",
        "mappings": output / "v920_development_locked_region_mappings.csv",
        "calibration": output / "v920_predicted_gain_calibration_bins.csv",
        "samples": output / "v920_outer_sample_decomposition.csv",
    }
    for path in files.values():
        require(path.is_file(), f"missing V9.20 output: {path}")

    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    folds = pd.read_csv(files["folds"])
    actions = pd.read_csv(files["actions"])
    aggregate_actions = pd.read_csv(files["aggregate_actions"])
    regions = pd.read_csv(files["regions"])
    aggregate_regions = pd.read_csv(files["aggregate_regions"])
    mappings = pd.read_csv(files["mappings"])
    calibration = pd.read_csv(files["calibration"])
    samples = pd.read_csv(files["samples"])

    require(summary["version"] == AUDIT_VERSION, "summary version mismatch")
    require(summary["models_trained"] is False, "audit claims model training")
    require(summary["official_validation_loaded"] is False, "Validation was loaded")
    require(summary["official_test_loaded"] is False, "Test was loaded")
    require(
        summary["provenance"]["no_optimizer_or_backward_pass"] is True,
        "no-training provenance missing",
    )
    require(len(folds) == cli.outer_folds, "outer fold count mismatch")
    require(set(folds["outer_fold"]) == set(range(cli.outer_folds)), "fold IDs mismatch")
    require(len(samples) == summary["sample_count"], "sample count mismatch")
    require(not samples["sample_id"].astype(str).duplicated().any(), "duplicate sample IDs")
    require(samples["outer_fold"].nunique() == cli.outer_folds, "sample fold count mismatch")

    prediction_columns = [f"prediction_{name}" for name in ACTION_NAMES]
    expected_columns = [f"expected_cost_{name}" for name in ACTION_NAMES]
    require(set(prediction_columns).issubset(samples.columns), "action predictions missing")
    require(set(expected_columns).issubset(samples.columns), "expected costs missing")

    errors = np.abs(samples[prediction_columns].to_numpy() - samples[["label"]].to_numpy())
    sample_oracle_error = np.abs(
        samples["sample_oracle_prediction"].to_numpy() - samples["label"].to_numpy()
    )
    require(
        np.all(sample_oracle_error <= errors.min(axis=1) + 2e-5),
        "sample oracle is not the minimum action error",
    )

    strategies = {
        "actual_gated_router": ("selected_prediction", "selected_action"),
        "sample_oracle": ("sample_oracle_prediction", "sample_oracle_action"),
        "semantic_region_oracle": (
            "semantic_region_prediction",
            "semantic_region_action",
        ),
        "development_locked_true_region_oracle": (
            "dev_locked_region_prediction",
            "dev_locked_region_action",
        ),
        "posthoc_true_region_oracle": (
            "posthoc_region_prediction",
            "posthoc_region_action",
        ),
        "counterfactual_frozen_router": (
            "counterfactual_router_prediction",
            "counterfactual_router_action",
        ),
        "expected_cost_argmin": (
            "expected_cost_argmin_prediction",
            "expected_cost_argmin_action",
        ),
    }
    for name, (prediction, action) in strategies.items():
        recomputed = strategy_metrics(samples, prediction, action)
        stored = summary["aggregate_strategies"][name]
        for metric in (
            "anchor_mae",
            "mae",
            "gain_vs_anchor",
            "coverage",
            "harm_over_010_rate",
        ):
            require(
                close(recomputed[metric], stored[metric]),
                f"aggregate strategy mismatch {name}.{metric}",
            )

    for outer_fold in range(cli.outer_folds):
        fold_samples = samples[samples["outer_fold"] == outer_fold].copy()
        fold_summary_path = (
            root
            / f"outer_fold_{outer_fold}"
            / "same_stack_nested_crossfit_v919_fold_summary.json"
        )
        original = json.loads(fold_summary_path.read_text(encoding="utf-8"))
        cutoff_value = original["outer_holdout_metrics"].get("gain_cutoff")
        cutoff = float("inf") if cutoff_value is None else float(cutoff_value)
        confidence = float(original["fixed_policy"]["min_region_confidence"])
        expected_trigger = (
            fold_samples["recomputed_predicted_gain"].astype(float) >= cutoff
        ) & (fold_samples["region_confidence"].astype(float) >= confidence)
        observed_trigger = fold_samples["counterfactual_router_triggered"].astype(bool)
        require(
            np.array_equal(expected_trigger.to_numpy(), observed_trigger.to_numpy()),
            f"counterfactual trigger mismatch in outer fold {outer_fold}",
        )
        allowed = list(original["fixed_policy"]["allowed_specialists"])
        if observed_trigger.any():
            proposed = (
                fold_samples.loc[
                    observed_trigger,
                    [f"expected_cost_{name}" for name in allowed],
                ]
                .astype(float)
                .idxmin(axis=1)
                .str.replace("expected_cost_", "", regex=False)
            )
            observed = fold_samples.loc[
                observed_trigger, "counterfactual_router_action"
            ].astype(str)
            require(
                np.array_equal(proposed.to_numpy(), observed.to_numpy()),
                f"counterfactual action mismatch in outer fold {outer_fold}",
            )

        fold_mapping = mappings[mappings["outer_fold"] == outer_fold]
        require(len(fold_mapping) == len(REGION_NAMES), "mapping row count mismatch")
        for row in fold_mapping.itertuples(index=False):
            sample_region = fold_samples[
                fold_samples["true_region"].astype(int) == int(row.region_index)
            ]
            if len(sample_region):
                require(
                    set(sample_region["dev_locked_region_action"].astype(str))
                    == {str(row.development_selected_action)},
                    f"development region mapping mismatch fold={outer_fold} region={row.region}",
                )

    require(
        set(actions["split"]) == {"inner_oof", "outer_holdout"},
        "action metric split mismatch",
    )
    require(
        len(aggregate_actions) == len(ACTION_NAMES),
        "aggregate action row count mismatch",
    )
    require(
        len(aggregate_regions) == len(REGION_NAMES) * len(ACTION_NAMES),
        "aggregate region-action shape mismatch",
    )
    require(
        len(regions)
        == cli.outer_folds * 2 * len(REGION_NAMES) * len(ACTION_NAMES),
        "fold region-action shape mismatch",
    )
    require(len(calibration) >= cli.outer_folds, "calibration output is empty")

    decomposition = summary["decomposition"]
    cf = summary["aggregate_strategies"]["counterfactual_frozen_router"]
    cf_positive_folds = int((folds["outer_counterfactual_router_gain"] > 0).sum())
    passes = (
        cf["gain_vs_anchor"] >= 0.005
        and cf_positive_folds >= 4
        and cf["harm_over_010_rate"] <= 0.05
    )
    expected_gate_verdict = (
        "potentially_overconservative"
        if passes
        else "protective_or_not_the_primary_problem"
    )
    require(
        decomposition["gate_verdict"] == expected_gate_verdict,
        "gate verdict does not match evidence",
    )

    print("V9.20 NO-TRAINING DECOMPOSITION AUDIT PASSED")
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


if __name__ == "__main__":
    main()
