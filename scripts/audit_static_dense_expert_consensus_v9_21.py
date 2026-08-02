"""Engineering audit for the V9.21 static dense consensus artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.static_dense_expert_consensus_v921 import (
    AUDIT_VERSION,
    PRIMARY_STRATEGY,
    ConsensusConfigV921,
    aggregate_prediction_frames,
    best_single_action,
    prediction_metrics,
    predict_strategy,
    success_gate,
    validate_simplex,
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


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def scalar_metrics(metrics):
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"prediction", "sample_gain"}
    }


def close(left, right, tolerance=3e-6):
    return abs(float(left) - float(right)) <= float(tolerance)


def main():
    cli = parse_args()
    root = Path(cli.root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v921_static_dense_expert_consensus"
    )
    required = {
        "summary": output / "v921_static_consensus_summary.json",
        "fold_metrics": output / "v921_strategy_metrics_by_fold.csv",
        "weights": output / "v921_convex_weights_by_fold.csv",
        "samples": output / "v921_outer_predictions.csv",
        "aggregate": output / "v921_aggregate_strategy_metrics.csv",
        "bootstrap": output / "v921_group_bootstrap_gain_ci.csv",
        "report": output / "v921_static_consensus_report.md",
    }
    for path in required.values():
        require(path.is_file(), f"missing V9.21 artifact: {path}")

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    require(summary["version"] == AUDIT_VERSION, "unexpected V9.21 version")
    provenance = summary["provenance"]
    require(provenance["uses_only_v919_saved_predictions"] is True, "wrong source")
    require(provenance["new_anchor_or_expert_models_trained"] is False, "model training detected")
    require(provenance["weights_fit_on_inner_oof_only"] is True, "outer fitting detected")
    require(provenance["official_validation_loaded"] is False, "Validation loaded")
    require(provenance["official_test_loaded"] is False, "Test loaded")
    require(provenance["sample_dependent_expert_weights"] is False, "dynamic weights detected")
    require(provenance["router_region_model_or_gate_used"] is False, "router detected")
    require(summary["primary_strategy"] == PRIMARY_STRATEGY, "primary strategy changed")

    config_keys = {field.name for field in fields(ConsensusConfigV921)}
    config = ConsensusConfigV921(
        **{key: value for key, value in summary["config"].items() if key in config_keys}
    )
    fold_metrics = pd.read_csv(required["fold_metrics"])
    weights = pd.read_csv(required["weights"])
    samples = pd.read_csv(required["samples"])
    aggregate_frame = pd.read_csv(required["aggregate"])
    bootstrap = pd.read_csv(required["bootstrap"])
    require(not samples["sample_id"].astype(str).duplicated().any(), "duplicate outer sample IDs")
    require(len(samples) == int(summary["sample_count"]), "sample count mismatch")

    strategies = sorted(
        column.removeprefix("prediction_")
        for column in samples.columns
        if column.startswith("prediction_")
    )
    require(PRIMARY_STRATEGY in strategies, "primary prediction column missing")
    require(set(bootstrap["strategy"]) == set(strategies), "bootstrap strategies mismatch")

    for fold in range(int(cli.outer_folds)):
        fold_dir = root / f"outer_fold_{fold}"
        inner_payload = torch.load(
            fold_dir / "inner_oof_same_stack_pool_v919.pth", map_location="cpu"
        )
        outer_payload = torch.load(
            fold_dir
            / "outer_deployment_stack"
            / "same_stack_target_pool_v919.pth",
            map_location="cpu",
        )
        inner_pool = normalize_pool(inner_payload)
        outer_pool = normalize_pool(outer_payload)
        frame = samples[samples["outer_fold"] == fold].reset_index(drop=True)
        require(frame["sample_id"].astype(str).tolist() == outer_pool.sample_ids, f"fold {fold} sample order mismatch")
        require(frame["group_id"].astype(str).tolist() == outer_pool.group_ids, f"fold {fold} group order mismatch")
        require(
            np.allclose(frame["label"].to_numpy(), outer_pool.labels.numpy(), atol=1e-6),
            f"fold {fold} labels mismatch",
        )
        for action_index, action in enumerate(ACTION_NAMES):
            require(
                np.allclose(
                    frame[f"action_prediction_{action}"].to_numpy(),
                    outer_pool.actions[:, action_index].numpy(),
                    atol=2e-6,
                ),
                f"fold {fold} action prediction mismatch: {action}",
            )

        selected_index = best_single_action(inner_pool)
        selected_name = ACTION_NAMES[selected_index]
        require(
            set(frame["inner_selected_single_action"].astype(str)) == {selected_name},
            f"fold {fold} selected single action mismatch",
        )
        checkpoint_path = output / f"outer_fold_{fold}_consensus_v921.pth"
        require(checkpoint_path.is_file(), f"missing consensus checkpoint {fold}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        require(checkpoint["inner_selected_single_action"] == selected_name, "checkpoint selected action mismatch")
        require(checkpoint["provenance"]["sample_dependent_weights"] is False, "checkpoint has dynamic weights")

        strategy_predictions = {
            "anchor": predict_strategy(outer_pool, "anchor"),
            **{
                f"fixed_{name}": predict_strategy(
                    outer_pool, "fixed_action", action_index=index
                )
                for index, name in enumerate(ACTION_NAMES[1:], start=1)
            },
            "inner_selected_single": predict_strategy(
                outer_pool, "fixed_action", action_index=selected_index
            ),
            "boundary_positive_mean": predict_strategy(
                outer_pool, "boundary_positive_mean"
            ),
            "all_action_mean": predict_strategy(outer_pool, "all_action_mean"),
            "all_action_median": predict_strategy(outer_pool, "all_action_median"),
            "trimmed_mean": predict_strategy(outer_pool, "trimmed_mean"),
        }
        for consensus_name in ("convex_mae", "convex_shrinkage"):
            subset = weights[
                (weights["outer_fold"] == fold)
                & (weights["strategy"] == consensus_name)
            ]
            by_action = subset.set_index("action")["weight"]
            require(set(by_action.index) == set(ACTION_NAMES), f"fold {fold} incomplete weights")
            weight_vector = validate_simplex(
                [by_action[action] for action in ACTION_NAMES], len(ACTION_NAMES)
            )
            checkpoint_weights = checkpoint["weights"][consensus_name].numpy()
            require(np.allclose(weight_vector, checkpoint_weights, atol=2e-6), "checkpoint weight mismatch")
            strategy_predictions[consensus_name] = predict_strategy(
                outer_pool, consensus_name, weights=weight_vector
            )

        require(set(strategy_predictions) == set(strategies), f"fold {fold} strategy set mismatch")
        for strategy, prediction in strategy_predictions.items():
            require(
                np.allclose(
                    frame[f"prediction_{strategy}"].to_numpy(),
                    prediction.numpy(),
                    atol=3e-6,
                ),
                f"fold {fold} prediction mismatch: {strategy}",
            )
            recomputed = scalar_metrics(
                prediction_metrics(
                    prediction, outer_pool.labels, outer_pool.actions[:, 0]
                )
            )
            stored_rows = fold_metrics[
                (fold_metrics["outer_fold"] == fold)
                & (fold_metrics["strategy"] == strategy)
            ]
            require(len(stored_rows) == 1, f"fold metric row missing: {fold}/{strategy}")
            stored = stored_rows.iloc[0]
            for key in (
                "anchor_mae",
                "mae",
                "gain_vs_anchor",
                "win_rate",
                "large_gain_rate_010",
                "large_harm_rate_010",
            ):
                require(close(recomputed[key], stored[key]), f"fold metric mismatch {fold}/{strategy}/{key}")

    aggregate = aggregate_prediction_frames(samples)
    stored_aggregate = aggregate_frame.set_index("strategy")
    require(set(stored_aggregate.index) == set(aggregate), "aggregate strategy set mismatch")
    for strategy, metrics in aggregate.items():
        for key in (
            "anchor_mae",
            "mae",
            "gain_vs_anchor",
            "win_rate",
            "large_gain_rate_010",
            "large_harm_rate_010",
        ):
            require(close(metrics[key], stored_aggregate.loc[strategy, key]), f"aggregate mismatch {strategy}/{key}")
            require(close(metrics[key], summary["aggregate_metrics"][strategy][key]), f"summary mismatch {strategy}/{key}")

    gate = success_gate(fold_metrics, aggregate, config)
    for key in (
        "aggregate_gain_vs_inner_selected_single",
        "nondegrading_outer_folds",
        "worst_fold_degradation",
        "passed",
    ):
        require(
            gate[key] == summary["success_gate"][key]
            or close(gate[key], summary["success_gate"][key]),
            f"gate mismatch: {key}",
        )

    print("V9.21 STATIC DENSE EXPERT CONSENSUS AUDIT PASSED")
    print("new expert models trained: False")
    print("sample-dependent routing: False")
    print("primary strategy:", PRIMARY_STRATEGY)
    print(
        "anchor/selected-single/primary MAE:",
        f"{aggregate['anchor']['mae']:.6f}",
        f"{aggregate['inner_selected_single']['mae']:.6f}",
        f"{aggregate[PRIMARY_STRATEGY]['mae']:.6f}",
    )
    print("success gate passed:", gate["passed"])


if __name__ == "__main__":
    main()
