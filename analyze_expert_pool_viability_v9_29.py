"""Strict V9.29 expert-pool viability audit using saved V9.19 predictions only."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.expert_pool_viability_audit_v929 import (
    AUDIT_VERSION,
    ViabilityAuditConfigV929,
    compute_fold_upper_bounds,
    fit_mae_simplex,
    make_viability_verdict,
    oracle_gap_closure,
    prediction_metrics,
)
from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    strategy_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--target-mae", type=float, default=0.70)
    parser.add_argument("--shrinkage-lambda", type=float, default=0.01)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_row(strategy, prediction, labels, anchor, deployable, cheating, scope):
    return {
        "strategy": strategy,
        "deployable": bool(deployable),
        "label_cheating": bool(cheating),
        "selection_scope": scope,
        **prediction_metrics(prediction, labels, anchor),
    }


def markdown_table(frame, columns):
    local = frame[[column for column in columns if column in frame]].copy()
    for column in local.columns:
        if pd.api.types.is_float_dtype(local[column]):
            local[column] = local[column].map(
                lambda value: "" if pd.isna(value) else f"{float(value):.6f}"
            )
    return "\n".join(
        [
            "|" + "|".join(local.columns) + "|",
            "|" + "|".join(["---"] * len(local.columns)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in local.itertuples(index=False, name=None)
            ],
        ]
    )


def main():
    cli = parse_args()
    root = Path(cli.root)
    output = Path(cli.output_dir) if cli.output_dir else root / "v929_expert_pool_viability_audit"
    output.mkdir(parents=True, exist_ok=True)
    config = ViabilityAuditConfigV929(target_mae=float(cli.target_mae))
    config.validate()
    consensus = ConsensusConfigV921(shrinkage_lambda=float(cli.shrinkage_lambda))

    fold_rows, action_rows, weight_rows, source_rows, sample_frames = [], [], [], [], []
    for fold in range(int(cli.outer_folds)):
        fold_dir = root / f"outer_fold_{fold}"
        inner_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        outer_path = fold_dir / "outer_deployment_stack" / "same_stack_target_pool_v919.pth"
        summary_path = fold_dir / "same_stack_nested_crossfit_v919_fold_summary.json"
        for role, path in (
            ("inner_oof_pool", inner_path),
            ("outer_holdout_pool", outer_path),
            ("v919_fold_summary", summary_path),
        ):
            if not path.is_file():
                raise FileNotFoundError(path)
            source_rows.append(
                {
                    "outer_fold": fold,
                    "role": role,
                    "path": str(path),
                    "sha256": file_sha256(path),
                    "size_bytes": path.stat().st_size,
                }
            )

        inner = normalize_pool(torch.load(inner_path, map_location="cpu"))
        outer = normalize_pool(torch.load(outer_path, map_location="cpu"))
        if set(inner.sample_ids) & set(outer.sample_ids):
            raise RuntimeError(f"outer fold {fold} has inner/outer sample overlap")
        if set(inner.group_ids) & set(outer.group_ids):
            raise RuntimeError(f"outer fold {fold} has inner/outer group overlap")

        actions = outer.actions.detach().cpu().double().numpy()
        labels = outer.labels.detach().cpu().double().numpy().reshape(-1)
        anchor = actions[:, 0]
        bounds = compute_fold_upper_bounds(actions, labels, ACTION_NAMES)
        reference = strategy_predictions(inner, outer, consensus)
        v921 = reference["predictions"]["convex_shrinkage"].detach().cpu().numpy().reshape(-1)
        selected = reference["predictions"]["inner_selected_single"].detach().cpu().numpy().reshape(-1)

        for action_index, action_name in enumerate(ACTION_NAMES):
            action_rows.append(
                {
                    "outer_fold": fold,
                    "action_index": action_index,
                    "action": action_name,
                    "is_fold_posthoc_best": action_index == bounds["best_single_index"],
                    **prediction_metrics(actions[:, action_index], labels, anchor),
                }
            )

        strategies = (
            ("anchor", anchor, True, False, "fixed_anchor"),
            ("inner_selected_single", selected, True, False, "selected_on_inner_oof"),
            ("v921_convex_shrinkage", v921, True, False, "weights_fit_on_inner_oof"),
            ("fold_best_single_cheating", bounds["best_single_prediction"], False, True, "outer_fold_labels"),
            ("fold_local_convex_cheating", bounds["convex_prediction"], False, True, "fit_and_evaluate_on_outer_fold"),
            ("sample_oracle", bounds["sample_oracle_prediction"], False, True, "true_label_per_sample"),
        )
        for name, prediction, deployable, cheating, scope in strategies:
            fold_rows.append(
                {
                    "outer_fold": fold,
                    "fold_posthoc_best_action": bounds["best_single_action"],
                    "inner_selected_single_action": reference["selected_single_action"],
                    **metric_row(name, prediction, labels, anchor, deployable, cheating, scope),
                }
            )
        for action_name, weight in zip(ACTION_NAMES, bounds["convex_weights"]):
            weight_rows.append(
                {
                    "fit_scope": "fold_local_outer_label_cheating",
                    "outer_fold": fold,
                    "action": action_name,
                    "weight": float(weight),
                }
            )

        frame = pd.DataFrame(
            {
                "outer_fold": fold,
                "sample_id": outer.sample_ids,
                "group_id": outer.group_ids,
                "label": labels,
                "prediction_anchor": anchor,
                "prediction_inner_selected_single": selected,
                "prediction_v921_convex_shrinkage": v921,
                "prediction_fold_best_single_cheating": bounds["best_single_prediction"],
                "fold_best_single_action_cheating": bounds["best_single_action"],
                "prediction_fold_local_convex_cheating": bounds["convex_prediction"],
                "prediction_sample_oracle": bounds["sample_oracle_prediction"],
                "sample_oracle_action": bounds["sample_oracle_action"],
            }
        )
        for index, action_name in enumerate(ACTION_NAMES):
            frame[f"action_prediction_{action_name}"] = actions[:, index]
        sample_frames.append(frame)

    samples = pd.concat(sample_frames, ignore_index=True)
    if samples["sample_id"].astype(str).duplicated().any():
        raise RuntimeError("outer aggregation contains duplicate sample IDs")
    labels = samples["label"].to_numpy(dtype=np.float64)
    action_columns = [f"action_prediction_{name}" for name in ACTION_NAMES]
    actions = samples[action_columns].to_numpy(dtype=np.float64)
    anchor = samples["prediction_anchor"].to_numpy(dtype=np.float64)

    action_mae = np.abs(actions - labels[:, None]).mean(axis=0)
    best_global_index = int(action_mae.argmin())
    best_global_action = ACTION_NAMES[best_global_index]
    samples["prediction_best_global_fixed_action_cheating"] = actions[:, best_global_index]
    pooled_weights = fit_mae_simplex(actions, labels)
    samples["prediction_pooled_global_convex_cheating"] = actions @ pooled_weights
    for action_name, weight in zip(ACTION_NAMES, pooled_weights):
        weight_rows.append(
            {
                "fit_scope": "pooled_global_outer_label_cheating",
                "outer_fold": "all",
                "action": action_name,
                "weight": float(weight),
            }
        )

    aggregate_specs = (
        ("anchor", "prediction_anchor", True, False, "fixed_anchor"),
        ("inner_selected_single", "prediction_inner_selected_single", True, False, "selected_on_inner_oof_per_fold"),
        ("v921_convex_shrinkage", "prediction_v921_convex_shrinkage", True, False, "weights_fit_on_inner_oof_per_fold"),
        ("best_global_fixed_action_cheating", "prediction_best_global_fixed_action_cheating", False, True, "one_action_selected_using_all_outer_labels"),
        ("fold_best_single_cheating", "prediction_fold_best_single_cheating", False, True, "one_action_selected_per_fold_using_outer_labels"),
        ("pooled_global_convex_cheating", "prediction_pooled_global_convex_cheating", False, True, "one_weight_vector_fit_on_all_outer_labels"),
        ("fold_local_convex_cheating", "prediction_fold_local_convex_cheating", False, True, "one_weight_vector_fit_per_outer_fold"),
        ("sample_oracle", "prediction_sample_oracle", False, True, "one_discrete_action_per_sample_using_true_label"),
    )
    aggregate_rows = []
    for name, column, deployable, cheating, scope in aggregate_specs:
        row = metric_row(name, samples[column], labels, anchor, deployable, cheating, scope)
        row["below_target_mae"] = row["mae"] < config.target_mae
        aggregate_rows.append(row)
    aggregate = pd.DataFrame(aggregate_rows)
    lookup = {row["strategy"]: {k: v for k, v in row.items() if k != "strategy"} for row in aggregate_rows}
    aggregate["discrete_oracle_gap_closure"] = aggregate.apply(
        lambda row: oracle_gap_closure(
            lookup["anchor"]["mae"], row["mae"], lookup["sample_oracle"]["mae"]
        ),
        axis=1,
    )

    action_aggregate = pd.DataFrame(
        [
            {
                "action_index": index,
                "action": name,
                "is_posthoc_best_global_action": index == best_global_index,
                **prediction_metrics(actions[:, index], labels, anchor),
            }
            for index, name in enumerate(ACTION_NAMES)
        ]
    ).sort_values("mae")
    verdict = make_viability_verdict(
        {name: {"mae": values["mae"]} for name, values in lookup.items()},
        config.target_mae,
    )
    fold_metrics = pd.DataFrame(fold_rows).sort_values(["outer_fold", "mae", "strategy"])
    action_by_fold = pd.DataFrame(action_rows).sort_values(["outer_fold", "mae", "action"])
    weights = pd.DataFrame(weight_rows)
    sources = pd.DataFrame(source_rows).sort_values(["outer_fold", "role"])
    local_weight_sum = weights[weights["fit_scope"] == "fold_local_outer_label_cheating"].groupby("outer_fold")["weight"].sum()
    if (local_weight_sum.sub(1.0).abs() > config.tolerance).any():
        raise RuntimeError("fold-local cheating weights are not simplex-normalized")

    summary = {
        "version": AUDIT_VERSION,
        "method": "strict_saved_prediction_expert_pool_viability_audit",
        "root": str(root),
        "output_dir": str(output),
        "config": asdict(config),
        "consensus_reference_config": {"shrinkage_lambda": float(cli.shrinkage_lambda)},
        "outer_fold_count": int(cli.outer_folds),
        "sample_count": len(samples),
        "action_names": list(ACTION_NAMES),
        "best_global_fixed_action_posthoc": best_global_action,
        "fold_posthoc_best_actions": {
            str(int(fold)): str(local["fold_posthoc_best_action"].iloc[0])
            for fold, local in fold_metrics[fold_metrics["strategy"] == "fold_best_single_cheating"].groupby("outer_fold")
        },
        "aggregate_metrics": lookup,
        "pooled_global_outer_label_cheating_weights": dict(zip(ACTION_NAMES, map(float, pooled_weights))),
        "viability_verdict": verdict,
        "provenance": {
            "uses_only_v919_saved_predictions": True,
            "new_models_trained": False,
            "v921_reference_weights_fit_on_inner_oof_only": True,
            "outer_labels_used_for_diagnostic_upper_bounds": True,
            "outer_label_cheating_methods_are_not_deployable": True,
            "sample_oracle_is_discrete_action_selection": True,
            "different_protocol_results_are_not_loaded": True,
            "inner_outer_sample_disjointness_checked": True,
            "inner_outer_group_disjointness_checked": True,
            "outer_sample_uniqueness_checked": True,
            "source_sha256_recorded": True,
        },
    }

    fold_metrics.to_csv(output / "v929_viability_metrics_by_fold.csv", index=False)
    action_by_fold.to_csv(output / "v929_action_metrics_by_fold.csv", index=False)
    action_aggregate.to_csv(output / "v929_action_metrics_aggregate.csv", index=False)
    weights.to_csv(output / "v929_outer_label_cheating_weights.csv", index=False)
    samples.to_csv(output / "v929_outer_predictions.csv", index=False)
    aggregate.sort_values("mae").to_csv(output / "v929_aggregate_bounds.csv", index=False)
    sources.to_csv(output / "v929_source_manifest.csv", index=False)
    (output / "v929_viability_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    key_fold = fold_metrics[fold_metrics["strategy"].isin(
        ["v921_convex_shrinkage", "fold_best_single_cheating", "fold_local_convex_cheating", "sample_oracle"]
    )]
    report = [
        "# V9.29 Expert-Pool Viability Audit",
        "",
        "No model is trained. Rows marked `label_cheating=True` are diagnostic upper bounds only.",
        "",
        "## Aggregate bounds",
        "",
        markdown_table(aggregate.sort_values("mae"), ["strategy", "deployable", "label_cheating", "mae", "gain_vs_anchor", "below_target_mae", "discrete_oracle_gap_closure"]),
        "",
        "## Single action families",
        "",
        markdown_table(action_aggregate, ["action", "is_posthoc_best_global_action", "mae", "gain_vs_anchor", "large_harm_rate_010"]),
        "",
        "## Key fold-local bounds",
        "",
        markdown_table(key_fold, ["outer_fold", "strategy", "fold_posthoc_best_action", "mae", "gain_vs_anchor"]),
        "",
        "## Decision",
        "",
        f"- Target MAE: `< {config.target_mae:.6f}`",
        f"- V9.21 MAE: `{lookup['v921_convex_shrinkage']['mae']:.6f}`",
        f"- Best global fixed action: `{best_global_action}` / `{lookup['best_global_fixed_action_cheating']['mae']:.6f}`",
        f"- Fold-best single: `{lookup['fold_best_single_cheating']['mae']:.6f}`",
        f"- Pooled cheating convex: `{lookup['pooled_global_convex_cheating']['mae']:.6f}`",
        f"- Fold-local cheating convex: `{lookup['fold_local_convex_cheating']['mae']:.6f}`",
        f"- Discrete sample oracle: `{lookup['sample_oracle']['mae']:.6f}`",
        f"- Verdict: `{verdict['verdict']}`",
        "",
        "Convex interpolation can be closer than every discrete expert, so cheating convex MAE need not be above discrete sample-oracle MAE.",
    ]
    (output / "v929_viability_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("V9.29 EXPERT-POOL VIABILITY AUDIT COMPLETE")
    print("new models trained: False")
    print("best global fixed action:", best_global_action)
    print(
        "V9.21 / pooled-convex / fold-convex / sample-oracle MAE:",
        f"{lookup['v921_convex_shrinkage']['mae']:.6f}",
        f"{lookup['pooled_global_convex_cheating']['mae']:.6f}",
        f"{lookup['fold_local_convex_cheating']['mae']:.6f}",
        f"{lookup['sample_oracle']['mae']:.6f}",
    )
    print("verdict:", verdict["verdict"])
    print("report:", output / "v929_viability_report.md")


if __name__ == "__main__":
    main()
