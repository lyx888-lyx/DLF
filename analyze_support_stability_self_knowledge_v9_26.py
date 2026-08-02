"""Run the strict V9.26 support-and-stability self-knowledge audit."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.expert_self_risk_v925 import (
    BINARY_TARGET_NAMES,
    CONTINUOUS_TARGET_NAMES,
    EXPERT_NAMES,
    add_rank_selection_flags,
    binary_metric_row,
    build_expert_features,
    build_expert_targets,
    continuous_metric_row,
    crossfit_risk_predictions,
    fit_outer_risk_bundle,
    group_bootstrap_selected_gain,
    predict_risk_bundle,
    selection_metric_row,
)
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.same_stack_expert_factory_v919 import sha256
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    inner_crossfit_consensus,
    strategy_predictions,
)
from trains.singleTask.support_stability_self_knowledge_v926 import (
    AUDIT_VERSION,
    BASE_METHOD,
    PRIMARY_METHOD,
    SupportStabilityConfigV926,
    average_precision,
    crossfit_support_stability_predictions,
    fit_outer_support_stability_bundle,
    predict_outer_support_stability,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--shrinkage-lambda", type=float, default=0.01)
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1111)
    return parser.parse_args()


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame.loc[:, [name for name in columns if name in frame]].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
            )
    return "\n".join(
        [
            "|" + "|".join(view.columns) + "|",
            "|" + "|".join(["---"] * len(view.columns)) + "|",
            *[
                "|" + "|".join(str(value) for value in row) + "|"
                for row in view.itertuples(index=False, name=None)
            ],
        ]
    )


def check_group_folds(payload, folds) -> None:
    frame = pd.DataFrame(
        {
            "group": [str(value) for value in payload["group_ids"]],
            "fold": np.asarray(folds, dtype=np.int64),
        }
    )
    if bool((frame.groupby("group")["fold"].nunique() != 1).any()):
        raise RuntimeError("conversation group crosses risk folds")


def make_frame(
    payload,
    outer_fold,
    split,
    method,
    expert,
    base_features,
    targets,
    predictions,
    risk_folds=None,
    support_min_distance=None,
):
    n = len(payload["sample_ids"])
    values = {
        "outer_fold": np.full(n, outer_fold),
        "split": np.full(n, split),
        "method": np.full(n, method),
        "sample_id": [str(value) for value in payload["sample_ids"]],
        "group_id": [str(value) for value in payload["group_ids"]],
        "expert": np.full(n, expert),
        "label": torch.as_tensor(payload["labels"]).view(-1).numpy(),
        "anchor_prediction": base_features["anchor_prediction"],
        "baseline_prediction": base_features["baseline_prediction"],
        "expert_prediction": base_features["expert_prediction"],
        "native_confidence": base_features["native_confidence"],
        **targets,
        **predictions,
    }
    if risk_folds is not None:
        values["risk_fold_index"] = np.asarray(risk_folds, dtype=np.int64)
    if support_min_distance is not None:
        values["support_min_distance"] = np.asarray(
            support_min_distance, dtype=np.float64
        )
    return pd.DataFrame(values)


def metric_artifacts(frame, config, split):
    binary_rows = []
    continuous_rows = []
    groups = [
        ((method, "all"), local)
        for method, local in frame.groupby("method", sort=True)
    ]
    groups.extend(
        ((method, expert), local)
        for (method, expert), local in frame.groupby(
            ["method", "expert"], sort=True
        )
    )
    for (method, expert), local in groups:
        for target in BINARY_TARGET_NAMES:
            probability = local[f"pred_{target}_prob"]
            binary_rows.append(
                {
                    "split": split,
                    "method": method,
                    "expert": expert,
                    "target": target,
                    **binary_metric_row(
                        local[target], probability, config.risk.calibration_bins
                    ),
                    "average_precision": average_precision(
                        local[target], probability
                    ),
                }
            )
        for target in CONTINUOUS_TARGET_NAMES:
            continuous_rows.append(
                {
                    "split": split,
                    "method": method,
                    "expert": expert,
                    "target": target,
                    **continuous_metric_row(
                        local[target], local[f"pred_{target}"]
                    ),
                }
            )
    return binary_rows, continuous_rows


def ranked_artifacts(frame, config, split):
    rows = []
    for (method, fold, expert), local in frame.groupby(
        ["method", "outer_fold", "expert"], sort=True
    ):
        for selector in ("selected_native_top", "selected_risk_top"):
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "outer_fold": fold,
                    "expert": expert,
                    "selector": selector,
                    **selection_metric_row(local, selector, config.risk),
                }
            )
    for (method, expert), local in frame.groupby(
        ["method", "expert"], sort=True
    ):
        for selector in ("selected_native_top", "selected_risk_top"):
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "outer_fold": "aggregate",
                    "expert": expert,
                    "selector": selector,
                    **selection_metric_row(local, selector, config.risk),
                }
            )
    for method, local in frame.groupby("method", sort=True):
        for selector in ("selected_native_top", "selected_risk_top"):
            rows.append(
                {
                    "split": split,
                    "method": method,
                    "outer_fold": "aggregate",
                    "expert": "all",
                    "selector": selector,
                    **selection_metric_row(local, selector, config.risk),
                }
            )
    return rows


def main():
    cli = parse_args()
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v926_support_stability_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "risk_models"
    model_dir.mkdir(parents=True, exist_ok=True)

    config = SupportStabilityConfigV926()
    config.validate()
    consensus = ConsensusConfigV921(
        shrinkage_lambda=cli.shrinkage_lambda,
        bootstrap_repetitions=cli.bootstrap_repetitions,
        bootstrap_seed=cli.bootstrap_seed,
    )

    inner_frames = []
    outer_frames = []
    split_rows = []
    source_rows = []
    model_rows = []
    schema_rows = []

    for outer_fold in range(cli.outer_folds):
        fold_dir = root / f"outer_fold_{outer_fold}"
        inner_path = fold_dir / "inner_oof_same_stack_pool_v919.pth"
        outer_path = (
            fold_dir
            / "outer_deployment_stack"
            / "same_stack_target_pool_v919.pth"
        )
        for path in (inner_path, outer_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        inner_payload = torch.load(inner_path, map_location="cpu")
        outer_payload = torch.load(outer_path, map_location="cpu")
        inner_pool = normalize_pool(inner_payload)
        outer_pool = normalize_pool(outer_payload)
        if set(inner_pool.sample_ids) & set(outer_pool.sample_ids):
            raise RuntimeError("inner/outer sample overlap")

        inner_baseline = inner_crossfit_consensus(
            inner_payload, consensus
        )["predictions"]["convex_shrinkage"]
        outer_baseline = strategy_predictions(
            inner_pool, outer_pool, consensus
        )["predictions"]["convex_shrinkage"]
        risk_folds = torch.as_tensor(
            inner_payload["fold_index"]
        ).view(-1).numpy()
        check_group_folds(inner_payload, risk_folds)

        for split, path, pool in (
            ("inner_oof", inner_path, inner_pool),
            ("outer_holdout", outer_path, outer_pool),
        ):
            source_rows.append(
                {
                    "outer_fold": outer_fold,
                    "split": split,
                    "path": str(path),
                    "sha256": sha256(path),
                    "sample_count": len(pool.labels),
                }
            )

        for expert in EXPERT_NAMES:
            inner_base = build_expert_features(
                inner_payload, inner_baseline, expert
            )
            inner_targets = build_expert_targets(
                inner_payload, inner_baseline, expert, config.risk
            )
            base_crossfit = crossfit_risk_predictions(
                inner_base["matrix"],
                inner_targets,
                risk_folds,
                config.risk,
            )
            inner_frames.append(
                make_frame(
                    inner_payload,
                    outer_fold,
                    "inner_oof",
                    BASE_METHOD,
                    expert,
                    inner_base,
                    inner_targets,
                    base_crossfit["predictions"],
                    risk_folds,
                )
            )

            support_crossfit = crossfit_support_stability_predictions(
                inner_payload,
                inner_baseline,
                expert,
                risk_folds,
                config,
            )
            inner_frames.append(
                make_frame(
                    inner_payload,
                    outer_fold,
                    "inner_oof",
                    PRIMARY_METHOD,
                    expert,
                    support_crossfit["inputs"]["base"],
                    inner_targets,
                    support_crossfit["predictions"],
                    risk_folds,
                )
            )
            for method, rows in (
                (BASE_METHOD, base_crossfit["split_rows"]),
                (PRIMARY_METHOD, support_crossfit["split_rows"]),
            ):
                for row in rows:
                    split_rows.append(
                        {
                            "outer_fold": outer_fold,
                            "expert": expert,
                            "method": method,
                            **row,
                        }
                    )

            base_fitted = fit_outer_risk_bundle(
                inner_base["matrix"],
                inner_targets,
                risk_folds,
                config.risk,
            )
            outer_base = build_expert_features(
                outer_payload, outer_baseline, expert
            )
            outer_targets = build_expert_targets(
                outer_payload, outer_baseline, expert, config.risk
            )
            outer_base_predictions = predict_risk_bundle(
                base_fitted["bundle"],
                outer_base["matrix"],
                config.risk,
            )
            outer_frames.append(
                make_frame(
                    outer_payload,
                    outer_fold,
                    "outer_holdout",
                    BASE_METHOD,
                    expert,
                    outer_base,
                    outer_targets,
                    outer_base_predictions,
                )
            )

            support_fitted = fit_outer_support_stability_bundle(
                inner_payload,
                inner_baseline,
                expert,
                risk_folds,
                config,
            )
            outer_support = predict_outer_support_stability(
                support_fitted,
                outer_payload,
                outer_baseline,
                expert,
                config,
            )
            outer_frames.append(
                make_frame(
                    outer_payload,
                    outer_fold,
                    "outer_holdout",
                    PRIMARY_METHOD,
                    expert,
                    outer_support["inputs"]["base"],
                    outer_targets,
                    outer_support["predictions"],
                    support_min_distance=outer_support["support"][
                        "minimum_neighbour_distance"
                    ],
                )
            )

            model_path = (
                model_dir
                / f"outer_fold_{outer_fold}_{expert}_support_risk_v926.pth"
            )
            torch.save(
                {
                    **support_fitted,
                    "outer_fold": outer_fold,
                    "expert": expert,
                    "provenance": {
                        "base_experts_frozen": True,
                        "v921_baseline_frozen_after_inner_fit": True,
                        "support_library_from_inner_fit_groups_only": True,
                        "same_conversation_neighbours_excluded": True,
                        "outer_labels_used_for_fit_calibration_or_support": False,
                        "router_or_action_selection_present": False,
                        "expert_predictions_replaced_or_mixed": False,
                        "raw_input_perturbation_claimed": False,
                        "cross_view_stability_from_saved_branches": True,
                    },
                },
                model_path,
            )
            model_rows.append(
                {
                    "outer_fold": outer_fold,
                    "expert": expert,
                    "model_path": str(model_path),
                    "model_sha256": sha256(model_path),
                    "calibration_fold": support_fitted[
                        "calibration_fold"
                    ],
                    "fit_sample_count": support_fitted[
                        "fit_sample_count"
                    ],
                    "calibration_sample_count": support_fitted[
                        "calibration_sample_count"
                    ],
                    "support_reference_count": support_fitted[
                        "support_reference_count"
                    ],
                    "feature_count": len(
                        support_fitted["feature_names"]
                    ),
                }
            )
            for index, name in enumerate(support_fitted["feature_names"]):
                schema_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "expert": expert,
                        "feature_index": index,
                        "feature_name": name,
                    }
                )

    inner = add_rank_selection_flags(
        pd.concat(inner_frames, ignore_index=True),
        config.risk.safe_rank_fraction,
        group_columns=("method", "outer_fold", "expert"),
    )
    outer = add_rank_selection_flags(
        pd.concat(outer_frames, ignore_index=True),
        config.risk.safe_rank_fraction,
        group_columns=("method", "outer_fold", "expert"),
    )
    if outer.duplicated(
        ["method", "outer_fold", "sample_id", "expert"]
    ).any():
        raise RuntimeError("duplicate outer method/expert/sample row")

    binary_rows = []
    continuous_rows = []
    for frame, split in (
        (inner, "inner_oof"),
        (outer, "outer_holdout"),
    ):
        binary, continuous = metric_artifacts(frame, config, split)
        binary_rows.extend(binary)
        continuous_rows.extend(continuous)
    binary = pd.DataFrame(binary_rows)
    continuous = pd.DataFrame(continuous_rows)
    ranked = pd.DataFrame(
        [
            *ranked_artifacts(inner, config, "inner_oof"),
            *ranked_artifacts(outer, config, "outer_holdout"),
        ]
    )

    support_outer = outer[outer["method"] == PRIMARY_METHOD].copy()
    base_outer = outer[outer["method"] == BASE_METHOD].copy()
    support_selected = selection_metric_row(
        support_outer, "selected_risk_top", config.risk
    )
    base_selected = selection_metric_row(
        base_outer, "selected_risk_top", config.risk
    )
    native_selected = selection_metric_row(
        support_outer, "selected_native_top", config.risk
    )
    support_bootstrap = group_bootstrap_selected_gain(
        support_outer,
        "selected_risk_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed,
    )
    base_bootstrap = group_bootstrap_selected_gain(
        base_outer,
        "selected_risk_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed + 1009,
    )

    fold_rows = []
    for fold, local in support_outer.groupby("outer_fold", sort=True):
        metrics = selection_metric_row(
            local, "selected_risk_top", config.risk
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "support_top20_gain": metrics[
                    "mean_gain_vs_baseline"
                ],
                "support_top20_win_rate": metrics["win_rate"],
                "support_top20_large_harm_rate": metrics[
                    "large_harm_rate_030"
                ],
                "support_distance_mean": float(
                    local.loc[
                        local["selected_risk_top"],
                        "support_min_distance",
                    ].mean()
                ),
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    positive_folds = int(
        (fold_frame["support_top20_gain"] > 0.0).sum()
    )

    aggregate_binary = binary[
        (binary["split"] == "outer_holdout")
        & (binary["expert"] == "all")
    ].set_index(["method", "target"])
    aggregate_continuous = continuous[
        (continuous["split"] == "outer_holdout")
        & (continuous["expert"] == "all")
    ].set_index(["method", "target"])

    support_win_auc = float(
        aggregate_binary.loc[
            (PRIMARY_METHOD, "win_vs_baseline"), "auc"
        ]
    )
    base_win_auc = float(
        aggregate_binary.loc[
            (BASE_METHOD, "win_vs_baseline"), "auc"
        ]
    )
    support_harm_auc = float(
        aggregate_binary.loc[
            (PRIMARY_METHOD, "large_harm_030"), "auc"
        ]
    )
    support_harm_ap = float(
        aggregate_binary.loc[
            (PRIMARY_METHOD, "large_harm_030"), "average_precision"
        ]
    )
    support_gain_spearman = float(
        aggregate_continuous.loc[
            (PRIMARY_METHOD, "gain_vs_baseline"), "spearman"
        ]
    )
    base_gain_spearman = float(
        aggregate_continuous.loc[
            (BASE_METHOD, "gain_vs_baseline"), "spearman"
        ]
    )
    gain_over_native = (
        support_selected["mean_gain_vs_baseline"]
        - native_selected["mean_gain_vs_baseline"]
    )
    win_auc_gain = support_win_auc - base_win_auc
    gain_spearman_gain = support_gain_spearman - base_gain_spearman

    passed = bool(
        support_win_auc >= config.required_win_auc
        and support_gain_spearman >= config.required_gain_spearman
        and support_harm_auc >= config.required_large_harm_auc
        and support_bootstrap["gain_ci_low"]
        > config.required_safe_gain_ci_low
        and positive_folds >= config.required_positive_outer_folds
        and gain_over_native >= config.required_gain_over_native_rank
        and support_selected["large_harm_rate_030"]
        <= native_selected["large_harm_rate_030"]
        and win_auc_gain >= config.required_win_auc_gain_over_v925
        and gain_spearman_gain
        >= config.required_gain_spearman_gain_over_v925
    )

    summary = {
        "version": AUDIT_VERSION,
        "method": "strict_support_and_cross_view_stability_self_knowledge",
        "config": asdict(config),
        "outer_fold_count": cli.outer_folds,
        "outer_rows_per_method": int(len(support_outer)),
        "risk_signal_supported": passed,
        "success_gate": {
            "passed": passed,
            "support_win_auc": support_win_auc,
            "v925_base_win_auc": base_win_auc,
            "win_auc_gain_over_v925": win_auc_gain,
            "support_gain_spearman": support_gain_spearman,
            "v925_base_gain_spearman": base_gain_spearman,
            "gain_spearman_gain_over_v925": gain_spearman_gain,
            "support_large_harm_auc": support_harm_auc,
            "support_large_harm_average_precision": support_harm_ap,
            "support_top20": support_selected,
            "v925_base_top20": base_selected,
            "native_confidence_top20": native_selected,
            "support_gain_over_native_top20": gain_over_native,
            "support_top20_bootstrap": support_bootstrap,
            "v925_base_top20_bootstrap": base_bootstrap,
            "positive_outer_folds": positive_folds,
        },
        "interpretation": (
            "Historical support and cross-view stability justify one separate pre-registered gate test on untouched data; V9.26 itself performs no routing."
            if passed
            else "Do not build a dynamic gate from V9.26: historical support and cross-view stability did not identify stable expert advantage over V9.21."
        ),
        "provenance": {
            "base_experts_trained_or_modified": False,
            "v921_predictions_modified": False,
            "risk_heads_low_capacity_linear": True,
            "support_uses_inner_fit_labels_only": True,
            "same_conversation_neighbours_excluded": True,
            "outer_labels_used_for_fit_calibration_or_support": False,
            "router_or_action_selection_executed": False,
            "expert_predictions_replaced_or_mixed": False,
            "true_raw_input_perturbations_performed": False,
            "stability_source": "saved_language_audio_vision_fusion_and_expert_predictions",
            "official_validation_or_test_used": False,
        },
    }

    inner.to_csv(output / "v926_inner_risk_predictions.csv", index=False)
    outer.to_csv(output / "v926_outer_risk_predictions.csv", index=False)
    binary.to_csv(output / "v926_binary_metrics.csv", index=False)
    continuous.to_csv(output / "v926_continuous_metrics.csv", index=False)
    ranked.to_csv(output / "v926_ranked_subset_metrics.csv", index=False)
    fold_frame.to_csv(output / "v926_outer_fold_safe_gain.csv", index=False)
    pd.DataFrame(split_rows).to_csv(
        output / "v926_risk_split_manifest.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(
        output / "v926_source_integrity.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(
        output / "v926_model_inventory.csv", index=False
    )
    pd.DataFrame(schema_rows).to_csv(
        output / "v926_feature_schema.csv", index=False
    )
    (output / "v926_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    binary_report = binary[
        (binary["split"] == "outer_holdout")
        & (binary["expert"] == "all")
        & binary["target"].isin(
            ["win_vs_baseline", "large_gain_010", "large_harm_030"]
        )
    ]
    continuous_report = continuous[
        (continuous["split"] == "outer_holdout")
        & (continuous["expert"] == "all")
    ]
    ranked_report = ranked[
        (ranked["split"] == "outer_holdout")
        & (ranked["outer_fold"].astype(str) == "aggregate")
        & (ranked["expert"] == "all")
    ]
    report = [
        "# V9.26 Support-and-Stability Self-Knowledge Audit",
        "",
        "Frozen V9.19 specialists and V9.21 predictions are unchanged. V9.26 adds strict inner-only historical-neighbour support and deterministic cross-view stability from saved language/audio/vision/fusion outputs. It does not perform raw-input perturbations, routing, prediction replacement, or mixture.",
        "",
        "## Aggregate binary comparison",
        "",
        markdown_table(
            binary_report,
            [
                "method",
                "target",
                "positive_rate",
                "auc",
                "average_precision",
                "brier",
                "ece",
            ],
        ),
        "",
        "## Aggregate continuous comparison",
        "",
        markdown_table(
            continuous_report,
            [
                "method",
                "target",
                "target_mean",
                "prediction_mean",
                "mae",
                "pearson",
                "spearman",
            ],
        ),
        "",
        "## Fixed top-20% diagnostic strata",
        "",
        markdown_table(
            ranked_report,
            [
                "method",
                "selector",
                "coverage",
                "mean_gain_vs_baseline",
                "win_rate",
                "large_harm_rate_030",
                "mean_absolute_error",
                "membership_rate",
            ],
        ),
        "",
        "## Outer-fold support-ranked gain",
        "",
        markdown_table(
            fold_frame,
            [
                "outer_fold",
                "support_top20_gain",
                "support_top20_win_rate",
                "support_top20_large_harm_rate",
                "support_distance_mean",
            ],
        ),
        "",
        "## Verdict",
        "",
        f"- Risk signal supported: `{passed}`",
        f"- Support win AUC: `{support_win_auc:.6f}`",
        f"- V9.25 base win AUC: `{base_win_auc:.6f}`",
        f"- Win-AUC gain: `{win_auc_gain:.6f}`",
        f"- Support gain Spearman: `{support_gain_spearman:.6f}`",
        f"- V9.25 base gain Spearman: `{base_gain_spearman:.6f}`",
        f"- Gain-Spearman improvement: `{gain_spearman_gain:.6f}`",
        f"- Large-harm AUC / AP: `{support_harm_auc:.6f}` / `{support_harm_ap:.6f}`",
        f"- Support top-20% gain: `{support_selected['mean_gain_vs_baseline']:.6f}`",
        f"- Support top-20% gain CI: `[{support_bootstrap['gain_ci_low']:.6f}, {support_bootstrap['gain_ci_high']:.6f}]`",
        f"- Positive outer folds: `{positive_folds}/{cli.outer_folds}`",
        "",
        summary["interpretation"],
    ]
    (output / "v926_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print("V9.26 SUPPORT-STABILITY AUDIT COMPLETE")
    print("base experts trained or modified: False")
    print("raw input perturbations performed: False")
    print("router or action selection executed: False")
    print("risk signal supported:", passed)
    print("report:", output / "v926_report.md")


if __name__ == "__main__":
    main()
