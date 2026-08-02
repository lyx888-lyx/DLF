"""Run the strict V9.25 expert self-risk audit on saved V9.19 pools."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.expert_self_risk_v925 import (
    AUDIT_VERSION,
    BINARY_TARGET_NAMES,
    CONTINUOUS_TARGET_NAMES,
    EXPERT_NAMES,
    ExpertSelfRiskConfigV925,
    add_rank_selection_flags,
    binary_metric_row,
    build_expert_features,
    build_expert_targets,
    confidently_wrong_metrics,
    continuous_metric_row,
    crossfit_risk_predictions,
    fit_outer_risk_bundle,
    group_bootstrap_selected_gain,
    predict_risk_bundle,
    selection_metric_row,
    upper_bound_metric_row,
)
from trains.singleTask.no_train_decomposition_v920 import normalize_pool
from trains.singleTask.region_cost_reporting_v918 import jsonable
from trains.singleTask.same_stack_expert_factory_v919 import sha256
from trains.singleTask.static_dense_expert_consensus_v921 import (
    ConsensusConfigV921,
    inner_crossfit_consensus,
    strategy_predictions,
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


def table(frame, columns):
    view = frame.loc[:, columns].copy()
    for column in view.columns:
        if pd.api.types.is_float_dtype(view[column]):
            view[column] = view[column].map(
                lambda value: "" if pd.isna(value) else f"{value:.6f}"
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


def check_group_folds(payload, folds):
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
    expert,
    features,
    targets,
    predictions,
    risk_folds=None,
):
    n = len(payload["sample_ids"])
    values = {
        "outer_fold": np.full(n, outer_fold),
        "split": np.full(n, split),
        "sample_id": [str(value) for value in payload["sample_ids"]],
        "group_id": [str(value) for value in payload["group_ids"]],
        "expert": np.full(n, expert),
        "label": torch.as_tensor(payload["labels"]).view(-1).numpy(),
        "anchor_prediction": features["anchor_prediction"],
        "baseline_prediction": features["baseline_prediction"],
        "expert_prediction": features["expert_prediction"],
        "native_confidence": features["native_confidence"],
        **targets,
        **predictions,
    }
    if risk_folds is not None:
        values["risk_fold_index"] = np.asarray(risk_folds, dtype=np.int64)
    return pd.DataFrame(values)


def metric_artifacts(frame, config, split):
    binary, continuous, upper, confident = [], [], [], []
    groups = [("all", frame), *list(frame.groupby("expert", sort=True))]
    for expert, local in groups:
        for target in BINARY_TARGET_NAMES:
            binary.append(
                {
                    "split": split,
                    "expert": expert,
                    "target": target,
                    "predictor": "calibrated_self_risk",
                    **binary_metric_row(
                        local[target],
                        local[f"pred_{target}_prob"],
                        config.calibration_bins,
                    ),
                }
            )
        for target in CONTINUOUS_TARGET_NAMES:
            continuous.append(
                {
                    "split": split,
                    "expert": expert,
                    "target": target,
                    **continuous_metric_row(
                        local[target], local[f"pred_{target}"]
                    ),
                }
            )
        for quantile in (50, 80, 95):
            upper.append(
                {
                    "split": split,
                    "expert": expert,
                    "nominal_coverage": quantile / 100.0,
                    **upper_bound_metric_row(
                        local["absolute_error"],
                        local[f"pred_error_upper_{quantile}"],
                    ),
                }
            )
        for target in ("membership", "win_vs_baseline"):
            binary.append(
                {
                    "split": split,
                    "expert": expert,
                    "target": target,
                    "predictor": "native_confidence",
                    **binary_metric_row(
                        local[target],
                        local["native_confidence"],
                        config.calibration_bins,
                    ),
                }
            )
        if expert != "all":
            confident.append(
                {
                    "split": split,
                    "expert": expert,
                    **confidently_wrong_metrics(local, config),
                }
            )
    return binary, continuous, upper, confident


def ranked_artifacts(frame, config, split):
    rows = []
    for fold, fold_frame in frame.groupby("outer_fold", sort=True):
        for expert, local in [
            ("all", fold_frame),
            *list(fold_frame.groupby("expert", sort=True)),
        ]:
            for selector in ("selected_native_top", "selected_risk_top"):
                rows.append(
                    {
                        "split": split,
                        "outer_fold": fold,
                        "expert": expert,
                        "selector": selector,
                        **selection_metric_row(local, selector, config),
                    }
                )
    for expert, local in [
        ("all", frame), *list(frame.groupby("expert", sort=True))
    ]:
        for selector in ("selected_native_top", "selected_risk_top"):
            rows.append(
                {
                    "split": split,
                    "outer_fold": "aggregate",
                    "expert": expert,
                    "selector": selector,
                    **selection_metric_row(local, selector, config),
                }
            )
    return rows


def main():
    cli = parse_args()
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v925_expert_self_risk_audit"
    )
    output.mkdir(parents=True, exist_ok=True)
    model_dir = output / "risk_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    config = ExpertSelfRiskConfigV925()
    config.validate()
    consensus = ConsensusConfigV921(
        shrinkage_lambda=cli.shrinkage_lambda,
        bootstrap_repetitions=cli.bootstrap_repetitions,
        bootstrap_seed=cli.bootstrap_seed,
    )

    inner_frames, outer_frames = [], []
    split_rows, source_rows, model_rows = [], [], []
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
            inner_features = build_expert_features(
                inner_payload, inner_baseline, expert
            )
            inner_targets = build_expert_targets(
                inner_payload, inner_baseline, expert, config
            )
            crossfit = crossfit_risk_predictions(
                inner_features["matrix"],
                inner_targets,
                risk_folds,
                config,
            )
            for row in crossfit["split_rows"]:
                split_rows.append(
                    {
                        "outer_fold": outer_fold,
                        "expert": expert,
                        **row,
                    }
                )
            inner_frames.append(
                make_frame(
                    inner_payload,
                    outer_fold,
                    "inner_oof",
                    expert,
                    inner_features,
                    inner_targets,
                    crossfit["predictions"],
                    risk_folds,
                )
            )

            fitted = fit_outer_risk_bundle(
                inner_features["matrix"],
                inner_targets,
                risk_folds,
                config,
            )
            outer_features = build_expert_features(
                outer_payload, outer_baseline, expert
            )
            outer_predictions = predict_risk_bundle(
                fitted["bundle"], outer_features["matrix"], config
            )
            outer_targets = build_expert_targets(
                outer_payload, outer_baseline, expert, config
            )
            outer_frames.append(
                make_frame(
                    outer_payload,
                    outer_fold,
                    "outer_holdout",
                    expert,
                    outer_features,
                    outer_targets,
                    outer_predictions,
                )
            )
            model_path = (
                model_dir
                / f"outer_fold_{outer_fold}_{expert}_risk_v925.pth"
            )
            torch.save(
                {
                    "version": AUDIT_VERSION,
                    "outer_fold": outer_fold,
                    "expert": expert,
                    "feature_names": inner_features["feature_names"],
                    "config": asdict(config),
                    "fit": fitted,
                    "provenance": {
                        "base_experts_frozen": True,
                        "risk_fit_on_inner_oof_only": True,
                        "calibration_on_disjoint_inner_group_fold": True,
                        "outer_labels_used_for_fit_or_calibration": False,
                        "router_or_action_selection_present": False,
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
                    "calibration_fold": fitted["calibration_fold"],
                    "fit_sample_count": fitted["fit_sample_count"],
                    "calibration_sample_count": fitted[
                        "calibration_sample_count"
                    ],
                    "feature_count": len(
                        inner_features["feature_names"]
                    ),
                }
            )

    inner = add_rank_selection_flags(
        pd.concat(inner_frames, ignore_index=True),
        config.safe_rank_fraction,
    )
    outer = add_rank_selection_flags(
        pd.concat(outer_frames, ignore_index=True),
        config.safe_rank_fraction,
    )
    if outer.duplicated(
        ["outer_fold", "sample_id", "expert"]
    ).any():
        raise RuntimeError("duplicate outer expert/sample row")

    binary_rows, continuous_rows = [], []
    upper_rows, confident_rows = [], []
    for frame, split in (
        (inner, "inner_oof"),
        (outer, "outer_holdout"),
    ):
        binary, continuous, upper, confident = metric_artifacts(
            frame, config, split
        )
        binary_rows.extend(binary)
        continuous_rows.extend(continuous)
        upper_rows.extend(upper)
        confident_rows.extend(confident)
    ranked_rows = ranked_artifacts(inner, config, "inner_oof")
    ranked_rows.extend(
        ranked_artifacts(outer, config, "outer_holdout")
    )

    binary = pd.DataFrame(binary_rows)
    continuous = pd.DataFrame(continuous_rows)
    upper = pd.DataFrame(upper_rows)
    ranked = pd.DataFrame(ranked_rows)
    confident = pd.DataFrame(confident_rows)
    risk_selected = selection_metric_row(
        outer, "selected_risk_top", config
    )
    native_selected = selection_metric_row(
        outer, "selected_native_top", config
    )
    risk_bootstrap = group_bootstrap_selected_gain(
        outer,
        "selected_risk_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed,
    )
    native_bootstrap = group_bootstrap_selected_gain(
        outer,
        "selected_native_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed + 1009,
    )
    fold_rows = []
    for fold, local in outer.groupby("outer_fold", sort=True):
        metrics = selection_metric_row(
            local, "selected_risk_top", config
        )
        fold_rows.append(
            {
                "outer_fold": fold,
                "risk_top20_gain": metrics[
                    "mean_gain_vs_baseline"
                ],
                "risk_top20_win_rate": metrics["win_rate"],
                "risk_top20_large_harm_rate": metrics[
                    "large_harm_rate_030"
                ],
            }
        )
    fold_frame = pd.DataFrame(fold_rows)
    positive_folds = int(
        (fold_frame["risk_top20_gain"] > 0.0).sum()
    )

    aggregate_binary = binary[
        (binary["split"] == "outer_holdout")
        & (binary["expert"] == "all")
        & (binary["predictor"] == "calibrated_self_risk")
    ].set_index("target")
    aggregate_continuous = continuous[
        (continuous["split"] == "outer_holdout")
        & (continuous["expert"] == "all")
    ].set_index("target")
    win_auc = float(
        aggregate_binary.loc["win_vs_baseline", "auc"]
    )
    harm_auc = float(
        aggregate_binary.loc["large_harm_030", "auc"]
    )
    error_spearman = float(
        aggregate_continuous.loc["absolute_error", "spearman"]
    )
    gain_over_native = (
        risk_selected["mean_gain_vs_baseline"]
        - native_selected["mean_gain_vs_baseline"]
    )
    passed = bool(
        win_auc >= config.required_win_auc
        and harm_auc >= config.required_large_harm_auc
        and error_spearman >= config.required_error_spearman
        and risk_bootstrap["gain_ci_low"]
        > config.required_safe_gain_ci_low
        and positive_folds
        >= config.required_positive_outer_folds
        and gain_over_native
        >= config.required_gain_over_native_rank
        and risk_selected["large_harm_rate_030"]
        <= native_selected["large_harm_rate_030"]
    )
    summary = {
        "version": AUDIT_VERSION,
        "method": "strict_oof_expert_self_risk_estimation",
        "config": asdict(config),
        "outer_fold_count": cli.outer_folds,
        "outer_sample_expert_rows": len(outer),
        "risk_signal_supported": passed,
        "success_gate": {
            "passed": passed,
            "aggregate_win_auc": win_auc,
            "aggregate_large_harm_auc": harm_auc,
            "aggregate_error_spearman": error_spearman,
            "risk_top20": risk_selected,
            "native_confidence_top20": native_selected,
            "risk_gain_over_native_top20": gain_over_native,
            "risk_top20_bootstrap": risk_bootstrap,
            "native_top20_bootstrap": native_bootstrap,
            "positive_outer_folds": positive_folds,
        },
        "interpretation": (
            "Self-risk signals justify a separate pre-registered gate on untouched data; V9.25 itself performs no routing."
            if passed
            else "Do not build a dynamic expert gate from these signals; self-knowledge was not stable on unseen conversations."
        ),
        "provenance": {
            "base_experts_trained": False,
            "base_experts_or_predictions_modified": False,
            "risk_heads_trained": True,
            "risk_heads_low_capacity_linear": True,
            "inner_oof_only_for_risk_fit": True,
            "disjoint_inner_group_fold_for_calibration": True,
            "outer_labels_used_for_risk_fit_or_calibration": False,
            "router_or_action_selection_executed": False,
            "expert_predictions_replaced_or_mixed": False,
            "v921_baseline_fitted_on_inner_oof_only": True,
            "official_validation_or_test_used": False,
        },
    }

    inner.to_csv(
        output / "v925_inner_oof_risk_predictions.csv", index=False
    )
    outer.to_csv(
        output / "v925_outer_risk_predictions.csv", index=False
    )
    binary.to_csv(output / "v925_binary_metrics.csv", index=False)
    continuous.to_csv(
        output / "v925_continuous_metrics.csv", index=False
    )
    upper.to_csv(
        output / "v925_error_upper_bound_metrics.csv", index=False
    )
    ranked.to_csv(
        output / "v925_ranked_subset_metrics.csv", index=False
    )
    confident.to_csv(
        output / "v925_confidently_wrong_metrics.csv", index=False
    )
    pd.DataFrame(split_rows).to_csv(
        output / "v925_risk_split_manifest.csv", index=False
    )
    pd.DataFrame(source_rows).to_csv(
        output / "v925_source_integrity.csv", index=False
    )
    pd.DataFrame(model_rows).to_csv(
        output / "v925_model_inventory.csv", index=False
    )
    fold_frame.to_csv(
        output / "v925_outer_fold_safe_gain.csv", index=False
    )
    (output / "v925_summary.json").write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    binary_report = binary[
        (binary["split"] == "outer_holdout")
        & (binary["expert"] == "all")
        & (binary["predictor"] == "calibrated_self_risk")
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
        "# V9.25 Expert Self-Risk Audit",
        "",
        "Frozen V9.19 specialists are unchanged. Calibrated linear risk heads are fitted on strict inner OOF pools and evaluated once on outer conversation holdouts. No expert is selected and no prediction is replaced or mixed.",
        "",
        "## Aggregate binary risk prediction",
        "",
        table(
            binary_report,
            ["target", "positive_rate", "auc", "brier", "ece"],
        ),
        "",
        "## Aggregate continuous risk prediction",
        "",
        table(
            continuous_report,
            [
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
        table(
            ranked_report,
            [
                "selector",
                "coverage",
                "mean_gain_vs_baseline",
                "win_rate",
                "large_harm_rate_030",
                "mean_absolute_error",
            ],
        ),
        "",
        "## Outer-fold risk-ranked gain",
        "",
        table(
            fold_frame,
            [
                "outer_fold",
                "risk_top20_gain",
                "risk_top20_win_rate",
                "risk_top20_large_harm_rate",
            ],
        ),
        "",
        "## Verdict",
        "",
        f"- Risk signal supported: `{passed}`",
        f"- Win AUC: `{win_auc:.6f}`",
        f"- Large-harm AUC: `{harm_auc:.6f}`",
        f"- Absolute-error Spearman: `{error_spearman:.6f}`",
        f"- Risk top-20% gain: `{risk_selected['mean_gain_vs_baseline']:.6f}`",
        f"- Risk top-20% gain CI: `[{risk_bootstrap['gain_ci_low']:.6f}, {risk_bootstrap['gain_ci_high']:.6f}]`",
        f"- Gain over native-confidence top-20%: `{gain_over_native:.6f}`",
        f"- Positive outer folds: `{positive_folds}/{cli.outer_folds}`",
        "",
        summary["interpretation"],
    ]
    (output / "v925_report.md").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )
    print("V9.25 EXPERT SELF-RISK AUDIT COMPLETE")
    print("base experts trained: False")
    print("risk heads trained: True")
    print("router or action selection executed: False")
    print("risk signal supported:", passed)
    print("report:", output / "v925_report.md")


if __name__ == "__main__":
    main()
