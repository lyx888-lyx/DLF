"""Engineering audit for V9.25 expert self-risk artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from trains.singleTask.expert_self_risk_v925 import (
    AUDIT_VERSION,
    EXPERT_NAMES,
    ExpertSelfRiskConfigV925,
    binary_metric_row,
    continuous_metric_row,
    group_bootstrap_selected_gain,
    selection_metric_row,
)
from trains.singleTask.same_stack_expert_factory_v919 import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--bootstrap-repetitions", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=1111)
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def close(left, right, atol=2e-6):
    return abs(float(left) - float(right)) <= float(atol)


def main():
    cli = parse_args()
    root = Path(cli.v919_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v925_expert_self_risk_audit"
    )
    names = (
        "v925_summary.json",
        "v925_report.md",
        "v925_inner_oof_risk_predictions.csv",
        "v925_outer_risk_predictions.csv",
        "v925_binary_metrics.csv",
        "v925_continuous_metrics.csv",
        "v925_error_upper_bound_metrics.csv",
        "v925_ranked_subset_metrics.csv",
        "v925_confidently_wrong_metrics.csv",
        "v925_risk_split_manifest.csv",
        "v925_source_integrity.csv",
        "v925_model_inventory.csv",
        "v925_outer_fold_safe_gain.csv",
    )
    for name in names:
        require((output / name).is_file(), f"missing artifact: {name}")
    summary = json.loads(
        (output / "v925_summary.json").read_text()
    )
    inner = pd.read_csv(
        output / "v925_inner_oof_risk_predictions.csv"
    )
    outer = pd.read_csv(output / "v925_outer_risk_predictions.csv")
    binary = pd.read_csv(output / "v925_binary_metrics.csv")
    continuous = pd.read_csv(output / "v925_continuous_metrics.csv")
    upper = pd.read_csv(
        output / "v925_error_upper_bound_metrics.csv"
    )
    ranked = pd.read_csv(output / "v925_ranked_subset_metrics.csv")
    confident = pd.read_csv(
        output / "v925_confidently_wrong_metrics.csv"
    )
    splits = pd.read_csv(output / "v925_risk_split_manifest.csv")
    sources = pd.read_csv(output / "v925_source_integrity.csv")
    models = pd.read_csv(output / "v925_model_inventory.csv")
    folds = pd.read_csv(output / "v925_outer_fold_safe_gain.csv")

    require(summary["version"] == AUDIT_VERSION, "version mismatch")
    provenance = summary["provenance"]
    for key in (
        "base_experts_trained",
        "base_experts_or_predictions_modified",
        "outer_labels_used_for_risk_fit_or_calibration",
        "router_or_action_selection_executed",
        "expert_predictions_replaced_or_mixed",
        "official_validation_or_test_used",
    ):
        require(provenance[key] is False, f"false provenance flag: {key}")
    for key in (
        "risk_heads_trained",
        "risk_heads_low_capacity_linear",
        "inner_oof_only_for_risk_fit",
        "disjoint_inner_group_fold_for_calibration",
        "v921_baseline_fitted_on_inner_oof_only",
    ):
        require(provenance[key] is True, f"true provenance flag: {key}")

    config = ExpertSelfRiskConfigV925(**summary["config"])
    config.validate()
    expected_folds = set(range(cli.outer_folds))
    for frame, name in ((inner, "inner"), (outer, "outer")):
        require(
            set(frame["outer_fold"].astype(int)) == expected_folds,
            f"{name} fold mismatch",
        )
        require(
            set(frame["expert"].astype(str)) == set(EXPERT_NAMES),
            f"{name} expert mismatch",
        )
        require(
            not frame.duplicated(
                ["outer_fold", "sample_id", "expert"]
            ).any(),
            f"duplicate {name} rows",
        )
        numeric = frame.select_dtypes(include=[np.number])
        require(
            bool(np.isfinite(numeric).all().all()),
            f"non-finite {name} values",
        )
        probability_columns = [
            column
            for column in frame
            if column.startswith("pred_")
            and column.endswith("_prob")
        ]
        require(probability_columns, "missing probability columns")
        for column in probability_columns:
            require(
                bool(
                    (
                        (frame[column] >= 0)
                        & (frame[column] <= 1)
                    ).all()
                ),
                f"invalid probability: {column}",
            )
        error = np.abs(
            frame["expert_prediction"] - frame["label"]
        )
        gain = np.abs(
            frame["baseline_prediction"] - frame["label"]
        ) - error
        require(
            np.allclose(
                error, frame["absolute_error"], atol=2e-6
            ),
            f"{name} error mismatch",
        )
        require(
            np.allclose(
                gain, frame["gain_vs_baseline"], atol=2e-6
            ),
            f"{name} gain mismatch",
        )
        require(
            np.array_equal(
                (
                    error > config.error_threshold_050
                ).astype(int),
                frame["large_error_050"].astype(int),
            ),
            f"{name} large error mismatch",
        )
        require(
            np.array_equal(
                (
                    gain < -config.large_harm_threshold
                ).astype(int),
                frame["large_harm_030"].astype(int),
            ),
            f"{name} harm mismatch",
        )

    for fold in expected_folds:
        local_inner = inner[inner["outer_fold"] == fold]
        local_outer = outer[outer["outer_fold"] == fold]
        require(
            set(local_inner["sample_id"]).isdisjoint(
                set(local_outer["sample_id"])
            ),
            f"sample leakage fold {fold}",
        )
        for expert in EXPERT_NAMES:
            local = local_outer[
                local_outer["expert"] == expert
            ]
            expected = max(
                1,
                int(
                    np.ceil(
                        len(local) * config.safe_rank_fraction
                    )
                ),
            )
            for column in (
                "selected_native_top",
                "selected_risk_top",
            ):
                require(
                    int(local[column].astype(bool).sum())
                    == expected,
                    f"selection count mismatch {fold}/{expert}/{column}",
                )

    require(
        len(sources) == cli.outer_folds * 2,
        "source count mismatch",
    )
    for row in sources.itertuples(index=False):
        path = Path(row.path)
        require(path.is_file(), f"missing source: {path}")
        require(
            sha256(path) == row.sha256,
            f"source hash mismatch: {path}",
        )
    require(
        len(models) == cli.outer_folds * len(EXPERT_NAMES),
        "model count mismatch",
    )
    for row in models.itertuples(index=False):
        path = Path(row.model_path)
        require(path.is_file(), f"missing model: {path}")
        require(
            sha256(path) == row.model_sha256,
            f"model hash mismatch: {path}",
        )
        payload = torch.load(path, map_location="cpu")
        require(
            payload["version"] == AUDIT_VERSION,
            "model version mismatch",
        )
        model_provenance = payload["provenance"]
        require(
            model_provenance["base_experts_frozen"] is True,
            "base experts not frozen",
        )
        require(
            model_provenance["risk_fit_on_inner_oof_only"]
            is True,
            "risk fit provenance",
        )
        require(
            model_provenance[
                "outer_labels_used_for_fit_or_calibration"
            ]
            is False,
            "outer leakage flag",
        )
        require(
            model_provenance[
                "router_or_action_selection_present"
            ]
            is False,
            "router flag",
        )

    require(
        bool(
            (
                splits[
                    [
                        "fit_sample_count",
                        "calibration_sample_count",
                        "holdout_sample_count",
                    ]
                ]
                > 0
            )
            .all()
            .all()
        ),
        "empty tri-split",
    )
    require(
        bool(
            (
                splits["holdout_fold"]
                != splits["calibration_fold"]
            ).all()
        ),
        "holdout used for calibration",
    )
    require(
        not binary.empty
        and not continuous.empty
        and not upper.empty,
        "empty metrics",
    )
    require(
        not ranked.empty
        and not confident.empty
        and not folds.empty,
        "empty diagnostics",
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
    win = binary_metric_row(
        outer["win_vs_baseline"],
        outer["pred_win_vs_baseline_prob"],
        config.calibration_bins,
    )
    harm = binary_metric_row(
        outer["large_harm_030"],
        outer["pred_large_harm_030_prob"],
        config.calibration_bins,
    )
    error = continuous_metric_row(
        outer["absolute_error"],
        outer["pred_absolute_error"],
    )
    require(
        close(
            win["auc"],
            aggregate_binary.loc[
                "win_vs_baseline", "auc"
            ],
        ),
        "win AUC mismatch",
    )
    require(
        close(
            harm["auc"],
            aggregate_binary.loc[
                "large_harm_030", "auc"
            ],
        ),
        "harm AUC mismatch",
    )
    require(
        close(
            error["spearman"],
            aggregate_continuous.loc[
                "absolute_error", "spearman"
            ],
        ),
        "error Spearman mismatch",
    )

    risk = selection_metric_row(
        outer, "selected_risk_top", config
    )
    native = selection_metric_row(
        outer, "selected_native_top", config
    )
    bootstrap = group_bootstrap_selected_gain(
        outer,
        "selected_risk_top",
        cli.bootstrap_repetitions,
        cli.bootstrap_seed,
    )
    positive_folds = int(
        sum(
            selection_metric_row(
                local, "selected_risk_top", config
            )["mean_gain_vs_baseline"]
            > 0
            for _, local in outer.groupby("outer_fold")
        )
    )
    gain_over_native = (
        risk["mean_gain_vs_baseline"]
        - native["mean_gain_vs_baseline"]
    )
    expected_pass = bool(
        win["auc"] >= config.required_win_auc
        and harm["auc"] >= config.required_large_harm_auc
        and error["spearman"]
        >= config.required_error_spearman
        and bootstrap["gain_ci_low"]
        > config.required_safe_gain_ci_low
        and positive_folds
        >= config.required_positive_outer_folds
        and gain_over_native
        >= config.required_gain_over_native_rank
        and risk["large_harm_rate_030"]
        <= native["large_harm_rate_030"]
    )
    gate = summary["success_gate"]
    require(
        close(gate["aggregate_win_auc"], win["auc"]),
        "summary win mismatch",
    )
    require(
        close(
            gate["aggregate_large_harm_auc"],
            harm["auc"],
        ),
        "summary harm mismatch",
    )
    require(
        close(
            gate["aggregate_error_spearman"],
            error["spearman"],
        ),
        "summary error mismatch",
    )
    require(
        close(
            gate["risk_top20"]["mean_gain_vs_baseline"],
            risk["mean_gain_vs_baseline"],
        ),
        "summary risk gain mismatch",
    )
    require(
        close(
            gate["risk_top20_bootstrap"]["gain_ci_low"],
            bootstrap["gain_ci_low"],
        ),
        "summary bootstrap mismatch",
    )
    require(
        int(gate["positive_outer_folds"])
        == positive_folds,
        "positive fold mismatch",
    )
    require(
        bool(summary["risk_signal_supported"])
        == expected_pass,
        "gate result mismatch",
    )
    require(
        bool(gate["passed"]) == expected_pass,
        "gate passed mismatch",
    )

    print("V9.25 EXPERT SELF-RISK ENGINEERING AUDIT PASSED")
    print("base experts trained: False")
    print("risk heads trained: True")
    print("router or action selection executed: False")
    print("risk signal supported:", expected_pass)


if __name__ == "__main__":
    main()
