"""Engineering audit for V9.24 region-gradient consolidation artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from trains.singleTask.gradient_conflict_audit_v923 import (
    FUSION_TAIL_MODULES,
)
from trains.singleTask.region_gradient_consolidation_v924 import (
    PRIMARY_STRATEGY,
    STRATEGIES,
    TRAINING_VERSION,
    SuccessGateConfigV924,
    regression_metrics,
)
from trains.singleTask.role_conditioned_experts_v9 import REGION_NAMES
from trains.singleTask.same_stack_expert_factory_v919 import sha256


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    return parser.parse_args()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def close(
    left: float,
    right: float,
    tolerance: float = 2e-6,
) -> bool:
    return abs(float(left) - float(right)) <= float(tolerance)


def _metrics(frame: pd.DataFrame, strategy: str):
    prediction = torch.tensor(
        frame[f"prediction_{strategy}"].to_numpy(),
        dtype=torch.float32,
    )
    labels = torch.tensor(
        frame["label"].to_numpy(),
        dtype=torch.float32,
    )
    anchor = torch.tensor(
        frame["prediction_anchor"].to_numpy(),
        dtype=torch.float32,
    )
    return regression_metrics(prediction, labels, anchor=anchor)


def main():
    cli = parse_args()
    root = Path(cli.v919_root)
    experiment = root / "v924_region_gradient_consolidation"
    all_frames = []
    selected_name_sets = []

    for fold in range(int(cli.outer_folds)):
        fold_dir = experiment / f"outer_fold_{fold}"
        summary_path = fold_dir / "v924_fold_summary.json"
        history_path = fold_dir / "v924_training_history.csv"
        weights_path = (
            fold_dir / "v924_region_weights_by_epoch.csv"
        )
        prediction_path = (
            fold_dir / "v924_outer_holdout_predictions.csv"
        )
        valid_prediction_path = (
            fold_dir / "v924_inner_valid_predictions.csv"
        )
        for path in (
            summary_path,
            history_path,
            weights_path,
            prediction_path,
            valid_prediction_path,
        ):
            require(
                path.is_file(),
                f"missing V9.24 artifact: {path}",
            )

        summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        require(
            summary["version"] == TRAINING_VERSION,
            "fold version mismatch",
        )
        require(summary["outer_fold"] == fold, "outer fold mismatch")
        require(
            summary["primary_strategy"] == PRIMARY_STRATEGY,
            "primary mismatch",
        )
        require(
            tuple(summary["strategies"]) == STRATEGIES,
            "strategy set mismatch",
        )
        provenance = summary["provenance"]
        for key in (
            "one_model_one_scalar_output",
            "same_initial_anchor_for_all_strategies",
            "fusion_tail_only_trainable",
            "full_batch_exact_mean_gradients_per_epoch",
            "plain_sgd_preserves_analyzed_direction",
            "inner_train_only_for_parameter_updates",
            "inner_valid_only_for_checkpoint_selection",
            "outer_holdout_evaluated_once_after_selection",
            "outer_holdout_not_used_for_hyperparameters",
        ):
            require(
                provenance.get(key) is True,
                f"false provenance: {key}",
            )
        for key in (
            "router_used",
            "consensus_or_residual_head_added",
            "expert_predictions_or_distillation_used",
            "official_validation_used",
            "official_test_used",
        ):
            require(
                provenance.get(key) is False,
                f"forbidden provenance: {key}",
            )

        source = Path(summary["source_checkpoint"])
        require(
            source.is_file(),
            f"missing source checkpoint: {source}",
        )
        current_hash = sha256(source)
        require(
            current_hash
            == summary["source_checkpoint_sha256_before"],
            "source hash differs from before",
        )
        require(
            current_hash
            == summary["source_checkpoint_sha256_after"],
            "source hash differs from after",
        )

        manifest = pd.read_csv(
            root / f"outer_fold_{fold}" / "outer_manifest.csv"
        )
        train_ids = set(
            manifest[
                manifest["partition"] == "inner_train"
            ]["sample_index"].astype(int).tolist()
        )
        valid_ids = set(
            manifest[
                manifest["partition"] == "inner_valid"
            ]["sample_index"].astype(int).tolist()
        )
        outer_ids = set(
            manifest[
                manifest["partition"] == "outer_holdout"
            ]["sample_index"].astype(int).tolist()
        )
        require(
            train_ids.isdisjoint(valid_ids),
            "train/valid overlap",
        )
        require(
            (train_ids | valid_ids).isdisjoint(outer_ids),
            "outer leakage",
        )

        history = pd.read_csv(history_path)
        weights = pd.read_csv(weights_path)
        predictions = pd.read_csv(prediction_path)
        valid_predictions = pd.read_csv(valid_prediction_path)
        require(
            set(predictions["sample_index"].astype(int))
            == outer_ids,
            "outer prediction index set mismatch",
        )
        require(
            set(valid_predictions["sample_index"].astype(int))
            == valid_ids,
            "valid prediction index set mismatch",
        )
        require(
            not predictions["sample_id"].astype(str).duplicated().any(),
            "duplicate outer sample IDs",
        )
        require(
            not valid_predictions[
                "sample_id"
            ].astype(str).duplicated().any(),
            "duplicate valid sample IDs",
        )

        initial_fingerprints = set()
        local_selected_names = None
        for strategy in STRATEGIES:
            local = summary["strategy_summaries"][strategy]
            checkpoint_path = Path(local["checkpoint"])
            require(
                checkpoint_path.is_file(),
                f"missing checkpoint: {checkpoint_path}",
            )
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
            )
            require(
                checkpoint["version"] == TRAINING_VERSION,
                "checkpoint version",
            )
            require(
                checkpoint["strategy"] == strategy,
                "checkpoint strategy",
            )
            require(
                checkpoint["outer_fold"] == fold,
                "checkpoint fold",
            )
            require(
                checkpoint["source_checkpoint_sha256"]
                == current_hash,
                "checkpoint source hash mismatch",
            )
            cp_provenance = checkpoint["provenance"]
            for key in (
                "initialized_from_same_v919_cfcompat_anchor",
                "fusion_tail_only_trainable",
                "plain_sgd_without_momentum_or_weight_decay",
                "inner_train_only_for_updates",
                "inner_valid_only_for_checkpoint_selection",
                "outer_holdout_not_used_for_training_or_selection",
            ):
                require(
                    cp_provenance.get(key) is True,
                    f"checkpoint provenance {key}",
                )
            for key in (
                "router_or_fusion_head_used",
                "expert_predictions_used",
                "official_validation_or_test_used",
            ):
                require(
                    cp_provenance.get(key) is False,
                    f"forbidden checkpoint {key}",
                )

            names = tuple(checkpoint["selected_parameter_names"])
            require(names, "empty selected parameter names")
            require(
                all(
                    any(
                        f"backbone.{module}." in name
                        for module in FUSION_TAIL_MODULES
                    )
                    for name in names
                ),
                "selected parameter outside fusion tail",
            )
            if local_selected_names is None:
                local_selected_names = names
            require(
                names == local_selected_names,
                "strategy parameter scopes differ",
            )
            selected_name_sets.append(names)
            initial_fingerprints.add(
                local["initial_parameter_fingerprint"]
            )

            strategy_history = history[
                history["strategy"] == strategy
            ]
            require(
                not strategy_history.empty,
                "empty strategy history",
            )
            require(
                int(local["best_epoch"])
                in set(strategy_history["epoch"].astype(int)),
                "best epoch absent from history",
            )
            best_row = strategy_history[
                strategy_history["epoch"].astype(int)
                == int(local["best_epoch"])
            ].iloc[0]
            require(
                close(
                    best_row["validation_objective"],
                    local["best_validation_objective"],
                ),
                "best validation objective mismatch",
            )
            if (
                strategy == "normalized_mgda"
                and int(local["best_epoch"]) > 0
            ):
                mgda_weights = weights[
                    (weights["strategy"] == strategy)
                    & (
                        weights["epoch"].astype(int)
                        == int(local["best_epoch"])
                    )
                ]
                require(
                    len(mgda_weights) == 5,
                    "missing selected MGDA weights",
                )
                require(
                    close(
                        mgda_weights["weight"].sum(),
                        1.0,
                        1e-5,
                    ),
                    "MGDA weights do not sum to one",
                )
                require(
                    bool(
                        (mgda_weights["weight"] >= -1e-8).all()
                    ),
                    "negative MGDA weight",
                )

            recalculated = _metrics(predictions, strategy)
            stored = local["outer_holdout_metrics"]
            for key in (
                "mae",
                "macro_region_mae",
                "gain_vs_anchor",
                "large_harm_rate_010",
            ):
                require(
                    close(recalculated[key], stored[key]),
                    f"metric mismatch {strategy} {key}",
                )
            for region in REGION_NAMES:
                require(
                    close(
                        recalculated["region_mae"][region],
                        stored[f"region_mae_{region}"],
                    ),
                    f"region metric mismatch {strategy} {region}",
                )

        require(
            len(initial_fingerprints) == 1,
            "strategies started differently",
        )
        require(
            next(iter(initial_fingerprints))
            == summary["initial_selected_parameter_fingerprint"],
            "summary initial fingerprint mismatch",
        )
        predictions.insert(0, "outer_fold", fold)
        all_frames.append(predictions)

    require(
        len(set(selected_name_sets)) == 1,
        "parameter scope changed across folds",
    )
    samples = pd.concat(all_frames, ignore_index=True)
    require(
        not samples["sample_id"].astype(str).duplicated().any(),
        "aggregate duplicate IDs",
    )

    aggregate_summary_path = (
        experiment / "v924_aggregate_summary.json"
    )
    aggregate_metrics_path = (
        experiment / "v924_aggregate_strategy_metrics.csv"
    )
    comparisons_path = (
        experiment / "v924_fold_primary_comparisons.csv"
    )
    aggregate_predictions_path = (
        experiment / "v924_outer_predictions.csv"
    )
    report_path = experiment / "v924_report.md"
    for path in (
        aggregate_summary_path,
        aggregate_metrics_path,
        comparisons_path,
        aggregate_predictions_path,
        report_path,
    ):
        require(
            path.is_file(),
            f"missing aggregate artifact: {path}",
        )
    aggregate_summary = json.loads(
        aggregate_summary_path.read_text(encoding="utf-8")
    )
    require(
        aggregate_summary["version"] == TRAINING_VERSION,
        "aggregate version",
    )
    require(
        aggregate_summary["primary_strategy"]
        == PRIMARY_STRATEGY,
        "aggregate primary",
    )
    require(
        aggregate_summary["sample_count"] == len(samples),
        "sample count mismatch",
    )

    primary = _metrics(samples, PRIMARY_STRATEGY)
    global_control = _metrics(samples, "global_mae")
    gain = float(
        global_control["mae"] - primary["mae"]
    )
    macro_gain = float(
        global_control["macro_region_mae"]
        - primary["macro_region_mae"]
    )
    sn_gain = float(
        global_control["region_mae"]["strong_negative"]
        - primary["region_mae"]["strong_negative"]
    )
    comparisons = pd.read_csv(comparisons_path)
    positive = int(
        (comparisons["gain_vs_global_control"] > 0.0).sum()
    )
    worst = float(
        max(
            0.0,
            -comparisons["gain_vs_global_control"].min(),
        )
    )
    gate = SuccessGateConfigV924()
    expected_pass = bool(
        gain >= gate.required_gain_vs_global
        and positive >= gate.required_positive_outer_folds
        and worst <= gate.maximum_worst_fold_degradation
        and (
            not gate.require_macro_region_improvement
            or macro_gain > 0.0
        )
        and (
            not gate.require_strong_negative_nondegradation
            or sn_gain >= 0.0
        )
    )
    stored_gate = aggregate_summary["success_gate"]
    require(
        close(
            stored_gate["aggregate_gain_vs_global_control"],
            gain,
        ),
        "aggregate gain",
    )
    require(
        stored_gate["positive_outer_folds"] == positive,
        "positive fold count",
    )
    require(
        close(stored_gate["worst_fold_degradation"], worst),
        "worst degradation",
    )
    require(
        close(
            stored_gate["macro_region_gain_vs_global_control"],
            macro_gain,
        ),
        "macro gain",
    )
    require(
        close(
            stored_gate[
                "strong_negative_gain_vs_global_control"
            ],
            sn_gain,
        ),
        "SN gain",
    )
    require(
        bool(stored_gate["passed"]) == expected_pass,
        "success gate mismatch",
    )

    print(
        "V9.24 REGION-GRADIENT CONSOLIDATION "
        "ENGINEERING AUDIT PASSED"
    )
    print("models trained: True")
    print("single model per strategy: True")
    print("router/fusion head/expert distillation: False")
    print("trainable scope: fusion_tail")
    print("primary strategy:", PRIMARY_STRATEGY)
    print("gain vs global control:", f"{gain:+.6f}")
    print(
        "positive folds:",
        f"{positive}/{int(cli.outer_folds)}",
    )
    print("success gate passed:", expected_pass)


if __name__ == "__main__":
    main()
