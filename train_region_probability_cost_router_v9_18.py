"""Train and evaluate V9.18 ordinal region-probability × action-cost routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch

from trains.singleTask.model.OrdinalRegionCostRouterV918 import (
    region_probabilities_from_logits,
)
from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES
from trains.singleTask.region_cost_reporting_v918 import (
    anchor_fallback_result,
    confusion_rows,
    jsonable,
    sample_rows,
    sha256,
    subset_pool,
)
from trains.singleTask.region_probability_cost_router_v918 import (
    ROUTER_VERSION,
    PolicyGridV918,
    RegionModelConfigV918,
    SafetyConfigV918,
    action_cost_matrix,
    apply_policy,
    group_bucket,
    nested_oof_region_probabilities,
    normalize_router_pool,
    policy_result_for_pool,
    region_metrics,
    save_model_checkpoint,
    select_policy,
    shrink_cost_matrix,
    train_fixed_epochs,
)
from utils import assign_gpu, setup_seed

REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)


def parse_float_tuple(value: str):
    return tuple(float(token.strip()) for token in value.split(",") if token.strip())


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strict OOF ordinal region probabilities times action-cost matrix routing."
    )
    parser.add_argument("--dataset", choices=["mosi", "mosei"], default="mosi")
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument(
        "--v917-root",
        default="result/fixed_expert_region_audit_v917/mosi/seed_1111",
    )
    parser.add_argument(
        "--strict-pool",
        default=(
            "result/strict_nested_frontier_coach_v98/mosi/seed_1111/"
            "strict_nested_frontier_pool/strict_nested_v93_frontier_pool_v98.pth"
        ),
    )
    parser.add_argument(
        "--save-root", default="result/region_probability_cost_router_v918"
    )
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-epochs", type=int, default=80)
    parser.add_argument("--early-stop", type=int, default=10)
    parser.add_argument("--gradient-clip", type=float, default=2.0)
    parser.add_argument("--score-penalty", type=float, default=1e-4)
    parser.add_argument(
        "--gain-margins", default="0,0.01,0.02,0.03,0.05,0.075,0.10"
    )
    parser.add_argument(
        "--min-region-confidences", default="0,0.35,0.45,0.55,0.65,0.75"
    )
    parser.add_argument("--temperatures", default="0.75,1,1.25,1.5,2")
    parser.add_argument("--min-oof-gain", type=float, default=0.001)
    parser.add_argument("--max-oof-harm", type=float, default=0.10)
    parser.add_argument("--min-validation-gain", type=float, default=0.0)
    parser.add_argument("--max-validation-harm", type=float, default=0.10)
    parser.add_argument("--cost-prior-strength", type=float, default=30.0)
    return parser.parse_args()


def metric_subset(result):
    keys = (
        "anchor_mae",
        "mae",
        "gain_vs_anchor",
        "harm_over_010_rate",
        "large_gain_rate_010",
        "coverage",
        "trigger_precision",
        "mean_trigger_gain",
        "action_counts",
    )
    return {key: result[key] for key in keys}


def write_cost_matrix(path: Path, matrix: torch.Tensor):
    pd.DataFrame(
        matrix.numpy(), index=REGION_NAMES, columns=ACTION_NAMES
    ).to_csv(path)


def main():
    cli = parse_args()
    setup_seed(cli.seed)
    device = assign_gpu([cli.gpu])

    model_config = RegionModelConfigV918(
        hidden_dim=cli.hidden_dim,
        dropout=cli.dropout,
        learning_rate=cli.learning_rate,
        weight_decay=cli.weight_decay,
        batch_size=cli.batch_size,
        max_epochs=cli.max_epochs,
        early_stop=cli.early_stop,
        gradient_clip=cli.gradient_clip,
        score_penalty=cli.score_penalty,
    )
    policy_grid = PolicyGridV918(
        gain_margins=parse_float_tuple(cli.gain_margins),
        min_region_confidences=parse_float_tuple(cli.min_region_confidences),
        temperatures=parse_float_tuple(cli.temperatures),
    )
    safety = SafetyConfigV918(
        min_oof_gain=cli.min_oof_gain,
        max_oof_harm_over_010_rate=cli.max_oof_harm,
        min_validation_gain=cli.min_validation_gain,
        max_validation_harm_over_010_rate=cli.max_validation_harm,
        cost_prior_strength=cli.cost_prior_strength,
    )

    strict_pool_path = Path(cli.strict_pool)
    v917_root = Path(cli.v917_root)
    valid_pool_path = v917_root / "valid_fixed_expert_pool_v917.pth"
    test_pool_path = v917_root / "test_fixed_expert_pool_v917.pth"
    for path in (strict_pool_path, valid_pool_path, test_pool_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    save_dir = Path(cli.save_root) / cli.dataset / f"seed_{cli.seed}"
    save_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "oof_candidates": save_dir / "v918_oof_policy_candidates.csv",
        "oof_folds": save_dir / "v918_oof_fold_metrics.csv",
        "training_history": save_dir / "v918_region_training_history.csv",
        "oof_predictions": save_dir / "v918_oof_router_predictions.csv",
        "router_checkpoint": save_dir / "ordinal_region_router_v918.pth",
        "train_cost_matrix": save_dir / "v918_train_oof_cost_matrix.csv",
        "validation_calibrated_cost_matrix": save_dir / "v918_validation_calibrated_cost_matrix.csv",
        "final_cost_matrix": save_dir / "v918_final_deployment_cost_matrix.csv",
        "validation_predictions": save_dir / "v918_validation_router_predictions.csv",
        "test_predictions": save_dir / "v918_test_router_predictions.csv",
        "confusion": save_dir / "v918_region_confusion.csv",
        "selection": save_dir / "v918_validation_selection.json",
        "test_summary": save_dir / "v918_test_summary.csv",
        "summary": save_dir / "region_probability_cost_router_v918_summary.json",
    }

    strict_raw = torch.load(strict_pool_path, map_location="cpu")
    strict_provenance = strict_raw.get("provenance", {})
    if strict_provenance.get("is_fully_nested_teacher_stack") is not True:
        raise ValueError("V9.18 requires the strict nested V9.8 pool")
    if strict_provenance.get("holdout_label_isolation") is not True:
        raise ValueError("strict OOF pool does not guarantee holdout isolation")
    train_pool = normalize_router_pool(
        strict_raw, "strict_nested_outer_fold_oof_shadow_experts"
    )

    nested = nested_oof_region_probabilities(
        train_pool, device, model_config, policy_grid, cli.seed
    )
    pd.DataFrame(nested["candidate_rows"]).to_csv(
        paths["oof_candidates"], index=False
    )
    pd.DataFrame(nested["fold_rows"]).to_csv(paths["oof_folds"], index=False)
    pd.DataFrame(nested["history_rows"]).to_csv(
        paths["training_history"], index=False
    )
    selected_policy = select_policy(nested["candidate_rows"], safety)

    selected_temperature = float(selected_policy["temperature"])
    oof_probabilities = region_probabilities_from_logits(
        nested["logits"], selected_temperature
    )
    oof_expected_costs = torch.einsum(
        "nr,nra->na", oof_probabilities, nested["cost_matrices"]
    )
    oof_result = apply_policy(
        oof_probabilities,
        oof_expected_costs,
        train_pool["actions_2d"],
        train_pool["labels"],
        float(selected_policy["gain_margin"]),
        float(selected_policy["min_region_confidence"]),
    )
    pd.DataFrame(
        sample_rows(
            "train_oof",
            train_pool,
            {
                "region_probabilities": oof_probabilities,
                "expected_costs": oof_expected_costs,
                **oof_result,
            },
        )
    ).to_csv(paths["oof_predictions"], index=False)

    all_train_indices = torch.arange(len(train_pool["labels"]), dtype=torch.long)
    final_epochs = int(nested["median_best_epoch"])
    final_model = train_fixed_epochs(
        train_pool["router_features"],
        train_pool["labels"],
        all_train_indices,
        device,
        model_config,
        cli.seed + 918000,
        final_epochs,
    )
    save_model_checkpoint(
        paths["router_checkpoint"],
        final_model,
        model_config,
        final_epochs,
        selected_temperature,
        train_pool["router_features"].shape[1],
    )

    train_cost, train_counts = action_cost_matrix(
        train_pool["actions_2d"],
        train_pool["labels"],
        train_pool["region_index"],
        all_train_indices,
    )
    write_cost_matrix(paths["train_cost_matrix"], train_cost)

    valid_raw = torch.load(valid_pool_path, map_location="cpu")
    valid_pool = normalize_router_pool(
        valid_raw, "frozen_full_train_deployment_experts"
    )
    if valid_pool["router_features"].shape[1] != train_pool["router_features"].shape[1]:
        raise RuntimeError("Train/Validation router feature dimension mismatch")

    calibration_indices = torch.tensor(
        [
            index
            for index, group in enumerate(valid_pool["group_ids"])
            if group_bucket(group, cli.seed + 9181, 2) == 0
        ],
        dtype=torch.long,
    )
    selection_indices = torch.tensor(
        [
            index
            for index, group in enumerate(valid_pool["group_ids"])
            if group_bucket(group, cli.seed + 9181, 2) != 0
        ],
        dtype=torch.long,
    )
    if len(calibration_indices) < 40 or len(selection_indices) < 40:
        raise RuntimeError("Validation calibration/selection group split is too small")

    calibrated_cost, calibration_counts = shrink_cost_matrix(
        train_cost,
        valid_pool["actions_2d"],
        valid_pool["labels"],
        valid_pool["region_index"],
        calibration_indices,
        safety.cost_prior_strength,
    )
    write_cost_matrix(paths["validation_calibrated_cost_matrix"], calibrated_cost)

    policy_fallback = bool(selected_policy.get("fallback", False))
    if policy_fallback:
        validation_full = anchor_fallback_result(
            final_model,
            valid_pool,
            device,
            model_config.batch_size,
            selected_temperature,
            calibrated_cost,
        )
        validation_selection = anchor_fallback_result(
            final_model,
            subset_pool(valid_pool, selection_indices),
            device,
            model_config.batch_size,
            selected_temperature,
            calibrated_cost,
        )
    else:
        validation_full = policy_result_for_pool(
            final_model,
            valid_pool,
            device,
            model_config.batch_size,
            selected_temperature,
            calibrated_cost,
            selected_policy["gain_margin"],
            selected_policy["min_region_confidence"],
        )
        validation_selection = policy_result_for_pool(
            final_model,
            subset_pool(valid_pool, selection_indices),
            device,
            model_config.batch_size,
            selected_temperature,
            calibrated_cost,
            selected_policy["gain_margin"],
            selected_policy["min_region_confidence"],
        )

    accepted = (
        not policy_fallback
        and validation_selection["gain_vs_anchor"] >= safety.min_validation_gain
        and validation_selection["harm_over_010_rate"]
        <= safety.max_validation_harm_over_010_rate
        and validation_selection["coverage"] > 0.0
    )
    rejection_reason = None if accepted else (
        selected_policy.get("reason")
        if policy_fallback
        else "validation_safety_gate_failed"
    )

    if accepted:
        final_cost, final_valid_counts = shrink_cost_matrix(
            train_cost,
            valid_pool["actions_2d"],
            valid_pool["labels"],
            valid_pool["region_index"],
            torch.arange(len(valid_pool["labels"])),
            safety.cost_prior_strength,
        )
    else:
        final_cost = calibrated_cost
        final_valid_counts = calibration_counts
    write_cost_matrix(paths["final_cost_matrix"], final_cost)

    selection_artifact = {
        "version": ROUTER_VERSION,
        "selected_policy": selected_policy,
        "accepted_for_test": accepted,
        "rejection_reason": rejection_reason,
        "validation_group_split": {
            "calibration_count": len(calibration_indices),
            "selection_count": len(selection_indices),
            "calibration_group_count": len(
                set(valid_pool["group_ids"][idx] for idx in calibration_indices.tolist())
            ),
            "selection_group_count": len(
                set(valid_pool["group_ids"][idx] for idx in selection_indices.tolist())
            ),
        },
        "validation_selection_metrics": metric_subset(validation_selection),
        "test_pool_loaded": False,
        "provenance": {
            "policy_hyperparameters_selected_on_strict_oof": True,
            "validation_cost_calibration_and_safety_groups_disjoint": True,
            "test_used_for_training_selection_or_calibration": False,
        },
    }
    paths["selection"].write_text(
        json.dumps(jsonable(selection_artifact), indent=2), encoding="utf-8"
    )
    pd.DataFrame(sample_rows("valid", valid_pool, validation_full)).to_csv(
        paths["validation_predictions"], index=False
    )

    # Deliberately load Test only after the immutable selection artifact exists.
    test_raw = torch.load(test_pool_path, map_location="cpu")
    test_pool = normalize_router_pool(
        test_raw, "frozen_full_train_deployment_experts"
    )
    if test_pool["router_features"].shape[1] != train_pool["router_features"].shape[1]:
        raise RuntimeError("Train/Test router feature dimension mismatch")
    if accepted:
        test_result = policy_result_for_pool(
            final_model,
            test_pool,
            device,
            model_config.batch_size,
            selected_temperature,
            final_cost,
            selected_policy["gain_margin"],
            selected_policy["min_region_confidence"],
        )
    else:
        test_result = anchor_fallback_result(
            final_model,
            test_pool,
            device,
            model_config.batch_size,
            selected_temperature,
            final_cost,
        )
    pd.DataFrame(sample_rows("test", test_pool, test_result)).to_csv(
        paths["test_predictions"], index=False
    )

    confusion = []
    confusion.extend(
        confusion_rows("train_oof", oof_probabilities, train_pool["region_index"])
    )
    confusion.extend(
        confusion_rows(
            "valid", validation_full["region_probabilities"], valid_pool["region_index"]
        )
    )
    confusion.extend(
        confusion_rows(
            "test", test_result["region_probabilities"], test_pool["region_index"]
        )
    )
    pd.DataFrame(confusion).to_csv(paths["confusion"], index=False)

    test_summary_rows = [
        {
            "model": "anchor",
            "mae": test_result["anchor_mae"],
            "gain_vs_anchor": 0.0,
            "coverage": 0.0,
        },
        {
            "model": (
                "validation_selected_region_cost_router"
                if accepted
                else "anchor_fallback"
            ),
            **metric_subset(test_result),
            **{
                f"count_{name}": test_result["action_counts"][name]
                for name in ACTION_NAMES
            },
        },
    ]
    pd.DataFrame(test_summary_rows).to_csv(paths["test_summary"], index=False)

    summary = {
        "version": ROUTER_VERSION,
        "method": "ordinal_region_probability_times_action_cost_v9_18",
        "dataset": cli.dataset,
        "seed": cli.seed,
        "model_config": model_config.__dict__,
        "policy_grid": policy_grid.__dict__,
        "safety_config": safety.__dict__,
        "selected_policy": selected_policy,
        "accepted_for_test": accepted,
        "rejection_reason": rejection_reason,
        "final_training_epochs": final_epochs,
        "feature_dim": int(train_pool["router_features"].shape[1]),
        "oof_selected_metrics": metric_subset(oof_result),
        "oof_region_metrics": region_metrics(
            oof_probabilities, train_pool["region_index"]
        ),
        "validation_full_metrics": metric_subset(validation_full),
        "validation_selection_metrics": metric_subset(validation_selection),
        "validation_region_metrics": region_metrics(
            validation_full["region_probabilities"], valid_pool["region_index"]
        ),
        "test_metrics": metric_subset(test_result),
        "test_region_metrics": region_metrics(
            test_result["region_probabilities"], test_pool["region_index"]
        ),
        "cost_region_counts": {
            "train_oof": train_counts,
            "validation_calibration": calibration_counts,
            "validation_final": final_valid_counts,
        },
        "checkpoint_paths": {
            "strict_oof_pool": str(strict_pool_path),
            "valid_frozen_pool": str(valid_pool_path),
            "test_frozen_pool": str(test_pool_path),
            "router": str(paths["router_checkpoint"]),
        },
        "checkpoint_sha256": {
            "strict_oof_pool": sha256(strict_pool_path),
            "valid_frozen_pool": sha256(valid_pool_path),
            "test_frozen_pool": sha256(test_pool_path),
            "router": sha256(paths["router_checkpoint"]),
        },
        "outputs": {key: str(value) for key, value in paths.items()},
        "provenance": {
            "strict_oof_experts_and_features_used_for_router_training": True,
            "each_train_expert_prediction_excludes_its_sample": True,
            "ordinal_region_targets_use_train_labels_only": True,
            "policy_grid_selected_on_nested_oof_only": True,
            "official_validation_used_for_cost_calibration": True,
            "validation_calibration_and_safety_groups_disjoint": True,
            "official_test_used_for_training_selection_or_calibration": False,
            "test_pool_loaded_after_selection_artifact_written": True,
            "action_costs_equal_region_probability_times_cost_matrix": True,
            "anchor_fallback_enabled": True,
            "neural_action_router_trained": False,
            "test_labels_used_for_reporting_only": True,
        },
    }
    paths["summary"].write_text(
        json.dumps(jsonable(summary), indent=2, sort_keys=True), encoding="utf-8"
    )

    print("V9.18 REGION-PROBABILITY ACTION-COST ROUTER COMPLETE")
    print("OOF selected policy:", selected_policy)
    print("Validation accepted for Test:", accepted, rejection_reason)
    print(
        "OOF anchor/router/gain:",
        f"{oof_result['anchor_mae']:.6f}",
        f"{oof_result['mae']:.6f}",
        f"{oof_result['gain_vs_anchor']:+.6f}",
    )
    print(
        "Validation selection anchor/router/gain:",
        f"{validation_selection['anchor_mae']:.6f}",
        f"{validation_selection['mae']:.6f}",
        f"{validation_selection['gain_vs_anchor']:+.6f}",
    )
    print(
        "Test anchor/router/gain:",
        f"{test_result['anchor_mae']:.6f}",
        f"{test_result['mae']:.6f}",
        f"{test_result['gain_vs_anchor']:+.6f}",
    )
    print("Test action counts:", test_result["action_counts"])


if __name__ == "__main__":
    main()
