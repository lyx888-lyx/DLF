"""Engineering and protocol audit for V9.11 predicted-region routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/predicted_region_router_v911/mosi/seed_1111",
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    root = Path(cli.root)
    summary_path = root / "predicted_region_router_v911_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    router = summary["region_router"]
    provenance = router.get("provenance") or {}

    assert summary["method"] == "predicted_region_router_v1"
    assert int(router["checkpoint_count"]) >= 2
    checkpoints = [Path(path) for path in router["ensemble_checkpoints"]]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
    assert provenance.get("is_fully_nested_teacher_stack") is True
    assert provenance.get("holdout_label_isolation") is True
    assert provenance.get("historical_full_train_teacher_reuse") is False

    required_csv = (
        "v911_region_oof_predictions.csv",
        "v911_region_oof_confusion.csv",
        "v911_region_action_maps.csv",
        "v911_region_ensemble.csv",
        "v911_oof_policy_profiles.csv",
        "v911_validation_profile_selection.csv",
        "v911_test_summary.csv",
        "v911_test_region_confusion.csv",
        "predicted_region_router_v911_predictions.csv",
    )
    for name in required_csv:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)

    oof = pd.read_csv(root / "v911_region_oof_predictions.csv")
    test = pd.read_csv(root / "predicted_region_router_v911_predictions.csv")
    profiles = pd.read_csv(root / "v911_oof_policy_profiles.csv")
    assert len(oof) == 1284
    assert len(test) == 686
    assert oof["sample_id"].nunique() == len(oof)
    assert oof["coach_fold"].nunique() == int(router["checkpoint_count"])
    probability_columns = [f"p_{name}" for name in summary["region_schema"]]
    assert not oof[probability_columns].isna().any().any()
    assert not test[probability_columns].isna().any().any()
    assert ((oof[probability_columns].sum(axis=1) - 1.0).abs() < 1e-4).all()
    assert ((test[probability_columns].sum(axis=1) - 1.0).abs() < 1e-4).all()
    assert set(profiles["profile"]) == {"conservative", "balanced", "broad"}

    metrics = router["region_metrics"]
    results = summary["test_results"]
    print("ENGINEERING AUDIT PASSED")
    print(f"predicted-region OOF samples: {len(oof)}")
    print(f"teacher stack fully nested: {provenance.get('is_fully_nested_teacher_stack')}")
    print(f"historical full-Train teacher reuse: {provenance.get('historical_full_train_teacher_reuse')}")
    print(f"holdout-label isolation: {provenance.get('holdout_label_isolation')}")
    print(f"ensemble checkpoints: {len(checkpoints)}")
    print(f"crossfit best epochs: {router.get('crossfit_best_epochs')}")
    print(f"OOF region accuracy: {metrics.get('accuracy'):.6f}")
    print(f"OOF region macro-F1: {metrics.get('macro_f1'):.6f}")
    print(f"OOF ordinal index MAE: {metrics.get('ordinal_index_mae'):.6f}")
    print(f"OOF anchor MAE: {router.get('oof_anchor_mae'):.6f}")
    print(
        "OOF predicted semantic route MAE: "
        f"{router.get('oof_predicted_semantic_route_mae'):.6f}"
    )
    print(
        "OOF predicted empirical route MAE: "
        f"{router.get('oof_predicted_empirical_route_mae'):.6f}"
    )
    print(
        "OOF true-region semantic upper bound: "
        f"{router.get('oof_true_region_semantic_route_mae'):.6f}"
    )
    print(f"selected mapping: {summary.get('selected_mapping')}")
    for row in summary["oof_policy_profiles"]:
        print(
            "  profile=%s gain=%+.6f lower=%+.6f harm=%.4f "
            "activation=%.4f eligible=%s"
            % (
                row["profile"],
                row["oof_gain"],
                row["bootstrap_gain_lower"],
                row["harm_over_010_rate"],
                row["activation_rate"],
                row["oof_eligible"],
            )
        )
    print(f"validation selected profile: {summary.get('selected_profile')}")
    print(f"test activation rate: {summary.get('test_activation_rate')}")
    print(f"test anchor MAE: {results['anchor']['MAE']:.6f}")
    print(
        "test predicted-region route MAE: "
        f"{results['predicted_region_chosen_mapping_all']['MAE']:.6f}"
    )
    print(
        "test probability-mixture MAE: "
        f"{results['region_probability_mixture_all']['MAE']:.6f}"
    )
    print(
        "test deployable region route MAE: "
        f"{results['predicted_region_valid_selected']['MAE']:.6f}"
    )
    print(
        "test true-region upper bound MAE: "
        f"{results['true_region_semantic_upper_bound']['MAE']:.6f}"
    )
    if (
        results["predicted_region_valid_selected"]["MAE"]
        >= results["anchor"]["MAE"]
    ):
        print("WARNING: V9.11 deployable router did not beat Anchor on Test")


if __name__ == "__main__":
    main()
