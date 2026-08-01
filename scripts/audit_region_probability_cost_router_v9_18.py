"""Engineering, alignment, and leakage audit for V9.18."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES  # noqa: E402
from trains.singleTask.region_probability_cost_router_v918 import (  # noqa: E402
    ROUTER_VERSION,
)

REGION_NAMES = (
    "strong_negative", "negative", "boundary", "positive", "strong_positive"
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/region_probability_cost_router_v918/mosi/seed_1111",
    )
    return parser.parse_args()


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, atol: float = 2e-6):
    return abs(float(left) - float(right)) <= atol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    cli = parse_args()
    root = Path(cli.root)
    files = {
        "candidates": root / "v918_oof_policy_candidates.csv",
        "folds": root / "v918_oof_fold_metrics.csv",
        "history": root / "v918_region_training_history.csv",
        "oof_predictions": root / "v918_oof_router_predictions.csv",
        "checkpoint": root / "ordinal_region_router_v918.pth",
        "train_cost": root / "v918_train_oof_cost_matrix.csv",
        "valid_cost": root / "v918_validation_calibrated_cost_matrix.csv",
        "final_cost": root / "v918_final_deployment_cost_matrix.csv",
        "valid_predictions": root / "v918_validation_router_predictions.csv",
        "test_predictions": root / "v918_test_router_predictions.csv",
        "confusion": root / "v918_region_confusion.csv",
        "selection": root / "v918_validation_selection.json",
        "test_summary": root / "v918_test_summary.csv",
        "summary": root / "region_probability_cost_router_v918_summary.json",
    }
    for path in files.values():
        require(path.is_file(), f"missing V9.18 output: {path}")

    summary = json.loads(files["summary"].read_text(encoding="utf-8"))
    selection = json.loads(files["selection"].read_text(encoding="utf-8"))
    candidates = pd.read_csv(files["candidates"])
    folds = pd.read_csv(files["folds"])
    oof = pd.read_csv(files["oof_predictions"])
    valid = pd.read_csv(files["valid_predictions"])
    test = pd.read_csv(files["test_predictions"])
    confusion = pd.read_csv(files["confusion"])
    final_cost = pd.read_csv(files["final_cost"], index_col=0)

    require(summary["version"] == ROUTER_VERSION, "summary version mismatch")
    require(selection["version"] == ROUTER_VERSION, "selection version mismatch")
    require(set(final_cost.index) == set(REGION_NAMES), "cost matrix region rows mismatch")
    require(tuple(final_cost.columns) == tuple(ACTION_NAMES), "cost matrix action columns mismatch")
    require(final_cost.shape == (5, 5), "cost matrix shape mismatch")
    require(len(candidates) == 210, "unexpected pre-registered policy candidate count")
    require(len(folds) >= 3, "too few strict OOF folds")
    require(not oof["sample_id"].astype(str).duplicated().any(), "duplicate OOF IDs")
    require(not valid["sample_id"].astype(str).duplicated().any(), "duplicate Validation IDs")
    require(not test["sample_id"].astype(str).duplicated().any(), "duplicate Test IDs")
    require(set(oof["sample_id"]).isdisjoint(set(valid["sample_id"])), "OOF/Validation overlap")
    require(set(oof["sample_id"]).isdisjoint(set(test["sample_id"])), "OOF/Test overlap")
    require(set(valid["sample_id"]).isdisjoint(set(test["sample_id"])), "Validation/Test overlap")

    provenance = summary["provenance"]
    require(provenance["strict_oof_experts_and_features_used_for_router_training"] is True, "strict OOF source missing")
    require(provenance["each_train_expert_prediction_excludes_its_sample"] is True, "sample isolation missing")
    require(provenance["policy_grid_selected_on_nested_oof_only"] is True, "policy selected outside OOF")
    require(provenance["validation_calibration_and_safety_groups_disjoint"] is True, "Validation split leakage")
    require(provenance["official_test_used_for_training_selection_or_calibration"] is False, "Test entered selection")
    require(provenance["test_pool_loaded_after_selection_artifact_written"] is True, "Test loaded before selection")
    require(provenance["action_costs_equal_region_probability_times_cost_matrix"] is True, "cost rule mismatch")
    require(provenance["anchor_fallback_enabled"] is True, "Anchor fallback missing")
    require(provenance["neural_action_router_trained"] is False, "unconstrained action router trained")
    require(selection["test_pool_loaded"] is False, "selection artifact claims Test was loaded")

    policy = summary["selected_policy"]
    if not policy["fallback"]:
        matched = candidates[
            (candidates["temperature"].astype(float) == float(policy["temperature"]))
            & (candidates["gain_margin"].astype(float) == float(policy["gain_margin"]))
            & (
                candidates["min_region_confidence"].astype(float)
                == float(policy["min_region_confidence"])
            )
        ]
        require(len(matched) == 1, "selected OOF policy absent or duplicated")

    probability_columns = [f"prob_{name}" for name in REGION_NAMES]
    cost_columns = [f"expected_cost_{name}" for name in ACTION_NAMES]
    prediction_columns = [f"prediction_{name}" for name in ACTION_NAMES]
    for frame_name, frame in (("OOF", oof), ("Validation", valid), ("Test", test)):
        require(set(probability_columns).issubset(frame.columns), f"{frame_name} probability columns missing")
        require(set(cost_columns).issubset(frame.columns), f"{frame_name} expected-cost columns missing")
        require(set(prediction_columns).issubset(frame.columns), f"{frame_name} action predictions missing")
        probabilities = torch.tensor(frame[probability_columns].to_numpy()).float()
        require(
            torch.allclose(
                probabilities.sum(dim=1), torch.ones(len(frame)), atol=2e-5
            ),
            f"{frame_name} probabilities do not sum to one",
        )
        require(bool((probabilities >= -1e-7).all()), f"{frame_name} has negative probability")

    probabilities = torch.tensor(test[probability_columns].to_numpy()).float()
    cost = torch.tensor(final_cost.to_numpy()).float()
    recomputed_expected = probabilities @ cost
    stored_expected = torch.tensor(test[cost_columns].to_numpy()).float()
    require(
        torch.allclose(recomputed_expected, stored_expected, atol=2e-5),
        "Test expected costs are not probability times cost matrix",
    )

    action_to_index = {name: index for index, name in enumerate(ACTION_NAMES)}
    selected_indices = torch.tensor(
        [action_to_index[name] for name in test["selected_action"]], dtype=torch.long
    )
    action_predictions = torch.tensor(test[prediction_columns].to_numpy()).float()
    recomputed_prediction = action_predictions.gather(
        1, selected_indices.view(-1, 1)
    ).view(-1)
    stored_prediction = torch.tensor(test["selected_prediction"].to_numpy()).float()
    require(
        torch.allclose(recomputed_prediction, stored_prediction, atol=2e-5),
        "selected action/prediction mismatch",
    )

    labels = torch.tensor(test["label"].to_numpy()).float()
    anchor = torch.tensor(test["anchor_prediction"].to_numpy()).float()
    anchor_error = torch.abs(anchor - labels)
    selected_error = torch.abs(stored_prediction - labels)
    anchor_mae = float(anchor_error.mean().item())
    selected_mae = float(selected_error.mean().item())
    gain = anchor_mae - selected_mae
    harm = float(((selected_error - anchor_error) > 0.10).float().mean().item())
    require(close(anchor_mae, summary["test_metrics"]["anchor_mae"]), "Test Anchor MAE mismatch")
    require(close(selected_mae, summary["test_metrics"]["mae"]), "Test router MAE mismatch")
    require(close(gain, summary["test_metrics"]["gain_vs_anchor"]), "Test gain mismatch")
    require(close(harm, summary["test_metrics"]["harm_over_010_rate"]), "Test harm mismatch")

    accepted = bool(summary["accepted_for_test"])
    require(accepted == bool(selection["accepted_for_test"]), "acceptance mismatch")
    if not accepted:
        require(set(test["selected_action"]) == {"anchor"}, "fallback selected an expert")
        require(close(selected_mae, anchor_mae), "fallback differs from Anchor")
    else:
        margin = float(policy["gain_margin"])
        confidence = float(policy["min_region_confidence"])
        specialist_cost = stored_expected[:, 1:].min(dim=1).values
        predicted_gain = stored_expected[:, 0] - specialist_cost
        region_confidence = probabilities.max(dim=1).values
        should_trigger = (predicted_gain >= margin) & (
            region_confidence >= confidence
        )
        stored_trigger = torch.tensor(test["triggered"].astype(bool).to_numpy())
        require(torch.equal(should_trigger, stored_trigger), "Test trigger rule mismatch")
        require(
            bool((selected_indices[~stored_trigger] == 0).all()),
            "non-trigger row did not fall back to Anchor",
        )
        proposed = stored_expected[:, 1:].argmin(dim=1) + 1
        require(
            torch.equal(selected_indices[stored_trigger], proposed[stored_trigger]),
            "trigger row did not select minimum expected-cost specialist",
        )

    for name, path_text in summary["checkpoint_paths"].items():
        path = Path(path_text)
        require(path.is_file(), f"missing checkpoint source: {name}")
        require(
            sha256(path) == summary["checkpoint_sha256"][name],
            f"checkpoint hash mismatch: {name}",
        )

    require(set(confusion["split"]) == {"train_oof", "valid", "test"}, "confusion split set mismatch")
    require(len(confusion) == 75, "confusion matrix row count mismatch")

    print("V9.18 ENGINEERING AUDIT PASSED")
    print("router: constrained ordinal region probabilities × action cost matrix")
    print("policy selected on strict OOF; Validation calibration/safety groups disjoint")
    print("accepted for Test:", accepted)
    print("selected policy:", policy)
    print(
        "OOF anchor/router/gain:",
        f"{summary['oof_selected_metrics']['anchor_mae']:.6f}",
        f"{summary['oof_selected_metrics']['mae']:.6f}",
        f"{summary['oof_selected_metrics']['gain_vs_anchor']:+.6f}",
    )
    print(
        "Validation safety gain/coverage:",
        f"{summary['validation_selection_metrics']['gain_vs_anchor']:+.6f}",
        f"{summary['validation_selection_metrics']['coverage']:.4f}",
    )
    print(
        "Test anchor/router/gain:",
        f"{anchor_mae:.6f}",
        f"{selected_mae:.6f}",
        f"{gain:+.6f}",
    )
    print("Test action counts:", summary["test_metrics"]["action_counts"])


if __name__ == "__main__":
    main()
