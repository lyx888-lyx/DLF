"""Engineering and leakage audit for V9.13 Validation-only fusion."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES  # noqa: E402
from trains.singleTask.position_aware_residual_fusion_v913 import (  # noqa: E402
    FUSION_VERSION,
    apply_config,
    evaluate_prediction,
    pool_tensors,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/position_aware_residual_fusion_v913/mosi/seed_1111",
    )
    return parser.parse_args()


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, atol: float = 1e-6):
    return abs(float(left) - float(right)) <= atol


def main():
    cli = parse_args()
    root = Path(cli.root)
    paths = {
        "valid_pool": root / "valid_expert_pool_v913.pth",
        "test_pool": root / "test_expert_pool_v913.pth",
        "candidates": root / "v913_validation_candidates.csv",
        "selection": root / "v913_validation_selection.json",
        "test_summary": root / "v913_test_summary.csv",
        "predictions": root / "v913_test_predictions.csv",
        "summary": root / "position_aware_residual_fusion_v913_summary.json",
    }
    for path in paths.values():
        require(path.is_file(), f"missing V9.13 output: {path}")

    valid_pool = torch.load(paths["valid_pool"], map_location="cpu")
    test_pool = torch.load(paths["test_pool"], map_location="cpu")
    selection = json.loads(paths["selection"].read_text(encoding="utf-8"))
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    candidates = pd.read_csv(paths["candidates"])
    test_summary = pd.read_csv(paths["test_summary"])
    predictions = pd.read_csv(paths["predictions"])

    require(summary["version"] == FUSION_VERSION, "unexpected V9.13 version")
    require(selection["version"] == FUSION_VERSION, "selection version mismatch")
    provenance = summary["provenance"]
    require(provenance["experts_frozen"] is True, "experts were not frozen")
    require(provenance["neural_router_trained"] is False, "neural router was trained")
    require(
        provenance["official_validation_used_for_selection"] is True,
        "Validation was not the selection split",
    )
    require(
        provenance["official_validation_used_for_expert_training"] is False,
        "Validation entered expert training",
    )
    require(
        provenance["test_labels_used_for_training_or_selection"] is False,
        "Test labels entered training or selection",
    )
    require(
        provenance["test_pool_collected_after_selection_artifact_written"] is True,
        "Test was not isolated behind the completed selection artifact",
    )
    require(selection["selected_by_validation_only"] is True, "selection leakage flag")
    require(selection["test_pool_collected"] is False, "selection artifact saw Test")

    valid_values = pool_tensors(valid_pool)
    test_values = pool_tensors(test_pool)
    require(
        not (set(valid_pool["sample_ids"]) & set(test_pool["sample_ids"])),
        "Validation/Test ID overlap",
    )
    require(len(predictions) == len(test_pool["sample_ids"]), "prediction row count")
    require(
        predictions["sample_id"].astype(str).tolist()
        == [str(value) for value in test_pool["sample_ids"]],
        "prediction sample order mismatch",
    )
    for index, name in enumerate(ACTION_NAMES):
        require(name in predictions.columns, f"missing prediction column: {name}")
        require(
            torch.allclose(
                torch.tensor(predictions[name].to_numpy()).float(),
                test_values["actions"][:, index],
                atol=1e-6,
            ),
            f"stored action mismatch: {name}",
        )

    selected_id = summary["selected_candidate_id"]
    require(selected_id == selection["selected"]["candidate_id"], "selected ID mismatch")
    require(selected_id in set(candidates["candidate_id"]), "selected candidate missing")
    selected_config = summary["selected_config"]
    require(selected_config == selection["selected"]["config"], "selected config mismatch")

    valid_prediction, _ = apply_config(valid_pool, selected_config)
    valid_metrics = evaluate_prediction(
        valid_prediction, valid_values["actions"], valid_values["labels"]
    )
    require(
        close(valid_metrics["mae"], summary["validation_selected_mae"]),
        "Validation selected MAE mismatch",
    )
    require(
        close(valid_metrics["gain_vs_anchor"], summary["validation_selected_gain"]),
        "Validation selected gain mismatch",
    )

    test_prediction, extras = apply_config(test_pool, selected_config)
    test_metrics = evaluate_prediction(
        test_prediction, test_values["actions"], test_values["labels"]
    )
    require(
        torch.allclose(
            test_prediction.view(-1),
            torch.tensor(predictions["selected_prediction"].to_numpy()).float(),
            atol=1e-6,
        ),
        "stored selected predictions mismatch",
    )
    require(close(test_metrics["mae"], summary["test_selected_mae"]), "Test MAE mismatch")
    require(
        close(test_metrics["gain_vs_anchor"], summary["test_selected_gain"]),
        "Test gain mismatch",
    )
    require(
        "validation_selected_deployable" in set(test_summary["model"]),
        "deployable Test row missing",
    )

    if selected_config["family"] == "position_aware":
        weights = extras["specialist_weights"]
        rho = float(selected_config["rho"])
        require(bool((weights >= 0.0).all()), "negative specialist weight")
        require(
            bool((weights.sum(dim=1) <= rho + 1e-6).all()),
            "specialist residual budget exceeded",
        )
        for name in SPECIALIST_NAMES:
            require(f"weight_{name}" in predictions.columns, f"missing {name} weight")

    for name, value in summary["checkpoint_paths"].items():
        require(Path(value).is_file(), f"missing frozen checkpoint {name}: {value}")
    require(len(summary["checkpoint_sha256"]) == 5, "expected five checkpoint hashes")

    print("V9.13 ENGINEERING AUDIT PASSED")
    print("protocol: frozen experts; Validation-only low-dimensional selection")
    print("neural router trained: False")
    print("simplex candidates:", summary["simplex_candidate_count"])
    print("selected candidate:", selected_id)
    print(
        "Validation anchor/selected/gain:",
        f"{summary['validation_anchor_mae']:.6f}",
        f"{summary['validation_selected_mae']:.6f}",
        f"{summary['validation_selected_gain']:+.6f}",
    )
    print(
        "Test anchor/selected/gain:",
        f"{summary['test_anchor_mae']:.6f}",
        f"{summary['test_selected_mae']:.6f}",
        f"{summary['test_selected_gain']:+.6f}",
    )
    print("Test harm_over_010_rate:", f"{summary['test_selected_harm_over_010_rate']:.4f}")


if __name__ == "__main__":
    main()
