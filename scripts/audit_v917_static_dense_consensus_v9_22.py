"""Independent engineering audit for V9.22 V9.17 static consensus outputs."""

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

from trains.singleTask.v917_static_dense_consensus_v922 import (
    AUDIT_VERSION,
    PRIMARY_STRATEGY,
    CompatibilityConfigV922,
    fit_full_validation,
    normalize_v917_pool,
    strategy_predictions,
    validation_group_crossfit,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v917-root",
        default="result/fixed_expert_region_audit_v917/mosi/seed_1111",
    )
    parser.add_argument("--output-dir", default="")
    return parser.parse_args()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def main():
    cli = parse_args()
    root = Path(cli.v917_root)
    output = (
        Path(cli.output_dir)
        if cli.output_dir
        else root / "v922_static_dense_consensus_compatibility"
    )
    summary_path = output / "v922_static_consensus_summary.json"
    checkpoint_path = output / "v922_v917_static_consensus.pth"
    test_predictions_path = output / "v922_test_predictions.csv"
    valid_predictions_path = output / "v922_validation_predictions.csv"
    metrics_path = output / "v922_test_strategy_metrics.csv"
    for path in (
        summary_path,
        checkpoint_path,
        test_predictions_path,
        valid_predictions_path,
        metrics_path,
    ):
        require(path.is_file(), f"missing V9.22 artifact: {path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    require(summary["version"] == AUDIT_VERSION, "summary version mismatch")
    require(checkpoint["version"] == AUDIT_VERSION, "checkpoint version mismatch")
    provenance = summary["provenance"]
    require(provenance["new_anchor_or_expert_models_trained"] is False, "models retrained")
    require(provenance["weights_fit_on_v917_validation_only"] is True, "weight source mismatch")
    require(provenance["test_used_for_weight_fitting_or_strategy_selection"] is False, "Test selection detected")
    require(provenance["sample_dependent_weights"] is False, "dynamic weights detected")
    require(provenance["router_region_model_or_gate_used"] is False, "router detected")
    require(provenance["result_is_not_an_independent_confirmatory_test"] is True, "exploratory status missing")

    config = CompatibilityConfigV922(**summary["config"])
    validation = normalize_v917_pool(
        torch.load(root / "valid_fixed_expert_pool_v917.pth", map_location="cpu"),
        "v917_frozen_validation",
    )
    test = normalize_v917_pool(
        torch.load(root / "test_fixed_expert_pool_v917.pth", map_location="cpu"),
        "v917_frozen_test",
    )
    require(set(validation.sample_ids).isdisjoint(set(test.sample_ids)), "Validation/Test overlap")

    refit = fit_full_validation(validation, config)
    require(
        int(refit["selected_single_index"])
        == int(checkpoint["validation_selected_single_index"]),
        "selected single was not reproduced from Validation",
    )
    for strategy in ("convex_mae", "convex_shrinkage"):
        expected = np.asarray(refit["weights"][strategy], dtype=np.float64)
        saved = checkpoint["weights"][strategy].double().numpy()
        require(np.allclose(expected, saved, atol=2e-6), f"{strategy} weights mismatch")
        require(np.all(saved >= -1e-8), f"{strategy} negative weight")
        require(abs(float(saved.sum()) - 1.0) < 2e-6, f"{strategy} weights do not sum to one")

    expected_test = strategy_predictions(refit, test)
    test_frame = pd.read_csv(test_predictions_path)
    require(test_frame["sample_id"].astype(str).tolist() == test.sample_ids, "Test order mismatch")
    require(not test_frame["sample_id"].astype(str).duplicated().any(), "duplicate Test IDs")
    for strategy, prediction in expected_test.items():
        observed = test_frame[f"prediction_{strategy}"].to_numpy(dtype=np.float64)
        require(
            np.allclose(observed, prediction.double().numpy(), atol=2e-6),
            f"Test prediction mismatch for {strategy}",
        )

    expected_valid = strategy_predictions(refit, validation)
    valid_frame = pd.read_csv(valid_predictions_path)
    require(valid_frame["sample_id"].astype(str).tolist() == validation.sample_ids, "Validation order mismatch")
    for strategy, prediction in expected_valid.items():
        observed = valid_frame[f"prediction_{strategy}"].to_numpy(dtype=np.float64)
        require(
            np.allclose(observed, prediction.double().numpy(), atol=2e-6),
            f"Validation prediction mismatch for {strategy}",
        )

    crossfit = validation_group_crossfit(validation, config)
    for strategy, prediction in crossfit["predictions"].items():
        observed = valid_frame[f"crossfit_prediction_{strategy}"].to_numpy(dtype=np.float64)
        require(
            np.allclose(observed, prediction.double().numpy(), atol=2e-6),
            f"Validation crossfit mismatch for {strategy}",
        )

    metric_frame = pd.read_csv(metrics_path).set_index("strategy")
    labels = test.labels.numpy()
    anchor = test.actions[:, 0].numpy()
    anchor_mae = float(np.abs(anchor - labels).mean())
    for strategy, prediction in expected_test.items():
        mae = float(np.abs(prediction.numpy() - labels).mean())
        gain = anchor_mae - mae
        require(abs(mae - float(metric_frame.loc[strategy, "mae"])) < 2e-6, f"MAE mismatch: {strategy}")
        require(abs(gain - float(metric_frame.loc[strategy, "gain_vs_anchor"])) < 2e-6, f"gain mismatch: {strategy}")

    primary = summary["test_exploratory_metrics"][PRIMARY_STRATEGY]
    require(
        abs(float(primary["mae"]) - float(metric_frame.loc[PRIMARY_STRATEGY, "mae"])) < 2e-6,
        "primary summary mismatch",
    )
    print("V9.22 V9.17 STATIC CONSENSUS ENGINEERING AUDIT PASSED")
    print("models trained: False")
    print("primary strategy:", PRIMARY_STRATEGY)
    print("validation selected single:", summary["validation_selected_single_action"])
    print(
        "test anchor/primary/gain:",
        f"{summary['test_exploratory_metrics']['anchor']['mae']:.6f}",
        f"{primary['mae']:.6f}",
        f"{primary['gain_vs_anchor']:+.6f}",
    )
    print("status: exploratory; MOSI Test previously observed")


if __name__ == "__main__":
    main()
