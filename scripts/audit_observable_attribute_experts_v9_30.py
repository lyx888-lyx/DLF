"""Independent leakage and output audit for V9.30."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trains.singleTask.observable_attribute_experts_v930 import (
    ACTION_NAMES_V930,
    AUDIT_VERSION,
    EXPERT_NAMES,
    MODE_NAMES,
    raw_attribute_scores,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--result-dir",
        default=(
            "result/full_same_stack_nested_crossfit_v919/mosi/seed_1111/"
            "v930_observable_attribute_experts"
        ),
    )
    parser.add_argument("--tolerance", type=float, default=3e-6)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_close(left: float, right: float, tolerance: float, message: str):
    if not np.isfinite(left) or not np.isfinite(right):
        raise AssertionError(f"{message}: non-finite {left}/{right}")
    if abs(float(left) - float(right)) > float(tolerance):
        raise AssertionError(f"{message}: {left} != {right}")


def main():
    cli = parse_args()
    result_dir = Path(cli.result_dir)
    required = {
        "summary": result_dir / "v930_summary.json",
        "report": result_dir / "v930_report.md",
        "fold": result_dir / "v930_metrics_by_fold.csv",
        "aggregate": result_dir / "v930_aggregate_metrics.csv",
        "weights": result_dir / "v930_weight_inventory.csv",
        "predictions": result_dir / "v930_outer_predictions.csv",
        "sources": result_dir / "v930_source_manifest.csv",
        "correlations": result_dir / "v930_residual_correlations.csv",
        "applicability": result_dir / "v930_applicability_summary.csv",
        "bootstrap": result_dir / "v930_group_bootstrap_gain_ci.csv",
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing V9.30 outputs: {missing}")

    summary = json.loads(required["summary"].read_text(encoding="utf-8"))
    if summary.get("version") != AUDIT_VERSION:
        raise AssertionError("unexpected V9.30 version")
    if tuple(summary.get("action_names", ())) != ACTION_NAMES_V930:
        raise AssertionError("V9.30 action order changed")
    provenance = summary.get("provenance", {})
    expected = {
        "attributes_computed_from_mode_predictions_only": True,
        "attributes_use_true_labels": False,
        "expert_training_uses_supervised_labels": True,
        "outer_labels_used_for_training_or_calibration": False,
        "primary_weights_fit_on_inner_oof_only": True,
        "winner_router_present": False,
        "oracle_winner_supervision_present": False,
        "knn_confidence_or_stability_gate_present": False,
    }
    for key, value in expected.items():
        if provenance.get(key) is not value:
            raise AssertionError(f"invalid provenance flag: {key}")

    predictions = pd.read_csv(required["predictions"])
    aggregate = pd.read_csv(required["aggregate"])
    fold = pd.read_csv(required["fold"])
    weights = pd.read_csv(required["weights"])
    sources = pd.read_csv(required["sources"])
    if predictions.empty or predictions["sample_id"].astype(str).duplicated().any():
        raise AssertionError("empty or duplicated outer predictions")
    if len(predictions) != int(summary["sample_count"]):
        raise AssertionError("sample count mismatch")
    if int(predictions["outer_fold"].nunique()) != int(summary["outer_fold_count"]):
        raise AssertionError("outer fold count mismatch")
    if (
        predictions.groupby(predictions["group_id"].astype(str))["outer_fold"]
        .nunique()
        .max()
        != 1
    ):
        raise AssertionError("conversation appears in multiple outer holdouts")

    action_columns = [f"action_prediction_{name}" for name in ACTION_NAMES_V930]
    applicability_columns = [
        f"applicability_{name}" for name in EXPERT_NAMES
    ]
    required_columns = {
        "label",
        "prediction_fixed_convex_shrinkage",
        "prediction_attribute_weighted",
        "prediction_old_v921_convex_shrinkage",
        "attribute_weight_anchor",
        *action_columns,
        *applicability_columns,
        *(f"attribute_weight_{name}" for name in EXPERT_NAMES),
    }
    missing_columns = required_columns - set(predictions.columns)
    if missing_columns:
        raise AssertionError(
            f"missing prediction columns: {sorted(missing_columns)}"
        )
    numeric = predictions[list(required_columns)].to_numpy(dtype=np.float64)
    if not np.isfinite(numeric).all():
        raise AssertionError("non-finite prediction output")
    applicability = predictions[applicability_columns].to_numpy(
        dtype=np.float64
    )
    floor = float(summary["config"]["applicability_floor"])
    if np.any(applicability < floor - cli.tolerance) or np.any(
        applicability > 1.0 + cli.tolerance
    ):
        raise AssertionError("applicability outside registered range")

    actions = predictions[action_columns].to_numpy(dtype=np.float64)
    attribute_weight_columns = ["attribute_weight_anchor"] + [
        f"attribute_weight_{name}" for name in EXPERT_NAMES
    ]
    attribute_weights = predictions[attribute_weight_columns].to_numpy(
        dtype=np.float64
    )
    if np.any(attribute_weights < -cli.tolerance):
        raise AssertionError("negative deterministic attribute weight")
    if not np.allclose(
        attribute_weights.sum(axis=1),
        1.0,
        atol=cli.tolerance,
        rtol=0.0,
    ):
        raise AssertionError("attribute weights do not sum to one")
    reconstructed_attribute = (actions * attribute_weights).sum(axis=1)
    if not np.allclose(
        reconstructed_attribute,
        predictions["prediction_attribute_weighted"].to_numpy(
            dtype=np.float64
        ),
        atol=cli.tolerance,
        rtol=0.0,
    ):
        raise AssertionError(
            "attribute-weighted prediction does not reconstruct"
        )

    primary_weights = weights[
        weights["weight_type"] == "inner_oof_fixed_convex"
    ].copy()
    for outer_fold, local in predictions.groupby("outer_fold", sort=True):
        local_weights = primary_weights[
            primary_weights["outer_fold"].astype(str) == str(int(outer_fold))
        ]
        if len(local_weights) != len(ACTION_NAMES_V930):
            raise AssertionError(
                f"fold {outer_fold} fixed weight vector incomplete"
            )
        vector = (
            local_weights.set_index("action")
            .loc[list(ACTION_NAMES_V930), "weight"]
            .to_numpy(dtype=np.float64)
        )
        if np.any(vector < -cli.tolerance):
            raise AssertionError(
                f"fold {outer_fold} has negative fixed weight"
            )
        assert_close(
            float(vector.sum()),
            1.0,
            cli.tolerance,
            f"fold {outer_fold} simplex",
        )
        local_actions = local[action_columns].to_numpy(dtype=np.float64)
        reconstructed = local_actions @ vector
        if not np.allclose(
            reconstructed,
            local["prediction_fixed_convex_shrinkage"].to_numpy(
                dtype=np.float64
            ),
            atol=cli.tolerance,
            rtol=0.0,
        ):
            raise AssertionError(
                f"fold {outer_fold} fixed prediction does not reconstruct"
            )

    aggregate_by_strategy = aggregate.set_index("strategy")
    labels = predictions["label"].to_numpy(dtype=np.float64)
    for column in [
        value
        for value in predictions.columns
        if value.startswith("prediction_")
    ]:
        strategy = column.removeprefix("prediction_")
        if strategy not in aggregate_by_strategy.index:
            raise AssertionError(f"aggregate table missing {strategy}")
        value = float(
            np.abs(
                predictions[column].to_numpy(dtype=np.float64) - labels
            ).mean()
        )
        assert_close(
            value,
            float(aggregate_by_strategy.loc[strategy, "mae"]),
            cli.tolerance,
            f"aggregate MAE {strategy}",
        )

    primary_mae = float(
        aggregate_by_strategy.loc["fixed_convex_shrinkage", "mae"]
    )
    old_mae = float(
        aggregate_by_strategy.loc["old_v921_convex_shrinkage", "mae"]
    )
    assert_close(
        primary_mae,
        float(summary["primary_mae"]),
        cli.tolerance,
        "primary summary",
    )
    assert_close(
        old_mae,
        float(summary["old_v921_mae"]),
        cli.tolerance,
        "V9.21 summary",
    )
    assert_close(
        old_mae - primary_mae,
        float(summary["gain_vs_old_v921"]),
        cli.tolerance,
        "gain summary",
    )

    expected_fold_strategies = {
        "anchor",
        *EXPERT_NAMES,
        "equal_mean",
        "attribute_weighted",
        "fixed_convex_shrinkage",
        "old_v921_convex_shrinkage",
        "fold_label_cheating_convex",
        "sample_oracle",
    }
    for outer_fold, local in fold.groupby("outer_fold"):
        if set(local["strategy"].astype(str)) != expected_fold_strategies:
            raise AssertionError(
                f"fold {outer_fold} strategy inventory incomplete"
            )

    if len(sources) != 2 * int(summary["outer_fold_count"]):
        raise AssertionError(
            "source manifest must contain inner and outer V9.19 pools"
        )
    for row in sources.itertuples(index=False):
        path = Path(row.path)
        if not path.is_file():
            raise FileNotFoundError(path)
        if sha256(path) != str(row.sha256):
            raise AssertionError(f"source checksum changed: {path}")

    stack_pools = sorted(
        result_dir.glob(
            "outer_fold_*/**/observable_attribute_target_pool_v930.pth"
        )
    )
    if not stack_pools:
        raise AssertionError("no V9.30 stack pools found")
    checkpoint_count = 0
    for path in stack_pools:
        payload = torch.load(path, map_location="cpu")
        if payload.get("version") != AUDIT_VERSION:
            raise AssertionError(f"stack version mismatch: {path}")
        stack_provenance = payload.get("provenance", {})
        if stack_provenance.get("labels_used_to_compute_attributes") is not False:
            raise AssertionError(
                f"label-derived attributes detected: {path}"
            )
        if stack_provenance.get("target_labels_used_for_training") is not False:
            raise AssertionError(f"target-label training detected: {path}")
        if stack_provenance.get("winner_or_regret_router_trained") is not False:
            raise AssertionError(f"winner router detected: {path}")
        attribute_inputs = set(
            stack_provenance.get("attribute_inputs", ())
        )
        if attribute_inputs != {
            "prediction_LAV",
            "prediction_L",
            "prediction_LA",
            "prediction_LV",
        }:
            raise AssertionError(
                f"unexpected attribute inputs: {attribute_inputs}"
            )
        mode_predictions = {
            mode: torch.as_tensor(payload["mode_predictions"][mode])
            .view(-1)
            .numpy()
            for mode in MODE_NAMES
        }
        recomputed_raw = raw_attribute_scores(mode_predictions)
        for expert_name in EXPERT_NAMES:
            stored_raw = (
                torch.as_tensor(
                    payload["raw_attribute_scores"][expert_name]
                )
                .view(-1)
                .numpy()
            )
            if not np.allclose(
                stored_raw,
                recomputed_raw[expert_name],
                atol=cli.tolerance,
                rtol=0.0,
            ):
                raise AssertionError(
                    "attribute formula does not recompute for "
                    f"{expert_name}: {path}"
                )
        local_checkpoints = list(path.parent.glob("*_expert_v930.pth"))
        if len(local_checkpoints) != len(EXPERT_NAMES):
            raise AssertionError(
                f"expert checkpoint inventory incomplete: {path.parent}"
            )
        checkpoint_count += len(local_checkpoints)

    audit = {
        "version": AUDIT_VERSION,
        "passed": True,
        "result_dir": str(result_dir),
        "sample_count": int(len(predictions)),
        "outer_fold_count": int(predictions["outer_fold"].nunique()),
        "stack_pool_count": len(stack_pools),
        "expert_checkpoint_count": checkpoint_count,
        "primary_mae": primary_mae,
        "old_v921_mae": old_mae,
        "gain_vs_old_v921": old_mae - primary_mae,
        "checks": {
            "attribute_inputs_are_prediction_only": True,
            "attribute_formulas_recomputed": True,
            "no_target_label_training": True,
            "no_winner_router": True,
            "fixed_predictions_recomputed": True,
            "attribute_predictions_recomputed": True,
            "aggregate_metrics_recomputed": True,
            "source_checksums_verified": True,
            "expert_checkpoints_verified": True,
        },
    }
    (result_dir / "v930_audit_check.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
    )
    print("V9.30 ENGINEERING AND LEAKAGE AUDIT PASSED")
    print("samples:", len(predictions))
    print("stack pools:", len(stack_pools))
    print("expert checkpoints:", checkpoint_count)
    print(
        "old V9.21 / V9.30 MAE:",
        f"{old_mae:.6f}",
        f"{primary_mae:.6f}",
    )
    print("audit check:", result_dir / "v930_audit_check.json")


if __name__ == "__main__":
    main()
