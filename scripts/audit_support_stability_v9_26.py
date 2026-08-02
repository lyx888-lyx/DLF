"""Independent engineering audit for V9.26 artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.same_stack_expert_factory_v919 import sha256  # noqa: E402
from trains.singleTask.support_stability_self_knowledge_v926 import (  # noqa: E402
    AUDIT_VERSION,
    BASE_METHOD,
    PRIMARY_METHOD,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--v919-root",
        default="result/full_same_stack_nested_crossfit_v919/mosi/seed_1111",
    )
    parser.add_argument("--outer-folds", type=int, default=5)
    return parser.parse_args()


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main():
    cli = parse_args()
    output = Path(cli.v919_root) / "v926_support_stability_audit"
    required = {
        name: require(output / name)
        for name in (
            "v926_summary.json",
            "v926_report.md",
            "v926_outer_risk_predictions.csv",
            "v926_binary_metrics.csv",
            "v926_continuous_metrics.csv",
            "v926_ranked_subset_metrics.csv",
            "v926_outer_fold_safe_gain.csv",
            "v926_risk_split_manifest.csv",
            "v926_source_integrity.csv",
            "v926_model_inventory.csv",
            "v926_feature_schema.csv",
        )
    }
    summary = json.loads(
        required["v926_summary.json"].read_text(encoding="utf-8")
    )
    if summary.get("version") != AUDIT_VERSION:
        raise RuntimeError("unexpected V9.26 summary version")
    provenance = summary.get("provenance", {})
    expected_false = (
        "base_experts_trained_or_modified",
        "v921_predictions_modified",
        "outer_labels_used_for_fit_calibration_or_support",
        "router_or_action_selection_executed",
        "expert_predictions_replaced_or_mixed",
        "true_raw_input_perturbations_performed",
        "official_validation_or_test_used",
    )
    for key in expected_false:
        if provenance.get(key) is not False:
            raise RuntimeError(f"invalid provenance flag: {key}")
    expected_true = (
        "risk_heads_low_capacity_linear",
        "support_uses_inner_fit_labels_only",
        "same_conversation_neighbours_excluded",
    )
    for key in expected_true:
        if provenance.get(key) is not True:
            raise RuntimeError(f"invalid provenance flag: {key}")

    outer = pd.read_csv(required["v926_outer_risk_predictions.csv"])
    methods = set(outer["method"].astype(str))
    if methods != {BASE_METHOD, PRIMARY_METHOD}:
        raise RuntimeError(f"unexpected methods: {methods}")
    if outer.duplicated(
        ["method", "outer_fold", "sample_id", "expert"]
    ).any():
        raise RuntimeError("duplicate outer method/expert/sample row")
    method_counts = outer.groupby("method").size()
    if method_counts.nunique() != 1:
        raise RuntimeError("base/support outer row counts differ")
    if outer[outer["method"] == PRIMARY_METHOD][
        "support_min_distance"
    ].isna().any():
        raise RuntimeError("missing support distance on primary rows")

    sources = pd.read_csv(required["v926_source_integrity.csv"])
    if len(sources) != cli.outer_folds * 2:
        raise RuntimeError("source inventory count mismatch")
    for row in sources.itertuples(index=False):
        path = require(Path(row.path))
        if sha256(path) != str(row.sha256):
            raise RuntimeError(f"source hash mismatch: {path}")

    inventory = pd.read_csv(required["v926_model_inventory.csv"])
    expected_models = cli.outer_folds * 4
    if len(inventory) != expected_models:
        raise RuntimeError("risk model inventory count mismatch")
    if (inventory["support_reference_count"] <= 0).any():
        raise RuntimeError("empty support library")
    if inventory["feature_count"].nunique() != 1:
        raise RuntimeError("V9.26 feature count changed across folds/experts")
    for row in inventory.itertuples(index=False):
        path = require(Path(row.model_path))
        if sha256(path) != str(row.model_sha256):
            raise RuntimeError(f"risk model hash mismatch: {path}")
        payload = torch.load(path, map_location="cpu")
        model_provenance = payload.get("provenance", {})
        if model_provenance.get(
            "outer_labels_used_for_fit_calibration_or_support"
        ) is not False:
            raise RuntimeError("outer-label provenance violation")
        if model_provenance.get(
            "same_conversation_neighbours_excluded"
        ) is not True:
            raise RuntimeError("same-group support exclusion missing")
        if model_provenance.get(
            "router_or_action_selection_present"
        ) is not False:
            raise RuntimeError("router unexpectedly present")
        if model_provenance.get("raw_input_perturbation_claimed") is not False:
            raise RuntimeError("raw perturbation incorrectly claimed")

    schema = pd.read_csv(required["v926_feature_schema.csv"])
    expected_schema_rows = int(inventory["feature_count"].sum())
    if len(schema) != expected_schema_rows:
        raise RuntimeError("feature schema inventory mismatch")
    allowed_prefixes = (
        "function_",
        "anchor",
        "baseline",
        "action_",
        "own_",
        "confidence_",
        "prediction_",
        "stability_",
        "support_",
    )
    invalid = ~schema["feature_name"].str.startswith(allowed_prefixes)
    if invalid.any():
        unknown = schema.loc[invalid, "feature_name"].unique()
        raise RuntimeError(f"unexpected feature names: {unknown[:5]}")

    manifest = pd.read_csv(required["v926_risk_split_manifest.csv"])
    if set(manifest["method"].astype(str)) != {BASE_METHOD, PRIMARY_METHOD}:
        raise RuntimeError("split manifest methods missing")
    support_manifest = manifest[manifest["method"] == PRIMARY_METHOD]
    if (support_manifest["support_reference_count"] <= 0).any():
        raise RuntimeError("split-local support library missing")

    print("V9.26 SUPPORT-STABILITY ENGINEERING AUDIT PASSED")
    print("base experts trained or modified: False")
    print("raw input perturbations performed: False")
    print("router or action selection executed: False")
    print("risk signal supported:", summary["risk_signal_supported"])


if __name__ == "__main__":
    main()
