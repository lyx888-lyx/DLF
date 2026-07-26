#!/usr/bin/env python3
"""Post-extraction integrity audit for all ten schema-v2 feature ledgers."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from stage23d_self_risk_common import (
    EXPERTS,
    FOLDS,
    MODE_MASKS,
    MODES,
    OUT,
    atomic_json,
    atomic_tsv,
    sha256_file,
    sha256_json,
    utc_now,
)


def main():
    split = pd.read_csv(
        OUT / "protocol" / "sample_split_manifest.csv",
        dtype={"sample_id": str, "video_id": str},
    )
    rows = []
    for fold in FOLDS:
        expected = split.loc[split["checkpoint_fold"].astype(int) == fold]
        expected_ids = set(expected["sample_id"])
        expected_role = expected.set_index("sample_id")["self_risk_role"].to_dict()
        expected_source = expected.set_index("sample_id")["video_id"].to_dict()
        for expert in EXPERTS:
            directory = OUT / "features" / f"checkpoint_fold{fold}" / expert
            manifest_path = directory / "extraction_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            checkpoint_sha = manifest["checkpoint_sha256"]
            audit = {
                "checkpoint_fold": fold,
                "expert_id": expert,
                "feature_schema_version": manifest.get("feature_schema_version"),
                "samples": manifest.get("samples"),
                "rows": manifest.get("rows"),
                "max_abs_final_replay_diff": manifest.get(
                    "max_abs_final_replay_diff"
                ),
                "hook_prediction_max_abs_diff": manifest.get(
                    "hook_prediction_max_abs_diff"
                ),
                "duplicates": 0,
                "missing": 0,
                "source_leakage": 0,
                "row_binding_mismatch": 0,
                "artifact_sha_mismatch": 0,
                "raw_binding_mismatch": 0,
                "label_leakage": 0,
                "inactive_head_leakage": 0,
                "nan_inf_count": 0,
                "a2_lfa_fusion_missing": 0,
            }
            ledgers = []
            for mode in MODES:
                artifact = manifest["feature_artifacts"][mode]
                path = directory / f"static_internal_features_{mode}.csv.gz"
                if sha256_file(path) != artifact["sha256"]:
                    audit["artifact_sha_mismatch"] += 1
                frame = pd.read_csv(
                    path, dtype={"sample_id": str, "video_id": str}
                )
                ledgers.append(frame)
                audit["duplicates"] += int(frame["sample_id"].duplicated().sum())
                audit["missing"] += len(
                    expected_ids.symmetric_difference(set(frame["sample_id"]))
                )
                audit["label_leakage"] += int("label" in frame.columns)
                numeric = frame.select_dtypes(include=[np.number]).to_numpy()
                audit["nan_inf_count"] += int(
                    numeric.size - np.isfinite(numeric).sum()
                )
                available = MODE_MASKS[mode]
                for name, flag in zip(("l", "a", "v"), available):
                    value_columns = [
                        column
                        for column in frame.columns
                        if column.startswith(f"specific_{name}__")
                        or column.startswith(f"aligned_shared_{name}__")
                    ]
                    if not flag:
                        audit["inactive_head_leakage"] += len(value_columns)
                required_a2 = {"fusion_input__l2", "lfa_l__l2"}
                if available[1]:
                    required_a2.update(
                        {"lfa_a__l2", "lfa_cross_a__l2", "lfa_ffn_a__l2"}
                    )
                if available[2]:
                    required_a2.update(
                        {"lfa_v__l2", "lfa_cross_v__l2", "lfa_ffn_v__l2"}
                    )
                audit["a2_lfa_fusion_missing"] += len(
                    required_a2 - set(frame.columns)
                )
                expected_bindings = [
                    sha256_json(
                        {
                            "sample_id": sample_id,
                            "mode": mode,
                            "expert_id": expert,
                            "checkpoint_sha256": checkpoint_sha,
                        }
                    )
                    for sample_id in frame["sample_id"]
                ]
                audit["row_binding_mismatch"] += int(
                    np.sum(
                        frame["row_binding_sha256"].astype(str).to_numpy()
                        != np.asarray(expected_bindings)
                    )
                )
                raw_info = manifest["raw_vector_artifacts"][mode]
                raw_path = directory / f"raw_internal_vectors_{mode}.npz"
                if sha256_file(raw_path) != raw_info["sha256"]:
                    audit["artifact_sha_mismatch"] += 1
                with np.load(raw_path) as raw:
                    audit["raw_binding_mismatch"] += int(
                        raw["features"].shape
                        != (raw_info["rows"], raw_info["dimensions"])
                    )
                    audit["raw_binding_mismatch"] += int(
                        np.sum(
                            raw["sample_id"].astype(str)
                            != frame["sample_id"].astype(str).to_numpy()
                        )
                    )
                    audit["raw_binding_mismatch"] += int(
                        np.sum(
                            raw["row_binding_sha256"].astype(str)
                            != frame["row_binding_sha256"].astype(str).to_numpy()
                        )
                    )
                for path_key, sha_key in (
                    ("development_label_path", "development_label_sha256"),
                    ("sealed_outer_label_path", "sealed_outer_label_sha256"),
                ):
                    if sha256_file(artifact[path_key]) != artifact[sha_key]:
                        audit["artifact_sha_mismatch"] += 1
            combined = pd.concat(ledgers, ignore_index=True)
            role_count = combined.groupby("video_id")["self_risk_role"].nunique()
            audit["source_leakage"] = int((role_count > 1).sum())
            audit["missing"] += int(
                sum(
                    expected_role[sample_id] != role
                    or expected_source[sample_id] != source
                    for sample_id, role, source in combined[
                        ["sample_id", "self_risk_role", "video_id"]
                    ].itertuples(index=False, name=None)
                )
            )
            integrity_fields = (
                "duplicates",
                "missing",
                "source_leakage",
                "row_binding_mismatch",
                "artifact_sha_mismatch",
                "raw_binding_mismatch",
                "label_leakage",
                "inactive_head_leakage",
                "nan_inf_count",
                "a2_lfa_fusion_missing",
            )
            audit["pass"] = bool(
                manifest.get("feature_schema_version") == 2
                and manifest.get("samples") == len(expected_ids)
                and manifest.get("rows") == len(expected_ids) * len(MODES)
                and manifest.get("max_abs_final_replay_diff", 1.0) <= 1e-6
                and manifest.get("hook_prediction_max_abs_diff", 1.0) <= 1e-12
                and all(audit[field] == 0 for field in integrity_fields)
            )
            rows.append(audit)
    table = pd.DataFrame(rows)
    output_dir = OUT / "audit"
    atomic_tsv(table, output_dir / "feature_extraction_audit.tsv")
    summary = {
        "stage": "Stage23D-A schema-v2 feature extraction integrity audit",
        "status": "PASS" if table["pass"].all() else "FAIL",
        "checkpoint_audits": len(table),
        "passed": int(table["pass"].sum()),
        "failed": int((~table["pass"]).sum()),
        "max_abs_final_replay_diff": float(
            table["max_abs_final_replay_diff"].max()
        ),
        "hook_prediction_max_abs_diff": float(
            table["hook_prediction_max_abs_diff"].max()
        ),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "completed_at": utc_now(),
    }
    atomic_json(output_dir / "feature_extraction_audit.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if summary["status"] != "PASS":
        raise RuntimeError("Feature extraction integrity gate failed")


if __name__ == "__main__":
    main()
