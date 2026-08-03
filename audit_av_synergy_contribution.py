"""Independent recomputation audit for AV synergy contribution artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from analyze_av_synergy_contribution import AuditConfig, VERSION, build_artifacts, jsonable


FILE_MAP = {
    "source_manifest": "avsc_source_manifest.csv",
    "per_seed_samples": "avsc_per_seed_samples.csv",
    "per_seed_metrics": "avsc_per_seed_metrics.csv",
    "ensemble_samples": "avsc_ensemble_samples.csv",
    "aggregate_metrics": "avsc_aggregate_metrics.csv",
    "group_bootstrap": "avsc_group_bootstrap.csv",
    "cross_seed_consistency": "avsc_cross_seed_consistency.csv",
    "conditional_regions": "avsc_conditional_regions.csv",
    "video_concentration": "avsc_video_concentration.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def config_from_summary(summary: Mapping[str, object]) -> AuditConfig:
    payload = dict(summary["config"])
    payload["seeds"] = tuple(int(value) for value in payload["seeds"])
    payload["splits"] = tuple(str(value) for value in payload["splits"])
    return AuditConfig(**payload)


def assert_frames_equal(expected: pd.DataFrame, observed: pd.DataFrame, name: str) -> None:
    if list(expected.columns) != list(observed.columns):
        raise AssertionError(f"{name} column mismatch")
    if len(expected) != len(observed):
        raise AssertionError(f"{name} row-count mismatch")
    for column in expected.columns:
        left = expected[column]
        right = observed[column]
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            if not np.allclose(
                left.to_numpy(float),
                right.to_numpy(float),
                atol=1e-10,
                rtol=1e-10,
                equal_nan=True,
            ):
                difference = np.nanmax(np.abs(left.to_numpy(float) - right.to_numpy(float)))
                raise AssertionError(f"{name} numeric mismatch in {column}: {difference}")
        else:
            left_values = left.fillna("<NA>").astype(str).tolist()
            right_values = right.fillna("<NA>").astype(str).tolist()
            if left_values != right_values:
                raise AssertionError(f"{name} value mismatch in {column}")


def assert_json_close(expected, observed, path="root") -> None:
    if isinstance(expected, dict):
        if not isinstance(observed, dict) or set(expected) != set(observed):
            raise AssertionError(f"JSON key mismatch at {path}")
        for key in expected:
            assert_json_close(expected[key], observed[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(observed, list) or len(expected) != len(observed):
            raise AssertionError(f"JSON list mismatch at {path}")
        for index, (left, right) in enumerate(zip(expected, observed)):
            assert_json_close(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if observed is None and isinstance(expected, float) and not math.isfinite(expected):
            return
        if not isinstance(observed, (int, float)) or not math.isclose(
            float(expected), float(observed), rel_tol=1e-10, abs_tol=1e-10
        ):
            raise AssertionError(f"JSON numeric mismatch at {path}: {expected} != {observed}")
        return
    if expected != observed:
        raise AssertionError(f"JSON mismatch at {path}: {expected!r} != {observed!r}")


def main() -> None:
    cli = parse_args()
    root = Path(cli.result_dir)
    summary_path = root / "avsc_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    observed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if observed_summary.get("version") != VERSION:
        raise AssertionError("Version mismatch")
    config = config_from_summary(observed_summary)
    config.validate()
    checks = {}

    observed_manifest = pd.read_csv(root / FILE_MAP["source_manifest"])
    for row in observed_manifest.itertuples(index=False):
        path = Path(str(row.path))
        if not path.is_file():
            raise AssertionError(f"Missing source: {path}")
        if sha256(path) != str(row.sha256):
            raise AssertionError(f"Source SHA mismatch: {path}")
    checks["source_hashes_verified"] = True

    recomputed = build_artifacts(config)
    for key, filename in FILE_MAP.items():
        observed = pd.read_csv(root / filename)
        expected = recomputed[key]
        assert isinstance(expected, pd.DataFrame)
        assert_frames_equal(expected, observed, key)
    checks["all_csv_artifacts_recomputed"] = True

    expected_summary = jsonable(recomputed["summary"])
    assert_json_close(expected_summary, observed_summary)
    checks["summary_and_gates_recomputed"] = True

    if observed_summary.get("models_loaded") is not False:
        raise AssertionError("Audit claims a model was loaded")
    if observed_summary.get("training_performed") is not False:
        raise AssertionError("Audit claims training was performed")
    if observed_summary.get("teacher_or_evaluator_loaded") is not False:
        raise AssertionError("Audit claims a Teacher/evaluator was loaded")
    if observed_summary.get("primary_decision_uses_test") is not False:
        raise AssertionError("Primary decision unexpectedly uses Test")
    checks["zero_training_and_valid_only_decision_verified"] = True

    per_seed = recomputed["per_seed_samples"]
    formula = (
        per_seed["LAV_pred"]
        - per_seed["LA_pred"]
        - per_seed["LV_pred"]
        + per_seed["L_pred"]
    )
    if not np.allclose(formula, per_seed["synergy_effect"], atol=1e-12):
        raise AssertionError("Synergy effect formula mismatch")
    reconstructed = per_seed["no_synergy_pred"] + per_seed["synergy_effect"]
    if not np.allclose(reconstructed, per_seed["LAV_pred"], atol=1e-12):
        raise AssertionError("Factorization reconstruction mismatch")
    checks["counterfactual_factorization_formula_verified"] = True

    result = {
        "version": VERSION,
        "passed": bool(all(checks.values())),
        "checks": checks,
        "verdict": observed_summary["verdict"],
        "decision_split": observed_summary["decision_split"],
        "synergy_gate": observed_summary["synergy_gate"],
        "marginal_gate": observed_summary["marginal_gate"],
    }
    (root / "avsc_audit_check.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("AV SYNERGY CONTRIBUTION INDEPENDENT AUDIT PASSED")
    for key, value in checks.items():
        print(f"{key}: {value}")
    print("verdict:", result["verdict"])


if __name__ == "__main__":
    main()
