"""Canonicalize the MOSI feature summary from its serialized sample table.

This utility performs no model loading or inference.  It makes
``modality_feature_sample_quality.csv`` the single numerical source for
``modality_feature_summary.csv`` and refreshes the corresponding manifest hash.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trains.singleTask.mosi_cfcompat_audit_utils import sha256_file


FEATURE_KEYS = ["Split", "Modality"]
FEATURE_COLUMNS = [
    "Split",
    "Modality",
    "sample_count",
    "finite_fraction_mean",
    "effective_length_mean",
    "effective_length_std",
    "zero_fraction_mean",
    "mean_abs_value",
    "sample_std_mean",
    "all_zero_fraction",
    "near_constant_fraction",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    return parser.parse_args()


def recompute_feature_summary(feature_samples: pd.DataFrame) -> pd.DataFrame:
    required = {
        "Split",
        "Modality",
        "sample_index",
        "finite_fraction",
        "effective_length",
        "zero_fraction",
        "mean_abs_value",
        "sample_std",
        "all_zero",
        "near_constant",
    }
    missing = required.difference(feature_samples.columns)
    if missing:
        raise ValueError(
            "Feature sample table lacks columns: {}".format(sorted(missing))
        )
    if feature_samples.duplicated(["Split", "Modality", "sample_index"]).any():
        raise ValueError("Feature sample table has duplicate sample bindings.")
    numeric = [
        "sample_index",
        "finite_fraction",
        "effective_length",
        "zero_fraction",
        "mean_abs_value",
        "sample_std",
        "all_zero",
        "near_constant",
    ]
    if not np.isfinite(feature_samples[numeric].to_numpy(dtype=float)).all():
        raise ValueError("Feature sample table contains NaN or Inf.")
    result = (
        feature_samples.groupby(FEATURE_KEYS, as_index=False, sort=True)
        .agg(
            sample_count=("sample_index", "size"),
            finite_fraction_mean=("finite_fraction", "mean"),
            effective_length_mean=("effective_length", "mean"),
            effective_length_std=("effective_length", "std"),
            zero_fraction_mean=("zero_fraction", "mean"),
            mean_abs_value=("mean_abs_value", "mean"),
            sample_std_mean=("sample_std", "mean"),
            all_zero_fraction=("all_zero", "mean"),
            near_constant_fraction=("near_constant", "mean"),
        )
    )
    return result[FEATURE_COLUMNS]


def canonicalize_result_dir(result_dir: Path) -> dict:
    root = Path(result_dir)
    sample_path = root / "modality_feature_sample_quality.csv"
    summary_path = root / "modality_feature_summary.csv"
    manifest_path = root / "source_manifest.json"
    for path in (sample_path, summary_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sample_artifact = manifest.get("artifacts", {}).get(sample_path.name)
    if not sample_artifact:
        raise ValueError("Manifest lacks the feature-sample artifact binding.")
    current_sample_sha = sha256_file(sample_path)
    if sample_artifact.get("sha256") != current_sample_sha:
        raise ValueError("Feature-sample artifact SHA does not match the manifest.")

    feature_samples = pd.read_csv(sample_path)
    canonical = recompute_feature_summary(feature_samples)
    canonical.to_csv(summary_path, index=False, float_format="%.17g")

    roundtrip = pd.read_csv(summary_path)
    if list(roundtrip.columns) != FEATURE_COLUMNS or len(roundtrip) != len(canonical):
        raise RuntimeError("Canonical feature summary changed shape after CSV roundtrip.")
    left = canonical.sort_values(FEATURE_KEYS).reset_index(drop=True)
    right = roundtrip.sort_values(FEATURE_KEYS).reset_index(drop=True)
    if not left[FEATURE_KEYS].astype(str).equals(right[FEATURE_KEYS].astype(str)):
        raise RuntimeError("Canonical feature-summary keys changed after roundtrip.")
    if not np.array_equal(
        left.sample_count.to_numpy(dtype=np.int64),
        right.sample_count.to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("Canonical feature-summary counts changed after roundtrip.")
    float_columns = [
        column for column in FEATURE_COLUMNS
        if column not in {"Split", "Modality", "sample_count"}
    ]
    if not np.allclose(
        left[float_columns].to_numpy(dtype=float),
        right[float_columns].to_numpy(dtype=float),
        atol=1e-12,
        rtol=1e-12,
        equal_nan=True,
    ):
        raise RuntimeError("Canonical feature summary is not CSV-roundtrip stable.")

    summary_sha = sha256_file(summary_path)
    manifest["artifacts"][summary_path.name] = {
        "path": str(summary_path.resolve()),
        "sha256": summary_sha,
    }
    manifest["feature_summary_canonicalization"] = {
        "version": "serialized_feature_samples_v1",
        "source_artifact": sample_path.name,
        "source_sha256": current_sample_sha,
        "summary_artifact": summary_path.name,
        "summary_sha256": summary_sha,
        "model_loading_performed": False,
        "inference_performed": False,
        "official_test_accessed": False,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return {
        "sample_count": int(len(feature_samples)),
        "group_count": int(len(canonical)),
        "sample_sha256": current_sample_sha,
        "summary_sha256": summary_sha,
    }


def main():
    cli = parse_args()
    result = canonicalize_result_dir(Path(cli.result_dir))
    print("MOSI feature summary canonicalized from serialized sample table")
    for key, value in result.items():
        print("{}: {}".format(key, value))
    print("official Test was not accessed")


if __name__ == "__main__":
    main()
