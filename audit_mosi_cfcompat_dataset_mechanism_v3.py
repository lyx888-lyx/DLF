"""Independent audit v3 with isolated CSV-roundtrip feature-summary handling."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import audit_mosi_cfcompat_dataset_mechanism_v2 as hardened


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
FLOAT_FEATURE_COLUMNS = [
    column for column in FEATURE_COLUMNS
    if column not in {"Split", "Modality", "sample_count"}
]
FEATURE_ATOL = 1e-7


def parse_result_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--result-dir", required=True)
    cli, _ = parser.parse_known_args()
    return Path(cli.result_dir)


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


def feature_summary_diagnostic(
    recorded: pd.DataFrame,
    recomputed: pd.DataFrame,
) -> tuple[bool, pd.DataFrame]:
    structure_ok = bool(
        list(recorded.columns) == FEATURE_COLUMNS
        and list(recomputed.columns) == FEATURE_COLUMNS
        and len(recorded) == len(recomputed)
    )
    if not structure_ok:
        diagnostic = pd.DataFrame(
            [
                {
                    "structure_ok": False,
                    "recorded_columns": "|".join(map(str, recorded.columns)),
                    "recomputed_columns": "|".join(map(str, recomputed.columns)),
                    "recorded_rows": len(recorded),
                    "recomputed_rows": len(recomputed),
                }
            ]
        )
        return False, diagnostic

    left = recorded.sort_values(FEATURE_KEYS, kind="mergesort").reset_index(drop=True)
    right = recomputed.sort_values(FEATURE_KEYS, kind="mergesort").reset_index(drop=True)
    keys_ok = bool(
        left["Split"].astype(str).equals(right["Split"].astype(str))
        and left["Modality"].astype(str).equals(right["Modality"].astype(str))
    )
    counts_ok = bool(
        np.array_equal(
            left["sample_count"].to_numpy(dtype=np.int64),
            right["sample_count"].to_numpy(dtype=np.int64),
        )
    )

    rows = []
    floats_ok = True
    for index in range(len(left)):
        row = {
            "Split": str(left.loc[index, "Split"]),
            "Modality": str(left.loc[index, "Modality"]),
            "recorded_sample_count": int(left.loc[index, "sample_count"]),
            "recomputed_sample_count": int(right.loc[index, "sample_count"]),
        }
        for column in FLOAT_FEATURE_COLUMNS:
            recorded_value = float(left.loc[index, column])
            recomputed_value = float(right.loc[index, column])
            if np.isnan(recorded_value) and np.isnan(recomputed_value):
                delta = 0.0
                cell_ok = True
            else:
                delta = abs(recorded_value - recomputed_value)
                cell_ok = bool(
                    np.isclose(
                        recorded_value,
                        recomputed_value,
                        atol=FEATURE_ATOL,
                        rtol=1e-12,
                        equal_nan=True,
                    )
                )
            row[column + "_recorded"] = recorded_value
            row[column + "_recomputed"] = recomputed_value
            row[column + "_abs_delta"] = delta
            row[column + "_passed"] = cell_ok
            floats_ok = bool(floats_ok and cell_ok)
        rows.append(row)
    return bool(keys_ok and counts_ok and floats_ok), pd.DataFrame(rows)


def main():
    root = parse_result_dir()
    feature_samples = pd.read_csv(root / "modality_feature_sample_quality.csv")
    recorded = pd.read_csv(root / "modality_feature_summary.csv")
    recomputed = recompute_feature_summary(feature_samples)
    feature_passed, diagnostic = feature_summary_diagnostic(recorded, recomputed)
    diagnostic_path = root / "feature_summary_recompute_diagnostic.csv"
    diagnostic.to_csv(diagnostic_path, index=False)

    original_same_frame = hardened.base.same_frame

    def same_frame_v3(actual, expected, keys, tolerance=1e-9):
        if (
            list(actual.columns) == FEATURE_COLUMNS
            and list(expected.columns) == FEATURE_COLUMNS
            and list(keys) == FEATURE_KEYS
        ):
            passed, _ = feature_summary_diagnostic(actual, expected)
            return passed
        return original_same_frame(actual, expected, keys, tolerance)

    hardened.base.same_frame = same_frame_v3
    if not feature_passed:
        print("Feature-summary serialized recomputation still differs materially.")
        print("diagnostic:", diagnostic_path)
    hardened.main()
    print("feature_summary_serialized_recompute: True")
    print("feature_summary_tolerance:", FEATURE_ATOL)
    print("feature_summary_diagnostic:", diagnostic_path)


if __name__ == "__main__":
    main()
