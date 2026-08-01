"""Engineering and statistical audit for V9.17 fixed expert region tables."""

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

from trains.singleTask.fixed_expert_region_audit_v917 import (  # noqa: E402
    ACTION_NAMES,
    AUDIT_VERSION,
    DESIGNATED_ACTION_BY_REGION,
    REGION_NAMES,
)
from trains.singleTask.role_conditioned_experts_v9 import region_index  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="result/fixed_expert_region_audit_v917/mosi/seed_1111",
    )
    return parser.parse_args()


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def close(left: float, right: float, atol: float = 1e-6):
    return abs(float(left) - float(right)) <= atol


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    cli = parse_args()
    root = Path(cli.root)
    paths = {
        "valid_pool": root / "valid_fixed_expert_pool_v917.pth",
        "test_pool": root / "test_fixed_expert_pool_v917.pth",
        "metrics": root / "v917_region_expert_metrics_long.csv",
        "global": root / "v917_global_expert_metrics.csv",
        "designated": root / "v917_designated_expert_summary.csv",
        "test_designated": root / "v917_test_designated_expert_table.csv",
        "test_designated_md": root / "v917_test_designated_expert_table.md",
        "samples": root / "v917_sample_level_advantages.csv",
        "oracle": root / "v917_region_oracle_summary.csv",
        "specialization": root / "v917_expert_specialization_summary.csv",
        "summary": root / "fixed_expert_region_audit_v917_summary.json",
    }
    for path in paths.values():
        require(path.is_file(), f"missing V9.17 output: {path}")

    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    metrics = pd.read_csv(paths["metrics"])
    global_metrics = pd.read_csv(paths["global"])
    designated = pd.read_csv(paths["designated"])
    test_designated = pd.read_csv(paths["test_designated"])
    samples = pd.read_csv(paths["samples"])
    oracle = pd.read_csv(paths["oracle"])
    specialization = pd.read_csv(paths["specialization"])
    valid_pool = torch.load(paths["valid_pool"], map_location="cpu")
    test_pool = torch.load(paths["test_pool"], map_location="cpu")

    require(summary["version"] == AUDIT_VERSION, "summary version mismatch")
    require(summary["action_names"] == list(ACTION_NAMES), "action order mismatch")
    require(summary["region_names"] == list(REGION_NAMES), "region order mismatch")
    require(
        summary["designated_action_by_region"] == DESIGNATED_ACTION_BY_REGION,
        "designated mapping mismatch",
    )
    require(
        sorted(round(float(value), 2) for value in summary["advantage_thresholds"])
        == [0.05, 0.10],
        "required advantage thresholds missing",
    )

    splits = ("train_oof", "valid", "test")
    require(set(metrics["split"]) == set(splits), "metrics split mismatch")
    require(len(metrics) == 3 * 30, "expected 30 metrics rows per split")
    require(len(global_metrics) == 3 * 5, "global row count mismatch")
    require(len(designated) == 3 * 5, "designated row count mismatch")
    require(len(oracle) == 3 * 5, "oracle row count mismatch")
    require(len(specialization) == 3 * 5, "specialization row count mismatch")
    require(len(test_designated) == 5, "Test designated table row count mismatch")

    region_metrics = metrics[metrics["region"].isin(REGION_NAMES)]
    require(
        set(region_metrics["expert"]) == set(ACTION_NAMES),
        "region metrics action set mismatch",
    )
    require(
        set(region_metrics["region"]) == set(REGION_NAMES),
        "region metrics region set mismatch",
    )
    require(
        bool((region_metrics["sample_count"].astype(int) > 0).all()),
        "empty true-label region",
    )
    rate_columns = [
        "win_rate",
        "tie_rate",
        "loss_rate",
        "large_gain_rate_005",
        "large_gain_rate_010",
        "large_harm_rate_005",
        "large_harm_rate_010",
        "oracle_best_rate",
        "bootstrap_positive_rate",
    ]
    for column in rate_columns:
        require(
            bool(region_metrics[column].astype(float).between(0.0, 1.0).all()),
            f"rate outside [0,1]: {column}",
        )

    expected_designated = {
        (split, region): expert
        for split in splits
        for region, expert in DESIGNATED_ACTION_BY_REGION.items()
    }
    actual_designated = {
        (str(row["split"]), str(row["region"])): str(row["designated_expert"])
        for _, row in designated.iterrows()
    }
    require(actual_designated == expected_designated, "designated rows mismatch")

    provenance = summary["provenance"]
    require(provenance["models_trained"] is False, "audit trained a model")
    require(provenance["router_trained"] is False, "audit trained a router")
    require(provenance["fusion_selected"] is False, "audit selected a fusion")
    require(
        provenance["train_statistics_use_strict_oof_shadow_experts"] is True,
        "Train did not use strict OOF experts",
    )
    require(
        provenance["each_train_prediction_excludes_its_sample"] is True,
        "Train holdout isolation missing",
    )
    require(
        provenance["validation_uses_frozen_full_train_experts"] is True,
        "Validation expert source mismatch",
    )
    require(
        provenance["test_uses_frozen_full_train_experts"] is True,
        "Test expert source mismatch",
    )
    require(
        provenance["test_labels_used_for_training_or_selection"] is False,
        "Test labels entered training/selection",
    )
    require(
        provenance["region_definitions_reused_from_v9_training"] is True,
        "region definitions changed",
    )

    labels = torch.tensor(samples["label"].to_numpy()).float()
    expected_regions = region_index(labels).tolist()
    stored_regions = [REGION_NAMES[int(index)] for index in expected_regions]
    require(
        stored_regions == samples["true_region"].astype(str).tolist(),
        "sample true-region assignment mismatch",
    )

    for split in splits:
        split_samples = samples[samples["split"] == split].reset_index(drop=True)
        require(
            not split_samples["sample_id"].astype(str).duplicated().any(),
            f"duplicate sample IDs in {split}",
        )
        for action in ACTION_NAMES:
            prediction = torch.tensor(
                split_samples[f"{action}_prediction"].to_numpy()
            ).float()
            local_labels = torch.tensor(split_samples["label"].to_numpy()).float()
            error = torch.abs(prediction - local_labels)
            stored_error = torch.tensor(
                split_samples[f"{action}_abs_error"].to_numpy()
            ).float()
            require(
                torch.allclose(error, stored_error, atol=1e-6),
                f"stored error mismatch: {split}/{action}",
            )
            anchor_error = torch.tensor(
                split_samples["anchor_abs_error"].to_numpy()
            ).float()
            advantage = anchor_error - error
            stored_advantage = torch.tensor(
                split_samples[f"{action}_gain_vs_anchor"].to_numpy()
            ).float()
            require(
                torch.allclose(advantage, stored_advantage, atol=1e-6),
                f"stored advantage mismatch: {split}/{action}",
            )

        for region in REGION_NAMES:
            local = split_samples[split_samples["true_region"] == region]
            require(len(local) > 0, f"empty sample region: {split}/{region}")
            anchor_error = torch.tensor(
                local["anchor_abs_error"].to_numpy()
            ).float()
            for action in ACTION_NAMES:
                error = torch.tensor(
                    local[f"{action}_abs_error"].to_numpy()
                ).float()
                advantage = anchor_error - error
                row = region_metrics[
                    (region_metrics["split"] == split)
                    & (region_metrics["region"] == region)
                    & (region_metrics["expert"] == action)
                ]
                require(len(row) == 1, f"missing metric row {split}/{region}/{action}")
                row = row.iloc[0]
                require(
                    int(row["sample_count"]) == len(local),
                    "metric sample count mismatch",
                )
                require(close(error.mean(), row["expert_mae"]), "MAE mismatch")
                require(
                    close(advantage.mean(), row["gain_vs_anchor"]),
                    "gain mismatch",
                )
                require(
                    close(
                        (advantage > 1e-8).float().mean(),
                        row["win_rate"],
                    ),
                    "win-rate mismatch",
                )
                require(
                    close(
                        (advantage > 0.10).float().mean(),
                        row["large_gain_rate_010"],
                    ),
                    "large-gain-rate mismatch",
                )
                require(
                    close(
                        (advantage < -0.10).float().mean(),
                        row["large_harm_rate_010"],
                    ),
                    "large-harm-rate mismatch",
                )

    require(
        len(set(valid_pool["sample_ids"]) & set(test_pool["sample_ids"])) == 0,
        "Validation/Test sample ID overlap",
    )
    require(
        len(samples[samples["split"] == "valid"]) == len(valid_pool["sample_ids"]),
        "Validation sample count mismatch",
    )
    require(
        len(samples[samples["split"] == "test"]) == len(test_pool["sample_ids"]),
        "Test sample count mismatch",
    )

    for split, matrix_paths in summary["outputs"]["matrices"].items():
        require(split in splits, f"unexpected matrix split: {split}")
        for name, value in matrix_paths.items():
            path = Path(value)
            require(path.is_file(), f"missing matrix {split}/{name}: {path}")
            frame = pd.read_csv(path)
            require(len(frame) == 5, f"matrix region count mismatch: {path}")
            require(
                list(frame["region"].astype(str)) == list(REGION_NAMES),
                f"matrix region order mismatch: {path}",
            )
            require(
                set(ACTION_NAMES).issubset(frame.columns),
                f"matrix action columns missing: {path}",
            )

    for name, value in summary["checkpoint_paths"].items():
        path = Path(value)
        require(path.is_file(), f"missing source checkpoint: {name}={path}")
        require(
            sha256(path) == summary["checkpoint_sha256"][name],
            f"source hash mismatch: {name}",
        )

    strict_path = Path(summary["checkpoint_paths"]["strict_oof_pool"])
    strict_pool = torch.load(strict_path, map_location="cpu")
    strict_provenance = strict_pool["provenance"]
    require(
        strict_provenance["is_fully_nested_teacher_stack"] is True,
        "strict Train pool provenance mismatch",
    )
    require(
        strict_provenance["holdout_label_isolation"] is True,
        "strict Train holdout isolation mismatch",
    )
    require(
        strict_provenance["historical_full_train_teacher_reuse"] is False,
        "full-train expert leakage into OOF pool",
    )

    print("V9.17 FIXED EXPERT REGION AUDIT PASSED")
    print("models/router/fusion trained: False")
    print("Train source: strict nested OOF shadow experts")
    print("Validation/Test source: frozen full-train deployment experts")
    print("\nTEST DESIGNATED-EXPERT TABLE")
    print(
        test_designated[
            [
                "region",
                "sample_count",
                "designated_expert",
                "anchor_mae",
                "designated_expert_mae",
                "improvement",
                "win_rate",
                "large_gain_rate_010",
                "large_harm_rate_010",
                "gain_ci_low",
                "gain_ci_high",
            ]
        ].to_string(index=False)
    )
    print("\nTEST BEST FIXED ACTION BY TRUE REGION")
    test_region = region_metrics[region_metrics["split"] == "test"]
    rows = []
    for region in REGION_NAMES:
        best = test_region[test_region["region"] == region].sort_values(
            ["expert_mae", "expert_index"]
        ).iloc[0]
        rows.append(
            {
                "region": region,
                "best_expert": best["expert"],
                "mae": best["expert_mae"],
                "gain_vs_anchor": best["gain_vs_anchor"],
                "win_rate": best["win_rate"],
            }
        )
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
