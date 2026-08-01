"""CPU smoke test for V9.17 fixed expert region audit primitives."""

from __future__ import annotations

import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from trains.singleTask.fixed_expert_region_audit_v917 import (  # noqa: E402
    ACTION_NAMES,
    DESIGNATED_ACTION_BY_REGION,
    REGION_NAMES,
    build_region_audit,
    normalize_expert_pool,
    specialization_rows,
)


def require(condition: bool, message: str):
    if not condition:
        raise AssertionError(message)


def synthetic_strict_pool():
    labels = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0]).view(-1, 1)
    anchor = torch.tensor([-1.5, -0.7, 0.4, 0.4, 1.7]).view(-1, 1)
    # Specialist order: strong_negative, boundary, positive, strong_positive.
    experts = torch.tensor(
        [
            [-1.9, -1.2, -1.1, -1.0],
            [-0.6, -0.8, -0.5, -0.4],
            [0.5, 0.05, 0.6, 0.8],
            [0.2, 0.5, 0.95, 0.6],
            [1.2, 1.4, 1.5, 1.95],
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)
    return {
        "sample_ids": [f"sample_{index}" for index in range(5)],
        "group_ids": [f"group_{index}" for index in range(5)],
        "labels": labels,
        "anchor": anchor,
        "expert_predictions": experts,
        "action_names": ACTION_NAMES,
        "fold_index": torch.arange(5),
    }


def synthetic_frozen_pool():
    strict = synthetic_strict_pool()
    return {
        "sample_ids": strict["sample_ids"],
        "group_ids": strict["group_ids"],
        "labels": strict["labels"],
        "anchor": strict["anchor"],
        "experts": {
            name: {
                "prediction": strict["expert_predictions"][:, index],
            }
            for index, name in enumerate(ACTION_NAMES[1:])
        },
    }


def main():
    strict = normalize_expert_pool(synthetic_strict_pool(), "strict_oof")
    frozen = normalize_expert_pool(synthetic_frozen_pool(), "frozen")
    require(
        torch.allclose(strict["actions"], frozen["actions"]),
        "strict/frozen normalization differs",
    )

    result = build_region_audit(
        strict,
        split="train_oof",
        thresholds=(0.05, 0.10),
        bootstrap_repeats=25,
        bootstrap_seed=917,
    )
    require(len(result["metrics_rows"]) == 25, "expected 5x5 region rows")
    require(len(result["global_rows"]) == 5, "expected five global rows")
    require(len(result["designated_rows"]) == 5, "expected five designated rows")
    require(len(result["sample_rows"]) == 5, "sample rows missing")
    require(len(result["oracle_region_rows"]) == 5, "oracle rows missing")

    region_order = [
        row["region"]
        for row in sorted(
            result["designated_rows"],
            key=lambda row: int(row["region_index"]),
        )
    ]
    require(region_order == list(REGION_NAMES), "region assignment mismatch")
    designated = {
        row["region"]: row["designated_expert"]
        for row in result["designated_rows"]
    }
    require(
        designated == DESIGNATED_ACTION_BY_REGION,
        "designated action mapping mismatch",
    )

    boundary = next(
        row
        for row in result["metrics_rows"]
        if row["region"] == "boundary" and row["expert"] == "boundary"
    )
    require(boundary["sample_count"] == 1, "boundary count mismatch")
    require(boundary["gain_vs_anchor"] > 0.30, "boundary gain not detected")
    require(boundary["win_rate"] == 1.0, "boundary win not detected")
    require(
        boundary["large_gain_rate_010"] == 1.0,
        "boundary large gain not detected",
    )

    strong_negative = next(
        row
        for row in result["metrics_rows"]
        if row["region"] == "strong_negative"
        and row["expert"] == "strong_negative"
    )
    require(
        strong_negative["gain_vs_anchor"] > 0.0,
        "strong-negative expert gain not detected",
    )

    specializations = specialization_rows(
        result["metrics_rows"],
        result["global_rows"],
    )
    require(len(specializations) == 5, "specialization rows missing")
    require(
        all(row["split"] == "train_oof" for row in specializations),
        "specialization split mismatch",
    )

    print("V9.17 FIXED EXPERT REGION AUDIT SMOKE TEST PASSED")
    print("actions:", ACTION_NAMES)
    print("regions:", REGION_NAMES)
    print("designated mapping:", DESIGNATED_ACTION_BY_REGION)


if __name__ == "__main__":
    main()
