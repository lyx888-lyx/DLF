"""Synthetic smoke test for the V9.26 support-and-stability audit."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from trains.singleTask.expert_self_risk_v925 import build_expert_targets  # noqa: E402
from trains.singleTask.support_stability_self_knowledge_v926 import (  # noqa: E402
    SupportStabilityConfigV926,
    average_precision,
    crossfit_support_stability_predictions,
    fit_outer_support_stability_bundle,
    predict_outer_support_stability,
)


def synthetic_payload(seed: int, group_count: int, rows_per_group: int):
    rng = np.random.default_rng(seed)
    n = group_count * rows_per_group
    group_index = np.repeat(np.arange(group_count), rows_per_group)
    latent = rng.normal(0.0, 1.25, size=n)
    group_shift = rng.normal(0.0, 0.25, size=group_count)[group_index]
    labels = np.clip(latent + group_shift, -3.0, 3.0)
    anchor = labels + rng.normal(0.0, 0.55, size=n)
    function_space = np.column_stack(
        [
            labels + rng.normal(0.0, scale, size=n)
            for scale in (0.55, 0.65, 0.70, 0.45)
        ]
    )

    regions = (
        labels < -1.5,
        np.abs(labels) <= 0.5,
        (labels > 0.5) & (labels <= 1.5),
        labels > 1.5,
    )
    predictions = []
    confidences = []
    corrections = []
    for index, mask in enumerate(regions):
        center = (-2.1, 0.0, 1.0, 2.1)[index]
        local_support = np.exp(-np.abs(labels - center))
        noise_scale = np.where(mask, 0.18, 0.80)
        prediction = labels + rng.normal(0.0, noise_scale, size=n)
        confidence = np.clip(
            0.10 + 0.82 * local_support + rng.normal(0.0, 0.04, size=n),
            0.0,
            1.0,
        )
        predictions.append(prediction)
        confidences.append(confidence)
        corrections.append(prediction - anchor)

    return {
        "labels": torch.tensor(labels, dtype=torch.float32).view(-1, 1),
        "anchor": torch.tensor(anchor, dtype=torch.float32).view(-1, 1),
        "function_space": torch.tensor(function_space, dtype=torch.float32),
        "expert_predictions": torch.tensor(
            np.column_stack(predictions), dtype=torch.float32
        ).unsqueeze(-1),
        "expert_confidences": torch.tensor(
            np.column_stack(confidences), dtype=torch.float32
        ).unsqueeze(-1),
        "expert_corrections": torch.tensor(
            np.column_stack(corrections), dtype=torch.float32
        ).unsqueeze(-1),
        "sample_ids": [f"sample-{seed}-{index}" for index in range(n)],
        "group_ids": [f"group-{seed}-{value}" for value in group_index],
    }


def main():
    config = SupportStabilityConfigV926()
    config.validate()
    inner = synthetic_payload(71, group_count=28, rows_per_group=9)
    outer = synthetic_payload(83, group_count=10, rows_per_group=8)
    inner_baseline = torch.as_tensor(inner["anchor"]).view(-1).numpy()
    outer_baseline = torch.as_tensor(outer["anchor"]).view(-1).numpy()
    folds = np.repeat(np.arange(4), 7 * 9)
    if len(folds) != len(inner["sample_ids"]):
        raise AssertionError("synthetic fold length changed")

    expert = "positive"
    crossfit = crossfit_support_stability_predictions(
        inner,
        inner_baseline,
        expert,
        folds,
        config,
    )
    if len(crossfit["predictions"]["pred_win_vs_baseline_prob"]) != len(
        inner["sample_ids"]
    ):
        raise AssertionError("cross-fit prediction length mismatch")
    if len(crossfit["feature_names"]) <= 60:
        raise AssertionError("support/stability features were not appended")

    fitted = fit_outer_support_stability_bundle(
        inner,
        inner_baseline,
        expert,
        folds,
        config,
    )
    outer_result = predict_outer_support_stability(
        fitted,
        outer,
        outer_baseline,
        expert,
        config,
    )
    probability = outer_result["predictions"][
        "pred_large_harm_030_prob"
    ]
    if not np.isfinite(probability).all():
        raise AssertionError("non-finite outer risk probability")
    if np.any((probability < 0.0) | (probability > 1.0)):
        raise AssertionError("outer probability outside [0,1]")
    if not np.isfinite(
        outer_result["support"]["minimum_neighbour_distance"]
    ).all():
        raise AssertionError("non-finite neighbour distance")

    targets = build_expert_targets(
        outer, outer_baseline, expert, config.risk
    )
    ap = average_precision(targets["large_harm_030"], probability)
    if not (np.isnan(ap) or 0.0 <= ap <= 1.0):
        raise AssertionError("invalid average precision")

    print("V9.26 SUPPORT-STABILITY SMOKE TEST PASSED")
    print("feature_count:", len(fitted["feature_names"]))
    print("support_reference_count:", fitted["support_reference_count"])


if __name__ == "__main__":
    main()
