"""Synthetic smoke test for V9.25 expert self-risk estimation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from trains.singleTask.expert_self_risk_v925 import (
    BINARY_TARGET_NAMES,
    EXPERT_NAMES,
    ExpertSelfRiskConfigV925,
    add_rank_selection_flags,
    binary_metric_row,
    build_expert_features,
    build_expert_targets,
    continuous_metric_row,
    crossfit_risk_predictions,
    fit_outer_risk_bundle,
    predict_risk_bundle,
    selection_metric_row,
)


def synthetic_pool(seed: int = 1111):
    rng = np.random.default_rng(seed)
    groups = np.repeat(np.arange(75), 4)
    count = len(groups)
    latent = rng.uniform(-2.7, 2.7, size=count)
    labels = np.clip(
        latent + rng.normal(0.0, 0.22, size=count),
        -3.0,
        3.0,
    )
    anchor = labels + rng.normal(0.0, 0.65, size=count)
    function_space = np.column_stack(
        [
            anchor
            + rng.normal(0.0, 0.18 + 0.04 * index, size=count)
            for index in range(4)
        ]
    )
    definitions = {
        "strong_negative": labels < -1.5,
        "boundary": np.abs(labels) <= 0.5,
        "positive": (labels > 0.5) & (labels <= 1.5),
        "strong_positive": labels > 1.5,
    }
    predictions, confidences, corrections = [], [], []
    for expert in EXPERT_NAMES:
        membership = definitions[expert]
        failure = rng.random(count) < (
            0.05 + 0.35 * (np.abs(latent) < 0.2)
        )
        noise_scale = np.where(
            membership & ~failure, 0.18, 0.85
        )
        prediction = labels + rng.normal(
            0.0, noise_scale
        )
        confidence = np.clip(
            0.12
            + 0.76 * membership
            + rng.normal(0.0, 0.10, size=count),
            0.0,
            1.0,
        )
        confidence[failure & membership] = np.maximum(
            confidence[failure & membership], 0.88
        )
        predictions.append(prediction)
        confidences.append(confidence)
        corrections.append(prediction - anchor)
    prediction_matrix = np.column_stack(predictions)
    payload = {
        "labels": torch.tensor(labels).view(-1, 1).float(),
        "anchor": torch.tensor(anchor).view(-1, 1).float(),
        "function_space": torch.tensor(function_space).float(),
        "expert_predictions": torch.tensor(
            prediction_matrix
        ).unsqueeze(-1).float(),
        "expert_confidences": torch.tensor(
            np.column_stack(confidences)
        ).unsqueeze(-1).float(),
        "expert_corrections": torch.tensor(
            np.column_stack(corrections)
        ).unsqueeze(-1).float(),
        "sample_ids": [f"sample_{index}" for index in range(count)],
        "group_ids": [f"group_{group}" for group in groups],
    }
    fold_index = groups % 5
    baseline = (
        0.70 * anchor
        + 0.30 * prediction_matrix.mean(axis=1)
    )
    return payload, baseline, fold_index


def main():
    payload, baseline, fold_index = synthetic_pool()
    config = ExpertSelfRiskConfigV925()
    frames = []
    for expert in EXPERT_NAMES:
        features = build_expert_features(
            payload, baseline, expert
        )
        targets = build_expert_targets(
            payload, baseline, expert, config
        )
        crossfit = crossfit_risk_predictions(
            features["matrix"],
            targets,
            fold_index,
            config,
        )
        predictions = crossfit["predictions"]
        assert set(BINARY_TARGET_NAMES).issubset(targets)
        for name, value in predictions.items():
            assert np.isfinite(value).all(), name
            if name.endswith("_prob"):
                assert (
                    (value >= 0.0) & (value <= 1.0)
                ).all(), name
        win = binary_metric_row(
            targets["win_vs_baseline"],
            predictions["pred_win_vs_baseline_prob"],
            config.calibration_bins,
        )
        error = continuous_metric_row(
            targets["absolute_error"],
            predictions["pred_absolute_error"],
        )
        assert win["auc"] > 0.50
        assert error["spearman"] > 0.0

        final_fit = fit_outer_risk_bundle(
            features["matrix"],
            targets,
            fold_index,
            config,
        )
        final_prediction = predict_risk_bundle(
            final_fit["bundle"],
            features["matrix"][:20],
            config,
        )
        assert all(
            len(value) == 20
            for value in final_prediction.values()
        )
        frames.append(
            pd.DataFrame(
                {
                    "outer_fold": 0,
                    "expert": expert,
                    "native_confidence": features[
                        "native_confidence"
                    ],
                    "absolute_error": targets[
                        "absolute_error"
                    ],
                    "gain_vs_baseline": targets[
                        "gain_vs_baseline"
                    ],
                    "membership": targets["membership"],
                    **predictions,
                }
            )
        )

    frame = add_rank_selection_flags(
        pd.concat(frames, ignore_index=True),
        config.safe_rank_fraction,
    )
    risk = selection_metric_row(
        frame, "selected_risk_top", config
    )
    native = selection_metric_row(
        frame, "selected_native_top", config
    )
    assert risk["selected_count"] > 0
    assert native["selected_count"] > 0
    assert 0.0 < risk["coverage"] < 1.0
    print("V9.25 EXPERT SELF-RISK SMOKE TEST PASSED")


if __name__ == "__main__":
    main()
