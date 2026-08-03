"""Synthetic smoke tests for V9.28 modality residual utilities."""

from __future__ import annotations

import numpy as np
import torch

from trains.singleTask.modality_residual_team_v928 import (
    ResidualHeadConfigV928,
    ResidualTeamConfigV928,
    apply_strategy,
    crossfit_residual_predictions,
    fit_residual_head,
    fit_strategy_weights,
    predict_residual_head,
    summarize_sequence,
)


def main():
    rng = np.random.default_rng(928)
    sequence = rng.normal(size=(12, 5)).astype(np.float32)
    sequence[9:] = 0.0
    summary = summarize_sequence(sequence, 9, temporal_bins=4)
    assert summary.shape == (5 * 6 + 1,)
    assert np.isfinite(summary).all()

    n = 180
    features = rng.normal(size=(n, 14)).astype(np.float32)
    anchor = rng.normal(scale=0.8, size=n)
    true_correction = 0.28 * np.tanh(features[:, 0] - 0.5 * features[:, 1])
    labels = anchor + true_correction + rng.normal(scale=0.03, size=n)
    folds = np.repeat(np.arange(3), n // 3)
    config = ResidualHeadConfigV928(
        temporal_bins=4,
        hidden_dim=32,
        dropout=0.0,
        correction_max=0.50,
        epochs=50,
        batch_size=32,
        learning_rate=3e-3,
        weight_decay=1e-4,
        correction_l1=0.0,
    )
    crossfit = crossfit_residual_predictions(
        features,
        anchor,
        labels,
        folds,
        config,
        torch.device("cpu"),
        seed=928,
    )
    correction = crossfit["prediction"]
    assert correction.shape == (n,)
    assert np.isfinite(correction).all()
    base_mae = np.abs(anchor - labels).mean()
    corrected_mae = np.abs(anchor + correction - labels).mean()
    assert corrected_mae < base_mae

    bundle = fit_residual_head(
        features,
        anchor,
        labels,
        config,
        torch.device("cpu"),
        seed=929,
    )
    full_correction = predict_residual_head(
        bundle,
        features,
        anchor,
        torch.device("cpu"),
    )
    assert np.isfinite(full_correction).all()

    correction_matrix = np.column_stack(
        [correction, rng.normal(scale=0.01, size=n), np.zeros(n)]
    )
    team = ResidualTeamConfigV928(
        head=config,
        coefficient_l2=1e-4,
        coefficient_upper_bound=1.0,
        bootstrap_repetitions=100,
    )
    weights = fit_strategy_weights(
        correction_matrix,
        anchor,
        labels,
        team,
    )
    primary = weights["text_plus_all_residuals"]
    assert primary.shape == (3,)
    assert np.all(primary >= -1e-12)
    assert np.all(primary <= 1.0 + 1e-12)
    prediction = apply_strategy(anchor, correction_matrix, primary)
    assert np.abs(prediction - labels).mean() <= base_mae + 1e-8

    print("V9.28 SMOKE TEST PASSED")
    print("summary_dim:", summary.size)
    print("base_mae:", f"{base_mae:.6f}")
    print("crossfit_corrected_mae:", f"{corrected_mae:.6f}")
    print("primary_weights:", primary.tolist())


if __name__ == "__main__":
    main()
