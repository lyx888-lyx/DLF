"""Synthetic smoke checks for DLF role-specialization audit utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd

from trains.singleTask.dlf_role_specialization_utils import (
    SENTIMENT_BINS,
    FixedOrdinalProbe,
    FixedPolarityProbe,
    effective_number_weights,
    intensity_labels,
    ordinal_probe_metrics,
    polarity_labels,
    polarity_probe_metrics,
    role_alignment_gate,
    sentiment_bins,
)


def main():
    labels = np.asarray(
        [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        dtype=float,
    )
    polarity = polarity_labels(labels)
    assert polarity.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2]
    intensity = intensity_labels(labels)
    assert intensity.tolist() == [3, 2, 2, 1, 1, 0, 0, 0, 1, 1, 2, 2, 3]
    seven = sentiment_bins(labels)
    assert set(seven.tolist()) == set(SENTIMENT_BINS)

    weights = effective_number_weights({-3: 2, -2: 4, -1: 8, 0: 16, 1: 8, 2: 4, 3: 2})
    assert len(weights) == 7
    assert np.isclose(weights.effective_weight.mean(), 1.0)
    assert np.isclose(weights.effective_weight_capped.mean(), 1.0)
    assert weights.loc[weights["class"].eq(-3), "effective_weight"].iloc[0] > weights.loc[
        weights["class"].eq(0), "effective_weight"
    ].iloc[0]

    rng = np.random.RandomState(7)
    train_y = np.repeat(np.arange(4), 30)
    train_x = np.stack(
        [train_y + rng.normal(scale=0.2, size=len(train_y)), rng.normal(size=len(train_y))],
        axis=1,
    )
    valid_y = np.repeat(np.arange(4), 10)
    valid_x = np.stack(
        [valid_y + rng.normal(scale=0.2, size=len(valid_y)), rng.normal(size=len(valid_y))],
        axis=1,
    )
    ordinal = FixedOrdinalProbe.fit(train_x, train_y)
    ordinal_prediction, probabilities = ordinal.predict(valid_x)
    assert probabilities.shape == (len(valid_y), 3)
    assert (probabilities[:, 0] + 1e-12 >= probabilities[:, 1]).all()
    assert (probabilities[:, 1] + 1e-12 >= probabilities[:, 2]).all()
    ordinal_metrics = ordinal_probe_metrics(valid_y, ordinal_prediction)
    assert ordinal_metrics["ordinal_mae"] < 0.5

    train_polarity = np.repeat(np.arange(3), 40)
    train_polarity_x = np.stack(
        [train_polarity + rng.normal(scale=0.15, size=len(train_polarity)), rng.normal(size=len(train_polarity))],
        axis=1,
    )
    valid_polarity = np.repeat(np.arange(3), 12)
    valid_polarity_x = np.stack(
        [valid_polarity + rng.normal(scale=0.15, size=len(valid_polarity)), rng.normal(size=len(valid_polarity))],
        axis=1,
    )
    polarity_probe = FixedPolarityProbe.fit(train_polarity_x, train_polarity)
    polarity_prediction = polarity_probe.predict(valid_polarity_x)
    polarity_metrics = polarity_probe_metrics(valid_polarity, polarity_prediction)
    assert polarity_metrics["macro_f1"] > 0.9

    comparisons = pd.DataFrame(
        [
            {
                "Seed": seed,
                "Mode": mode,
                "polarity_advantage": 0.02,
                "intensity_advantage": 0.03,
            }
            for seed in (1111, 1114)
            for mode in ("LAV", "LA", "LV", "L")
        ]
    )
    gate = role_alignment_gate(comparisons)
    assert gate["passed"]
    assert gate["status"] == "ROLE_ALIGNMENT_SUPPORTED"
    print("DLF role-specialization utility smoke test passed")


if __name__ == "__main__":
    main()
