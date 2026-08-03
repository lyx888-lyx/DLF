"""Synthetic smoke tests for V9.30 observable-attribute expert utilities."""

from __future__ import annotations

import numpy as np

from trains.singleTask.observable_attribute_experts_v930 import (
    ACTION_NAMES_V930,
    EXPERT_NAMES,
    ObservableAttributeConfigV930,
    apply_attribute_calibrator,
    attribute_weighted_prediction,
    fit_attribute_calibrator,
    inner_crossfit_fixed_convex,
    raw_attribute_scores,
    residual_correlation,
    success_gate,
)


def main():
    rng = np.random.default_rng(930)
    sample_count = 240
    text = rng.normal(0.0, 0.8, sample_count)
    audio_effect = rng.normal(0.0, 0.25, sample_count)
    vision_effect = rng.normal(0.0, 0.25, sample_count)
    lav = text + audio_effect + vision_effect
    mode_predictions = {
        "LAV": lav,
        "L": text,
        "LA": text + audio_effect,
        "LV": text + vision_effect,
    }
    raw = raw_attribute_scores(mode_predictions)
    assert tuple(raw) == EXPERT_NAMES
    assert raw["text_stable"].shape == (sample_count,)

    calibrator = fit_attribute_calibrator(raw, 0.10, 2.0)
    applicability = apply_attribute_calibrator(raw, calibrator)
    assert applicability.shape == (sample_count, len(EXPERT_NAMES))
    assert np.isfinite(applicability).all()
    assert float(applicability.min()) >= 0.10 - 1e-12
    assert float(applicability.max()) <= 1.0 + 1e-12

    labels = text + 0.8 * audio_effect + 0.6 * vision_effect
    anchor = lav + rng.normal(0.0, 0.22, sample_count)
    expert_predictions = np.column_stack(
        [
            text + rng.normal(0.0, 0.12, sample_count),
            text + audio_effect + rng.normal(0.0, 0.12, sample_count),
            text + vision_effect + rng.normal(0.0, 0.12, sample_count),
            labels + rng.normal(0.0, 0.10, sample_count),
        ]
    )
    actions = np.column_stack([anchor, expert_predictions])
    folds = np.repeat(np.arange(3), sample_count // 3)
    consensus = inner_crossfit_fixed_convex(actions, labels, folds, 0.01)
    assert consensus["prediction"].shape == (sample_count,)
    assert consensus["final_weights"].shape == (len(ACTION_NAMES_V930),)
    assert np.all(consensus["final_weights"] >= -1e-10)
    assert np.isclose(consensus["final_weights"].sum(), 1.0)

    attribute = attribute_weighted_prediction(actions, applicability, 1.0)
    assert attribute["prediction"].shape == (sample_count,)
    assert attribute["weights"].shape == actions.shape
    assert np.allclose(attribute["weights"].sum(axis=1), 1.0)

    correlation = residual_correlation(actions, labels)
    assert correlation.shape == (
        len(ACTION_NAMES_V930),
        len(ACTION_NAMES_V930),
    )
    assert np.isfinite(correlation).all()

    config = ObservableAttributeConfigV930(
        required_gain_vs_v921=0.005,
        required_nondegrading_folds=2,
        max_worst_fold_degradation=0.005,
    )
    gate = success_gate([0.01, 0.006, -0.002], 0.007, config)
    assert gate["passed"] is True
    failed = success_gate([0.01, -0.02, 0.001], 0.002, config)
    assert failed["passed"] is False

    print("V9.30 SMOKE TEST PASSED")
    print("action_names:", ACTION_NAMES_V930)
    print(
        "applicability_range:",
        float(applicability.min()),
        float(applicability.max()),
    )
    print("fixed_weights:", consensus["final_weights"].tolist())


if __name__ == "__main__":
    main()
