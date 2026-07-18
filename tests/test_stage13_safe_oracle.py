import inspect

import numpy as np

from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
)
from trains.singleTask.decision_feasible_interval import (
    bounded_decision_interval,
    decision_cell,
    normalized_half,
    safe_oracle_predictions,
)


def test_decision_cell_reuses_official_evaluator():
    values = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float32)
    decisions = evaluator_decisions(values, "mosi")
    for index, value in enumerate(values):
        assert decision_cell(value) == (
            int(decisions[0][index]),
            int(decisions[1][index]),
            bool(decisions[2][index]),
        )


def test_safe_oracle_is_label_free_projection_api_at_inference_boundary():
    assert "label" not in inspect.signature(project_array).parameters
    baseline = np.array([-0.6, 0.0, 0.6], dtype=np.float32)
    labels = np.array([3.0, -3.0, -3.0], dtype=np.float32)
    oracle, _ = safe_oracle_predictions(baseline, labels)
    for left, right in zip(
        evaluator_decisions(baseline, "mosi"),
        evaluator_decisions(oracle, "mosi"),
    ):
        assert np.array_equal(left, right)


def test_bounded_interval_is_finite_and_contains_anchor():
    for value in (-4.0, -2.5, 0.0, 2.5, 4.0):
        interval = bounded_decision_interval(value)
        anchor = np.clip(np.float32(value), -3.0, 3.0)
        assert -3.0 <= interval.lower <= interval.upper <= 3.0
        assert interval.lower <= anchor <= interval.upper


def test_half_definition_is_frozen_to_two_bins():
    assert normalized_half(-0.49) == 0
    assert normalized_half(-0.01) == 1
