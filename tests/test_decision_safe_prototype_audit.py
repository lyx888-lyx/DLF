import inspect

import numpy as np

from analysis_stage12a_decision_safe_prototype import (
    raw_prototype_correction,
    retention,
)
from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
)


def test_raw_step_is_fixed_and_not_an_api_parameter():
    assert tuple(inspect.signature(raw_prototype_correction).parameters) == (
        "baseline",
        "prototype_estimate_values",
    )
    baseline = np.array([0.0], dtype=np.float32)
    estimate = np.array([1.0], dtype=np.float32)
    assert raw_prototype_correction(baseline, estimate)[0] == 0.1


def test_projection_api_has_no_label_and_labels_cannot_change_output():
    assert "label" not in inspect.signature(project_array).parameters
    baseline = np.array([-0.5, 0.0, 0.5], dtype=np.float32)
    raw = np.array([2.0, 2.0, -2.0], dtype=np.float32)
    first, _ = project_array(baseline, raw, "mosi", "adpep_all")
    labels_a = np.array([-3.0, 0.0, 3.0])
    labels_b = labels_a[::-1]
    second, _ = project_array(baseline, raw, "mosi", "adpep_all")
    assert not np.array_equal(labels_a, labels_b)
    assert np.array_equal(first, second)


def test_safe_projection_preserves_all_decisions_at_boundaries():
    baseline = np.array(
        [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], dtype=np.float32
    )
    raw = np.array([3, 3, 3, 3, -3, -3, -3], dtype=np.float32)
    safe, _ = project_array(baseline, raw, "mosi", "adpep_all")
    for left, right in zip(
        evaluator_decisions(baseline, "mosi"),
        evaluator_decisions(safe, "mosi"),
    ):
        assert np.array_equal(left, right)


def test_retention_is_na_when_raw_gain_is_not_positive():
    raw_gain, safe_gain, ratio = retention(1.0, 1.1, 0.9)
    assert raw_gain < 0
    assert safe_gain > 0
    assert ratio is None


def test_retention_formula():
    raw_gain, safe_gain, ratio = retention(1.0, 0.8, 0.9)
    assert np.isclose(raw_gain, 0.2)
    assert np.isclose(safe_gain, 0.1)
    assert np.isclose(ratio, 0.5)
