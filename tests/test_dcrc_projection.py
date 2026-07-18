import inspect

import numpy as np

from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
)


def test_projection_api_has_no_label():
    assert "label" not in inspect.signature(project_array).parameters


def test_dcrc_projection_inherits_all_anchor_decisions():
    anchor = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float32)
    target = np.array([3.0, 3.0, -3.0, -3.0, -3.0], dtype=np.float32)
    projected, _ = project_array(anchor, target, "mosi", "adpep_all")
    for left, right in zip(
        evaluator_decisions(anchor, "mosi"),
        evaluator_decisions(projected, "mosi"),
    ):
        assert np.array_equal(left, right)
