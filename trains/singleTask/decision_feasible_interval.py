"""Stage 13 helpers backed by the frozen Stage 9B decision implementation."""

from dataclasses import dataclass

import numpy as np

from trains.singleTask.anchor_decision_projection import (
    evaluator_decisions,
    project_array,
    safe_interval,
)


OUTPUT_MIN = np.float32(-3.0)
OUTPUT_MAX = np.float32(3.0)


@dataclass(frozen=True)
class BoundedDecisionInterval:
    lower: np.float32
    upper: np.float32
    lower_closed: bool
    upper_closed: bool

    @property
    def width(self):
        return float(self.upper - self.lower)


def decision_cell(prediction, dataset="mosi"):
    """Return the official Acc7/Acc5/Acc2 signature."""
    decisions = evaluator_decisions([prediction], dataset)
    return int(decisions[0][0]), int(decisions[1][0]), bool(decisions[2][0])


def bounded_decision_interval(prediction, dataset="mosi"):
    """Intersect the maximal Stage 9B interval with the MOSI label domain."""
    interval = safe_interval(prediction, dataset, "adpep_all")
    lower = np.maximum(interval.lower, OUTPUT_MIN).astype(np.float32)
    upper = np.minimum(interval.upper, OUTPUT_MAX).astype(np.float32)
    if lower > upper:
        raise RuntimeError("Bounded decision interval is empty.")
    lower_closed = (
        interval.lower_closed if interval.lower >= OUTPUT_MIN else True
    )
    upper_closed = (
        interval.upper_closed if interval.upper <= OUTPUT_MAX else True
    )
    return BoundedDecisionInterval(
        lower=lower,
        upper=upper,
        lower_closed=bool(lower_closed),
        upper_closed=bool(upper_closed),
    )


def safe_oracle_predictions(baseline, labels, dataset="mosi"):
    """Project labels into the anchor decision cell using formal Stage 9B code."""
    return project_array(baseline, labels, dataset, "adpep_all")


def normalized_half(prediction, dataset="mosi"):
    """Frozen two-bin location within the bounded official decision cell."""
    interval = bounded_decision_interval(prediction, dataset)
    denominator = float(interval.upper - interval.lower)
    if denominator <= 0:
        raise RuntimeError("Decision cell has non-positive width.")
    u = (float(np.float32(prediction)) - float(interval.lower)) / denominator
    return 0 if u < 0.5 else 1
