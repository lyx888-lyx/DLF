import numpy as np

from trains.singleTask.cross_seed_median_residual import (
    is_strong_group,
    median_absolute_deviation,
    minimum_group_count,
    sign_agreement,
)


def test_n_min_formula_is_frozen():
    assert minimum_group_count(100) == 8
    assert minimum_group_count(2000) == 10


def test_cross_seed_strong_group_requires_four_supported_same_sign():
    medians = [0.1, 0.2, 0.15, 0.12, -0.1]
    counts = [8, 8, 8, 8, 8]
    assert is_strong_group(medians, counts, 8, required=4)
    assert not is_strong_group(medians, [8, 8, 8, 7, 7], 8, required=4)


def test_minimum_absolute_residual_is_enforced():
    assert not is_strong_group(
        [0.001, 0.002, 0.003, 0.004, 0.005],
        [8, 8, 8, 8, 8],
        8,
    )


def test_robust_statistics():
    assert np.isclose(median_absolute_deviation([1, 2, 3]), 1)
    assert sign_agreement([1, 2, -1, 3, 4]) == (4, 1)
