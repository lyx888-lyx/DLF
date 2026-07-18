import inspect

import numpy as np

from trains.singleTask.robust_cross_seed_consensus import (
    KAPPA,
    MAD_SCALE,
    MAX_ITERATIONS,
    SCALE_FLOOR,
    huber_center,
)


def test_huber_api_has_no_label_or_forbidden_input():
    parameters = inspect.signature(huber_center).parameters
    assert tuple(parameters) == ("values",)


def test_frozen_constants():
    assert KAPPA == 1.345
    assert MAD_SCALE == 1.4826
    assert SCALE_FLOOR == 1e-6
    assert MAX_ITERATIONS == 20


def test_permutation_invariance():
    values = np.array([-1.0, 0.0, 0.2, 0.3, 3.0])
    first = huber_center(values).center
    second = huber_center(values[[4, 2, 0, 3, 1]]).center
    assert first == second


def test_all_equal_uses_degenerate_mean_fallback():
    result = huber_center([0.4] * 5)
    assert result.center == 0.4
    assert result.degenerate_mad
    assert result.fallback_reason == "degenerate_mad_arithmetic_mean"


def test_one_outlier_has_less_effect_than_mean():
    values = np.array([0.0, 0.0, 0.1, 0.1, 3.0])
    center = huber_center(values).center
    assert abs(center - np.median(values)) < abs(
        np.mean(values) - np.median(values)
    )


def test_even_member_median_and_mad_are_midpoint_definitions():
    result = huber_center([0.0, 1.0, 2.0, 100.0])
    assert result.median == 1.5
    assert result.mad == 1.0


def test_center_is_clipped():
    assert huber_center([10.0] * 5).center == 3.0
    assert huber_center([-10.0] * 5).center == -3.0


def test_nan_is_rejected():
    try:
        huber_center([0.0, 1.0, np.nan, 2.0, 3.0])
    except FloatingPointError:
        pass
    else:
        raise AssertionError("NaN must be rejected")
