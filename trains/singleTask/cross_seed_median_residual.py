"""Frozen train-only grouping primitives for Stage 13 CS-DFMRC."""

import math

import numpy as np


def minimum_group_count(train_samples):
    return max(8, int(math.ceil(0.005 * int(train_samples))))


def residual_sign(value):
    value = float(value)
    return 1 if value > 0 else (-1 if value < 0 else 0)


def median_absolute_deviation(values):
    values = np.asarray(values, dtype=np.float64)
    center = np.median(values)
    return float(np.median(np.abs(values - center)))


def sign_agreement(values):
    signs = np.asarray([residual_sign(value) for value in values])
    positive = int(np.count_nonzero(signs > 0))
    negative = int(np.count_nonzero(signs < 0))
    return max(positive, negative), (1 if positive >= negative else -1)


def is_strong_group(seed_medians, seed_counts, n_min, required=4):
    supported = [
        float(median)
        for median, count in zip(seed_medians, seed_counts)
        if int(count) >= int(n_min) and np.isfinite(median)
    ]
    if len(supported) < required:
        return False
    agreement, _ = sign_agreement(supported)
    return (
        agreement >= required
        and abs(float(np.median(supported))) >= 0.01
    )
