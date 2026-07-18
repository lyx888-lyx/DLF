import inspect

import numpy as np

from trains.singleTask.ordered_prototype_geometry import (
    LEVELS,
    acc7_bins,
    compute_prototypes,
    cross_mode_retrieval,
    fit_sentiment_axis,
    ordinal_spearman,
    prototype_estimate,
    prototype_temperature,
)


def test_acc7_bins_reuse_clip_and_round_to_even():
    values = np.array([-4, -2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 4])
    expected = np.round(np.clip(values.astype(np.float32), -3, 3)).astype(int)
    assert np.array_equal(acc7_bins(values), expected)


def test_empty_bins_are_not_synthesized():
    representations = np.array([[0.0], [1.0]])
    prototypes, counts, variances = compute_prototypes(
        representations, np.array([-3.0, 3.0])
    )
    assert prototypes[0] is None
    assert counts[0] == 0
    assert variances[0] is None


def test_ordered_geometry_and_retrieval():
    labels = LEVELS.astype(float)
    representations = LEVELS.astype(float)[:, None]
    prototypes, _, _ = compute_prototypes(representations, labels)
    axis = fit_sentiment_axis(prototypes)
    assert ordinal_spearman(prototypes, axis) == 1.0
    retrieval = cross_mode_retrieval(prototypes, prototypes)
    assert retrieval["SameBinRetrievalCount"] == 7
    assert retrieval["OrderedRetrievalError"] == 0


def test_temperature_and_soft_estimate_are_train_only_interfaces():
    labels = LEVELS.astype(float)
    representations = LEVELS.astype(float)[:, None]
    prototypes, _, _ = compute_prototypes(representations, labels)
    temperature = prototype_temperature(prototypes)
    estimate = prototype_estimate(np.array([[2.9]]), prototypes, temperature)
    assert temperature > 0
    assert estimate.shape == (1,)
    assert "labels" not in inspect.signature(prototype_estimate).parameters
