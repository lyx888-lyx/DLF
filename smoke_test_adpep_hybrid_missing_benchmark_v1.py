"""Synthetic checks for the offline ADPEP/Hybrid missing benchmark."""
import numpy as np
import pandas as pd

from benchmark_adpep_hybrid_missing_offline_v1 import (
    MISSING_MODES,
    MODES,
    _soft_region_membership,
    metric_bundle,
    paired_missing_vs_original,
)


def frame(values, labels):
    result = pd.DataFrame(
        {
            "sample_index": np.arange(len(labels)),
            "sample_id": ["s{}".format(i) for i in range(len(labels))],
            "label": labels,
        }
    )
    for mode in MODES:
        result["{}_pred".format(mode)] = values
    return result


def main():
    membership = _soft_region_membership(np.asarray([-2.2, 0.0, 2.2]), 0.55)
    assert membership.shape == (3, 5)
    np.testing.assert_allclose(membership.sum(axis=1), np.ones(3), atol=1e-12, rtol=0.0)
    assert int(np.argmax(membership[0])) == 0
    assert int(np.argmax(membership[1])) == 2
    assert int(np.argmax(membership[2])) == 4

    labels = np.asarray([-1.0, 0.5, 1.5], dtype=np.float64)
    original = frame(np.asarray([-0.5, 0.0, 1.0]), labels)
    candidate = frame(np.asarray([-0.8, 0.4, 1.4]), labels)
    paired = paired_missing_vs_original(candidate, original)
    assert paired["PairedMissingN"] == len(labels) * len(MISSING_MODES)
    assert paired["PairedMissingMeanGainVsOriginal"] > 0.0
    assert paired["PairedMissingWinRateVsOriginal"] == 1.0

    metrics = metric_bundle(candidate)
    assert np.isfinite(metrics["TestJ"])
    assert np.isfinite(metrics["MissingMacroMAE"])
    assert abs(metrics["MissingMacroMAE"] - metrics["LA_MAE"]) < 1e-12
    print("ADPEP/Hybrid offline benchmark synthetic smoke: PASS")


if __name__ == "__main__":
    main()
