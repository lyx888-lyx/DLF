import numpy as np
import pandas as pd

from analyze_sentiment_region_missing_robustness import (
    INTENSITY_ORDER,
    compose_fixed_blend,
    intensity_gain_table,
    intensity_region,
    overall_mode_metrics,
    sample_error_gain,
)


def _frame(offset=0.0):
    labels = np.array([-3.0, -2.0, -1.0, -0.25, 0.25, 1.0, 2.0, 3.0])
    frame = pd.DataFrame(
        {
            "sample_index": np.arange(len(labels)),
            "sample_id": [f"s{i}" for i in range(len(labels))],
            "label": labels,
        }
    )
    for mode, extra in (("LAV", 0.0), ("LA", 0.1), ("LV", 0.15), ("L", 0.3)):
        frame[f"{mode}_pred"] = labels + offset + extra
    return frame


def test_intensity_regions_are_frozen():
    values = [-3.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.0]
    regions = [intensity_region(value) for value in values]
    assert regions == [
        "extreme",
        "moderate",
        "mild",
        "near_neutral",
        "near_neutral",
        "near_neutral",
        "mild",
        "moderate",
        "extreme",
    ]
    assert INTENSITY_ORDER == ("near_neutral", "mild", "moderate", "extreme")


def test_fixed_blend_is_prediction_level_composition():
    left = _frame(offset=0.0)
    right = _frame(offset=1.0)
    result = compose_fixed_blend(left, right, 0.25)
    expected = 0.75 * left.LAV_pred.to_numpy() + 0.25 * right.LAV_pred.to_numpy()
    assert np.allclose(result.LAV_pred.to_numpy(), expected)
    assert np.array_equal(result.sample_index.to_numpy(), left.sample_index.to_numpy())


def test_error_gain_positive_when_ours_is_closer():
    baseline = _frame(offset=0.5)
    ours = _frame(offset=0.1)
    modes = ("LAV", "LA", "LV", "L")
    sample = sample_error_gain(baseline, ours, modes)
    assert (sample.ErrorGain > 0).all()
    gain = intensity_gain_table(sample)
    assert (gain.MAEGain > 0).all()
    assert (gain.WinRate == 1.0).all()


def test_missing_degradation_is_measured_relative_to_each_method_lav():
    baseline = _frame(offset=0.0)
    ours = _frame(offset=0.0)
    overall = overall_mode_metrics(baseline, ours, ("LAV", "LA", "LV", "L"), "DLF", "Ours")
    dlf = overall.loc[overall.Method.eq("DLF")].set_index("Mode")
    assert np.isclose(dlf.loc["LAV", "MAE_degradation_vs_LAV"], 0.0)
    assert dlf.loc["L", "MAE_degradation_vs_LAV"] > dlf.loc["LA", "MAE_degradation_vs_LAV"]
