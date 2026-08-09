"""Synthetic checks for the aggregate-only exploratory Test viability helpers.

No MOSI split, checkpoint, or model is loaded here.
"""
import pandas as pd

from trains.singleTask.cfcompat_exploratory_test_utils import (
    METHOD_ORDER,
    add_reference_deltas,
    build_missing_events,
    comparison_row,
    transfer_summary,
    v13_route_decision,
)


def prediction_frame(values):
    rows = []
    for index, (label, la, lv, l) in enumerate(values):
        rows.append(
            {
                "sample_index": index,
                "label": float(label),
                "LAV_pred": float(label),
                "LA_pred": float(la),
                "LV_pred": float(lv),
                "L_pred": float(l),
            }
        )
    return pd.DataFrame(rows)


def reference_frame():
    return pd.DataFrame(
        [
            {
                "sample_index": 0,
                "label": 1.0,
                "teacher_prediction": 1.0,
                "baseline_LAV_pred": 1.0,
                "baseline_LA_pred": 0.5,
                "baseline_LV_pred": 0.5,
                "baseline_L_pred": 0.5,
            },
            {
                "sample_index": 1,
                "label": -1.0,
                "teacher_prediction": 0.0,
                "baseline_LAV_pred": -1.0,
                "baseline_LA_pred": -0.5,
                "baseline_LV_pred": -0.5,
                "baseline_L_pred": -0.5,
            },
        ]
    )


def fake_metrics(j):
    return {
        "TestJ": float(j),
        "MissingMacroMAE": float(j),
        "LAV_MAE": float(j),
        "LA_MAE": float(j),
        "LV_MAE": float(j),
        "L_MAE": float(j),
    }


def main():
    reference = reference_frame()
    original_pred = prediction_frame(
        [
            (1.0, 0.4, 0.4, 0.4),
            (-1.0, -0.4, -0.4, -0.4),
        ]
    )
    safer_pred = prediction_frame(
        [
            (1.0, 0.8, 0.8, 0.8),
            (-1.0, -0.55, -0.55, -0.55),
        ]
    )
    original_events = build_missing_events(original_pred, reference, "original_cfcompat_v1")
    safer_events = build_missing_events(safer_pred, reference, "adam_step_safety_v13")
    original_transfer = transfer_summary(original_events)
    safer_transfer = transfer_summary(safer_events)

    # Sample 0 is Teacher-beneficial and must improve strongly under safer_pred.
    assert original_transfer["teacher_beneficial"]["negative_transfer_rate"] > safer_transfer["teacher_beneficial"]["negative_transfer_rate"]

    rows = []
    for index, method in enumerate(METHOD_ORDER):
        if method == "original_cfcompat_v1":
            transfer = original_transfer
            j = 0.70
        elif method == "adam_step_safety_v13":
            transfer = safer_transfer
            j = 0.69
        else:
            transfer = original_transfer
            j = 0.71 + index * 0.001
        rows.append(comparison_row(method, fake_metrics(j), transfer))
    comparison = add_reference_deltas(pd.DataFrame(rows))
    assert list(comparison.Method) == list(METHOD_ORDER)
    assert float(
        comparison.loc[
            comparison.Method.eq("adam_step_safety_v13"), "DeltaJVsOriginal"
        ].iloc[0]
    ) < 0.0

    route = v13_route_decision(comparison)
    assert route["checks"]["test_J_improves_original"] is True
    assert route["checks"]["beneficial_NTR_improves_original"] is True
    print("CFCompat exploratory Test aggregate smoke: PASS")
    print("route:", route["verdict"])


if __name__ == "__main__":
    main()
