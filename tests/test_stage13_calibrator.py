import json

import pandas as pd

from trains.singleTask.cs_dfmrc_calibrator import (
    build_calibrator,
    calibrator_sha,
    correction_for_group,
)


def training_frame(sign=1.0):
    rows = []
    for seed in (1111, 1112, 1113, 1114):
        for mode in ("LAV", "LA"):
            for _ in range(8):
                rows.append(
                    {
                        "Seed": seed,
                        "Mode": mode,
                        "Cell": "c7=0|c5=0|c2=0",
                        "Half": 0,
                        "SafeResidual": sign * (0.1 + seed % 3 * 0.01),
                    }
                )
    return pd.DataFrame(rows)


def test_calibrator_is_train_only_and_serializable():
    calibrator = build_calibrator(
        training_frame(), (1111, 1112, 1113, 1114), 8
    )
    assert calibrator["TrainingSeeds"] == [1111, 1112, 1113, 1114]
    assert "valid_label" in calibrator["ForbiddenInputs"]
    json.dumps(calibrator)
    assert len(calibrator_sha(calibrator)) == 64


def test_shrinkage_formula_is_frozen():
    calibrator = build_calibrator(
        training_frame(), (1111, 1112, 1113, 1114), 8
    )
    correction, source, weight = correction_for_group(
        calibrator, "LAV", "c7=0|c5=0|c2=0", 0
    )
    assert source == "LocalShrinkage"
    assert weight == 4 / 6
    assert correction > 0


def test_zero_fallback_when_pool_is_unavailable():
    calibrator = build_calibrator(
        training_frame(), (1111, 1112, 1113, 1114), 100
    )
    correction, source, weight = correction_for_group(
        calibrator, "LAV", "c7=0|c5=0|c2=0", 0
    )
    assert (correction, source, weight) == (0.0, "Zero", 0.0)


def test_sign_guard_rejects_two_vs_two():
    frame = training_frame()
    frame.loc[frame.Seed.isin([1113, 1114]), "SafeResidual"] *= -1
    calibrator = build_calibrator(
        frame, (1111, 1112, 1113, 1114), 8
    )
    correction, source, _ = correction_for_group(
        calibrator, "LAV", "c7=0|c5=0|c2=0", 0
    )
    assert correction == 0.0
    assert source == "Zero"
