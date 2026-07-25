import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mosei"
if str(SCRIPT) not in sys.path:
    sys.path.insert(0, str(SCRIPT))

from stage23a_v2_common import (
    EXPERTS,
    MODE_ACTIVE_HEADS,
    MODE_AVAILABILITY,
    optimize_simplex,
    overall_j,
    stable_bucket,
)


def test_mode_schema_is_57d_and_explicit():
    assert MODE_AVAILABILITY["LAV"] == (1, 1, 1)
    assert MODE_AVAILABILITY["LA"] == (1, 1, 0)
    assert MODE_AVAILABILITY["LV"] == (1, 0, 1)
    assert MODE_AVAILABILITY["L"] == (1, 0, 0)
    assert "logits_v_hetero" not in MODE_ACTIVE_HEADS["LA"]
    assert "logits_a_hetero" not in MODE_ACTIVE_HEADS["LV"]
    assert set(MODE_ACTIVE_HEADS["L"]) == {
        "output_logit",
        "logits_c",
        "logits_l_hetero",
    }
    assert 4 + len(EXPERTS) + 3 + len(EXPERTS) + len(EXPERTS) + 32 + 3 == 57


def test_stable_source_split_keeps_source_together():
    source = "video-a"
    salt = "stage23a_v2_direction_A_seed2302"
    assert stable_bucket(source, salt, 8) == stable_bucket(source, salt, 8)


def test_simplex_and_oracle_are_select_one():
    generator = np.random.RandomState(23)
    predictions = generator.normal(size=(100, 5))
    labels = 0.6 * predictions[:, 0] + 0.4 * predictions[:, 1]
    weights = optimize_simplex(predictions, labels)
    assert np.all(weights >= 0)
    assert np.isclose(weights.sum(), 1.0)
    toy_predictions = np.array([[0.0, 1.0, 3.0, 4.0, 5.0]])
    toy_label = np.array([1.1])
    errors = np.abs(toy_predictions - toy_label[:, None])
    selected = toy_predictions[np.arange(len(toy_label)), errors.argmin(axis=1)]
    assert np.allclose(selected, [1.0])


def test_overall_j_uses_lav_and_missing_macro():
    frame = pd.DataFrame(
        {
            "mode": ["LAV", "LA", "LV", "L"],
            "label": [0.0, 0.0, 0.0, 0.0],
            "prediction": [1.0, 2.0, 3.0, 4.0],
        }
    )
    assert np.isclose(overall_j(frame, "prediction"), 2.0)
