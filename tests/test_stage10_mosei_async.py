import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.mosei.stage10_common import (
    SEEDS,
    STAGES,
    STATES,
    atomic_json,
    sha256,
    validate_stage_manifest,
)
from trains.singleTask.anchor_decision_projection import (
    project_array,
    select_anchor_seed,
)
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    require_locked_members,
)
from trains.singleTask.missing_utils import TRAINING_SPLITS, validation_objective


def test_stage10_frozen_seeds_and_state_machine():
    assert require_locked_members(SEEDS) == SEEDS
    assert STAGES == ("clean", "moddrop", "compatibility", "cfcompat")
    assert {"PREPARING", "RUNNING_WORKERS", "TEST_UNLOCK_CHECK", "COMPLETED", "FAILED"} <= set(STATES)


def test_training_splits_exclude_test_and_projection_has_no_label_argument():
    assert TRAINING_SPLITS == ("train", "valid")
    assert "label" not in inspect.signature(project_array).parameters


def test_official_mosei_config_points_to_manual_asset():
    config = json.loads((Path(__file__).parents[1] / "config/config.json").read_text())
    mosei = config["datasetCommonParams"]["mosei"]["aligned"]
    assert mosei["featurePath"] == "/data4t/lyx/datasets/MOSEI/Processed/aligned_50.pkl"
    assert mosei["train_samples"] == 16326
    assert mosei["feature_dims"] == [768, 74, 35]


def test_anchor_uses_validation_j_and_lower_seed_tie_break():
    rows = [{"Seed": seed, "J": 1.0} for seed in SEEDS]
    rows[-1]["J"] = 0.5
    rows[-2]["J"] = 0.5
    assert select_anchor_seed(rows, SEEDS) == 1114


def test_adpep_all_preserves_mosei_decisions():
    anchor = np.array([-2.5, -0.5, 0.0, 0.5, 2.5], dtype=np.float32)
    pe5 = np.array([3.0, 2.0, 2.0, -2.0, -3.0], dtype=np.float32)
    projected, _ = project_array(anchor, pe5, "mosei", "adpep_all")
    from trains.singleTask.anchor_decision_projection import evaluator_decisions
    for left, right in zip(
        evaluator_decisions(anchor, "mosei"),
        evaluator_decisions(projected, "mosei"),
    ):
        assert np.array_equal(left, right)


def test_validation_j_formula():
    metrics = {
        "LAV": {"MAE": 1.0},
        "LA": {"MAE": 2.0},
        "LV": {"MAE": 3.0},
        "L": {"MAE": 4.0},
    }
    assert validation_objective(metrics) == 2.0


def test_manifest_rejects_sha_mismatch(tmp_path):
    output = tmp_path / "asset.bin"
    output.write_bytes(b"good")
    manifest = tmp_path / "stage_manifest.json"
    atomic_json(
        manifest,
        {
            "Seed": 1111,
            "Stage": "clean",
            "Commit": "abc",
            "Dataset": "mosei",
            "ValidationSelected": True,
            "NoTestAccess": True,
            "Outputs": [{"Path": str(output), "SHA256": sha256(output)}],
        },
    )
    validate_stage_manifest(manifest, 1111, "clean", "abc")
    output.write_bytes(b"changed")
    with pytest.raises(RuntimeError):
        validate_stage_manifest(manifest, 1111, "clean", "abc")
