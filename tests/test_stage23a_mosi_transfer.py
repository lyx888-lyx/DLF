import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mosei"
if str(SCRIPT) not in sys.path:
    sys.path.insert(0, str(SCRIPT))

from stage23a_mosi_common import (
    EXPERTS,
    N_FOLDS,
    SPLIT_SEED,
    inner_valid,
    judge_fold,
    overall_j,
    source_video,
)


def test_frozen_candidate_count_and_fold_count():
    assert len(EXPERTS) == 5
    assert N_FOLDS == 5
    assert SPLIT_SEED == 2301


def test_source_level_helpers_are_deterministic():
    video = source_video("video-A$_$17")
    assert video == "video-A"
    assert judge_fold(video) == judge_fold(video)
    assert inner_valid(video, 3) == inner_valid(video, 3)


def test_test_name_is_not_exposed_by_protocol_adapter():
    import stage23a_mosi_common as common

    assert not hasattr(common, "build_test_loader")
    assert not hasattr(common, "load_test")

