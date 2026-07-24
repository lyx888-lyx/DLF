import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "mosei"
if str(SCRIPT) not in sys.path:
    sys.path.insert(0, str(SCRIPT))

from stage23a_common import (
    judge_fold,
    outer_fold,
    project_simplex_rows,
    source_video,
)


def test_source_video_and_splits_are_deterministic():
    sample = "-3g5yACwYnA$_$10"
    video = source_video(sample)
    assert video == "-3g5yACwYnA"
    assert outer_fold(video) in (0, 1)
    assert outer_fold(video) == outer_fold(video)
    assert judge_fold(video) == judge_fold(video)


def test_simplex_projection_rows():
    values = np.array([[2.0, -1.0, 0.5], [-0.2, -0.1, -0.3]])
    projected = project_simplex_rows(values)
    assert np.all(projected >= 0)
    assert np.allclose(projected.sum(axis=1), 1.0)
    assert np.allclose(projected[0], [1.0, 0.0, 0.0])


def test_source_split_is_source_level():
    samples = ["video-a$_$0", "video-a$_$1", "video-b$_$0"]
    folds = [outer_fold(source_video(value)) for value in samples]
    assert folds[0] == folds[1]

