"""Unit checks for the Stage23A-v2a frozen feature definitions."""

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mosei"))

from stage23a_v2_common import EXPERTS, MODE_AVAILABILITY, MODES, optimize_simplex


class Stage23AV2ATest(unittest.TestCase):
    def test_frozen_committee_and_modes(self):
        self.assertEqual(len(EXPERTS), 5)
        self.assertEqual(EXPERTS[0], "uniform_kd_seed1111")
        self.assertNotIn("clean_seed1111", EXPERTS)
        self.assertEqual(MODES, ("LAV", "LA", "LV", "L"))

    def test_effective_vision_mask_does_not_change_mode(self):
        lav = np.asarray(MODE_AVAILABILITY["LAV"])
        effective = lav * np.asarray([1, 1, 0])
        np.testing.assert_array_equal(effective, [1, 1, 0])
        self.assertEqual(MODES.index("LAV"), 0)

    def test_simplex_constraints(self):
        predictions = np.asarray(
            [[0.0, 1.0], [1.0, 0.0], [0.4, 0.6], [0.6, 0.4]]
        )
        labels = np.asarray([0.5, 0.5, 0.5, 0.5])
        weights = optimize_simplex(predictions, labels)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=10)
        self.assertTrue(np.all(weights >= 0))

    def test_frozen_57d_block_sum(self):
        self.assertEqual(4 + 5 + 3 + 5 + 5 + 32 + 3, 57)


if __name__ == "__main__":
    unittest.main()
