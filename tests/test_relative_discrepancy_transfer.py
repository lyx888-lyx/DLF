"""Synthetic regression checks; no GPU, checkpoints, real datasets, or Test access."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_relative_discrepancy_transfer import (
    forbid_test_path, midpoint_compatibility, long_table,
    nearest_same_sample_pairs, intensity_summary, gap_bin_summary,
    correlation_summary, validate_alignment
)


class RelativeTransferAuditTests(unittest.TestCase):
    def setup_frames(self):
        idx = [0, 1, 2, 3]
        ids = ["s0", "s1", "s2", "s3"]
        y = np.array([-3., 2.5, -1.5, 1.7])
        ref = pd.DataFrame({
            "sample_index": idx, "sample_id": ids, "label": y,
            "LAV_pred": [-1., 2.4, -1.5, 1.5],
            "LA_pred": [-1.3, 2.0, -1.2, 1.1],
            "LV_pred": [-.7, 2.2, -1.0, 1.4],
            "L_pred": [-1.0, 1.6, -.9, 1.0],
        })
        uni = ref.copy()
        cf = ref.copy()
        uni["LA_pred"] = [-2.0, 2.0, -1.0, 1.3]
        uni["LV_pred"] = [-1.5, 2.1, -1.1, 1.5]
        uni["L_pred"] = [-1.2, 1.4, -.7, .9]
        cf["LA_pred"] = [-2.6, 2.2, -1.4, 1.6]
        cf["LV_pred"] = [-2.7, 2.3, -1.4, 1.6]
        cf["L_pred"] = [-2.1, 1.8, -1.2, 1.2]
        cache = pd.DataFrame({
            "sample_index": list(range(5)),
            "delta_LA": [.05,.10,.15,.20,.30],
            "delta_LV": [.30,.40,.55,.65,.80],
            "delta_L": [.08,.20,.35,.50,.90],
        })
        teacher = pd.DataFrame({
            "sample_index":idx,"sample_id":ids,"label":y,
            "teacher_LAV_pred":[-2.9, 2.4, -1.4, 1.7]
        })
        return ref, uni, cf, cache, teacher

    def test_equal_raw_gaps_differ_between_modes(self):
        a = midpoint_compatibility([.05,.10,.15,.20,.30], [.30])[0]
        b = midpoint_compatibility([.30,.40,.55,.65,.80], [.30])[0]
        self.assertAlmostEqual(a, .1)
        self.assertAlmostEqual(b, .9)

    def test_modewise_not_extra_information_within_mode(self):
        ref, uni, ours, cache, teacher = self.setup_frames()
        long = long_table(ref,uni,ours,cache,teacher)
        la = long.loc[long["mode"] == "LA"].sort_values("delta")
        self.assertTrue((np.diff(la.compat_mode) <= 1e-10).all())
        self.assertIn("teacher_student_gap", long)
        self.assertEqual(len(long), 12)

    def test_same_sample_matched_case(self):
        ref,uni,ours,cache,teacher=self.setup_frames()
        long=long_table(ref,uni,ours,cache,teacher)
        pairs=nearest_same_sample_pairs(long, tolerance=1e-8, min_rank_difference=.4)
        relevant=pairs.loc[(pairs.sample_index==0) &
                           (pairs.high_compat_mode=="LV") &
                           (pairs.low_compat_mode=="LA")]
        self.assertEqual(len(relevant),1)
        # Subtracted floats can fall just above the train 0.30 boundary.
        # The qualitative claim is a large rank separation, not exact 0.80.
        self.assertGreater(float(relevant.iloc[0].compat_separation), .6)

    def test_intensity_and_summary(self):
        ref,uni,ours,cache,teacher=self.setup_frames()
        long=long_table(ref,uni,ours,cache,teacher)
        intensity=intensity_summary(long,bootstrap_reps=8,seed=6)
        self.assertTrue(np.isfinite(intensity.MAE_reduction).all())
        self.assertTrue((intensity.n_view_pairs>0).all())
        bins=gap_bin_summary(long,cache)
        self.assertGreater(len(bins),0)
        corrs=correlation_summary(long,bootstrap_reps=8,seed=5)
        self.assertEqual(len(corrs),16)

    def test_reject_test_input(self):
        forbid_test_path("result/analysis/valid_predictions.csv")
        with self.assertRaises(ValueError):
            forbid_test_path("result/mosi_best_test_predictions.csv")
        with self.assertRaises(ValueError):
            forbid_test_path("/data/test/predictions.csv")

    def test_alignment_rejects_wrong_labels(self):
        ref,uni,*_=self.setup_frames()
        uni.loc[0,"label"]=1.0
        with self.assertRaises(ValueError):
            validate_alignment(ref,uni,"uniform")


if __name__ == "__main__":
    unittest.main()
