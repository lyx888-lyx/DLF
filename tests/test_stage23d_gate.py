import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mosei"))

from stage23d_aggregate import gate_table, markdown_table
from stage23d_self_risk_common import EXPERTS, MODES


class Stage23DGateTests(unittest.TestCase):
    def fixtures(self, r2_spearman=0.32):
        metrics = []
        model_values = {
            "R2": (r2_spearman, 0.70, 0.70),
            "R1": (0.20, 0.56, 0.56),
            "N2_length_mask_only": (0.10, 0.52, 0.52),
            "N5_label_bin_prior": (0.10, 0.52, 0.52),
            "N0_shuffle_seed23071": (0.05, 0.52, 0.52),
            "N0_shuffle_seed23072": (0.04, 0.51, 0.51),
            "N0_shuffle_seed23073": (0.03, 0.50, 0.50),
        }
        for fold in (0, 1):
            for expert in EXPERTS:
                for mode in MODES:
                    for model, values in model_values.items():
                        metrics.append(
                            {
                                "checkpoint_fold": fold,
                                "expert_id": expert,
                                "mode": mode,
                                "model": model,
                                "Error_Spearman": values[0],
                                "bad20_AUROC": values[1],
                                "confident_wrong_AUROC": values[2],
                            }
                        )
        coverage = []
        for fold in (0, 1):
            for expert in EXPERTS:
                for mode in MODES:
                    for value in (0.1, 0.2, 0.3, 0.5, 0.7, 1.0):
                        coverage.append(
                            {
                                "checkpoint_fold": fold,
                                "expert_id": expert,
                                "mode": mode,
                                "model": "R2",
                                "coverage": value,
                                "retained_actual_MAE": value,
                                "improvement_vs_random": 1.0 - value,
                            }
                        )
        return pd.DataFrame(metrics), pd.DataFrame(coverage)

    def test_all_frozen_conditions_can_pass(self):
        metrics, coverage = self.fixtures()
        table, _, details = gate_table(metrics, coverage)
        self.assertTrue(table["pass"].all())
        self.assertEqual(details["decision"], "PASS")

    def test_weak_is_not_promoted_to_pass(self):
        metrics, coverage = self.fixtures(r2_spearman=0.05)
        table, _, details = gate_table(metrics, coverage)
        self.assertFalse(table["pass"].all())
        self.assertEqual(details["decision"], "FAIL")

    def test_markdown_renderer_has_no_optional_dependency(self):
        rendered = markdown_table(pd.DataFrame({"a": [1], "b": [0.25]}))
        self.assertIn("| a | b |", rendered)
        self.assertIn("0.2500", rendered)


if __name__ == "__main__":
    unittest.main()
