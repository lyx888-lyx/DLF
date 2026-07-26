import sys
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "mosei"))

from stage23d_aggregate import gate_table, markdown_table
from stage23d_phase1 import core_pca_features, cross_source_shuffle
from stage23d_self_risk_common import EXPERTS, MODES, enable_dropout_only


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

    def test_core_pca_layout_excludes_middle_a2_blocks(self):
        raw = pd.DataFrame([range(1200)]).to_numpy(dtype="float32")
        core = core_pca_features(raw, "L")
        self.assertEqual(core.shape, (1, 550))
        self.assertEqual(core[0, 449], 449)
        self.assertEqual(core[0, 450], 1100)

    def test_shuffle_control_never_uses_same_source(self):
        sources = pd.Series(["a", "a", "b", "b", "c", "c"]).to_numpy()
        values = pd.DataFrame({"a": [0, 0, 1, 1, 2, 2]}).to_numpy()
        shuffled = cross_source_shuffle(values, sources, 23071).reshape(-1)
        original = values.reshape(-1)
        self.assertTrue((shuffled != original).all())

    def test_a5_enables_only_dropout_modules(self):
        import torch

        model = torch.nn.Sequential(
            torch.nn.Linear(3, 3),
            torch.nn.BatchNorm1d(3),
            torch.nn.Dropout(0.2),
        )
        before = [parameter.detach().clone() for parameter in model.parameters()]
        count = enable_dropout_only(model)
        self.assertEqual(count, 1)
        self.assertFalse(model[1].training)
        self.assertTrue(model[2].training)
        self.assertTrue(
            all(
                torch.equal(left, right)
                for left, right in zip(before, model.parameters())
            )
        )


if __name__ == "__main__":
    unittest.main()
