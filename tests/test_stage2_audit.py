import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import audit_stage2
from trains.singleTask.missing_utils import build_single_split_loader
from trains.singleTask.stage2_audit_utils import (
    AUDIT_MODES,
    assert_frozen_eval_model,
    compare_metrics_to_reference,
    contribution_statistics,
    counterfactual_contributions,
    describe_abs_differences,
    error_change_statistics,
    label_distribution,
    metrics_from_predictions,
    total_variation_distance,
    validate_counterfactual_identities,
)


class Stage2AuditTests(unittest.TestCase):
    def test_counterfactual_identities_and_missing_identities(self):
        predictions = {
            "L": np.array([1.0, -1.0]),
            "LA": np.array([3.0, 0.0]),
            "LV": np.array([4.0, 2.0]),
            "LAV": np.array([8.0, 4.0]),
        }
        contributions = counterfactual_contributions(predictions)
        np.testing.assert_allclose(contributions["r_A"], [2.0, 1.0])
        np.testing.assert_allclose(contributions["r_V"], [3.0, 3.0])
        np.testing.assert_allclose(contributions["r_AV"], [2.0, 1.0])
        maxima = validate_counterfactual_identities(contributions)
        self.assertTrue(all(value <= 1e-6 for value in maxima.values()))

    def test_counterfactual_identity_violation_raises(self):
        contributions = counterfactual_contributions(
            {"L": [0.0], "LA": [1.0], "LV": [2.0], "LAV": [4.0]}
        )
        contributions["identity_error"][0] = 1e-4
        with self.assertRaises(RuntimeError):
            validate_counterfactual_identities(contributions)

    def test_mode_difference_statistics_and_zero_sign_rule(self):
        stats = describe_abs_differences(
            np.array([0.0, 1.0, -1.0, 0.2]),
            np.array([0.0, -1.0, -1.0, 0.3]),
        )
        self.assertAlmostEqual(stats["mean"], (0.0 + 2.0 + 0.0 + 0.1) / 4.0)
        self.assertAlmostEqual(stats["median"], 0.05)
        self.assertAlmostEqual(stats["p90"], np.percentile([0.0, 2.0, 0.0, 0.1], 90))
        self.assertAlmostEqual(stats["sign_flip"], 0.25)

    def test_error_change_categories_are_fixed(self):
        stats = error_change_statistics(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.2, 3.0, 0.0]),
            np.array([1.0, 1.0, 0.0]),
        )
        self.assertEqual(stats["improved_count"], 1)
        self.assertEqual(stats["worsened_count"], 1)
        self.assertEqual(stats["unchanged_count"], 1)
        self.assertAlmostEqual(stats["mean_improvement_among_improved"], 0.2)
        self.assertAlmostEqual(stats["mean_degradation_among_worsened"], 1.0)

    def test_label_bin_boundaries(self):
        distribution = label_distribution(np.array([-3.0, -1.0, 0.0, 1.0, 3.0]))
        self.assertEqual([item["count"] for item in distribution["bins"]], [1, 1, 1, 1, 1])
        self.assertEqual(distribution["negative_fraction"], 0.4)
        self.assertEqual(distribution["zero_fraction"], 0.2)
        self.assertEqual(distribution["positive_fraction"], 0.4)

    def test_total_variation_distance(self):
        self.assertAlmostEqual(total_variation_distance([0.5, 0.5], [1.0, 0.0]), 0.5)

    def test_reference_comparison_matches_and_rejects_difference(self):
        metrics = {
            mode: {
                "acc_7": 0.1,
                "acc_5": 0.2,
                "acc_2": 0.3,
                "F1_score": 0.4,
                "Corr": 0.5,
                "MAE": 0.6,
                "Loss": 0.7,
            }
            for mode in AUDIT_MODES
        }
        with tempfile.TemporaryDirectory() as directory:
            row = {"Seed": 1111}
            for mode in AUDIT_MODES:
                for key, value in metrics[mode].items():
                    row["{}_{}".format(mode, key)] = value
            path = Path(directory) / "reference.csv"
            pd.DataFrame([row]).to_csv(path, index=False)
            self.assertTrue(compare_metrics_to_reference(metrics, path, 1111)["matched"])
            metrics["LAV"]["MAE"] = 0.7
            with self.assertRaises(RuntimeError):
                compare_metrics_to_reference(metrics, path, 1111)

    def test_marker_guard_refuses_completed_or_partial_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "test"
            marker = output / "AUDIT_COMPLETED.json"
            output.mkdir()
            with self.assertRaises(RuntimeError):
                audit_stage2.assert_test_not_previously_completed(marker, output)
            marker.write_text('{"success": true}', encoding="utf-8")
            with self.assertRaises(RuntimeError):
                audit_stage2.assert_test_not_previously_completed(marker, output)

    def test_test_cli_requires_explicit_confirmation(self):
        with mock.patch.object(sys, "argv", ["audit_stage2.py", "--split", "test"]):
            with self.assertRaises(SystemExit):
                audit_stage2.parse_args()

    def test_valid_parse_does_not_create_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "AUDIT_COMPLETED.json"
            with mock.patch.object(sys, "argv", ["audit_stage2.py", "--split", "valid"]):
                parsed = audit_stage2.parse_args()
            self.assertEqual(parsed.split, "valid")
            self.assertFalse(marker.exists())

    def test_frozen_eval_model_and_no_grad(self):
        model = nn.Linear(2, 1).eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        with torch.inference_mode():
            model(torch.ones(1, 2))
        assert_frozen_eval_model(model, "tiny")

    def test_checkpoint_paths_are_distinct(self):
        paths = audit_stage2.checkpoint_paths(
            SimpleNamespace(model_save_dir="pt", seed=1111),
            SimpleNamespace(dataset_name="mosi"),
        )
        self.assertEqual(len(set(paths.values())), 3)

    def test_stable_sample_ids_and_indices(self):
        batch = {"index": torch.tensor([0, 1]), "id": ["a", "b"]}
        identifiers, indices = audit_stage2._batch_sample_ids(batch, 0)
        self.assertEqual(identifiers, ["a", "b"])
        self.assertEqual(indices, [0, 1])
        with self.assertRaises(RuntimeError):
            audit_stage2._batch_sample_ids({"index": torch.tensor([1]), "id": ["x"]}, 0)

    def test_metric_sample_count_alignment_and_raw_scale(self):
        predictions = np.array([0.0, 1.0])
        labels = np.array([0.0, 2.0])
        metrics = metrics_from_predictions(predictions, labels, [0.5])
        self.assertAlmostEqual(metrics["MAE"], 0.5)
        self.assertAlmostEqual(metrics["Loss"], 0.5)

    def test_audit_source_is_inference_only_and_loader_is_not_shuffled(self):
        source = Path("audit_stage2.py").read_text(encoding="utf-8")
        self.assertIn("torch.inference_mode()", source)
        self.assertNotIn("optimizer", source.lower())
        self.assertNotIn(".backward(", source)
        self.assertIn("shuffle=False", inspect.getsource(build_single_split_loader))

    def test_contribution_statistics_cover_signs(self):
        statistics = contribution_statistics(np.array([-1.0, 0.0, 2.0]))
        self.assertAlmostEqual(statistics["positive_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(statistics["negative_fraction"], 1.0 / 3.0)
        self.assertAlmostEqual(statistics["near_zero_fraction_0.01"], 1.0 / 3.0)


if __name__ == "__main__":
    unittest.main()
