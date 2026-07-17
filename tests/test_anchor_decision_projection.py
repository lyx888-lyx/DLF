import inspect
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score

import eval_anchor_decision_preserving_ensemble as evaluator
from trains.singleTask.anchor_decision_projection import (
    DecisionInterval,
    decision_signature,
    evaluator_decisions,
    project_array,
    project_prediction,
    retention_ratio,
    safe_interval,
    select_anchor_seed,
)
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    metrics_from_predictions,
)
from trains.singleTask.missing_utils import regression_metrics


class AnchorDecisionProjectionTests(unittest.TestCase):
    def test_anchor_selection_uses_validation_j_and_lower_seed_tie(self):
        rows = [
            {"Seed": 9, "J": 0.7},
            {"Seed": 3, "J": 0.6},
            {"Seed": 7, "J": 0.6},
            {"Seed": 5, "J": 0.8},
            {"Seed": 1, "J": 0.9},
        ]
        self.assertEqual(select_anchor_seed(rows, (1, 3, 5, 7, 9)), 3)

    def test_anchor_selection_is_not_hard_coded_to_1114(self):
        rows = [
            {"Seed": seed, "J": value}
            for seed, value in zip((1111, 1112, 1113, 1114, 1115), (0.5, 0.4, 0.3, 0.2, 0.1))
        ]
        self.assertEqual(select_anchor_seed(rows, evaluator.SEEDS), 1115)
        self.assertNotIn("1114", inspect.getsource(select_anchor_seed))

    def test_mosi_reference_replay_selects_1114(self):
        rows = [
            {"Seed": 1111, "J": 0.6779637237389882},
            {"Seed": 1112, "J": 0.7278208533922832},
            {"Seed": 1113, "J": 0.7005979816118877},
            {"Seed": 1114, "J": 0.6667473018169403},
            {"Seed": 1115, "J": 0.6784075995286305},
        ]
        self.assertEqual(select_anchor_seed(rows, evaluator.SEEDS), 1114)
        self.assertLess(abs(rows[3]["J"] - 0.666747), 1e-4)

    def test_projection_api_has_no_label_argument(self):
        for function in (project_prediction, project_array, safe_interval):
            self.assertNotIn("label", inspect.signature(function).parameters)

    def test_exact_evaluator_boundaries_and_round_half_to_even(self):
        boundaries = np.asarray(
            [-2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5], dtype=np.float32
        )
        values = np.concatenate(
            [
                np.nextafter(boundaries, np.float32(-np.inf), dtype=np.float32),
                boundaries,
                np.nextafter(boundaries, np.float32(np.inf), dtype=np.float32),
            ]
        )
        acc7, acc5, acc2 = evaluator_decisions(values, "mosi")
        np.testing.assert_array_equal(
            acc7, np.round(np.clip(values, np.float32(-3), np.float32(3)))
        )
        np.testing.assert_array_equal(
            acc5, np.round(np.clip(values, np.float32(-2), np.float32(2)))
        )
        np.testing.assert_array_equal(acc2, values > np.float32(0))
        self.assertEqual(decision_signature(-2.5, "mosi", "adpep_all"), (-2, -2, False))
        self.assertEqual(decision_signature(-1.5, "mosi", "adpep_all"), (-2, -2, False))
        self.assertEqual(decision_signature(-0.5, "mosi", "adpep_all"), (0, 0, False))
        self.assertEqual(decision_signature(0.0, "mosi", "adpep_all"), (0, 0, False))
        self.assertEqual(decision_signature(0.5, "mosi", "adpep_all"), (0, 0, True))
        self.assertEqual(decision_signature(1.5, "mosi", "adpep_all"), (2, 2, True))
        self.assertEqual(decision_signature(2.5, "mosi", "adpep_all"), (2, 2, True))

    def test_decisions_match_frozen_stage9a_metric_evaluator(self):
        prediction = np.asarray(
            [-3.7, -2.5, -1.5001, -1.5, -0.5, 0, 0.5, 1.5, 2.5, 3.7],
            dtype=np.float32,
        )
        target = np.asarray(
            [-3, -2, -1, -2, -0.4, 0.2, 0.6, 2, 3, 3], dtype=np.float32
        )
        values = regression_metrics(
            torch.tensor(prediction), torch.tensor(target)
        )
        pred7, pred5, pred2 = evaluator_decisions(prediction, "mosei")
        true7, true5, true2 = evaluator_decisions(target, "mosei")
        nonzero = target != 0
        self.assertAlmostEqual(values["acc_7"], np.mean(pred7 == true7))
        self.assertAlmostEqual(values["acc_5"], np.mean(pred5 == true5))
        self.assertAlmostEqual(values["acc_2"], np.mean(pred2[nonzero] == true2[nonzero]))
        self.assertAlmostEqual(
            values["F1_score"],
            f1_score(true2[nonzero], pred2[nonzero], average="weighted", zero_division=0),
        )

    def test_zero_and_has_zero_protocol_are_explicit(self):
        _, _, binary = evaluator_decisions([-0.0, 0.0, np.float32(1e-45)], "mosi")
        np.testing.assert_array_equal(binary, [False, False, True])
        # Stage 9A excludes target==0 from Acc2/F1, but the projection preserves
        # prediction > 0 for every sample without seeing which labels are zero.

    def test_open_boundary_uses_nextafter(self):
        result = project_prediction(1.2, 2.0, "mosi", "adpep_all")
        expected = np.nextafter(
            np.float32(1.5), np.float32(-np.inf), dtype=np.float32
        )
        self.assertEqual(result.value, expected)
        self.assertTrue(result.boundary_adjusted)
        self.assertEqual(
            decision_signature(result.value, "mosi", "adpep_all"),
            decision_signature(1.2, "mosi", "adpep_all"),
        )

    def test_closed_boundaries_positive_and_negative(self):
        self.assertEqual(
            project_prediction(-2.0, -3.0, "mosi", "adpep_all").value,
            np.float32(-2.5),
        )
        self.assertEqual(
            project_prediction(2.0, 3.0, "mosi", "adpep_all").value,
            np.float32(2.5),
        )
        self.assertEqual(
            project_prediction(-0.5, 0.2, "mosi", "adpep_all").value,
            np.float32(0),
        )

    def test_pe5_inside_left_and_right_projection(self):
        inside = project_prediction(0.2, 0.4, "mosi", "adpep_all")
        left = project_prediction(0.2, -2.0, "mosi", "adpep_all")
        right = project_prediction(0.2, 2.0, "mosi", "adpep_all")
        self.assertTrue(inside.pe5_already_feasible)
        self.assertEqual(inside.value, np.float32(0.4))
        self.assertEqual(
            left.value,
            np.nextafter(np.float32(0), np.float32(np.inf), dtype=np.float32),
        )
        self.assertEqual(right.value, np.float32(0.5))

    def test_safe_interval_contains_anchor_for_all_boundary_neighbors(self):
        boundaries = np.asarray(
            [-2.5, -1.5, -0.5, 0, 0.5, 1.5, 2.5], dtype=np.float32
        )
        anchors = []
        for boundary in boundaries:
            anchors.extend(
                [
                    np.nextafter(boundary, np.float32(-np.inf), dtype=np.float32),
                    boundary,
                    np.nextafter(boundary, np.float32(np.inf), dtype=np.float32),
                ]
            )
        for variant in ("adpep57", "adpep_all"):
            for anchor in anchors:
                self.assertTrue(safe_interval(anchor, "mosi", variant).contains(anchor))

    def test_adpep57_and_all_preserve_required_classes(self):
        anchors = np.linspace(-4, 4, 161, dtype=np.float32)
        ensembles = anchors[::-1].copy()
        for variant, required in (("adpep57", 2), ("adpep_all", 3)):
            projected, details = project_array(
                anchors, ensembles, "mosei", variant
            )
            anchor_decisions = evaluator_decisions(anchors, "mosei")
            final_decisions = evaluator_decisions(projected, "mosei")
            for left, right in zip(
                anchor_decisions[:required], final_decisions[:required]
            ):
                np.testing.assert_array_equal(left, right)
            self.assertFalse(any(item.fallback_to_anchor for item in details))

    def test_projection_rejects_nonfinite_and_shape_mismatch(self):
        with self.assertRaises(FloatingPointError):
            project_array([0, np.nan], [0, 1], "mosi", "adpep_all")
        with self.assertRaises(ValueError):
            project_array([0], [0, 1], "mosi", "adpep_all")

    def test_safety_fallback_returns_anchor(self):
        wrong_interval = DecisionInterval(
            np.float32(-3), np.float32(-2), True, True
        )
        with mock.patch(
            "trains.singleTask.anchor_decision_projection.safe_interval",
            return_value=wrong_interval,
        ):
            result = project_prediction(1.2, -2.5, "mosi", "adpep_all")
        self.assertTrue(result.fallback_to_anchor)
        self.assertEqual(result.value, np.float32(1.2))
        self.assertEqual(result.fallback_reason, "post_projection_decision_mismatch")

    def test_sample_index_binding_not_row_binding(self):
        anchor = pd.DataFrame(
            {"sample_index": [0, 1], "sample_id": ["a", "b"]}
        )
        pe5 = anchor.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(RuntimeError):
            evaluator._bind_without_labels(anchor, pe5)

    def test_split_and_duplicate_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            pd.DataFrame(
                {
                    "sample_index": [0, 0],
                    "sample_id": ["a", "a"],
                    "label": [100, -100],
                    "LAV_pred": [0, 0],
                    "LA_pred": [0, 0],
                    "LV_pred": [0, 0],
                    "L_pred": [0, 0],
                    "Split": ["valid", "test"],
                    "Method": ["Online", "Online"],
                }
            ).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                evaluator._prediction_frame(path, "valid", "Online", use_labels=False)

    def test_label_column_is_not_loaded_for_projection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pred.csv"
            pd.DataFrame(
                {
                    "sample_index": [0],
                    "sample_id": ["a"],
                    "label": [12345],
                    "LAV_pred": [0],
                    "LA_pred": [0],
                    "LV_pred": [0],
                    "L_pred": [0],
                    "Split": ["valid"],
                    "Method": ["Online"],
                }
            ).to_csv(path, index=False)
            frame = evaluator._prediction_frame(
                path, "valid", "Online", use_labels=False
            )
            self.assertNotIn("label", frame.columns)

    def test_input_hash_detects_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "x"
            path.write_bytes(b"before")
            original = evaluator.sha256(path)
            path.write_bytes(b"after")
            self.assertNotEqual(original, evaluator.sha256(path))

    def test_retention_formula(self):
        gain_pe5, gain_method, ratio = retention_ratio(10, 6, 8, True)
        self.assertEqual((gain_pe5, gain_method, ratio), (4, 2, 0.5))
        gain_pe5, gain_method, ratio = retention_ratio(0.5, 0.9, 0.7, False)
        self.assertAlmostEqual(gain_pe5, 0.4)
        self.assertAlmostEqual(gain_method, 0.2)
        self.assertAlmostEqual(ratio, 0.5)
        self.assertIsNone(retention_ratio(1, 2, 1.5, True)[2])

    def test_missing_macro_is_arithmetic_mean(self):
        frame = pd.DataFrame(
            {
                "sample_index": [0, 1, 2],
                "sample_id": ["a", "b", "c"],
                "label": [-1, 0, 1],
                "LAV_pred": [-1, 0, 1],
                "LA_pred": [-1, 0, 0],
                "LV_pred": [-1, 1, 1],
                "L_pred": [0, 0, 1],
                "Split": ["valid"] * 3,
                "Method": ["x"] * 3,
            }
        )
        metrics, objective = metrics_from_predictions(frame)
        for metric in ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE", "Loss"):
            self.assertAlmostEqual(
                metrics["MissingMacro"][metric],
                np.mean([metrics[mode][metric] for mode in ("LA", "LV", "L")]),
            )
        self.assertAlmostEqual(
            objective,
            0.5 * metrics["LAV"]["MAE"]
            + 0.5 * metrics["MissingMacro"]["MAE"],
        )

    def test_cli_exposes_mosi_and_mosei_without_mosi_anchor_rule(self):
        with mock.patch(
            "sys.argv",
            [
                "x",
                "--dataset",
                "mosei",
                "--seeds",
                "1111",
                "1112",
                "1113",
                "1114",
                "1115",
            ],
        ):
            args = evaluator.parse_args()
        self.assertEqual(args.dataset, "mosei")
        self.assertIn("mosei", args.input_root)
        self.assertNotIn("anchor_seed = 1114", inspect.getsource(evaluator))


if __name__ == "__main__":
    unittest.main()
