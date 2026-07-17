import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch

import aggregate_cfcompat_prediction_ensemble as aggregate
import eval_cfcompat_prediction_ensemble as evaluator
from trains.singleTask.cfcompat_prediction_ensemble_utils import (
    PE5_SEEDS,
    aligned_prediction_mean,
    compatibility_quartiles,
    equal_prediction_ensemble,
    j_contribution,
    load_locked_checkpoints,
    max_prediction_difference,
    metric_max_difference,
    metrics_from_predictions,
    paired_bootstrap,
    require_locked_members,
    validate_prediction_frame,
)
from trains.singleTask.fixed_kd_utils import checkpoint_sha256
from trains.singleTask.missing_utils import regression_metrics


def prediction_frame(seed, split="valid", offset=0.0):
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "c", "d"],
            "sample_index": [0, 1, 2, 3],
            "label": [-1.0, 0.0, 1.0, 2.0],
            "LAV_pred": [-1.0, 0.5, 1.0, 1.0],
            "LA_pred": [-0.5, 0.0, 1.5, 2.0],
            "LV_pred": [-1.0, 0.0, 0.5, 2.5],
            "L_pred": [-1.5, 0.5, 1.0, 2.0],
        }
    )
    for column in ("LAV_pred", "LA_pred", "LV_pred", "L_pred"):
        frame[column] += float(offset)
    frame["Seed"] = int(seed)
    frame["Method"] = "Online"
    frame["Split"] = split
    frame["SelectedBy"] = "validation_J"
    return frame


class CFCompatPredictionEnsembleTests(unittest.TestCase):
    def test_only_locked_five_seed_order_is_allowed(self):
        self.assertEqual(require_locked_members(PE5_SEEDS), PE5_SEEDS)
        for seeds in (
            PE5_SEEDS[:-1],
            PE5_SEEDS + (1116,),
            tuple(reversed(PE5_SEEDS)),
        ):
            with self.assertRaises(ValueError):
                require_locked_members(seeds)

    def test_sample_index_alignment_is_not_row_alignment(self):
        frames = [
            prediction_frame(seed, offset=index)
            for index, seed in enumerate(PE5_SEEDS)
        ]
        frames[2] = frames[2].iloc[::-1].reset_index(drop=True)
        result = equal_prediction_ensemble(frames, "valid")
        self.assertEqual(result.sample_index.tolist(), [0, 1, 2, 3])
        self.assertTrue(
            np.allclose(
                result.LAV_pred,
                prediction_frame(1111).LAV_pred.to_numpy() + 2.0,
            )
        )

    def test_duplicate_index_is_rejected(self):
        frame = prediction_frame(1111)
        frame.loc[1, "sample_index"] = 0
        with self.assertRaises(ValueError):
            validate_prediction_frame(frame, "valid", 1111)

    def test_label_mismatch_is_rejected(self):
        frames = [prediction_frame(seed) for seed in PE5_SEEDS]
        frames[-1].loc[0, "label"] = 99
        with self.assertRaises(RuntimeError):
            equal_prediction_ensemble(frames, "valid")

    def test_label_binding_uses_exact_original_float32_value(self):
        frames = [prediction_frame(seed) for seed in PE5_SEEDS]
        frames[0].loc[1, "label"] = 0.2
        for frame in frames[1:]:
            frame.loc[1, "label"] = float(np.float32(0.2))
        equal_prediction_ensemble(frames, "valid")

    def test_split_mixing_is_rejected(self):
        frame = prediction_frame(1111)
        frame.loc[0, "Split"] = "test"
        with self.assertRaises(ValueError):
            validate_prediction_frame(frame, "valid", 1111)

    def test_nonfinite_prediction_is_rejected(self):
        frame = prediction_frame(1111)
        frame.loc[0, "L_pred"] = np.inf
        with self.assertRaises(FloatingPointError):
            validate_prediction_frame(frame, "valid", 1111)

    def test_equal_weight_formula(self):
        frames = [
            prediction_frame(seed, offset=index)
            for index, seed in enumerate(PE5_SEEDS)
        ]
        result = equal_prediction_ensemble(frames, "valid")
        expected = np.mean(
            [frame.LA_pred.to_numpy() for frame in frames], axis=0
        )
        self.assertTrue(np.array_equal(result.LA_pred.to_numpy(), expected))
        self.assertEqual(set(result.MemberCount), {5})
        self.assertEqual(set(result.EqualWeights), {True})

    def test_aligned_mean_supports_leave_one_out_without_selection(self):
        frames = [prediction_frame(seed, offset=index) for index, seed in enumerate(PE5_SEEDS[:4])]
        result = aligned_prediction_mean(frames, "valid", "omit1115")
        self.assertEqual(set(result.MemberCount), {4})
        self.assertTrue(np.allclose(result.LAV_pred, prediction_frame(1111).LAV_pred + 1.5))

    def test_j_formula_and_original_classification_metrics(self):
        frame = equal_prediction_ensemble(
            [prediction_frame(seed) for seed in PE5_SEEDS], "valid"
        )
        metrics, objective = metrics_from_predictions(frame)
        expected = 0.5 * metrics["LAV"]["MAE"] + 0.5 * np.mean(
            [metrics[mode]["MAE"] for mode in ("LA", "LV", "L")]
        )
        self.assertAlmostEqual(objective, expected)
        original = regression_metrics(
            torch.tensor(frame.LAV_pred.to_numpy(), dtype=torch.float32),
            torch.tensor(frame.label.to_numpy(), dtype=torch.float32),
        )
        for metric in ("acc_7", "acc_5", "acc_2", "F1_score", "Corr", "MAE"):
            self.assertEqual(metrics["LAV"][metric], original[metric])

    def test_online_offline_comparison_requires_exact_binding(self):
        left = prediction_frame(1111)
        right = prediction_frame(1111, offset=1e-7)
        self.assertLess(max_prediction_difference(left, right, "valid"), 1e-6)
        self.assertEqual(metric_max_difference(left, left.copy()), 0.0)
        right.loc[0, "sample_id"] = "wrong"
        with self.assertRaises(RuntimeError):
            max_prediction_difference(left, right, "valid")

    def test_compatibility_quartiles_use_counterfactual_difference(self):
        source = prediction_frame(1111).rename(
            columns={
                "LAV_pred": "moddrop_LAV_pred",
                "LA_pred": "moddrop_LA_pred",
                "LV_pred": "moddrop_LV_pred",
                "L_pred": "moddrop_L_pred",
            }
        )
        result = compatibility_quartiles(source)
        self.assertEqual(
            set(result.CompatibilityQuartile_LA),
            {"Q1_low", "Q2", "Q3", "Q4_high"},
        )
        self.assertTrue(
            np.all((result.Compatibility_LA > 0) & (result.Compatibility_LA < 1))
        )

    def test_j_contribution_is_sample_level_protocol_formula(self):
        frame = prediction_frame(1111)
        value = j_contribution(frame)
        manual = 0.5 * np.abs(frame.LAV_pred - frame.label) + 0.5 * (
            np.abs(frame.LA_pred - frame.label)
            + np.abs(frame.LV_pred - frame.label)
            + np.abs(frame.L_pred - frame.label)
        ) / 3
        self.assertTrue(np.allclose(value, manual))

    def test_paired_bootstrap_is_fixed_and_reproducible(self):
        left = np.arange(10, dtype=float)
        right = left + 1
        first = paired_bootstrap(left, right)
        second = paired_bootstrap(left, right)
        self.assertEqual(first, second)
        self.assertEqual(first["BootstrapSamples"], 2000)
        self.assertEqual(first["MeanDifference"], -1.0)
        with self.assertRaises(ValueError):
            paired_bootstrap(left, right, samples=100)

    def test_best_single_seed_uses_validation_only(self):
        rows = []
        for seed in PE5_SEEDS:
            rows.extend(
                [
                    {
                        "Split": "valid",
                        "Mode": "LAV",
                        "Seed": seed,
                        "J": float(seed),
                    },
                    {
                        "Split": "test",
                        "Mode": "LAV",
                        "Seed": seed,
                        "J": float(-seed),
                    },
                ]
            )
        self.assertEqual(aggregate.best_validation_seed(pd.DataFrame(rows)), 1111)

    def test_pairwise_diversity_has_all_ten_pairs_per_mode_split(self):
        frames = {
            split: {
                seed: prediction_frame(seed, split, offset=index * 0.1)
                for index, seed in enumerate(PE5_SEEDS)
            }
            for split in ("valid", "test")
        }
        diversity = aggregate.pairwise_diversity(frames)
        self.assertEqual(len(diversity), 2 * 4 * 10)
        self.assertTrue(np.isfinite(diversity.select_dtypes(include=[np.number])).all().all())

    def test_manifest_rejects_ema_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stage8 = (
                root
                / "missing_baseline"
                / "cfcompat_stability_v1"
                / "mosi"
            )
            stage8.mkdir(parents=True)
            methods = []
            for seed in PE5_SEEDS:
                checkpoint = root / "ema_{}_online_best_valid.pth".format(seed)
                checkpoint.write_bytes(b"x")
                methods.append(
                    {
                        "Seed": seed,
                        "Method": "Online",
                        "Checkpoint": str(checkpoint),
                        "CheckpointSHA256": checkpoint_sha256(checkpoint),
                        "BestValidEpoch": 1,
                    }
                )
            (stage8 / "checkpoint_manifest.json").write_text(
                json.dumps(
                    {
                        "AllOnlineReplaysPassed": True,
                        "NoTestSelectedCheckpoint": True,
                        "Methods": methods,
                    }
                )
            )
            with self.assertRaises(RuntimeError):
                load_locked_checkpoints(root, "mosi")

    def test_cli_rejects_member_mode_split_and_worker_changes(self):
        invalid = (
            ["x", "--seeds", "1111", "1112", "1113", "1114"],
            ["x", "--split", "test", "valid"],
            ["x", "--modes", "LAV", "LA", "LV"],
            ["x", "--num-workers", "1"],
        )
        for argv in invalid:
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                evaluator.parse_args()

    def test_aggregate_requires_formal_replay_flag_and_2000_bootstraps(self):
        for argv in (
            ["x"],
            ["x", "--verify-online-offline", "--bootstrap-samples", "100"],
        ):
            with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit):
                aggregate.parse_args()

    def test_inference_source_is_student_only_and_sequential(self):
        source = Path(evaluator.__file__).read_text()
        self.assertNotIn("build_frozen_teacher", source)
        self.assertNotIn("build_frozen_evaluator", source)
        self.assertNotIn("load_counterfactual_cache", source)
        self.assertIn("del student", source)
        self.assertIn("torch.cuda.empty_cache()", source)
        self.assertIn('"MaximumResidentModels": 1', source)
        self.assertNotIn("optimizer", source)
        self.assertNotIn("backward", source)


if __name__ == "__main__":
    unittest.main()
