import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
import torch.nn as nn

import train_lds
from trains.singleTask import lds_utils
from trains.singleTask.lds_utils import (
    LDS_BIN_COUNT,
    assert_fixed_lds_config,
    compute_full_dlf_loss_lds,
    compute_missing_task_loss_lds,
    compute_task_loss_per_sample,
    density_group_thresholds,
    gaussian_kernel,
    label_to_bin_indices,
    lds_checkpoint_path,
    lds_v1_config,
    prepare_train_label_weights,
    sample_weights_for_indices,
    validation_diagnostics,
    weighted_task_loss,
    write_density_audit,
)
from trains.singleTask.missing_utils import build_single_split_loader, compute_task_loss


class LDSModDropTests(unittest.TestCase):
    def good_train_labels(self):
        return np.concatenate((np.full(160, -2.85), np.full(80, -1.25), np.full(40, 0.25), np.full(20, 2.85)))

    def five_head_output(self):
        return {
            "output_logit": torch.tensor([[1.0], [-2.0]]),
            "logits_c": torch.tensor([[2.0], [-1.0]]),
            "logits_l_hetero": torch.tensor([[3.0], [-3.0]]),
            "logits_v_hetero": torch.tensor([[4.0], [-4.0]]),
            "logits_a_hetero": torch.tensor([[5.0], [-5.0]]),
        }

    def test_fixed_config_is_exact_and_rejects_mismatch(self):
        config = lds_v1_config()
        self.assertEqual(config["version"], "LDS-v1")
        self.assertEqual(config["bin_count"], 60)
        self.assertEqual(config["alpha"], 0.5)
        self.assertEqual(config["kernel_sigma"], 1.0)
        self.assertEqual(config["clip_min"], 0.2)
        self.assertEqual(config["clip_max"], 5.0)
        assert_fixed_lds_config(config)
        changed = dict(config)
        changed["alpha"] = 0.7
        with self.assertRaises(ValueError):
            assert_fixed_lds_config(changed)

    def test_bin_boundaries_and_out_of_range_are_exact(self):
        values = np.array([-3.0, -2.9, -1.00001, -1.0, -0.99999, 0.0, 2.99999, 3.0])
        self.assertEqual(label_to_bin_indices(values).tolist(), [0, 1, 19, 20, 20, 30, 59, 59])
        with self.assertRaises(ValueError):
            label_to_bin_indices([-3.001])
        with self.assertRaises(ValueError):
            label_to_bin_indices([3.001])

    def test_kernel_is_normalized_and_fixed_size(self):
        kernel = gaussian_kernel()
        self.assertEqual(kernel.shape, (5,))
        self.assertAlmostEqual(float(kernel.sum()), 1.0, places=12)
        self.assertTrue(np.all(kernel > 0.0))

    def test_density_uses_train_labels_only(self):
        train = self.good_train_labels()
        valid = np.array([-3.0, 3.0, 3.0])
        held_out = np.array([3.0, 3.0, 3.0, 3.0])
        first = prepare_train_label_weights(train)
        _ = valid, held_out
        second = prepare_train_label_weights(train.copy())
        np.testing.assert_allclose(first.sample_weights, second.sample_weights)
        np.testing.assert_array_equal(first.counts, second.counts)

    def test_same_train_labels_have_same_weights_and_safety(self):
        artifacts = prepare_train_label_weights(self.good_train_labels())
        repeated = prepare_train_label_weights(self.good_train_labels())
        np.testing.assert_allclose(artifacts.sample_weights, repeated.sample_weights)
        self.assertAlmostEqual(float(artifacts.sample_weights.mean()), 1.0, places=6)
        self.assertTrue(np.all(np.isfinite(artifacts.sample_weights)))
        self.assertTrue(np.all(artifacts.sample_weights > 0.0))
        self.assertLessEqual(float(artifacts.sample_weights.max()), 5.5)
        self.assertGreaterEqual(len(np.unique(artifacts.sample_weights)), 3)

    def test_lower_density_non_decreasing_weight_and_same_bin_weight(self):
        artifacts = prepare_train_label_weights(self.good_train_labels())
        order = np.argsort(artifacts.smooth_counts)
        self.assertTrue(np.all(np.diff(artifacts.final_weights_by_bin[order]) <= 1e-12))
        same = np.flatnonzero(artifacts.bin_indices == artifacts.bin_indices[0])
        self.assertTrue(np.all(artifacts.sample_weights[same] == artifacts.sample_weights[same[0]]))

    def test_stable_index_binding_and_shuffled_gather(self):
        artifacts = prepare_train_label_weights(self.good_train_labels())
        indices = torch.tensor([5, 299, 1, 160, 42])
        gathered = sample_weights_for_indices(artifacts, indices, torch.device("cpu"))
        np.testing.assert_allclose(gathered.numpy(), artifacts.sample_weights[indices.numpy()])
        permuted = torch.tensor([42, 1, 5, 160, 299])
        remapped = sample_weights_for_indices(artifacts, permuted, torch.device("cpu"))
        self.assertAlmostEqual(float(gathered[0]), float(remapped[2]), places=7)
        with self.assertRaises(IndexError):
            sample_weights_for_indices(artifacts, torch.tensor([9999]))

    def test_no_weighted_sampler_and_train_loader_boundary(self):
        source = Path("train_lds.py").read_text(encoding="utf-8")
        self.assertNotIn("WeightedRandomSampler", source)
        self.assertNotIn('dataloader["test"]', source)
        self.assertNotIn("dataloader['test']", source)
        self.assertIn('{"train", "valid"}', source)
        self.assertIn("shuffle=False", inspect.getsource(build_single_split_loader))

    def test_all_one_weights_match_stage1_task_loss(self):
        output, labels = self.five_head_output(), torch.tensor([[0.0], [1.0]])
        stage1, _ = compute_task_loss(output, labels, nn.L1Loss())
        weighted = weighted_task_loss(compute_task_loss_per_sample(output, labels), torch.ones(2))
        torch.testing.assert_close(weighted, stage1, rtol=1e-6, atol=1e-7)

    def test_five_head_factors_are_1_1_3_1_1(self):
        output = {name: torch.ones(1, 1) for name in ("output_logit", "logits_c", "logits_l_hetero", "logits_v_hetero", "logits_a_hetero")}
        output["logits_l_hetero"] = torch.full((1, 1), 2.0)
        self.assertAlmostEqual(float(compute_task_loss_per_sample(output, torch.zeros(1, 1))[0]), 10.0, places=6)

    def test_only_task_is_replaced_in_full_loss(self):
        output, labels = self.five_head_output(), torch.tensor([[0.0], [1.0]])
        old_task = torch.tensor(5.0)
        base_total = torch.tensor(17.0)
        details = {"task_loss": old_task, "reconstruction_loss": torch.tensor(2.0), "specific_loss": torch.tensor(3.0), "orthogonality_loss": torch.tensor(4.0), "similarity_loss": torch.tensor(5.0)}
        weights = torch.tensor([1.0, 2.0])
        with mock.patch.object(lds_utils, "compute_full_dlf_loss", return_value=(base_total, details)):
            total, revised = compute_full_dlf_loss_lds(output, labels, nn.L1Loss(), None, None, weights)
        expected = base_total - old_task + weighted_task_loss(compute_task_loss_per_sample(output, labels), weights)
        torch.testing.assert_close(total, expected)
        self.assertEqual(float(revised["reconstruction_loss"]), 2.0)
        self.assertEqual(float(revised["specific_loss"]), 3.0)
        self.assertEqual(float(revised["orthogonality_loss"]), 4.0)
        self.assertEqual(float(revised["similarity_loss"]), 5.0)

    def test_missing_loss_has_no_auxiliary_return(self):
        weighted, unweighted = compute_missing_task_loss_lds(self.five_head_output(), torch.tensor([[0.0], [1.0]]), torch.tensor([1.0, 2.0]), nn.L1Loss())
        self.assertEqual(weighted.ndim, 0)
        self.assertEqual(unweighted.ndim, 0)

    def test_total_objective_is_full_plus_missing_at_eta_one(self):
        full, missing = torch.tensor(2.5), torch.tensor(7.0)
        self.assertEqual(float(full + 1.0 * missing), 9.5)
        with mock.patch.object(sys, "argv", ["train_lds.py", "--eta", "0.5"]):
            with self.assertRaises(SystemExit):
                train_lds.parse_args()

    def test_no_lds_tuning_cli(self):
        source = inspect.getsource(train_lds.parse_args)
        for forbidden in ("--alpha", "--sigma", "--bin-width", "--clip-min", "--clip-max"):
            self.assertNotIn(forbidden, source)

    def test_checkpoint_and_smoke_paths_are_isolated(self):
        formal = lds_checkpoint_path("pt", "mosi", 1111)
        self.assertEqual(str(formal), "pt/missing_baseline/lds_moddrop_v1/DLF_mosi_seed1111_best.pth")
        args = SimpleNamespace(model_save_dir="pt", smoke_test=True)
        smoke = train_lds.checkpoint_for_run(args, "mosi", 1111)
        self.assertIn("lds_moddrop_v1/smoke", str(smoke))
        self.assertNotEqual(str(formal), str(smoke))

    def test_fixed_bin_macro_ignores_empty_bins_without_coercing_nan(self):
        artifacts = prepare_train_label_weights(self.good_train_labels())
        bins, _, macro = validation_diagnostics(np.array([-2.0, -2.5]), np.array([-2.5, -2.0]), artifacts, 1111, 1)
        self.assertAlmostEqual(macro, 0.5, places=6)
        empty = next(row for row in bins if row["Bin"] == "{0}")
        self.assertTrue(np.isnan(empty["MAE"]))

    def test_density_group_thresholds_depend_on_train_weights(self):
        artifacts = prepare_train_label_weights(self.good_train_labels())
        first = density_group_thresholds(artifacts.sample_weights)
        second = density_group_thresholds(artifacts.sample_weights.copy())
        self.assertEqual(first, second)

    def test_density_audit_writes_60_bins_and_fixed_files(self):
        artifacts = prepare_train_label_weights(self.good_train_labels())
        with tempfile.TemporaryDirectory() as tmp:
            summary = write_density_audit(artifacts, tmp)
            self.assertEqual(summary["train_sample_count"], 300)
            self.assertEqual(len((Path(tmp) / "train_label_density_bins.csv").read_text().splitlines()), LDS_BIN_COUNT + 1)
            self.assertTrue((Path(tmp) / "train_sample_weights_summary.json").is_file())
            self.assertTrue((Path(tmp) / "train_fixed_emotion_bins.csv").is_file())
            self.assertTrue((Path(tmp) / "LDS_V1_CONFIG.json").is_file())

    def test_audit_function_has_no_model_or_optimizer(self):
        source = inspect.getsource(train_lds.run_density_audit)
        self.assertNotIn("DLF(", source)
        self.assertNotIn("optimizer", source)
        self.assertNotIn("MMDataLoader", source)

    def test_nan_inf_and_shape_checks_fail_closed(self):
        with self.assertRaises(ValueError):
            label_to_bin_indices([np.nan])
        with self.assertRaises(ValueError):
            weighted_task_loss(torch.ones(2), torch.ones(3))
        with self.assertRaises(FloatingPointError):
            weighted_task_loss(torch.ones(2), torch.tensor([1.0, float("inf")]))


if __name__ == "__main__":
    unittest.main()
