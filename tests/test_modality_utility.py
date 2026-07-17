import inspect
import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import analyze_modality_utility as analyze
from trains.singleTask.modality_utility_utils import (
    assign_quartiles, bootstrap_summary, capture_rng_state, classify_change,
    classify_utility, clone_buffers, clone_parameters, fit_quartile_edges,
    gain_values, infer_padding_mask, label_bin, masked_input_sensitivity,
    permute_bound_fields, representation_pair_summary, restore_rng_state,
    rng_states_equal, sattolo_derangement, shuffle_damage, tensor_maps_equal,
    validate_derangement, vector_from_json, vector_to_json,
)


class ModalityUtilityTests(unittest.TestCase):
    def test_sattolo_derangement_is_bijection_without_fixed_points(self):
        mapping = sattolo_derangement(1284, 270710)
        self.assertTrue(validate_derangement(mapping, 1284))
        np.testing.assert_array_equal(np.sort(mapping), np.arange(1284))
        self.assertFalse(np.any(mapping == np.arange(1284)))

    def test_derangement_is_fixed_and_split_specific(self):
        first = sattolo_derangement(20, 9)
        np.testing.assert_array_equal(first, sattolo_derangement(20, 9))
        self.assertFalse(np.array_equal(first, sattolo_derangement(20, 10)))

    def test_invalid_derangements_are_rejected(self):
        for mapping in ([0, 2, 1], [1, 1, 0], [1, 2]):
            with self.assertRaises(ValueError):
                validate_derangement(mapping, 3)

    def test_bound_feature_mask_length_and_extra_field_move_together(self):
        mapping = np.array([1, 2, 0])
        result = permute_bound_fields(
            torch.tensor([[10], [20], [30]]), mapping,
            mask=torch.tensor([[1], [0], [1]]),
            lengths=np.array([5, 6, 7]),
            extra_fields={"quality": np.array([.1, .2, .3])},
        )
        torch.testing.assert_close(result["features"].view(-1), torch.tensor([20, 30, 10]))
        torch.testing.assert_close(result["mask"].view(-1), torch.tensor([0, 1, 1]))
        np.testing.assert_array_equal(result["lengths"], [6, 7, 5])
        np.testing.assert_allclose(result["quality"], [.2, .3, .1])

    def test_gain_formulas_have_registered_sign(self):
        label = np.array([0., 0.])
        text = np.array([2., 1.])
        with_audio = np.array([1., 2.])
        np.testing.assert_allclose(gain_values(text, with_audio, label), [1., -1.])

    def test_shuffle_damage_formula(self):
        correct = np.array([.5, 2.])
        shuffled = np.array([1.5, 1.])
        label = np.zeros(2)
        np.testing.assert_allclose(shuffle_damage(correct, shuffled, label), [1., -1.])

    def test_bootstrap_is_paired_fixed_and_complete(self):
        first = bootstrap_summary([1., -1., 2., 0.], 200, 11)
        second = bootstrap_summary([1., -1., 2., 0.], 200, 11)
        self.assertEqual(first, second)
        self.assertEqual(first["count"], 4)
        self.assertEqual(first["positive_fraction"], .5)
        self.assertEqual(first["negative_fraction"], .25)
        self.assertEqual(first["zero_fraction"], .25)

    def test_padding_is_excluded_from_sensitivity(self):
        features = torch.tensor([[[2.], [0.]]])
        gradients = torch.tensor([[[3.], [1000.]]])
        mask = infer_padding_mask(features)
        gradnorm, sensitivity = masked_input_sensitivity(features, gradients, mask)
        np.testing.assert_allclose(gradnorm, [3.])
        np.testing.assert_allclose(sensitivity, [3.])

    def test_explicit_lengths_create_padding_mask(self):
        features = torch.ones(2, 4, 1)
        mask = infer_padding_mask(features, [2, 3])
        torch.testing.assert_close(
            mask, torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.bool)
        )

    def test_autograd_input_gradient_does_not_update_parameters(self):
        model = nn.Linear(2, 1)
        before = clone_parameters(model)
        value = torch.ones(3, 2, requires_grad=True)
        gradient = torch.autograd.grad(model(value).sum(), value)[0]
        self.assertGreater(float(gradient.norm()), 0)
        self.assertTrue(tensor_maps_equal(before, clone_parameters(model)))
        self.assertTrue(all(parameter.grad is None for parameter in model.parameters()))

    def test_parameter_buffer_and_rng_snapshots_are_exact(self):
        model = nn.BatchNorm1d(2)
        parameters, buffers = clone_parameters(model), clone_buffers(model)
        random.seed(4); np.random.seed(4); torch.manual_seed(4)
        state = capture_rng_state()
        random.random(); np.random.rand(); torch.rand(1)
        restore_rng_state(state)
        self.assertTrue(rng_states_equal(state, capture_rng_state()))
        self.assertTrue(tensor_maps_equal(parameters, clone_parameters(model)))
        self.assertTrue(tensor_maps_equal(buffers, clone_buffers(model)))

    def test_representation_summary_and_vector_roundtrip(self):
        base = np.array([[1., 0.], [0., 1.], [1., 1.]])
        other = base + np.array([[1., 0.], [1., 0.], [1., 0.]])
        summary = representation_pair_summary(base, other)
        self.assertEqual(summary["Count"], 3)
        self.assertGreater(summary["NormMean"], 0)
        vector = np.array([1.25, -2.5, 3.])
        np.testing.assert_allclose(vector_from_json(vector_to_json(vector)), vector)

    def test_train_edges_are_reused_for_valid_quartiles(self):
        train = np.arange(8, dtype=float)
        valid = np.array([0., 7.])
        edges = fit_quartile_edges(train)
        np.testing.assert_array_equal(assign_quartiles(valid, edges), ["Q1_low", "Q4_high"])

    def test_fixed_label_bins(self):
        values = [-3, -1, -.5, 0, .5, 1, 3]
        expected = ["[-3,-1)", "[-1,0)", "[-1,0)", "[0,1)", "[0,1)", "[1,3]", "[1,3]"]
        np.testing.assert_array_equal(label_bin(values), expected)

    def test_classification_A_utility_supported(self):
        gain = {"ci_low": .01, "mean": .02}
        damage = {"ci_low": .02, "mean": .03}
        self.assertEqual(
            classify_utility(gain, damage, .2, 1., .2),
            "A. Utility Supported",
        )

    def test_classification_B_underuse_supported(self):
        gain = {"ci_low": -.01, "mean": 0.}
        damage = {"ci_low": -.02, "mean": 0.}
        self.assertEqual(
            classify_utility(gain, damage, .05, 1., .05),
            "B. Underuse Supported",
        )

    def test_classification_C_used_but_unreliable(self):
        gain = {"ci_low": -.01, "mean": -.02}
        damage = {"ci_low": -.02, "mean": -.01}
        self.assertEqual(
            classify_utility(gain, damage, .2, 1., .2),
            "C. Used but Unreliable",
        )

    def test_classification_D_mixed(self):
        gain = {"ci_low": -.01, "mean": .02}
        damage = {"ci_low": .01, "mean": .02}
        self.assertEqual(
            classify_utility(gain, damage, .05, 1., .05),
            "D. Mixed / Inconclusive",
        )

    def test_change_classification_requires_paired_evidence(self):
        positive = {"ci_low": .01, "ci_high": .03}
        negative = {"ci_low": -.03, "ci_high": -.01}
        overlap = {"ci_low": -.01, "ci_high": .01}
        self.assertEqual(classify_change("A", "A", positive, positive), "Enhanced")
        self.assertEqual(classify_change("A", "A", negative, negative), "Weakened")
        self.assertIn("Maintained", classify_change("A", "A", overlap, overlap))

    def test_checkpoint_locator_requires_manifest_valid_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_root = root / "missing_baseline" / "cf_compat_kd_v1" / "benchmark_multiseed"
            manifest_root.mkdir(parents=True)
            gate, cf = root / "gate.pth", root / "best_valid.pth"
            torch.save({"x": torch.tensor(1)}, gate)
            torch.save({"x": torch.tensor(2)}, cf)
            from trains.singleTask.fixed_kd_utils import checkpoint_sha256
            pd.DataFrame([{
                "Seed": 1111, "Checkpoint": str(gate), "SHA256": checkpoint_sha256(gate),
                "Verified": True, "Protocol": "Gate3 validation-best", "BestEpoch": 3,
            }]).to_csv(manifest_root / "gate3_checkpoint_manifest.csv", index=False)
            pd.DataFrame([{
                "Seed": 1111, "CFCompatCheckpoint": str(cf),
                "CFCompatSHA256": checkpoint_sha256(cf),
            }]).to_csv(manifest_root / "checkpoint_manifest.csv", index=False)
            paths = analyze.locate_validation_selected_states(root, 1111)
            self.assertEqual(paths["cfcompat"], cf)
            bad = pd.read_csv(manifest_root / "checkpoint_manifest.csv")
            bad.CFCompatCheckpoint = str(root / "best_test_diagnostic.pth")
            bad.to_csv(manifest_root / "checkpoint_manifest.csv", index=False)
            with self.assertRaises(ValueError):
                analyze.locate_validation_selected_states(root, 1111)

    def test_only_train_and_valid_datasets_are_constructed(self):
        calls = []

        class FakeDataset:
            def __init__(self, args, mode):
                calls.append(mode)
            def __len__(self):
                return 2
            def __getitem__(self, index):
                return {"x": index}

        args = mock.Mock(batch_size=1)
        with mock.patch.object(analyze, "MMDataset", FakeDataset):
            analyze.build_audit_datasets(args, 0)
        self.assertEqual(calls, ["train", "valid"])

    def test_cli_is_locked(self):
        bad = (
            ["x", "--seed", "2"], ["x", "--shuffle-repeats", "2"],
            ["x", "--bootstrap-samples", "20"], ["x", "--num-workers", "1"],
        )
        for argv in bad:
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    analyze.parse_args()

    def test_summary_only_requires_verification(self):
        with mock.patch.object(sys, "argv", ["x", "--summary-only"]):
            with self.assertRaises(SystemExit):
                analyze.parse_args()

    def test_source_has_no_training_or_checkpoint_write(self):
        source = Path("analyze_modality_utility.py").read_text()
        self.assertNotIn("torch.optim", source)
        self.assertNotIn(".backward(", source)
        self.assertNotIn("torch.save", source)
        self.assertNotIn("git switch -c", source)

    def test_source_uses_exact_stage6a_representation_hook(self):
        source = Path("analyze_modality_utility.py").read_text()
        self.assertIn("model.backbone.proj1.register_forward_pre_hook", source)

    def test_two_frozen_states_and_required_outputs_declared(self):
        self.assertEqual(analyze.STATES, ("gate3_init", "cfcompat_best_valid"))
        source = Path("analyze_modality_utility.py").read_text()
        for name in (
            "standard_mode_metrics.csv", "sample_predictions.csv",
            "sample_modality_gains.csv", "shuffle_sample_metrics.csv",
            "input_sensitivity_samples.csv", "representation_contribution_samples.csv",
            "conditional_label_bins.csv", "conditional_compatibility_quartiles.csv",
            "conditional_text_error_quartiles.csv", "model_state_comparison.csv",
            "audit_summary.json", "stage7a_modality_utility_audit.md",
        ):
            self.assertIn(name, source)

    def test_protocol_freezes_base_and_stop_boundary(self):
        text = Path("MODALITY_UTILITY_AUDIT_PROTOCOL.md").read_text()
        self.assertIn("13a6eb7d7e9708e6480033f80be7f48312739c04", text)
        self.assertIn("does not create Stage 7B", text)

    def test_summary_only_regenerates_before_hash_comparison(self):
        source = inspect.getsource(analyze.verify_existing)
        self.assertIn("regenerate_summaries", source)
        self.assertIn("old_manifest != new_manifest", source)


if __name__ == "__main__":
    unittest.main()
