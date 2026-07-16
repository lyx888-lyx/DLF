import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train_gradient_aligned_cfcompat as train
from trains.singleTask.gradient_aligned_cfcompat_utils import (
    GRADIENT_POLICIES, MissingSequenceDigest, add_gradient_tuples,
    clone_gradients, combine_task_kd_gradients, compare_tensor_tuples,
    finite_quantiles, gradient_cosine, gradient_dot, gradient_norm,
    group_gradient_metrics, ordered_autograd, trainable_named_parameters,
    replay_corrected_total, subtract_gradient_tuples, write_parameter_gradients,
)


class Tiny(nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.first = nn.Parameter(torch.tensor([1.0, -2.0], dtype=dtype))
        self.unused = nn.Parameter(torch.tensor([3.0], dtype=dtype))

    def forward(self):
        return self.first


class GradientAlignedCFCompatTests(unittest.TestCase):
    def setUp(self):
        self.model = Tiny()
        self.names, self.parameters = trainable_named_parameters(self.model)

    def test_policy_registry_is_exact(self):
        self.assertEqual(GRADIENT_POLICIES, ("manual_replay", "conflict_drop", "task_anchored_projection"))
        self.assertEqual(set(train.METHODS), set(GRADIENT_POLICIES))

    def test_named_parameter_order_is_canonical(self):
        self.assertEqual(self.names, [name for name, value in self.model.named_parameters() if value.requires_grad])
        self.assertEqual([id(value) for value in self.parameters], [id(value) for value in self.model.parameters()])

    def test_duplicate_trainable_parameter_is_rejected(self):
        class Duplicate(nn.Module):
            def __init__(self):
                super().__init__(); value = nn.Parameter(torch.ones(1)); self.register_parameter("a", value)
                self.__dict__["_parameters"]["b"] = value
        with self.assertRaises(ValueError): trainable_named_parameters(Duplicate())

    def test_ordered_autograd_preserves_none(self):
        gradients = ordered_autograd(self.model().sum(), self.parameters, retain_graph=False)
        torch.testing.assert_close(gradients[0], torch.ones(2)); self.assertIsNone(gradients[1])

    def test_ordered_autograd_rejects_nonfinite(self):
        loss = (self.model() * torch.tensor(float("nan"))).sum()
        with self.assertRaises(FloatingPointError): ordered_autograd(loss, self.parameters, False)

    def test_manual_replay_equals_total_backward(self):
        task_loss = (self.model() ** 2).sum(); kd_loss = (self.model() - 4).abs().sum()
        task = ordered_autograd(task_loss, self.parameters, True); kd = ordered_autograd(kd_loss, self.parameters, True)
        _, total, _ = combine_task_kd_gradients(self.parameters, task, kd, "manual_replay")
        self.model.zero_grad(set_to_none=True); (task_loss + kd_loss).backward()
        comparison = compare_tensor_tuples(clone_gradients(self.parameters), total)
        self.assertGreaterEqual(comparison["cosine"], .999999)
        self.assertLessEqual(comparison["max_abs_difference"], 1e-6)

    def test_manual_replay_preserves_raw_kd(self):
        task = (torch.tensor([1., 0.]), None); kd = (torch.tensor([0., 2.]), None)
        used, total, metrics = combine_task_kd_gradients(self.parameters, task, kd, "manual_replay")
        torch.testing.assert_close(used[0], kd[0]); torch.testing.assert_close(total[0], torch.tensor([1., 2.]))
        self.assertEqual(metrics["RemovedComponentNorm"], 0.)

    def test_replay_correction_exactly_recovers_reference(self):
        task = (torch.tensor([1., 2.]), None); raw = (torch.tensor([3., 4.]), None)
        reference = (torch.tensor([4.0001, 6.]), None)
        corrected = replay_corrected_total(reference, task, raw, raw)
        torch.testing.assert_close(corrected[0], reference[0])

    def test_conflict_definition_is_strictly_negative(self):
        for kd, expected in ((torch.tensor([0., 1.]), False), (torch.tensor([-1., 0.]), True)):
            _, _, metrics = combine_task_kd_gradients(self.parameters, (torch.tensor([1., 0.]), None), (kd, None), "conflict_drop")
            self.assertEqual(metrics["ConflictFlag"], expected)

    def test_conflict_drop_zeros_entire_kd(self):
        task = (torch.tensor([1., 0.]), torch.tensor([1.]))
        kd = (torch.tensor([-1., 4.]), torch.tensor([-7.]))
        used, total, metrics = combine_task_kd_gradients(self.parameters, task, kd, "conflict_drop")
        self.assertIsNone(used[0]); self.assertIsNone(used[1])
        torch.testing.assert_close(total[0], task[0]); torch.testing.assert_close(total[1], task[1])
        self.assertEqual(metrics["GlobalKDUsedGradNorm"], 0.)

    def test_conflict_drop_keeps_nonconflicting_kd(self):
        task = (torch.tensor([1., 0.]), None); kd = (torch.tensor([1., 4.]), None)
        used, _, _ = combine_task_kd_gradients(self.parameters, task, kd, "conflict_drop")
        torch.testing.assert_close(used[0], kd[0])

    def test_projection_formula_is_exact(self):
        task = (torch.tensor([2., 0.]), None); kd = (torch.tensor([-1., 3.]), None)
        used, _, metrics = combine_task_kd_gradients(self.parameters, task, kd, "task_anchored_projection")
        torch.testing.assert_close(used[0], torch.tensor([0., 3.]))
        self.assertAlmostEqual(metrics["ProjectionCoefficient"], -.5)

    def test_projection_makes_global_dot_zero(self):
        task = (torch.tensor([2., 1.]), torch.tensor([1.]))
        kd = (torch.tensor([-4., 3.]), torch.tensor([-2.]))
        used, _, metrics = combine_task_kd_gradients(self.parameters, task, kd, "task_anchored_projection")
        self.assertLessEqual(abs(gradient_dot(task, used)), metrics["ProjectionTolerance"])

    def test_projection_does_not_modify_task(self):
        task = (torch.tensor([2., 1.]), None); original = task[0].clone()
        combine_task_kd_gradients(self.parameters, task, (torch.tensor([-4., 0.]), None), "task_anchored_projection")
        torch.testing.assert_close(task[0], original)

    def test_projection_does_not_restore_kd_norm(self):
        task = (torch.tensor([1., 0.]), None); kd = (torch.tensor([-1., 2.]), None)
        used, _, _ = combine_task_kd_gradients(self.parameters, task, kd, "task_anchored_projection")
        self.assertLess(gradient_norm(used), gradient_norm(kd))

    def test_projection_can_create_coordinate_for_none_kd(self):
        task = (torch.tensor([1., 0.]), torch.tensor([2.]))
        kd = (torch.tensor([-3., 0.]), None)
        used, _, _ = combine_task_kd_gradients(self.parameters, task, kd, "task_anchored_projection")
        self.assertIsNotNone(used[1]); self.assertNotEqual(float(used[1]), 0.)

    def test_nonconflicting_projection_is_identity(self):
        task = (torch.tensor([1., 0.]), None); kd = (torch.tensor([0., 2.]), None)
        used, _, metrics = combine_task_kd_gradients(self.parameters, task, kd, "task_anchored_projection")
        torch.testing.assert_close(used[0], kd[0]); self.assertEqual(metrics["ProjectionCoefficient"], 0.)

    def test_projection_output_dtype_matches_parameter(self):
        model = Tiny(torch.float64); _, parameters = trainable_named_parameters(model)
        used, _, _ = combine_task_kd_gradients(parameters, (torch.tensor([1., 0.], dtype=torch.float64), None),
                                                (torch.tensor([-1., 2.], dtype=torch.float64), None), "task_anchored_projection")
        self.assertEqual(used[0].dtype, torch.float64)

    def test_global_dot_and_norm_are_finite(self):
        first = (torch.tensor([3., 4.]), None); second = (torch.tensor([1., 2.]), None)
        self.assertEqual(gradient_norm(first), 5.); self.assertEqual(gradient_dot(first, second), 11.)
        self.assertTrue(np.isfinite(gradient_cosine(first, second)))

    def test_write_gradients_does_not_create_unused_zero(self):
        self.model.zero_grad(set_to_none=True)
        write_parameter_gradients(self.parameters, (torch.ones(2), None), accumulate=False)
        self.assertIsNotNone(self.model.first.grad); self.assertIsNone(self.model.unused.grad)

    def test_write_gradients_accumulates_at_frozen_boundary(self):
        self.model.zero_grad(set_to_none=True)
        write_parameter_gradients(self.parameters, (torch.ones(2), None), accumulate=True)
        write_parameter_gradients(self.parameters, (2 * torch.ones(2), None), accumulate=True)
        torch.testing.assert_close(self.model.first.grad, 3 * torch.ones(2))

    def test_add_gradient_tuples_handles_none(self):
        value = add_gradient_tuples((None, torch.tensor([1.])), (torch.tensor([2., 3.]), None))
        torch.testing.assert_close(value[0], torch.tensor([2., 3.])); torch.testing.assert_close(value[1], torch.tensor([1.]))

    def test_policy_delta_is_used_minus_raw_kd(self):
        value = subtract_gradient_tuples((None, torch.tensor([3.])), (torch.tensor([2., 1.]), torch.tensor([1.])))
        torch.testing.assert_close(value[0], torch.tensor([-2., -1.])); torch.testing.assert_close(value[1], torch.tensor([2.]))

    def test_compare_tensor_tuples_reports_mismatch(self):
        result = compare_tensor_tuples((torch.ones(2), None), (torch.tensor([1., 2.]), torch.ones(1)))
        self.assertGreater(result["mismatched_parameter_count"], 0); self.assertGreater(result["max_abs_difference"], 0)

    def test_group_metrics_are_diagnostic_only(self):
        task = (torch.tensor([1., 0.]), None); raw = (torch.tensor([-1., 2.]), None)
        used, _, _ = combine_task_kd_gradients(self.parameters, task, raw, "task_anchored_projection")
        rows = group_gradient_metrics(task, raw, used, self.parameters, {"all": [0, 1]})
        self.assertEqual(rows[0]["ParameterGroup"], "all"); self.assertIn("RemovedNormFraction", rows[0])

    def test_missing_sequence_digest_is_deterministic(self):
        masks = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        first, second = MissingSequenceDigest(), MissingSequenceDigest()
        first.update(masks); second.update(masks)
        self.assertEqual(first.hexdigest(), second.hexdigest()); self.assertEqual(first.count, 2)

    def test_missing_sequence_digest_is_order_sensitive(self):
        masks = torch.tensor([[1., 1., 0.], [1., 0., 1.]])
        first, second = MissingSequenceDigest(), MissingSequenceDigest(); first.update(masks); second.update(masks.flip(0))
        self.assertNotEqual(first.hexdigest(), second.hexdigest())

    def test_finite_quantiles_ignore_nonfinite(self):
        result = finite_quantiles([1., 2., np.nan]); self.assertEqual(result["Mean"], 1.5); self.assertEqual(result["Max"], 2.)

    def test_cli_fixes_seed_and_workers(self):
        with mock.patch.object(sys, "argv", ["x", "--gradient-policy", "manual_replay", "--seeds", "1112"]):
            with self.assertRaises(SystemExit): train.parse_args()
        with mock.patch.object(sys, "argv", ["x", "--gradient-policy", "manual_replay", "--num-workers", "1"]):
            with self.assertRaises(SystemExit): train.parse_args()

    def test_cli_smoke_caps_epochs(self):
        with mock.patch.object(sys, "argv", ["x", "--gradient-policy", "manual_replay", "--smoke-test", "--max-epochs", "9"]):
            self.assertEqual(train.parse_args().max_epochs, 2)

    def test_paths_are_policy_and_checkpoint_isolated(self):
        paths = []
        for policy in GRADIENT_POLICIES:
            with mock.patch.object(sys, "argv", ["x", "--gradient-policy", policy]): cli = train.parse_args()
            result, main, diagnostic = train.method_paths(cli, "mosi"); paths.append(str(result))
            self.assertIn("best_valid", str(main)); self.assertIn("diagnostic", str(diagnostic)); self.assertNotEqual(main, diagnostic)
        self.assertEqual(len(set(paths)), 3)

    def test_nonmanual_formal_run_requires_manual_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(sys, "argv", ["x", "--gradient-policy", "conflict_drop", "--result-root", tmp]): cli = train.parse_args()
            with self.assertRaises(RuntimeError): train.assert_manual_formal_gate(cli)

    def test_manual_formal_gate_accepts_locked_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "missing_baseline/cfcompat_manual_replay_v1/benchmark_train"; source.mkdir(parents=True)
            pd.DataFrame([{**train.EXPECTED_MANUAL}]).to_csv(source / "mosi_per_seed.csv", index=False)
            with mock.patch.object(sys, "argv", ["x", "--gradient-policy", "conflict_drop", "--result-root", tmp]): cli = train.parse_args()
            train.assert_manual_formal_gate(cli)

    def test_stage3_forward_helpers_are_reused(self):
        source = Path("train_gradient_aligned_cfcompat.py").read_text()
        for token in ("compute_full_dlf_loss", "compute_task_loss", "gated_kd_loss", "compatibility_for_modes", "teacher_lav_prediction"):
            self.assertIn(token, source)

    def test_fixed_missing_rng_and_counts_are_guarded(self):
        source = Path("train_gradient_aligned_cfcompat.py").read_text()
        self.assertIn("seed + 104729", source); self.assertIn('Counter({"LA": 435, "LV": 430, "L": 419})', source)

    def test_no_new_kd_hyperparameter_or_threshold(self):
        source = Path("train_gradient_aligned_cfcompat.py").read_text()
        for forbidden in ("--lambda-kd", "--temperature", "--threshold", "per_layer", "per_mode"):
            self.assertNotIn(forbidden, source)

    def test_training_uses_separate_diagnostics_and_one_reference_backward(self):
        source = inspect.getsource(train.train_one_seed)
        self.assertIn("ordered_autograd(full_loss + missing_loss", source)
        self.assertIn("ordered_autograd(kd_loss", source)
        self.assertEqual(source.count(".backward()"), 1)
        self.assertIn("subtract_gradient_tuples(used_gradients, kd_gradients)", source)

    def test_reference_backward_is_confined_to_equivalence_gate(self):
        source = inspect.getsource(train.run_manual_equivalence_gate)
        self.assertIn("(full + missing + kd).backward()", source)

    def test_main_selection_is_validation_only(self):
        source = inspect.getsource(train.train_one_seed)
        self.assertIn("j_valid <= best_valid_j", source); self.assertIn("j_test <= best_test_j", source)
        self.assertIn("main_checkpoint", source); self.assertIn("diagnostic_checkpoint", source)

    def test_eval_source_is_student_only(self):
        source = Path("eval_gradient_aligned_cfcompat.py").read_text()
        self.assertIn("StudentOnly", source); self.assertNotIn("build_frozen_teacher", source)
        self.assertNotIn("load_counterfactual_cache", source)

    def test_parameter_group_mapping_is_reused(self):
        source = Path("train_gradient_aligned_cfcompat.py").read_text()
        self.assertIn("build_parameter_groups(student)", source); self.assertIn("group_gradient_metrics", source)

    def test_probe_and_representation_are_train_only(self):
        source = inspect.getsource(train.train_one_seed)
        self.assertIn('build_single_split_loader(args, "train"', source)
        self.assertIn("gradient_probe", source); self.assertIn("representation_audit", source)

    def test_output_schema_contains_required_diagnostics(self):
        source = Path("train_gradient_aligned_cfcompat.py").read_text()
        for token in ("mosi_gradient_step_metrics.csv", "mosi_gradient_epoch_summary.csv", "mosi_gradient_group_summary.csv",
                      "mosi_gradient_probe_init.csv", "mosi_representation_best_valid.json", "MissingSequenceSHA256",
                      "ActualTaskDescentFraction", "SelectionRegret"):
            self.assertIn(token, source)

    def test_protocol_records_stage3_accumulation_and_no_clipping(self):
        protocol = Path("GRADIENT_ALIGNED_CF_COMPAT_PROTOCOL.md").read_text()
        self.assertIn("update_epochs=10", protocol); self.assertIn("adds none", protocol)


if __name__ == "__main__":
    unittest.main()
