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

import train_cf_compat_kd
from trains.singleTask.cf_compat_kd_utils import (
    CACHE_COLUMNS,
    build_counterfactual_cache,
    cache_paths,
    cache_summary,
    compatibility_for_modes,
    compatibility_from_deltas,
    effective_sample_size,
    gate_weights,
    gated_kd_loss,
    load_counterfactual_cache,
    locate_stage1_evaluator,
    modes_from_masks,
    preserve_rng,
    stable_average_ranks,
    write_counterfactual_cache,
)


class FakeEvaluator(nn.Module):
    def forward(self, text, audio, vision, mask):
        value = mask[:, 1] + 2 * mask[:, 2]
        return {"output_logit": value.view(-1, 1).to(audio)}


def fake_batch(indices):
    count = len(indices)
    return {
        "text": torch.zeros(count, 2, 2),
        "audio": torch.zeros(count, 2, 2),
        "vision": torch.zeros(count, 2, 2),
        "labels": {"M": torch.arange(count, dtype=torch.float32).view(-1, 1)},
        "index": torch.tensor(indices),
        "id": ["id{}".format(index) for index in indices],
    }


def cache_frame():
    rows = []
    for index in range(4):
        rows.append({
            "sample_index": index, "sample_id": "id{}".format(index), "label": float(index),
            "evaluator_LAV_pred": 3.0, "evaluator_LA_pred": 2.0 + index * .1,
            "evaluator_LV_pred": 1.0 + index * .2, "evaluator_L_pred": float(index),
            "delta_LA": .1 + index, "delta_LV": .2 + index, "delta_L": .3 + index,
            "rank_LA": index + 1., "rank_LV": index + 1., "rank_L": index + 1.,
            "q_LA": (index + .5) / 4, "q_LV": (index + .5) / 4, "q_L": (index + .5) / 4,
            "compat_LA": 1 - (index + .5) / 4, "compat_LV": 1 - (index + .5) / 4,
            "compat_L": 1 - (index + .5) / 4,
        })
    return pd.DataFrame(rows).loc[:, CACHE_COLUMNS]


class CounterfactualCompatibilityTests(unittest.TestCase):
    def test_stable_average_rank_and_ties(self):
        np.testing.assert_allclose(stable_average_ranks([2., 1., 1., 3.]), [3., 1.5, 1.5, 4.])

    def test_rank_uses_mergesort_and_nonfinite_rejected(self):
        self.assertIn('kind="mergesort"', inspect.getsource(stable_average_ranks))
        with self.assertRaises(ValueError):
            stable_average_ranks([0., np.nan])

    def test_q_and_compatibility_strict_ranges_and_monotonicity(self):
        rank, q, compat = compatibility_from_deltas([.1, .3, .2, .4])
        self.assertTrue(np.all((q > 0) & (q < 1)))
        self.assertTrue(np.all((compat > 0) & (compat < 1)))
        self.assertTrue(np.all(np.diff(compat[np.argsort([.1, .3, .2, .4])]) <= 0))
        np.testing.assert_allclose(q, (rank - .5) / 4)
        np.testing.assert_allclose(compat, 1 - q)

    def test_mode_ranks_are_independent(self):
        _, _, first = compatibility_from_deltas([0., 1., 2.])
        _, _, second = compatibility_from_deltas([2., 1., 0.])
        self.assertGreater(first[0], first[-1])
        self.assertLess(second[0], second[-1])

    def test_mask_mapping_and_invalid_mask(self):
        mask = torch.tensor([[1., 1., 0.], [1., 0., 1.], [1., 0., 0.]])
        self.assertEqual(modes_from_masks(mask), ["LA", "LV", "L"])
        with self.assertRaises(ValueError):
            modes_from_masks(torch.tensor([[1., 1., 1.]]))

    def test_c_only_gate_equals_compatibility_and_detaches(self):
        compat = torch.tensor([.2, .8], requires_grad=True)
        gate, reliability = gate_weights(compat, gate_mode="compat")
        torch.testing.assert_close(gate, compat.detach())
        torch.testing.assert_close(reliability, torch.ones_like(compat))
        self.assertFalse(gate.requires_grad)

    def test_reliability_times_compatibility_exact_stage3a_formula(self):
        compat = torch.tensor([.5, .25], requires_grad=True)
        teacher = torch.tensor([[0.], [2.]], requires_grad=True)
        labels = torch.tensor([[0.], [0.]], requires_grad=True)
        gate, reliability = gate_weights(compat, teacher, labels, "reliability_compat")
        torch.testing.assert_close(reliability, torch.exp(-torch.tensor([0., 2.])))
        torch.testing.assert_close(gate, reliability * compat.detach())
        self.assertFalse(gate.requires_grad)

    def test_gated_kd_reduction_and_equal_weights_mean_equivalence(self):
        student = torch.tensor([[1.], [3.]], requires_grad=True)
        teacher = torch.tensor([[0.], [0.]])
        loss, each = gated_kd_loss(student, teacher, torch.ones(2))
        torch.testing.assert_close(loss, nn.SmoothL1Loss()(student, teacher))
        weighted, _ = gated_kd_loss(student, teacher, torch.tensor([1., .5]))
        torch.testing.assert_close(weighted, (torch.tensor([1., .5]) * each).sum() / 1.5)
        weighted.backward()
        self.assertGreater(float(student.grad.norm()), 0.)

    def test_cache_binding_lookup_and_missing_binding_failure(self):
        frame = cache_frame()
        table = {int(row.sample_index): row._asdict() for row in frame.itertuples(index=False)}
        got = compatibility_for_modes(table, [0, 1, 2], ["LA", "LV", "L"], "cpu", torch.float32)
        self.assertTrue(torch.all((got > 0) & (got < 1)))
        with self.assertRaises(KeyError):
            compatibility_for_modes(table, [9], ["LA"], "cpu", torch.float32)

    def test_cache_build_visits_every_train_sample_once_and_train_only(self):
        generator = torch.Generator().manual_seed(9)
        before = generator.get_state().clone()
        frame = build_counterfactual_cache(FakeEvaluator(), [fake_batch([2, 0]), fake_batch([1, 3])], "cpu", generator)
        self.assertEqual(list(frame.columns), list(CACHE_COLUMNS))
        self.assertEqual(frame.sample_index.tolist(), [0, 1, 2, 3])
        self.assertEqual(frame.sample_index.nunique(), 4)
        self.assertTrue(torch.equal(before, generator.get_state()))
        for mode in ("LA", "LV", "L"):
            self.assertTrue(np.all(frame["delta_{}".format(mode)] >= 0))
            self.assertTrue(np.all((frame["compat_{}".format(mode)] > 0) & (frame["compat_{}".format(mode)] < 1)))

    def test_duplicate_or_missing_train_indices_are_rejected(self):
        with self.assertRaises(RuntimeError):
            build_counterfactual_cache(FakeEvaluator(), [fake_batch([0, 0])], "cpu", torch.Generator())

    def test_rng_preservation_covers_python_numpy_torch_and_missing_generator(self):
        random.seed(3); np.random.seed(3); torch.manual_seed(3)
        generator = torch.Generator().manual_seed(11)
        expected_python, expected_numpy = random.random(), np.random.rand()
        expected_torch, expected_generator = torch.rand(1), torch.rand(1, generator=generator)
        random.seed(3); np.random.seed(3); torch.manual_seed(3); generator.manual_seed(11)
        with preserve_rng(generator):
            random.random(); np.random.rand(); torch.rand(1); torch.rand(1, generator=generator)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(np.random.rand(), expected_numpy)
        torch.testing.assert_close(torch.rand(1), expected_torch)
        torch.testing.assert_close(torch.rand(1, generator=generator), expected_generator)

    def test_cache_files_summary_metadata_and_spearman(self):
        frame = cache_frame()
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "evaluator.pth"
            torch.save({"weight": torch.tensor([1.])}, checkpoint)
            paths, config = write_counterfactual_cache(frame, tmp, "mosi", checkpoint, 21)
            self.assertTrue(all(path.is_file() for key, path in paths.items() if key != "directory"))
            self.assertEqual(config["rank_method"], "average")
            self.assertEqual(config["sort_kind"], "mergesort")
            self.assertEqual(config["source"], "train_only")
            loaded, _ = load_counterfactual_cache(tmp, "mosi")
            self.assertEqual(len(loaded), 4)
            summary = json.loads(paths["summary"].read_text())
            self.assertAlmostEqual(summary["LA"]["corr_delta_compat_spearman"], -1.0)
            self.assertIn("config_sha256", config)

    def test_stage1_checkpoint_is_located_from_result_csv_not_guessed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "recorded.pth"
            torch.save({"x": torch.tensor(1)}, checkpoint)
            source = root / "missing_baseline" / "moddrop" / "train"
            source.mkdir(parents=True)
            pd.DataFrame([{"Seed": 1111, "BestEpoch": 21, "Checkpoint": str(checkpoint)}]).to_csv(source / "mosi_per_seed.csv", index=False)
            path, epoch, csv = locate_stage1_evaluator(root, "mosi", 1111)
            self.assertEqual(path, checkpoint)
            self.assertEqual(epoch, 21)
            self.assertEqual(csv.name, "mosi_per_seed.csv")

    def test_ess_and_cache_summary_are_finite(self):
        self.assertAlmostEqual(effective_sample_size([1., 1., 1.]), 3.)
        summary = cache_summary(cache_frame())
        self.assertEqual(set(summary), {"LA", "LV", "L"})
        self.assertTrue(all(np.isfinite(value["compat"]["mean"]) for value in summary.values()))

    def test_cli_rejects_tuning_and_has_required_gate_modes(self):
        with mock.patch.object(sys, "argv", ["x", "--eta", ".5"]):
            with self.assertRaises(SystemExit):
                train_cf_compat_kd.parse_args()
        with mock.patch.object(sys, "argv", ["x", "--lambda-kd", ".5"]):
            with self.assertRaises(SystemExit):
                train_cf_compat_kd.parse_args()
        self.assertEqual(set(train_cf_compat_kd.METHODS), {"compat", "reliability_compat"})

    def test_main_and_diagnostic_paths_are_isolated(self):
        with mock.patch.object(sys, "argv", ["x", "--gate-mode", "compat"]):
            cli = train_cf_compat_kd.parse_args()
        _, main, diagnostic = train_cf_compat_kd.method_paths(cli, "mosi")
        self.assertNotEqual(main, diagnostic)
        self.assertIn("diagnostic", str(diagnostic))
        self.assertIn("best_valid", str(main))
        self.assertIn("best_test_diagnostic", str(diagnostic))

    def test_protocol_source_guards(self):
        source = Path("train_cf_compat_kd.py").read_text()
        for name in ("build_single_split_loader(args, \"train\"", "build_single_split_loader(args, \"test\"", "IsBestValid", "IsBestTestDiagnostic", "torch.save(student.state_dict(), main_checkpoint)", "torch.save(student.state_dict(), diagnostic_checkpoint)"):
            self.assertIn(name, source)
        self.assertNotIn("--temperature", source); self.assertNotIn("--tau", source); self.assertNotIn("WeightedRandomSampler", source)

    def test_gate_quartiles_and_reliability_compat_2x2(self):
        records=[{"sample_index":i,"mode":("LA","LV","L")[i%3],"delta":float(i),"compat":(i+1)/10,"reliability":(i+2)/10,"gate":(i+1)/20,"kd":.1*i,"weighted_kd_contribution":.01*i,"student_missing_abs_label_error":.2*i,"teacher_error":.3*i} for i in range(8)]
        summary, quartiles=train_cf_compat_kd._diagnostic_rows(records,1111,1,"compat"); self.assertEqual(summary["SampleCount"],8); self.assertEqual(sum(row["count"] for row in quartiles),8)
        _, two_by_two=train_cf_compat_kd._diagnostic_rows(records,1111,1,"reliability_compat"); self.assertTrue(any(row["GroupType"]=="reliability_compat_2x2" for row in two_by_two))

    def test_output_schema_names_and_diagnostic_metadata(self):
        source = Path("train_cf_compat_kd.py").read_text()
        for name in ("write_result_csvs", "_epoch_metrics.csv", "gate_summary", "gate_quartiles", "best_valid_predictions", "best_test_diagnostic_predictions", "selected_by", "diagnostic_only", "not_main_result"):
            self.assertIn(name, source)

    def test_full_teacher_and_evaluator_are_frozen_helpers(self):
        source = Path("trains/singleTask/cf_compat_kd_utils.py").read_text()
        self.assertIn("freeze_teacher(evaluator)", source)
        self.assertIn("torch.inference_mode()", source)
        self.assertIn("capture_rng_state()", source)
        self.assertIn("restore_rng_state(state)", source)


if __name__ == "__main__":
    unittest.main()
