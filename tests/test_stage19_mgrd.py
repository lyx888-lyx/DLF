import inspect
import json
import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch

from scripts.mosei import run_stage19_mgrd as runner
from trains.singleTask.anchor_decision_projection import evaluator_decisions
from trains.singleTask import mgrd_utils
from trains.singleTask.mgrd_utils import (
    BOUNDARIES,
    GRANULARITIES,
    _shuffle_flat,
    canonical_ids,
    hinge_equivalence_report,
    multigranular_kd_loss,
    ordinal_probability,
    ordered_id_sha,
    recover_evaluator_decisions,
    recoverability,
)


class Stage19TestLock(unittest.TestCase):
    def test_forbidden_split_hard_fails_before_dataset_construction(self):
        with self.assertRaisesRegex(RuntimeError, "train/valid only"):
            runner.locked_dataset(SimpleNamespace(), "test")

    def test_runner_has_no_allow_test_flag(self):
        source = Path(inspect.getsourcefile(runner)).read_text()
        self.assertNotIn("--allow-test", source)
        self.assertNotIn('choices=("train", "valid", "test")', source)


class ThresholdTests(unittest.TestCase):
    def test_dense_grid_matches_frozen_evaluator(self):
        grid = np.linspace(-3.0, 3.0, 12001, dtype=np.float32)
        acc7, acc5, acc2 = evaluator_decisions(grid, "mosei")
        expected = {"Acc7": acc7.astype(np.int64), "Acc5": acc5.astype(np.int64), "Acc2": acc2.astype(np.int64)}
        for granularity in GRANULARITIES:
            np.testing.assert_array_equal(recover_evaluator_decisions(grid, granularity), expected[granularity])

    def test_boundaries_are_extracted(self):
        self.assertEqual(BOUNDARIES["Acc2"], (0.0,))
        self.assertEqual(BOUNDARIES["Acc5"], (-1.5, -0.5, 0.5, 1.5))
        self.assertEqual(BOUNDARIES["Acc7"], (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5))

    def test_ordinal_monotonicity(self):
        for value in torch.linspace(-3, 3, 101):
            for boundaries in BOUNDARIES.values():
                probability = ordinal_probability(value.view(1), boundaries)
                self.assertTrue(bool(torch.all(probability[:, :-1] >= probability[:, 1:])))


class LossTests(unittest.TestCase):
    def setUp(self):
        self.student = torch.tensor([-1.2, -0.1, 0.3, 1.7, -2.1, 0.8], requires_grad=True)
        self.teacher = torch.tensor([-1.0, 0.2, 0.5, 1.4, -1.8, 1.1])
        self.reference = torch.tensor([-0.9, -0.4, 0.7, 0.9, -2.4, 0.4])
        self.modes = ["LA", "LV", "L", "LA", "LV", "L"]

    def test_mgd_finite_backward_and_weights(self):
        loss, details = multigranular_kd_loss("mgd", self.student, self.teacher, self.modes)
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        self.assertGreater(float(self.student.grad.abs().sum()), 0)
        self.assertAlmostEqual(details["weighted_reg"] + details["weighted_ordinal"], float(loss.detach()), places=6)
        self.assertTrue(all(details[key] > 0 for key in ("L_Acc2", "L_Acc5", "L_Acc7")))

    def test_recoverability_range_side_and_distance(self):
        opposite = recoverability(torch.tensor([[0.9]]), torch.tensor([[0.1]]))
        near = recoverability(torch.tensor([[0.6]]), torch.tensor([[0.6]]))
        far = recoverability(torch.tensor([[0.9]]), torch.tensor([[0.9]]))
        self.assertEqual(float(opposite), 0.0)
        self.assertGreater(float(far), float(near))
        self.assertTrue(0 <= float(far) <= 1)

    def test_all_one_gate_reduces_to_mgd_ordinal(self):
        mgd, mgd_details = multigranular_kd_loss("mgd", self.student, self.teacher, self.modes)
        with mock.patch.object(mgrd_utils, "recoverability", side_effect=lambda left, right: torch.ones_like(left)):
            mgrd, mgrd_details = multigranular_kd_loss(
                "mgrd", self.student, self.teacher, self.modes, self.reference
            )
        self.assertAlmostEqual(float(mgd), float(mgrd), places=6)
        for key in ("L_Acc2", "L_Acc5", "L_Acc7"):
            self.assertAlmostEqual(mgd_details[key], mgrd_details[key], places=6)

    def test_all_zero_gate_preserves_continuous_kd(self):
        with mock.patch.object(mgrd_utils, "recoverability", side_effect=lambda left, right: torch.zeros_like(left)):
            loss, details = multigranular_kd_loss(
                "mgrd", self.student, self.teacher, self.modes, self.reference
            )
        self.assertGreater(details["L_reg"], 0)
        self.assertAlmostEqual(float(loss), 0.5 * details["L_reg"], places=6)

    def test_shuffled_gate_preserves_multiset_and_mass(self):
        values = torch.arange(18, dtype=torch.float32).view(6, 3)
        shuffled = _shuffle_flat(values, torch.Generator().manual_seed(19))
        self.assertEqual(float(values.sum()), float(shuffled.sum()))
        self.assertEqual(sorted(values.view(-1).tolist()), sorted(shuffled.view(-1).tolist()))
        self.assertFalse(torch.equal(values, shuffled))


class CacheTests(unittest.TestCase):
    def test_duplicate_sample_id_hard_fails(self):
        with self.assertRaisesRegex(RuntimeError, "Duplicate"):
            canonical_ids(["a", "a"])

    def test_ordered_sha_changes_with_order(self):
        self.assertNotEqual(ordered_id_sha(["a", "b"]), ordered_id_sha(["b", "a"]))

    def test_bad_cache_order_sha_hard_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "seed1111"
            root.mkdir(parents=True)
            cache = root / "train_predictions.npz"
            np.savez(
                cache,
                sample_id=np.asarray(["a", "b"]),
                teacher_lav=np.asarray([0, 1], dtype=np.float32),
                moddrop_LA=np.asarray([0, 1], dtype=np.float32),
                moddrop_LV=np.asarray([0, 1], dtype=np.float32),
                moddrop_L=np.asarray([0, 1], dtype=np.float32),
            )
            entry = {
                "path": str(cache),
                "sha256": mgrd_utils.sha256_file(cache),
                "sample_count": 2,
                "ordered_sample_id_sha256": "bad",
            }
            manifest = {"locked_test_access_count": 0, "entries": {"train": entry, "valid": entry}}
            (root / "cache_manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(RuntimeError, "ordered sample-ID SHA"):
                runner.PredictionCache(directory, 1111)


class ResumeAndHingeTests(unittest.TestCase):
    def test_rng_state_roundtrip(self):
        sampler = torch.Generator().manual_seed(1)
        missing = torch.Generator().manual_seed(2)
        shuffle = torch.Generator().manual_seed(3)
        state = runner.capture_rng(sampler, missing, shuffle)
        expected = (
            torch.rand(3),
            torch.rand(3, generator=sampler),
            torch.rand(3, generator=missing),
            torch.rand(3, generator=shuffle),
        )
        runner.restore_rng(state, sampler, missing, shuffle)
        actual = (
            torch.rand(3),
            torch.rand(3, generator=sampler),
            torch.rand(3, generator=missing),
            torch.rand(3, generator=shuffle),
        )
        for left, right in zip(expected, actual):
            torch.testing.assert_close(left, right, rtol=0, atol=0)

    def test_epoch_boundary_resume_preserves_training_trajectory(self):
        class ToyDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 12

            def __getitem__(self, index):
                return {"index": index, "x": torch.tensor([float(index), 1.0])}

        def make_state():
            torch.manual_seed(91)
            model = torch.nn.Linear(2, 1)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=1
            )
            sampler = torch.Generator().manual_seed(1111)
            missing = torch.Generator().manual_seed(1111 + 104729)
            shuffle = torch.Generator().manual_seed(1111 + 314159)
            return model, optimizer, scheduler, sampler, missing, shuffle

        def run_epoch(model, optimizer, sampler, missing, shuffle, limit=None):
            loader = torch.utils.data.DataLoader(
                ToyDataset(), batch_size=3, shuffle=True, generator=sampler
            )
            records = []
            optimizer_steps = 0
            for number, batch in enumerate(loader, 1):
                choices = torch.randint(3, (len(batch["index"]),), generator=missing)
                gate = torch.rand((len(batch["index"]), 2), generator=shuffle)
                prediction = model(batch["x"])
                loss = prediction.square().mean() + 0.01 * gate.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                records.append(
                    {
                        "ids": batch["index"].tolist(),
                        "modes": choices.tolist(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss": float(loss.detach()),
                        "prediction": prediction.detach().clone(),
                        "gate_mean": float(gate.mean()),
                        "optimizer_step": optimizer_steps,
                    }
                )
                if limit and number >= limit:
                    break
            return records

        model, optimizer, scheduler, sampler, missing, shuffle = make_state()
        run_epoch(model, optimizer, sampler, missing, shuffle)
        scheduler.step(1.0)
        saved = {
            "model": {key: value.clone() for key, value in model.state_dict().items()},
            "optimizer": copy.deepcopy(optimizer.state_dict()),
            "scheduler": copy.deepcopy(scheduler.state_dict()),
            "rng": runner.capture_rng(sampler, missing, shuffle),
            "epoch": 1,
            "optimizer_step": 4,
            "patience": 0,
        }
        uninterrupted = run_epoch(model, optimizer, sampler, missing, shuffle, limit=2)

        resumed_model, resumed_optimizer, resumed_scheduler, resumed_sampler, resumed_missing, resumed_shuffle = make_state()
        resumed_model.load_state_dict(saved["model"])
        resumed_optimizer.load_state_dict(saved["optimizer"])
        resumed_scheduler.load_state_dict(saved["scheduler"])
        runner.restore_rng(
            saved["rng"], resumed_sampler, resumed_missing, resumed_shuffle
        )
        resumed = run_epoch(
            resumed_model,
            resumed_optimizer,
            resumed_sampler,
            resumed_missing,
            resumed_shuffle,
            limit=2,
        )
        self.assertEqual(saved["epoch"], 1)
        self.assertEqual(saved["optimizer_step"], 4)
        self.assertEqual(saved["patience"], 0)
        self.assertEqual(resumed_scheduler.state_dict(), saved["scheduler"])
        for left, right in zip(uninterrupted, resumed):
            self.assertEqual(left["ids"], right["ids"])
            self.assertEqual(left["modes"], right["modes"])
            self.assertEqual(left["lr"], right["lr"])
            self.assertEqual(left["optimizer_step"], right["optimizer_step"])
            self.assertAlmostEqual(left["loss"], right["loss"], places=10)
            self.assertAlmostEqual(left["gate_mean"], right["gate_mean"], places=10)
            torch.testing.assert_close(left["prediction"], right["prediction"], rtol=0, atol=0)

    def test_hinge_old_new_equivalence_gate_is_reported(self):
        report = hinge_equivalence_report()
        self.assertLessEqual(report["loss_abs_difference"], 1e-7)
        self.assertLessEqual(report["gradient_max_difference"], 1e-6)
        self.assertTrue(report["accepted"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
    def test_amp_fixed_batch_finite_and_fp32_fallback(self):
        layer = torch.nn.Linear(8, 1).cuda()
        values = torch.randn(16, 8, device="cuda")
        with torch.autocast("cuda", dtype=torch.float16):
            amp = layer(values)
        fp32 = layer(values)
        self.assertTrue(bool(torch.isfinite(amp).all()))
        self.assertTrue(bool(torch.isfinite(fp32).all()))


if __name__ == "__main__":
    unittest.main()
