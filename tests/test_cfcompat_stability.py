import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import aggregate_cfcompat_stability as aggregate
import run_cfcompat_stability_multiseed as runner
from trains.singleTask.cfcompat_stability_utils import (
    EMA_DECAY,
    MissingSequenceHasher,
    capture_rng_state,
    expected_missing_sequence_sha,
    initialize_ema,
    optimizer_step_and_update_ema,
    preserve_rng_state,
    rank_trajectory,
    rng_states_equal,
    select_main_strategy,
    state_distance,
    uniform_soup,
    update_ema,
    verify_online_replay,
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)
        self.register_buffer("counter", torch.tensor(0, dtype=torch.int64))

    def forward(self, values):
        return self.linear(values)


def state(value, counter=0, dtype=torch.float32):
    return {
        "weight": torch.tensor([value], dtype=dtype),
        "counter": torch.tensor(counter, dtype=torch.int64),
    }


def entries(seed=1111, source="Online"):
    return [
        {
            "Seed": seed,
            "Epoch": epoch,
            "J_valid": score,
            "SourceMethod": source,
            "state": state(float(epoch)),
        }
        for epoch, score in ((1, .5), (2, .4), (3, .3), (4, .2), (5, .1))
    ]


class CFCompatStabilityTests(unittest.TestCase):
    def test_ema_decay_is_frozen(self):
        self.assertEqual(EMA_DECAY, .999)
        online, ema = ToyModel(), ToyModel()
        with self.assertRaises(ValueError):
            update_ema(ema, online, decay=.99)

    def test_ema_initialization_does_not_change_rng(self):
        torch.manual_seed(7)
        online = ToyModel()
        before = capture_rng_state()
        initialize_ema(online)
        after = capture_rng_state()
        self.assertTrue(rng_states_equal(before, after))

    def test_ema_is_frozen_and_not_in_optimizer(self):
        online = ToyModel()
        ema = initialize_ema(online)
        optimizer = torch.optim.SGD(online.parameters(), lr=.1)
        optimizer_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertTrue(all(not p.requires_grad for p in ema.parameters()))
        self.assertTrue(all(id(p) not in optimizer_ids for p in ema.parameters()))

    def test_ema_formula_and_nonfloating_copy(self):
        online, ema = ToyModel(), ToyModel()
        with torch.no_grad():
            ema.linear.weight.fill_(0)
            online.linear.weight.fill_(2)
            ema.counter.fill_(1)
            online.counter.fill_(9)
        update_ema(ema, online)
        self.assertTrue(
            torch.allclose(
                ema.linear.weight,
                torch.full_like(ema.linear.weight, .002),
                atol=0,
                rtol=0,
            )
        )
        self.assertEqual(int(ema.counter), 9)

    def test_optimizer_wrapper_updates_ema_exactly_once_after_step(self):
        online = ToyModel()
        with torch.no_grad():
            online.linear.weight.zero_()
            online.linear.bias.zero_()
        ema = initialize_ema(online)
        optimizer = torch.optim.SGD(online.parameters(), lr=1.0)
        loss = online(torch.tensor([[1.0, 0.0]])).sum()
        loss.backward()
        optimizer_step_and_update_ema(optimizer, ema, online)
        self.assertEqual(float(online.linear.weight[0, 0]), -1.0)
        self.assertAlmostEqual(float(ema.linear.weight[0, 0]), -.001, places=7)

    def test_enabling_ema_does_not_change_online_gradient(self):
        torch.manual_seed(4)
        left = ToyModel()
        right = ToyModel()
        right.load_state_dict(left.state_dict())
        initialize_ema(right)
        values = torch.tensor([[1.0, 2.0]])
        left(values).sum().backward()
        right(values).sum().backward()
        for a, b in zip(left.parameters(), right.parameters()):
            self.assertTrue(torch.equal(a.grad, b.grad))

    def test_online_single_step_is_unchanged_by_ema(self):
        torch.manual_seed(5)
        left = ToyModel()
        right = ToyModel()
        right.load_state_dict(left.state_dict())
        ema = initialize_ema(right)
        left_opt = torch.optim.SGD(left.parameters(), lr=.2)
        right_opt = torch.optim.SGD(right.parameters(), lr=.2)
        values = torch.tensor([[1.0, 2.0]])
        left(values).sum().backward()
        right(values).sum().backward()
        left_opt.step()
        optimizer_step_and_update_ema(right_opt, ema, right)
        for a, b in zip(left.parameters(), right.parameters()):
            self.assertTrue(torch.equal(a, b))

    def test_auxiliary_context_restores_rng(self):
        before = capture_rng_state()
        with preserve_rng_state():
            torch.rand(10)
            np.random.rand(10)
        after = capture_rng_state()
        self.assertTrue(rng_states_equal(before, after))

    def test_top5_uses_validation_j_and_earlier_epoch_tie(self):
        ranked = rank_trajectory(
            [
                {"Epoch": 9, "J_valid": .2, "J_test": .1},
                {"Epoch": 3, "J_valid": .2, "J_test": .9},
                {"Epoch": 5, "J_valid": .1, "J_test": 99},
            ]
        )
        self.assertEqual([row["Epoch"] for row in ranked], [5, 3, 9])

    def test_soup3_formula_and_cpu_fp64(self):
        result, selected = uniform_soup(entries(), 3, 1111)
        self.assertEqual([row["Epoch"] for row in selected], [5, 4, 3])
        self.assertEqual(result["weight"].device.type, "cpu")
        self.assertEqual(result["weight"].dtype, torch.float32)
        self.assertEqual(float(result["weight"]), 4.0)

    def test_soup5_formula(self):
        result, selected = uniform_soup(entries(), 5, 1111)
        self.assertEqual(len(selected), 5)
        self.assertEqual(float(result["weight"]), 3.0)

    def test_soup_rejects_nonfloating_difference(self):
        sources = entries()
        sources[-1]["state"]["counter"].fill_(1)
        with self.assertRaises(RuntimeError):
            uniform_soup(sources, 5, 1111)

    def test_soup_rejects_cross_seed(self):
        sources = entries()
        sources[-1]["Seed"] = 1112
        with self.assertRaises(RuntimeError):
            uniform_soup(sources, 5, 1111)

    def test_soup_rejects_ema_sources(self):
        with self.assertRaises(RuntimeError):
            uniform_soup(entries(source="EMA"), 5, 1111)

    def test_soup_rejects_unregistered_topk(self):
        with self.assertRaises(ValueError):
            uniform_soup(entries(), 4, 1111)

    def test_state_distance_validates_nonfloating_state(self):
        left, right = state(1, counter=0), state(2, counter=1)
        with self.assertRaises(RuntimeError):
            state_distance(left, right)

    def test_main_strategy_uses_mean_validation_only(self):
        frame = pd.DataFrame(
            [
                {"Seed": seed, "Method": method, "J_valid": value, "J_test": test}
                for seed in range(1111, 1116)
                for method, value, test in (
                    ("EMA", .2, 100),
                    ("Soup-3", .1, 200),
                    ("Soup-5", .3, -100),
                )
            ]
        )
        selected, _ = select_main_strategy(frame)
        self.assertEqual(selected, "Soup-3")

    def test_main_strategy_tie_priority(self):
        frame = pd.DataFrame(
            [
                {"Seed": seed, "Method": method, "J_valid": .1}
                for seed in range(1111, 1116)
                for method in ("EMA", "Soup-3", "Soup-5")
            ]
        )
        selected, _ = select_main_strategy(frame)
        self.assertEqual(selected, "EMA")

    def test_online_replay_checks_epoch_j_and_four_maes(self):
        current = {
            "BestValidEpoch": 9,
            "J_valid": .6,
            "J_test_at_valid_best": .7,
            **{
                "test_at_valid_best_{}_MAE".format(mode): .8
                for mode in ("LAV", "LA", "LV", "L")
            },
        }
        self.assertTrue(verify_online_replay(current, current)["Passed"])
        changed = dict(current)
        changed["test_at_valid_best_L_MAE"] += .001
        self.assertFalse(verify_online_replay(changed, current)["Passed"])

    def test_missing_sequence_sha_replays_stage3_generator(self):
        expected, count = expected_missing_sequence_sha(1111, 2, [4, 3])
        generator = torch.Generator().manual_seed(1111 + 104729)
        hasher = MissingSequenceHasher()
        names = ("LA", "LV", "L")
        for _ in range(2):
            for size in (4, 3):
                values = torch.randint(3, (size,), generator=generator)
                hasher.update([names[int(value)] for value in values])
        self.assertEqual(hasher.hexdigest(), expected)
        self.assertEqual(hasher.count, count)

    def test_fixed_cli_seed_order_and_formal_max_epoch_lock(self):
        with mock.patch.object(
            sys,
            "argv",
            ["x", "--seed", "1111", "--max-epochs", "2"],
        ), self.assertRaises(SystemExit):
            runner.parse_args()
        with mock.patch.object(
            sys, "argv", ["x", "--seed", "9999"]
        ), self.assertRaises(SystemExit):
            runner.parse_args()

    def test_frozen_online_loss_expression_is_stage3_sum(self):
        source = Path(runner.__file__).read_text()
        self.assertIn("total_loss = full_loss + missing_loss + kd_loss", source)
        self.assertIn('"compat"', source)
        self.assertNotIn("lambda_ema", source)

    def test_paired_deltas_are_recomputed_from_per_seed_rows(self):
        rows = []
        for seed in range(1111, 1116):
            for method, offset in (
                ("Online", 0.0),
                ("EMA", -.1),
                ("Soup-3", -.2),
                ("Soup-5", -.3),
            ):
                rows.append(
                    {
                        "Seed": seed,
                        "Method": method,
                        "BestValidEpoch": 1,
                        "J_valid": 1 + offset,
                        "J_test_at_valid_best": 2 + offset,
                    }
                )
        paired = aggregate.build_paired_deltas(pd.DataFrame(rows))
        soup5 = paired.loc[paired.Strategy.eq("Soup-5")]
        self.assertTrue(np.allclose(soup5.Delta_J_valid, -.3))

    def test_equal_weight_ensemble_is_diagnostic_only(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for seed in range(1111, 1116):
                target = aggregate.seed_directory(root, "mosi", seed)
                target.mkdir(parents=True)
                frame = pd.DataFrame(
                    {
                        "sample_index": [0, 1],
                        "sample_id": ["a", "b"],
                        "label": [0.0, 1.0],
                        **{
                            "{}_pred".format(mode): [float(seed), float(seed)]
                            for mode in ("LAV", "LA", "LV", "L")
                        },
                    }
                )
                frame.to_csv(target / "online_valid_predictions.csv", index=False)
            predictions, rows = aggregate.ensemble_split(root, "mosi", "valid")
            self.assertTrue(
                np.allclose(predictions.LAV_pred.to_numpy(), 1113.0)
            )
            self.assertTrue(all(row["DiagnosticOnly"] for row in rows))
            self.assertTrue(
                all(not row["ParticipatesInMainStrategySelection"] for row in rows)
            )


if __name__ == "__main__":
    unittest.main()
