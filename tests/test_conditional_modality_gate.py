import inspect
import math
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import train_conditional_modality_gate as train
from trains.singleTask.conditional_modality_gate_utils import (
    ConditionalModalityGateWrapper, build_utility_groups,
    first_epoch_counts_ok, load_stage7a_derangements, matched_shuffle_loss,
    qualification_report, reliable_target_table, summarize_gate_samples,
    targets_for_indices, utility_bce,
)
from trains.singleTask.missing_utils import MissingModalityWrapper, mode_to_mask


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = 3, 2, 2
        self.d_l = self.d_a = self.d_v = 2
        self.proj_l = nn.Conv1d(3, 2, 1, bias=False)
        self.proj_a = nn.Conv1d(2, 2, 1, bias=False)
        self.proj_v = nn.Conv1d(2, 2, 1, bias=False)
        self.out_layer = nn.Linear(6, 1)

    def forward(self, text, audio, vision, fusion_residual=None):
        l = self.proj_l(text.transpose(1, 2)).mean(2)
        a = self.proj_a(audio.transpose(1, 2)).mean(2)
        v = self.proj_v(vision.transpose(1, 2)).mean(2)
        fused = torch.cat([l, a, v], dim=1)
        if fusion_residual is not None:
            fused = fused + fusion_residual
        return {
            "output_logit": self.out_layer(fused),
            "projected_a": a, "projected_v": v,
        }


def fake_inputs(count=3):
    return (
        torch.randn(count, 4, 3),
        torch.randn(count, 4, 2),
        torch.randn(count, 4, 2),
    )


def make_stage7a_fixture(directory):
    gain_rows, shuffle_rows = [], []
    signs = {
        0: (1., 1., 1., 1.),
        1: (-1., -1., -1., -1.),
        2: (1., -1., 1., -1.),
    }
    for split in ("train", "valid"):
        for modality in ("A", "V"):
            for index, values in signs.items():
                for state, offset in (("gate3_init", 0), ("cfcompat_best_valid", 2)):
                    gain_rows.append({
                        "State": state, "Split": split, "SampleIndex": index,
                        "SampleID": "{}{}".format(split, index), "Label": 0.,
                        "Modality": modality, "Gain": values[offset],
                    })
                for repeat in range(10):
                    shuffle_rows.append({
                        "State": "gate3_init", "Split": split, "Repeat": repeat,
                        "SampleIndex": index, "SampleID": "{}{}".format(split, index),
                        "Label": 0., "ShuffleDamage_A": values[1],
                        "ShuffleDamage_V": values[1],
                    })
                    shuffle_rows.append({
                        "State": "cfcompat_best_valid", "Split": split, "Repeat": repeat,
                        "SampleIndex": index, "SampleID": "{}{}".format(split, index),
                        "Label": 0., "ShuffleDamage_A": values[3],
                        "ShuffleDamage_V": values[3],
                    })
    pd.DataFrame(gain_rows).to_csv(directory / "sample_modality_gains.csv", index=False)
    pd.DataFrame(shuffle_rows).to_csv(directory / "shuffle_sample_metrics.csv", index=False)


class ConditionalModalityGateTests(unittest.TestCase):
    def test_stable_utility_label_formula_and_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_stage7a_fixture(directory)
            frame = build_utility_groups(directory)
            groups = frame[
                frame.Modality.eq("A") & frame.Split.eq("train")
            ].sort_values("SampleIndex").UtilityGroup.tolist()
            self.assertEqual(groups, ["positive", "negative", "ambiguous"])

    def test_ambiguous_target_is_nan_and_excluded(self):
        probability = torch.tensor([.8, .2, .9], requires_grad=True)
        target = torch.tensor([1., 0., -1.])
        loss, count = utility_bce(probability, target, torch.ones(3, dtype=torch.bool))
        self.assertEqual(count, 2)
        expected = nn.functional.binary_cross_entropy(probability[:2], target[:2])
        torch.testing.assert_close(loss, expected)

    def test_utility_only_when_modality_present(self):
        probability = torch.tensor([.8, .2], requires_grad=True)
        target = torch.tensor([1., 0.])
        loss, count = utility_bce(probability, target, torch.tensor([True, False]))
        self.assertEqual(count, 1)
        torch.testing.assert_close(loss, -torch.log(probability[0]))

    def test_valid_targets_cannot_enter_training_table(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            make_stage7a_fixture(directory)
            groups = build_utility_groups(directory)
            with self.assertRaises(ValueError):
                reliable_target_table(groups, "valid")
            table = reliable_target_table(groups, "train")
            self.assertIsNone(table["A"][2])

    def test_targets_lookup_keeps_ambiguous_sentinel(self):
        table = {"A": {0: 1, 1: 0, 2: None}}
        got = targets_for_indices(table, [0, 1, 2], "A", "cpu", torch.float32)
        torch.testing.assert_close(got, torch.tensor([1., 0., -1.]))

    def test_qualification_thresholds(self):
        rows = []
        for modality, counts in (("A", (32, 32, 8, 8)), ("V", (31, 40, 9, 9))):
            for split, positive, negative in (
                ("train", counts[0], counts[1]), ("valid", counts[2], counts[3])
            ):
                for group, count in (("positive", positive), ("negative", negative)):
                    rows.extend({
                        "Modality": modality, "Split": split, "UtilityGroup": group
                    } for _ in range(count))
        report = qualification_report(pd.DataFrame(rows))
        self.assertTrue(report["A"]["qualified"])
        self.assertFalse(report["V"]["qualified"])

    def test_zero_initialization_q_half_g_one(self):
        model = ConditionalModalityGateWrapper(FakeBackbone(), 2, 2, ("A", "V"))
        text, audio, vision = fake_inputs()
        output = model(text, audio, vision, mode_to_mask("LAV", 3))
        torch.testing.assert_close(output["utility_q_A"], torch.full((3,), .5))
        torch.testing.assert_close(output["utility_q_V"], torch.full((3,), .5))
        torch.testing.assert_close(output["utility_g_A"], torch.ones(3))
        torch.testing.assert_close(output["utility_g_V"], torch.ones(3))

    def test_identity_replay_exact_all_modes(self):
        backbone = FakeBackbone()
        reference = MissingModalityWrapper(backbone, 2, 2)
        gated = ConditionalModalityGateWrapper(FakeBackbone(), 2, 2, ())
        gated.load_state_dict(reference.state_dict(), strict=True)
        text, audio, vision = fake_inputs()
        for mode in ("LAV", "LA", "LV", "L"):
            mask = mode_to_mask(mode, 3)
            torch.testing.assert_close(
                gated(text, audio, vision, mask)["output_logit"],
                reference(text, audio, vision, mask)["output_logit"],
                rtol=0, atol=0,
            )

    def test_missing_token_is_not_scaled(self):
        model = ConditionalModalityGateWrapper(FakeBackbone(), 2, 2, ("A",))
        model.audio_utility_gate.bias.data.fill_(math.log(3.))
        text, audio, vision = fake_inputs()
        output = model(text, audio, vision, mode_to_mask("LV", 3))
        torch.testing.assert_close(output["utility_g_A"], torch.ones(3))

    def test_four_mode_gate_rules(self):
        model = ConditionalModalityGateWrapper(FakeBackbone(), 2, 2, ("A", "V"))
        model.audio_utility_gate.bias.data.fill_(math.log(3.))
        model.vision_utility_gate.bias.data.fill_(math.log(3.))
        text, audio, vision = fake_inputs()
        expected = {
            "LAV": (1.5, 1.5), "LA": (1.5, 1.), "LV": (1., 1.5), "L": (1., 1.),
        }
        for mode, values in expected.items():
            output = model(text, audio, vision, mode_to_mask(mode, 3))
            torch.testing.assert_close(output["utility_g_A"], torch.full((3,), values[0]))
            torch.testing.assert_close(output["utility_g_V"], torch.full((3,), values[1]))

    def test_unqualified_modality_has_no_trainable_gate(self):
        model = ConditionalModalityGateWrapper(FakeBackbone(), 2, 2, ("A",))
        self.assertIsNotNone(model.audio_utility_gate)
        self.assertIsNone(model.vision_utility_gate)
        self.assertFalse(any("vision_utility_gate" in name for name, _ in model.named_parameters()))

    def test_matched_loss_positive_only_and_no_margin(self):
        matched = torch.tensor([2., 2., 2.])
        shuffled = torch.tensor([0., 3., 0.])
        labels = torch.zeros(3)
        loss, me, se, count = matched_shuffle_loss(
            matched, shuffled, labels, torch.tensor([True, False, False])
        )
        self.assertEqual(count, 1)
        torch.testing.assert_close(loss, torch.relu(me[0] - se[0]))
        self.assertNotIn("margin", inspect.getsource(matched_shuffle_loss).lower())

    def test_stage7a_derangements_are_reused_and_validated(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = []
            for split in ("train", "valid"):
                for repeat in range(10):
                    mapping = [1, 2, 0]
                    for source, target in enumerate(mapping):
                        rows.append({
                            "Split": split, "Repeat": repeat, "SourceIndex": source,
                            "TargetIndex": target, "AudioTargetIndex": target,
                            "VisionTargetIndex": target, "CrossSplit": False,
                        })
            pd.DataFrame(rows).to_csv(Path(tmp) / "shuffle_manifest.csv", index=False)
            got = load_stage7a_derangements(tmp)
            np.testing.assert_array_equal(got["train"][0], [1, 2, 0])

    def test_epoch1_missing_counts_lock(self):
        self.assertTrue(first_epoch_counts_ok(Counter({"LA": 435, "LV": 430, "L": 419})))
        self.assertFalse(first_epoch_counts_ok(Counter({"LA": 434, "LV": 431, "L": 419})))

    def test_gate_summary_has_auroc_gap_and_saturation(self):
        frame = pd.DataFrame([
            {"Epoch": 1, "Split": "valid", "Modality": "A", "q": .8, "g": 1.6,
             "UtilityGroup": "positive", "UtilityTarget": 1.},
            {"Epoch": 1, "Split": "valid", "Modality": "A", "q": .2, "g": .4,
             "UtilityGroup": "negative", "UtilityTarget": 0.},
        ])
        summary = summarize_gate_samples(frame).iloc[0]
        self.assertEqual(summary.UtilityAUROC, 1.)
        self.assertGreater(summary.PositiveNegativeGateGap, 0)
        self.assertIn("q_low_saturation_fraction", summary.index)

    def test_cli_locks_seed_workers_and_variant(self):
        for argv in (
            ["x", "--seed", "2", "--build-utility-groups-only"],
            ["x", "--num-workers", "1", "--build-utility-groups-only"],
            ["x"],
        ):
            with mock.patch.object(sys, "argv", argv):
                with self.assertRaises(SystemExit):
                    train.parse_args()

    def test_source_checkpoint_selected_only_by_valid_j(self):
        source = Path("train_conditional_modality_gate.py").read_text()
        self.assertIn("is_best_valid = j_valid <=", source)
        self.assertIn("torch.save(student.state_dict(), main_checkpoint)", source)
        self.assertNotIn("torch.save(student.state_dict(), main_checkpoint) if is_best_test", source)

    def test_student_only_eval_and_test_not_utility_target(self):
        source = Path("train_conditional_modality_gate.py").read_text()
        self.assertIn('"StudentOnlyEval": True', source)
        self.assertIn('"TestUsedForUtilityTargets": False', source)
        self.assertIn('reliable_target_table(groups, "train")', source)
        self.assertNotIn('reliable_target_table(groups, "test")', source)

    def test_total_loss_is_fixed_sum(self):
        source = Path("train_conditional_modality_gate.py").read_text()
        self.assertIn(
            "full_loss + missing_loss + kd_loss + utility_loss + match_loss", source
        )
        self.assertNotIn("--lambda", source)
        self.assertNotIn("--temperature", source)

    def test_gate_is_prefusion_not_prediction_scalar(self):
        source = inspect.getsource(ConditionalModalityGateWrapper)
        self.assertIn("backbone.proj_a.register_forward_hook", source)
        self.assertIn("backbone.proj_v.register_forward_hook", source)
        self.assertNotIn("output_logit", inspect.getsource(
            ConditionalModalityGateWrapper._gate_projection
        ))

    def test_no_prohibited_model_features(self):
        source = Path("train_conditional_modality_gate.py").read_text().lower()
        for token in ("lora", "gradient surgery", "contrastive", "recoverability"):
            self.assertNotIn(token, source)

    def test_required_outputs_declared(self):
        source = Path("train_conditional_modality_gate.py").read_text()
        for token in (
            "mosi", "_per_seed.csv", "_epoch_metrics.csv", "_gate_summary.csv",
            "_gate_samples_train.csv", "_gate_samples_valid.csv",
            "_utility_groups.csv", "_matched_shuffle_summary.csv",
            "_best_valid_predictions.csv",
        ):
            self.assertIn(token, source)
        self.assertIn("_valid_modality_utility.csv", Path(
            "eval_conditional_modality_gate.py"
        ).read_text())

    def test_protocol_freezes_base_and_replay_gate(self):
        text = Path("CONDITIONAL_MODALITY_GATE_PROTOCOL.md").read_text()
        self.assertIn("1e5ab2430307576127ad0845fde32b433a0da744", text)
        self.assertIn("Identity Replay", text)


if __name__ == "__main__":
    unittest.main()
