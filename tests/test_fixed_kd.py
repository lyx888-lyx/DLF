import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn

import eval_missing
import train_fixed_kd
from config import get_config_regression
from trains.singleTask.fixed_kd_utils import (
    assert_initial_lav_equivalence,
    assert_student_state_dict_has_no_teacher,
    assert_teacher_not_in_optimizer,
    build_frozen_teacher,
    fixed_kd_checkpoint_path,
    freeze_teacher,
    prediction_kd_loss,
    teacher_grad_count,
    teacher_lav_prediction,
)
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    build_single_split_loader,
    mode_to_mask,
    sample_missing_masks,
)
from trains.singleTask.model.DLF import DLF


class TinyBackbone(nn.Module):
    def __init__(self, text_dim=2, audio_dim=2, vision_dim=2):
        super().__init__()
        self.out_layer = nn.Linear(text_dim + audio_dim + vision_dim, 1)

    def forward(self, text, audio, vision, fusion_residual=None):
        fusion = torch.cat([text.mean(dim=1), audio.mean(dim=1), vision.mean(dim=1)], dim=1)
        if fusion_residual is not None:
            fusion = fusion + fusion_residual
        return {"output_logit": self.out_layer(fusion)}


class FixedKDTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)

    def _batch(self):
        return (
            torch.randn(3, 4, 2),
            torch.randn(3, 5, 2),
            torch.randn(3, 6, 2),
        )

    def test_teacher_freeze_and_optimizer_exclusion(self):
        teacher = freeze_teacher(TinyBackbone())
        student = MissingModalityWrapper(TinyBackbone(), 2, 2)
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-4)
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))
        assert_teacher_not_in_optimizer(teacher, optimizer)

    def test_teacher_and_student_lav_match_at_shared_initialization(self):
        source = TinyBackbone()
        teacher = TinyBackbone()
        student_backbone = TinyBackbone()
        teacher.load_state_dict(source.state_dict(), strict=True)
        student_backbone.load_state_dict(source.state_dict(), strict=True)
        freeze_teacher(teacher)
        student = MissingModalityWrapper(student_backbone, 2, 2).eval()
        text, audio, vision = self._batch()
        assert_initial_lav_equivalence(teacher, student, text, audio, vision)

    def test_kd_uses_only_primary_output_and_has_expected_value(self):
        criterion = nn.SmoothL1Loss()
        student_prediction = torch.tensor([[0.2], [0.5]], requires_grad=True)
        teacher_prediction = torch.tensor([[0.2], [0.5]])
        equal_loss = prediction_kd_loss(
            criterion,
            {"output_logit": student_prediction, "logits_c": torch.tensor([[99.0]])},
            {"output_logit": teacher_prediction, "logits_c": torch.tensor([[88.0]])},
        )
        self.assertEqual(equal_loss.item(), 0.0)
        different_loss = prediction_kd_loss(
            criterion,
            {"output_logit": student_prediction + 1.0},
            {"output_logit": teacher_prediction},
        )
        self.assertGreater(different_loss.item(), 0.0)
        source = inspect.getsource(prediction_kd_loss)
        self.assertIn('["output_logit"]', source)
        for forbidden in ("logits_c", "hetero", "feature", "softmax", "temperature"):
            self.assertNotIn(forbidden, source)

    def test_kd_backward_updates_student_but_not_teacher(self):
        teacher = freeze_teacher(TinyBackbone())
        student = MissingModalityWrapper(TinyBackbone(), 2, 2)
        text, audio, vision = self._batch()
        with torch.inference_mode():
            teacher_output = teacher(text, audio, vision)
        missing_output = student(text, audio, vision, mode_to_mask("L", 3))
        loss = prediction_kd_loss(nn.SmoothL1Loss(), missing_output, teacher_output)
        gradients = torch.autograd.grad(
            loss,
            [parameter for parameter in student.parameters() if parameter.requires_grad],
            retain_graph=True,
            allow_unused=True,
        )
        self.assertGreater(
            sum(gradient.abs().sum().item() for gradient in gradients if gradient is not None),
            0.0,
        )
        loss.backward()
        self.assertEqual(teacher_grad_count(teacher), 0)
        self.assertGreater(
            sum(
                parameter.grad.abs().sum().item()
                for parameter in student.parameters()
                if parameter.grad is not None
            ),
            0.0,
        )

    def test_fixed_checkpoint_is_isolated_and_student_state_has_no_teacher(self):
        fixed = fixed_kd_checkpoint_path("pt", "mosi", 1111)
        self.assertEqual(str(fixed), "pt/missing_baseline/fixed_kd/DLF_mosi_seed1111_best.pth")
        self.assertNotEqual(str(fixed), "pt/DLF_mosi_seed1111_best.pth")
        student = MissingModalityWrapper(TinyBackbone(), 2, 2)
        assert_student_state_dict_has_no_teacher(student.state_dict())
        with self.assertRaises(RuntimeError):
            assert_student_state_dict_has_no_teacher({"teacher.out_layer.weight": torch.zeros(1)})

    def test_missing_sampling_sequence_matches_stage1_generator_contract(self):
        stage1_generator = torch.Generator().manual_seed(1111 + 104729)
        stage2_generator = torch.Generator().manual_seed(1111 + 104729)
        torch.testing.assert_close(
            sample_missing_masks(1284, stage1_generator),
            sample_missing_masks(1284, stage2_generator),
        )

    def test_cli_rejects_nonfixed_weights(self):
        with mock.patch.object(sys, "argv", ["train_fixed_kd.py", "--eta", "0.5"]):
            with self.assertRaises(SystemExit):
                train_fixed_kd.parse_args()
        with mock.patch.object(sys, "argv", ["train_fixed_kd.py", "--lambda-kd", "0.5"]):
            with self.assertRaises(SystemExit):
                train_fixed_kd.parse_args()

    def test_training_and_eval_protocols_are_student_only_and_test_guarded(self):
        training_source = Path("train_fixed_kd.py").read_text(encoding="utf-8")
        self.assertNotIn('dataloader["test"]', training_source)
        self.assertNotIn("dataloader['test']", training_source)
        self.assertIn("torch.inference_mode()", inspect.getsource(teacher_lav_prediction))
        self.assertIn("fixedkd", inspect.getsource(eval_missing.load_model))
        self.assertNotIn("teacher", inspect.getsource(eval_missing.load_model))
        self.assertIn("shuffle=False", inspect.getsource(build_single_split_loader))
        with mock.patch.object(
            sys,
            "argv",
            ["eval_missing.py", "--method", "fixedkd", "--split", "test"],
        ):
            with self.assertRaises(SystemExit):
                eval_missing.parse_args()

    def test_gate3_checkpoint_initializes_teacher_and_student_equally(self):
        checkpoint = Path("pt/DLF_mosi_seed1111_best.pth")
        dataset = Path("dataset/MOSI/Processed/aligned_50.pkl")
        if not checkpoint.is_file() or not dataset.is_file():
            self.skipTest("Gate 3 checkpoint or MOSI validation features are unavailable.")
        args = get_config_regression("DLF", "mosi", "config/config.json")
        args.feature_T = ""
        args.feature_A = ""
        args.feature_V = ""
        args.mode = "train"
        args.device = torch.device("cpu")
        teacher = build_frozen_teacher(DLF, args, checkpoint)
        student_backbone = DLF(args)
        student_backbone.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        student = MissingModalityWrapper(student_backbone, args.feature_dims[1], args.feature_dims[2]).eval()
        batch = next(iter(build_single_split_loader(args, "valid", num_workers=0)))
        assert_initial_lav_equivalence(
            teacher,
            student,
            batch["text"],
            batch["audio"],
            batch["vision"],
        )
        self.assertFalse(teacher.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in teacher.parameters()))


if __name__ == "__main__":
    unittest.main()
