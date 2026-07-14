import inspect
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from config import get_config_regression

from trains.singleTask.missing_utils import (
    MISSING_MODES,
    MissingModalityWrapper,
    apply_direct_mask,
    apply_moddrop_tokens,
    build_single_split_loader,
    clean_checkpoint_path,
    count_missing_modes,
    missing_checkpoint_path,
    mode_to_mask,
    sample_missing_masks,
)
from trains.singleTask.model.DLF import DLF


class TinyBackbone(nn.Module):
    def __init__(self, text_dim, audio_dim, vision_dim):
        super().__init__()
        self.out_layer = nn.Linear(text_dim + audio_dim + vision_dim, 1)

    def forward(self, text, audio, vision, fusion_residual=None):
        fusion = torch.cat([text.mean(dim=1), audio.mean(dim=1), vision.mean(dim=1)], dim=1)
        if fusion_residual is not None:
            fusion = fusion + fusion_residual
        output = self.out_layer(fusion)
        return {
            "output_logit": output,
            "logits_c": output,
            "logits_l_hetero": output,
            "logits_v_hetero": output,
            "logits_a_hetero": output,
        }


def small_dlf_args():
    return SimpleNamespace(
        use_bert=False,
        dataset_name="mosi",
        need_data_aligned=True,
        feature_dims=[5, 3, 6],
        dst_feature_dim_nheads=[4, 1],
        nlevels=1,
        attn_dropout=0.0,
        attn_dropout_a=0.0,
        attn_dropout_v=0.0,
        relu_dropout=0.0,
        embed_dropout=0.0,
        res_dropout=0.0,
        output_dropout=0.0,
        text_dropout=0.0,
        attn_mask=True,
        conv1d_kernel_size_l=3,
        conv1d_kernel_size_a=3,
        conv1d_kernel_size_v=3,
    )


class MissingModalityTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)

    def test_modality_mask_mapping(self):
        self.assertEqual(mode_to_mask("LAV").tolist(), [1.0, 1.0, 1.0])
        self.assertEqual(mode_to_mask("LA").tolist(), [1.0, 1.0, 0.0])
        self.assertEqual(mode_to_mask("LV").tolist(), [1.0, 0.0, 1.0])
        self.assertEqual(mode_to_mask("L").tolist(), [1.0, 0.0, 0.0])

    def test_direct_mask_zeroes_only_missing_modalities(self):
        audio = torch.arange(24, dtype=torch.float32).view(2, 3, 4)
        vision = torch.arange(30, dtype=torch.float32).view(2, 3, 5)
        la_audio, la_vision = apply_direct_mask(audio, vision, mode_to_mask("LA", 2))
        lv_audio, lv_vision = apply_direct_mask(audio, vision, mode_to_mask("LV", 2))
        l_audio, l_vision = apply_direct_mask(audio, vision, mode_to_mask("L", 2))
        torch.testing.assert_close(la_audio, audio)
        self.assertTrue(torch.count_nonzero(la_vision) == 0)
        self.assertTrue(torch.count_nonzero(lv_audio) == 0)
        torch.testing.assert_close(lv_vision, vision)
        self.assertTrue(torch.count_nonzero(l_audio) == 0)
        self.assertTrue(torch.count_nonzero(l_vision) == 0)

    def test_moddrop_tokens_broadcast_and_receive_gradients(self):
        audio = torch.randn(3, 4, 2)
        vision = torch.randn(3, 5, 3)
        audio_token = nn.Parameter(torch.zeros(1, 1, 2))
        vision_token = nn.Parameter(torch.zeros(1, 1, 3))
        masked_audio, masked_vision = apply_moddrop_tokens(
            audio, vision, mode_to_mask("L", 3), audio_token, vision_token
        )
        self.assertEqual(masked_audio.shape, audio.shape)
        self.assertEqual(masked_vision.shape, vision.shape)
        (masked_audio.sum() + masked_vision.sum()).backward()
        self.assertGreater(audio_token.grad.abs().sum().item(), 0.0)
        self.assertGreater(vision_token.grad.abs().sum().item(), 0.0)

    def test_mask_adapter_is_exactly_zero_for_lav(self):
        wrapper = MissingModalityWrapper(TinyBackbone(2, 3, 4), 3, 4)
        residual = wrapper.mask_adapter(1.0 - mode_to_mask("LAV", 5))
        self.assertTrue(torch.equal(residual, torch.zeros_like(residual)))

    def test_lav_wrapper_matches_original_dlf(self):
        model = DLF(small_dlf_args()).eval()
        wrapper = MissingModalityWrapper(model, 3, 6).eval()
        text = torch.randn(2, 50, 5)
        audio = torch.randn(2, 50, 3)
        vision = torch.randn(2, 50, 6)
        original_output = model(text, audio, vision)["output_logit"]
        wrapped_output = wrapper(text, audio, vision, mode_to_mask("LAV", 2))["output_logit"]
        torch.testing.assert_close(original_output, wrapped_output, rtol=1e-5, atol=1e-6)

    def test_per_sample_sampling_is_limited_and_near_uniform(self):
        generator = torch.Generator().manual_seed(1111)
        masks = sample_missing_masks(12000, generator)
        counts = count_missing_modes(masks)
        self.assertEqual(sum(counts.values()), 12000)
        self.assertEqual(set(counts), set(MISSING_MODES))
        for count in counts.values():
            self.assertLess(abs(count / 12000.0 - 1.0 / 3.0), 0.03)

    def test_training_and_validation_loader_protocol_is_static(self):
        training_source = Path("train_missing.py").read_text(encoding="utf-8")
        self.assertNotIn('dataloader["test"]', training_source)
        self.assertNotIn("dataloader['test']", training_source)
        loader_source = inspect.getsource(build_single_split_loader)
        self.assertIn("shuffle=False", loader_source)

    def test_missing_checkpoint_is_isolated_from_gate3(self):
        clean = clean_checkpoint_path("pt", "mosi", 1111)
        missing = missing_checkpoint_path("pt", "mosi", 1111)
        self.assertEqual(str(clean), "pt/DLF_mosi_seed1111_best.pth")
        self.assertEqual(str(missing), "pt/missing_baseline/moddrop/DLF_mosi_seed1111_best.pth")
        self.assertNotEqual(clean, missing)

    def test_gate3_checkpoint_lav_wrapper_matches_base_on_valid_batch(self):
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
        base = DLF(args).eval()
        base.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        wrapper = MissingModalityWrapper(base, args.feature_dims[1], args.feature_dims[2]).eval()
        batch = next(iter(build_single_split_loader(args, "valid", num_workers=0)))
        with torch.no_grad():
            original_output = base(batch["text"], batch["audio"], batch["vision"])["output_logit"]
            wrapped_output = wrapper(
                batch["text"],
                batch["audio"],
                batch["vision"],
                mode_to_mask("LAV", batch["text"].size(0)),
            )["output_logit"]
        torch.testing.assert_close(original_output, wrapped_output, rtol=1e-5, atol=1e-6)



if __name__ == "__main__":
    unittest.main()
