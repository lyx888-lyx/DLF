import json
import random
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import eval_cf_residual_recovery as evaluator_entry
import train_cf_residual_recovery as trainer
from trains.singleTask.cf_residual_utils import (
    CFRR_CACHE_VERSION,
    MODE_ORDER,
    CounterfactualResidualStudent,
    build_residual_cache,
    complementary_residual_loss,
    derive_residual_frame,
    load_residual_cache,
    modes_from_any_mask,
    prediction_rows,
    residual_cache_paths,
    residual_diagnostic_rows,
    residual_targets_for_modes,
    rng_states_equal,
    signed_distribution,
)
from trains.singleTask.fixed_kd_utils import capture_rng_state, checkpoint_sha256
from trains.singleTask.missing_utils import MissingModalityWrapper, mode_to_mask


class TinyBackbone(nn.Module):
    def __init__(self, fusion_dim=4):
        super().__init__()
        self.out_layer = nn.Linear(fusion_dim, 1)

    def forward(self, text, audio, vision, fusion_residual=None):
        feature = text[:, 0, :self.out_layer.in_features]
        if fusion_residual is not None:
            feature = feature + fusion_residual
        output = self.out_layer(feature)
        zeros = torch.zeros_like(output)
        return {
            "fusion_feature": feature, "output_logit": output,
            "logits_c": zeros, "logits_l_hetero": zeros,
            "logits_v_hetero": zeros, "logits_a_hetero": zeros,
        }


def tiny_student(scales=None):
    backbone = TinyBackbone()
    base = MissingModalityWrapper(backbone, audio_dim=2, vision_dim=3)
    return CounterfactualResidualStudent(base, scales or {"LA": .1, "LV": .2, "L": .3})


def source_cache_frame(count=1284):
    index = np.arange(count)
    lav = np.sin(index / 17.0) * .2
    residuals = {
        "LA": ((index % 11) - 5) * .003,
        "LV": ((index % 13) - 6) * .002,
        "L": ((index % 17) - 8) * .0015,
    }
    frame = pd.DataFrame({"sample_index": index, "sample_id": ["id{}".format(i) for i in index],
                          "label": np.cos(index / 19.0)})
    frame["evaluator_LAV_pred"] = lav
    for mode in MODE_ORDER:
        frame["evaluator_{}_pred".format(mode)] = lav - residuals[mode]
        frame["delta_{}".format(mode)] = np.abs(residuals[mode])
        frame["compat_{}".format(mode)] = (count - index - .5) / count
    return frame


def install_stage3_fixture(root, corrupt_sha=False):
    root = Path(root)
    source = root / "counterfactual_compatibility/cf_compat_v1/mosi"
    source.mkdir(parents=True)
    csv = source / "train_counterfactual_compatibility.csv"
    source_cache_frame().to_csv(csv, index=False)
    evaluator = root / "evaluator_valid.pth"; evaluator.write_bytes(b"valid evaluator")
    config = {"source": "train_only", "train_sample_count": 1284,
              "evaluator_sha256": checkpoint_sha256(evaluator)}
    (source / "cf_compat_config.json").write_text(json.dumps(config))
    manifest = root / "missing_baseline/cf_compat_kd_v1/benchmark_multiseed/checkpoint_manifest.csv"
    manifest.parent.mkdir(parents=True)
    pd.DataFrame([{
        "Seed": 1111, "Status": "locked_existing", "CachePath": str(csv),
        "CacheSHA256": "bad" if corrupt_sha else checkpoint_sha256(csv),
        "EvaluatorCheckpoint": str(evaluator), "EvaluatorSHA256": checkpoint_sha256(evaluator),
    }]).to_csv(manifest, index=False)
    return csv, evaluator


class ResidualRecoveryTests(unittest.TestCase):
    def test_signed_residual_definitions_and_absolute_delta(self):
        source = source_cache_frame()
        result = derive_residual_frame(source)
        for mode in MODE_ORDER:
            expected = source.evaluator_LAV_pred - source["evaluator_{}_pred".format(mode)]
            np.testing.assert_allclose(result["residual_{}".format(mode)], expected, atol=1e-12)
            np.testing.assert_allclose(result["residual_{}".format(mode)].abs(), source["delta_{}".format(mode)], atol=1e-12)
            self.assertTrue((result["residual_{}".format(mode)] > 0).any())
            self.assertTrue((result["residual_{}".format(mode)] < 0).any())

    def test_residual_cache_is_exact_train_index_set(self):
        result = derive_residual_frame(source_cache_frame())
        self.assertEqual(len(result), 1284)
        self.assertEqual(result.sample_index.nunique(), 1284)
        np.testing.assert_array_equal(result.sample_index, np.arange(1284))
        with self.assertRaises(ValueError):
            derive_residual_frame(source_cache_frame(1283))

    def test_abs_residual_mismatch_is_rejected(self):
        source = source_cache_frame(); source.loc[0, "delta_LA"] += 1e-4
        with self.assertRaises(ValueError):
            derive_residual_frame(source)

    def test_cache_manifest_sha_binding_and_population_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            install_stage3_fixture(tmp)
            before = capture_rng_state()
            paths, config, frame = build_residual_cache(tmp)
            after = capture_rng_state()
            self.assertTrue(rng_states_equal(before, after))
            self.assertEqual(config["TrainSampleCount"], 1284)
            self.assertTrue(config["CreatedFromTrainOnly"])
            self.assertEqual(config["ScaleFormula"], "population_std(residual_m,ddof=0)")
            for mode in MODE_ORDER:
                self.assertAlmostEqual(config["ModeScales"][mode], frame["residual_{}".format(mode)].std(ddof=0), places=14)
                self.assertGreater(config["ModeScales"][mode], 1e-8)
            self.assertEqual(config["ResidualCacheSHA256"], checkpoint_sha256(paths["csv"]))
            loaded, by_index, scales, loaded_config = load_residual_cache(tmp)
            self.assertEqual(len(loaded), 1284); self.assertEqual(len(by_index), 1284)
            self.assertEqual(scales, config["ModeScales"]); self.assertEqual(loaded_config["ConfigSHA256"], config["ConfigSHA256"])

    def test_stage3_source_sha_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            install_stage3_fixture(tmp, corrupt_sha=True)
            with self.assertRaises(ValueError):
                build_residual_cache(tmp)

    def test_cache_builder_source_contains_no_split_loader(self):
        source = Path("trains/singleTask/cf_residual_utils.py").read_text()
        body = source[source.index("def build_residual_cache"):source.index("def load_residual_cache")]
        self.assertNotIn("valid", body.lower()); self.assertNotIn("test", body.lower())
        self.assertIn("SourceCompatibilityCacheSHA256", body)

    def test_signed_distribution_uses_population_std_and_fractions(self):
        stats = signed_distribution([-2., 0., 1., 3.])
        self.assertAlmostEqual(stats["std"], np.std([-2., 0., 1., 3.], ddof=0))
        self.assertEqual(stats["positive_fraction"], .5)
        self.assertEqual(stats["negative_fraction"], .25)
        self.assertEqual(stats["zero_fraction"], .25)

    def test_residual_heads_are_independent_linear_zero_initialized(self):
        student = tiny_student()
        self.assertEqual(set(student.residual_heads), set(MODE_ORDER))
        self.assertEqual(len({id(head) for head in student.residual_heads.values()}), 3)
        for head in student.residual_heads.values():
            self.assertIsInstance(head, nn.Linear)
            self.assertEqual(head.out_features, 1)
            self.assertEqual(torch.count_nonzero(head.weight), 0)
            self.assertEqual(torch.count_nonzero(head.bias), 0)

    def test_residual_head_creation_preserves_all_rng_states(self):
        random.seed(7); np.random.seed(7); torch.manual_seed(7)
        backbone = TinyBackbone(); base = MissingModalityWrapper(backbone, 2, 3)
        before = capture_rng_state(); CounterfactualResidualStudent(base, {"LA": .1, "LV": .2, "L": .3})
        after = capture_rng_state(); self.assertTrue(rng_states_equal(before, after))

    def test_invalid_scales_are_rejected(self):
        for value in (0., 1e-9, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                tiny_student({"LA": value, "LV": .2, "L": .3})

    def test_zero_init_corrected_equals_base_and_lav_residual_zero(self):
        student = tiny_student(); text = torch.randn(4, 1, 4); audio = torch.randn(4, 2, 2); vision = torch.randn(4, 2, 3)
        for mode in ("LAV",) + MODE_ORDER:
            output = student(text, audio, vision, mode_to_mask(mode, 4))
            torch.testing.assert_close(output["output_logit"], output["base_output_logit"], rtol=0, atol=0)
            self.assertEqual(torch.count_nonzero(output["predicted_residual"]), 0)
        self.assertEqual(output["fusion_feature"].shape[-1], 4)

    def test_mixed_mode_batch_routes_corresponding_heads_and_scales(self):
        student = tiny_student()
        with torch.no_grad():
            student.residual_heads["LA"].bias.fill_(1.)
            student.residual_heads["LV"].bias.fill_(2.)
            student.residual_heads["L"].bias.fill_(3.)
        text = torch.randn(4, 1, 4); audio = torch.randn(4, 2, 2); vision = torch.randn(4, 2, 3)
        mask = torch.tensor([[1,1,1],[1,1,0],[1,0,1],[1,0,0]], dtype=torch.float32)
        output = student(text, audio, vision, mask)
        torch.testing.assert_close(output["predicted_residual_z"].view(-1), torch.tensor([0.,1.,2.,3.]))
        torch.testing.assert_close(output["predicted_residual"].view(-1), torch.tensor([0.,.1,.4,.9]))
        self.assertEqual(output["residual_mode"], ("LAV", "LA", "LV", "L"))

    def test_mask_decoder_rejects_unknown_patterns(self):
        with self.assertRaises(ValueError):
            modes_from_any_mask(torch.tensor([[0., 1., 1.]]))

    def test_residual_target_binding_is_signed_and_mode_scaled(self):
        cache = {0: {"compat_LA": .2, "residual_LA": -.1}, 1: {"compat_L": .8, "residual_L": .6}}
        compatibility, raw, z = residual_targets_for_modes(cache, [0,1], ["LA","L"], {"LA": .1,"LV":.2,"L":.3}, "cpu", torch.float32)
        torch.testing.assert_close(compatibility, torch.tensor([.2,.8]))
        torch.testing.assert_close(raw, torch.tensor([-.1,.6]))
        torch.testing.assert_close(z, torch.tensor([-1.,2.]))
        self.assertFalse(compatibility.requires_grad); self.assertFalse(z.requires_grad)

    def test_complementary_loss_exact_formula_and_detachment(self):
        prediction = torch.tensor([0., .5], requires_grad=True)
        target = torch.tensor([1., -1.], requires_grad=True)
        compatibility = torch.tensor([.25, .75], requires_grad=True)
        total, each, weight = complementary_residual_loss(prediction, target, compatibility)
        expected = ((1-.25)*each[0] + (1-.75)*each[1]) / ((1-.25)+(1-.75)+1e-8)
        torch.testing.assert_close(total, expected)
        total.backward()
        self.assertIsNotNone(prediction.grad); self.assertIsNone(target.grad); self.assertIsNone(compatibility.grad)
        torch.testing.assert_close(weight, torch.tensor([.75,.25]))

    def test_equal_residual_weights_reduce_to_mean(self):
        prediction = torch.tensor([0., 2., -1.]); target = torch.tensor([1., 1., 1.]); compat = torch.full((3,), .4)
        total, each, _ = complementary_residual_loss(prediction, target, compat)
        torch.testing.assert_close(total, each.mean(), rtol=1e-6, atol=1e-7)

    def test_residual_heads_receive_gradient(self):
        student = tiny_student(); text = torch.randn(3,1,4); audio = torch.randn(3,2,2); vision = torch.randn(3,2,3)
        output = student(text,audio,vision,torch.tensor([[1,1,0],[1,0,1],[1,0,0]],dtype=torch.float32))
        loss, _, _ = complementary_residual_loss(output["predicted_residual_z"], torch.tensor([1.,-1.,.5]), torch.tensor([.2,.3,.4]))
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in student.residual_heads.parameters() if p.grad is not None), 0)

    def test_checkpoint_schema_contains_heads_scales_not_frozen_models(self):
        keys = set(tiny_student().state_dict())
        self.assertIn("mode_scales", keys)
        for mode in MODE_ORDER:
            self.assertIn("residual_heads.{}.weight".format(mode), keys)
        self.assertFalse(any(key.startswith(("teacher.", "evaluator.")) for key in keys))

    def test_residual_diagnostics_include_modes_and_quartiles(self):
        records=[]
        for i in range(12):
            mode=MODE_ORDER[i%3]; target=(-1)**i*.1*(i+1); pred=target*.5; base=.2; label=.1
            records.append({"sample_index":i,"mode":mode,"compatibility":(i+.5)/12,
                            "target_residual":target,"predicted_residual":pred,
                            "base_prediction":base,"corrected_prediction":base+pred,"label":label})
        summary,modes,quartiles=residual_diagnostic_rows(records,1111,1,"x")
        self.assertEqual(summary["SampleCount"],12); self.assertEqual({row["Mode"] for row in modes},set(MODE_ORDER))
        self.assertEqual([row["Quartile"] for row in quartiles],["Q1_low","Q2","Q3","Q4_high"])
        for row in modes:
            for key in ("ResidualMAE","ResidualRMSE","Pearson","Spearman","SignAccuracy","ExplainedVariance",
                        "BaseLabelMAE","CorrectedLabelMAE","MeanErrorReduction","fraction_corrected_better","fraction_corrected_worse"):
                self.assertIn(key,row)

    def test_cli_locks_seed_methods_gates_and_weights(self):
        invalid = [
            ["--method","cfrr_only","--seeds","1112"],
            ["--method","cfrr_only","--gate-mode","reliability"],
            ["--method","cfrr_only","--eta",".5"],
            ["--method","cfrr_only","--lambda-kd",".5"],
            ["--method","cfrr_only","--lambda-residual",".5"],
        ]
        for argv in invalid:
            with self.assertRaises(SystemExit): trainer.parse_args(argv)
        self.assertEqual(trainer.parse_args(["--method","cfcompat_cfrr"]).method,"cfcompat_cfrr")

    def test_cfrr_only_and_combination_loss_semantics_are_explicit(self):
        source = Path("train_cf_residual_recovery.py").read_text()
        self.assertIn('gated_kd_loss(missing_output["base_output_logit"]', source)
        self.assertNotIn('gated_kd_loss(missing_output["output_logit"]', source)
        self.assertIn('direct_loss = torch.zeros', source)
        self.assertIn('full_loss + missing_loss + direct_loss + residual_loss', source)
        self.assertIn('1.0 - compatibility.detach()', Path("trains/singleTask/cf_residual_utils.py").read_text())

    def test_evaluation_entry_rejects_unconfirmed_test(self):
        with self.assertRaises(SystemExit): evaluator_entry.parse_args(["--method","cfrr_only","--split","test"])

    def test_evaluation_source_is_student_only(self):
        source = Path("eval_cf_residual_recovery.py").read_text()
        self.assertNotIn("build_frozen_teacher", source)
        self.assertNotIn("build_frozen_evaluator", source)
        self.assertNotIn("load_residual_cache", source)
        self.assertIn("load_residual_student_for_eval", source)

    def test_training_records_required_checkpoint_selection_and_counts(self):
        source = Path("train_cf_residual_recovery.py").read_text()
        for token in ('j_valid <= best_valid_j - 1e-6','j_test <= best_test_j - 1e-6',
                      'main_checkpoint','diagnostic_checkpoint','LA=435 LV=430 L=419',
                      'evaluate_residual_modes(student, loaders["valid"]','evaluate_residual_modes(student, test_loader'):
            self.assertIn(token, source)

    def test_initial_equivalence_is_checked_with_student_in_eval_mode(self):
        source = Path("train_cf_residual_recovery.py").read_text()
        start = source.index("def initialize_models")
        end = source.index("def _formal_output_guard")
        body = source[start:end]
        self.assertLess(body.index("student.eval()"), body.index("torch.testing.assert_close"))

    def test_required_result_files_are_registered(self):
        source = Path("train_cf_residual_recovery.py").read_text()
        for name in ("mosi_per_seed.csv","mosi_summary.csv","mosi_epoch_metrics.csv","mosi_residual_summary.csv",
                     "mosi_residual_mode_metrics.csv","mosi_residual_quartiles.csv","mosi_best_valid_predictions.csv",
                     "mosi_best_test_diagnostic_predictions.csv"):
            if name == "mosi_per_seed.csv":
                self.assertIn('"{}_per_seed.csv"', Path("trains/singleTask/missing_utils.py").read_text())
            elif name == "mosi_summary.csv":
                self.assertIn('"{}_summary.csv"', Path("trains/singleTask/missing_utils.py").read_text())
            else:
                self.assertIn(name, source)

    def test_dlf_exposes_exact_final_fusion_feature_without_parameters(self):
        source = Path("trains/singleTask/model/DLF.py").read_text()
        self.assertIn("'fusion_feature': last_hs_proj", source)
        self.assertIn("output = self.out_layer(last_hs_proj)", source)

    def test_existing_five_head_task_loss_formula_is_unchanged(self):
        source = Path("trains/singleTask/missing_utils.py").read_text()
        self.assertIn('criterion(output["output_logit"], labels)', source)
        self.assertIn('3.0 * components["task_l_hetero"]', source)

    def test_main_and_diagnostic_prediction_metadata(self):
        source = Path("train_cf_residual_recovery.py").read_text()
        self.assertIn('valid_predictions["selected_by"] = "validation"', source)
        self.assertIn('diagnostic_predictions["selected_by"] = "test"', source)
        self.assertIn('diagnostic_predictions["not_main_result"] = True', source)

    def test_formal_outputs_cannot_be_selectively_overwritten(self):
        source = Path("train_cf_residual_recovery.py").read_text()
        self.assertIn("Formal Stage 4A outputs already exist", source)

    def test_formal_guard_allows_smoke_subdirectory_but_rejects_formal_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); result = root / "benchmark_train"
            (result / "smoke").mkdir(parents=True)
            cli = SimpleNamespace(smoke_test=False)
            main = root / "main.pth"; diagnostic = root / "diagnostic.pth"
            trainer._formal_output_guard(cli, result, main, diagnostic)
            (result / "mosi_per_seed.csv").write_text("formal")
            with self.assertRaises(FileExistsError):
                trainer._formal_output_guard(cli, result, main, diagnostic)

    def test_final_report_compares_six_methods_and_stops(self):
        source = Path("generate_stage4a_audit_report.py").read_text()
        for method in ("ModDrop","FixedKD","ReliabilityKD","CFCompatKD","CFRR-only","CFCompatKD-CFRR"):
            self.assertIn(method, source)
        self.assertIn("Awaiting human audit", source)


if __name__ == "__main__":
    unittest.main()
