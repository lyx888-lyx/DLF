import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

import eval_coherent_routed_residual as evaluator
import train_coherent_routed_residual as trainer
from trains.singleTask.cf_residual_utils import CounterfactualResidualStudent
from trains.singleTask.coherent_routed_residual_utils import (
    ROUTE_VARIANTS,
    _alignment_pair,
    direct_prediction,
    gradient_alignment_audit,
    missing_sequence_sha256,
    route_diagnostic_rows,
    shared_route_loss,
)
from trains.singleTask.fixed_kd_utils import capture_rng_state
from trains.singleTask.missing_utils import MissingModalityWrapper


class TinyBackbone(nn.Module):
    def __init__(self, fusion_dim=4):
        super().__init__(); self.out_layer = nn.Linear(fusion_dim, 1)

    def forward(self, text, audio, vision, fusion_residual=None):
        feature = text[:, 0, :self.out_layer.in_features]
        if fusion_residual is not None:
            feature = feature + fusion_residual
        output = self.out_layer(feature); zeros = torch.zeros_like(output)
        return {"fusion_feature": feature, "output_logit": output, "logits_c": zeros,
                "logits_l_hetero": zeros, "logits_v_hetero": zeros, "logits_a_hetero": zeros}


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__(); self.linear = nn.Linear(4, 1)

    def forward(self, text, audio, vision):
        return {"output_logit": self.linear(text[:, 0, :4])}


def tiny_student():
    base = MissingModalityWrapper(TinyBackbone(), audio_dim=2, vision_dim=3)
    return CounterfactualResidualStudent(base, {"LA": .1, "LV": .2, "L": .3})


def tiny_output():
    base = torch.tensor([[.2], [-.4]], requires_grad=True)
    residual = torch.tensor([[.1], [.3]], requires_grad=True)
    return {"base_output_logit": base, "predicted_residual": residual,
            "corrected_output_logit": base + residual,
            "output_logit": base + residual, "predicted_residual_z": residual}


def tiny_loader():
    batches = []
    for batch in range(4):
        start = batch * 2
        batches.append({
            "text": torch.randn(2, 1, 4), "audio": torch.randn(2, 2, 2), "vision": torch.randn(2, 2, 3),
            "labels": {"M": torch.randn(2, 1)}, "index": torch.tensor([start, start + 1]),
            "id": ["id{}".format(start), "id{}".format(start + 1)],
        })
    return batches


def tiny_cache():
    return {index: {"compat_LA": .2, "compat_LV": .5, "compat_L": .8,
                    "residual_LA": -.1, "residual_LV": .2, "residual_L": -.3}
            for index in range(8)}


class CoherentRoutingTests(unittest.TestCase):
    def test_fixed_variant_names_and_methods(self):
        self.assertEqual(tuple(ROUTE_VARIANTS), ("base_shared", "corrected_joint_shared", "corrected_stopres_shared"))
        self.assertEqual(ROUTE_VARIANTS["corrected_stopres_shared"]["method"], "DLF-CCRRD-v1")

    def test_base_shared_direct_prediction_is_base(self):
        output = tiny_output(); self.assertIs(direct_prediction(output, "base_shared"), output["base_output_logit"])

    def test_corrected_joint_direct_prediction_is_corrected(self):
        output = tiny_output(); self.assertIs(direct_prediction(output, "corrected_joint_shared"), output["corrected_output_logit"])

    def test_stopres_forward_value_equals_corrected(self):
        output = tiny_output()
        torch.testing.assert_close(direct_prediction(output, "corrected_stopres_shared"), output["corrected_output_logit"])

    def test_stopres_direct_gradient_does_not_reach_residual(self):
        output = tiny_output(); prediction = direct_prediction(output, "corrected_stopres_shared")
        loss = prediction.square().mean(); base_grad, residual_grad = torch.autograd.grad(loss, [output["base_output_logit"], output["predicted_residual"]], allow_unused=True)
        self.assertGreater(float(base_grad.abs().sum()), 0); self.assertIsNone(residual_grad)

    def test_joint_direct_gradient_reaches_residual(self):
        output = tiny_output(); loss = direct_prediction(output, "corrected_joint_shared").square().mean()
        residual_grad, = torch.autograd.grad(loss, [output["predicted_residual"]])
        self.assertGreater(float(residual_grad.abs().sum()), 0)

    def test_shared_route_is_exact_batch_mean(self):
        output = tiny_output(); teacher = torch.tensor([[1.], [-1.]])
        target = torch.tensor([.5, -.25]); compatibility = torch.tensor([.2, .8])
        routed = shared_route_loss(output, teacher, target, compatibility, "base_shared")
        expected = (compatibility * routed["direct_each"] + (1-compatibility) * routed["residual_each"]).mean()
        torch.testing.assert_close(routed["loss"], expected)

    def test_route_weights_sum_strictly_to_one(self):
        output = tiny_output(); compatibility = torch.tensor([.123, .987])
        routed = shared_route_loss(output, torch.zeros(2,1), torch.zeros(2), compatibility, "base_shared")
        torch.testing.assert_close(routed["compatibility"] + (1-routed["compatibility"]), torch.ones(2), rtol=0, atol=0)

    def test_persisted_weight_sum_guard_only_allows_serialization_precision(self):
        source = Path("train_coherent_routed_residual.py").read_text()
        self.assertIn('math.isclose(route_summary["MeanRouteWeightSum"], 1.0, rel_tol=0.0, abs_tol=1e-7)', source)

    def test_route_does_not_separately_renormalize_channels(self):
        source = Path("trains/singleTask/coherent_routed_residual_utils.py").read_text()
        body = source[source.index("def shared_route_loss"):source.index("def missing_sequence_sha256")]
        self.assertIn("route_each.mean()", body)
        self.assertNotIn("torch.sum(weighted_direct) /", body)
        self.assertNotIn("torch.sum(weighted_residual) /", body)

    def test_teacher_target_and_compatibility_are_detached(self):
        output = tiny_output(); teacher = torch.randn(2,1,requires_grad=True); compat = torch.tensor([.2,.8],requires_grad=True)
        routed = shared_route_loss(output, teacher, torch.zeros(2,requires_grad=True), compat, "base_shared")
        routed["loss"].backward(); self.assertIsNone(teacher.grad); self.assertIsNone(compat.grad)

    def test_routed_shapes_must_be_batch_vectors(self):
        output = tiny_output()
        with self.assertRaises(ValueError):
            shared_route_loss(output, torch.zeros(3,1), torch.zeros(2), torch.ones(2), "base_shared")

    def test_unknown_variant_is_rejected(self):
        with self.assertRaises(ValueError): direct_prediction(tiny_output(), "unknown")

    def test_lambda_route_and_seed_are_locked(self):
        with self.assertRaises(SystemExit): trainer.parse_args(["--route-variant","base_shared","--lambda-route",".5"])
        with self.assertRaises(SystemExit): trainer.parse_args(["--route-variant","base_shared","--seeds","1112"])

    def test_old_lambda_arguments_are_rejected(self):
        for flag in ("--lambda-kd", "--lambda-residual"):
            with self.assertRaises(SystemExit): trainer.parse_args(["--route-variant","base_shared",flag,"1"])

    def test_alignment_only_does_not_require_variant(self):
        cli = trainer.parse_args(["--build-alignment-audit-only"]); self.assertTrue(cli.build_alignment_audit_only)

    def test_smoke_is_capped_at_two_epochs(self):
        cli = trainer.parse_args(["--route-variant","base_shared","--smoke-test","--max-epochs","9"])
        self.assertEqual(cli.max_epochs, 2)

    def test_missing_sequence_hash_is_deterministic_and_order_sensitive(self):
        first = missing_sequence_sha256(["LA","LV","L"]); second = missing_sequence_sha256(["LA","LV","L"])
        self.assertEqual(first, second); self.assertNotEqual(first, missing_sequence_sha256(["L","LV","LA"]))

    def test_missing_sequence_rejects_lav(self):
        with self.assertRaises(ValueError): missing_sequence_sha256(["LAV"])

    def test_route_diagnostics_have_shared_contributions_and_quartiles(self):
        records=[]
        for index in range(12):
            mode=("LA","LV","L")[index%3]; c=(index+.5)/12
            records.append({"sample_index":index,"mode":mode,"compatibility":c,"complement":1-c,
                            "direct_raw":.2,"residual_raw":.4,"weighted_direct":c*.2,"weighted_residual":(1-c)*.4,
                            "predicted_residual":.01,"target_residual":.02,"base_prediction":.2,
                            "corrected_prediction":.21,"label":.1})
        summary, quartiles = route_diagnostic_rows(records,1111,1,"m","base_shared")
        self.assertAlmostEqual(summary["MeanRouteWeightSum"],1.0); self.assertEqual(len(quartiles),4)
        self.assertAlmostEqual(summary["RouteLoss"],summary["WeightedDirectContribution"]+summary["WeightedResidualContribution"])

    def test_alignment_pair_definitions(self):
        result = _alignment_pair([1.,-1.],[.5,-.5])
        self.assertEqual(result["MAE"],.5); self.assertEqual(result["RMSE"],.5); self.assertEqual(result["SignAgreement"],1.)

    def test_gradient_audit_joint_and_stopres_contracts(self):
        for variant in ("corrected_joint_shared", "corrected_stopres_shared"):
            random.seed(3); np.random.seed(3); torch.manual_seed(3)
            student=tiny_student(); teacher=TinyTeacher(); loader=tiny_loader(); before=capture_rng_state()
            rows=gradient_alignment_audit(student,teacher,loader,tiny_cache(),{"LA":.1,"LV":.2,"L":.3},"cpu",variant,nn.L1Loss(),"test")
            direct = next(row["DirectGradNorm"] for row in rows if row["ParameterGroup"]=="residual_heads")
            self.assertGreater(direct,0) if variant=="corrected_joint_shared" else self.assertEqual(direct,0)
            after=capture_rng_state(); self.assertTrue(torch.equal(before["torch"],after["torch"]))

    def test_training_uses_corrected_task_output_and_one_route_loss(self):
        source=Path("train_coherent_routed_residual.py").read_text()
        self.assertIn("compute_task_loss(missing_output",source)
        self.assertIn('full_loss + missing_loss + routed["loss"]',source)
        self.assertNotIn("complementary_residual_loss",source); self.assertNotIn("gated_kd_loss",source)

    def test_locked_cache_sha_is_enforced(self):
        source=Path("train_coherent_routed_residual.py").read_text()
        self.assertIn("8702d2ff15b80594094ea74469ad393a681992ecce1891c7095208f8c1004790",source)

    def test_epoch1_counts_and_sequence_are_recorded(self):
        source=Path("train_coherent_routed_residual.py").read_text()
        self.assertIn('Counter({"LA": 435, "LV": 430, "L": 419})',source)
        self.assertIn("MissingSequenceSHA256",source)

    def test_main_and_diagnostic_selection_are_separate(self):
        source=Path("train_coherent_routed_residual.py").read_text()
        self.assertIn("j_valid <= best_valid_j - 1e-6",source); self.assertIn("j_test <= best_test_j - 1e-6",source)
        self.assertIn('diagnostic_predictions["not_main_result"] = True',source)

    def test_formal_guard_allows_smoke_only_and_rejects_formal_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); result=root/"benchmark_train"; (result/"smoke").mkdir(parents=True)
            cli=SimpleNamespace(smoke_test=False); trainer._formal_output_guard(cli,result,root/"main",root/"diag")
            (result/"mosi_per_seed.csv").write_text("x")
            with self.assertRaises(FileExistsError): trainer._formal_output_guard(cli,result,root/"main",root/"diag")

    def test_required_outputs_are_registered(self):
        for name in ("mosi_route_summary.csv","mosi_route_quartiles.csv","mosi_residual_mode_metrics.csv",
                     "gradient_alignment_init.csv","gradient_alignment_best_valid.csv"):
            self.assertIn(name,trainer.RESULT_FILES)

    def test_alignment_audit_is_train_only_diagnostic(self):
        source=Path("trains/singleTask/coherent_routed_residual_utils.py").read_text()
        start=source.index("def build_teacher_evaluator_alignment"); end=source.index("def _parameter_groups")
        body=source[start:end]
        self.assertIn('build_single_split_loader(args, "train"',body)
        self.assertIn('"DiagnosticOnly": True',body); self.assertNotIn('build_single_split_loader(args, "valid"',body)
        self.assertNotIn('build_single_split_loader(args, "test"',body)

    def test_teacher_needed_residual_formula_is_exact(self):
        source=Path("trains/singleTask/coherent_routed_residual_utils.py").read_text()
        self.assertIn('teacher_value - evaluator',source)

    def test_gradient_audit_has_no_optimizer_step(self):
        source=Path("trains/singleTask/coherent_routed_residual_utils.py").read_text()
        body=source[source.index("def gradient_alignment_audit"):]
        self.assertNotIn("optimizer.step",body); self.assertIn("parameter_snapshot",body); self.assertIn("restore_rng_state",body)

    def test_eval_is_student_only_and_test_guarded(self):
        source=Path("eval_coherent_routed_residual.py").read_text()
        for forbidden in ("build_frozen_teacher","load_residual_cache","Evaluator"):
            self.assertNotIn(forbidden,source)
        with self.assertRaises(SystemExit): evaluator.parse_args(["--route-variant","base_shared","--split","test"])

    def test_report_compares_all_seven_methods_and_stops(self):
        source=Path("generate_stage4a1_audit_report.py").read_text()
        for name in ("ModDrop","CFCompatKD","CFRR-only","Old CFCompatKD-CFRR","Base-Shared","Corrected-Joint-Shared","Corrected-StopResidual-Shared"):
            self.assertIn(name,source)
        self.assertIn("Awaiting human audit",source)
        self.assertIn('metric_map(cf, "test_at_valid_best")', source)
        self.assertIn('metric_map(cfrr, "corrected_test_at_valid_best")', source)


if __name__ == "__main__":
    unittest.main()
