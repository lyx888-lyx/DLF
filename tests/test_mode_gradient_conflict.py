import inspect
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

import analyze_mode_gradient_conflict as analyze
from trains.singleTask.gradient_conflict_utils import (
    ALL_GROUP, GROUPS, MODE_PAIRS, add_gradients, assign_compatibility_quartiles,
    bootstrap_intervals, build_parameter_groups, capture_rng_state,
    classify_conflicts, cosine_summary, effective_rank, flatten_group_gradient,
    gradient_cosine, gradient_dot, group_gradient_stats, linear_cka,
    ordered_gradients, parameter_group_for_name, protected_audit_state,
    representation_statistics, restore_rng_state, rng_states_equal, states_equal,
)


class FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__(); self.proj_l=nn.Linear(2,2); self.proj_a=nn.Linear(2,2); self.proj_v=nn.Linear(2,2)
        self.encoder_c=nn.Linear(2,2); self.trans_l_mem=nn.Linear(2,2); self.proj1=nn.Linear(2,2); self.out_layer=nn.Linear(2,1)


class FakeStudent(nn.Module):
    def __init__(self):
        super().__init__(); self.missing_audio_token=nn.Parameter(torch.zeros(1,1,2)); self.missing_vision_token=nn.Parameter(torch.zeros(1,1,2))
        self.backbone=FakeBackbone(); self.mask_adapter=nn.Linear(3,2,bias=False); self.register_buffer("seen",torch.zeros(1))
    def forward(self,x): return self.backbone.out_layer(self.backbone.proj1(self.backbone.encoder_c(self.backbone.proj_l(x))))


def synthetic_task(fusion_values=None,head_values=None):
    rows=[]; fusion_values=fusion_values or {pair:[.2,.3,.1] for pair in MODE_PAIRS}; head_values=head_values or {pair:[.2,.3,.1] for pair in MODE_PAIRS}
    for group,values in (("shared_multimodal_fusion",fusion_values),("prediction_and_task_heads",head_values),(ALL_GROUP,fusion_values)):
        for pair,x in values.items():
            rows.append({"RowType":"summary","State":"cfcompat_best_valid","ModeA":pair[0],"ModeB":pair[1],"ParameterGroup":group,
                         "median":float(np.median(x)),"negative_fraction":float((np.array(x)<0).mean()),"bootstrap_mean_ci_high":float(np.mean(x)+.01),
                         "mean":float(np.mean(x)),"count":len(x)})
    return pd.DataFrame(rows)


def synthetic_kd(value=.2,negative_fraction=0.0):
    return pd.DataFrame([{"RowType":"summary","State":"cfcompat_best_valid","AlignmentType":"missing_task_vs_kd","Mode":mode,"ParameterGroup":group,
                          "median":value,"negative_fraction":negative_fraction,"mean":value,"count":3,"GradientNormRatio":1.0}
                         for mode in ("LA","LV","L") for group in ("shared_multimodal_fusion","prediction_and_task_heads",ALL_GROUP)])


class ModeGradientConflictTests(unittest.TestCase):
    def test_parameter_name_mapping_real_roots(self):
        cases={"backbone.text_model.model.x":"text_backbone_or_projection","backbone.proj_a.weight":"audio_backbone_or_projection",
               "backbone.proj_v.weight":"vision_backbone_or_projection","backbone.encoder_c.x":"shared_multimodal_fusion",
               "backbone.trans_l_with_a.x":"shared_multimodal_fusion","backbone.proj1.weight":"prediction_and_task_heads",
               "mask_adapter.weight":"missing_tokens_and_mask_adapter","missing_audio_token":"missing_tokens_and_mask_adapter"}
        for name,expected in cases.items(): self.assertEqual(parameter_group_for_name(name),expected)

    def test_parameter_groups_are_mutually_exclusive_and_complete(self):
        model=FakeStudent(); names,params,indices,frame,manifest=build_parameter_groups(model)
        assigned=[i for group in GROUPS for i in indices[group]]
        self.assertEqual(sorted(assigned),list(range(len(params)))); self.assertEqual(len(assigned),len(set(assigned)))
        self.assertEqual(frame.ParameterName.nunique(),len(params)); self.assertTrue(manifest["mapping_complete"])

    def test_other_group_threshold_stops(self):
        model=nn.Sequential(nn.Linear(10,10))
        with self.assertRaises(RuntimeError): build_parameter_groups(model)

    def test_gradient_order_and_none_zero_fill(self):
        a=nn.Parameter(torch.tensor([1.,2.])); b=nn.Parameter(torch.tensor([3.]))
        grads=ordered_gradients((a*a).sum(),[a,b],retain_graph=False)
        flat=flatten_group_gradient(grads,[a,b],[0,1]); torch.testing.assert_close(flat,torch.tensor([2.,4.,0.]))

    def test_gradient_stats_dimension_and_zero_fraction(self):
        a=nn.Parameter(torch.tensor([1.,2.])); b=nn.Parameter(torch.tensor([3.]))
        stats=group_gradient_stats((torch.tensor([2.,0.]),None),[a,b],[0,1])
        self.assertEqual(stats["VectorDimensionality"],3); self.assertAlmostEqual(stats["ZeroGradientFraction"],2/3); self.assertEqual(stats["GradientNorm"],2)

    def test_cosine_exact(self):
        p=nn.Parameter(torch.zeros(2)); a=(torch.tensor([1.,0.]),); b=(torch.tensor([1.,1.]),)
        self.assertAlmostEqual(gradient_cosine(a,b,[p],[0]),1/np.sqrt(2)); self.assertEqual(gradient_dot(a,b,[0]),1)

    def test_zero_norm_cosine_is_nan(self):
        p=nn.Parameter(torch.zeros(2)); self.assertTrue(np.isnan(gradient_cosine((None,),(torch.ones(2),),[p],[0])))

    def test_add_gradients_handles_none(self):
        got=add_gradients((None,torch.tensor([1.])),(torch.tensor([2.]),torch.tensor([3.])))
        torch.testing.assert_close(got[0],torch.tensor([2.])); torch.testing.assert_close(got[1],torch.tensor([4.]))

    def test_nonfinite_gradient_rejected(self):
        p=nn.Parameter(torch.tensor(float("nan")))
        with self.assertRaises(FloatingPointError): ordered_gradients(p*p,[p],retain_graph=False)

    def test_bootstrap_is_fixed_and_batch_only(self):
        a=bootstrap_intervals([-.2,.1,.3],2000,7); b=bootstrap_intervals([-.2,.1,.3],2000,7); self.assertEqual(a,b)

    def test_cosine_summary_fractions(self):
        x=cosine_summary([-.3,-.1,0,.2],50,3); self.assertEqual(x["negative_fraction"],.5); self.assertEqual(x["strong_negative_fraction"],.25); self.assertEqual(x["positive_fraction"],.25)

    def test_quartiles_are_mode_specific_complete(self):
        frame=pd.DataFrame({"sample_index":range(8),"compat_LA":range(8),"compat_LV":range(7,-1,-1),"compat_L":[1,3,2,4,8,6,7,5]})
        got=assign_compatibility_quartiles(frame); self.assertEqual(len(got),24); self.assertEqual(got[(0,"LA")],"Q1_low"); self.assertEqual(got[(0,"LV")],"Q4_high")

    def test_linear_cka_identity(self):
        x=np.arange(24).reshape(8,3); self.assertAlmostEqual(linear_cka(x,x),1)

    def test_representation_cosine_and_cka(self):
        x=np.eye(4); reps={mode:x.copy() for mode in ("LAV","LA","LV","L")}; frame,summary=representation_statistics(reps)
        self.assertTrue(np.allclose(frame.SameSampleCosineMean,1)); self.assertTrue(np.allclose(frame.LinearCKA,1)); self.assertEqual(set(summary),set(reps))

    def test_effective_rank_finite(self): self.assertTrue(np.isfinite(effective_rank(np.eye(5))))

    def test_protection_restores_rng_state_mode_parameter_and_buffer(self):
        model=FakeStudent(); model.train(); before={k:v.clone() for k,v in model.state_dict().items()}; random.seed(4); np.random.seed(4); torch.manual_seed(4); rng=capture_rng_state()
        with protected_audit_state(model):
            self.assertFalse(model.training); random.random(); np.random.rand(); torch.rand(1); model.seen.add_(1); model.mask_adapter.weight.data.add_(1)
        self.assertTrue(model.training); self.assertTrue(states_equal(before,model.state_dict())); self.assertTrue(rng_states_equal(rng,capture_rng_state()))

    def test_rng_roundtrip(self):
        torch.manual_seed(9); state=capture_rng_state(); expected=torch.rand(2); restore_rng_state(state); torch.testing.assert_close(torch.rand(2),expected)

    def test_g1_shared_fusion_rule_a(self):
        bad={pair:[-.4,-.3,.1] if pair in MODE_PAIRS[:2] else [.2,.3,.1] for pair in MODE_PAIRS}
        flags=classify_conflicts(synthetic_task(fusion_values=bad),synthetic_kd()); self.assertTrue(flags["shared_fusion_conflict"])

    def test_g1_shared_fusion_rule_b(self):
        task=synthetic_task(); mask=(task.ParameterGroup=="shared_multimodal_fusion")&(task.ModeA=="LAV")&(task.ModeB=="LA"); task.loc[mask,"bootstrap_mean_ci_high"]=-.01
        self.assertTrue(classify_conflicts(task,synthetic_kd())["shared_fusion_conflict"])

    def test_g2_prediction_head_only(self):
        bad={pair:[-.4,-.3,.1] if pair in MODE_PAIRS[:2] else [.2,.3,.1] for pair in MODE_PAIRS}
        flags=classify_conflicts(synthetic_task(head_values=bad),synthetic_kd()); self.assertTrue(flags["prediction_head_only_conflict"]); self.assertFalse(flags["shared_fusion_conflict"])

    def test_g3_kd_task_conflict(self):
        flags=classify_conflicts(synthetic_task(),synthetic_kd(-.2,.7)); self.assertTrue(flags["cfkd_task_conflict"])

    def test_g5_no_systematic_conflict(self):
        flags=classify_conflicts(synthetic_task(),synthetic_kd()); self.assertTrue(flags["no_systematic_conflict"])

    def test_six_task_pairs_are_registered(self): self.assertEqual(len(MODE_PAIRS),6)

    def test_cli_locks_seed_workers_and_bootstrap(self):
        for argv in (["x","--seed","1112"],["x","--num-workers","1"],["x","--bootstrap-samples","10"]):
            with mock.patch.object(sys,"argv",argv):
                with self.assertRaises(SystemExit): analyze.parse_args()

    def test_summary_only_requires_verification(self):
        with mock.patch.object(sys,"argv",["x","--summary-only"]):
            with self.assertRaises(SystemExit): analyze.parse_args()

    def test_source_only_constructs_train_loader(self):
        source=Path("analyze_mode_gradient_conflict.py").read_text(); self.assertIn('build_single_split_loader(args,"train"',source)
        self.assertNotIn('build_single_split_loader(args,"valid"',source); self.assertNotIn('build_single_split_loader(args,"test"',source)

    def test_source_has_no_optimizer_or_step(self):
        source=Path("analyze_mode_gradient_conflict.py").read_text(); self.assertNotIn("optim.",source); self.assertNotIn("optimizer.step",source); self.assertNotIn(".backward(",source)

    def test_source_has_original_task_losses(self):
        source=Path("analyze_mode_gradient_conflict.py").read_text(); self.assertIn("compute_full_dlf_loss",source); self.assertIn("compute_task_loss",source)

    def test_source_reuses_locked_cfcompat_loss(self):
        source=Path("analyze_mode_gradient_conflict.py").read_text(); self.assertIn("gated_kd_loss",source); self.assertNotIn("reliability",source.lower()); self.assertNotIn("mode_teacher",source.lower())

    def test_historical_cache_reader_requires_train_only_manifest(self):
        from trains.singleTask.cf_compat_kd_utils import CACHE_COLUMNS
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/"counterfactual_compatibility"/"cf_compat_v1"/"mosi"; root.mkdir(parents=True)
            rows=[]
            for i in range(1284):
                row={name:0.0 for name in CACHE_COLUMNS}; row.update(sample_index=i,sample_id="id{}".format(i),label=0.)
                for mode in ("LA","LV","L"): row["compat_{}".format(mode)]=(i+.5)/1284
                rows.append(row)
            pd.DataFrame(rows,columns=CACHE_COLUMNS).to_csv(root/"train_counterfactual_compatibility.csv",index=False)
            (root/"cf_compat_config.json").write_text('{"version":"cf_compat_v1","source":"train_only","train_sample_count":1284,"evaluator_sha256":"x"}')
            frame,table,_,_=analyze.load_locked_audit_cache(tmp,"mosi","x"); self.assertEqual(len(frame),1284); self.assertEqual(len(table),1284)
            (root/"cf_compat_config.json").write_text('{"version":"cf_compat_v1","source":"test","train_sample_count":1284,"evaluator_sha256":"x"}')
            with self.assertRaises(ValueError): analyze.load_locked_audit_cache(tmp,"mosi","x")

    def test_source_uses_autograd_grad(self): self.assertIn("torch.autograd.grad",inspect.getsource(ordered_gradients))

    def test_source_has_no_adapter_or_head_creation(self):
        source=Path("analyze_mode_gradient_conflict.py").read_text(); self.assertNotIn("nn.Linear(",source); self.assertNotIn("Adapter(",source)

    def test_source_saves_no_checkpoint(self): self.assertNotIn("torch.save",Path("analyze_mode_gradient_conflict.py").read_text())

    def test_total_gradient_is_task_plus_kd(self): self.assertIn("add_gradients(task_grads[mode],kd_grads[mode])",Path("analyze_mode_gradient_conflict.py").read_text())

    def test_representation_hook_is_exact_final_proj_input(self): self.assertIn("backbone.proj1.register_forward_pre_hook",Path("analyze_mode_gradient_conflict.py").read_text())

    def test_protocol_freezes_base_and_stop(self):
        text=Path("MODE_GRADIENT_CONFLICT_AUDIT_PROTOCOL.md").read_text(); self.assertIn("276fc26e6e7d1998f7dd746edfca1efbbb691e55",text); self.assertIn("does not create Stage 6B",text)

    def test_required_output_names_are_declared(self):
        source=Path("analyze_mode_gradient_conflict.py").read_text()
        for token in ("parameter_group_manifest.csv","audit_sample_manifest.csv","task_gradient_cosines_","gradient_transfer_","task_kd_alignment_","compatibility_quartile_gradient_","gradient_norms_","representation_alignment_","audit_summary.json","stage6a_mode_gradient_conflict_audit.md"):
            self.assertIn(token,source)


if __name__=="__main__": unittest.main()
