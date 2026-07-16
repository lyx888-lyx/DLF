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

import train_compat_routed_dual_teacher as train
from trains.singleTask.compat_routed_dual_teacher_utils import (
    TEACHER_ROUTES, build_dual_teacher_suitability_audit, cache_table,
    capture_rng_state, compatibility_targets, dual_teacher_audit_paths,
    lookup_mode_values, mode_teacher_targets, preserve_rng_and_modes,
    restore_rng_state, route_alpha, routed_dual_teacher_loss,
    validate_train_cache,
)


def small_frame(count=4):
    rows=[]
    for i in range(count):
        row={"sample_index":i,"sample_id":"id{}".format(i),"label":float(i%3)-1,
             "evaluator_LAV_pred":.1*i,"evaluator_LA_pred":.1*i+.01,
             "evaluator_LV_pred":.1*i+.02,"evaluator_L_pred":.1*i+.03}
        for mode,offset in (("LA",.1),("LV",.2),("L",.3)):
            row["delta_{}".format(mode)]=offset+i*.001
            row["compat_{}".format(mode)]=(i+.5)/count
        rows.append(row)
    return pd.DataFrame(rows)


def full_frame():
    frame=small_frame(1284)
    for mode in ("LA","LV","L"):
        frame["compat_{}".format(mode)]=(np.arange(1284)+.5)/1284
    return frame


class FakeStudent(nn.Module):
    def __init__(self): super().__init__(); self.scale=nn.Parameter(torch.tensor(1.))
    def forward(self,text,audio,vision,mask):
        return {"output_logit":(text[:,0,0]*self.scale+mask[:,1]*.01+mask[:,2]*.02).view(-1,1)}


class FakeTeacher(nn.Module):
    def forward(self,text,audio,vision): return {"output_logit":text[:,0,0].view(-1,1)}


def fake_loader():
    rows=[]
    for start in (0,642):
        idx=torch.arange(start,start+642)
        rows.append({"text":idx.float().view(-1,1,1)/100,"audio":torch.zeros(642,1,1),"vision":torch.zeros(642,1,1),
                     "labels":{"M":((idx%3).float()-1).view(-1,1)},"index":idx,"id":["id{}".format(i) for i in idx.tolist()]})
    return rows


class CompatibilityRoutedDualTeacherTests(unittest.TestCase):
    def test_registered_variants_are_exact(self):
        self.assertEqual(set(TEACHER_ROUTES),{"mode_only","uniform_dual","compatibility_routed"})
        self.assertEqual(TEACHER_ROUTES["compatibility_routed"][0],"DLF-CRDTD-v1")

    def test_mode_only_alpha_is_zero(self):
        a=route_alpha("mode_only",torch.tensor([.2,.8])); torch.testing.assert_close(a,torch.zeros(2))

    def test_uniform_alpha_is_half(self):
        a=route_alpha("uniform_dual",torch.tensor([.2,.8])); torch.testing.assert_close(a,torch.full((2,),.5))

    def test_compatibility_alpha_is_exact_and_detached(self):
        c=torch.tensor([.2,.8],requires_grad=True); a=route_alpha("compatibility_routed",c)
        torch.testing.assert_close(a,c.detach()); self.assertFalse(a.requires_grad)

    def test_compatibility_strict_range(self):
        for bad in (torch.tensor([0.]),torch.tensor([1.]),torch.tensor([float("nan")])):
            with self.assertRaises(ValueError): route_alpha("compatibility_routed",bad)

    def test_route_weight_sum_exact(self):
        for route in TEACHER_ROUTES:
            a=route_alpha(route,torch.tensor([.125,.875])); self.assertTrue(torch.equal(a+(1-a),torch.ones_like(a)))

    def test_unknown_route_rejected(self):
        with self.assertRaises(ValueError): route_alpha("tuned",torch.tensor([.5]))

    def test_routed_loss_formula(self):
        s=torch.tensor([1.,3.],requires_grad=True); f=torch.tensor([0.,0.]); m=torch.tensor([2.,2.]); a=torch.tensor([.25,.75])
        loss,df,dm,each=routed_dual_teacher_loss(s,f,m,a)
        torch.testing.assert_close(each,a*df+(1-a)*dm); torch.testing.assert_close(loss,each.mean())
        loss.backward(); self.assertGreater(float(s.grad.norm()),0)

    def test_alpha_one_equals_full_mean(self):
        loss,df,_,_=routed_dual_teacher_loss(torch.tensor([1.,3.]),torch.zeros(2),torch.ones(2),torch.ones(2))
        torch.testing.assert_close(loss,df.mean())

    def test_alpha_zero_equals_mode_mean(self):
        loss,_,dm,_=routed_dual_teacher_loss(torch.tensor([1.,3.]),torch.zeros(2),torch.ones(2),torch.zeros(2))
        torch.testing.assert_close(loss,dm.mean())

    def test_alpha_half_equals_uniform_dual(self):
        loss,df,dm,_=routed_dual_teacher_loss(torch.tensor([1.,3.]),torch.zeros(2),torch.ones(2),torch.full((2,),.5))
        torch.testing.assert_close(loss,(.5*df+.5*dm).mean())

    def test_identical_teachers_make_all_variants_equal(self):
        s=torch.tensor([1.,3.]); target=torch.tensor([0.,2.]); c=torch.tensor([.2,.8]); losses=[]
        for route in TEACHER_ROUTES: losses.append(routed_dual_teacher_loss(s,target,target,route_alpha(route,c))[0])
        torch.testing.assert_close(losses[0],losses[1]); torch.testing.assert_close(losses[1],losses[2])

    def test_loss_reshapes_column_vectors_safely(self):
        loss,df,dm,each=routed_dual_teacher_loss(torch.ones(3,1),torch.zeros(3,1),torch.ones(3),torch.ones(3)*.5)
        self.assertEqual(df.shape,(3,)); self.assertEqual(dm.shape,(3,)); self.assertEqual(each.shape,(3,)); self.assertEqual(loss.ndim,0)

    def test_broadcasting_mismatch_is_rejected(self):
        with self.assertRaises(ValueError): routed_dual_teacher_loss(torch.ones(3),torch.zeros(2),torch.ones(3),torch.ones(3))

    def test_targets_are_detached(self):
        s=torch.ones(2,requires_grad=True); f=torch.zeros(2,requires_grad=True); m=torch.zeros(2,requires_grad=True)
        routed_dual_teacher_loss(s,f,m,torch.ones(2)*.5)[0].backward()
        self.assertIsNone(f.grad); self.assertIsNone(m.grad); self.assertIsNotNone(s.grad)

    def test_mode_target_exact_columns(self):
        table=cache_table(small_frame()); got=mode_teacher_targets(table,[0,1,2],["LA","LV","L"],"cpu",torch.float32)
        torch.testing.assert_close(got,torch.tensor([.01,.12,.23]))

    def test_mode_target_never_uses_lav(self):
        table=cache_table(small_frame()); table[0]["evaluator_LA_pred"]=99
        self.assertEqual(float(mode_teacher_targets(table,[0],["LA"],"cpu",torch.float32)[0]),99)
        with self.assertRaises(ValueError): lookup_mode_values(table,[0],["LAV"],"evaluator","cpu",torch.float32)

    def test_compatibility_exact_mode_binding(self):
        table=cache_table(small_frame()); got=compatibility_targets(table,[0,1,2],["LA","LV","L"],"cpu",torch.float64)
        np.testing.assert_allclose(got.numpy(),[.125,.375,.625])

    def test_missing_binding_fails_immediately(self):
        table=cache_table(small_frame())
        with self.assertRaises(KeyError): mode_teacher_targets(table,[9],["LA"],"cpu",torch.float32)

    def test_duplicate_cache_index_rejected(self):
        frame=full_frame(); frame.loc[1,"sample_index"]=0
        with self.assertRaises(ValueError): validate_train_cache(frame)

    def test_missing_cache_index_rejected(self):
        frame=full_frame(); frame.loc[1283,"sample_index"]=2000
        with self.assertRaises(ValueError): validate_train_cache(frame)

    def test_cache_requires_1284_train_samples(self):
        with self.assertRaises(ValueError): validate_train_cache(small_frame())
        validate_train_cache(full_frame())

    def test_cache_requires_all_three_mode_predictions(self):
        with self.assertRaises(ValueError): validate_train_cache(full_frame().drop(columns=["evaluator_LV_pred"]))

    def test_rng_context_preserves_python_numpy_torch_and_model_mode(self):
        model=FakeStudent(); model.train(); random.seed(5); np.random.seed(5); torch.manual_seed(5)
        expected=(random.random(),np.random.rand(),torch.rand(1)); random.seed(5); np.random.seed(5); torch.manual_seed(5)
        with preserve_rng_and_modes(model): random.random(); np.random.rand(); torch.rand(1); model.eval()
        self.assertTrue(model.training); self.assertEqual(random.random(),expected[0]); self.assertEqual(np.random.rand(),expected[1]); torch.testing.assert_close(torch.rand(1),expected[2])

    def test_capture_restore_roundtrip(self):
        torch.manual_seed(7); state=capture_rng_state(); expected=torch.rand(2); restore_rng_state(state); torch.testing.assert_close(torch.rand(2),expected)

    def test_audit_paths_are_isolated(self):
        paths=dual_teacher_audit_paths("root","mosi",1111)
        self.assertIn("dual_teacher",str(paths["directory"])); self.assertIn("crdtd_v1",str(paths["directory"]))

    def test_suitability_audit_is_train_only_complete_and_parameter_preserving(self):
        student=FakeStudent(); teacher=FakeTeacher(); frame=full_frame(); before=student.scale.detach().clone()
        with tempfile.TemporaryDirectory() as tmp:
            full=Path(tmp)/"full.pth"; mode=Path(tmp)/"mode.pth"; torch.save(teacher.state_dict(),full); torch.save(student.state_dict(),mode)
            paths,summary=build_dual_teacher_suitability_audit(student,teacher,fake_loader(),"cpu",frame,full,mode,{"config_sha256":"abc"},tmp)
            self.assertEqual(summary["source_split"],"train"); self.assertEqual(summary["train_sample_count"],1284)
            self.assertEqual(len(pd.read_csv(paths["samples"])),3852); self.assertEqual(len(pd.read_csv(paths["targets"])),1284)
            self.assertEqual(len(pd.read_csv(paths["quartiles"])),12); torch.testing.assert_close(student.scale,before)
            manifest=json.loads(paths["manifest"].read_text()); self.assertTrue(manifest["rng_state_preserved"])

    def test_cli_rejects_route_tuning(self):
        for argv in (["x","--lambda-route",".5"],["x","--lambda-kd","1"],["x","--temperature","2"],["x","--route-gamma","2"]):
            with mock.patch.object(sys,"argv",argv):
                with self.assertRaises(SystemExit): train.parse_args()

    def test_cli_rejects_other_seed(self):
        with mock.patch.object(sys,"argv",["x","--seeds","1112"]):
            with self.assertRaises(SystemExit): train.parse_args()

    def test_main_and_diagnostic_paths_are_isolated(self):
        with mock.patch.object(sys,"argv",["x","--teacher-route","mode_only"]): cli=train.parse_args()
        _,main,diag=train.method_paths(cli,"mosi"); self.assertIn("best_valid",str(main)); self.assertIn("diagnostic",str(diag)); self.assertNotEqual(main,diag)

    def test_three_variants_have_isolated_directories(self):
        paths=[]
        for route in TEACHER_ROUTES:
            with mock.patch.object(sys,"argv",["x","--teacher-route",route]): cli=train.parse_args()
            paths.append(str(train.method_paths(cli,"mosi")[0]))
        self.assertEqual(len(set(paths)),3)

    def test_missing_sequence_is_deterministic_and_locked(self):
        self.assertEqual(train.missing_sequence_sha(1111),train.missing_sequence_sha(1111)); self.assertNotEqual(train.missing_sequence_sha(1111),train.missing_sequence_sha(1112))

    def test_source_has_online_full_teacher_and_no_mode_forward(self):
        source=Path("train_compat_routed_dual_teacher.py").read_text()
        self.assertIn("teacher_lav_prediction(teacher,text,audio,vision)",source)
        self.assertIn("mode_teacher_targets(table,indices,modes",source)
        self.assertNotIn("build_frozen_evaluator",source)

    def test_source_has_no_residual_training_route(self):
        source=Path("train_compat_routed_dual_teacher.py").read_text().lower()
        self.assertNotIn("residual_head",source); self.assertNotIn("residual_loss",source); self.assertNotIn("residual_cache",source)

    def test_source_guards_student_only_validation_and_checkpoints(self):
        source=Path("train_compat_routed_dual_teacher.py").read_text()
        for token in ("evaluate_all_modes(student","torch.save(student.state_dict(),main_checkpoint)","torch.save(student.state_dict(),diagnostic_checkpoint)","IsBestValid","IsBestTestDiagnostic"):
            self.assertIn(token,source)

    def test_independent_eval_has_no_teacher_or_cache_import(self):
        source=Path("eval_compat_routed_dual_teacher.py").read_text()
        self.assertNotIn("teacher_lav_prediction",source); self.assertNotIn("locate_stage3_cache",source); self.assertIn("student_only",source)

    def test_protocol_records_fixed_branch_and_formula(self):
        text=Path("COMPAT_ROUTED_DUAL_TEACHER_PROTOCOL.md").read_text() if Path("COMPAT_ROUTED_DUAL_TEACHER_PROTOCOL.md").exists() else ""
        self.assertIn("276fc26e6e7d1998f7dd746edfca1efbbb691e55",text)
        self.assertIn("alpha * d_full + (1 - alpha) * d_mode",text)


if __name__=="__main__": unittest.main()
