import inspect,sys,unittest
from pathlib import Path
from unittest import mock
import torch
import torch.nn as nn
import train_reliability_kd
from trains.singleTask.reliability_kd_utils import reliability_weights,reliability_kd_loss,reliability_checkpoint_path
from trains.singleTask.fixed_kd_utils import freeze_teacher,assert_teacher_not_in_optimizer,teacher_grad_count
from trains.singleTask.missing_utils import MissingModalityWrapper
class Tiny(nn.Module):
 def __init__(self):super().__init__();self.out_layer=nn.Linear(6,1)
 def forward(self,t,a,v,fusion_residual=None):
  x=torch.cat([t.mean(1),a.mean(1),v.mean(1)],1);return {'output_logit':self.out_layer(x)}
class T(unittest.TestCase):
 def test_formula_monotonic_finite_and_detached(self):
  p=torch.tensor([[0.],[1.],[2.]],requires_grad=True);y=torch.tensor([[0.],[0.],[0.]],requires_grad=True);w=reliability_weights(p,y);torch.testing.assert_close(w,torch.exp(-torch.tensor([0.,1.,2.])));self.assertEqual(w.requires_grad,False);self.assertEqual(float(w[0]),1.);self.assertGreater(float(w[1]),float(w[2]));self.assertTrue(torch.all((w>0)&(w<=1)))
 def test_nonfinite_rejected(self):
  with self.assertRaises(FloatingPointError):reliability_weights(torch.tensor([float('nan')]),torch.tensor([0.]))
 def test_weighted_kd_normalization(self):
  s=torch.tensor([[1.],[3.]],requires_grad=True);t=torch.tensor([[0.],[0.]])
  for w in (torch.ones(2),torch.full((2,),.3)):
   x,_=reliability_kd_loss(s,t,w);torch.testing.assert_close(x,nn.SmoothL1Loss()(s,t),rtol=1e-6,atol=1e-7)
  x,each=reliability_kd_loss(s,t,torch.tensor([1.,.5]));torch.testing.assert_close(x,(torch.tensor([1.,.5])*each).sum()/(1.5+1e-8))
 def test_teacher_frozen_and_excluded(self):
  teacher=freeze_teacher(Tiny());student=MissingModalityWrapper(Tiny(),2,2);opt=torch.optim.Adam(student.parameters());self.assertFalse(teacher.training);self.assertTrue(all(not p.requires_grad for p in teacher.parameters()));assert_teacher_not_in_optimizer(teacher,opt);self.assertEqual(teacher_grad_count(teacher),0)
 def test_checkpoint_and_cli(self):
  self.assertIn('reliability_kd_v1',str(reliability_checkpoint_path('pt','mosi',1111)))
  with mock.patch.object(sys,'argv',['x','--eta','.5']):
   with self.assertRaises(SystemExit):train_reliability_kd.parse_args()
  with mock.patch.object(sys,'argv',['x','--lambda-kd','.5']):
   with self.assertRaises(SystemExit):train_reliability_kd.parse_args()
 def test_benchmark_protocol_and_no_temperature(self):
  src=Path('train_reliability_kd.py').read_text();self.assertIn("test_loader",src);self.assertIn("J_valid",src);self.assertNotIn('--temperature',src);self.assertNotIn('--tau',src);self.assertIn("torch.inference_mode",inspect.getsource(__import__('trains.singleTask.fixed_kd_utils',fromlist=['teacher_lav_prediction']).teacher_lav_prediction))
if __name__=='__main__':unittest.main()
