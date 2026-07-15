"""Fixed reliability-gated prediction KD helpers for Stage 3A."""
import math
import torch
from pathlib import Path
from .fixed_kd_utils import build_frozen_teacher, checkpoint_sha256, assert_initial_lav_equivalence, assert_teacher_not_in_optimizer, teacher_grad_count, teacher_lav_prediction

def reliability_weights(teacher_prediction, labels):
    prediction = teacher_prediction.detach().clone().view(-1)
    target = labels.detach().view(-1)
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise FloatingPointError("Teacher prediction and labels must be finite.")
    weight = torch.exp(-torch.abs(prediction - target))
    if not torch.isfinite(weight).all() or torch.any(weight <= 0) or torch.any(weight > 1):
        raise FloatingPointError("Reliability must be finite and in (0,1].")
    return weight

def reliability_kd_loss(student_prediction, teacher_prediction, weights):
    student = student_prediction.view(-1)
    teacher = teacher_prediction.detach().clone().view(-1)
    weight = weights.detach().view(-1).to(student)
    loss = torch.nn.functional.smooth_l1_loss(student, teacher, reduction="none")
    if loss.shape != weight.shape: raise ValueError("KD and reliability shape mismatch.")
    value = torch.sum(weight * loss) / (torch.sum(weight) + 1e-8)
    if not torch.isfinite(value): raise FloatingPointError("Reliability KD is non-finite.")
    return value, loss

def reliability_checkpoint_path(root,dataset,seed):
    return Path(root)/"missing_baseline"/"reliability_kd_v1"/f"DLF_{dataset}_seed{seed}_best.pth"

def stats(values):
    x=torch.as_tensor(values,dtype=torch.float64).view(-1)
    qs=torch.quantile(x,torch.tensor([.1,.25,.5,.75,.9,.95],dtype=torch.float64))
    return {"mean":float(x.mean()),"std":float(x.std(unbiased=False)),"min":float(x.min()),"p10":float(qs[0]),"p25":float(qs[1]),"median":float(qs[2]),"p75":float(qs[3]),"p90":float(qs[4]),"p95":float(qs[5]),"max":float(x.max())}
