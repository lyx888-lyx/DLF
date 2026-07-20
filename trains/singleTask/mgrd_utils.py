"""Stage 19 multi-granular recoverable distillation primitives.

The module is train/valid only.  It contains no test loader or test-evaluation
entry point.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .anchor_decision_projection import evaluator_decisions
from .HingeLoss import HingeLoss


METHODS = ("uniform_kd", "mgd", "mgrd", "mgrd_shuffled_gate")
GRANULARITIES = ("Acc2", "Acc5", "Acc7")
MISSING_MODES = ("LA", "LV", "L")
TAU = 0.5
EPSILON = 1e-12


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_ids(ids):
    result = [str(value) for value in ids]
    if len(result) != len(set(result)):
        raise RuntimeError("Duplicate sample ID is forbidden.")
    return result


def ordered_id_sha(ids):
    payload = json.dumps(canonical_ids(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def unordered_id_sha(ids):
    payload = json.dumps(sorted(canonical_ids(ids)), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _metric_component(values, granularity):
    acc7, acc5, acc2 = evaluator_decisions(values, "mosei")
    return {"Acc2": acc2.astype(np.int64), "Acc5": acc5.astype(np.int64), "Acc7": acc7.astype(np.int64)}[
        granularity
    ]


def extract_decision_boundaries():
    """Numerically extract boundaries from the frozen project evaluator.

    Candidate points are half steps because the evaluator is round-after-clip.
    A point is retained only when the evaluator differs immediately to its left
    and right, so the returned set is derived from code behavior.
    """
    candidates = np.arange(-3.5, 3.5001, 0.5, dtype=np.float32)
    result = {}
    for granularity in GRANULARITIES:
        boundaries = []
        for value in candidates:
            left = np.nextafter(value, np.float32(-np.inf), dtype=np.float32)
            right = np.nextafter(value, np.float32(np.inf), dtype=np.float32)
            if _metric_component([left], granularity)[0] != _metric_component([right], granularity)[0]:
                boundaries.append(float(value))
        result[granularity] = tuple(boundaries)
    expected_counts = {"Acc2": 1, "Acc5": 4, "Acc7": 6}
    if {key: len(value) for key, value in result.items()} != expected_counts:
        raise RuntimeError("Unexpected evaluator decision-boundary structure: {}".format(result))
    return result


BOUNDARIES = extract_decision_boundaries()


def recover_evaluator_decisions(values, granularity):
    """Recover the evaluator class while respecting round-half-to-even ties."""
    values = np.asarray(values, dtype=np.float32)
    boundaries = BOUNDARIES[granularity]
    start = int(_metric_component([np.float32(-4.0)], granularity)[0])
    result = np.full(values.shape, start, dtype=np.int64)
    for boundary in boundaries:
        boundary = np.float32(boundary)
        at = int(_metric_component([boundary], granularity)[0])
        right = int(
            _metric_component(
                [np.nextafter(boundary, np.float32(np.inf), dtype=np.float32)],
                granularity,
            )[0]
        )
        crossed = values >= boundary if at == right else values > boundary
        result += crossed.astype(np.int64)
    return result


def ordinal_probability(values, boundaries, tau=TAU):
    values = values.float().view(-1, 1)
    boundary = torch.as_tensor(boundaries, dtype=torch.float32, device=values.device).view(1, -1)
    return torch.sigmoid((values - boundary) / float(tau))


def recoverability(teacher_probability, reference_probability):
    value = F.relu((2.0 * teacher_probability - 1.0) * (2.0 * reference_probability - 1.0))
    return value.clamp_(0.0, 1.0)


def _weight_stats(weights):
    flat = weights.detach().float().reshape(-1)
    if flat.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "nonzero_rate": 0.0, "ess": 0.0}
    total = flat.sum()
    ess = total.square() / (flat.square().sum() + EPSILON)
    return {
        "mean": float(flat.mean().cpu()),
        "std": float(flat.std(unbiased=False).cpu()),
        "nonzero_rate": float((flat > 0).float().mean().cpu()),
        "ess": float(ess.cpu()),
    }


def _shuffle_flat(values, generator):
    flat = values.reshape(-1)
    permutation = torch.randperm(flat.numel(), generator=generator, device="cpu").to(flat.device)
    return flat.index_select(0, permutation).view_as(values)


def multigranular_kd_loss(
    method,
    student_prediction,
    teacher_prediction,
    modes,
    reference_prediction=None,
    shuffle_generator=None,
    tau=TAU,
):
    """Return the frozen Stage 19 KD objective and mechanism statistics."""
    if method not in METHODS:
        raise ValueError("Unknown Stage 19 method: {}".format(method))
    student = student_prediction.float().view(-1)
    teacher = teacher_prediction.detach().float().view(-1)
    if student.shape != teacher.shape or len(modes) != student.numel():
        raise ValueError("Prediction/mode shapes differ.")
    l_reg = F.smooth_l1_loss(student, teacher, reduction="mean")
    if method == "uniform_kd":
        return l_reg, {
            "L_reg": float(l_reg.detach().cpu()),
            "L_Acc2": 0.0,
            "L_Acc5": 0.0,
            "L_Acc7": 0.0,
            "weighted_reg": float(l_reg.detach().cpu()),
            "weighted_ordinal": 0.0,
            "gate": {},
        }

    reference = None
    if method in ("mgrd", "mgrd_shuffled_gate"):
        if reference_prediction is None:
            raise ValueError("MGRD requires a frozen ModDrop reference.")
        reference = reference_prediction.detach().float().view(-1)
        if reference.shape != student.shape:
            raise ValueError("Reference prediction shape differs.")
        if method == "mgrd_shuffled_gate" and shuffle_generator is None:
            raise ValueError("Shuffled gate requires a dedicated generator.")

    ordinal_losses = {}
    gate_stats = {}
    for granularity in GRANULARITIES:
        boundaries = BOUNDARIES[granularity]
        q_student = ordinal_probability(student, boundaries, tau)
        q_teacher = ordinal_probability(teacher, boundaries, tau).detach()
        element_bce = F.binary_cross_entropy(q_student, q_teacher, reduction="none")
        if method == "mgd":
            level_loss = element_bce.mean(dim=1).mean()
        else:
            q_reference = ordinal_probability(reference, boundaries, tau).detach()
            weights = recoverability(q_teacher, q_reference)
            weighted_sum = student.new_zeros(())
            sample_total = 0
            for mode in MISSING_MODES:
                index = torch.as_tensor(
                    [position for position, value in enumerate(modes) if value == mode],
                    dtype=torch.long,
                    device=student.device,
                )
                if index.numel() == 0:
                    continue
                mode_weights = weights.index_select(0, index)
                if method == "mgrd_shuffled_gate":
                    mode_weights = _shuffle_flat(mode_weights, shuffle_generator)
                mode_bce = element_bce.index_select(0, index)
                mode_loss = (mode_weights * mode_bce).sum() / (mode_weights.sum() + EPSILON)
                # Sample-count weighting makes the all-one gate exactly reduce
                # to the unweighted MGD mean while retaining per-mode normalization.
                weighted_sum = weighted_sum + index.numel() * mode_loss
                sample_total += int(index.numel())
                gate_stats["{}_{}".format(mode, granularity)] = _weight_stats(mode_weights)
            level_loss = weighted_sum / max(sample_total, 1)
        ordinal_losses[granularity] = level_loss

    ordinal_mean = sum(ordinal_losses.values()) / float(len(ordinal_losses))
    total = 0.5 * l_reg + 0.5 * ordinal_mean
    details = {
        "L_reg": float(l_reg.detach().cpu()),
        "L_Acc2": float(ordinal_losses["Acc2"].detach().cpu()),
        "L_Acc5": float(ordinal_losses["Acc5"].detach().cpu()),
        "L_Acc7": float(ordinal_losses["Acc7"].detach().cpu()),
        "weighted_reg": float((0.5 * l_reg).detach().cpu()),
        "weighted_ordinal": float((0.5 * ordinal_mean).detach().cpu()),
        "gate": gate_stats,
    }
    return total, details


class VectorizedHingeLoss(nn.Module):
    """Experimental vectorization preserving the original masks and reduction."""

    def forward(self, ids, feats, margin=0.1):
        del margin  # The original implementation also replaces this argument.
        n_rows, feature_dim = feats.shape
        source = feats.repeat(1, n_rows).view(-1, feature_dim)
        target = feats.repeat(n_rows, 1)
        source_norm = torch.sqrt(torch.sum(torch.pow(source, 2), 1) + 1e-8)
        source_norm = torch.max(source_norm, 1e-8 * torch.ones_like(source_norm))
        target_norm = torch.sqrt(torch.sum(torch.pow(target, 2), 1) + 1e-8)
        target_norm = torch.max(target_norm, 1e-8 * torch.ones_like(target_norm))
        cosine = (torch.sum(source * target, 1) / (source_norm * target_norm)).view(n_rows, n_rows)

        identity = torch.eye(n_rows, dtype=torch.bool, device=feats.device)
        source_ids = ids.view(n_rows, 1).expand(n_rows, n_rows)
        target_ids = ids.view(1, n_rows).expand(n_rows, n_rows)
        same = (source_ids == target_ids) & ~identity
        different = (source_ids != target_ids) & ~identity
        pair_margin = 0.15 * torch.abs(source_ids - target_ids)

        # [anchor, positive, negative], exactly the tensors used by the old
        # per-anchor loop.  Per-anchor means and the final mean are unchanged.
        triplet = F.relu(
            pair_margin[:, None, :]
            - cosine[:, :, None]
            + cosine[:, None, :]
        )
        valid = same[:, :, None] & different[:, None, :]
        counts = valid.sum(dim=(1, 2))
        per_anchor = (triplet * valid).sum(dim=(1, 2)) / counts.clamp_min(1)
        active = counts > 0
        if not bool(active.any()):
            return feats.sum() * 0.0
        return per_anchor[active].mean()


def hinge_equivalence_report(device="cpu", seed=19):
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    ids = torch.tensor([-1.0, -1.0, 0.0, 0.0, 1.0, 1.0] * 3, device=device).view(-1, 1)
    base = torch.randn((ids.size(0), 50), generator=generator, dtype=torch.float32)
    old_input = base.clone().to(device).requires_grad_(True)
    new_input = base.clone().to(device).requires_grad_(True)
    old_loss = HingeLoss()(ids, old_input)
    new_loss = VectorizedHingeLoss()(ids, new_input)
    old_grad = torch.autograd.grad(old_loss, old_input)[0]
    new_grad = torch.autograd.grad(new_loss, new_input)[0]
    loss_difference = float(torch.abs(old_loss - new_loss).detach().cpu())
    gradient_difference = float(torch.max(torch.abs(old_grad - new_grad)).detach().cpu())
    return {
        "old_loss": float(old_loss.detach().cpu()),
        "new_loss": float(new_loss.detach().cpu()),
        "loss_abs_difference": loss_difference,
        "gradient_max_difference": gradient_difference,
        "accepted": loss_difference <= 1e-7 and gradient_difference <= 1e-6,
    }
