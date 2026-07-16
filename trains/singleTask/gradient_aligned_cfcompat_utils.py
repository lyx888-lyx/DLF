"""Gradient-only interventions for Stage 6B GA-CFCompatKD.

The forward losses are intentionally defined elsewhere by the frozen Stage 3
helpers.  This module only separates, diagnoses, combines, and replays their
gradients before the unchanged optimizer update.
"""
import hashlib
import math

import numpy as np
import torch


GRADIENT_POLICIES = (
    "manual_replay",
    "conflict_drop",
    "task_anchored_projection",
)


def trainable_named_parameters(model):
    """Return the one canonical named-parameter order used by every operation."""
    raw = []
    def visit(module, prefix=""):
        for local_name, parameter in module._parameters.items():
            if parameter is not None and parameter.requires_grad:
                raw.append((prefix + local_name, parameter))
        for local_name, child in module._modules.items():
            if child is not None:
                visit(child, prefix + local_name + ".")
    visit(model)
    if len({id(parameter) for _, parameter in raw}) != len(raw):
        raise ValueError("A trainable parameter occurs more than once.")
    named = [(name, parameter) for name, parameter in model.named_parameters()
             if parameter.requires_grad]
    if not named:
        raise ValueError("Student has no trainable parameters.")
    if [name for name, _ in raw] != [name for name, _ in named]:
        raise ValueError("Canonical named-parameter traversal changed.")
    return [name for name, _ in named], [parameter for _, parameter in named]


def ordered_autograd(loss, parameters, retain_graph):
    """Compute finite gradients, preserving ``None`` for unused parameters."""
    gradients = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        create_graph=False,
        allow_unused=True,
    )
    for gradient in gradients:
        if gradient is not None and not torch.isfinite(gradient).all():
            raise FloatingPointError("A non-finite Student gradient was produced.")
    return tuple(gradients)


def _sum_products(left, right=None, indices=None):
    """Return a global FP32 dot product (or squared norm)."""
    if right is None:
        right = left
    if len(left) != len(right):
        raise ValueError("Gradient tuples have different lengths.")
    selected = range(len(left)) if indices is None else indices
    total = None
    for index in selected:
        first, second = left[index], right[index]
        if first is None or second is None:
            continue
        value = torch.sum(first.detach().float() * second.detach().float())
        total = value if total is None else total + value
    return 0.0 if total is None else float(total.detach().cpu())


def gradient_norm(gradients, indices=None):
    return math.sqrt(max(0.0, _sum_products(gradients, indices=indices)))


def gradient_dot(left, right, indices=None):
    return _sum_products(left, right, indices)


def gradient_cosine(left, right, indices=None):
    first = gradient_norm(left, indices)
    second = gradient_norm(right, indices)
    if first < 1e-12 or second < 1e-12:
        return float("nan")
    return gradient_dot(left, right, indices) / (first * second)


def _combine_one(task, kd, coefficient, policy):
    if policy == "manual_replay":
        used = kd
    elif policy == "conflict_drop":
        used = None
    elif policy == "task_anchored_projection":
        if task is None:
            used = kd
        elif kd is None:
            used = (-coefficient * task.detach().float()).to(task)
        else:
            used = (kd.detach().float() - coefficient * task.detach().float()).to(kd)
    else:
        raise ValueError("Unknown gradient policy: {}".format(policy))
    if task is None:
        total = used
    elif used is None:
        total = task
    else:
        total = task + used.to(task)
    return used, total


def combine_task_kd_gradients(parameters, task_gradients, kd_gradients, policy):
    """Apply the fixed global, asymmetric Stage 6B policy.

    The task tuple is never mutated.  All conflict math and the projection
    coefficient are evaluated in FP32.  Projected tensors are converted back to
    the parameter gradient dtype/device before being returned.
    """
    if policy not in GRADIENT_POLICIES:
        raise ValueError("Unknown gradient policy: {}".format(policy))
    if not (len(parameters) == len(task_gradients) == len(kd_gradients)):
        raise ValueError("Parameter and gradient tuple lengths differ.")
    raw_dot = gradient_dot(task_gradients, kd_gradients)
    task_norm_sq = max(0.0, _sum_products(task_gradients))
    task_norm = math.sqrt(task_norm_sq)
    kd_raw_norm = gradient_norm(kd_gradients)
    conflict = bool(raw_dot < 0.0)
    coefficient = raw_dot / (task_norm_sq + 1e-12) if conflict else 0.0
    effective_policy = policy
    if not conflict:
        effective_policy = "manual_replay"
    used, total = [], []
    for parameter, task, kd in zip(parameters, task_gradients, kd_gradients):
        local_used, local_total = _combine_one(task, kd, coefficient, effective_policy)
        if local_used is not None:
            local_used = local_used.to(device=parameter.device, dtype=parameter.dtype)
        if local_total is not None:
            local_total = local_total.to(device=parameter.device, dtype=parameter.dtype)
        used.append(local_used)
        total.append(local_total)
    used, total = tuple(used), tuple(total)
    numerical_refinement = 0.0
    if policy == "task_anchored_projection" and conflict:
        # One deterministic FP32 residual-removal pass compensates only for
        # round-off introduced when the global coefficient is written into
        # hundreds of separate parameter tensors.
        numerical_refinement = gradient_dot(task_gradients, used) / (task_norm_sq + 1e-12)
        refined_used, refined_total = [], []
        for parameter, task, current in zip(parameters, task_gradients, used):
            if task is None:
                value = current
            elif current is None:
                value = (-numerical_refinement * task.detach().float()).to(task)
            else:
                value = (current.detach().float() - numerical_refinement * task.detach().float()).to(current)
            if value is not None:
                value = value.to(device=parameter.device, dtype=parameter.dtype)
            if task is None:
                combined = value
            elif value is None:
                combined = task
            else:
                combined = task + value.to(task)
            refined_used.append(value); refined_total.append(combined)
        used, total = tuple(refined_used), tuple(refined_total)
    projected_dot = gradient_dot(task_gradients, used)
    used_norm = gradient_norm(used)
    total_norm = gradient_norm(total)
    removed = []
    for raw, current, parameter in zip(kd_gradients, used, parameters):
        if raw is None and current is None:
            removed.append(None)
        else:
            raw_value = torch.zeros_like(parameter) if raw is None else raw
            used_value = torch.zeros_like(parameter) if current is None else current
            removed.append(raw_value - used_value)
    removed_norm = gradient_norm(tuple(removed))
    tolerance = 1e-6 * max(1.0, task_norm * kd_raw_norm)
    if policy == "task_anchored_projection" and conflict and abs(projected_dot) > tolerance:
        raise FloatingPointError(
            "Projected task-KD dot {} exceeds tolerance {}.".format(projected_dot, tolerance))
    metrics = {
        "GlobalTaskGradNorm": task_norm,
        "GlobalKDRawGradNorm": kd_raw_norm,
        "GlobalKDUsedGradNorm": used_norm,
        "GlobalTotalGradNorm": total_norm,
        "RawDot": raw_dot,
        "RawCosine": raw_dot / (task_norm * kd_raw_norm) if task_norm > 1e-12 and kd_raw_norm > 1e-12 else float("nan"),
        "ConflictFlag": conflict,
        "ProjectionCoefficient": coefficient if policy == "task_anchored_projection" and conflict else 0.0,
        "ProjectionNumericalRefinement": numerical_refinement if policy == "task_anchored_projection" and conflict else 0.0,
        "RemovedComponentNorm": removed_norm,
        "RemovedFraction": removed_norm / (kd_raw_norm + 1e-12),
        "KDRetainedNormRatio": used_norm / (kd_raw_norm + 1e-12),
        "ProjectedDot": projected_dot,
        "ProjectedCosine": projected_dot / (task_norm * used_norm) if task_norm > 1e-12 and used_norm > 1e-12 else float("nan"),
        "FirstOrderTaskInterferenceRaw": -raw_dot,
        "FirstOrderTaskInterferenceUsed": -projected_dot,
        "ProjectionTolerance": tolerance,
    }
    return used, total, metrics


def write_parameter_gradients(parameters, gradients, accumulate):
    """Replay gradients without manufacturing zeros for unused parameters."""
    if len(parameters) != len(gradients):
        raise ValueError("Parameter and gradient tuple lengths differ.")
    for parameter, gradient in zip(parameters, gradients):
        if gradient is None:
            continue
        value = gradient.detach().to(device=parameter.device, dtype=parameter.dtype)
        if accumulate and parameter.grad is not None:
            parameter.grad.add_(value)
        else:
            parameter.grad = value.clone()


def add_gradient_tuples(left, right):
    result = []
    for first, second in zip(left, right):
        if first is None and second is None:
            result.append(None)
        elif first is None:
            result.append(second.detach().clone())
        elif second is None:
            result.append(first.detach().clone())
        else:
            result.append(first.detach() + second.detach())
    return tuple(result)


def subtract_gradient_tuples(left, right):
    """Return ``left - right`` while preserving unused coordinates."""
    result = []
    for first, second in zip(left, right):
        if first is None and second is None:
            result.append(None)
        elif first is None:
            result.append(-second.detach())
        elif second is None:
            result.append(first.detach())
        else:
            result.append(first.detach() - second.detach())
    return tuple(result)


def replay_corrected_total(reference_total, task, raw_kd, used_kd):
    """Anchor policy updates to the exact Stage 3 backward accumulation order.

    Large models can show tiny but optimizer-visible round-off differences
    between ``grad(L_sup + L_kd)`` and separately evaluated gradients.  The
    correction below is only that numerical residual.  It makes manual replay
    exactly equal to the reference and applies the same baseline to interventions.
    """
    baseline = add_gradient_tuples(task, raw_kd)
    candidate = add_gradient_tuples(task, used_kd)
    corrected = []
    for reference, original, current in zip(reference_total, baseline, candidate):
        if reference is None and original is None:
            correction = None
        elif reference is None:
            correction = -original
        elif original is None:
            correction = reference
        else:
            correction = reference - original
        if current is None:
            corrected.append(correction)
        elif correction is None:
            corrected.append(current)
        else:
            corrected.append(current + correction.to(current))
    return tuple(corrected)


def clone_gradients(parameters):
    return tuple(None if parameter.grad is None else parameter.grad.detach().clone()
                 for parameter in parameters)


def compare_tensor_tuples(reference, candidate):
    """Per-tensor and global equivalence diagnostics for gradients/parameters."""
    if len(reference) != len(candidate):
        raise ValueError("Tensor tuple lengths differ.")
    max_abs = 0.0
    max_relative = 0.0
    mismatched = 0
    dot = ref_sq = candidate_sq = 0.0
    for first, second in zip(reference, candidate):
        if first is None or second is None:
            if first is not None or second is not None:
                mismatched += 1
            continue
        a, b = first.detach().float(), second.detach().float()
        difference = torch.abs(a - b)
        local_abs = float(difference.max().cpu()) if difference.numel() else 0.0
        denominator = torch.maximum(torch.abs(a), torch.abs(b)).clamp_min(1e-12)
        local_relative = float((difference / denominator).max().cpu()) if difference.numel() else 0.0
        max_abs = max(max_abs, local_abs)
        max_relative = max(max_relative, local_relative)
        if not (local_abs <= 1e-6 or local_relative <= 1e-5):
            mismatched += 1
        dot += float(torch.sum(a * b).cpu())
        ref_sq += float(torch.sum(a * a).cpu())
        candidate_sq += float(torch.sum(b * b).cpu())
    cosine = dot / math.sqrt(ref_sq * candidate_sq) if ref_sq > 0 and candidate_sq > 0 else float("nan")
    return {
        "max_abs_difference": max_abs,
        "max_relative_difference": max_relative,
        "cosine": cosine,
        "mismatched_parameter_count": mismatched,
    }


def group_gradient_metrics(task, raw, used, parameters, group_indices):
    rows = []
    for group, indices in group_indices.items():
        task_norm = gradient_norm(task, indices)
        raw_norm = gradient_norm(raw, indices)
        used_norm = gradient_norm(used, indices)
        raw_cosine = gradient_cosine(task, raw, indices)
        used_cosine = gradient_cosine(task, used, indices)
        removed_sq = 0.0
        for index in indices:
            first = torch.zeros_like(parameters[index]) if raw[index] is None else raw[index]
            second = torch.zeros_like(parameters[index]) if used[index] is None else used[index]
            removed_sq += float(torch.sum((first.detach().float() - second.detach().float()) ** 2).cpu())
        rows.append({
            "ParameterGroup": group,
            "TaskGradNorm": task_norm,
            "RawKDGradNorm": raw_norm,
            "UsedKDGradNorm": used_norm,
            "RawTaskKDCosine": raw_cosine,
            "UsedTaskKDCosine": used_cosine,
            "RemovedNormFraction": math.sqrt(max(0.0, removed_sq)) / (raw_norm + 1e-12),
        })
    return rows


class MissingSequenceDigest:
    """Streaming, device-independent SHA256 for the exact sampled masks."""
    def __init__(self):
        self._digest = hashlib.sha256()
        self.count = 0

    def update(self, masks):
        array = masks.detach().cpu().to(torch.uint8).contiguous().numpy()
        self._digest.update(array.tobytes(order="C"))
        self.count += int(array.shape[0])

    def hexdigest(self):
        return self._digest.hexdigest()


def finite_quantiles(values):
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {key: float("nan") for key in ("Mean", "Median", "Min", "P10", "P25", "P75", "P90", "Max")}
    quantiles = np.quantile(array, [.1, .25, .5, .75, .9])
    return {
        "Mean": float(array.mean()), "Median": float(quantiles[2]),
        "Min": float(array.min()), "P10": float(quantiles[0]),
        "P25": float(quantiles[1]), "P75": float(quantiles[3]),
        "P90": float(quantiles[4]), "Max": float(array.max()),
    }
