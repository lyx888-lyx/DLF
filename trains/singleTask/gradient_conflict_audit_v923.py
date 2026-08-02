"""No-training gradient-space conflict audit for V9.23.

The audit loads a frozen V9.19 CFCompat deployment anchor, computes exact
mean-MAE gradients for the global development set and the five fixed semantic
regions, and analyzes their geometry. It never creates an optimizer, calls
``backward()``, or updates a model parameter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from scipy.optimize import minimize

from .role_conditioned_experts_v9 import REGION_NAMES, region_index

AUDIT_VERSION = "gradient_space_conflict_audit_v923_v1"
TASK_NAMES = ("global", *REGION_NAMES)
REGION_TASK_NAMES = tuple(REGION_NAMES)
FUSION_TAIL_MODULES = (
    "projector_l",
    "projector_a",
    "projector_v",
    "projector_c",
    "proj1",
    "proj2",
    "out_layer",
)


@dataclass(frozen=True)
class GradientAuditConfigV923:
    parameter_scope: str = "fusion_tail"
    conflict_tolerance: float = 1e-12
    common_cosine_margin: float = 0.02
    minimum_mgda_norm: float = 0.05
    required_feasible_outer_folds: int = 4
    mgda_max_iterations: int = 2000
    mgda_ftol: float = 1e-12


@dataclass(frozen=True)
class ParameterRecordV923:
    name: str
    group: str
    parameter: torch.nn.Parameter


Gradient = tuple[torch.Tensor, ...]


def select_parameter_records(
    model: torch.nn.Module,
    scope: str = "fusion_tail",
) -> list[ParameterRecordV923]:
    """Select one fixed shared-parameter scope from a CFCompat wrapper."""
    backbone = getattr(model, "backbone", model)
    if scope == "fusion_tail":
        module_names = FUSION_TAIL_MODULES
    elif scope == "output_head":
        module_names = ("out_layer",)
    else:
        raise ValueError("parameter_scope must be fusion_tail or output_head")

    records: list[ParameterRecordV923] = []
    seen: set[int] = set()
    for module_name in module_names:
        module = getattr(backbone, module_name, None)
        if module is None:
            raise AttributeError(f"anchor backbone has no module {module_name}")
        local_count = 0
        for local_name, parameter in module.named_parameters(recurse=True):
            if not parameter.requires_grad or id(parameter) in seen:
                continue
            seen.add(id(parameter))
            local_count += int(parameter.numel())
            records.append(
                ParameterRecordV923(
                    name=f"backbone.{module_name}.{local_name}",
                    group=module_name,
                    parameter=parameter,
                )
            )
        if local_count == 0:
            raise RuntimeError(f"parameter group {module_name} is empty")
    if not records:
        raise RuntimeError("no parameters selected for gradient audit")
    return records


def selected_parameter_fingerprint(
    records: Sequence[ParameterRecordV923],
) -> str:
    import hashlib

    digest = hashlib.sha256()
    for record in records:
        value = record.parameter.detach().cpu().contiguous()
        digest.update(record.name.encode("utf-8"))
        digest.update(str(tuple(value.shape)).encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _zeros(records: Sequence[ParameterRecordV923]) -> Gradient:
    return tuple(
        torch.zeros_like(record.parameter, device="cpu", dtype=torch.float32)
        for record in records
    )


def _clone(gradient: Gradient) -> Gradient:
    return tuple(value.clone() for value in gradient)


def gradient_dot(left: Gradient, right: Gradient) -> float:
    if len(left) != len(right):
        raise ValueError("gradient lengths differ")
    total = 0.0
    for first, second in zip(left, right):
        total += float(torch.sum(first.double() * second.double()).item())
    return total


def gradient_norm(gradient: Gradient) -> float:
    return float(max(gradient_dot(gradient, gradient), 0.0) ** 0.5)


def gradient_scale(gradient: Gradient, scale: float) -> Gradient:
    return tuple(value * float(scale) for value in gradient)


def gradient_add(
    left: Gradient,
    right: Gradient,
    scale: float = 1.0,
) -> Gradient:
    if len(left) != len(right):
        raise ValueError("gradient lengths differ")
    return tuple(
        first + float(scale) * second
        for first, second in zip(left, right)
    )


def gradient_mean(gradients: Sequence[Gradient]) -> Gradient:
    if not gradients:
        raise ValueError("cannot average an empty gradient collection")
    result = _zeros_from_gradient(gradients[0])
    for gradient in gradients:
        result = gradient_add(result, gradient)
    return gradient_scale(result, 1.0 / len(gradients))


def _zeros_from_gradient(gradient: Gradient) -> Gradient:
    return tuple(torch.zeros_like(value) for value in gradient)


def gradient_normalize(
    gradient: Gradient,
    eps: float = 1e-12,
) -> Gradient:
    norm = gradient_norm(gradient)
    if norm <= float(eps):
        raise RuntimeError("cannot normalize a zero gradient")
    return gradient_scale(gradient, 1.0 / norm)


def _group_dot(
    left: Gradient,
    right: Gradient,
    records: Sequence[ParameterRecordV923],
    group: str,
) -> float:
    total = 0.0
    found = False
    for first, second, record in zip(left, right, records):
        if record.group != group:
            continue
        found = True
        total += float(torch.sum(first.double() * second.double()).item())
    if not found:
        raise ValueError(f"unknown parameter group: {group}")
    return total


def _active_task_masks(labels: torch.Tensor) -> Dict[str, torch.Tensor]:
    labels = labels.view(-1)
    regions = region_index(labels)
    result = {"global": torch.ones_like(regions, dtype=torch.bool)}
    for index, name in enumerate(REGION_NAMES):
        result[name] = regions == int(index)
    return result


def collect_mean_mae_gradients(
    model: torch.nn.Module,
    loader: Iterable[object],
    records: Sequence[ParameterRecordV923],
    forward_batch: Callable[
        [torch.nn.Module, object],
        tuple[torch.Tensor, torch.Tensor],
    ],
) -> Dict[str, object]:
    """Accumulate exact dataset-mean task gradients without updates."""
    parameters = [record.parameter for record in records]
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("selected parameter already has a populated .grad")

    model.eval()
    gradient_sums = {name: _zeros(records) for name in TASK_NAMES}
    sample_counts = {name: 0 for name in TASK_NAMES}
    loss_sums = {name: 0.0 for name in TASK_NAMES}
    batch_count = 0

    with torch.enable_grad():
        for batch in loader:
            prediction, labels = forward_batch(model, batch)
            prediction = prediction.view(-1)
            labels = labels.view(-1).to(prediction)
            if len(prediction) != len(labels):
                raise ValueError("prediction/label batch length mismatch")
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("non-finite anchor prediction")
            errors = torch.abs(prediction - labels)
            masks = _active_task_masks(labels)
            active = [
                name for name in TASK_NAMES if bool(masks[name].any())
            ]
            for offset, task in enumerate(active):
                mask = masks[task]
                loss_sum = errors[mask].sum()
                gradients = torch.autograd.grad(
                    loss_sum,
                    parameters,
                    retain_graph=offset < len(active) - 1,
                    create_graph=False,
                    allow_unused=True,
                )
                accumulated = list(gradient_sums[task])
                for index, gradient in enumerate(gradients):
                    if gradient is not None:
                        accumulated[index].add_(
                            gradient.detach().cpu().float()
                        )
                gradient_sums[task] = tuple(accumulated)
                count = int(mask.sum().item())
                sample_counts[task] += count
                loss_sums[task] += float(loss_sum.detach().item())
            batch_count += 1
            del prediction, labels, errors

    if batch_count == 0:
        raise RuntimeError("gradient audit loader is empty")
    missing = [
        name for name in TASK_NAMES if sample_counts[name] == 0
    ]
    if missing:
        raise RuntimeError(
            f"gradient audit tasks have no samples: {missing}"
        )
    if any(parameter.grad is not None for parameter in parameters):
        raise RuntimeError("autograd.grad unexpectedly populated .grad")

    gradients = {
        name: gradient_scale(
            gradient_sums[name],
            1.0 / sample_counts[name],
        )
        for name in TASK_NAMES
    }
    if any(
        gradient_norm(value) <= 0.0 for value in gradients.values()
    ):
        raise RuntimeError("one or more task gradients are zero")
    return {
        "gradients": gradients,
        "sample_counts": sample_counts,
        "mean_losses": {
            name: loss_sums[name] / sample_counts[name]
            for name in TASK_NAMES
        },
        "batch_count": batch_count,
    }


def pairwise_geometry_rows(
    gradients: Mapping[str, Gradient],
    records: Sequence[ParameterRecordV923],
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    overall: list[Dict[str, object]] = []
    layerwise: list[Dict[str, object]] = []
    tolerance = 1e-12
    groups = sorted({record.group for record in records})
    for left_name in TASK_NAMES:
        for right_name in TASK_NAMES:
            left = gradients[left_name]
            right = gradients[right_name]
            dot = gradient_dot(left, right)
            left_norm = gradient_norm(left)
            right_norm = gradient_norm(right)
            denominator = max(
                left_norm * right_norm,
                tolerance,
            )
            cosine = dot / denominator
            overall.append(
                {
                    "left_task": left_name,
                    "right_task": right_name,
                    "dot": dot,
                    "cosine": cosine,
                    "conflict": bool(dot < 0.0),
                }
            )
            for group in groups:
                group_dot = _group_dot(left, right, records, group)
                left_sq = _group_dot(left, left, records, group)
                right_sq = _group_dot(right, right, records, group)
                group_denominator = max(
                    max(left_sq, 0.0) ** 0.5
                    * max(right_sq, 0.0) ** 0.5,
                    tolerance,
                )
                layerwise.append(
                    {
                        "parameter_group": group,
                        "left_task": left_name,
                        "right_task": right_name,
                        "dot": group_dot,
                        "cosine": group_dot / group_denominator,
                        "conflict": bool(group_dot < 0.0),
                        "left_norm": max(left_sq, 0.0) ** 0.5,
                        "right_norm": max(right_sq, 0.0) ** 0.5,
                    }
                )
    return overall, layerwise


def deterministic_symmetric_pcgrad(
    gradients: Mapping[str, Gradient],
    task_names: Sequence[str] = REGION_TASK_NAMES,
) -> Gradient:
    """Average forward- and reverse-order PCGrad projections."""
    names = tuple(task_names)
    originals = {name: gradients[name] for name in names}

    def one_order(reverse: bool) -> Dict[str, Gradient]:
        projected: Dict[str, Gradient] = {}
        ordered_references = (
            tuple(reversed(names)) if reverse else names
        )
        for name in names:
            current = _clone(originals[name])
            for other in ordered_references:
                if other == name:
                    continue
                reference = originals[other]
                dot = gradient_dot(current, reference)
                denominator = gradient_dot(reference, reference)
                if dot < 0.0 and denominator > 1e-20:
                    current = gradient_add(
                        current,
                        reference,
                        scale=-dot / denominator,
                    )
            projected[name] = current
        return projected

    forward = one_order(False)
    reverse = one_order(True)
    symmetric = {
        name: gradient_scale(
            gradient_add(forward[name], reverse[name]),
            0.5,
        )
        for name in names
    }
    return gradient_mean([symmetric[name] for name in names])


def fit_normalized_mgda(
    gradients: Mapping[str, Gradient],
    config: GradientAuditConfigV923,
    task_names: Sequence[str] = REGION_TASK_NAMES,
) -> Dict[str, object]:
    """Find the minimum-norm convex combination of unit gradients."""
    names = tuple(task_names)
    normalized = [
        gradient_normalize(gradients[name]) for name in names
    ]
    gram = np.asarray(
        [
            [gradient_dot(left, right) for right in normalized]
            for left in normalized
        ],
        dtype=np.float64,
    )
    gram = 0.5 * (gram + gram.T)
    count = len(names)
    initial = np.full(count, 1.0 / count, dtype=np.float64)

    def objective(alpha: np.ndarray) -> float:
        return float(0.5 * alpha @ gram @ alpha)

    def jacobian(alpha: np.ndarray) -> np.ndarray:
        return gram @ alpha

    result = minimize(
        objective,
        initial,
        jac=jacobian,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * count,
        constraints=[
            {
                "type": "eq",
                "fun": lambda value: value.sum() - 1.0,
            }
        ],
        options={
            "maxiter": int(config.mgda_max_iterations),
            "ftol": float(config.mgda_ftol),
            "disp": False,
        },
    )
    if not result.success:
        raise RuntimeError(
            f"MGDA optimization failed: {result.message}"
        )
    alpha = np.clip(result.x, 0.0, 1.0)
    alpha /= alpha.sum()
    direction = _zeros_from_gradient(normalized[0])
    for weight, gradient in zip(alpha.tolist(), normalized):
        direction = gradient_add(
            direction,
            gradient,
            scale=weight,
        )
    return {
        "task_names": names,
        "weights": alpha,
        "direction": direction,
        "direction_norm": gradient_norm(direction),
        "gram": gram,
        "objective": objective(alpha),
    }


def build_candidate_directions(
    gradients: Mapping[str, Gradient],
    config: GradientAuditConfigV923,
) -> Dict[str, object]:
    mgda = fit_normalized_mgda(gradients, config)
    directions = {
        "global_gradient": gradients["global"],
        "equal_region_mean": gradient_mean(
            [gradients[name] for name in REGION_TASK_NAMES]
        ),
        "pcgrad_regions": deterministic_symmetric_pcgrad(gradients),
        "mgda_normalized_regions": mgda["direction"],
    }
    return {"directions": directions, "mgda": mgda}


def direction_effect_rows(
    gradients: Mapping[str, Gradient],
    directions: Mapping[str, Gradient],
    tolerance: float,
) -> tuple[list[Dict[str, object]], list[Dict[str, object]]]:
    task_rows: list[Dict[str, object]] = []
    summary_rows: list[Dict[str, object]] = []
    for direction_name, direction in directions.items():
        direction_norm = gradient_norm(direction)
        if direction_norm <= 0.0:
            raise RuntimeError(
                f"zero candidate direction: {direction_name}"
            )
        local_rows = []
        for task in TASK_NAMES:
            task_norm = gradient_norm(gradients[task])
            dot = gradient_dot(gradients[task], direction)
            cosine = dot / max(
                task_norm * direction_norm,
                1e-12,
            )
            row = {
                "direction": direction_name,
                "task": task,
                "dot": dot,
                "cosine": cosine,
                "first_order_improvement_per_unit_step": (
                    dot / direction_norm
                ),
                "would_improve": bool(cosine > float(tolerance)),
            }
            task_rows.append(row)
            local_rows.append(row)
        region_rows = [
            row
            for row in local_rows
            if row["task"] in REGION_TASK_NAMES
        ]
        global_row = next(
            row for row in local_rows if row["task"] == "global"
        )
        summary_rows.append(
            {
                "direction": direction_name,
                "direction_norm": direction_norm,
                "region_improved_count": int(
                    sum(
                        bool(row["would_improve"])
                        for row in region_rows
                    )
                ),
                "all_regions_improve": bool(
                    all(
                        bool(row["would_improve"])
                        for row in region_rows
                    )
                ),
                "global_improves": bool(
                    global_row["would_improve"]
                ),
                "all_tasks_improve": bool(
                    all(
                        bool(row["would_improve"])
                        for row in local_rows
                    )
                ),
                "minimum_region_cosine": float(
                    min(
                        float(row["cosine"])
                        for row in region_rows
                    )
                ),
                "mean_region_cosine": float(
                    np.mean(
                        [
                            float(row["cosine"])
                            for row in region_rows
                        ]
                    )
                ),
                "global_cosine": float(global_row["cosine"]),
                "worst_region": min(
                    region_rows,
                    key=lambda row: float(row["cosine"]),
                )["task"],
            }
        )
    return task_rows, summary_rows


def global_region_decomposition_error(
    gradients: Mapping[str, Gradient],
    sample_counts: Mapping[str, int],
) -> float:
    total = int(sample_counts["global"])
    reconstructed = _zeros_from_gradient(gradients["global"])
    for name in REGION_TASK_NAMES:
        reconstructed = gradient_add(
            reconstructed,
            gradients[name],
            scale=float(sample_counts[name]) / total,
        )
    difference = gradient_add(
        gradients["global"],
        reconstructed,
        scale=-1.0,
    )
    return gradient_norm(difference) / max(
        gradient_norm(gradients["global"]),
        1e-12,
    )


def task_stat_rows(
    gradients: Mapping[str, Gradient],
    sample_counts: Mapping[str, int],
    mean_losses: Mapping[str, float],
) -> list[Dict[str, object]]:
    total = int(sample_counts["global"])
    global_norm = gradient_norm(gradients["global"])
    rows = []
    for task in TASK_NAMES:
        norm = gradient_norm(gradients[task])
        share = (
            1.0
            if task == "global"
            else float(sample_counts[task]) / total
        )
        rows.append(
            {
                "task": task,
                "sample_count": int(sample_counts[task]),
                "sample_share": share,
                "mean_mae": float(mean_losses[task]),
                "gradient_norm": norm,
                "norm_ratio_to_global": (
                    norm / max(global_norm, 1e-12)
                ),
                "weighted_contribution_norm": norm * share,
                "weighted_contribution_ratio_to_global": (
                    norm * share / max(global_norm, 1e-12)
                ),
            }
        )
    return rows


def fold_geometry_summary(
    pairwise_rows: Sequence[Mapping[str, object]],
    direction_summary_rows: Sequence[Mapping[str, object]],
    mgda: Mapping[str, object],
    decomposition_error: float,
    config: GradientAuditConfigV923,
) -> Dict[str, object]:
    region_pairs = [
        row
        for row in pairwise_rows
        if row["left_task"] in REGION_TASK_NAMES
        and row["right_task"] in REGION_TASK_NAMES
        and REGION_TASK_NAMES.index(str(row["left_task"]))
        < REGION_TASK_NAMES.index(str(row["right_task"]))
    ]
    global_pairs = [
        row
        for row in pairwise_rows
        if row["left_task"] == "global"
        and row["right_task"] in REGION_TASK_NAMES
    ]
    direction_map = {
        str(row["direction"]): row
        for row in direction_summary_rows
    }
    mgda_row = direction_map["mgda_normalized_regions"]
    feasible = bool(
        mgda_row["all_regions_improve"]
        and mgda_row["global_improves"]
        and float(mgda_row["minimum_region_cosine"])
        >= float(config.common_cosine_margin)
        and float(mgda["direction_norm"])
        >= float(config.minimum_mgda_norm)
    )
    return {
        "region_pair_conflict_fraction": float(
            np.mean(
                [bool(row["conflict"]) for row in region_pairs]
            )
        ),
        "region_pair_mean_cosine": float(
            np.mean(
                [float(row["cosine"]) for row in region_pairs]
            )
        ),
        "global_conflicting_region_count": int(
            sum(bool(row["conflict"]) for row in global_pairs)
        ),
        "global_region_decomposition_relative_error": float(
            decomposition_error
        ),
        "equal_mean_all_regions_improve": bool(
            direction_map["equal_region_mean"]
            ["all_regions_improve"]
        ),
        "pcgrad_all_regions_improve": bool(
            direction_map["pcgrad_regions"]
            ["all_regions_improve"]
        ),
        "mgda_all_regions_improve": bool(
            mgda_row["all_regions_improve"]
        ),
        "mgda_global_improves": bool(
            mgda_row["global_improves"]
        ),
        "mgda_minimum_region_cosine": float(
            mgda_row["minimum_region_cosine"]
        ),
        "mgda_global_cosine": float(mgda_row["global_cosine"]),
        "mgda_direction_norm": float(mgda["direction_norm"]),
        "fold_geometry_feasible": feasible,
    }


__all__ = [
    "AUDIT_VERSION",
    "TASK_NAMES",
    "REGION_TASK_NAMES",
    "GradientAuditConfigV923",
    "ParameterRecordV923",
    "select_parameter_records",
    "selected_parameter_fingerprint",
    "gradient_dot",
    "gradient_norm",
    "gradient_scale",
    "gradient_add",
    "gradient_mean",
    "collect_mean_mae_gradients",
    "pairwise_geometry_rows",
    "deterministic_symmetric_pcgrad",
    "fit_normalized_mgda",
    "build_candidate_directions",
    "direction_effect_rows",
    "global_region_decomposition_error",
    "task_stat_rows",
    "fold_geometry_summary",
]
