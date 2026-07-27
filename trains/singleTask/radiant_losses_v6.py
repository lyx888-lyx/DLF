"""Losses and diagnostics for RADIANT-DLF V6."""

from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .HingeLoss import HingeLoss
from .model.RADIANT_DLF import MODE_NAMES


REGION_NAMES = (
    "strong_negative",
    "negative",
    "boundary",
    "positive",
    "strong_positive",
)


class _MSE(nn.Module):
    def forward(self, prediction, target):
        difference = target - prediction
        return difference.square().sum() / max(1, difference.numel())


def region_index(labels: torch.Tensor) -> torch.Tensor:
    values = labels.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


def safe_corr(prediction: torch.Tensor, target: torch.Tensor) -> float:
    x = prediction.view(-1).double()
    y = target.view(-1).double()
    x = x - x.mean()
    y = y - y.mean()
    denominator = torch.sqrt(x.square().sum() * y.square().sum()).clamp_min(1e-12)
    return float((x * y).sum().div(denominator).item())


def balanced_sign_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    predicted = prediction.view(-1) >= 0
    truth = target.view(-1) >= 0
    recalls = []
    for value in (False, True):
        mask = truth == value
        if mask.any():
            recalls.append((predicted[mask] == truth[mask]).float().mean())
    return float(torch.stack(recalls).mean().item()) if recalls else 0.0


def discrete_energy_score(
    candidates: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target_distance = torch.abs(candidates - target)
    first = (weights * target_distance).sum(dim=1)
    pairwise = torch.abs(candidates.unsqueeze(2) - candidates.unsqueeze(1))
    pair_weight = weights.unsqueeze(2) * weights.unsqueeze(1)
    second = 0.5 * (pair_weight * pairwise).sum(dim=(1, 2))
    return (first - second).mean()


def latent_energy_score(
    candidates: torch.Tensor,
    weights: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    target_distance = torch.mean(
        torch.abs(candidates - target.unsqueeze(1)), dim=2
    )
    first = (weights * target_distance).sum(dim=1)
    pairwise = torch.mean(
        torch.abs(candidates.unsqueeze(2) - candidates.unsqueeze(1)), dim=3
    )
    pair_weight = weights.unsqueeze(2) * weights.unsqueeze(1)
    second = 0.5 * (pair_weight * pairwise).sum(dim=(1, 2))
    return (first - second).mean()


def dlf_backbone_loss(output: Mapping[str, torch.Tensor], labels: torch.Tensor, dataset: str):
    l1 = F.l1_loss
    task = (
        l1(output["output_logit"], labels)
        + l1(output["logits_c"], labels)
        + 3.0 * l1(output["logits_l_hetero"], labels)
        + l1(output["logits_v_hetero"], labels)
        + l1(output["logits_a_hetero"], labels)
    )
    mse = _MSE()
    reconstruction = (
        mse(output["recon_l"], output["origin_l"])
        + mse(output["recon_v"], output["origin_v"])
        + mse(output["recon_a"], output["origin_a"])
    )
    specific = (
        mse(output["s_l"].permute(1, 2, 0), output["s_l_r"])
        + mse(output["s_v"].permute(1, 2, 0), output["s_v_r"])
        + mse(output["s_a"].permute(1, 2, 0), output["s_a_r"])
    )
    feature_width = 50 if dataset == "mosi" else 10
    cosine = nn.CosineEmbeddingLoss()
    s_l_flat = output["s_l"].reshape(-1, feature_width)
    s_v_flat = output["s_v"].reshape(-1, feature_width)
    s_a_flat = output["s_a"].reshape(-1, feature_width)
    negative_l = torch.full((s_l_flat.size(0),), -1.0, device=labels.device)
    negative_v = torch.full((s_v_flat.size(0),), -1.0, device=labels.device)
    negative_a = torch.full((s_a_flat.size(0),), -1.0, device=labels.device)
    orthogonal = (
        cosine(s_l_flat, output["c_l"].reshape(-1, feature_width), negative_l)
        + cosine(s_v_flat, output["c_v"].reshape(-1, feature_width), negative_v)
        + cosine(s_a_flat, output["c_a"].reshape(-1, feature_width), negative_a)
    )
    features, identities = [], []
    for index in range(labels.size(0)):
        for value in (
            output["c_l_sim"][index],
            output["c_v_sim"][index],
            output["c_a_sim"][index],
        ):
            features.append(value.view(1, -1))
            identities.append(labels[index].view(1, -1))
    similarity = HingeLoss()(torch.cat(identities, dim=0), torch.cat(features, dim=0))
    return task + 0.1 * (specific + reconstruction + 0.1 * (similarity + orthogonal))


def _region_balanced_mae(prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    regions = region_index(labels)
    terms = []
    for index in range(len(REGION_NAMES)):
        mask = regions == index
        if mask.any():
            terms.append(torch.abs(prediction[mask] - labels[mask]).mean())
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def _region_bias(prediction: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    regions = region_index(labels)
    terms = []
    for index in range(len(REGION_NAMES)):
        mask = regions == index
        if mask.any():
            terms.append((labels[mask] - prediction[mask]).mean().square())
    return torch.stack(terms).mean() if terms else prediction.new_zeros(())


def _mode_monotonic_loss(views: Mapping[str, Mapping[str, torch.Tensor]], labels: torch.Tensor):
    errors = {
        mode: torch.abs(view["mean_prediction"] - labels)
        for mode, view in views.items()
    }
    comparisons = []
    for richer, poorer in (
        ("lav", "la"),
        ("lav", "lv"),
        ("la", "l"),
        ("lv", "l"),
        ("lav", "l"),
    ):
        if richer in errors and poorer in errors:
            comparisons.append(F.relu(errors[richer] - errors[poorer] + 0.01).mean())
    return torch.stack(comparisons).mean() if comparisons else labels.new_zeros(())


def _voi_targets(views: Mapping[str, Mapping[str, torch.Tensor]], labels: torch.Tensor):
    error = {
        mode: torch.abs(view["mean_prediction"].detach() - labels)
        for mode, view in views.items()
    }
    targets = {}
    if all(name in error for name in ("l", "la", "lv", "lav")):
        targets["l"] = torch.cat(
            [error["l"] - error["la"], error["l"] - error["lv"], error["l"] - error["lav"]],
            dim=1,
        )
        targets["la"] = torch.cat(
            [torch.zeros_like(error["la"]), error["la"] - error["lav"], error["la"] - error["lav"]],
            dim=1,
        )
        targets["lv"] = torch.cat(
            [error["lv"] - error["lav"], torch.zeros_like(error["lv"]), error["lv"] - error["lav"]],
            dim=1,
        )
        targets["lav"] = torch.zeros_like(views["lav"]["voi"])
    return targets


def radiant_loss(
    views: Mapping[str, Mapping[str, torch.Tensor]],
    labels: torch.Tensor,
    dataset: str,
    weights: Mapping[str, float],
    include_backbone_loss: bool,
) -> Dict[str, torch.Tensor]:
    posterior_energy = []
    posterior_mean = []
    latent_energy = []
    latent_mean = []
    support = []
    sign = []
    scale = []
    region = []
    bias = []
    for mode in MODE_NAMES:
        if mode not in views:
            continue
        view = views[mode]
        posterior_energy.append(
            discrete_energy_score(
                view["candidate_predictions"], view["component_weights"], labels
            )
        )
        posterior_mean.append(F.smooth_l1_loss(view["mean_prediction"], labels))
        latent_energy.append(
            latent_energy_score(
                view["latent_candidates"],
                view["component_weights"],
                view["target_full"].detach(),
            )
        )
        latent_mean.append(
            F.smooth_l1_loss(view["expected_latent"], view["target_full"].detach())
        )
        support.append(
            torch.abs(view["candidate_predictions"] - labels).min(dim=1).values.mean()
        )
        residual_target = labels - view["anchor"].detach()
        sign.append(
            F.binary_cross_entropy_with_logits(
                view["sign_logit"], (residual_target >= 0).float()
            )
        )
        absolute_error = torch.abs(view["mean_prediction"].detach() - labels)
        scale.append(F.smooth_l1_loss(view["predicted_scale"], absolute_error))
        region.append(_region_balanced_mae(view["mean_prediction"], labels))
        bias.append(_region_bias(view["mean_prediction"], labels))

    def average(items: Iterable[torch.Tensor]):
        items = list(items)
        return torch.stack(items).mean() if items else labels.new_zeros(())

    voi_target = _voi_targets(views, labels)
    voi_terms = [
        F.smooth_l1_loss(views[mode]["voi"], target)
        for mode, target in voi_target.items()
    ]
    losses = {
        "posterior_energy": average(posterior_energy),
        "posterior_mean": average(posterior_mean),
        "latent_energy": average(latent_energy),
        "latent_mean": average(latent_mean),
        "support": average(support),
        "sign": average(sign),
        "scale": average(scale),
        "region": average(region),
        "bias": average(bias),
        "monotonic": _mode_monotonic_loss(views, labels),
        "voi": average(voi_terms),
        "backbone": (
            dlf_backbone_loss(views["lav"]["backbone"], labels, dataset)
            if include_backbone_loss
            else labels.new_zeros(())
        ),
    }
    total = labels.new_zeros(())
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["total"] = total
    return losses


def posterior_diagnostics(view: Mapping[str, torch.Tensor], labels: torch.Tensor):
    residual_target = labels - view["anchor"]
    residual_prediction = view["mean_prediction"] - view["anchor"]
    error = torch.abs(view["mean_prediction"] - labels)
    support_error = torch.abs(view["candidate_predictions"] - labels).min(dim=1).values
    return {
        "mae": float(error.mean().item()),
        "anchor_mae": float(torch.abs(view["anchor"] - labels).mean().item()),
        "residual_corr": safe_corr(residual_prediction, residual_target),
        "sign_balanced_accuracy": balanced_sign_accuracy(
            residual_prediction, residual_target
        ),
        "posterior_support_mae": float(support_error.mean().item()),
        "mean_entropy": float(view["entropy"].mean().item()),
        "mean_spread": float(view["spread"].mean().item()),
        "scale_error_corr": safe_corr(view["predicted_scale"], error),
    }
