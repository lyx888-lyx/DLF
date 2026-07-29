"""Training and evaluation for region-balanced ordinal soft-mixture V9."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from .complementarity_v71 import (
    apply_global_committee,
    apply_region_committee,
    fit_committee_cv,
)
from .expert_analysis import normalize_batch_ids
from .ordinal_consistency_system_v74 import _fast_metrics, _fold_metrics


LOGGER = logging.getLogger("MMSA")
THRESHOLDS_7 = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
THRESHOLDS_5 = (-1.5, -0.5, 0.5, 1.5)
REGION_NAMES = (
    "strong_negative",
    "ordinary_negative",
    "neutral",
    "ordinary_positive",
    "strong_positive",
)


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _cache_index(sample_ids):
    mapping = {}
    for index, value in enumerate(sample_ids):
        key = str(value)
        if key in mapping:
            raise RuntimeError(f"Duplicate sample id in teacher cache: {key}")
        mapping[key] = index
    return mapping


def _batch_cache_indices(batch_ids, mapping):
    values = normalize_batch_ids(batch_ids)
    try:
        return torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    except KeyError as error:
        raise KeyError(f"Sample id missing from teacher cache: {error}") from error


def sentiment_region_ids(
    labels: torch.Tensor,
    strong_threshold: float = 1.0,
    neutral_radius: float = 1e-6,
) -> torch.Tensor:
    """Map continuous labels into five auditable sentiment regions."""
    values = labels.view(-1)
    strong = float(strong_threshold)
    neutral = max(float(neutral_radius), 0.0)
    region = torch.full_like(values, 2, dtype=torch.long)
    region[values < -strong] = 0
    region[(values >= -strong) & (values < -neutral)] = 1
    region[(values > neutral) & (values <= strong)] = 3
    region[values > strong] = 4
    return region


def polarity_targets(labels: torch.Tensor, neutral_radius: float = 1e-6):
    values = labels.view(-1)
    neutral = max(float(neutral_radius), 0.0)
    target = torch.full_like(values, 1, dtype=torch.long)
    target[values < -neutral] = 0
    target[values > neutral] = 2
    return target


def _manual_huber(error: torch.Tensor, delta: float):
    delta = max(float(delta), 1e-6)
    absolute = error.abs()
    quadratic = torch.minimum(absolute, absolute.new_tensor(delta))
    linear = absolute - quadratic
    return 0.5 * quadratic.square() / delta + linear


def build_region_weights(
    labels: torch.Tensor,
    strong_threshold: float,
    neutral_radius: float,
    mode: str = "inverse_sqrt",
    min_weight: float = 0.50,
    max_weight: float = 2.00,
    effective_beta: float = 0.999,
):
    region = sentiment_region_ids(labels, strong_threshold, neutral_radius)
    counts = torch.bincount(region, minlength=5).float()
    if bool((counts == 0).any()):
        missing = [REGION_NAMES[i] for i in range(5) if counts[i] == 0]
        raise RuntimeError(f"Training split has empty sentiment regions: {missing}")
    mode = str(mode).lower()
    if mode == "none":
        weights = torch.ones_like(counts)
    elif mode == "inverse_sqrt":
        weights = counts.rsqrt()
    elif mode == "effective_number":
        beta = min(max(float(effective_beta), 0.0), 0.999999)
        weights = (1.0 - beta) / (1.0 - counts.new_tensor(beta).pow(counts))
    else:
        raise ValueError(f"Unknown region weighting mode: {mode}")

    # Normalize the sample-frequency-weighted mean to one, then clip.
    weights = weights / ((weights * counts).sum() / counts.sum()).clamp_min(1e-8)
    weights = weights.clamp(float(min_weight), float(max_weight))
    weights = weights / ((weights * counts).sum() / counts.sum()).clamp_min(1e-8)
    return counts, weights


def region_diagnostics(
    prediction: torch.Tensor,
    labels: torch.Tensor,
    strong_threshold: float,
    neutral_radius: float,
):
    prediction = prediction.view(-1).detach().cpu()
    labels = labels.view(-1).detach().cpu()
    region = sentiment_region_ids(labels, strong_threshold, neutral_radius)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = region == index
        count = int(mask.sum().item())
        if count == 0:
            rows.append({
                "region": name,
                "count": 0,
                "label_mean": float("nan"),
                "prediction_mean": float("nan"),
                "signed_bias": float("nan"),
                "mae": float("nan"),
                "rmse": float("nan"),
                "cross_zero_rate": float("nan"),
            })
            continue
        y = labels[mask]
        p = prediction[mask]
        error = p - y
        if index in (0, 1):
            cross = (p > 0.0).float().mean()
        elif index in (3, 4):
            cross = (p < 0.0).float().mean()
        else:
            cross = (p.abs() > max(float(neutral_radius), 1e-6)).float().mean()
        rows.append({
            "region": name,
            "count": count,
            "label_mean": float(y.mean().item()),
            "prediction_mean": float(p.mean().item()),
            "signed_bias": float(error.mean().item()),
            "mae": float(error.abs().mean().item()),
            "rmse": float(error.square().mean().sqrt().item()),
            "cross_zero_rate": float(cross.item()),
        })

    available = [row for row in rows if row["count"] > 0]
    ordinary_positive = rows[3]
    aggregate = {
        "worst_region_mae": max(row["mae"] for row in available),
        "ordinary_positive_mae": ordinary_positive["mae"],
        "ordinary_positive_signed_bias": ordinary_positive["signed_bias"],
        "ordinary_positive_cross_zero_rate": ordinary_positive["cross_zero_rate"],
    }
    return aggregate, rows


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor):
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


class RegionBalancedOrdinalMixtureTrainerV9:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths: Sequence[Path],
        init_index: int,
        source_summary: Mapping[str, object],
        max_epochs: int = 16,
        early_stop: int = 5,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-3,
        strong_threshold: float = 1.0,
        neutral_radius: float = 1e-6,
        region_weight_mode: str = "inverse_sqrt",
        region_weight_min: float = 0.50,
        region_weight_max: float = 2.00,
        effective_beta: float = 0.999,
        huber_delta: float = 0.50,
        group_dro_weight: float = 0.0,
        group_dro_eta: float = 0.05,
        valid_mae_tolerance: float = 0.001,
        fold_mae_tolerance: float = 0.010,
        worst_region_tolerance: float = 0.10,
        robust_selection_weight: float = 0.05,
        ordinary_positive_selection_weight: float = 0.02,
        stability_selection_weight: float = 0.05,
        folds: int = 3,
        beta_grid=(0.40, 0.50, 0.60),
        alpha_grid=(0.50, 0.75, 1.00),
        loss_weights=None,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache = teacher_cache
        self.teacher_paths = [Path(value) for value in teacher_paths]
        self.init_index = int(init_index)
        self.source_summary = dict(source_summary)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.strong_threshold = float(strong_threshold)
        self.neutral_radius = float(neutral_radius)
        self.huber_delta = float(huber_delta)
        self.group_dro_weight = float(group_dro_weight)
        self.group_dro_eta = float(group_dro_eta)
        self.valid_mae_tolerance = float(valid_mae_tolerance)
        self.fold_mae_tolerance = float(fold_mae_tolerance)
        self.worst_region_tolerance = float(worst_region_tolerance)
        self.robust_selection_weight = float(robust_selection_weight)
        self.ordinary_positive_selection_weight = float(
            ordinary_positive_selection_weight
        )
        self.stability_selection_weight = float(stability_selection_weight)
        self.folds = int(folds)
        self.beta_grid = tuple(float(value) for value in beta_grid)
        self.alpha_grid = tuple(float(value) for value in alpha_grid)
        self.loss_weights = loss_weights or {
            "supervised": 1.00,
            "group_dro": 1.00,
            "polarity": 0.20,
            "expert": 0.20,
            "ordinal7": 0.15,
            "ordinal5": 0.10,
            "ordinal_mae": 0.10,
            "ordinal_consistency": 0.08,
            "cross_zero": 0.10,
            "legacy_preservation": 0.04,
            "blend_regularization": 0.01,
        }
        self.index = {
            split: _cache_index(self.cache["splits"][split]["sample_ids"])
            for split in ("train", "valid", "test")
        }
        counts, weights = build_region_weights(
            self.cache["splits"]["train"]["labels"],
            self.strong_threshold,
            self.neutral_radius,
            mode=region_weight_mode,
            min_weight=region_weight_min,
            max_weight=region_weight_max,
            effective_beta=effective_beta,
        )
        self.region_counts = counts
        self.region_weights = weights
        self.dro_logits = torch.zeros(5)
        pd.DataFrame({
            "region": REGION_NAMES,
            "count": counts.tolist(),
            "weight": weights.tolist(),
        }).to_csv(self.save_dir / "v9_train_region_weights.csv", index=False)
        self.committee = self._fit_committee()

    def _fit_committee(self):
        valid = self.cache["splits"]["valid"]
        anchor = valid["predictions"][:, self.init_index]
        fitted = fit_committee_cv(
            valid["predictions"],
            valid["labels"],
            anchor,
            valid["sample_ids"],
            temperature=0.55,
            steps=600,
        )
        pd.DataFrame(fitted["cv_rows"]).to_csv(
            self.save_dir / "v9_committee_cv.csv", index=False
        )
        return fitted

    def _committee_predictions(self, split_name):
        split = self.cache["splits"][split_name]
        predictions = split["predictions"].float()
        anchor = predictions[:, self.init_index]
        return {
            "global_simplex": apply_global_committee(
                predictions, self.committee["global_weights"]
            ),
            "region_simplex": apply_region_committee(
                predictions,
                anchor,
                self.committee["region_weights"],
                0.55,
            ),
            "anchor": anchor,
        }

    def _optimizer(self, model):
        parameters = model.trainable_parameters()
        if not parameters:
            raise RuntimeError("V9 has no trainable mixture parameters.")
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _loss(self, output, labels):
        labels = labels.view(-1, 1)
        region = sentiment_region_ids(
            labels, self.strong_threshold, self.neutral_radius
        )
        polarity = polarity_targets(labels, self.neutral_radius)
        sample_weights = self.region_weights.to(labels.device)[region].view(-1, 1)

        prediction_error = output["prediction"] - labels
        per_sample_huber = _manual_huber(prediction_error, self.huber_delta)
        supervised = _weighted_mean(per_sample_huber, sample_weights)

        group_means = labels.new_zeros(5)
        present = torch.zeros(5, dtype=torch.bool, device=labels.device)
        for index in range(5):
            mask = region == index
            if bool(mask.any()):
                group_means[index] = per_sample_huber.view(-1)[mask].mean()
                present[index] = True
        if bool(present.any()):
            dro_logits = self.dro_logits.to(labels.device)
            masked_logits = dro_logits[present]
            dro_probs = torch.softmax(masked_logits, dim=0)
            group_dro = (dro_probs * group_means[present]).sum()
        else:
            group_dro = supervised.new_zeros(())

        polarity_loss = F.cross_entropy(
            output["polarity_logits"], polarity, reduction="none"
        ).view(-1, 1)
        polarity_loss = _weighted_mean(polarity_loss, sample_weights)

        selected_expert = output["expert_values"].gather(
            1, polarity.view(-1, 1)
        )
        expert_loss = _weighted_mean(
            _manual_huber(selected_expert - labels, self.huber_delta),
            sample_weights,
        )

        thresholds7 = labels.new_tensor(THRESHOLDS_7).view(1, -1)
        thresholds5 = labels.new_tensor(THRESHOLDS_5).view(1, -1)
        target7 = (labels > thresholds7).float()
        target5 = (labels.clamp(-2.0, 2.0) > thresholds5).float()
        ordinal7 = F.binary_cross_entropy_with_logits(
            output["ordinal7_logits"], target7, reduction="none"
        ).mean(dim=1, keepdim=True)
        ordinal5 = F.binary_cross_entropy_with_logits(
            output["ordinal5_logits"], target5, reduction="none"
        ).mean(dim=1, keepdim=True)
        ordinal7 = _weighted_mean(ordinal7, sample_weights)
        ordinal5 = _weighted_mean(ordinal5, sample_weights)
        ordinal_mae = _weighted_mean(
            _manual_huber(output["ordinal_consensus"] - labels, self.huber_delta),
            sample_weights,
        )
        ordinal_consistency = _weighted_mean(
            _manual_huber(
                output["mixture_value"] - output["ordinal_consensus"],
                self.huber_delta,
            ),
            sample_weights,
        )

        sign = torch.sign(labels)
        non_neutral = (labels.abs() > max(self.neutral_radius, 1e-6)).float()
        distance_weight = (
            labels.abs() / max(self.strong_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        cross_zero = F.relu(-sign * output["prediction"])
        cross_weight = sample_weights * non_neutral * distance_weight
        cross_zero = (
            (cross_zero * cross_weight).sum()
            / cross_weight.sum().clamp_min(1e-8)
        )

        legacy_preservation = F.smooth_l1_loss(
            output["prediction"], output["legacy_prediction"].detach()
        )
        blend_regularization = output["mixture_blend"].square().mean()
        losses = {
            "supervised": supervised,
            "group_dro": self.group_dro_weight * group_dro,
            "polarity": polarity_loss,
            "expert": expert_loss,
            "ordinal7": ordinal7,
            "ordinal5": ordinal5,
            "ordinal_mae": ordinal_mae,
            "ordinal_consistency": ordinal_consistency,
            "cross_zero": cross_zero,
            "legacy_preservation": legacy_preservation,
            "blend_regularization": blend_regularization,
        }
        total = prediction_error.new_zeros(())
        for name, value in losses.items():
            total = total + float(self.loss_weights.get(name, 0.0)) * value
        losses["total"] = total
        return losses, group_means.detach(), present.detach()

    def _train_epoch(self, model, dataloader, optimizer):
        model.set_train_mode()
        totals: Dict[str, float] = {}
        optimizer.zero_grad()
        for batch in tqdm(dataloader, leave=False):
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            losses, group_means, present = self._loss(output, labels)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()

            if self.group_dro_weight > 0.0 and self.group_dro_eta > 0.0:
                update = torch.zeros_like(self.dro_logits)
                update[present.cpu()] = group_means[present].cpu()
                self.dro_logits += self.group_dro_eta * update
                self.dro_logits -= self.dro_logits.mean()

            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
        count = max(1, len(dataloader))
        row = {name: value / count for name, value in totals.items()}
        for index, name in enumerate(REGION_NAMES):
            row[f"dro_probability_{name}"] = float(
                torch.softmax(self.dro_logits, dim=0)[index].item()
            )
        return row

    @torch.no_grad()
    def collect(self, model, dataloader, split_name):
        model.eval()
        mapping = self.index[split_name]
        split = self.cache["splits"][split_name]
        count = len(split["sample_ids"])
        keys = (
            "base_prediction",
            "legacy_prediction",
            "prediction",
            "mixture_value",
            "polarity_probs",
            "expert_values",
            "ordinal7_value",
            "ordinal5_value",
            "ordinal_consensus",
        )
        buffers = None
        seen = torch.zeros(count, dtype=torch.bool)
        for batch in tqdm(dataloader, leave=False):
            indices = _batch_cache_indices(batch.get("id"), mapping)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            if buffers is None:
                buffers = {
                    key: torch.empty(
                        (count,) + tuple(output[key].shape[1:]),
                        dtype=output[key].dtype,
                    )
                    for key in keys
                }
            for key in keys:
                buffers[key][indices] = output[key].detach().cpu()
            seen[indices] = True
        if buffers is None or not bool(seen.all()):
            raise RuntimeError(
                f"Incomplete V9 collection for {split_name}: "
                f"missing={int((~seen).sum().item())}"
            )
        return {
            "sample_ids": list(split["sample_ids"]),
            "labels": split["labels"].clone(),
            **buffers,
        }

    def _reference_prediction(self, collected, committees):
        policy = self.source_summary["hybrid_policy"]
        alpha = float(self.source_summary["student_policy"]["alpha"])
        legacy = collected["base_prediction"] + alpha * (
            collected["legacy_prediction"] - collected["base_prediction"]
        )
        beta = float(policy["beta"])
        committee_name = str(policy["committee"])
        return beta * committees[committee_name] + (1.0 - beta) * legacy

    def _policy_metrics(self, prediction, labels, sample_ids):
        fast = _fast_metrics(prediction, labels)
        folds = _fold_metrics(prediction, labels, sample_ids, self.folds)
        fold_std = float(
            torch.tensor(folds["fold_maes"]).std(unbiased=False).item()
        )
        region, rows = region_diagnostics(
            prediction,
            labels,
            self.strong_threshold,
            self.neutral_radius,
        )
        return {
            **fast,
            **region,
            "fold_mae_mean": folds["fold_mae_mean"],
            "fold_mae_std": fold_std,
            "fold_maes": folds["fold_maes"],
            "region_rows": rows,
        }

    def _selection_objective(self, metrics):
        return (
            metrics["mae"]
            + self.robust_selection_weight * metrics["worst_region_mae"]
            + self.ordinary_positive_selection_weight
            * metrics["ordinary_positive_mae"]
            + self.stability_selection_weight * metrics["fold_mae_std"]
        )

    def _select_policy(self, collected, committees):
        labels = collected["labels"]
        sample_ids = collected["sample_ids"]
        reference = self._reference_prediction(collected, committees)
        reference_metrics = self._policy_metrics(reference, labels, sample_ids)
        reference_row = {
            "source": "legacy_reference",
            "committee": str(self.source_summary["hybrid_policy"]["committee"]),
            "beta": float(self.source_summary["hybrid_policy"]["beta"]),
            "alpha": float(self.source_summary["student_policy"]["alpha"]),
            "objective": self._selection_objective(reference_metrics),
            "feasible": True,
            **{
                key: value
                for key, value in reference_metrics.items()
                if key not in ("region_rows", "fold_maes")
            },
            "fold_maes": reference_metrics["fold_maes"],
        }
        rows = [reference_row]
        feasible = [reference_row]
        mae_limit = reference_metrics["mae"] + self.valid_mae_tolerance
        fold_limits = [
            value + self.fold_mae_tolerance
            for value in reference_metrics["fold_maes"]
        ]
        worst_limit = (
            reference_metrics["worst_region_mae"] + self.worst_region_tolerance
        )

        source_committee = str(self.source_summary["hybrid_policy"]["committee"])
        committee_names = tuple(dict.fromkeys((
            source_committee,
            "global_simplex",
            "region_simplex",
        )))
        source_beta = float(self.source_summary["hybrid_policy"]["beta"])
        beta_grid = tuple(dict.fromkeys((*self.beta_grid, source_beta)))
        for committee_name in committee_names:
            if committee_name not in committees:
                continue
            for beta in beta_grid:
                for alpha in self.alpha_grid:
                    mixture_student = collected["legacy_prediction"] + alpha * (
                        collected["prediction"] - collected["legacy_prediction"]
                    )
                    prediction = beta * committees[committee_name] + (
                        1.0 - beta
                    ) * mixture_student
                    metrics = self._policy_metrics(prediction, labels, sample_ids)
                    is_feasible = (
                        metrics["mae"] <= mae_limit + 1e-12
                        and metrics["worst_region_mae"] <= worst_limit + 1e-12
                        and all(
                            metrics["fold_maes"][index]
                            <= fold_limits[index] + 1e-12
                            for index in range(self.folds)
                        )
                    )
                    row = {
                        "source": "region_balanced_ordinal_mixture_v9",
                        "committee": committee_name,
                        "beta": float(beta),
                        "alpha": float(alpha),
                        "objective": self._selection_objective(metrics),
                        "feasible": bool(is_feasible),
                        **{
                            key: value
                            for key, value in metrics.items()
                            if key not in ("region_rows", "fold_maes")
                        },
                        "fold_maes": metrics["fold_maes"],
                    }
                    rows.append(row)
                    if is_feasible:
                        feasible.append(row)

        selected = min(
            feasible,
            key=lambda row: (
                row["objective"],
                row["mae"],
                row["worst_region_mae"],
                row["ordinary_positive_mae"],
                row["source"] != "legacy_reference",
            ),
        )
        return selected, rows, reference, reference_metrics

    def train(self, model, dataloaders):
        model.freeze_legacy()
        optimizer = self._optimizer(model)
        history = []
        policy_rows = []
        valid_committees = self._committee_predictions("valid")

        initial_valid = self.collect(model, dataloaders["valid"], "valid")
        initial_policy, initial_rows, _, _ = self._select_policy(
            initial_valid, valid_committees
        )
        policy_rows.extend({"epoch": 0, **row} for row in initial_rows)
        best = {
            "epoch": 0,
            "objective": float(initial_policy["objective"]),
            "state": _cpu_state_dict(model),
            "policy": dict(initial_policy),
        }
        last_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_row = self._train_epoch(model, dataloaders["train"], optimizer)
            valid = self.collect(model, dataloaders["valid"], "valid")
            selected, rows, _, reference_metrics = self._select_policy(
                valid, valid_committees
            )
            policy_rows.extend({"epoch": epoch, **row} for row in rows)
            blend = float(
                model.max_mixture_blend
                * torch.sigmoid(model.mixture_blend_logit.detach()).item()
            )
            row = {
                "epoch": epoch,
                **train_row,
                "selected_source": selected["source"],
                "selected_objective": selected["objective"],
                "selected_mae": selected["mae"],
                "selected_worst_region_mae": selected["worst_region_mae"],
                "selected_ordinary_positive_mae": selected["ordinary_positive_mae"],
                "reference_mae": reference_metrics["mae"],
                "mixture_blend": blend,
            }
            history.append(row)
            LOGGER.info(
                "V9 epoch=%d source=%s Valid(MAE=%.4f worst=%.4f op=%.4f) "
                "blend=%.4f",
                epoch,
                selected["source"],
                selected["mae"],
                selected["worst_region_mae"],
                selected["ordinary_positive_mae"],
                blend,
            )
            improved = (
                selected["objective"] < best["objective"] - 1e-6
                or (
                    abs(selected["objective"] - best["objective"]) <= 1e-6
                    and selected["mae"] < best["policy"]["mae"] - 1e-6
                )
            )
            if improved:
                best = {
                    "epoch": epoch,
                    "objective": float(selected["objective"]),
                    "state": _cpu_state_dict(model),
                    "policy": dict(selected),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.early_stop:
                break

        pd.DataFrame(history).to_csv(
            self.save_dir / "v9_training_history.csv", index=False
        )
        serializable_policy_rows = []
        for row in policy_rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            serializable_policy_rows.append(value)
        pd.DataFrame(serializable_policy_rows).to_csv(
            self.save_dir / "v9_valid_policy_search.csv", index=False
        )
        torch.save(best, self.save_dir / "region_balanced_ordinal_mixture_v9_best.pth")
        return best

    def _apply_policy(self, collected, committees, policy):
        if policy["source"] == "legacy_reference":
            return self._reference_prediction(collected, committees)
        mixture_student = collected["legacy_prediction"] + float(policy["alpha"]) * (
            collected["prediction"] - collected["legacy_prediction"]
        )
        return float(policy["beta"]) * committees[policy["committee"]] + (
            1.0 - float(policy["beta"])
        ) * mixture_student

    def _bootstrap_mae_gain(
        self,
        reference,
        candidate,
        labels,
        mask=None,
        samples=2000,
        seed=1201,
    ):
        reference = reference.view(-1).detach().cpu()
        candidate = candidate.view(-1).detach().cpu()
        labels = labels.view(-1).detach().cpu()
        if mask is not None:
            mask = mask.view(-1).detach().cpu().bool()
            reference = reference[mask]
            candidate = candidate[mask]
            labels = labels[mask]
        n = labels.numel()
        if n == 0:
            return {
                "count": 0,
                "mean": float("nan"),
                "ci95_low": float("nan"),
                "ci95_high": float("nan"),
                "probability_positive": float("nan"),
            }
        generator = torch.Generator().manual_seed(int(seed))
        values = torch.empty(int(samples))
        for index in range(int(samples)):
            draw = torch.randint(0, n, (n,), generator=generator)
            ref_mae = (reference[draw] - labels[draw]).abs().mean()
            cand_mae = (candidate[draw] - labels[draw]).abs().mean()
            values[index] = ref_mae - cand_mae
        ordered = values.sort().values
        low_index = max(0, int(math.floor(0.025 * samples)))
        high_index = min(samples - 1, int(math.ceil(0.975 * samples)) - 1)
        return {
            "count": int(n),
            "mean": float(values.mean().item()),
            "ci95_low": float(ordered[low_index].item()),
            "ci95_high": float(ordered[high_index].item()),
            "probability_positive": float((values > 0).float().mean().item()),
        }

    def evaluate_and_save(self, model, dataloaders, best, bootstrap_samples=2000):
        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        valid_committees = self._committee_predictions("valid")
        selected, rows, _, valid_reference_metrics = self._select_policy(
            valid, valid_committees
        )
        final_rows = []
        for row in rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            final_rows.append(value)
        pd.DataFrame(final_rows).to_csv(
            self.save_dir / "v9_final_valid_policy.csv", index=False
        )

        # Test is touched only after the epoch and policy are frozen by Valid.
        test = self.collect(model, dataloaders["test"], "test")
        test_committees = self._committee_predictions("test")
        test_reference = self._reference_prediction(test, test_committees)
        test_selected = self._apply_policy(test, test_committees, selected)

        named = {
            "anchor": test["base_prediction"],
            "global_simplex": test_committees["global_simplex"],
            "original_v71_hybrid": test_reference,
            "region_balanced_ordinal_mixture_v9": test_selected,
        }
        results = {}
        comparison_rows = []
        region_rows = []
        for name, prediction in named.items():
            metric = {
                key: float(value)
                for key, value in self.metrics_fn(
                    prediction.detach().cpu(), test["labels"].detach().cpu()
                ).items()
            }
            aggregate, per_region = region_diagnostics(
                prediction,
                test["labels"],
                self.strong_threshold,
                self.neutral_radius,
            )
            results[name] = {**metric, **aggregate}
            comparison_rows.append({"model": name, **metric, **aggregate})
            region_rows.extend({"model": name, **row} for row in per_region)
        pd.DataFrame(comparison_rows).to_csv(
            self.save_dir / "v9_test_comparison.csv", index=False
        )
        pd.DataFrame(region_rows).to_csv(
            self.save_dir / "v9_test_region_diagnostics.csv", index=False
        )

        probs = test["polarity_probs"]
        experts = test["expert_values"]
        pd.DataFrame({
            "sample_id": test["sample_ids"],
            "label": test["labels"].view(-1).tolist(),
            "original_v71_hybrid": test_reference.view(-1).tolist(),
            "v9_selected": test_selected.view(-1).tolist(),
            "legacy_student": test["legacy_prediction"].view(-1).tolist(),
            "raw_v9_prediction": test["prediction"].view(-1).tolist(),
            "mixture_value": test["mixture_value"].view(-1).tolist(),
            "gate_negative": probs[:, 0].tolist(),
            "gate_neutral": probs[:, 1].tolist(),
            "gate_positive": probs[:, 2].tolist(),
            "expert_negative": experts[:, 0].tolist(),
            "expert_neutral": experts[:, 1].tolist(),
            "expert_positive": experts[:, 2].tolist(),
            "ordinal7_value": test["ordinal7_value"].view(-1).tolist(),
            "ordinal5_value": test["ordinal5_value"].view(-1).tolist(),
        }).to_csv(self.save_dir / "v9_test_predictions.csv", index=False)

        test_region = sentiment_region_ids(
            test["labels"], self.strong_threshold, self.neutral_radius
        )
        summary = {
            "method": "region_balanced_ordinal_soft_mixture_v9",
            "selection_protocol": (
                "The V7.1 backbone and residual Student are frozen. New heads are "
                "trained on Train with region-balanced Huber and auxiliary polarity/"
                "ordinal objectives. Epoch and compact hybrid policy are selected only "
                "on Validation. Test is evaluated once after freezing the policy."
            ),
            "region_definition": {
                "strong_threshold": self.strong_threshold,
                "neutral_radius": self.neutral_radius,
                "names": list(REGION_NAMES),
            },
            "train_region_counts": self.region_counts.tolist(),
            "train_region_weights": self.region_weights.tolist(),
            "selected_epoch": int(best["epoch"]),
            "selected_policy": selected,
            "valid_reference_metrics": {
                key: value
                for key, value in valid_reference_metrics.items()
                if key not in ("region_rows",)
            },
            "valid_selected_metrics": self._policy_metrics(
                self._apply_policy(valid, valid_committees, selected),
                valid["labels"],
                valid["sample_ids"],
            ),
            "test_results": results,
            "bootstrap_vs_original_v71": {
                "overall_mae_gain_positive_is_better": self._bootstrap_mae_gain(
                    test_reference,
                    test_selected,
                    test["labels"],
                    samples=int(bootstrap_samples),
                    seed=int(getattr(self.args, "seed", 1111)) + 901,
                ),
                "ordinary_positive_mae_gain_positive_is_better": (
                    self._bootstrap_mae_gain(
                        test_reference,
                        test_selected,
                        test["labels"],
                        mask=test_region == 3,
                        samples=int(bootstrap_samples),
                        seed=int(getattr(self.args, "seed", 1111)) + 902,
                    )
                ),
            },
            "loss_weights": dict(self.loss_weights),
            "group_dro_weight": self.group_dro_weight,
            "group_dro_eta": self.group_dro_eta,
            "committee": {
                "global_weights": self.committee["global_weights"].tolist(),
                "region_weights": self.committee["region_weights"].tolist(),
                "global_cv_score": self.committee["global_cv_score"],
                "region_cv_score": self.committee["region_cv_score"],
            },
        }
        # Remove nested per-region rows from the JSON metric object; they are in CSV.
        summary["valid_selected_metrics"].pop("region_rows", None)
        (self.save_dir / "region_balanced_ordinal_mixture_v9_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "V9 final source=%s Test(MAE=%.4f worst=%.4f op=%.4f) reference_MAE=%.4f",
            selected["source"],
            results["region_balanced_ordinal_mixture_v9"]["MAE"],
            results["region_balanced_ordinal_mixture_v9"]["worst_region_mae"],
            results["region_balanced_ordinal_mixture_v9"]["ordinary_positive_mae"],
            results["original_v71_hybrid"]["MAE"],
        )
        return summary
