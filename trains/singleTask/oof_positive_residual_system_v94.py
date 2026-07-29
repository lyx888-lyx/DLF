"""Cross-fitted calibration and staged positive-residual specialist V9.4.

V9.4 is deliberately explicit about what is and is not out-of-fold.  The
historical DLF/Teacher predictions are frozen and may have been trained on the
full MOSI Train split.  The *calibration layer* is cross-fitted with grouped
folds, so every Train calibration prediction is produced by a calibrator that
did not see that sample's label.  Magnitude and activation are then trained in
separate stages against those cross-fitted residual targets.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from .positive_residual_specialist_system_v92 import (
    LOGGER,
    PositiveResidualSpecialistTrainerV92,
    _cpu_state_dict,
    _manual_huber,
    _weighted_mean,
    sentiment_region_ids,
)


def _group_key(sample_id) -> str:
    """Keep clips from the same source video in one cross-fitting fold."""
    value = str(sample_id)
    for separator in ("$_$", "___", "::", "#"):
        if separator in value:
            return value.split(separator, 1)[0]
    return value


def _grouped_stratified_folds(
    sample_ids: Sequence[object],
    labels: torch.Tensor,
    fold_count: int,
    strong_threshold: float,
    neutral_radius: float,
) -> tuple[torch.Tensor, list[str]]:
    """Deterministic group-aware approximate stratification over five regions."""
    fold_count = max(2, int(fold_count))
    labels = labels.view(-1).cpu()
    region = sentiment_region_ids(
        labels.view(-1, 1), strong_threshold, neutral_radius
    ).view(-1)
    groups: Dict[str, list[int]] = {}
    for index, sample_id in enumerate(sample_ids):
        groups.setdefault(_group_key(sample_id), []).append(index)

    by_region: Dict[int, list[tuple[str, list[int]]]] = {index: [] for index in range(5)}
    for key, indices in groups.items():
        group_regions = region[torch.tensor(indices, dtype=torch.long)]
        counts = torch.bincount(group_regions, minlength=5)
        dominant = int(torch.argmax(counts).item())
        by_region[dominant].append((key, indices))

    assignment = torch.full((len(sample_ids),), -1, dtype=torch.long)
    group_keys = [""] * len(sample_ids)
    fold_load = [0] * fold_count
    for region_id in range(5):
        ordered = sorted(
            by_region[region_id],
            key=lambda item: hashlib.sha1(item[0].encode("utf-8")).hexdigest(),
        )
        for key, indices in ordered:
            fold = min(range(fold_count), key=lambda value: (fold_load[value], value))
            assignment[torch.tensor(indices, dtype=torch.long)] = fold
            fold_load[fold] += len(indices)
            for index in indices:
                group_keys[index] = key
    if bool((assignment < 0).any()):
        raise RuntimeError("Cross-fitting fold assignment is incomplete.")
    return assignment, group_keys


def _robust_ridge_fit(
    features: torch.Tensor,
    target: torch.Tensor,
    l2: float,
    huber_delta: float = 0.35,
    iterations: int = 12,
) -> Mapping[str, torch.Tensor]:
    """IRLS Huber ridge with fold-local standardization and an unpenalized intercept."""
    x = features.detach().cpu().double()
    y = target.detach().cpu().double().view(-1, 1)
    mean = x.mean(dim=0, keepdim=True)
    scale = x.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    z = (x - mean) / scale
    design = torch.cat([torch.ones((z.size(0), 1), dtype=z.dtype), z], dim=1)
    penalty = torch.eye(design.size(1), dtype=design.dtype) * float(l2)
    penalty[0, 0] = 0.0
    weights = torch.ones((design.size(0), 1), dtype=design.dtype)
    coef = torch.zeros((design.size(1), 1), dtype=design.dtype)
    for _ in range(max(1, int(iterations))):
        weighted = design * weights.sqrt()
        weighted_target = y * weights.sqrt()
        lhs = weighted.T @ weighted + penalty
        rhs = weighted.T @ weighted_target
        try:
            coef = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            coef = torch.linalg.pinv(lhs) @ rhs
        residual = design @ coef - y
        absolute = residual.abs().clamp_min(1e-8)
        weights = torch.minimum(
            torch.ones_like(absolute),
            absolute.new_full(absolute.shape, float(huber_delta)) / absolute,
        )
    return {"mean": mean, "scale": scale, "coef": coef}


def _robust_ridge_predict(
    model: Mapping[str, torch.Tensor], features: torch.Tensor
) -> torch.Tensor:
    x = features.detach().cpu().double()
    z = (x - model["mean"]) / model["scale"]
    design = torch.cat([torch.ones((z.size(0), 1), dtype=z.dtype), z], dim=1)
    return (design @ model["coef"]).float()


def _average_precision(labels: torch.Tensor, scores: torch.Tensor) -> float:
    labels = labels.view(-1).float().cpu()
    scores = scores.view(-1).float().cpu()
    positives = int((labels > 0.5).sum().item())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True)
    ordered = labels[order]
    true_positive = torch.cumsum(ordered, dim=0)
    rank = torch.arange(1, len(ordered) + 1, dtype=torch.float32)
    precision = true_positive / rank
    return float((precision * ordered).sum().item() / positives)


def _binary_metrics(labels: torch.Tensor, probabilities: torch.Tensor) -> Dict[str, float]:
    labels = labels.view(-1).bool().cpu()
    probabilities = probabilities.view(-1).float().cpu()
    prediction = probabilities >= 0.5
    tp = int((prediction & labels).sum().item())
    tn = int((~prediction & ~labels).sum().item())
    fp = int((prediction & ~labels).sum().item())
    fn = int((~prediction & labels).sum().item())
    recall = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    return {
        "average_precision": _average_precision(labels.float(), probabilities),
        "recall": float(recall),
        "specificity": float(specificity),
        "precision": float(precision),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "false_positive_rate": float(fp / max(1, fp + tn)),
        "active_rate": float(labels.float().mean().item()),
        "predicted_active_rate": float(prediction.float().mean().item()),
    }


class OOFPositiveResidualTrainerV94(PositiveResidualSpecialistTrainerV92):
    """Cross-fit a reference calibrator, then train magnitude and gate separately."""

    def __init__(
        self,
        *args,
        crossfit_folds: int = 5,
        calibrator_l2_grid=(0.1, 1.0, 10.0, 100.0),
        calibrator_blend_grid=(0.0, 0.25, 0.50, 0.75, 1.0),
        max_calibration_shift: float = 0.75,
        magnitude_epochs: int = 20,
        magnitude_early_stop: int = 6,
        gate_epochs: int = 15,
        gate_early_stop: int = 5,
        gate_focal_gamma: float = 1.5,
        gate_pos_weight_cap: float = 8.0,
        magnitude_active_weight: float = 1.50,
        magnitude_zero_weight: float = 0.05,
        magnitude_over_weight: float = 0.05,
        gate_threshold_grid=(0.30, 0.40, 0.50, 0.60, 0.70),
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.crossfit_folds = max(2, int(crossfit_folds))
        self.calibrator_l2_grid = tuple(sorted({float(value) for value in calibrator_l2_grid}))
        self.calibrator_blend_grid = tuple(
            sorted({float(value) for value in calibrator_blend_grid})
        )
        if 0.0 not in self.calibrator_blend_grid:
            self.calibrator_blend_grid = (0.0, *self.calibrator_blend_grid)
        self.max_calibration_shift = max(0.0, float(max_calibration_shift))
        self.magnitude_epochs = max(1, int(magnitude_epochs))
        self.magnitude_early_stop = max(1, int(magnitude_early_stop))
        self.gate_epochs = max(1, int(gate_epochs))
        self.gate_early_stop = max(1, int(gate_early_stop))
        self.gate_focal_gamma = max(0.0, float(gate_focal_gamma))
        self.gate_pos_weight_cap = max(1.0, float(gate_pos_weight_cap))
        self.magnitude_active_weight = max(0.0, float(magnitude_active_weight))
        self.magnitude_zero_weight = max(0.0, float(magnitude_zero_weight))
        self.magnitude_over_weight = max(0.0, float(magnitude_over_weight))
        self.gate_threshold_grid = tuple(
            sorted({float(value) for value in gate_threshold_grid if 0.0 < float(value) < 1.0})
        )
        if not self.gate_threshold_grid:
            raise ValueError("gate_threshold_grid must contain values in (0, 1).")
        self.calibrator = None
        self.train_oof_reference = None
        self.train_residual_target = None
        self.crossfit_assignment = None
        self.crossfit_group_keys = None
        self.feature_names = None

    def _split_name(self, collected) -> str:
        observed = [str(value) for value in collected["sample_ids"]]
        for split_name, split in self.cache["splits"].items():
            expected = [str(value) for value in split["sample_ids"]]
            if observed == expected:
                return str(split_name)
        raise RuntimeError("Collected sample ids do not match any teacher-cache split.")

    def _design_matrix(self, collected, committees, split_name):
        split = self.cache["splits"][split_name]
        expected = [str(value) for value in split["sample_ids"]]
        observed = [str(value) for value in collected["sample_ids"]]
        if observed != expected:
            raise RuntimeError(f"Teacher-cache order mismatch for split={split_name}.")
        teacher = split["predictions"].detach().cpu().float()
        if teacher.ndim == 3 and teacher.size(-1) == 1:
            teacher = teacher.squeeze(-1)
        if teacher.ndim != 2:
            raise RuntimeError(f"Unexpected teacher prediction shape: {tuple(teacher.shape)}")
        original = self._reference_prediction(collected, committees).view(-1, 1).cpu()
        legacy = self._legacy_student(
            collected["base_prediction"], collected["legacy_prediction"]
        ).view(-1, 1).cpu()
        base = collected["base_prediction"].view(-1, 1).cpu()
        global_simplex = committees["global_simplex"].view(-1, 1).cpu()
        region_simplex = committees["region_simplex"].view(-1, 1).cpu()
        teacher_mean = teacher.mean(dim=1, keepdim=True)
        teacher_median = teacher.median(dim=1, keepdim=True).values
        teacher_std = teacher.std(dim=1, unbiased=False, keepdim=True)
        teacher_range = teacher.max(dim=1, keepdim=True).values - teacher.min(
            dim=1, keepdim=True
        ).values
        matrix = torch.cat(
            [
                original,
                legacy,
                base,
                global_simplex,
                region_simplex,
                teacher,
                teacher_mean,
                teacher_median,
                teacher_std,
                teacher_range,
            ],
            dim=1,
        )
        names = [
            "original_v71_hybrid",
            "legacy_student",
            "anchor",
            "global_simplex",
            "region_simplex",
            *[f"teacher_{index}" for index in range(teacher.size(1))],
            "teacher_mean",
            "teacher_median",
            "teacher_std",
            "teacher_range",
        ]
        if self.feature_names is None:
            self.feature_names = names
        elif self.feature_names != names:
            raise RuntimeError("Calibrator feature layout changed across splits.")
        return matrix, original

    def _crossfit_objective(self, prediction, labels, sample_ids, fold_ids):
        metrics = self._extended_metrics(prediction, labels, sample_ids)
        fold_maes = []
        for fold in range(self.crossfit_folds):
            mask = fold_ids == fold
            if bool(mask.any()):
                fold_maes.append(
                    float(
                        torch.abs(
                            prediction.view(-1)[mask] - labels.view(-1)[mask]
                        ).mean().item()
                    )
                )
        fold_std = (
            float(torch.tensor(fold_maes).std(unbiased=False).item())
            if fold_maes
            else 0.0
        )
        objective = (
            metrics["mae"]
            + self.robust_selection_weight * metrics["worst_region_mae"]
            + self.stability_selection_weight * fold_std
        )
        return float(objective), metrics, fold_maes, fold_std

    def _build_crossfit_calibrator(self, train_collected):
        committees = self.committees_by_split["train"]
        x, anchor = self._design_matrix(train_collected, committees, "train")
        labels = train_collected["labels"].view(-1, 1).cpu()
        fold_ids, group_keys = _grouped_stratified_folds(
            train_collected["sample_ids"],
            labels,
            self.crossfit_folds,
            self.strong_threshold,
            self.neutral_radius,
        )
        target_delta = (labels - anchor).clamp(
            -self.max_calibration_shift, self.max_calibration_shift
        )
        rows = []
        candidates = []
        for l2 in self.calibrator_l2_grid:
            oof_delta = torch.empty_like(target_delta)
            for fold in range(self.crossfit_folds):
                holdout = fold_ids == fold
                fit_mask = ~holdout
                if not bool(holdout.any()) or not bool(fit_mask.any()):
                    raise RuntimeError(f"Invalid cross-fitting fold {fold}.")
                model = _robust_ridge_fit(
                    x[fit_mask],
                    target_delta[fit_mask],
                    l2=l2,
                    huber_delta=max(self.huber_delta, 0.10),
                )
                oof_delta[holdout] = _robust_ridge_predict(model, x[holdout])
            oof_delta = oof_delta.clamp(
                -self.max_calibration_shift, self.max_calibration_shift
            )
            for blend in self.calibrator_blend_grid:
                prediction = anchor + float(blend) * oof_delta
                objective, metrics, fold_maes, fold_std = self._crossfit_objective(
                    prediction,
                    labels,
                    train_collected["sample_ids"],
                    fold_ids,
                )
                row = {
                    "l2": float(l2),
                    "blend": float(blend),
                    "objective": objective,
                    "mae": metrics["mae"],
                    "worst_region_mae": metrics["worst_region_mae"],
                    "ordinary_positive_mae": metrics["ordinary_positive_mae"],
                    "nonpositive_mae": metrics["nonpositive_mae"],
                    "fold_mae_std": fold_std,
                    "fold_maes": json.dumps(fold_maes),
                }
                rows.append(row)
                candidates.append((objective, metrics["mae"], float(blend), float(l2), oof_delta))
        selected = min(candidates, key=lambda item: (item[0], item[1], item[2], item[3]))
        _, _, blend, l2, selected_delta = selected
        oof_reference = anchor + blend * selected_delta
        full_model = _robust_ridge_fit(
            x,
            target_delta,
            l2=l2,
            huber_delta=max(self.huber_delta, 0.10),
        )
        self.calibrator = {
            "l2": float(l2),
            "blend": float(blend),
            "max_shift": float(self.max_calibration_shift),
            "model": full_model,
        }
        self.train_oof_reference = oof_reference.detach().cpu()
        self.crossfit_assignment = fold_ids
        self.crossfit_group_keys = group_keys
        target, _, _, _ = self._specialist_targets(labels, self.train_oof_reference)
        self.train_residual_target = target.detach().cpu()

        pd.DataFrame(rows).to_csv(
            self.save_dir / "v94_crossfit_calibrator_search.csv", index=False
        )
        pd.DataFrame(
            {
                "sample_id": train_collected["sample_ids"],
                "group_key": group_keys,
                "fold": fold_ids.tolist(),
                "label": labels.view(-1).tolist(),
                "original_reference": anchor.view(-1).tolist(),
                "oof_calibrated_reference": self.train_oof_reference.view(-1).tolist(),
                "positive_residual_target": self.train_residual_target.view(-1).tolist(),
            }
        ).to_csv(self.save_dir / "v94_oof_assignments.csv", index=False)
        LOGGER.info(
            "V9.4 crossfit calibrator l2=%.4g blend=%.2f "
            "Train-OOF(MAE=%.4f active=%.3f)",
            l2,
            blend,
            float(torch.abs(self.train_oof_reference - labels).mean().item()),
            float((self.train_residual_target > self.residual_margin).float().mean().item()),
        )

    def _calibrated_reference(self, collected, committees):
        split_name = self._split_name(collected)
        if split_name == "train" and self.train_oof_reference is not None:
            return self.train_oof_reference.clone()
        if self.calibrator is None:
            raise RuntimeError("V9.4 calibrator has not been fitted.")
        x, anchor = self._design_matrix(collected, committees, split_name)
        delta = _robust_ridge_predict(self.calibrator["model"], x).clamp(
            -self.calibrator["max_shift"], self.calibrator["max_shift"]
        )
        return anchor + self.calibrator["blend"] * delta

    def _sample_weights(self, labels):
        region = sentiment_region_ids(
            labels, self.strong_threshold, self.neutral_radius
        ).view(-1)
        weights = self.region_weights.to(labels.device)[region].view(-1, 1)
        return weights * torch.where(
            (region == 3).view(-1, 1),
            weights.new_full(weights.shape, self.ordinary_positive_boost),
            weights.new_ones(weights.shape),
        )

    def _magnitude_epoch(self, model, dataloader, optimizer):
        model.set_magnitude_train_mode()
        totals: Dict[str, float] = {}
        for batch in tqdm(dataloader, leave=False):
            indices = self._batch_cache_indices(batch.get("id"), "train")
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            target = self.train_residual_target[indices].to(self.args.device)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            magnitude = output["magnitude"]
            weights = self._sample_weights(labels)
            all_loss = _weighted_mean(
                _manual_huber(magnitude - target, self.huber_delta), weights
            )
            active = target > self.residual_margin
            active_loss = (
                _weighted_mean(
                    _manual_huber(
                        magnitude[active] - target[active], self.huber_delta
                    ),
                    weights[active],
                )
                if bool(active.any())
                else magnitude.new_zeros(())
            )
            zero = target <= 1e-8
            zero_loss = (
                _weighted_mean(magnitude[zero], weights[zero])
                if bool(zero.any())
                else magnitude.new_zeros(())
            )
            over_loss = _weighted_mean(F.relu(magnitude - target), weights)
            total = (
                all_loss
                + self.magnitude_active_weight * active_loss
                + self.magnitude_zero_weight * zero_loss
                + self.magnitude_over_weight * over_loss
            )
            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), 2.0)
            optimizer.step()
            values = {
                "total": total,
                "all_huber": all_loss,
                "active_huber": active_loss,
                "zero_mean": zero_loss,
                "over_mean": over_loss,
                "magnitude_mean": magnitude.mean(),
                "magnitude_max": magnitude.max(),
            }
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    def _magnitude_validation(self, collected, committees):
        reference = self._calibrated_reference(collected, committees).view(-1, 1)
        labels = collected["labels"].view(-1, 1)
        target, _, _, _ = self._specialist_targets(labels, reference)
        magnitude = collected["magnitude"].view(-1, 1)
        active = target > self.residual_margin
        zero = target <= 1e-8
        overall = float(_manual_huber(magnitude - target, self.huber_delta).mean().item())
        active_huber = (
            float(
                _manual_huber(
                    magnitude[active] - target[active], self.huber_delta
                ).mean().item()
            )
            if bool(active.any())
            else 0.0
        )
        false_mean = float(magnitude[zero].mean().item()) if bool(zero.any()) else 0.0
        objective = active_huber + 0.25 * overall + 0.05 * false_mean
        return {
            "objective": float(objective),
            "overall_huber": overall,
            "active_huber": active_huber,
            "false_magnitude_mean": false_mean,
            "active_rate": float(active.float().mean().item()),
            "magnitude_mean": float(magnitude.mean().item()),
            "magnitude_max": float(magnitude.max().item()),
        }

    def _batch_cache_indices(self, batch_ids, split_name):
        from .positive_residual_specialist_system_v92 import _batch_cache_indices

        return _batch_cache_indices(batch_ids, self.index[split_name])

    def _gate_pos_weight(self):
        active = self.train_residual_target.view(-1) > self.residual_margin
        positive = int(active.sum().item())
        negative = int((~active).sum().item())
        if positive == 0:
            raise RuntimeError("V9.4 Train OOF targets contain no active residuals.")
        return min(self.gate_pos_weight_cap, negative / max(1, positive))

    def _gate_epoch(self, model, dataloader, optimizer, pos_weight):
        model.set_gate_train_mode()
        totals: Dict[str, float] = {}
        positive_weight = torch.tensor(
            [float(pos_weight)], device=self.args.device, dtype=torch.float32
        )
        for batch in tqdm(dataloader, leave=False):
            indices = self._batch_cache_indices(batch.get("id"), "train")
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            target = (
                self.train_residual_target[indices].to(self.args.device)
                > self.residual_margin
            ).float()
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            logits = output["gate_logit"]
            bce = F.binary_cross_entropy_with_logits(
                logits, target, reduction="none", pos_weight=positive_weight
            )
            probability = torch.sigmoid(logits)
            pt = torch.where(target > 0.5, probability, 1.0 - probability)
            focal = (1.0 - pt).pow(self.gate_focal_gamma)
            weights = self._sample_weights(labels)
            loss = _weighted_mean(bce * focal, weights)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), 2.0)
            optimizer.step()
            totals["loss"] = totals.get("loss", 0.0) + float(loss.detach().item())
            totals["probability_mean"] = totals.get("probability_mean", 0.0) + float(
                probability.detach().mean().item()
            )
        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    def _gate_validation(self, collected, committees, pos_weight):
        reference = self._calibrated_reference(collected, committees).view(-1, 1)
        labels = collected["labels"].view(-1, 1)
        target, _, _, _ = self._specialist_targets(labels, reference)
        active = (target > self.residual_margin).float()
        probabilities = collected["gate_probability"].view(-1, 1)
        logits = torch.logit(probabilities.clamp(1e-6, 1.0 - 1e-6))
        positive_weight = torch.tensor([float(pos_weight)], dtype=logits.dtype)
        bce = float(
            F.binary_cross_entropy_with_logits(
                logits, active, pos_weight=positive_weight
            ).item()
        )
        metrics = _binary_metrics(active, probabilities)
        objective = (
            bce
            + 0.50 * (1.0 - metrics["average_precision"])
            + 0.10 * metrics["false_positive_rate"]
        )
        return {"objective": float(objective), "bce": bce, **metrics}

    def _new_candidate_row(
        self,
        source,
        prediction,
        labels,
        sample_ids,
        reference_metrics,
        gamma=0.0,
        gate_mode="none",
        threshold=0.0,
    ):
        row = self._candidate_row(
            source,
            "oof_calibrated",
            0.0,
            0.0,
            gamma,
            prediction,
            labels,
            sample_ids,
            reference_metrics,
        )
        row["gate_mode"] = str(gate_mode)
        row["threshold"] = float(threshold)
        return row

    def _select_policy(self, collected, committees):
        safe = dict(collected)
        safe["positive_correction"] = torch.zeros_like(collected["positive_correction"])
        _, inherited_rows, reference, reference_metrics = (
            PositiveResidualSpecialistTrainerV92._select_policy(
                self, safe, committees
            )
        )
        rows = [
            dict(row, gate_mode="none", threshold=0.0)
            for row in inherited_rows
            if row["source"] != "positive_residual"
        ]
        feasible = [row for row in rows if row["feasible"]]

        calibrated = self._calibrated_reference(collected, committees)
        row = self._new_candidate_row(
            "oof_calibrated_reference",
            calibrated,
            collected["labels"],
            collected["sample_ids"],
            reference_metrics,
        )
        rows.append(row)
        if row["feasible"]:
            feasible.append(row)

        magnitude = collected["magnitude"].view(-1, 1)
        gate_probability = collected["gate_probability"].view(-1, 1)
        correction_families = [
            ("oof_magnitude_only", "magnitude_only", 0.0, magnitude),
            (
                "oof_gated_residual",
                "soft",
                0.0,
                gate_probability * magnitude,
            ),
        ]
        for threshold in self.gate_threshold_grid:
            correction_families.append(
                (
                    "oof_gated_residual",
                    "hard",
                    float(threshold),
                    (gate_probability >= threshold).float() * magnitude,
                )
            )
        for source, mode, threshold, correction in correction_families:
            for gamma in self.gamma_grid:
                prediction = calibrated + float(gamma) * correction
                row = self._new_candidate_row(
                    source,
                    prediction,
                    collected["labels"],
                    collected["sample_ids"],
                    reference_metrics,
                    gamma=float(gamma),
                    gate_mode=mode,
                    threshold=threshold,
                )
                rows.append(row)
                if row["feasible"]:
                    feasible.append(row)
        if not feasible:
            raise RuntimeError("V9.4 policy search produced no feasible candidate.")
        complexity = {
            "legacy_reference": 0,
            "legacy_beta_search": 1,
            "zero_shrinkage": 2,
            "oof_calibrated_reference": 3,
            "oof_magnitude_only": 4,
            "oof_gated_residual": 5,
        }
        selected = min(
            feasible,
            key=lambda item: (
                item["objective"],
                item["mae"],
                item["worst_region_mae"],
                item["nonpositive_mae"],
                item["fold_mae_std"],
                complexity.get(item["source"], 99),
            ),
        )
        return selected, rows, reference, reference_metrics

    def _apply_policy(self, collected, committees, policy):
        source = str(policy["source"])
        if source in ("legacy_reference", "legacy_beta_search", "zero_shrinkage"):
            return PositiveResidualSpecialistTrainerV92._apply_policy(
                self, collected, committees, policy
            )
        calibrated = self._calibrated_reference(collected, committees)
        if source == "oof_calibrated_reference":
            return calibrated
        magnitude = collected["magnitude"].view(-1, 1)
        if source == "oof_magnitude_only":
            correction = magnitude
        elif source == "oof_gated_residual":
            probability = collected["gate_probability"].view(-1, 1)
            if policy.get("gate_mode") == "hard":
                correction = (
                    probability >= float(policy.get("threshold", 0.5))
                ).float() * magnitude
            else:
                correction = probability * magnitude
        else:
            raise ValueError(f"Unknown V9.4 policy source: {source}")
        return calibrated + float(policy["gamma"]) * correction

    def train(self, model, dataloaders):
        model.freeze_for_magnitude()
        self.max_correction = float(model.max_correction)
        train_collected = self.collect(model, dataloaders["train"], "train")
        self._build_crossfit_calibrator(train_collected)
        valid_committees = self.committees_by_split["valid"]

        magnitude_optimizer = optim.AdamW(
            model.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        magnitude_history = []
        best_magnitude = None
        last_improvement = 0
        for epoch in range(1, self.magnitude_epochs + 1):
            train_row = self._magnitude_epoch(
                model, dataloaders["train"], magnitude_optimizer
            )
            valid = self.collect(model, dataloaders["valid"], "valid")
            metric = self._magnitude_validation(valid, valid_committees)
            magnitude_history.append(
                {"epoch": epoch, **train_row, **{f"valid_{k}": v for k, v in metric.items()}}
            )
            LOGGER.info(
                "V9.4 magnitude epoch=%d Valid(overall=%.4f active=%.4f "
                "false=%.4f mean=%.4f max=%.4f)",
                epoch,
                metric["overall_huber"],
                metric["active_huber"],
                metric["false_magnitude_mean"],
                metric["magnitude_mean"],
                metric["magnitude_max"],
            )
            if (
                best_magnitude is None
                or metric["objective"] < best_magnitude["objective"] - 1e-6
            ):
                best_magnitude = {
                    "epoch": epoch,
                    "objective": float(metric["objective"]),
                    "metrics": dict(metric),
                    "state": _cpu_state_dict(model),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.magnitude_early_stop:
                break
        if best_magnitude is None:
            raise RuntimeError("V9.4 completed no magnitude epochs.")
        model.load_state_dict(best_magnitude["state"])
        model.to(self.args.device)

        model.freeze_for_gate()
        pos_weight = self._gate_pos_weight()
        gate_optimizer = optim.AdamW(
            model.trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        gate_history = []
        valid = self.collect(model, dataloaders["valid"], "valid")
        initial_gate = self._gate_validation(valid, valid_committees, pos_weight)
        best_gate = {
            "epoch": 0,
            "objective": float(initial_gate["objective"]),
            "metrics": dict(initial_gate),
            "state": _cpu_state_dict(model),
        }
        last_improvement = 0
        for epoch in range(1, self.gate_epochs + 1):
            train_row = self._gate_epoch(
                model, dataloaders["train"], gate_optimizer, pos_weight
            )
            valid = self.collect(model, dataloaders["valid"], "valid")
            metric = self._gate_validation(valid, valid_committees, pos_weight)
            gate_history.append(
                {"epoch": epoch, **train_row, **{f"valid_{k}": v for k, v in metric.items()}}
            )
            LOGGER.info(
                "V9.4 gate epoch=%d Valid(BCE=%.4f AP=%.3f recall=%.3f "
                "specificity=%.3f predicted_active=%.3f)",
                epoch,
                metric["bce"],
                metric["average_precision"],
                metric["recall"],
                metric["specificity"],
                metric["predicted_active_rate"],
            )
            if metric["objective"] < best_gate["objective"] - 1e-6:
                best_gate = {
                    "epoch": epoch,
                    "objective": float(metric["objective"]),
                    "metrics": dict(metric),
                    "state": _cpu_state_dict(model),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.gate_early_stop:
                break
        model.load_state_dict(best_gate["state"])
        model.to(self.args.device)

        valid = self.collect(model, dataloaders["valid"], "valid")
        selected, policy_rows, _, _ = self._select_policy(valid, valid_committees)
        pd.DataFrame(magnitude_history).to_csv(
            self.save_dir / "v94_magnitude_history.csv", index=False
        )
        pd.DataFrame(gate_history).to_csv(
            self.save_dir / "v94_gate_history.csv", index=False
        )
        serializable = []
        for row in policy_rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            serializable.append(value)
        pd.DataFrame(serializable).to_csv(
            self.save_dir / "v94_valid_policy_search.csv", index=False
        )
        best = {
            "epoch": int(best_magnitude["epoch"]),
            "gate_epoch": int(best_gate["epoch"]),
            "state": _cpu_state_dict(model),
            "policy": dict(selected),
            "objective": float(selected["objective"]),
            "magnitude_metrics": dict(best_magnitude["metrics"]),
            "gate_metrics": dict(best_gate["metrics"]),
            "gate_pos_weight": float(pos_weight),
        }
        torch.save(best, self.save_dir / "oof_positive_residual_v94_best.pth")
        LOGGER.info(
            "V9.4 frozen magnitude=%d gate=%d; selected source=%s gate_mode=%s "
            "gamma=%.2f Valid(MAE=%.4f op=%.4f)",
            best["epoch"],
            best["gate_epoch"],
            selected["source"],
            selected.get("gate_mode", "none"),
            selected["gamma"],
            selected["mae"],
            selected["ordinary_positive_mae"],
        )
        return best

    def _serialized_calibrator(self):
        if self.calibrator is None:
            return None
        model = self.calibrator["model"]
        return {
            "l2": self.calibrator["l2"],
            "blend": self.calibrator["blend"],
            "max_shift": self.calibrator["max_shift"],
            "feature_names": list(self.feature_names or []),
            "mean": model["mean"].view(-1).tolist(),
            "scale": model["scale"].view(-1).tolist(),
            "coef": model["coef"].view(-1).tolist(),
        }

    def evaluate_and_save(self, model, dataloaders, best, bootstrap_samples=2000):
        summary = super().evaluate_and_save(
            model, dataloaders, best, bootstrap_samples=bootstrap_samples
        )
        selected = summary["selected_policy"]
        residual_sources = {"oof_magnitude_only", "oof_gated_residual"}
        learned_sources = residual_sources | {"oof_calibrated_reference"}
        summary.update(
            {
                "method": "crossfit_calibrated_staged_positive_residual_v9_4",
                "oof_scope": (
                    "OOF applies to the grouped calibration layer. Historical frozen "
                    "DLF/Teacher predictions may still be in-sample because their original "
                    "checkpoints were trained on the full Train split."
                ),
                "crossfit_folds": self.crossfit_folds,
                "calibrator": self._serialized_calibrator(),
                "magnitude_metrics": best["magnitude_metrics"],
                "gate_metrics": best["gate_metrics"],
                "gate_epoch": best["gate_epoch"],
                "gate_pos_weight": best["gate_pos_weight"],
                "calibration_contributed": bool(
                    selected["source"] in learned_sources
                    and self.calibrator is not None
                    and self.calibrator["blend"] > 0.0
                ),
                "residual_training_contributed": bool(
                    selected["source"] in residual_sources
                    and float(selected["gamma"]) > 0.0
                ),
                "training_contributed": bool(selected["source"] in learned_sources),
            }
        )
        target = self.save_dir / "oof_positive_residual_v94_summary.json"
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        selected_name = "positive_residual_specialist_v92"
        result = summary["test_results"][selected_name]
        LOGGER.info(
            "V9.4 final source=%s magnitude=%d gate=%d residual_contributed=%s "
            "Test(MAE=%.4f worst=%.4f op=%.4f nonpos=%.4f) reference_MAE=%.4f",
            selected["source"],
            best["epoch"],
            best["gate_epoch"],
            summary["residual_training_contributed"],
            result["MAE"],
            result["worst_region_mae"],
            result["ordinary_positive_mae"],
            result["nonpositive_mae"],
            summary["test_results"]["original_v71_hybrid"]["MAE"],
        )
        return summary
