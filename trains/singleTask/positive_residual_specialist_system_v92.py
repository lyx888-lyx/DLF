"""Training and attribution for constrained positive-residual specialist V9.2."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids
from .region_balanced_ordinal_mixture_system_v9 import (
    REGION_NAMES,
    RegionBalancedOrdinalMixtureTrainerV9,
    _cpu_state_dict,
    _manual_huber,
    _weighted_mean,
    region_diagnostics,
    sentiment_region_ids,
)


LOGGER = logging.getLogger("MMSA")


def _batch_cache_indices(batch_ids, mapping):
    values = normalize_batch_ids(batch_ids)
    try:
        return torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    except KeyError as error:
        raise KeyError(f"Sample id missing from teacher cache: {error}") from error


class PositiveResidualSpecialistTrainerV92(RegionBalancedOrdinalMixtureTrainerV9):
    """Train a local positive correction, then perform one Valid-only attribution."""

    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths: Sequence[Path],
        init_index: int,
        source_summary: Mapping[str, object],
        max_epochs: int = 30,
        early_stop: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-3,
        strong_threshold: float = 1.0,
        neutral_radius: float = 1e-6,
        region_weight_mode: str = "inverse_sqrt",
        region_weight_min: float = 0.50,
        region_weight_max: float = 2.00,
        effective_beta: float = 0.999,
        huber_delta: float = 0.30,
        ordinary_positive_boost: float = 1.50,
        residual_margin: float = 0.05,
        group_dro_weight: float = 0.0,
        group_dro_eta: float = 0.05,
        valid_mae_tolerance: float = 0.001,
        fold_mae_tolerance: float = 0.010,
        worst_region_tolerance: float = 0.10,
        nonpositive_mae_tolerance: float = 0.010,
        robust_selection_weight: float = 0.05,
        stability_selection_weight: float = 0.05,
        nonpositive_selection_weight: float = 0.02,
        folds: int = 3,
        beta_grid=(0.40, 0.50, 0.60),
        gamma_grid=(0.25, 0.50, 0.75, 1.00),
        zero_shrinkage_grid=(0.02, 0.05, 0.10),
        loss_weights=None,
    ):
        super().__init__(
            args=args,
            metrics_fn=metrics_fn,
            save_dir=save_dir,
            teacher_cache=teacher_cache,
            teacher_paths=teacher_paths,
            init_index=init_index,
            source_summary=source_summary,
            max_epochs=max_epochs,
            early_stop=early_stop,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            strong_threshold=strong_threshold,
            neutral_radius=neutral_radius,
            region_weight_mode=region_weight_mode,
            region_weight_min=region_weight_min,
            region_weight_max=region_weight_max,
            effective_beta=effective_beta,
            huber_delta=huber_delta,
            group_dro_weight=group_dro_weight,
            group_dro_eta=group_dro_eta,
            valid_mae_tolerance=valid_mae_tolerance,
            fold_mae_tolerance=fold_mae_tolerance,
            worst_region_tolerance=worst_region_tolerance,
            robust_selection_weight=robust_selection_weight,
            ordinary_positive_selection_weight=0.0,
            stability_selection_weight=stability_selection_weight,
            folds=folds,
            beta_grid=beta_grid,
            alpha_grid=(1.0,),
        )
        self.ordinary_positive_boost = max(float(ordinary_positive_boost), 1.0)
        self.residual_margin = max(float(residual_margin), 0.0)
        self.nonpositive_mae_tolerance = max(
            float(nonpositive_mae_tolerance), 0.0
        )
        self.nonpositive_selection_weight = max(
            float(nonpositive_selection_weight), 0.0
        )
        self.gamma_grid = tuple(
            sorted({float(value) for value in gamma_grid if float(value) > 0.0})
        )
        self.zero_shrinkage_grid = tuple(
            sorted(
                {float(value) for value in zero_shrinkage_grid if float(value) > 0.0}
            )
        )
        if not self.gamma_grid:
            raise ValueError("gamma_grid must contain at least one positive value.")
        if not self.zero_shrinkage_grid:
            raise ValueError(
                "zero_shrinkage_grid must contain at least one positive value."
            )
        if any(value > 1.0 for value in (*self.gamma_grid, *self.zero_shrinkage_grid)):
            raise ValueError("All gamma/shrinkage values must be in (0, 1].")

        self.loss_weights = loss_weights or {
            "correction": 1.00,
            "gate": 0.25,
            "magnitude": 0.20,
            "no_harm": 0.40,
            "overcorrection": 0.20,
            "sparsity": 0.02,
        }
        self.committees_by_split = {
            split: self._committee_predictions(split)
            for split in ("train", "valid", "test")
        }
        pd.DataFrame(
            {
                "region": REGION_NAMES,
                "count": self.region_counts.tolist(),
                "weight": self.region_weights.tolist(),
            }
        ).to_csv(self.save_dir / "v92_train_region_weights.csv", index=False)

    def _fit_committee(self):
        fitted = super()._fit_committee()
        pd.DataFrame(fitted["cv_rows"]).to_csv(
            Path(self.save_dir) / "v92_committee_cv.csv", index=False
        )
        return fitted

    def _optimizer(self, model):
        parameters = model.trainable_parameters()
        if not parameters:
            raise RuntimeError("V9.2 has no trainable specialist parameters.")
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _legacy_student(self, base_prediction, legacy_prediction):
        alpha = float(self.source_summary["student_policy"]["alpha"])
        return base_prediction + alpha * (legacy_prediction - base_prediction)

    def _batch_reference(self, output, indices, split_name):
        source_policy = self.source_summary["hybrid_policy"]
        committee_name = str(source_policy["committee"])
        beta = float(source_policy["beta"])
        committee = self.committees_by_split[split_name][committee_name][
            indices
        ].to(output["base_prediction"].device)
        legacy = self._legacy_student(
            output["base_prediction"], output["legacy_prediction"]
        )
        return beta * committee + (1.0 - beta) * legacy

    def _specialist_targets(self, labels, reference):
        positive_label = labels > max(self.neutral_radius, 1e-6)
        raw_residual = labels - reference.detach()
        target = raw_residual.clamp_min(0.0)
        target = torch.minimum(
            target, target.new_full(target.shape, float(self.max_correction))
        )
        target = target * positive_label.float()
        gate_target = (target > self.residual_margin).float()
        return target, gate_target, raw_residual, positive_label

    def _loss(self, output, labels, reference):
        labels = labels.view(-1, 1)
        region = sentiment_region_ids(
            labels, self.strong_threshold, self.neutral_radius
        )
        sample_weights = self.region_weights.to(labels.device)[region].view(-1, 1)
        sample_weights = sample_weights * torch.where(
            (region == 3).view(-1, 1),
            sample_weights.new_full(sample_weights.shape, self.ordinary_positive_boost),
            sample_weights.new_ones(sample_weights.shape),
        )

        target, gate_target, raw_residual, positive_label = self._specialist_targets(
            labels, reference
        )
        correction = output["positive_correction"]
        correction_loss = _weighted_mean(
            _manual_huber(correction - target, self.huber_delta),
            sample_weights,
        )

        gate_loss = F.binary_cross_entropy_with_logits(
            output["gate_logit"], gate_target, reduction="none"
        )
        gate_loss = _weighted_mean(gate_loss, sample_weights)

        active = gate_target > 0.5
        if bool(active.any()):
            magnitude_loss = _weighted_mean(
                _manual_huber(
                    output["magnitude"][active] - target[active],
                    self.huber_delta,
                ),
                sample_weights[active],
            )
        else:
            magnitude_loss = correction.new_zeros(())

        target_zero = target <= 1e-8
        if bool(target_zero.any()):
            no_harm = _weighted_mean(
                correction[target_zero],
                sample_weights[target_zero],
            )
        else:
            no_harm = correction.new_zeros(())

        allowed = raw_residual.clamp_min(0.0)
        overcorrection = F.relu(correction - allowed)
        overcorrection = _weighted_mean(overcorrection, sample_weights)

        # Sparse activation is desirable because this path is a specialist, not
        # a replacement generalist.
        sparsity = output["gate_probability"].mean()

        losses = {
            "correction": correction_loss,
            "gate": gate_loss,
            "magnitude": magnitude_loss,
            "no_harm": no_harm,
            "overcorrection": overcorrection,
            "sparsity": sparsity,
        }
        total = correction.new_zeros(())
        for name, value in losses.items():
            total = total + float(self.loss_weights.get(name, 0.0)) * value
        losses["total"] = total
        diagnostics = {
            "target_mean": target.mean().detach(),
            "target_active_rate": gate_target.mean().detach(),
            "correction_mean": correction.mean().detach(),
            "gate_mean": output["gate_probability"].mean().detach(),
            "positive_label_rate": positive_label.float().mean().detach(),
        }
        return losses, diagnostics

    def _train_epoch(self, model, dataloader, optimizer):
        model.set_train_mode()
        totals: Dict[str, float] = {}
        optimizer.zero_grad()
        for batch in tqdm(dataloader, leave=False):
            indices = _batch_cache_indices(batch.get("id"), self.index["train"])
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            reference = self._batch_reference(output, indices, "train")
            losses, diagnostics = self._loss(output, labels, reference)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()

            for name, value in {**losses, **diagnostics}.items():
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().item()
                )
        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    @torch.no_grad()
    def collect(self, model, dataloader, split_name):
        model.eval()
        mapping = self.index[split_name]
        split = self.cache["splits"][split_name]
        count = len(split["sample_ids"])
        keys = (
            "base_prediction",
            "legacy_prediction",
            "gate_probability",
            "magnitude",
            "positive_correction",
            "specialist_prediction",
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
                f"Incomplete V9.2 collection for {split_name}: "
                f"missing={int((~seen).sum().item())}"
            )
        return {
            "sample_ids": list(split["sample_ids"]),
            "labels": split["labels"].clone(),
            **buffers,
        }

    def _reference_prediction(self, collected, committees):
        source_policy = self.source_summary["hybrid_policy"]
        beta = float(source_policy["beta"])
        committee = committees[str(source_policy["committee"])]
        legacy = self._legacy_student(
            collected["base_prediction"], collected["legacy_prediction"]
        )
        return beta * committee + (1.0 - beta) * legacy

    def _extended_metrics(self, prediction, labels, sample_ids):
        metrics = self._policy_metrics(prediction, labels, sample_ids)
        values = labels.view(-1)
        errors = (prediction.view(-1) - values).abs()
        nonpositive = values <= max(self.neutral_radius, 1e-6)
        positive = values > max(self.neutral_radius, 1e-6)
        metrics["nonpositive_mae"] = float(
            errors[nonpositive].mean().item()
        ) if bool(nonpositive.any()) else float("nan")
        metrics["positive_mae"] = float(
            errors[positive].mean().item()
        ) if bool(positive.any()) else float("nan")
        return metrics

    def _selection_objective(self, metrics):
        return (
            metrics["mae"]
            + self.robust_selection_weight * metrics["worst_region_mae"]
            + self.stability_selection_weight * metrics["fold_mae_std"]
            + self.nonpositive_selection_weight * metrics["nonpositive_mae"]
        )

    def _candidate_row(
        self,
        source,
        committee,
        beta,
        shrinkage,
        gamma,
        prediction,
        labels,
        sample_ids,
        reference_metrics,
    ):
        metrics = self._extended_metrics(prediction, labels, sample_ids)
        fold_limits = [
            value + self.fold_mae_tolerance
            for value in reference_metrics["fold_maes"]
        ]
        feasible = (
            metrics["mae"]
            <= reference_metrics["mae"] + self.valid_mae_tolerance + 1e-12
            and metrics["worst_region_mae"]
            <= reference_metrics["worst_region_mae"]
            + self.worst_region_tolerance
            + 1e-12
            and metrics["nonpositive_mae"]
            <= reference_metrics["nonpositive_mae"]
            + self.nonpositive_mae_tolerance
            + 1e-12
            and all(
                metrics["fold_maes"][index] <= fold_limits[index] + 1e-12
                for index in range(self.folds)
            )
        )
        return {
            "source": str(source),
            "committee": str(committee),
            "beta": float(beta),
            "shrinkage": float(shrinkage),
            "gamma": float(gamma),
            "objective": float(self._selection_objective(metrics)),
            "feasible": bool(feasible),
            **{
                key: value
                for key, value in metrics.items()
                if key not in ("region_rows", "fold_maes")
            },
            "fold_maes": metrics["fold_maes"],
        }

    def _select_policy(self, collected, committees):
        labels = collected["labels"]
        sample_ids = collected["sample_ids"]
        reference = self._reference_prediction(collected, committees)
        reference_metrics = self._extended_metrics(
            reference, labels, sample_ids
        )
        source_policy = self.source_summary["hybrid_policy"]
        reference_row = {
            "source": "legacy_reference",
            "committee": str(source_policy["committee"]),
            "beta": float(source_policy["beta"]),
            "shrinkage": 0.0,
            "gamma": 0.0,
            "objective": float(self._selection_objective(reference_metrics)),
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

        legacy = self._legacy_student(
            collected["base_prediction"], collected["legacy_prediction"]
        )
        correction = collected["positive_correction"]
        source_committee = str(source_policy["committee"])
        committee_names = tuple(
            dict.fromkeys(
                (source_committee, "global_simplex", "region_simplex")
            )
        )
        source_beta = float(source_policy["beta"])
        beta_grid = tuple(dict.fromkeys((*self.beta_grid, source_beta)))

        bases = []
        for committee_name in committee_names:
            if committee_name not in committees:
                continue
            committee_prediction = committees[committee_name]
            for beta in beta_grid:
                beta = float(beta)
                beta_only = beta * committee_prediction + (1.0 - beta) * legacy
                row = self._candidate_row(
                    "legacy_beta_search",
                    committee_name,
                    beta,
                    0.0,
                    0.0,
                    beta_only,
                    labels,
                    sample_ids,
                    reference_metrics,
                )
                rows.append(row)
                if row["feasible"]:
                    feasible.append(row)
                bases.append(
                    {
                        "committee": committee_name,
                        "beta": beta,
                        "shrinkage": 0.0,
                        "prediction": beta_only,
                    }
                )

                for shrinkage in self.zero_shrinkage_grid:
                    shrunk_legacy = (1.0 - shrinkage) * legacy
                    zero_prediction = (
                        beta * committee_prediction
                        + (1.0 - beta) * shrunk_legacy
                    )
                    row = self._candidate_row(
                        "zero_shrinkage",
                        committee_name,
                        beta,
                        shrinkage,
                        0.0,
                        zero_prediction,
                        labels,
                        sample_ids,
                        reference_metrics,
                    )
                    rows.append(row)
                    if row["feasible"]:
                        feasible.append(row)
                    bases.append(
                        {
                            "committee": committee_name,
                            "beta": beta,
                            "shrinkage": float(shrinkage),
                            "prediction": zero_prediction,
                        }
                    )

        for base in bases:
            for gamma in self.gamma_grid:
                prediction = base["prediction"] + float(gamma) * correction
                row = self._candidate_row(
                    "positive_residual",
                    base["committee"],
                    base["beta"],
                    base["shrinkage"],
                    gamma,
                    prediction,
                    labels,
                    sample_ids,
                    reference_metrics,
                )
                rows.append(row)
                if row["feasible"]:
                    feasible.append(row)

        selected = min(
            feasible,
            key=lambda row: (
                row["objective"],
                row["mae"],
                row["worst_region_mae"],
                row["nonpositive_mae"],
                row["fold_mae_std"],
                row["source"] != "legacy_reference",
            ),
        )
        return selected, rows, reference, reference_metrics

    def _apply_policy(self, collected, committees, policy):
        if policy["source"] == "legacy_reference":
            return self._reference_prediction(collected, committees)
        legacy = self._legacy_student(
            collected["base_prediction"], collected["legacy_prediction"]
        )
        committee = committees[str(policy["committee"])]
        shrunk_legacy = (1.0 - float(policy["shrinkage"])) * legacy
        prediction = (
            float(policy["beta"]) * committee
            + (1.0 - float(policy["beta"])) * shrunk_legacy
        )
        if policy["source"] == "positive_residual":
            prediction = prediction + float(policy["gamma"]) * collected[
                "positive_correction"
            ]
        return prediction

    def _expert_validation(self, collected, committees):
        reference = self._reference_prediction(collected, committees)
        candidate = reference + collected["positive_correction"]
        metrics = self._extended_metrics(
            candidate, collected["labels"], collected["sample_ids"]
        )
        target, gate_target, _, _ = self._specialist_targets(
            collected["labels"].view(-1, 1), reference.view(-1, 1)
        )
        correction = collected["positive_correction"].view(-1, 1)
        residual_huber = float(
            _manual_huber(correction - target, self.huber_delta).mean().item()
        )
        target_zero = target <= 1e-8
        false_correction = float(
            correction[target_zero].mean().item()
        ) if bool(target_zero.any()) else 0.0
        gate_prediction = collected["gate_probability"].view(-1, 1) > 0.5
        gate_accuracy = float(
            (gate_prediction == (gate_target > 0.5)).float().mean().item()
        )
        objective = self._selection_objective(metrics)
        return {
            "objective": float(objective),
            "candidate_metrics": metrics,
            "residual_huber": residual_huber,
            "false_correction_mean": false_correction,
            "gate_accuracy": gate_accuracy,
            "target_active_rate": float(gate_target.mean().item()),
            "correction_mean": float(correction.mean().item()),
        }

    def train(self, model, dataloaders):
        model.freeze_legacy()
        self.max_correction = float(model.max_correction)
        optimizer = self._optimizer(model)
        history = []
        valid_committees = self.committees_by_split["valid"]
        best = None
        last_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_row = self._train_epoch(
                model, dataloaders["train"], optimizer
            )
            valid = self.collect(model, dataloaders["valid"], "valid")
            expert = self._expert_validation(valid, valid_committees)
            metrics = expert["candidate_metrics"]
            row = {
                "epoch": epoch,
                **train_row,
                "expert_objective": expert["objective"],
                "expert_candidate_mae": metrics["mae"],
                "expert_candidate_worst_region_mae": metrics[
                    "worst_region_mae"
                ],
                "expert_candidate_ordinary_positive_mae": metrics[
                    "ordinary_positive_mae"
                ],
                "expert_candidate_nonpositive_mae": metrics[
                    "nonpositive_mae"
                ],
                "expert_fold_mae_std": metrics["fold_mae_std"],
                "residual_huber": expert["residual_huber"],
                "false_correction_mean": expert["false_correction_mean"],
                "gate_accuracy": expert["gate_accuracy"],
                "valid_target_active_rate": expert["target_active_rate"],
                "valid_correction_mean": expert["correction_mean"],
            }
            history.append(row)
            LOGGER.info(
                "V9.2 specialist epoch=%d Valid(full-gamma MAE=%.4f worst=%.4f "
                "op=%.4f nonpos=%.4f) residual=%.4f gate=%.3f corr=%.4f",
                epoch,
                metrics["mae"],
                metrics["worst_region_mae"],
                metrics["ordinary_positive_mae"],
                metrics["nonpositive_mae"],
                expert["residual_huber"],
                expert["gate_accuracy"],
                expert["correction_mean"],
            )

            improved = (
                best is None
                or expert["objective"] < best["expert_objective"] - 1e-6
                or (
                    abs(expert["objective"] - best["expert_objective"]) <= 1e-6
                    and metrics["mae"] < best["expert_metrics"]["mae"] - 1e-6
                )
            )
            if improved:
                best = {
                    "epoch": epoch,
                    "expert_objective": float(expert["objective"]),
                    "expert_metrics": {
                        key: value
                        for key, value in metrics.items()
                        if key != "region_rows"
                    },
                    "state": _cpu_state_dict(model),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.early_stop:
                break

        if best is None:
            raise RuntimeError("V9.2 completed no specialist-training epochs.")

        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        selected, policy_rows, _, _ = self._select_policy(
            valid, valid_committees
        )
        best["policy"] = dict(selected)
        best["objective"] = float(selected["objective"])

        for row in history:
            row.update(
                {
                    "selected_source": selected["source"],
                    "selected_beta": selected["beta"],
                    "selected_shrinkage": selected["shrinkage"],
                    "selected_gamma": selected["gamma"],
                    "selected_mae": selected["mae"],
                }
            )
        pd.DataFrame(history).to_csv(
            self.save_dir / "v92_training_history.csv", index=False
        )

        serializable_rows = []
        for row in policy_rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            serializable_rows.append(value)
        pd.DataFrame(serializable_rows).to_csv(
            self.save_dir / "v92_valid_policy_search.csv", index=False
        )
        torch.save(
            best,
            self.save_dir / "positive_residual_specialist_v92_best.pth",
        )
        LOGGER.info(
            "V9.2 frozen specialist epoch=%d; selected source=%s committee=%s "
            "beta=%.2f shrink=%.2f gamma=%.2f Valid(MAE=%.4f op=%.4f)",
            best["epoch"],
            selected["source"],
            selected["committee"],
            selected["beta"],
            selected["shrinkage"],
            selected["gamma"],
            selected["mae"],
            selected["ordinary_positive_mae"],
        )
        return best

    def _best_family_row(self, rows, source):
        candidates = [row for row in rows if row["source"] == source]
        if not candidates:
            return None
        feasible = [row for row in candidates if row["feasible"]]
        pool = feasible or candidates
        return min(
            pool,
            key=lambda row: (
                row["objective"],
                row["mae"],
                row["worst_region_mae"],
            ),
        )

    def evaluate_and_save(self, model, dataloaders, best, bootstrap_samples=2000):
        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        valid_committees = self.committees_by_split["valid"]
        selected, rows, _, valid_reference_metrics = self._select_policy(
            valid, valid_committees
        )

        serializable_rows = []
        for row in rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            serializable_rows.append(value)
        pd.DataFrame(serializable_rows).to_csv(
            self.save_dir / "v92_final_valid_policy.csv", index=False
        )

        # Test is collected only after the specialist checkpoint and final policy
        # have both been frozen by Validation.
        test = self.collect(model, dataloaders["test"], "test")
        test_committees = self.committees_by_split["test"]
        test_reference = self._reference_prediction(test, test_committees)
        test_selected = self._apply_policy(test, test_committees, selected)

        family_rows = {
            source: self._best_family_row(rows, source)
            for source in (
                "legacy_beta_search",
                "zero_shrinkage",
                "positive_residual",
            )
        }
        named = {
            "anchor": test["base_prediction"],
            "global_simplex": test_committees["global_simplex"],
            "original_v71_hybrid": test_reference,
        }
        for source, row in family_rows.items():
            if row is not None:
                named[f"best_valid_{source}"] = self._apply_policy(
                    test, test_committees, row
                )
        named["positive_residual_specialist_v92"] = test_selected

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
            extended = self._extended_metrics(
                prediction, test["labels"], test["sample_ids"]
            )
            extra = {
                "nonpositive_mae": extended["nonpositive_mae"],
                "positive_mae": extended["positive_mae"],
            }
            results[name] = {**metric, **aggregate, **extra}
            comparison_rows.append(
                {"model": name, **metric, **aggregate, **extra}
            )
            region_rows.extend({"model": name, **row} for row in per_region)

        pd.DataFrame(comparison_rows).to_csv(
            self.save_dir / "v92_test_comparison.csv", index=False
        )
        pd.DataFrame(region_rows).to_csv(
            self.save_dir / "v92_test_region_diagnostics.csv", index=False
        )

        prediction_columns = {
            "sample_id": test["sample_ids"],
            "label": test["labels"].view(-1).tolist(),
            "original_v71_hybrid": test_reference.view(-1).tolist(),
            "v92_selected": test_selected.view(-1).tolist(),
            "legacy_student": self._legacy_student(
                test["base_prediction"], test["legacy_prediction"]
            ).view(-1).tolist(),
            "positive_correction": test["positive_correction"].view(-1).tolist(),
            "gate_probability": test["gate_probability"].view(-1).tolist(),
            "magnitude": test["magnitude"].view(-1).tolist(),
        }
        for source, row in family_rows.items():
            if row is not None:
                prediction_columns[f"best_valid_{source}"] = self._apply_policy(
                    test, test_committees, row
                ).view(-1).tolist()
        pd.DataFrame(prediction_columns).to_csv(
            self.save_dir / "v92_test_predictions.csv", index=False
        )

        test_region = sentiment_region_ids(
            test["labels"], self.strong_threshold, self.neutral_radius
        )
        training_contributed = bool(
            selected["source"] == "positive_residual"
            and float(selected["gamma"]) > 0.0
        )
        valid_selected_metrics = self._extended_metrics(
            self._apply_policy(valid, valid_committees, selected),
            valid["labels"],
            valid["sample_ids"],
        )
        valid_selected_metrics.pop("region_rows", None)

        summary = {
            "method": "constrained_positive_residual_specialist_v9_2",
            "selection_protocol": (
                "The V7.1 predictor is frozen. A bounded non-negative correction "
                "is trained on Train against under-prediction residuals of the frozen "
                "V7.1 hybrid. The specialist checkpoint is selected on Validation "
                "using a fixed full-correction candidate. A single compact attribution "
                "search then selects reference, committee reweighting, zero shrinkage, "
                "or positive residual. Test is evaluated once after both choices freeze."
            ),
            "selected_epoch": int(best["epoch"]),
            "selected_policy": selected,
            "training_contributed": training_contributed,
            "max_correction": float(model.max_correction),
            "residual_margin": self.residual_margin,
            "ordinary_positive_boost": self.ordinary_positive_boost,
            "train_region_counts": self.region_counts.tolist(),
            "train_region_weights": self.region_weights.tolist(),
            "expert_valid_metrics": best["expert_metrics"],
            "valid_reference_metrics": {
                key: value
                for key, value in valid_reference_metrics.items()
                if key != "region_rows"
            },
            "valid_selected_metrics": valid_selected_metrics,
            "test_results": results,
            "bootstrap_vs_original_v71": {
                "overall_mae_gain_positive_is_better": self._bootstrap_mae_gain(
                    test_reference,
                    test_selected,
                    test["labels"],
                    samples=int(bootstrap_samples),
                    seed=int(getattr(self.args, "seed", 1111)) + 921,
                ),
                "ordinary_positive_mae_gain_positive_is_better": (
                    self._bootstrap_mae_gain(
                        test_reference,
                        test_selected,
                        test["labels"],
                        mask=test_region == 3,
                        samples=int(bootstrap_samples),
                        seed=int(getattr(self.args, "seed", 1111)) + 922,
                    )
                ),
            },
            "loss_weights": dict(self.loss_weights),
            "family_valid_policies": family_rows,
            "committee": {
                "global_weights": self.committee["global_weights"].tolist(),
                "region_weights": self.committee["region_weights"].tolist(),
                "global_cv_score": self.committee["global_cv_score"],
                "region_cv_score": self.committee["region_cv_score"],
            },
        }
        (self.save_dir / "positive_residual_specialist_v92_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "V9.2 final source=%s epoch=%d training_contributed=%s "
            "Test(MAE=%.4f worst=%.4f op=%.4f nonpos=%.4f) reference_MAE=%.4f",
            selected["source"],
            best["epoch"],
            training_contributed,
            results["positive_residual_specialist_v92"]["MAE"],
            results["positive_residual_specialist_v92"]["worst_region_mae"],
            results["positive_residual_specialist_v92"]["ordinary_positive_mae"],
            results["positive_residual_specialist_v92"]["nonpositive_mae"],
            results["original_v71_hybrid"]["MAE"],
        )
        return summary
