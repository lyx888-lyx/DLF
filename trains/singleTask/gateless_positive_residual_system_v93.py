"""Gateless positive-residual training and attribution for V9.3.

The V9.2 gate collapsed to an almost always-off solution. V9.3 separates the
questions "can the residual magnitude be learned?" and "how much of it should
be deployed?" by training a bounded positive correction directly and selecting
its global deployment scale only on Validation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

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


class GatelessPositiveResidualTrainerV93(PositiveResidualSpecialistTrainerV92):
    """Train correction magnitude directly; freeze policy selection until after."""

    def __init__(
        self,
        *args,
        no_harm_weight: float = 0.05,
        overcorrection_weight: float = 0.05,
        positive_target_weight: float = 1.50,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.no_harm_weight = float(no_harm_weight)
        self.overcorrection_weight = float(overcorrection_weight)
        self.positive_target_weight = float(positive_target_weight)
        self.loss_weights = {
            "correction": 1.00,
            "positive_target": self.positive_target_weight,
            "no_harm": self.no_harm_weight,
            "overcorrection": self.overcorrection_weight,
        }

    def _optimizer(self, model):
        parameters = model.trainable_parameters()
        if not parameters:
            raise RuntimeError("V9.3 has no trainable residual parameters.")
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

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

        target, _, raw_residual, positive_label = self._specialist_targets(
            labels, reference
        )
        correction = output["positive_correction"]
        active = target > self.residual_margin

        correction_loss = _weighted_mean(
            _manual_huber(correction - target, self.huber_delta),
            sample_weights,
        )

        if bool(active.any()):
            positive_target = _weighted_mean(
                _manual_huber(
                    correction[active] - target[active], self.huber_delta
                ),
                sample_weights[active],
            )
        else:
            positive_target = correction.new_zeros(())

        target_zero = target <= 1e-8
        if bool(target_zero.any()):
            no_harm = _weighted_mean(
                correction[target_zero], sample_weights[target_zero]
            )
        else:
            no_harm = correction.new_zeros(())

        allowed = raw_residual.clamp_min(0.0)
        overcorrection = _weighted_mean(
            F.relu(correction - allowed), sample_weights
        )

        losses = {
            "correction": correction_loss,
            "positive_target": positive_target,
            "no_harm": no_harm,
            "overcorrection": overcorrection,
        }
        total = correction.new_zeros(())
        for name, value in losses.items():
            total = total + float(self.loss_weights.get(name, 0.0)) * value
        losses["total"] = total

        diagnostics = {
            "target_mean": target.mean().detach(),
            "target_active_rate": active.float().mean().detach(),
            "correction_mean": correction.mean().detach(),
            "correction_max": correction.max().detach(),
            "positive_label_rate": positive_label.float().mean().detach(),
        }
        return losses, diagnostics

    def _train_epoch(self, model, dataloader, optimizer):
        model.set_train_mode()
        totals: Dict[str, float] = {}
        optimizer.zero_grad()
        for batch in tqdm(dataloader, leave=False):
            indices = self._batch_indices(batch.get("id"), "train")
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
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    def _batch_indices(self, batch_ids, split_name):
        from .positive_residual_specialist_system_v92 import _batch_cache_indices

        return _batch_cache_indices(batch_ids, self.index[split_name])

    def _expert_validation(self, collected, committees):
        reference = self._reference_prediction(collected, committees)
        labels = collected["labels"].view(-1, 1)
        correction = collected["positive_correction"].view(-1, 1)
        target, _, _, _ = self._specialist_targets(labels, reference.view(-1, 1))
        residual_huber = float(
            _manual_huber(correction - target, self.huber_delta).mean().item()
        )
        active = target > self.residual_margin
        active_huber = float(
            _manual_huber(
                correction[active] - target[active], self.huber_delta
            ).mean().item()
        ) if bool(active.any()) else 0.0
        target_zero = target <= 1e-8
        false_correction = float(
            correction[target_zero].mean().item()
        ) if bool(target_zero.any()) else 0.0

        # Checkpoint selection measures residual learning directly, before any
        # committee/beta/gamma search is allowed.
        objective = (
            residual_huber
            + self.no_harm_weight * false_correction
            + 0.25 * active_huber
        )
        return {
            "objective": float(objective),
            "residual_huber": residual_huber,
            "active_huber": active_huber,
            "false_correction_mean": false_correction,
            "target_active_rate": float(active.float().mean().item()),
            "correction_mean": float(correction.mean().item()),
            "correction_max": float(correction.max().item()),
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
            train_row = self._train_epoch(model, dataloaders["train"], optimizer)
            valid = self.collect(model, dataloaders["valid"], "valid")
            expert = self._expert_validation(valid, valid_committees)
            row = {"epoch": epoch, **train_row, **{
                f"valid_{key}": value for key, value in expert.items()
            }}
            history.append(row)
            LOGGER.info(
                "V9.3 magnitude epoch=%d Valid(residual=%.4f active=%.4f "
                "false=%.4f corr_mean=%.4f corr_max=%.4f)",
                epoch,
                expert["residual_huber"],
                expert["active_huber"],
                expert["false_correction_mean"],
                expert["correction_mean"],
                expert["correction_max"],
            )
            improved = (
                best is None
                or expert["objective"] < best["expert_objective"] - 1e-6
            )
            if improved:
                best = {
                    "epoch": epoch,
                    "expert_objective": float(expert["objective"]),
                    "expert_metrics": dict(expert),
                    "state": _cpu_state_dict(model),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.early_stop:
                break

        if best is None:
            raise RuntimeError("V9.3 completed no magnitude-training epochs.")

        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        selected, policy_rows, _, _ = self._select_policy(valid, valid_committees)

        for row in history:
            row.update({
                "selected_source": selected["source"],
                "selected_beta": selected["beta"],
                "selected_shrinkage": selected["shrinkage"],
                "selected_gamma": selected["gamma"],
                "selected_mae": selected["mae"],
            })
        pd.DataFrame(history).to_csv(
            self.save_dir / "v93_training_history.csv", index=False
        )
        serializable = []
        for row in policy_rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            serializable.append(value)
        pd.DataFrame(serializable).to_csv(
            self.save_dir / "v93_valid_policy_search.csv", index=False
        )

        best.update({
            "objective": float(selected["objective"]),
            "policy": dict(selected),
        })
        torch.save(best, self.save_dir / "gateless_positive_residual_v93_best.pth")
        LOGGER.info(
            "V9.3 frozen magnitude epoch=%d; selected source=%s committee=%s "
            "beta=%.2f shrink=%.2f gamma=%.2f Valid(MAE=%.4f op=%.4f)",
            best["epoch"], selected["source"], selected["committee"],
            selected["beta"], selected["shrinkage"], selected["gamma"],
            selected["mae"], selected["ordinary_positive_mae"],
        )
        return best

    def evaluate_and_save(self, model, dataloaders, best, bootstrap_samples=2000):
        summary = super().evaluate_and_save(
            model, dataloaders, best, bootstrap_samples=bootstrap_samples
        )
        # Keep the inherited auditable outputs and add an explicit V9.3 marker.
        summary["method"] = "gateless_positive_residual_specialist_v9_3"
        summary["training_protocol"] = (
            "Bounded positive residual magnitude is trained directly on Train. "
            "Its checkpoint is selected by residual prediction quality on Valid. "
            "Committee, shrinkage and deployment gamma are selected once after "
            "the magnitude checkpoint is frozen; Test is evaluated once."
        )
        summary["gate_removed"] = True
        summary["training_contributed"] = bool(
            summary["selected_policy"]["source"] == "positive_residual"
            and float(summary["selected_policy"]["gamma"]) > 0.0
        )
        target = self.save_dir / "gateless_positive_residual_v93_summary.json"
        target.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
