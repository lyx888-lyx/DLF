"""Decoupled training and attribution for region-balanced ordinal mixture V9.1.

V9.1 fixes the main V9 failure mode: the new expert path is trained directly,
rather than through a tiny blend into the frozen V7.1 prediction.  Final
combination weights are selected only on Validation, with explicit attribution
candidates for committee reweighting, zero shrinkage, and the trained mixture.
"""

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

from .ordinal_consistency_system_v74 import _fast_metrics, _fold_metrics
from .region_balanced_ordinal_mixture_system_v9 import (
    REGION_NAMES,
    THRESHOLDS_5,
    THRESHOLDS_7,
    RegionBalancedOrdinalMixtureTrainerV9,
    _cpu_state_dict,
    _manual_huber,
    _weighted_mean,
    polarity_targets,
    region_diagnostics,
    sentiment_region_ids,
)


LOGGER = logging.getLogger("MMSA")


class DecoupledRegionMixtureTrainerV91(RegionBalancedOrdinalMixtureTrainerV9):
    """Train the new mixture directly, then select a frozen Valid-only blend."""

    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths: Sequence[Path],
        init_index: int,
        source_summary: Mapping[str, object],
        max_epochs: int = 20,
        early_stop: int = 6,
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
        stability_selection_weight: float = 0.05,
        folds: int = 3,
        beta_grid=(0.40, 0.50, 0.60),
        gamma_grid=(0.05, 0.10, 0.20, 0.30),
        zero_shrinkage_grid=(0.02, 0.05, 0.10),
        loss_weights=None,
    ):
        # The V9 base supplies aligned collection, committee fitting, region
        # weights, diagnostics, and bootstrap utilities.  Ordinary-positive
        # error is deliberately not part of policy selection because MOSI Valid
        # contains too few samples in that region for stable tuning.
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
        self.gamma_grid = tuple(
            sorted({float(value) for value in gamma_grid if float(value) > 0.0})
        )
        self.zero_shrinkage_grid = tuple(
            sorted({float(value) for value in zero_shrinkage_grid if float(value) > 0.0})
        )
        if not self.gamma_grid:
            raise ValueError("gamma_grid must contain at least one positive value.")
        if not self.zero_shrinkage_grid:
            raise ValueError(
                "zero_shrinkage_grid must contain at least one positive value."
            )
        if any(value > 1.0 for value in (*self.gamma_grid, *self.zero_shrinkage_grid)):
            raise ValueError("All gamma values must be in (0, 1].")

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
            "gate_balance": 0.01,
        }
        pd.DataFrame(
            {
                "region": REGION_NAMES,
                "count": self.region_counts.tolist(),
                "weight": self.region_weights.tolist(),
            }
        ).to_csv(self.save_dir / "v91_train_region_weights.csv", index=False)

    def _fit_committee(self):
        fitted = super()._fit_committee()
        # Keep a versioned copy even though the inherited implementation also
        # writes the V9 filename.
        pd.DataFrame(fitted["cv_rows"]).to_csv(
            Path(self.save_dir) / "v91_committee_cv.csv", index=False
        )
        return fitted

    def _optimizer(self, model):
        parameters = model.trainable_parameters()
        if not parameters:
            raise RuntimeError("V9.1 has no trainable mixture parameters.")
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _direct_loss(self, output, labels):
        """Optimize the standalone mixture, not the tiny V9 blended prediction."""

        labels = labels.view(-1, 1)
        region = sentiment_region_ids(
            labels, self.strong_threshold, self.neutral_radius
        )
        polarity = polarity_targets(labels, self.neutral_radius)
        sample_weights = self.region_weights.to(labels.device)[region].view(-1, 1)

        direct_error = output["mixture_value"] - labels
        per_sample_huber = _manual_huber(direct_error, self.huber_delta)
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
            dro_probs = torch.softmax(dro_logits[present], dim=0)
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
            _manual_huber(
                output["ordinal_consensus"] - labels, self.huber_delta
            ),
            sample_weights,
        )
        ordinal_consistency = _weighted_mean(
            _manual_huber(
                output["mixture_value"] - output["ordinal_consensus"],
                self.huber_delta,
            ),
            sample_weights,
        )

        # Penalize genuine sign crossings, softly down-weighting labels close to
        # zero where annotation ambiguity is highest.
        sign = torch.sign(labels)
        non_neutral = (labels.abs() > max(self.neutral_radius, 1e-6)).float()
        distance_weight = (
            labels.abs() / max(self.strong_threshold, 1e-6)
        ).clamp(0.0, 1.0)
        cross_zero = F.relu(-sign * output["mixture_value"])
        cross_weight = sample_weights * non_neutral * distance_weight
        cross_zero = (
            (cross_zero * cross_weight).sum()
            / cross_weight.sum().clamp_min(1e-8)
        )

        # A very small marginal-balance term prevents immediate gate collapse.
        # It does not impose a uniform per-sample gate.
        mean_gate = output["polarity_probs"].mean(dim=0)
        gate_balance = (mean_gate * mean_gate.clamp_min(1e-8).log()).sum()

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
            "gate_balance": gate_balance,
        }
        total = direct_error.new_zeros(())
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
            losses, group_means, present = self._direct_loss(output, labels)
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
                totals[name] = totals.get(name, 0.0) + float(
                    value.detach().item()
                )

        count = max(1, len(dataloader))
        row = {name: value / count for name, value in totals.items()}
        for index, name in enumerate(REGION_NAMES):
            row[f"dro_probability_{name}"] = float(
                torch.softmax(self.dro_logits, dim=0)[index].item()
            )
        return row

    def _legacy_student(self, collected):
        alpha = float(self.source_summary["student_policy"]["alpha"])
        return collected["base_prediction"] + alpha * (
            collected["legacy_prediction"] - collected["base_prediction"]
        )

    def _selection_objective(self, metrics):
        # Overall MAE remains primary. Worst-region and fold terms are small
        # robustness regularizers; ordinary-positive Valid MAE is diagnostic only.
        return (
            metrics["mae"]
            + self.robust_selection_weight * metrics["worst_region_mae"]
            + self.stability_selection_weight * metrics["fold_mae_std"]
        )

    def _candidate_row(
        self,
        source,
        committee,
        beta,
        gamma,
        prediction,
        labels,
        sample_ids,
        reference_metrics,
    ):
        metrics = self._policy_metrics(prediction, labels, sample_ids)
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
            and all(
                metrics["fold_maes"][index] <= fold_limits[index] + 1e-12
                for index in range(self.folds)
            )
        )
        return {
            "source": str(source),
            "committee": str(committee),
            "beta": float(beta),
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

    def _select_policy(self, collected, committees, allow_trained: bool):
        labels = collected["labels"]
        sample_ids = collected["sample_ids"]
        reference = self._reference_prediction(collected, committees)
        reference_metrics = self._policy_metrics(reference, labels, sample_ids)
        source_policy = self.source_summary["hybrid_policy"]
        reference_row = {
            "source": "legacy_reference",
            "committee": str(source_policy["committee"]),
            "beta": float(source_policy["beta"]),
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

        legacy = self._legacy_student(collected)
        source_committee = str(source_policy["committee"])
        committee_names = tuple(
            dict.fromkeys(
                (source_committee, "global_simplex", "region_simplex")
            )
        )
        source_beta = float(source_policy["beta"])
        beta_grid = tuple(dict.fromkeys((*self.beta_grid, source_beta)))

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
                    beta_only,
                    labels,
                    sample_ids,
                    reference_metrics,
                )
                rows.append(row)
                if row["feasible"]:
                    feasible.append(row)

                for gamma in self.zero_shrinkage_grid:
                    shrunk = (1.0 - gamma) * legacy
                    prediction = (
                        beta * committee_prediction + (1.0 - beta) * shrunk
                    )
                    row = self._candidate_row(
                        "zero_shrinkage",
                        committee_name,
                        beta,
                        gamma,
                        prediction,
                        labels,
                        sample_ids,
                        reference_metrics,
                    )
                    rows.append(row)
                    if row["feasible"]:
                        feasible.append(row)

                # Epoch 0 is intentionally excluded from this family. The
                # symmetric initialization is approximately zero and would
                # otherwise masquerade as a learned expert.
                if allow_trained:
                    for gamma in self.gamma_grid:
                        mixed = (1.0 - gamma) * legacy + gamma * collected[
                            "mixture_value"
                        ]
                        prediction = (
                            beta * committee_prediction
                            + (1.0 - beta) * mixed
                        )
                        row = self._candidate_row(
                            "trained_mixture",
                            committee_name,
                            beta,
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
                row["fold_mae_std"],
                row["source"] != "legacy_reference",
            ),
        )
        return selected, rows, reference, reference_metrics

    def train(self, model, dataloaders):
        model.freeze_legacy()
        # The old V9 scalar blend is not trained or used for selection. This
        # removes the tiny-gradient bottleneck that caused V9 to select epoch 0.
        model.mixture_blend_logit.requires_grad_(False)
        optimizer = self._optimizer(model)

        history = []
        policy_rows = []
        valid_committees = self._committee_predictions("valid")

        initial_valid = self.collect(model, dataloaders["valid"], "valid")
        initial_policy, initial_rows, _, _ = self._select_policy(
            initial_valid, valid_committees, allow_trained=False
        )
        policy_rows.extend({"epoch": 0, **row} for row in initial_rows)
        direct_initial = self._policy_metrics(
            initial_valid["mixture_value"],
            initial_valid["labels"],
            initial_valid["sample_ids"],
        )
        best = {
            "epoch": 0,
            "objective": float(initial_policy["objective"]),
            "state": _cpu_state_dict(model),
            "policy": dict(initial_policy),
        }
        last_improvement = 0

        LOGGER.info(
            "V9.1 epoch=0 source=%s Valid(MAE=%.4f worst=%.4f op=%.4f) "
            "direct_mixture_MAE=%.4f",
            initial_policy["source"],
            initial_policy["mae"],
            initial_policy["worst_region_mae"],
            initial_policy["ordinary_positive_mae"],
            direct_initial["mae"],
        )

        for epoch in range(1, self.max_epochs + 1):
            train_row = self._train_epoch(model, dataloaders["train"], optimizer)
            valid = self.collect(model, dataloaders["valid"], "valid")
            selected, rows, _, reference_metrics = self._select_policy(
                valid, valid_committees, allow_trained=True
            )
            policy_rows.extend({"epoch": epoch, **row} for row in rows)
            direct_metrics = self._policy_metrics(
                valid["mixture_value"], valid["labels"], valid["sample_ids"]
            )
            row = {
                "epoch": epoch,
                **train_row,
                "selected_source": selected["source"],
                "selected_objective": selected["objective"],
                "selected_mae": selected["mae"],
                "selected_worst_region_mae": selected["worst_region_mae"],
                "selected_ordinary_positive_mae": selected[
                    "ordinary_positive_mae"
                ],
                "selected_beta": selected["beta"],
                "selected_gamma": selected["gamma"],
                "reference_mae": reference_metrics["mae"],
                "direct_mixture_mae": direct_metrics["mae"],
                "direct_mixture_worst_region_mae": direct_metrics[
                    "worst_region_mae"
                ],
                "direct_mixture_ordinary_positive_mae": direct_metrics[
                    "ordinary_positive_mae"
                ],
            }
            history.append(row)
            LOGGER.info(
                "V9.1 epoch=%d source=%s Valid(MAE=%.4f worst=%.4f op=%.4f) "
                "beta=%.2f gamma=%.2f direct_MAE=%.4f",
                epoch,
                selected["source"],
                selected["mae"],
                selected["worst_region_mae"],
                selected["ordinary_positive_mae"],
                selected["beta"],
                selected["gamma"],
                direct_metrics["mae"],
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
            self.save_dir / "v91_training_history.csv", index=False
        )
        serializable_rows = []
        for row in policy_rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            serializable_rows.append(value)
        pd.DataFrame(serializable_rows).to_csv(
            self.save_dir / "v91_valid_policy_search.csv", index=False
        )
        torch.save(best, self.save_dir / "decoupled_region_mixture_v91_best.pth")
        return best

    def _apply_policy(self, collected, committees, policy):
        if policy["source"] == "legacy_reference":
            return self._reference_prediction(collected, committees)

        beta = float(policy["beta"])
        gamma = float(policy.get("gamma", 0.0))
        committee = committees[policy["committee"]]
        legacy = self._legacy_student(collected)

        if policy["source"] == "legacy_beta_search":
            student = legacy
        elif policy["source"] == "zero_shrinkage":
            student = (1.0 - gamma) * legacy
        elif policy["source"] == "trained_mixture":
            student = (
                (1.0 - gamma) * legacy
                + gamma * collected["mixture_value"]
            )
        else:
            raise ValueError(f"Unknown V9.1 policy source: {policy['source']}")
        return beta * committee + (1.0 - beta) * student

    def _attribution_predictions(self, collected, committees, policy):
        beta = float(policy["beta"])
        gamma = float(policy.get("gamma", 0.0))
        committee = committees[policy["committee"]]
        legacy = self._legacy_student(collected)
        beta_only = beta * committee + (1.0 - beta) * legacy
        zero_shrinkage = beta * committee + (1.0 - beta) * (
            (1.0 - gamma) * legacy
        )
        trained_mixture = beta * committee + (1.0 - beta) * (
            (1.0 - gamma) * legacy + gamma * collected["mixture_value"]
        )
        return {
            "selected_beta_only": beta_only,
            "selected_zero_shrinkage": zero_shrinkage,
            "selected_trained_mixture": trained_mixture,
        }

    def evaluate_and_save(
        self, model, dataloaders, best, bootstrap_samples=2000
    ):
        model.load_state_dict(best["state"])
        model.to(self.args.device)

        valid = self.collect(model, dataloaders["valid"], "valid")
        valid_committees = self._committee_predictions("valid")
        allow_trained = int(best["epoch"]) > 0
        selected, rows, _, valid_reference_metrics = self._select_policy(
            valid, valid_committees, allow_trained=allow_trained
        )
        final_rows = []
        for row in rows:
            value = dict(row)
            value["fold_maes"] = json.dumps(value["fold_maes"])
            final_rows.append(value)
        pd.DataFrame(final_rows).to_csv(
            self.save_dir / "v91_final_valid_policy.csv", index=False
        )

        if (
            selected["source"] != best["policy"]["source"]
            or selected["committee"] != best["policy"]["committee"]
            or abs(float(selected["beta"]) - float(best["policy"]["beta"])) > 1e-8
            or abs(float(selected["gamma"]) - float(best["policy"]["gamma"])) > 1e-8
        ):
            raise RuntimeError(
                "Frozen V9.1 Valid policy does not reproduce the checkpoint policy: "
                f"checkpoint={best['policy']} recomputed={selected}"
            )

        # Test is touched only after the epoch and policy are frozen.
        test = self.collect(model, dataloaders["test"], "test")
        test_committees = self._committee_predictions("test")
        test_reference = self._reference_prediction(test, test_committees)
        test_selected = self._apply_policy(test, test_committees, selected)
        attribution = self._attribution_predictions(
            test, test_committees, selected
        )

        named = {
            "anchor": test["base_prediction"],
            "global_simplex": test_committees["global_simplex"],
            "original_v71_hybrid": test_reference,
            **attribution,
            "decoupled_region_mixture_v91": test_selected,
            "direct_trained_mixture": test["mixture_value"],
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
            self.save_dir / "v91_test_comparison.csv", index=False
        )
        pd.DataFrame(region_rows).to_csv(
            self.save_dir / "v91_test_region_diagnostics.csv", index=False
        )

        probs = test["polarity_probs"]
        experts = test["expert_values"]
        pd.DataFrame(
            {
                "sample_id": test["sample_ids"],
                "label": test["labels"].view(-1).tolist(),
                "original_v71_hybrid": test_reference.view(-1).tolist(),
                "v91_selected": test_selected.view(-1).tolist(),
                "selected_beta_only": attribution[
                    "selected_beta_only"
                ].view(-1).tolist(),
                "selected_zero_shrinkage": attribution[
                    "selected_zero_shrinkage"
                ].view(-1).tolist(),
                "selected_trained_mixture": attribution[
                    "selected_trained_mixture"
                ].view(-1).tolist(),
                "legacy_student": self._legacy_student(test).view(-1).tolist(),
                "direct_trained_mixture": test["mixture_value"].view(-1).tolist(),
                "gate_negative": probs[:, 0].tolist(),
                "gate_neutral": probs[:, 1].tolist(),
                "gate_positive": probs[:, 2].tolist(),
                "expert_negative": experts[:, 0].tolist(),
                "expert_neutral": experts[:, 1].tolist(),
                "expert_positive": experts[:, 2].tolist(),
                "ordinal7_value": test["ordinal7_value"].view(-1).tolist(),
                "ordinal5_value": test["ordinal5_value"].view(-1).tolist(),
            }
        ).to_csv(self.save_dir / "v91_test_predictions.csv", index=False)

        test_region = sentiment_region_ids(
            test["labels"], self.strong_threshold, self.neutral_radius
        )
        training_contributed = bool(
            int(best["epoch"]) > 0
            and selected["source"] == "trained_mixture"
            and float(selected["gamma"]) > 0.0
        )
        summary = {
            "method": "decoupled_region_balanced_ordinal_mixture_v9_1",
            "selection_protocol": (
                "The V7.1 backbone and residual Student are frozen. The new "
                "mixture is trained directly on Train. Epoch and final policy "
                "are selected only on Validation. Epoch-0 mixture predictions "
                "are excluded from the trained-mixture family. Test is evaluated "
                "once after freezing the complete policy."
            ),
            "selected_epoch": int(best["epoch"]),
            "selected_policy": selected,
            "training_contributed": training_contributed,
            "region_definition": {
                "strong_threshold": self.strong_threshold,
                "neutral_radius": self.neutral_radius,
                "names": list(REGION_NAMES),
            },
            "train_region_counts": self.region_counts.tolist(),
            "train_region_weights": self.region_weights.tolist(),
            "valid_reference_metrics": {
                key: value
                for key, value in valid_reference_metrics.items()
                if key != "region_rows"
            },
            "valid_selected_metrics": self._policy_metrics(
                self._apply_policy(valid, valid_committees, selected),
                valid["labels"],
                valid["sample_ids"],
            ),
            "valid_direct_mixture_metrics": self._policy_metrics(
                valid["mixture_value"],
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
                    seed=int(getattr(self.args, "seed", 1111)) + 911,
                ),
                "ordinary_positive_mae_gain_positive_is_better": (
                    self._bootstrap_mae_gain(
                        test_reference,
                        test_selected,
                        test["labels"],
                        mask=test_region == 3,
                        samples=int(bootstrap_samples),
                        seed=int(getattr(self.args, "seed", 1111)) + 912,
                    )
                ),
            },
            "loss_weights": dict(self.loss_weights),
            "group_dro_weight": self.group_dro_weight,
            "group_dro_eta": self.group_dro_eta,
            "beta_grid": list(self.beta_grid),
            "gamma_grid": list(self.gamma_grid),
            "zero_shrinkage_grid": list(self.zero_shrinkage_grid),
            "committee": {
                "global_weights": self.committee["global_weights"].tolist(),
                "region_weights": self.committee["region_weights"].tolist(),
                "global_cv_score": self.committee["global_cv_score"],
                "region_cv_score": self.committee["region_cv_score"],
            },
        }
        summary["valid_selected_metrics"].pop("region_rows", None)
        summary["valid_direct_mixture_metrics"].pop("region_rows", None)
        (self.save_dir / "decoupled_region_mixture_v91_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "V9.1 final source=%s epoch=%d training_contributed=%s "
            "Test(MAE=%.4f worst=%.4f op=%.4f) reference_MAE=%.4f",
            selected["source"],
            best["epoch"],
            training_contributed,
            results["decoupled_region_mixture_v91"]["MAE"],
            results["decoupled_region_mixture_v91"]["worst_region_mae"],
            results["decoupled_region_mixture_v91"][
                "ordinary_positive_mae"
            ],
            results["original_v71_hybrid"]["MAE"],
        )
        return summary
