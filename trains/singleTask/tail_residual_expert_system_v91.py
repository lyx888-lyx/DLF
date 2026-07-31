"""Leakage-controlled training and evaluation for V9.1 tail experts."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids
from .role_conditioned_experts_v9 import apply_teacher_weights
from .tail_residual_experts_v91 import (
    MECHANISM_NAMES,
    TAIL_ROLE_NAMES,
    TailLossWeights,
    exact_tail_mask,
    residual_diagnostic_rows,
    residual_mechanism_index,
    tail_capability_metrics,
    tail_residual_loss,
)
from .model.TailResidualDLF import TailResidualDLF


logger = logging.getLogger("MMSA")


def _cpu_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _cache_index(sample_ids: Sequence[object]) -> Dict[str, int]:
    mapping: Dict[str, int] = {}
    for index, value in enumerate(sample_ids):
        key = str(value)
        if key in mapping:
            raise RuntimeError("Duplicate sample id in V9.1 teacher cache: %s" % key)
        mapping[key] = index
    return mapping


def _batch_cache_indices(batch_ids, mapping: Mapping[str, int]) -> torch.Tensor:
    values = normalize_batch_ids(batch_ids)
    try:
        return torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    except KeyError as error:
        raise KeyError("V9.1 sample id missing from teacher cache: %s" % error)


def _safe_metrics(metrics_fn, prediction, labels) -> Dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _candidate_mask_for_region(labels: torch.Tensor, region_name: str):
    if region_name == "strong_negative":
        return labels.view(-1) < -1.5
    if region_name == "strong_positive":
        return labels.view(-1) > 1.5
    raise ValueError("unsupported tail region: %s" % region_name)


class TailResidualExpertTrainerV91:
    """Train shared and sign-specific tail residual candidates from one anchor."""

    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths: Sequence[Path],
        anchor_index: int,
        global_teacher_weights: torch.Tensor,
        role_teacher_weights: torch.Tensor,
        hidden_dim: int = 192,
        dropout: float = 0.15,
        residual_max: float = 1.50,
        max_epochs: int = 24,
        early_stop: int = 7,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-3,
        membership_temperature: float = 0.25,
        gain_margin: float = 0.08,
        gain_fraction: float = 0.20,
        global_tolerance: float = 0.012,
        max_harm_rate: float = 0.15,
        min_teacher_gain: float = 0.002,
        loss_weights: TailLossWeights = TailLossWeights(),
    ) -> None:
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache = teacher_cache
        self.teacher_paths = [Path(value) for value in teacher_paths]
        self.anchor_index = int(anchor_index)
        self.global_teacher_weights = global_teacher_weights.detach().cpu().float()
        self.role_teacher_weights = role_teacher_weights.detach().cpu().float()
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual_max = float(residual_max)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.membership_temperature = float(membership_temperature)
        self.gain_margin = float(gain_margin)
        self.gain_fraction = float(gain_fraction)
        self.global_tolerance = float(global_tolerance)
        self.max_harm_rate = float(max_harm_rate)
        self.min_teacher_gain = float(min_teacher_gain)
        self.loss_weights = loss_weights
        self.index = {
            split: _cache_index(self.cache["splits"][split]["sample_ids"])
            for split in ("train", "valid", "test")
        }
        if self.role_teacher_weights.shape != (5, len(self.teacher_paths)):
            raise ValueError("V9.1 role teacher matrix must have shape [5, K].")

        self.teacher_policies = self._build_teacher_policies()
        pd.DataFrame(self.teacher_policies.values()).to_csv(
            self.save_dir / "v91_tail_teacher_diagnostics.csv", index=False
        )
        self._write_anchor_residual_diagnostics()

    def _fixed_anchor(self, split_name: str) -> torch.Tensor:
        return self.cache["splits"][split_name]["predictions"][:, self.anchor_index].float()

    def _teacher_weights_for_role(self, role_name: str) -> torch.Tensor:
        if role_name == "strong_negative":
            return self.role_teacher_weights[0]
        if role_name == "strong_positive":
            return self.role_teacher_weights[4]
        shared = self.role_teacher_weights[0] + self.role_teacher_weights[4]
        return shared / shared.sum().clamp_min(1e-8)

    def _build_teacher_policies(self):
        valid = self.cache["splits"]["valid"]
        predictions = valid["predictions"].float()
        labels = valid["labels"].float()
        anchor = self._fixed_anchor("valid")
        policies = {}
        for role_name in TAIL_ROLE_NAMES:
            mask = exact_tail_mask(labels, role_name)
            weights = self._teacher_weights_for_role(role_name)
            teacher = apply_teacher_weights(predictions, weights)
            anchor_mae = float(torch.abs(anchor[mask] - labels[mask]).mean().item())
            teacher_mae = float(torch.abs(teacher[mask] - labels[mask]).mean().item())
            gain = anchor_mae - teacher_mae
            enabled = bool(gain >= self.min_teacher_gain)
            policies[role_name] = {
                "role": role_name,
                "count": int(mask.sum().item()),
                "anchor_mae": anchor_mae,
                "teacher_mae": teacher_mae,
                "teacher_gain": gain,
                "teacher_enabled": enabled,
                "teacher_scale": 1.0 if enabled else 0.0,
                "teacher_weights": weights.tolist(),
            }
        return policies

    def _write_anchor_residual_diagnostics(self) -> None:
        rows = []
        for split_name in ("train", "valid", "test"):
            split = self.cache["splits"][split_name]
            rows.extend(
                residual_diagnostic_rows(
                    self._fixed_anchor(split_name),
                    split["labels"],
                    split["sample_ids"],
                    split_name,
                )
            )
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v91_anchor_residual_diagnostics.csv", index=False
        )

    def _new_model(self) -> TailResidualDLF:
        model = TailResidualDLF(
            self.args,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            residual_max=self.residual_max,
        ).to(self.args.device)
        incompatible = model.load_anchor_checkpoint(
            self.teacher_paths[self.anchor_index], map_location=self.args.device
        )
        model.freeze_anchor()
        logger.info(
            "Loaded immutable V9.1 anchor (missing=%d unexpected=%d): %s",
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
            self.teacher_paths[self.anchor_index],
        )
        return model

    def _optimizer(self, model: TailResidualDLF) -> optim.Optimizer:
        return optim.AdamW(
            model.parameter_groups(self.learning_rate),
            weight_decay=self.weight_decay,
        )

    def _teacher_target(self, split_name: str, role_name: str) -> torch.Tensor:
        split = self.cache["splits"][split_name]
        weights = self._teacher_weights_for_role(role_name)
        return apply_teacher_weights(split["predictions"].float(), weights)

    def _train_epoch(self, model, dataloader, optimizer, role_name: str):
        model.set_train_mode()
        anchor_all = self._fixed_anchor("train")
        teacher_all = self._teacher_target("train", role_name)
        teacher_scale = float(self.teacher_policies[role_name]["teacher_scale"])
        totals: Dict[str, float] = {}
        accumulation = max(1, int(getattr(self.args, "update_epochs", 1)))
        optimizer.zero_grad()

        for step, batch in enumerate(tqdm(dataloader, leave=False), start=1):
            indices = _batch_cache_indices(batch.get("id"), self.index["train"])
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            losses = tail_residual_loss(
                output,
                labels,
                anchor_all[indices].to(self.args.device),
                teacher_all[indices].to(self.args.device),
                role=role_name,
                membership_temperature=self.membership_temperature,
                gain_margin=self.gain_margin,
                gain_fraction=self.gain_fraction,
                teacher_scale=teacher_scale,
                weights=self.loss_weights,
            )
            (losses["total"] / accumulation).backward()
            if step % accumulation == 0 or step == len(dataloader):
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_norm=2.0
                )
                optimizer.step()
                optimizer.zero_grad()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    @torch.no_grad()
    def collect(self, model, dataloader, split_name: str):
        model.eval()
        split = self.cache["splits"][split_name]
        mapping = self.index[split_name]
        sample_count = len(split["sample_ids"])
        keys = (
            "anchor_prediction",
            "prediction",
            "raw_correction",
            "correction",
            "applicability_prob",
            "mechanism_probs",
        )
        buffers = None
        seen = torch.zeros(sample_count, dtype=torch.bool)
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
                        (sample_count,) + tuple(output[key].shape[1:]),
                        dtype=output[key].dtype,
                    )
                    for key in keys
                }
            for key in keys:
                buffers[key][indices] = output[key].detach().cpu()
            seen[indices] = True
        if buffers is None or not bool(seen.all()):
            raise RuntimeError(
                "Incomplete V9.1 collection for %s: missing=%d"
                % (split_name, int((~seen).sum().item()))
            )

        fixed_anchor = self._fixed_anchor(split_name)
        max_anchor_diff = float(
            torch.abs(buffers["anchor_prediction"] - fixed_anchor).max().item()
        )
        if max_anchor_diff > 5e-5:
            raise RuntimeError(
                "Model anchor differs from fixed teacher cache for %s: %.8f"
                % (split_name, max_anchor_diff)
            )
        return {
            "sample_ids": list(split["sample_ids"]),
            "labels": split["labels"].clone(),
            "fixed_anchor": fixed_anchor.clone(),
            "max_anchor_diff": max_anchor_diff,
            **buffers,
        }

    def _validation_row(self, collected, role_name: str):
        labels = collected["labels"]
        prediction = collected["prediction"]
        anchor = collected["fixed_anchor"]
        mask = exact_tail_mask(labels, role_name)
        global_mae = float(torch.abs(prediction - labels).mean().item())
        tail_mae = float(torch.abs(prediction[mask] - labels[mask]).mean().item())
        anchor_global = float(torch.abs(anchor - labels).mean().item())
        anchor_tail = float(torch.abs(anchor[mask] - labels[mask]).mean().item())
        realized_gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
        harm_rate = float((realized_gain < -0.10).float().mean().item())
        global_delta = global_mae - anchor_global
        tail_gain = anchor_tail - tail_mae
        global_excess = max(0.0, global_delta - self.global_tolerance)
        harm_excess = max(0.0, harm_rate - self.max_harm_rate)
        objective = (
            0.80 * tail_mae
            + 0.20 * global_mae
            + 3.0 * global_excess
            + 0.20 * harm_excess
        )

        gate_target = exact_tail_mask(labels, role_name)
        gate_prediction = collected["applicability_prob"].view(-1) >= 0.5
        gate_accuracy = float((gate_prediction == gate_target).float().mean().item())
        mechanism_target = residual_mechanism_index(anchor, labels)
        mechanism_prediction = collected["mechanism_probs"].argmax(dim=1)
        mechanism_accuracy = float(
            (mechanism_prediction == mechanism_target).float().mean().item()
        )
        eligible = bool(
            tail_gain > 0.0
            and global_delta <= self.global_tolerance
            and harm_rate <= self.max_harm_rate
        )
        return {
            "objective": objective,
            "eligible": eligible,
            "global_mae": global_mae,
            "tail_mae": tail_mae,
            "anchor_global_mae": anchor_global,
            "anchor_tail_mae": anchor_tail,
            "tail_gain": tail_gain,
            "global_delta": global_delta,
            "harm_over_010_rate": harm_rate,
            "gate_accuracy": gate_accuracy,
            "mechanism_accuracy": mechanism_accuracy,
            "mean_abs_correction": float(
                torch.abs(collected["correction"]).mean().item()
            ),
            "max_anchor_diff": float(collected["max_anchor_diff"]),
        }

    def train_candidate(self, role_name: str, dataloaders):
        candidate_dir = self.save_dir / role_name
        candidate_dir.mkdir(parents=True, exist_ok=True)

        role_seed = int(getattr(self.args, "seed", 0))
        torch.manual_seed(role_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(role_seed)
        model = self._new_model()
        optimizer = self._optimizer(model)
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
        )

        initial_valid = self.collect(model, dataloaders["valid"], "valid")
        initial_row = self._validation_row(initial_valid, role_name)
        best = {
            "objective": float(initial_row["objective"]),
            "epoch": 0,
            "state": _cpu_state_dict(model),
            "valid": dict(initial_row),
            "anchor_fallback": True,
        }
        history: List[Dict[str, object]] = [
            {"epoch": 0, "stage": "anchor", **{"valid_" + k: v for k, v in initial_row.items()}}
        ]
        best_training_objective = float("inf")
        no_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            sampler = getattr(dataloaders["train"], "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            train_row = self._train_epoch(
                model, dataloaders["train"], optimizer, role_name
            )
            valid = self.collect(model, dataloaders["valid"], "valid")
            valid_row = self._validation_row(valid, role_name)
            scheduler.step(valid_row["objective"])
            history.append(
                {
                    "epoch": epoch,
                    "stage": "residual",
                    **{"train_" + k: v for k, v in train_row.items()},
                    **{"valid_" + k: v for k, v in valid_row.items()},
                }
            )
            logger.info(
                "V9.1 candidate=%s epoch=%d objective=%.6f tail_mae=%.6f "
                "tail_gain=%+.6f global_delta=%+.6f harm=%.4f eligible=%s",
                role_name,
                epoch,
                valid_row["objective"],
                valid_row["tail_mae"],
                valid_row["tail_gain"],
                valid_row["global_delta"],
                valid_row["harm_over_010_rate"],
                valid_row["eligible"],
            )

            if valid_row["objective"] < best_training_objective - 1e-6:
                best_training_objective = float(valid_row["objective"])
                no_improvement = 0
            else:
                no_improvement += 1

            if valid_row["eligible"] and (
                best["anchor_fallback"]
                or valid_row["objective"] < best["objective"] - 1e-6
            ):
                best = {
                    "objective": float(valid_row["objective"]),
                    "epoch": int(epoch),
                    "state": _cpu_state_dict(model),
                    "valid": dict(valid_row),
                    "anchor_fallback": False,
                }
            if no_improvement >= self.early_stop:
                break

        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        test = self.collect(model, dataloaders["test"], "test")
        pd.DataFrame(history).to_csv(
            candidate_dir / "training_history.csv", index=False
        )
        torch.save(
            {
                "method": "tail_residual_expert_v9_1",
                "role": role_name,
                "anchor_checkpoint": str(self.teacher_paths[self.anchor_index]),
                "anchor_index": self.anchor_index,
                "selected_epoch": int(best["epoch"]),
                "anchor_fallback": bool(best["anchor_fallback"]),
                "selected_valid": dict(best["valid"]),
                "teacher_policy": dict(self.teacher_policies[role_name]),
                "state_dict": best["state"],
            },
            candidate_dir / "tail_residual_expert_v91_best.pth",
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return {
            "role_name": role_name,
            "best": best,
            "valid": valid,
            "test": test,
        }

    def _capability_rows(self, results, split_name: str):
        labels = results[0][split_name]["labels"]
        anchor = results[0][split_name]["fixed_anchor"]
        rows = [{"expert": "anchor", **tail_capability_metrics(anchor, labels)}]
        for result in results:
            rows.append(
                {
                    "expert": result["role_name"],
                    **tail_capability_metrics(result[split_name]["prediction"], labels),
                }
            )
        return rows

    def _select_tail_policy(self, valid_rows):
        anchor_row = next(row for row in valid_rows if row["expert"] == "anchor")
        policy = {}
        for region_name in ("strong_negative", "strong_positive"):
            column = region_name + "_mae"
            eligible = [anchor_row]
            for row in valid_rows:
                if row["expert"] == "anchor":
                    continue
                if row["global_mae"] - anchor_row["global_mae"] <= self.global_tolerance:
                    eligible.append(row)
            best = min(eligible, key=lambda row: row[column])
            if best[column] >= anchor_row[column] - 1e-8:
                best = anchor_row
            policy[region_name] = {
                "expert": best["expert"],
                "valid_mae": float(best[column]),
                "anchor_valid_mae": float(anchor_row[column]),
                "valid_gain": float(anchor_row[column] - best[column]),
                "global_delta": float(best["global_mae"] - anchor_row["global_mae"]),
            }
        return policy

    @staticmethod
    def _result_by_name(results):
        return {result["role_name"]: result for result in results}

    def _prediction_for_expert(self, results_by_name, split_name, expert_name):
        if expert_name == "anchor":
            return next(iter(results_by_name.values()))[split_name]["fixed_anchor"]
        return results_by_name[expert_name][split_name]["prediction"]

    def _calibrate_anchor_gate(self, results, policy):
        by_name = self._result_by_name(results)
        anchor = results[0]["valid"]["fixed_anchor"]
        labels = results[0]["valid"]["labels"]
        negative = self._prediction_for_expert(
            by_name, "valid", policy["strong_negative"]["expert"]
        )
        positive = self._prediction_for_expert(
            by_name, "valid", policy["strong_positive"]["expert"]
        )
        rows = []
        best = None
        for negative_threshold in (-2.0, -1.5, -1.0, -0.5, 0.0):
            for positive_threshold in (0.0, 0.5, 1.0, 1.5, 2.0):
                for beta in (0.25, 0.50, 0.75, 1.0):
                    prediction = anchor.clone()
                    negative_mask = anchor.view(-1) <= float(negative_threshold)
                    positive_mask = anchor.view(-1) >= float(positive_threshold)
                    prediction[negative_mask] = (
                        (1.0 - beta) * anchor[negative_mask]
                        + beta * negative[negative_mask]
                    )
                    prediction[positive_mask] = (
                        (1.0 - beta) * anchor[positive_mask]
                        + beta * positive[positive_mask]
                    )
                    mae = float(torch.abs(prediction - labels).mean().item())
                    gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
                    harm = float((gain < -0.10).float().mean().item())
                    objective = mae + 0.03 * harm
                    row = {
                        "negative_threshold": negative_threshold,
                        "positive_threshold": positive_threshold,
                        "beta": beta,
                        "mae": mae,
                        "harm_over_010_rate": harm,
                        "objective": objective,
                    }
                    rows.append(row)
                    if best is None or (objective, mae) < (best["objective"], best["mae"]):
                        best = row
        if best is None:
            raise RuntimeError("V9.1 anchor gate calibration produced no candidates.")
        return best, rows

    def _apply_anchor_gate(self, results, split_name, policy, gate_policy):
        by_name = self._result_by_name(results)
        anchor = results[0][split_name]["fixed_anchor"]
        negative = self._prediction_for_expert(
            by_name, split_name, policy["strong_negative"]["expert"]
        )
        positive = self._prediction_for_expert(
            by_name, split_name, policy["strong_positive"]["expert"]
        )
        beta = float(gate_policy["beta"])
        prediction = anchor.clone()
        negative_mask = anchor.view(-1) <= float(gate_policy["negative_threshold"])
        positive_mask = anchor.view(-1) >= float(gate_policy["positive_threshold"])
        prediction[negative_mask] = (
            (1.0 - beta) * anchor[negative_mask] + beta * negative[negative_mask]
        )
        prediction[positive_mask] = (
            (1.0 - beta) * anchor[positive_mask] + beta * positive[positive_mask]
        )
        return prediction

    def train_all(self, dataloaders):
        results = [
            self.train_candidate(role_name, dataloaders)
            for role_name in TAIL_ROLE_NAMES
        ]
        valid_rows = self._capability_rows(results, "valid")
        test_rows = self._capability_rows(results, "test")
        pd.DataFrame(valid_rows).to_csv(
            self.save_dir / "v91_valid_tail_capability_matrix.csv", index=False
        )
        pd.DataFrame(test_rows).to_csv(
            self.save_dir / "v91_test_tail_capability_matrix.csv", index=False
        )

        policy = self._select_tail_policy(valid_rows)
        gate_policy, gate_rows = self._calibrate_anchor_gate(results, policy)
        pd.DataFrame(gate_rows).to_csv(
            self.save_dir / "v91_anchor_gate_calibration.csv", index=False
        )
        return self._evaluate_and_save(results, policy, gate_policy)

    def _evaluate_and_save(self, results, policy, gate_policy):
        by_name = self._result_by_name(results)
        labels = results[0]["test"]["labels"]
        anchor = results[0]["test"]["fixed_anchor"]
        deployable_gate = self._apply_anchor_gate(
            results, "test", policy, gate_policy
        )

        true_region_policy = anchor.clone()
        negative_mask = _candidate_mask_for_region(labels, "strong_negative")
        positive_mask = _candidate_mask_for_region(labels, "strong_positive")
        negative_prediction = self._prediction_for_expert(
            by_name, "test", policy["strong_negative"]["expert"]
        )
        positive_prediction = self._prediction_for_expert(
            by_name, "test", policy["strong_positive"]["expert"]
        )
        true_region_policy[negative_mask] = negative_prediction[negative_mask]
        true_region_policy[positive_mask] = positive_prediction[positive_mask]

        all_predictions = [anchor] + [
            result["test"]["prediction"] for result in results
        ]
        stacked = torch.stack(all_predictions, dim=1)
        sample_errors = torch.abs(stacked - labels.unsqueeze(1))
        best_indices = sample_errors.squeeze(-1).argmin(dim=1)
        sample_oracle = stacked[torch.arange(stacked.size(0)), best_indices]

        named_predictions = {
            "anchor": anchor,
            **{
                result["role_name"]: result["test"]["prediction"]
                for result in results
            },
            "anchor_score_gate_valid_selected": deployable_gate,
            "true_region_valid_selected_tail_policy": true_region_policy,
            "sample_oracle_upper_bound": sample_oracle,
        }
        rows = []
        payload = {}
        for name, prediction in named_predictions.items():
            metrics = _safe_metrics(self.metrics_fn, prediction, labels)
            metrics.update(tail_capability_metrics(prediction, labels))
            rows.append({"model": name, **metrics})
            payload[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v91_test_tail_summary.csv", index=False
        )

        frame: Dict[str, object] = {
            "sample_id": results[0]["test"]["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": anchor.view(-1).tolist(),
            "anchor_score_gate": deployable_gate.view(-1).tolist(),
            "true_region_tail_policy": true_region_policy.view(-1).tolist(),
            "sample_oracle_upper_bound": sample_oracle.view(-1).tolist(),
        }
        for result in results:
            name = result["role_name"]
            test = result["test"]
            frame[name + "_prediction"] = test["prediction"].view(-1).tolist()
            frame[name + "_correction"] = test["correction"].view(-1).tolist()
            frame[name + "_applicability"] = (
                test["applicability_prob"].view(-1).tolist()
            )
            for index, mechanism_name in enumerate(MECHANISM_NAMES):
                frame[name + "_p_" + mechanism_name] = (
                    test["mechanism_probs"][:, index].tolist()
                )
        pd.DataFrame(frame).to_csv(
            self.save_dir / "tail_residual_experts_v91_predictions.csv",
            index=False,
        )

        summary = {
            "method": "bidirectional_tail_residual_experts_v9_1",
            "dataset": str(self.args.dataset_name),
            "seed": int(getattr(self.args, "seed", 0)),
            "anchor_checkpoint": str(self.teacher_paths[self.anchor_index]),
            "anchor_index": self.anchor_index,
            "teacher_paths": [str(value) for value in self.teacher_paths],
            "teacher_policies": self.teacher_policies,
            "candidates": {
                result["role_name"]: {
                    "selected_epoch": int(result["best"]["epoch"]),
                    "anchor_fallback": bool(result["best"]["anchor_fallback"]),
                    "selected_valid": dict(result["best"]["valid"]),
                }
                for result in results
            },
            "validation_selected_tail_policy": policy,
            "anchor_score_gate_policy": gate_policy,
            "test_results": payload,
            "loss_weights": self.loss_weights.__dict__,
            "selection_protocol": (
                "The anchor, teacher enablement, candidate checkpoints, tail-role "
                "assignment, anchor-score thresholds, and shrinkage beta are "
                "selected from train/validation only. Test labels are used only "
                "for final metrics and explicitly named oracle analyses."
            ),
        }
        (self.save_dir / "tail_residual_experts_v91_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info(
            "V9.1 TEST anchor=%.6f gate=%.6f true-region-tail=%.6f",
            payload["anchor"]["MAE"],
            payload["anchor_score_gate_valid_selected"]["MAE"],
            payload["true_region_valid_selected_tail_policy"]["MAE"],
        )
        return summary
