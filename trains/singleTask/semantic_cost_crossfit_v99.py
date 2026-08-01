"""Cross-fitted semantic per-action cost coaching and safe deployment for V9.9."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .coach_utils_v93 import cpu_state, safe_group_folds, safe_metrics
from .model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_DIM,
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    SPECIALIST_NAMES,
    SemanticCostCoachV99,
    actual_action_costs,
    cost_soft_mixture,
    global_context_features,
    select_action_from_cost,
    semantic_cost_loss,
    stack_action_predictions,
    stack_action_signatures,
)
from .oof_group_splits_v92 import conversation_group_id


@dataclass(frozen=True)
class SemanticCostCoachConfigV99:
    hidden_dim: int = 128
    action_embedding_dim: int = 12
    dropout: float = 0.15
    minimum_scale: float = 0.01
    folds: int = 5
    max_epochs: int = 50
    early_stop: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    rank_margin: float = 0.02
    rank_temperature: float = 0.05
    selection_temperature: float = 0.08


POLICY_PROFILES = {
    "conservative": {
        "risk_aversion": 0.75,
        "confidence_z": 0.75,
        "min_predicted_gain": 0.050,
        "min_lower_gain": 0.005,
        "min_cost_margin": 0.030,
        "max_selected_scale": 0.150,
        "min_expert_confidence": 0.55,
        "beta": 0.50,
        "max_activation_rate": 0.20,
    },
    "balanced": {
        "risk_aversion": 0.50,
        "confidence_z": 0.50,
        "min_predicted_gain": 0.030,
        "min_lower_gain": 0.000,
        "min_cost_margin": 0.015,
        "max_selected_scale": 0.220,
        "min_expert_confidence": 0.40,
        "beta": 0.75,
        "max_activation_rate": 0.35,
    },
    "broad": {
        "risk_aversion": 0.25,
        "confidence_z": 0.25,
        "min_predicted_gain": 0.015,
        "min_lower_gain": -0.010,
        "min_cost_margin": 0.005,
        "max_selected_scale": 0.350,
        "min_expert_confidence": 0.00,
        "beta": 1.00,
        "max_activation_rate": 0.50,
    },
}


def _expert_tensor(pool, key: str) -> torch.Tensor:
    return torch.stack(
        [
            pool["experts"][name][key].float()
            for name in SPECIALIST_NAMES
        ],
        dim=1,
    )


def pool_tensors(pool):
    predictions = _expert_tensor(pool, "prediction")
    signatures = _expert_tensor(pool, "signature")
    actions = stack_action_predictions(pool["anchor"].float(), predictions)
    action_signatures = stack_action_signatures(
        pool["anchor"].float(), signatures
    )
    context = global_context_features(
        pool["function_space"].float(), actions
    )
    return context, actions, action_signatures


def _action_counts(indices: torch.Tensor) -> Dict[str, int]:
    return {
        name: int((indices.view(-1) == index).sum().item())
        for index, name in enumerate(ACTION_NAMES)
    }


def _region_index(values: torch.Tensor) -> torch.Tensor:
    values = values.view(-1)
    result = torch.full_like(values, 2, dtype=torch.long)
    result[values < -1.5] = 0
    result[(values >= -1.5) & (values < -0.5)] = 1
    result[(values > 0.5) & (values <= 1.5)] = 3
    result[values > 1.5] = 4
    return result


class SemanticCostCoachCrossFitterV99:
    def __init__(self, args, save_dir, pool_path, config):
        self.args = args
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pool_path = Path(pool_path)
        self.pool = torch.load(self.pool_path, map_location="cpu")
        self.config = config
        required = {
            "sample_ids",
            "group_ids",
            "labels",
            "anchor",
            "function_space",
            "fold_index",
            "expert_predictions",
            "expert_signatures",
            "signature_version",
            "signature_fields",
        }
        if not required.issubset(self.pool):
            raise ValueError(
                f"V9.9 semantic pool missing {sorted(required-set(self.pool))}"
            )
        if self.pool.get("signature_version") != SIGNATURE_VERSION:
            raise ValueError("V9.9 semantic signature version mismatch")
        if tuple(self.pool.get("signature_fields", ())) != SIGNATURE_FIELDS:
            raise ValueError("V9.9 semantic signature field mismatch")
        self.actions = stack_action_predictions(
            self.pool["anchor"].float(),
            self.pool["expert_predictions"].float(),
        )
        self.signatures = stack_action_signatures(
            self.pool["anchor"].float(),
            self.pool["expert_signatures"].float(),
        )
        self.context = global_context_features(
            self.pool["function_space"].float(), self.actions
        )
        self.labels = self.pool["labels"].float().view(-1, 1)

    def new_model(self):
        return SemanticCostCoachV99(
            context_dim=self.context.size(1),
            signature_dim=SIGNATURE_DIM,
            hidden_dim=self.config.hidden_dim,
            action_embedding_dim=self.config.action_embedding_dim,
            dropout=self.config.dropout,
            minimum_scale=self.config.minimum_scale,
        ).to(self.args.device)

    def loader(self, indices, shuffle, seed):
        index = torch.as_tensor(indices, dtype=torch.long)
        generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
        return DataLoader(
            TensorDataset(
                self.context[index],
                self.signatures[index],
                self.actions[index],
                self.labels[index],
            ),
            batch_size=self.config.batch_size,
            shuffle=bool(shuffle),
            generator=generator,
            drop_last=False,
        )

    @torch.no_grad()
    def collect(self, model, context, signatures):
        model.eval()
        buffers = {"predicted_cost": [], "predicted_scale": []}
        for start in range(0, len(context), self.config.batch_size):
            output = model(
                context[start : start + self.config.batch_size].to(
                    self.args.device
                ),
                signatures[start : start + self.config.batch_size].to(
                    self.args.device
                ),
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        return {key: torch.cat(value, dim=0) for key, value in buffers.items()}

    def _loss(self, output, actions, labels):
        return semantic_cost_loss(
            output,
            actions,
            labels,
            rank_margin=self.config.rank_margin,
            rank_temperature=self.config.rank_temperature,
            selection_temperature=self.config.selection_temperature,
        )

    def train_fold(self, train_indices, valid_indices, seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = self.loader(train_indices, True, seed)
        valid_index = torch.as_tensor(valid_indices, dtype=torch.long)
        best = None
        stale = 0
        history = []
        for epoch in range(1, self.config.max_epochs + 1):
            model.train()
            totals: Dict[str, float] = {}
            for context, signatures, actions, labels in loader:
                output = model(
                    context.to(self.args.device),
                    signatures.to(self.args.device),
                )
                losses = self._loss(
                    output,
                    actions.to(self.args.device),
                    labels.to(self.args.device),
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().item()
                    )
            valid_output = self.collect(
                model,
                self.context[valid_index],
                self.signatures[valid_index],
            )
            valid_losses = self._loss(
                valid_output,
                self.actions[valid_index],
                self.labels[valid_index],
            )
            selected = select_action_from_cost(
                valid_output, self.actions[valid_index]
            )
            nearest_mae = float(
                torch.abs(
                    selected["selected_value"] - self.labels[valid_index]
                )
                .mean()
                .item()
            )
            objective = float(valid_losses["total"].item()) + 0.30 * nearest_mae
            history.append(
                {
                    "epoch": epoch,
                    "valid_objective": objective,
                    "valid_cost_selected_mae": nearest_mae,
                    **{
                        f"train_{key}": value / max(1, len(loader))
                        for key, value in totals.items()
                    },
                    **{
                        f"valid_{key}": float(value.item())
                        for key, value in valid_losses.items()
                    },
                }
            )
            if best is None or objective < best["objective"] - 1e-6:
                best = {
                    "objective": objective,
                    "epoch": epoch,
                    "state": cpu_state(model),
                    "selected_mae": nearest_mae,
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.config.early_stop:
                break
        if best is None:
            raise RuntimeError("V9.9 semantic cost fold selected no epoch")
        model.load_state_dict(best["state"])
        return model, best, history

    def fit(self):
        specs, manifest = safe_group_folds(
            self.pool["sample_ids"],
            self.labels.view(-1).tolist(),
            self.config.folds,
            int(self.args.seed) + 99001,
        )
        manifest.to_csv(
            self.save_dir / "v99_semantic_cost_group_manifest.csv", index=False
        )
        n = len(self.labels)
        buffers = {
            "predicted_cost": torch.full((n, 5), float("nan")),
            "predicted_scale": torch.full((n, 5), float("nan")),
        }
        best_epochs = []
        history_rows = []
        for spec in specs:
            seed = int(self.args.seed) + 13001 * (spec.outer_fold + 1)
            model, best, history = self.train_fold(
                spec.inner_train_indices,
                spec.inner_valid_indices,
                seed,
            )
            holdout = torch.as_tensor(
                spec.outer_holdout_indices, dtype=torch.long
            )
            output = self.collect(
                model,
                self.context[holdout],
                self.signatures[holdout],
            )
            for key in buffers:
                buffers[key][holdout] = output[key]
            best_epochs.append(int(best["epoch"]))
            history_rows.extend(
                {"fold": spec.outer_fold, **row} for row in history
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if any(not torch.isfinite(value).all() for value in buffers.values()):
            raise FloatingPointError("V9.9 semantic cost OOF incomplete")
        pd.DataFrame(history_rows).to_csv(
            self.save_dir / "v99_semantic_cost_fold_history.csv", index=False
        )
        output = dict(buffers)
        target_cost = actual_action_costs(self.actions, self.labels)
        selected = select_action_from_cost(output, self.actions)
        selected_index = selected["selected_index"]
        oracle_index = target_cost.argmin(dim=1)
        selected_actual_cost = target_cost.gather(
            1, selected_index.view(-1, 1)
        )
        actual_selected_gain = target_cost[:, :1] - selected_actual_cost
        predicted_selected_gain = selected["predicted_gain"]

        oof_frame = {
            "sample_id": self.pool["sample_ids"],
            "group_id": self.pool["group_ids"],
            "label": self.labels.view(-1).tolist(),
            "anchor": self.pool["anchor"].view(-1).tolist(),
            "oracle_action": [ACTION_NAMES[i] for i in oracle_index.tolist()],
            "selected_action": [
                ACTION_NAMES[i] for i in selected_index.tolist()
            ],
            "selected_value": selected["selected_value"].view(-1).tolist(),
            "predicted_selected_gain": predicted_selected_gain
            .view(-1)
            .tolist(),
            "actual_selected_gain": actual_selected_gain.view(-1).tolist(),
            "predicted_lower_gain": selected["lower_gain"].view(-1).tolist(),
            "predicted_cost_margin": selected["cost_margin"].view(-1).tolist(),
            "selected_scale": selected["selected_scale"].view(-1).tolist(),
        }
        for index, name in enumerate(ACTION_NAMES):
            oof_frame[f"{name}_actual_cost"] = target_cost[:, index].tolist()
            oof_frame[f"{name}_predicted_cost"] = output[
                "predicted_cost"
            ][:, index].tolist()
            oof_frame[f"{name}_predicted_scale"] = output[
                "predicted_scale"
            ][:, index].tolist()
        pd.DataFrame(oof_frame).to_csv(
            self.save_dir / "v99_semantic_cost_oof_predictions.csv",
            index=False,
        )

        full_epochs = max(1, int(round(median(best_epochs))))
        torch.manual_seed(int(self.args.seed) + 99003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.args.seed) + 99003)
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = self.loader(range(n), True, int(self.args.seed) + 99003)
        full_history = []
        for epoch in range(1, full_epochs + 1):
            model.train()
            totals: Dict[str, float] = {}
            for context, signatures, actions, labels in loader:
                batch_output = model(
                    context.to(self.args.device),
                    signatures.to(self.args.device),
                )
                losses = self._loss(
                    batch_output,
                    actions.to(self.args.device),
                    labels.to(self.args.device),
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().item()
                    )
            full_history.append(
                {
                    "epoch": epoch,
                    **{
                        key: value / max(1, len(loader))
                        for key, value in totals.items()
                    },
                }
            )
        pd.DataFrame(full_history).to_csv(
            self.save_dir / "v99_semantic_cost_full_history.csv", index=False
        )
        checkpoint = self.save_dir / "semantic_cost_coach_v99.pth"
        torch.save(
            {
                "method": "semantic_signature_per_action_cost_coach_v9_9",
                "context_dim": int(self.context.size(1)),
                "signature_version": SIGNATURE_VERSION,
                "signature_fields": list(SIGNATURE_FIELDS),
                "config": self.config.__dict__,
                "crossfit_best_epochs": best_epochs,
                "full_train_epochs": full_epochs,
                "state_dict": cpu_state(model),
            },
            checkpoint,
        )
        cost_mae = torch.abs(output["predicted_cost"] - target_cost).mean(dim=0)
        scale_error = torch.abs(output["predicted_cost"] - target_cost)
        return model, output, {
            "checkpoint": str(checkpoint),
            "crossfit_best_epochs": best_epochs,
            "full_train_epochs": full_epochs,
            "oof_per_action_cost_mae": {
                name: float(cost_mae[index].item())
                for index, name in enumerate(ACTION_NAMES)
            },
            "oof_selected_action_accuracy": float(
                (selected_index == oracle_index).float().mean().item()
            ),
            "oof_cost_selected_mae": float(selected_actual_cost.mean().item()),
            "oof_anchor_mae": float(target_cost[:, 0].mean().item()),
            "oof_selected_gain_mae": float(
                torch.abs(
                    predicted_selected_gain - actual_selected_gain
                ).mean().item()
            ),
            "oof_scale_coverage_1x": float(
                (scale_error <= output["predicted_scale"])
                .float()
                .mean()
                .item()
            ),
            "oof_scale_coverage_2x": float(
                (scale_error <= 2.0 * output["predicted_scale"])
                .float()
                .mean()
                .item()
            ),
        }
