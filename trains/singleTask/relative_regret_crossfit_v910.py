"""Cross-fitted relative-regret coaching and fold-ensemble inference for V9.10."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .coach_utils_v93 import cpu_state, safe_group_folds
from .model.RelativeRegretCoachV910 import (
    REGRET_VERSION,
    RelativeRegretCoachV910,
    relative_regret_loss,
    relative_regret_targets,
    select_action_from_regret,
)
from .model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_DIM,
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    SPECIALIST_NAMES,
    global_context_features,
    stack_action_predictions,
    stack_action_signatures,
)


@dataclass(frozen=True)
class RelativeRegretCoachConfigV910:
    hidden_dim: int = 128
    action_embedding_dim: int = 12
    dropout: float = 0.15
    minimum_scale: float = 0.01
    regret_max: float = 1.75
    folds: int = 5
    max_epochs: int = 50
    early_stop: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    sign_margin: float = 0.02
    rank_margin: float = 0.02
    rank_temperature: float = 0.05
    selection_temperature: float = 0.08


POLICY_PROFILES = {
    "conservative": {
        "risk_aversion": 0.75,
        "confidence_z": 0.75,
        "min_predicted_gain": 0.040,
        "min_lower_gain": 0.005,
        "min_beat_probability": 0.70,
        "min_regret_margin": 0.020,
        "max_selected_scale": 0.20,
        "min_expert_confidence": 0.55,
        "beta": 0.50,
        "max_activation_rate": 0.20,
    },
    "balanced": {
        "risk_aversion": 0.50,
        "confidence_z": 0.50,
        "min_predicted_gain": 0.025,
        "min_lower_gain": 0.000,
        "min_beat_probability": 0.62,
        "min_regret_margin": 0.010,
        "max_selected_scale": 0.30,
        "min_expert_confidence": 0.40,
        "beta": 0.75,
        "max_activation_rate": 0.35,
    },
    "broad": {
        "risk_aversion": 0.25,
        "confidence_z": 0.25,
        "min_predicted_gain": 0.010,
        "min_lower_gain": -0.005,
        "min_beat_probability": 0.55,
        "min_regret_margin": 0.004,
        "max_selected_scale": 0.45,
        "min_expert_confidence": 0.00,
        "beta": 1.00,
        "max_activation_rate": 0.50,
    },
}


def _expert_tensor(pool, key: str) -> torch.Tensor:
    return torch.stack(
        [pool["experts"][name][key].float() for name in SPECIALIST_NAMES],
        dim=1,
    )


def pool_tensors(pool):
    predictions = _expert_tensor(pool, "prediction")
    signatures = _expert_tensor(pool, "signature")
    actions = stack_action_predictions(pool["anchor"].float(), predictions)
    full_signatures = stack_action_signatures(pool["anchor"].float(), signatures)
    context = global_context_features(pool["function_space"].float(), actions)
    return context, actions, full_signatures[:, 1:]


def _action_counts(indices: torch.Tensor) -> Dict[str, int]:
    return {
        name: int((indices.view(-1) == index).sum().item())
        for index, name in enumerate(ACTION_NAMES)
    }


def _spearman(actual: torch.Tensor, predicted: torch.Tensor) -> float:
    return float(
        pd.Series(actual.detach().cpu().numpy()).corr(
            pd.Series(predicted.detach().cpu().numpy()), method="spearman"
        )
    )


class RelativeRegretCoachCrossFitterV910:
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
                f"V9.10 semantic pool missing {sorted(required - set(self.pool))}"
            )
        if self.pool.get("signature_version") != SIGNATURE_VERSION:
            raise ValueError("V9.10 semantic signature version mismatch")
        if tuple(self.pool.get("signature_fields", ())) != SIGNATURE_FIELDS:
            raise ValueError("V9.10 semantic signature field mismatch")
        self.context, self.actions, self.signatures = pool_tensors(self.pool)
        self.labels = self.pool["labels"].float().view(-1, 1)
        self.ensemble_checkpoints: List[Path] = []

    def new_model(self):
        return RelativeRegretCoachV910(
            context_dim=self.context.size(1),
            signature_dim=SIGNATURE_DIM,
            hidden_dim=self.config.hidden_dim,
            action_embedding_dim=self.config.action_embedding_dim,
            dropout=self.config.dropout,
            minimum_scale=self.config.minimum_scale,
            regret_max=self.config.regret_max,
        ).to(self.args.device)

    def loader(self, indices: Iterable[int], shuffle: bool, seed: int):
        index = torch.as_tensor(list(indices), dtype=torch.long)
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
        buffers = {
            "predicted_delta": [],
            "predicted_scale": [],
            "beat_logit": [],
            "beat_probability": [],
        }
        for start in range(0, len(context), self.config.batch_size):
            output = model(
                context[start : start + self.config.batch_size].to(self.args.device),
                signatures[start : start + self.config.batch_size].to(
                    self.args.device
                ),
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        return {key: torch.cat(value, dim=0) for key, value in buffers.items()}

    def _loss(self, output, actions, labels):
        return relative_regret_loss(
            output,
            actions,
            labels,
            sign_margin=self.config.sign_margin,
            rank_margin=self.config.rank_margin,
            rank_temperature=self.config.rank_temperature,
            selection_temperature=self.config.selection_temperature,
        )

    def train_for_selection(self, train_indices, valid_indices, seed):
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
                    totals[key] = totals.get(key, 0.0) + float(value.detach().item())
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
            selected = select_action_from_regret(
                valid_output,
                self.actions[valid_index],
            )
            selected_mae = float(
                torch.abs(
                    selected["selected_value"] - self.labels[valid_index]
                ).mean().item()
            )
            target = relative_regret_targets(
                self.actions[valid_index], self.labels[valid_index]
            )
            beat_accuracy = float(
                (
                    (valid_output["beat_probability"] >= 0.5)
                    == target["beat_anchor"].bool()
                ).float().mean().item()
            )
            objective = (
                float(valid_losses["total"].item())
                + 0.35 * selected_mae
                + 0.10 * (1.0 - beat_accuracy)
            )
            history.append(
                {
                    "epoch": epoch,
                    "valid_objective": objective,
                    "valid_selected_mae": selected_mae,
                    "valid_beat_accuracy": beat_accuracy,
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
                    "selected_mae": selected_mae,
                    "beat_accuracy": beat_accuracy,
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.config.early_stop:
                break
        if best is None:
            raise RuntimeError("V9.10 relative-regret fold selected no epoch")
        return best, history

    def fit_fixed_epochs(self, indices, epochs, seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = self.loader(indices, True, seed)
        history = []
        for epoch in range(1, int(epochs) + 1):
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
                    totals[key] = totals.get(key, 0.0) + float(value.detach().item())
            history.append(
                {
                    "epoch": epoch,
                    **{
                        key: value / max(1, len(loader))
                        for key, value in totals.items()
                    },
                }
            )
        return model, history

    @torch.no_grad()
    def collect_ensemble(self, context, signatures):
        if not self.ensemble_checkpoints:
            raise RuntimeError("V9.10 ensemble checkpoints are unavailable")
        means, scales, probabilities = [], [], []
        for checkpoint in self.ensemble_checkpoints:
            payload = torch.load(checkpoint, map_location="cpu")
            model = self.new_model()
            model.load_state_dict(payload["state_dict"], strict=True)
            output = self.collect(model, context, signatures)
            means.append(output["predicted_delta"])
            scales.append(output["predicted_scale"])
            probabilities.append(output["beat_probability"])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        mean_stack = torch.stack(means, dim=0)
        scale_stack = torch.stack(scales, dim=0)
        probability_stack = torch.stack(probabilities, dim=0)
        predicted_delta = mean_stack.mean(dim=0)
        epistemic_var = mean_stack.var(dim=0, unbiased=False)
        aleatoric_var = scale_stack.square().mean(dim=0)
        total_scale = torch.sqrt((epistemic_var + aleatoric_var).clamp_min(1e-8))
        beat_probability = probability_stack.mean(dim=0).clamp(1e-6, 1.0 - 1e-6)
        return {
            "predicted_delta": predicted_delta,
            "predicted_scale": total_scale,
            "beat_probability": beat_probability,
            "beat_logit": torch.logit(beat_probability),
            "ensemble_std": torch.sqrt(epistemic_var.clamp_min(0.0)),
            "aleatoric_scale": torch.sqrt(aleatoric_var.clamp_min(0.0)),
        }

    def fit(self):
        specs, manifest = safe_group_folds(
            self.pool["sample_ids"],
            self.labels.view(-1).tolist(),
            self.config.folds,
            int(self.args.seed) + 910001,
        )
        manifest.to_csv(
            self.save_dir / "v910_relative_regret_group_manifest.csv", index=False
        )
        n = len(self.labels)
        buffers = {
            "predicted_delta": torch.full((n, 4), float("nan")),
            "predicted_scale": torch.full((n, 4), float("nan")),
            "beat_logit": torch.full((n, 4), float("nan")),
            "beat_probability": torch.full((n, 4), float("nan")),
        }
        best_epochs = []
        selection_history = []
        refit_history = []
        checkpoint_rows = []
        self.ensemble_checkpoints = []
        for spec in specs:
            fold = int(spec.outer_fold)
            selection_seed = int(self.args.seed) + 17011 * (fold + 1)
            best, history = self.train_for_selection(
                spec.inner_train_indices,
                spec.inner_valid_indices,
                selection_seed,
            )
            best_epochs.append(int(best["epoch"]))
            selection_history.extend(
                {"fold": fold, **row} for row in history
            )
            development = sorted(
                set(spec.inner_train_indices) | set(spec.inner_valid_indices)
            )
            refit_seed = int(self.args.seed) + 19001 * (fold + 1)
            model, fixed_history = self.fit_fixed_epochs(
                development,
                int(best["epoch"]),
                refit_seed,
            )
            refit_history.extend(
                {"fold": fold, **row} for row in fixed_history
            )
            holdout = torch.as_tensor(spec.outer_holdout_indices, dtype=torch.long)
            output = self.collect(
                model,
                self.context[holdout],
                self.signatures[holdout],
            )
            for key in buffers:
                buffers[key][holdout] = output[key]
            checkpoint = self.save_dir / f"relative_regret_fold_{fold}_v910.pth"
            torch.save(
                {
                    "method": "relative_regret_coach_v9_10",
                    "regret_version": REGRET_VERSION,
                    "signature_version": SIGNATURE_VERSION,
                    "signature_fields": list(SIGNATURE_FIELDS),
                    "context_dim": int(self.context.size(1)),
                    "config": self.config.__dict__,
                    "fold": fold,
                    "selected_epoch": int(best["epoch"]),
                    "development_count": len(development),
                    "holdout_count": len(holdout),
                    "state_dict": cpu_state(model),
                },
                checkpoint,
            )
            self.ensemble_checkpoints.append(checkpoint)
            checkpoint_rows.append(
                {
                    "fold": fold,
                    "checkpoint": str(checkpoint),
                    "selected_epoch": int(best["epoch"]),
                    "development_count": len(development),
                    "holdout_count": len(holdout),
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if any(not torch.isfinite(value).all() for value in buffers.values()):
            raise FloatingPointError("V9.10 relative-regret OOF output incomplete")
        pd.DataFrame(selection_history).to_csv(
            self.save_dir / "v910_relative_regret_selection_history.csv", index=False
        )
        pd.DataFrame(refit_history).to_csv(
            self.save_dir / "v910_relative_regret_refit_history.csv", index=False
        )
        pd.DataFrame(checkpoint_rows).to_csv(
            self.save_dir / "v910_relative_regret_ensemble.csv", index=False
        )

        output = dict(buffers)
        target = relative_regret_targets(self.actions, self.labels)
        selected = select_action_from_regret(output, self.actions)
        selected_index = selected["selected_index"]
        selected_actual_cost = target["costs"].gather(
            1, selected_index.view(-1, 1)
        )
        actual_selected_gain = target["anchor_cost"] - selected_actual_cost
        oracle_index = target["oracle_index"]

        frame = {
            "sample_id": self.pool["sample_ids"],
            "group_id": self.pool["group_ids"],
            "label": self.labels.view(-1).tolist(),
            "anchor": self.pool["anchor"].view(-1).tolist(),
            "oracle_action": [ACTION_NAMES[index] for index in oracle_index.tolist()],
            "selected_action": [
                ACTION_NAMES[index] for index in selected_index.tolist()
            ],
            "selected_value": selected["selected_value"].view(-1).tolist(),
            "predicted_selected_gain": selected["predicted_gain"].view(-1).tolist(),
            "actual_selected_gain": actual_selected_gain.view(-1).tolist(),
            "predicted_lower_gain": selected["lower_gain"].view(-1).tolist(),
            "predicted_regret_margin": selected["regret_margin"].view(-1).tolist(),
            "selected_scale": selected["selected_scale"].view(-1).tolist(),
            "selected_beat_probability": selected["beat_probability"].view(-1).tolist(),
        }
        for index, name in enumerate(SPECIALIST_NAMES):
            frame[f"{name}_actual_delta"] = target["delta"][:, index].tolist()
            frame[f"{name}_predicted_delta"] = output["predicted_delta"][:, index].tolist()
            frame[f"{name}_predicted_scale"] = output["predicted_scale"][:, index].tolist()
            frame[f"{name}_beat_probability"] = output["beat_probability"][:, index].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "v910_relative_regret_oof_predictions.csv", index=False
        )

        delta_error = torch.abs(output["predicted_delta"] - target["delta"])
        delta_mae = delta_error.mean(dim=0)
        beat_prediction = output["beat_probability"] >= 0.5
        beat_target = target["beat_anchor"].bool()
        beat_accuracy = (beat_prediction == beat_target).float().mean(dim=0)
        beat_brier = (
            output["beat_probability"] - target["beat_anchor"]
        ).square().mean(dim=0)
        delta_spearman = {
            name: _spearman(target["delta"][:, index], output["predicted_delta"][:, index])
            for index, name in enumerate(SPECIALIST_NAMES)
        }
        summary = {
            "method": "relative_regret_coach_v9_10",
            "regret_version": REGRET_VERSION,
            "ensemble_checkpoints": [str(path) for path in self.ensemble_checkpoints],
            "crossfit_best_epochs": best_epochs,
            "oof_per_expert_delta_mae": {
                name: float(delta_mae[index].item())
                for index, name in enumerate(SPECIALIST_NAMES)
            },
            "oof_per_expert_delta_spearman": delta_spearman,
            "oof_per_expert_beat_accuracy": {
                name: float(beat_accuracy[index].item())
                for index, name in enumerate(SPECIALIST_NAMES)
            },
            "oof_per_expert_beat_brier": {
                name: float(beat_brier[index].item())
                for index, name in enumerate(SPECIALIST_NAMES)
            },
            "oof_selected_action_accuracy": float(
                (selected_index == oracle_index).float().mean().item()
            ),
            "oof_regret_selected_mae": float(selected_actual_cost.mean().item()),
            "oof_anchor_mae": float(target["anchor_cost"].mean().item()),
            "oof_unrestricted_gain": float(
                target["anchor_cost"].mean().item() - selected_actual_cost.mean().item()
            ),
            "oof_selected_gain_mae": float(
                torch.abs(selected["predicted_gain"] - actual_selected_gain)
                .mean().item()
            ),
            "oof_selected_benefit_rate": float(
                (actual_selected_gain.view(-1) > 0.0).float().mean().item()
            ),
            "oof_selected_harm_over_010_rate": float(
                (actual_selected_gain.view(-1) < -0.10).float().mean().item()
            ),
            "oof_delta_scale_coverage_1x": float(
                (delta_error <= output["predicted_scale"]).float().mean().item()
            ),
            "oof_delta_scale_coverage_2x": float(
                (delta_error <= 2.0 * output["predicted_scale"])
                .float().mean().item()
            ),
            "oof_selected_action_counts": _action_counts(selected_index),
        }
        return output, summary
