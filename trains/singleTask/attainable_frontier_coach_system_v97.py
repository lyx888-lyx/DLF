"""Cross-fitted attainable-frontier coach and pre-registered safe routing."""

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
from .model.AttainableFrontierCoachV97 import (
    ACTION_NAMES,
    SPECIALIST_NAMES,
    AttainableFrontierCoachV97,
    attainable_frontier_loss,
    attainable_frontier_targets,
    frontier_input_features,
    nearest_action_from_frontier,
    stack_action_predictions,
)
from .oof_group_splits_v92 import conversation_group_id


@dataclass(frozen=True)
class FrontierCoachConfigV97:
    hidden_dim: int = 96
    dropout: float = 0.15
    gain_max: float = 2.0
    folds: int = 5
    max_epochs: int = 45
    early_stop: int = 8
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    useful_margin: float = 0.02


POLICY_PROFILES = {
    "conservative": {
        "min_predicted_gain": 0.05,
        "min_useful_probability": 0.70,
        "min_nearest_margin": 0.05,
        "min_expert_confidence": 0.55,
        "beta": 0.50,
        "max_activation_rate": 0.20,
    },
    "balanced": {
        "min_predicted_gain": 0.025,
        "min_useful_probability": 0.60,
        "min_nearest_margin": 0.025,
        "min_expert_confidence": 0.45,
        "beta": 0.75,
        "max_activation_rate": 0.35,
    },
    "broad": {
        "min_predicted_gain": 0.01,
        "min_useful_probability": 0.55,
        "min_nearest_margin": 0.01,
        "min_expert_confidence": 0.00,
        "beta": 1.00,
        "max_activation_rate": 0.50,
    },
}


def _expert_tensor(pool, key: str) -> torch.Tensor:
    return torch.stack(
        [
            pool["experts"][name][key].float().view(-1, 1)
            for name in SPECIALIST_NAMES
        ],
        dim=1,
    )


def _pool_tensors(pool):
    expert_predictions = _expert_tensor(pool, "prediction")
    expert_confidences = _expert_tensor(pool, "confidence")
    actions = stack_action_predictions(pool["anchor"].float(), expert_predictions)
    features = frontier_input_features(
        pool["function_space"].float(), actions, expert_confidences
    )
    return features, actions, expert_confidences


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


class FrontierCoachCrossFitterV97:
    def __init__(self, args, save_dir, pool_path, config: FrontierCoachConfigV97):
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
            "expert_confidences",
        }
        if not required.issubset(self.pool):
            raise ValueError(
                f"V9.7 frontier pool missing {sorted(required-set(self.pool))}"
            )
        self.actions = stack_action_predictions(
            self.pool["anchor"].float(),
            self.pool["expert_predictions"].float(),
        )
        self.features = frontier_input_features(
            self.pool["function_space"].float(),
            self.actions,
            self.pool["expert_confidences"].float(),
        )
        self.labels = self.pool["labels"].float().view(-1, 1)

    def new_model(self):
        return AttainableFrontierCoachV97(
            input_dim=self.features.size(1),
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            gain_max=self.config.gain_max,
        ).to(self.args.device)

    def loader(self, indices, shuffle, seed):
        index = torch.as_tensor(indices, dtype=torch.long)
        generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
        return DataLoader(
            TensorDataset(
                self.features[index], self.actions[index], self.labels[index]
            ),
            batch_size=self.config.batch_size,
            shuffle=bool(shuffle),
            generator=generator,
            drop_last=False,
        )

    @torch.no_grad()
    def collect(self, model, features, actions):
        model.eval()
        buffers = {
            key: []
            for key in (
                "frontier_value",
                "predicted_gain",
                "useful_probability",
                "action_logits",
            )
        }
        for start in range(0, len(features), self.config.batch_size):
            output = model(
                features[start : start + self.config.batch_size].to(
                    self.args.device
                ),
                actions[start : start + self.config.batch_size].to(
                    self.args.device
                ),
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        result = {
            key: torch.cat(value, dim=0) for key, value in buffers.items()
        }
        result["useful_logit"] = torch.logit(
            result["useful_probability"].clamp(1e-6, 1.0 - 1e-6)
        )
        return result

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
            for features, actions, labels in loader:
                output = model(
                    features.to(self.args.device), actions.to(self.args.device)
                )
                losses = attainable_frontier_loss(
                    output,
                    actions.to(self.args.device),
                    labels.to(self.args.device),
                    useful_margin=self.config.useful_margin,
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
                self.features[valid_index],
                self.actions[valid_index],
            )
            valid_losses = attainable_frontier_loss(
                valid_output,
                self.actions[valid_index],
                self.labels[valid_index],
                useful_margin=self.config.useful_margin,
            )
            nearest = nearest_action_from_frontier(
                valid_output, self.actions[valid_index]
            )
            nearest_mae = float(
                torch.abs(
                    nearest["selected_value"] - self.labels[valid_index]
                )
                .mean()
                .item()
            )
            objective = float(valid_losses["total"].item()) + 0.20 * nearest_mae
            history.append(
                {
                    "epoch": epoch,
                    "valid_objective": objective,
                    "valid_nearest_action_mae": nearest_mae,
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
                    "nearest_action_mae": nearest_mae,
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.config.early_stop:
                break
        if best is None:
            raise RuntimeError("V9.7 frontier fold failed to select an epoch")
        model.load_state_dict(best["state"])
        return model, best, history

    def fit(self):
        specs, manifest = safe_group_folds(
            self.pool["sample_ids"],
            self.labels.view(-1).tolist(),
            self.config.folds,
            int(self.args.seed) + 97001,
        )
        manifest.to_csv(
            self.save_dir / "v97_frontier_group_manifest.csv", index=False
        )
        n = len(self.labels)
        buffers = {
            "frontier_value": torch.full((n, 1), float("nan")),
            "predicted_gain": torch.full((n, 1), float("nan")),
            "useful_probability": torch.full((n, 1), float("nan")),
            "action_logits": torch.full(
                (n, len(ACTION_NAMES)), float("nan")
            ),
        }
        best_epochs = []
        history_rows = []
        for spec in specs:
            seed = int(self.args.seed) + 11003 * (spec.outer_fold + 1)
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
                self.features[holdout],
                self.actions[holdout],
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
            raise FloatingPointError("V9.7 frontier coach OOF output incomplete")
        pd.DataFrame(history_rows).to_csv(
            self.save_dir / "v97_frontier_fold_history.csv", index=False
        )
        output = dict(buffers)
        output["useful_logit"] = torch.logit(
            output["useful_probability"].clamp(1e-6, 1.0 - 1e-6)
        )
        target = attainable_frontier_targets(self.actions, self.labels)
        nearest = nearest_action_from_frontier(output, self.actions)
        oof_frame = {
            "sample_id": self.pool["sample_ids"],
            "group_id": self.pool["group_ids"],
            "label": self.labels.view(-1).tolist(),
            "anchor": self.pool["anchor"].view(-1).tolist(),
            "oracle_action": [
                ACTION_NAMES[i] for i in target["oracle_index"].tolist()
            ],
            "oracle_value": target["oracle_value"].view(-1).tolist(),
            "oracle_gain": target["oracle_gain"].view(-1).tolist(),
            "oracle_cost_margin": target["cost_margin"].view(-1).tolist(),
            "predicted_frontier_value": output["frontier_value"]
            .view(-1)
            .tolist(),
            "predicted_gain": output["predicted_gain"].view(-1).tolist(),
            "useful_probability": output["useful_probability"]
            .view(-1)
            .tolist(),
            "nearest_action": [
                ACTION_NAMES[i] for i in nearest["selected_index"].tolist()
            ],
            "nearest_margin": nearest["nearest_margin"].view(-1).tolist(),
        }
        pd.DataFrame(oof_frame).to_csv(
            self.save_dir / "v97_frontier_coach_oof_predictions.csv",
            index=False,
        )

        full_epochs = max(1, int(round(median(best_epochs))))
        torch.manual_seed(int(self.args.seed) + 77003)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.args.seed) + 77003)
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        loader = self.loader(range(n), True, int(self.args.seed) + 77003)
        full_history = []
        for epoch in range(1, full_epochs + 1):
            model.train()
            totals: Dict[str, float] = {}
            for features, actions, labels in loader:
                batch_output = model(
                    features.to(self.args.device),
                    actions.to(self.args.device),
                )
                losses = attainable_frontier_loss(
                    batch_output,
                    actions.to(self.args.device),
                    labels.to(self.args.device),
                    useful_margin=self.config.useful_margin,
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
            self.save_dir / "v97_frontier_full_history.csv", index=False
        )
        checkpoint = self.save_dir / "attainable_frontier_coach_v97.pth"
        torch.save(
            {
                "method": "attainable_frontier_coach_v9_7",
                "feature_dim": int(self.features.size(1)),
                "config": self.config.__dict__,
                "crossfit_best_epochs": best_epochs,
                "full_train_epochs": full_epochs,
                "state_dict": cpu_state(model),
            },
            checkpoint,
        )
        return model, output, {
            "checkpoint": str(checkpoint),
            "crossfit_best_epochs": best_epochs,
            "full_train_epochs": full_epochs,
            "oof_frontier_value_mae": float(
                torch.abs(
                    output["frontier_value"] - target["oracle_value"]
                )
                .mean()
                .item()
            ),
            "oof_predicted_gain_mae": float(
                torch.abs(output["predicted_gain"] - target["oracle_gain"])
                .mean()
                .item()
            ),
            "oof_nearest_action_accuracy": float(
                (
                    nearest["selected_index"] == target["oracle_index"]
                )
                .float()
                .mean()
                .item()
            ),
            "oof_nearest_action_mae": float(
                torch.abs(nearest["selected_value"] - self.labels)
                .mean()
                .item()
            ),
            "oof_anchor_mae": float(
                torch.abs(self.pool["anchor"] - self.labels).mean().item()
            ),
        }


class AttainableFrontierTrainerV97:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        frontier_pool_path,
        valid_pool,
        test_pool,
        coach_config: FrontierCoachConfigV97,
        minimum_oof_gain=0.002,
        minimum_validation_gain=0.0015,
        bootstrap_quantile=0.20,
        bootstrap_repeats=500,
        maximum_harm_rate=0.03,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.valid_pool = valid_pool
        self.test_pool = test_pool
        self.minimum_oof_gain = float(minimum_oof_gain)
        self.minimum_validation_gain = float(minimum_validation_gain)
        self.bootstrap_quantile = float(bootstrap_quantile)
        self.bootstrap_repeats = int(bootstrap_repeats)
        self.maximum_harm_rate = float(maximum_harm_rate)
        self.crossfitter = FrontierCoachCrossFitterV97(
            args, self.save_dir, frontier_pool_path, coach_config
        )
        self._bootstrap_cache = {}

    def _bootstrap_lower(
        self, anchor_error, prediction_error, sample_ids, salt
    ):
        groups = np.asarray(
            [conversation_group_id(value) for value in sample_ids]
        )
        unique = np.unique(groups)
        gains = np.asarray(
            [
                float(
                    anchor_error[groups == group].mean()
                    - prediction_error[groups == group].mean()
                )
                for group in unique
            ],
            dtype=np.float64,
        )
        key = (len(gains), int(salt))
        if key not in self._bootstrap_cache:
            rng = np.random.default_rng(
                int(self.args.seed) + 99001 + int(salt)
            )
            self._bootstrap_cache[key] = rng.integers(
                0,
                len(gains),
                size=(self.bootstrap_repeats, len(gains)),
            )
        values = gains[self._bootstrap_cache[key]].mean(axis=1)
        return float(np.quantile(values, self.bootstrap_quantile))

    @staticmethod
    def _apply_profile(actions, confidences, output, profile):
        nearest = nearest_action_from_frontier(output, actions)
        selected = nearest["selected_index"].view(-1)
        action_values = actions.squeeze(-1)
        selected_values = action_values.gather(
            1, selected.view(-1, 1)
        ).view(-1)
        anchor = action_values[:, 0]
        confidence_matrix = torch.cat(
            (torch.ones(len(actions), 1), confidences.squeeze(-1)), dim=1
        )
        selected_confidence = confidence_matrix.gather(
            1, selected.view(-1, 1)
        ).view(-1)
        activate = selected > 0
        activate &= output["predicted_gain"].view(-1) >= float(
            profile["min_predicted_gain"]
        )
        activate &= output["useful_probability"].view(-1) >= float(
            profile["min_useful_probability"]
        )
        activate &= nearest["nearest_margin"].view(-1) >= float(
            profile["min_nearest_margin"]
        )
        activate &= selected_confidence >= float(
            profile["min_expert_confidence"]
        )
        toward_frontier = (
            (selected_values - anchor)
            * (output["frontier_value"].view(-1) - anchor)
            > 0
        )
        activate &= toward_frontier
        beta = float(profile["beta"])
        prediction = anchor + beta * (
            selected_values - anchor
        ) * activate.float()
        deployed = torch.where(
            activate, selected, torch.zeros_like(selected)
        )
        return {
            "prediction": prediction.view(-1, 1),
            "activated": activate,
            "proposed_action": selected,
            "deployed_action": deployed,
            "selected_confidence": selected_confidence,
            "nearest_margin": nearest["nearest_margin"].view(-1),
        }

    def _profile_rows(self, output):
        pool = self.crossfitter.pool
        actions = self.crossfitter.actions
        confidences = pool["expert_confidences"].float()
        labels = pool["labels"].float().view(-1)
        anchor = pool["anchor"].float().view(-1)
        anchor_error = torch.abs(anchor - labels).numpy()
        rows = []
        for salt, (name, profile) in enumerate(
            POLICY_PROFILES.items(), start=1
        ):
            route = self._apply_profile(
                actions, confidences, output, profile
            )
            prediction = route["prediction"].view(-1)
            error = torch.abs(prediction - labels).numpy()
            gain_values = anchor_error - error
            fold_gains = []
            fold_values = pool["fold_index"].numpy()
            for fold in sorted(pool["fold_index"].unique().tolist()):
                mask = fold_values == int(fold)
                fold_gains.append(float(gain_values[mask].mean()))
            activation = float(
                route["activated"].float().mean().item()
            )
            gain = float(anchor_error.mean() - error.mean())
            lower = self._bootstrap_lower(
                anchor_error, error, pool["sample_ids"], salt
            )
            harm = float(np.mean(gain_values < -0.10))
            positive_folds = int(sum(value > 0 for value in fold_gains))
            eligible = (
                gain >= self.minimum_oof_gain
                and lower >= 0.0
                and harm <= self.maximum_harm_rate
                and 0.0
                < activation
                <= float(profile["max_activation_rate"])
                and positive_folds >= max(2, len(fold_gains) - 1)
            )
            rows.append(
                {
                    "profile": name,
                    **profile,
                    "oof_mae": float(error.mean()),
                    "oof_gain": gain,
                    "bootstrap_gain_lower": lower,
                    "harm_over_010_rate": harm,
                    "activation_rate": activation,
                    "positive_outer_folds": positive_folds,
                    "outer_fold_gains": json.dumps(fold_gains),
                    "oof_eligible": bool(eligible),
                }
            )
        frame = pd.DataFrame(rows)
        frame.to_csv(
            self.save_dir / "v97_oof_policy_profiles.csv", index=False
        )
        return rows

    def _apply_model(self, model, pool):
        features, actions, confidences = _pool_tensors(pool)
        output = self.crossfitter.collect(model, features, actions)
        return output, actions, confidences

    def _validation_select(self, model, profile_rows):
        output, actions, confidences = self._apply_model(
            model, self.valid_pool
        )
        labels = self.valid_pool["labels"].float().view(-1)
        anchor = self.valid_pool["anchor"].float().view(-1)
        anchor_error = torch.abs(anchor - labels).numpy()
        rows = [
            {
                "profile": "anchor",
                "valid_mae": float(anchor_error.mean()),
                "valid_gain": 0.0,
                "harm_over_010_rate": 0.0,
                "activation_rate": 0.0,
                "eligible": True,
            }
        ]
        payload = {"anchor": None}
        for row in profile_rows:
            if not bool(row["oof_eligible"]):
                continue
            profile = {
                key: row[key]
                for key in POLICY_PROFILES[row["profile"]]
            }
            route = self._apply_profile(
                actions, confidences, output, profile
            )
            error = torch.abs(
                route["prediction"].view(-1) - labels
            ).numpy()
            gain_values = anchor_error - error
            gain = float(anchor_error.mean() - error.mean())
            harm = float(np.mean(gain_values < -0.10))
            activation = float(
                route["activated"].float().mean().item()
            )
            eligible = (
                gain >= self.minimum_validation_gain
                and harm <= self.maximum_harm_rate
                and activation <= float(profile["max_activation_rate"])
            )
            rows.append(
                {
                    "profile": row["profile"],
                    "valid_mae": float(error.mean()),
                    "valid_gain": gain,
                    "harm_over_010_rate": harm,
                    "activation_rate": activation,
                    "eligible": bool(eligible),
                }
            )
            payload[row["profile"]] = profile
        frame = pd.DataFrame(rows)
        frame.to_csv(
            self.save_dir / "v97_validation_profile_selection.csv",
            index=False,
        )
        eligible = frame[frame["eligible"] == True].sort_values(  # noqa: E712
            ["valid_mae", "activation_rate"], kind="mergesort"
        )
        selected_name = (
            "anchor" if eligible.empty else str(eligible.iloc[0]["profile"])
        )
        return selected_name, payload.get(selected_name), output, rows

    @staticmethod
    def _true_region(pool):
        labels = pool["labels"].float().view(-1)
        regions = _region_index(labels)
        prediction = pool["anchor"].float().view(-1).clone()
        mapping = {
            0: "strong_negative",
            2: "boundary",
            3: "positive",
            4: "strong_positive",
        }
        for region, name in mapping.items():
            mask = regions == region
            prediction[mask] = (
                pool["experts"][name]["prediction"].view(-1)[mask]
            )
        return prediction.view(-1, 1)

    @staticmethod
    def _oracle(pool):
        _, actions, _ = _pool_tensors(pool)
        labels = pool["labels"].float().view(-1, 1, 1)
        indices = torch.abs(actions - labels).squeeze(-1).argmin(dim=1)
        return actions[torch.arange(len(actions)), indices]

    def train_all(self):
        model, oof_output, coach_summary = self.crossfitter.fit()
        profile_rows = self._profile_rows(oof_output)
        selected_name, selected_profile, _, valid_rows = (
            self._validation_select(model, profile_rows)
        )
        test_output, test_actions, test_confidences = self._apply_model(
            model, self.test_pool
        )
        nearest_all = nearest_action_from_frontier(
            test_output, test_actions
        )
        if selected_name == "anchor":
            n = len(test_actions)
            deploy = {
                "prediction": self.test_pool["anchor"]
                .float()
                .view(-1, 1),
                "activated": torch.zeros(n, dtype=torch.bool),
                "proposed_action": nearest_all["selected_index"],
                "deployed_action": torch.zeros(n, dtype=torch.long),
                "selected_confidence": torch.ones(n),
                "nearest_margin": nearest_all["nearest_margin"].view(-1),
            }
        else:
            deploy = self._apply_profile(
                test_actions,
                test_confidences,
                test_output,
                selected_profile,
            )
        labels = self.test_pool["labels"].float().view(-1, 1)
        named = {
            "anchor": self.test_pool["anchor"].float().view(-1, 1),
            "frontier_value_direct": test_output["frontier_value"].view(
                -1, 1
            ),
            "nearest_candidate_to_frontier_all": nearest_all[
                "selected_value"
            ].view(-1, 1),
            "attainable_frontier_valid_selected": deploy["prediction"],
            "true_region_expert_policy": self._true_region(self.test_pool),
            "sample_oracle_upper_bound": self._oracle(self.test_pool),
        }
        for name in SPECIALIST_NAMES:
            named[name] = (
                self.test_pool["experts"][name]["prediction"]
                .float()
                .view(-1, 1)
            )
        rows, results = [], {}
        for name, prediction in named.items():
            metrics = safe_metrics(self.metrics_fn, prediction, labels)
            rows.append({"model": name, **metrics})
            results[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v97_test_summary.csv", index=False
        )

        action_values = test_actions.squeeze(-1)
        frame = {
            "sample_id": self.test_pool["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": named["anchor"].view(-1).tolist(),
            "predicted_frontier_value": test_output["frontier_value"]
            .view(-1)
            .tolist(),
            "predicted_gain": test_output["predicted_gain"]
            .view(-1)
            .tolist(),
            "useful_probability": test_output["useful_probability"]
            .view(-1)
            .tolist(),
            "nearest_margin": deploy["nearest_margin"].view(-1).tolist(),
            "activated": deploy["activated"].tolist(),
            "proposed_action": [
                ACTION_NAMES[index]
                for index in deploy["proposed_action"].tolist()
            ],
            "deployed_action": [
                ACTION_NAMES[index]
                for index in deploy["deployed_action"].tolist()
            ],
            "attainable_frontier_prediction": deploy["prediction"]
            .view(-1)
            .tolist(),
            "selected_confidence": deploy["selected_confidence"]
            .view(-1)
            .tolist(),
        }
        for index, name in enumerate(ACTION_NAMES):
            frame[f"{name}_prediction"] = action_values[:, index].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "attainable_frontier_coach_v97_predictions.csv",
            index=False,
        )
        summary = {
            "method": "attainable_frontier_coach_v9_7",
            "dataset": str(self.args.dataset_name),
            "seed": int(self.args.seed),
            "action_schema": ACTION_NAMES,
            "frontier_pool": {
                "path": str(self.crossfitter.pool_path),
                "version": self.crossfitter.pool.get("version"),
                "provenance": self.crossfitter.pool.get("provenance"),
            },
            "frontier_coach": coach_summary,
            "oof_policy_profiles": profile_rows,
            "validation_profile_rows": valid_rows,
            "selected_profile": selected_name,
            "selected_policy": selected_profile,
            "test_activation_rate": float(
                deploy["activated"].float().mean().item()
            ),
            "test_proposed_action_counts": _action_counts(
                deploy["proposed_action"]
            ),
            "test_deployed_action_counts": _action_counts(
                deploy["deployed_action"]
            ),
            "test_results": results,
            "selection_protocol": (
                "Candidate specialists are group-cross-fitted on Train with the "
                "original V9.3 role/tail architectures. The coach receives every "
                "candidate prediction and learns the best attainable candidate "
                "value, oracle gain, useful-call probability, and auxiliary action. "
                "Only three pre-registered OOF-screened policies are compared on "
                "Validation, with an explicit Anchor fallback."
            ),
        }
        (
            self.save_dir / "attainable_frontier_coach_v97_summary.json"
        ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "V9.7 TEST anchor=%.6f frontier=%.6f coach=%.6f "
            "true-region=%.6f oracle=%.6f"
            % (
                results["anchor"]["MAE"],
                results["frontier_value_direct"]["MAE"],
                results["attainable_frontier_valid_selected"]["MAE"],
                results["true_region_expert_policy"]["MAE"],
                results["sample_oracle_upper_bound"]["MAE"],
            )
        )
        return summary
