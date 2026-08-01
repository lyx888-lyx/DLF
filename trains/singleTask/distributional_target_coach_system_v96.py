"""Grouped OOF target-distribution training and safe nearest-expert routing for V9.6."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .coach_utils_v93 import cpu_state, safe_group_folds, safe_metrics
from .model.DistributionalTargetCoachV96 import (
    ACTION_NAMES,
    QUANTILE_LEVELS,
    SPECIALIST_NAMES,
    DistributionalTargetCoachV96,
    action_risk,
    coach_input_features,
    distributional_target_loss,
    stack_action_predictions,
    target_region_index,
)
from .oof_group_splits_v92 import conversation_group_id


class DistributionalTargetCrossFitterV96:
    def __init__(
        self,
        args,
        save_dir,
        oof_cache_path,
        hidden_dim=48,
        dropout=0.10,
        residual_max=1.50,
        folds=5,
        max_epochs=50,
        early_stop=8,
        learning_rate=2e-4,
        weight_decay=1e-3,
        batch_size=64,
    ):
        self.args = args
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.oof = torch.load(oof_cache_path, map_location="cpu")
        required = {
            "sample_ids", "group_ids", "labels", "oof_prediction", "oof_feature"
        }
        if not required.issubset(self.oof):
            raise ValueError(
                f"OOF cache missing keys: {sorted(required - set(self.oof))}"
            )
        if self.oof.get("feature_space") != "auxiliary_prediction_logits_v1":
            raise ValueError(
                "V9.6 requires the aligned V9.2 function-space OOF cache"
            )
        self.anchor = self.oof["oof_prediction"].float().view(-1, 1)
        self.labels = self.oof["labels"].float().view(-1, 1)
        self.features = coach_input_features(
            self.oof["oof_feature"].float(), self.anchor
        )
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual_max = float(residual_max)
        self.folds = int(folds)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)

    def new_model(self):
        return DistributionalTargetCoachV96(
            input_dim=self.features.size(1),
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            residual_max=self.residual_max,
        ).to(self.args.device)

    def loader(self, indices, shuffle, seed):
        index = torch.as_tensor(indices, dtype=torch.long)
        dataset = TensorDataset(
            self.features[index], self.anchor[index], self.labels[index]
        )
        generator = torch.Generator().manual_seed(int(seed))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=bool(shuffle),
            generator=generator if shuffle else None,
            drop_last=False,
        )

    @torch.no_grad()
    def collect(self, model, features, anchor):
        model.eval()
        buffers = {
            key: []
            for key in (
                "quantiles",
                "median",
                "correction",
                "interval_width_80",
                "interval_width_50",
            )
        }
        for start in range(0, len(features), self.batch_size):
            output = model(
                features[start : start + self.batch_size].to(self.args.device),
                anchor[start : start + self.batch_size].to(self.args.device),
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        return {
            key: torch.cat(value, dim=0) for key, value in buffers.items()
        }

    @staticmethod
    def diagnostics(output, labels, anchor):
        labels = labels.view(-1)
        anchor = anchor.view(-1)
        quantiles = output["quantiles"]
        median_prediction = output["median"].view(-1)
        return {
            "anchor_mae": float(torch.abs(anchor - labels).mean().item()),
            "median_mae": float(
                torch.abs(median_prediction - labels).mean().item()
            ),
            "median_gain": float(
                (
                    torch.abs(anchor - labels)
                    - torch.abs(median_prediction - labels)
                ).mean().item()
            ),
            "mean_width_80": float(
                output["interval_width_80"].mean().item()
            ),
            "mean_width_50": float(
                output["interval_width_50"].mean().item()
            ),
            "coverage_80": float(
                (
                    (labels >= quantiles[:, 0])
                    & (labels <= quantiles[:, 4])
                ).float().mean().item()
            ),
            "coverage_50": float(
                (
                    (labels >= quantiles[:, 1])
                    & (labels <= quantiles[:, 3])
                ).float().mean().item()
            ),
        }

    def train_fold(self, train_indices, valid_indices, seed):
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        train_loader = self.loader(train_indices, True, seed)
        valid_index = torch.as_tensor(valid_indices, dtype=torch.long)
        best = None
        history = []
        stale = 0
        for epoch in range(1, self.max_epochs + 1):
            model.train()
            totals = {}
            for features, anchor, labels in train_loader:
                output = model(
                    features.to(self.args.device),
                    anchor.to(self.args.device),
                )
                losses = distributional_target_loss(
                    output,
                    labels.to(self.args.device),
                    anchor.to(self.args.device),
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(
                        value.detach().item()
                    )
            valid = self.collect(
                model,
                self.features[valid_index],
                self.anchor[valid_index],
            )
            valid_losses = distributional_target_loss(
                valid,
                self.labels[valid_index],
                self.anchor[valid_index],
            )
            diagnostics = self.diagnostics(
                valid,
                self.labels[valid_index],
                self.anchor[valid_index],
            )
            objective = float(valid_losses["total"].item())
            row = {
                "epoch": epoch,
                "valid_objective": objective,
                **{
                    f"train_{key}": value / max(1, len(train_loader))
                    for key, value in totals.items()
                },
                **{
                    f"valid_{key}": float(value.item())
                    for key, value in valid_losses.items()
                },
                **{
                    f"valid_{key}": value
                    for key, value in diagnostics.items()
                },
            }
            history.append(row)
            if best is None or objective < best["objective"] - 1e-6:
                best = {
                    "objective": objective,
                    "epoch": epoch,
                    "state": cpu_state(model),
                    "valid": diagnostics,
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.early_stop:
                break
        if best is None:
            raise RuntimeError(
                "target fold failed to select a checkpoint"
            )
        model.load_state_dict(best["state"])
        return model, best, history

    def fit(self):
        sample_ids = self.oof["sample_ids"]
        specs, manifest = safe_group_folds(
            sample_ids,
            self.labels.view(-1).tolist(),
            self.folds,
            int(self.args.seed) + 26011,
        )
        manifest.to_csv(
            self.save_dir / "v96_target_group_manifest.csv", index=False
        )
        quantiles = torch.full(
            (len(sample_ids), len(QUANTILE_LEVELS)), float("nan")
        )
        medians = torch.full((len(sample_ids), 1), float("nan"))
        corrections = torch.full_like(medians, float("nan"))
        width80 = torch.full_like(medians, float("nan"))
        width50 = torch.full_like(medians, float("nan"))
        best_epochs = []
        history_rows = []
        for spec in specs:
            fold_seed = int(self.args.seed) + 4001 * (
                spec.outer_fold + 1
            )
            model, best, history = self.train_fold(
                spec.inner_train_indices,
                spec.inner_valid_indices,
                fold_seed,
            )
            holdout = torch.as_tensor(
                spec.outer_holdout_indices, dtype=torch.long
            )
            output = self.collect(
                model,
                self.features[holdout],
                self.anchor[holdout],
            )
            quantiles[holdout] = output["quantiles"]
            medians[holdout] = output["median"]
            corrections[holdout] = output["correction"]
            width80[holdout] = output["interval_width_80"]
            width50[holdout] = output["interval_width_50"]
            best_epochs.append(int(best["epoch"]))
            history_rows.extend(
                {"fold": spec.outer_fold, **row}
                for row in history
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if not all(
            torch.isfinite(value).all()
            for value in (
                quantiles,
                medians,
                corrections,
                width80,
                width50,
            )
        ):
            raise RuntimeError(
                "V9.6 target OOF predictions are incomplete"
            )
        if not bool(
            (quantiles[:, 1:] >= quantiles[:, :-1]).all()
        ):
            raise RuntimeError(
                "V9.6 target OOF quantiles are not monotone"
            )
        pd.DataFrame(history_rows).to_csv(
            self.save_dir / "v96_target_fold_history.csv",
            index=False,
        )
        frame = {
            "sample_id": sample_ids,
            "group_id": self.oof["group_ids"],
            "label": self.labels.view(-1).tolist(),
            "oof_anchor": self.anchor.view(-1).tolist(),
            "target_median": medians.view(-1).tolist(),
            "target_correction": corrections.view(-1).tolist(),
            "interval_width_80": width80.view(-1).tolist(),
            "interval_width_50": width50.view(-1).tolist(),
        }
        for index, level in enumerate(QUANTILE_LEVELS):
            frame[f"q{int(level * 100):02d}"] = (
                quantiles[:, index].tolist()
            )
        pd.DataFrame(frame).to_csv(
            self.save_dir / "v96_target_oof_predictions.csv",
            index=False,
        )
        oof_output = {
            "quantiles": quantiles,
            "median": medians,
            "correction": corrections,
            "interval_width_80": width80,
            "interval_width_50": width50,
        }
        oof_metrics = self.diagnostics(
            oof_output, self.labels, self.anchor
        )
        full_epochs = max(1, int(round(median(best_epochs))))
        torch.manual_seed(int(self.args.seed) + 70001)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.args.seed) + 70001)
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        full_loader = self.loader(
            range(len(sample_ids)),
            True,
            int(self.args.seed) + 70001,
        )
        full_history = []
        for epoch in range(1, full_epochs + 1):
            model.train()
            totals = {}
            for features, anchor, labels in full_loader:
                output = model(
                    features.to(self.args.device),
                    anchor.to(self.args.device),
                )
                losses = distributional_target_loss(
                    output,
                    labels.to(self.args.device),
                    anchor.to(self.args.device),
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
                        key: value / max(1, len(full_loader))
                        for key, value in totals.items()
                    },
                }
            )
        pd.DataFrame(full_history).to_csv(
            self.save_dir / "v96_target_full_history.csv",
            index=False,
        )
        checkpoint = (
            self.save_dir / "distributional_target_coach_v96.pth"
        )
        torch.save(
            {
                "method": "distributional_target_coach_v9_6",
                "feature_dim": int(self.features.size(1)),
                "hidden_dim": self.hidden_dim,
                "dropout": self.dropout,
                "residual_max": self.residual_max,
                "quantile_levels": QUANTILE_LEVELS,
                "crossfit_best_epochs": best_epochs,
                "full_train_epochs": full_epochs,
                "oof_metrics": oof_metrics,
                "state_dict": cpu_state(model),
            },
            checkpoint,
        )
        return model, {
            "checkpoint": str(checkpoint),
            "oof_metrics": oof_metrics,
            "best_epochs": best_epochs,
            "full_train_epochs": full_epochs,
            "oof_output": oof_output,
        }


class DistributionalNearestExpertTrainerV96:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        oof_cache_path,
        valid_pool,
        test_pool,
        hidden_dim=48,
        dropout=0.10,
        residual_max=1.50,
        folds=5,
        max_epochs=50,
        early_stop=8,
        learning_rate=2e-4,
        weight_decay=1e-3,
        batch_size=64,
        minimum_validation_gain=0.0015,
        bootstrap_quantile=0.20,
        bootstrap_repeats=400,
        maximum_activation_rate=0.35,
        maximum_harm_rate=0.03,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.valid_pool = valid_pool
        self.test_pool = test_pool
        self.minimum_validation_gain = float(
            minimum_validation_gain
        )
        self.bootstrap_quantile = float(bootstrap_quantile)
        self.bootstrap_repeats = int(bootstrap_repeats)
        self.maximum_activation_rate = float(
            maximum_activation_rate
        )
        self.maximum_harm_rate = float(maximum_harm_rate)
        self._bootstrap_draw_cache = {}
        self.target_fitter = DistributionalTargetCrossFitterV96(
            args=args,
            save_dir=self.save_dir,
            oof_cache_path=oof_cache_path,
            hidden_dim=hidden_dim,
            dropout=dropout,
            residual_max=residual_max,
            folds=folds,
            max_epochs=max_epochs,
            early_stop=early_stop,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
        )

    @staticmethod
    def _expert_tensor(pool, key):
        return torch.stack(
            [
                pool["experts"][name][key].float().view(-1, 1)
                for name in SPECIALIST_NAMES
            ],
            dim=1,
        )

    def _actions(self, pool):
        return stack_action_predictions(
            pool["anchor"].float(),
            self._expert_tensor(pool, "prediction"),
        )

    def _apply_target(self, model, pool):
        features = coach_input_features(
            pool["function_space"].float(),
            pool["anchor"].float(),
        )
        return self.target_fitter.collect(
            model,
            features,
            pool["anchor"].float(),
        )

    def _bootstrap_gain_lower(
        self,
        anchor_error,
        prediction_error,
        sample_ids,
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
        count = len(gains)
        if count not in self._bootstrap_draw_cache:
            rng = np.random.default_rng(
                int(self.args.seed) + 88001 + count
            )
            self._bootstrap_draw_cache[count] = rng.integers(
                0,
                count,
                size=(self.bootstrap_repeats, count),
            )
        values = gains[
            self._bootstrap_draw_cache[count]
        ].mean(axis=1)
        return float(
            np.quantile(values, self.bootstrap_quantile)
        )

    def _route(self, pool, target_output, policy):
        actions = self._actions(pool)
        risks = action_risk(
            actions,
            target_output,
            str(policy["risk_mode"]),
        )
        selected = risks.argmin(dim=1)
        selected_risk = risks.gather(
            1, selected.view(-1, 1)
        ).view(-1)
        anchor_risk = risks[:, 0]
        estimated_advantage = anchor_risk - selected_risk
        median_value = target_output["median"].view(-1)
        action_values = actions.squeeze(-1)
        median_distances = torch.abs(
            action_values - median_value.view(-1, 1)
        )
        median_advantage = (
            median_distances[:, 0]
            - median_distances.gather(
                1, selected.view(-1, 1)
            ).view(-1)
        )
        width = target_output["interval_width_80"].view(-1)
        confidences = torch.cat(
            [
                torch.ones(len(actions), 1),
                self._expert_tensor(
                    pool, "confidence"
                ).squeeze(-1),
            ],
            dim=1,
        )
        selected_confidence = confidences.gather(
            1, selected.view(-1, 1)
        ).view(-1)
        selected_values = action_values.gather(
            1, selected.view(-1, 1)
        ).view(-1)
        anchor = action_values[:, 0]
        direction_ok = (
            (selected_values - anchor)
            * (median_value - anchor)
            > 0
        )
        activate = selected > 0
        activate &= (
            estimated_advantage
            >= float(policy["min_estimated_advantage"])
        )
        activate &= (
            median_advantage
            >= float(policy["min_median_advantage"])
        )
        activate &= (
            width <= float(policy["max_interval_width"])
        )
        activate &= (
            selected_confidence
            >= float(policy["min_expert_confidence"])
        )
        activate &= direction_ok
        beta = float(policy["beta"])
        prediction = (
            anchor
            + beta
            * (selected_values - anchor)
            * activate.float()
        )
        deployed = torch.where(
            activate,
            selected,
            torch.zeros_like(selected),
        )
        return {
            "prediction": prediction.view(-1, 1),
            "activated": activate,
            "proposed_action": selected,
            "deployed_action": deployed,
            "risks": risks,
            "estimated_advantage": estimated_advantage,
            "median_advantage": median_advantage,
            "selected_confidence": selected_confidence,
            "interval_width_80": width,
        }

    def _calibrate(self, valid_target):
        labels = self.valid_pool["labels"].float().view(-1)
        anchor = self.valid_pool["anchor"].float().view(-1)
        anchor_error = torch.abs(anchor - labels).numpy()
        anchor_mae = float(anchor_error.mean())
        anchor_row = {
            "mode": "anchor",
            "risk_mode": "median_distance",
            "min_estimated_advantage": 999.0,
            "min_median_advantage": 999.0,
            "max_interval_width": 0.0,
            "min_expert_confidence": 1.0,
            "beta": 0.0,
            "mae": anchor_mae,
            "gain": 0.0,
            "bootstrap_gain_lower": 0.0,
            "harm_over_010_rate": 0.0,
            "activation_rate": 0.0,
            "eligible": True,
            "objective": anchor_mae,
        }
        rows = [anchor_row]
        for risk_mode in (
            "median_distance",
            "quantile_risk",
        ):
            for estimated_advantage in (
                0.00,
                0.02,
                0.05,
                0.10,
            ):
                for median_advantage in (
                    0.00,
                    0.02,
                    0.05,
                ):
                    for max_width in (
                        0.50,
                        1.00,
                        1.50,
                        2.50,
                    ):
                        for confidence in (
                            0.00,
                            0.45,
                            0.60,
                        ):
                            for beta in (
                                0.25,
                                0.50,
                                0.75,
                                1.00,
                            ):
                                policy = {
                                    "risk_mode": risk_mode,
                                    "min_estimated_advantage": (
                                        estimated_advantage
                                    ),
                                    "min_median_advantage": (
                                        median_advantage
                                    ),
                                    "max_interval_width": max_width,
                                    "min_expert_confidence": (
                                        confidence
                                    ),
                                    "beta": beta,
                                }
                                route = self._route(
                                    self.valid_pool,
                                    valid_target,
                                    policy,
                                )
                                prediction = route[
                                    "prediction"
                                ].view(-1)
                                error = torch.abs(
                                    prediction - labels
                                ).numpy()
                                gain_values = (
                                    anchor_error - error
                                )
                                mae = float(error.mean())
                                gain = anchor_mae - mae
                                activation = float(
                                    route["activated"]
                                    .float()
                                    .mean()
                                    .item()
                                )
                                lower = self._bootstrap_gain_lower(
                                    anchor_error,
                                    error,
                                    self.valid_pool["sample_ids"],
                                )
                                harm = float(
                                    np.mean(gain_values < -0.10)
                                )
                                eligible = (
                                    gain
                                    >= self.minimum_validation_gain
                                    and lower >= 0.0
                                    and 0.0
                                    < activation
                                    <= self.maximum_activation_rate
                                    and harm
                                    <= self.maximum_harm_rate
                                )
                                rows.append(
                                    {
                                        "mode": (
                                            "distributional_nearest_expert"
                                        ),
                                        **policy,
                                        "mae": mae,
                                        "gain": gain,
                                        "bootstrap_gain_lower": lower,
                                        "harm_over_010_rate": harm,
                                        "activation_rate": activation,
                                        "eligible": bool(eligible),
                                        "objective": (
                                            mae
                                            + 0.05 * harm
                                            + 0.002 * activation
                                            + (
                                                0.0
                                                if eligible
                                                else 1.0
                                            )
                                        ),
                                    }
                                )
        frame = pd.DataFrame(rows)
        frame.to_csv(
            self.save_dir / "v96_route_calibration.csv",
            index=False,
        )
        eligible = frame.loc[
            frame["eligible"] == True  # noqa: E712
        ].sort_values(
            ["objective", "mae", "activation_rate"],
            kind="mergesort",
        )
        selected = (
            anchor_row
            if eligible.empty
            else eligible.iloc[0].to_dict()
        )
        return {
            key: (
                value.item()
                if hasattr(value, "item")
                else value
            )
            for key, value in selected.items()
        }

    def _true_region(self, pool):
        labels = pool["labels"].view(-1)
        regions = target_region_index(labels)
        prediction = (
            pool["anchor"].float().view(-1).clone()
        )
        mapping = {
            0: "strong_negative",
            2: "boundary",
            3: "positive",
            4: "strong_positive",
        }
        for region, name in mapping.items():
            mask = regions == region
            prediction[mask] = (
                pool["experts"][name]["prediction"]
                .float()
                .view(-1)[mask]
            )
        return prediction.view(-1, 1)

    def _oracle(self, pool):
        labels = pool["labels"].float().view(-1, 1)
        actions = self._actions(pool)
        best = torch.abs(
            actions - labels.unsqueeze(1)
        ).squeeze(-1).argmin(dim=1)
        return actions[
            torch.arange(len(labels)), best
        ]

    @staticmethod
    def _action_counts(indices):
        return {
            ACTION_NAMES[index]: int(
                (indices == index).sum().item()
            )
            for index in range(len(ACTION_NAMES))
        }

    def train_all(self):
        model, target_summary = self.target_fitter.fit()
        valid_target = self._apply_target(
            model, self.valid_pool
        )
        test_target = self._apply_target(
            model, self.test_pool
        )
        target_summary = {
            key: value
            for key, value in target_summary.items()
            if key != "oof_output"
        }
        target_summary[
            "valid_metrics"
        ] = self.target_fitter.diagnostics(
            valid_target,
            self.valid_pool["labels"],
            self.valid_pool["anchor"],
        )
        target_summary[
            "test_metrics_diagnostic"
        ] = self.target_fitter.diagnostics(
            test_target,
            self.test_pool["labels"],
            self.test_pool["anchor"],
        )
        policy = self._calibrate(valid_target)
        if policy["mode"] == "anchor":
            n = len(self.test_pool["labels"])
            deploy = {
                "prediction": (
                    self.test_pool["anchor"]
                    .float()
                    .view(-1, 1)
                ),
                "activated": torch.zeros(
                    n, dtype=torch.bool
                ),
                "proposed_action": torch.zeros(
                    n, dtype=torch.long
                ),
                "deployed_action": torch.zeros(
                    n, dtype=torch.long
                ),
                "risks": action_risk(
                    self._actions(self.test_pool),
                    test_target,
                    "median_distance",
                ),
                "estimated_advantage": torch.zeros(n),
                "median_advantage": torch.zeros(n),
                "selected_confidence": torch.ones(n),
                "interval_width_80": test_target[
                    "interval_width_80"
                ].view(-1),
            }
        else:
            deploy = self._route(
                self.test_pool,
                test_target,
                policy,
            )
        unrestricted = {}
        for risk_mode in (
            "median_distance",
            "quantile_risk",
        ):
            unrestricted[risk_mode] = self._route(
                self.test_pool,
                test_target,
                {
                    "risk_mode": risk_mode,
                    "min_estimated_advantage": -999.0,
                    "min_median_advantage": -999.0,
                    "max_interval_width": 999.0,
                    "min_expert_confidence": 0.0,
                    "beta": 1.0,
                },
            )["prediction"]
        labels = self.test_pool["labels"].float().view(-1, 1)
        named = {
            "anchor": (
                self.test_pool["anchor"].float().view(-1, 1)
            ),
            "target_median_direct": (
                test_target["median"].float().view(-1, 1)
            ),
            "nearest_expert_to_target_median_all": (
                unrestricted["median_distance"]
            ),
            "minimum_quantile_risk_expert_all": (
                unrestricted["quantile_risk"]
            ),
            "distributional_target_nearest_expert_valid_selected": (
                deploy["prediction"]
            ),
            "true_region_expert_policy": self._true_region(
                self.test_pool
            ),
            "sample_oracle_upper_bound": self._oracle(
                self.test_pool
            ),
        }
        for name in SPECIALIST_NAMES:
            named[name] = (
                self.test_pool["experts"][name]["prediction"]
                .float()
                .view(-1, 1)
            )
        rows, results = [], {}
        for name, prediction in named.items():
            metrics = safe_metrics(
                self.metrics_fn, prediction, labels
            )
            rows.append({"model": name, **metrics})
            results[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v96_test_summary.csv",
            index=False,
        )

        actions = self._actions(
            self.test_pool
        ).squeeze(-1)
        frame = {
            "sample_id": self.test_pool["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": named["anchor"].view(-1).tolist(),
            "target_median": (
                test_target["median"].view(-1).tolist()
            ),
            "interval_width_80": (
                test_target["interval_width_80"]
                .view(-1)
                .tolist()
            ),
            "distributional_target_nearest_expert": (
                deploy["prediction"].view(-1).tolist()
            ),
            "activated": deploy["activated"].tolist(),
            "proposed_action_index": (
                deploy["proposed_action"].tolist()
            ),
            "deployed_action_index": (
                deploy["deployed_action"].tolist()
            ),
            "proposed_action": [
                ACTION_NAMES[index]
                for index in deploy[
                    "proposed_action"
                ].tolist()
            ],
            "deployed_action": [
                ACTION_NAMES[index]
                for index in deploy[
                    "deployed_action"
                ].tolist()
            ],
            "estimated_advantage": (
                deploy["estimated_advantage"].tolist()
            ),
            "median_advantage": (
                deploy["median_advantage"].tolist()
            ),
            "selected_confidence": (
                deploy["selected_confidence"].tolist()
            ),
        }
        for index, level in enumerate(QUANTILE_LEVELS):
            frame[f"q{int(level * 100):02d}"] = (
                test_target["quantiles"][:, index].tolist()
            )
        risk_mode = str(
            policy.get("risk_mode", "median_distance")
        )
        deployed_risks = action_risk(
            self._actions(self.test_pool),
            test_target,
            risk_mode,
        )
        for index, name in enumerate(ACTION_NAMES):
            frame[f"{name}_prediction"] = (
                actions[:, index].tolist()
            )
            frame[f"{name}_estimated_risk"] = (
                deployed_risks[:, index].tolist()
            )
        pd.DataFrame(frame).to_csv(
            self.save_dir
            / "distributional_target_nearest_expert_v96_predictions.csv",
            index=False,
        )
        summary = {
            "method": (
                "distributional_target_nearest_expert_v9_6"
            ),
            "dataset": str(self.args.dataset_name),
            "seed": int(self.args.seed),
            "action_schema": ACTION_NAMES,
            "target_coach": target_summary,
            "route_policy": policy,
            "test_activation_rate": float(
                deploy["activated"].float().mean().item()
            ),
            "test_proposed_action_counts": (
                self._action_counts(
                    deploy["proposed_action"]
                )
            ),
            "test_deployed_action_counts": (
                self._action_counts(
                    deploy["deployed_action"]
                )
            ),
            "test_results": results,
            "selection_protocol": (
                "The target distribution is trained by source-video grouped "
                "cross-fitting on the V9.2 Train OOF cache. Frozen V9.3 "
                "experts are used only on Validation/Test. Validation "
                "calibrates risk mode and safety thresholds with an explicit "
                "Anchor fallback. Test labels are used only for final metrics "
                "and named diagnostics."
            ),
        }
        (
            self.save_dir
            / "distributional_target_nearest_expert_v96_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        anchor_mae = results["anchor"]["MAE"]
        coach_mae = results[
            "distributional_target_nearest_expert_valid_selected"
        ]["MAE"]
        print(
            "V9.6 TEST anchor=%.6f target-median=%.6f coach=%.6f "
            "true-region=%.6f oracle=%.6f"
            % (
                anchor_mae,
                results["target_median_direct"]["MAE"],
                coach_mae,
                results["true_region_expert_policy"]["MAE"],
                results["sample_oracle_upper_bound"]["MAE"],
            )
        )
        return summary
