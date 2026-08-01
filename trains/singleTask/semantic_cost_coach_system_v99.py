"""Safe OOF/Validation policy selection for the V9.9 semantic cost coach."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .coach_utils_v93 import safe_metrics
from .model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    SPECIALIST_NAMES,
    cost_soft_mixture,
    select_action_from_cost,
)
from .oof_group_splits_v92 import conversation_group_id
from .semantic_cost_crossfit_v99 import (
    POLICY_PROFILES,
    SemanticCostCoachConfigV99,
    SemanticCostCoachCrossFitterV99,
    _action_counts,
    _region_index,
    pool_tensors,
)


class SemanticCostTrainerV99:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        semantic_pool_path,
        valid_pool,
        test_pool,
        coach_config: SemanticCostCoachConfigV99,
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
        self.crossfitter = SemanticCostCoachCrossFitterV99(
            args, self.save_dir, semantic_pool_path, coach_config
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
                int(self.args.seed) + 199001 + int(salt)
            )
            self._bootstrap_cache[key] = rng.integers(
                0,
                len(gains),
                size=(self.bootstrap_repeats, len(gains)),
            )
        values = gains[self._bootstrap_cache[key]].mean(axis=1)
        return float(np.quantile(values, self.bootstrap_quantile))

    @staticmethod
    def _apply_profile(actions, signatures, output, profile):
        selected = select_action_from_cost(
            output,
            actions,
            risk_aversion=float(profile["risk_aversion"]),
            confidence_z=float(profile["confidence_z"]),
        )
        indices = selected["selected_index"].view(-1)
        action_values = actions.squeeze(-1)
        selected_values = action_values.gather(
            1, indices.view(-1, 1)
        ).view(-1)
        selected_confidence = signatures[:, :, 3].gather(
            1, indices.view(-1, 1)
        ).view(-1)
        anchor = action_values[:, 0]
        activate = indices > 0
        activate &= selected["predicted_gain"].view(-1) >= float(
            profile["min_predicted_gain"]
        )
        activate &= selected["lower_gain"].view(-1) >= float(
            profile["min_lower_gain"]
        )
        activate &= selected["cost_margin"].view(-1) >= float(
            profile["min_cost_margin"]
        )
        activate &= selected["selected_scale"].view(-1) <= float(
            profile["max_selected_scale"]
        )
        activate &= selected_confidence >= float(
            profile["min_expert_confidence"]
        )
        beta = float(profile["beta"])
        prediction = anchor + beta * (
            selected_values - anchor
        ) * activate.float()
        deployed = torch.where(
            activate, indices, torch.zeros_like(indices)
        )
        return {
            "prediction": prediction.view(-1, 1),
            "activated": activate,
            "proposed_action": indices,
            "deployed_action": deployed,
            "selected_confidence": selected_confidence,
            **selected,
        }

    def _profile_rows(self, output):
        pool = self.crossfitter.pool
        actions = self.crossfitter.actions
        signatures = self.crossfitter.signatures
        labels = pool["labels"].float().view(-1)
        anchor = pool["anchor"].float().view(-1)
        anchor_error = torch.abs(anchor - labels).numpy()
        rows = []
        for salt, (name, profile) in enumerate(
            POLICY_PROFILES.items(), start=1
        ):
            route = self._apply_profile(
                actions, signatures, output, profile
            )
            prediction = route["prediction"].view(-1)
            error = torch.abs(prediction - labels).numpy()
            gain_values = anchor_error - error
            fold_gains = []
            fold_values = pool["fold_index"].numpy()
            for fold in sorted(pool["fold_index"].unique().tolist()):
                mask = fold_values == int(fold)
                fold_gains.append(float(gain_values[mask].mean()))
            activation = float(route["activated"].float().mean().item())
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
                and 0.0 < activation <= float(profile["max_activation_rate"])
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
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v99_oof_policy_profiles.csv", index=False
        )
        return rows

    def _apply_model(self, model, pool):
        context, actions, signatures = pool_tensors(pool)
        output = self.crossfitter.collect(model, context, signatures)
        return output, actions, signatures

    def _validation_select(self, model, profile_rows):
        output, actions, signatures = self._apply_model(
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
                key: row[key] for key in POLICY_PROFILES[row["profile"]]
            }
            route = self._apply_profile(
                actions, signatures, output, profile
            )
            error = torch.abs(
                route["prediction"].view(-1) - labels
            ).numpy()
            gain_values = anchor_error - error
            gain = float(anchor_error.mean() - error.mean())
            harm = float(np.mean(gain_values < -0.10))
            activation = float(route["activated"].float().mean().item())
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
            self.save_dir / "v99_validation_profile_selection.csv",
            index=False,
        )
        eligible = frame[frame["eligible"] == True].sort_values(  # noqa: E712
            ["valid_mae", "activation_rate"], kind="mergesort"
        )
        selected_name = (
            "anchor" if eligible.empty else str(eligible.iloc[0]["profile"])
        )
        return selected_name, payload.get(selected_name), rows

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
            prediction[mask] = pool["experts"][name]["prediction"].view(-1)[
                mask
            ]
        return prediction.view(-1, 1)

    @staticmethod
    def _oracle(pool):
        _, actions, _ = pool_tensors(pool)
        labels = pool["labels"].float().view(-1, 1, 1)
        indices = torch.abs(actions - labels).squeeze(-1).argmin(dim=1)
        return actions[torch.arange(len(actions)), indices]

    def train_all(self):
        model, oof_output, coach_summary = self.crossfitter.fit()
        profile_rows = self._profile_rows(oof_output)
        selected_name, selected_profile, valid_rows = self._validation_select(
            model, profile_rows
        )
        test_output, test_actions, test_signatures = self._apply_model(
            model, self.test_pool
        )
        hard_all = select_action_from_cost(test_output, test_actions)
        risk_all = select_action_from_cost(
            test_output, test_actions, risk_aversion=0.50, confidence_z=0.50
        )
        soft_all = cost_soft_mixture(
            test_output,
            test_actions,
            temperature=self.crossfitter.config.selection_temperature,
            risk_aversion=0.25,
        )
        if selected_name == "anchor":
            n = len(test_actions)
            deploy = {
                "prediction": self.test_pool["anchor"].float().view(-1, 1),
                "activated": torch.zeros(n, dtype=torch.bool),
                "proposed_action": risk_all["selected_index"],
                "deployed_action": torch.zeros(n, dtype=torch.long),
                "selected_confidence": torch.ones(n),
                **risk_all,
            }
        else:
            deploy = self._apply_profile(
                test_actions,
                test_signatures,
                test_output,
                selected_profile,
            )
        labels = self.test_pool["labels"].float().view(-1, 1)
        named = {
            "anchor": self.test_pool["anchor"].float().view(-1, 1),
            "minimum_predicted_cost_action_all": hard_all[
                "selected_value"
            ].view(-1, 1),
            "risk_adjusted_cost_action_all": risk_all[
                "selected_value"
            ].view(-1, 1),
            "semantic_cost_soft_mixture_all": soft_all["prediction"],
            "semantic_cost_valid_selected": deploy["prediction"],
            "true_region_expert_policy": self._true_region(self.test_pool),
            "sample_oracle_upper_bound": self._oracle(self.test_pool),
        }
        for name in SPECIALIST_NAMES:
            named[name] = self.test_pool["experts"][name]["prediction"].float()
        rows, results = [], {}
        for name, prediction in named.items():
            metrics = safe_metrics(self.metrics_fn, prediction, labels)
            rows.append({"model": name, **metrics})
            results[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v99_test_summary.csv", index=False
        )

        action_values = test_actions.squeeze(-1)
        frame = {
            "sample_id": self.test_pool["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": named["anchor"].view(-1).tolist(),
            "activated": deploy["activated"].tolist(),
            "proposed_action": [
                ACTION_NAMES[index]
                for index in deploy["proposed_action"].tolist()
            ],
            "deployed_action": [
                ACTION_NAMES[index]
                for index in deploy["deployed_action"].tolist()
            ],
            "predicted_selected_gain": deploy["predicted_gain"]
            .view(-1)
            .tolist(),
            "predicted_lower_gain": deploy["lower_gain"].view(-1).tolist(),
            "predicted_cost_margin": deploy["cost_margin"].view(-1).tolist(),
            "selected_scale": deploy["selected_scale"].view(-1).tolist(),
            "selected_confidence": deploy["selected_confidence"]
            .view(-1)
            .tolist(),
            "semantic_cost_prediction": deploy["prediction"]
            .view(-1)
            .tolist(),
        }
        for index, name in enumerate(ACTION_NAMES):
            frame[f"{name}_prediction"] = action_values[:, index].tolist()
            frame[f"{name}_predicted_cost"] = test_output[
                "predicted_cost"
            ][:, index].tolist()
            frame[f"{name}_predicted_scale"] = test_output[
                "predicted_scale"
            ][:, index].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "semantic_cost_coach_v99_predictions.csv",
            index=False,
        )
        summary = {
            "method": "semantic_signature_per_action_cost_coach_v9_9",
            "dataset": str(self.args.dataset_name),
            "seed": int(self.args.seed),
            "action_schema": ACTION_NAMES,
            "signature_version": SIGNATURE_VERSION,
            "signature_fields": list(SIGNATURE_FIELDS),
            "semantic_pool": {
                "path": str(self.crossfitter.pool_path),
                "version": self.crossfitter.pool.get("version"),
                "provenance": self.crossfitter.pool.get("provenance"),
            },
            "semantic_cost_coach": coach_summary,
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
                "Every action exposes a fixed semantic signature. The coach "
                "predicts the absolute-error mean and scale for each action, "
                "uses cost-sensitive pairwise ranking, and computes safety gain "
                "from the action it actually proposes. Only three pre-registered "
                "OOF-screened profiles are compared on Validation."
            ),
        }
        (
            self.save_dir / "semantic_cost_coach_v99_summary.json"
        ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "V9.9 TEST anchor=%.6f hard=%.6f soft=%.6f deploy=%.6f "
            "true-region=%.6f oracle=%.6f"
            % (
                results["anchor"]["MAE"],
                results["minimum_predicted_cost_action_all"]["MAE"],
                results["semantic_cost_soft_mixture_all"]["MAE"],
                results["semantic_cost_valid_selected"]["MAE"],
                results["true_region_expert_policy"]["MAE"],
                results["sample_oracle_upper_bound"]["MAE"],
            )
        )
        return summary
