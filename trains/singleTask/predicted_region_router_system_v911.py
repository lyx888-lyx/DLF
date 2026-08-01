"""Safe OOF/Validation selection for the V9.11 predicted-region router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

from .coach_utils_v93 import safe_metrics
from .model.PredictedRegionRouterV911 import (
    REGION_NAMES,
    REGION_ROUTER_VERSION,
    SEMANTIC_REGION_ACTION_MAP,
    action_indices_from_regions,
    region_classification_metrics,
    region_index,
    route_by_regions,
)
from .model.SemanticCostCoachV99 import ACTION_NAMES, SPECIALIST_NAMES
from .oof_group_splits_v92 import conversation_group_id
from .predicted_region_router_crossfit_v911 import (
    PredictedRegionRouterConfigV911,
    PredictedRegionRouterCrossFitterV911,
    pool_action_tensors,
)


POLICY_PROFILES = {
    "conservative": {
        "min_max_probability": 0.65,
        "min_probability_margin": 0.25,
        "max_entropy": 1.10,
        "min_model_confidence": 0.55,
        "beta": 0.50,
        "max_activation_rate": 0.20,
    },
    "balanced": {
        "min_max_probability": 0.55,
        "min_probability_margin": 0.15,
        "max_entropy": 1.35,
        "min_model_confidence": 0.50,
        "beta": 0.75,
        "max_activation_rate": 0.35,
    },
    "broad": {
        "min_max_probability": 0.45,
        "min_probability_margin": 0.08,
        "max_entropy": 1.50,
        "min_model_confidence": 0.45,
        "beta": 1.00,
        "max_activation_rate": 0.55,
    },
}


class PredictedRegionRouterTrainerV911:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        semantic_pool_path,
        valid_pool,
        test_pool,
        router_config: PredictedRegionRouterConfigV911,
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
        self.crossfitter = PredictedRegionRouterCrossFitterV911(
            args, self.save_dir, semantic_pool_path, router_config
        )
        self._bootstrap_cache: Dict[tuple, np.ndarray] = {}

    def _bootstrap_lower(self, anchor_error, prediction_error, sample_ids, salt):
        groups = np.asarray([conversation_group_id(value) for value in sample_ids])
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
                int(self.args.seed) + 911991 + int(salt)
            )
            self._bootstrap_cache[key] = rng.integers(
                0,
                len(gains),
                size=(self.bootstrap_repeats, len(gains)),
            )
        values = gains[self._bootstrap_cache[key]].mean(axis=1)
        return float(np.quantile(values, self.bootstrap_quantile))

    @staticmethod
    def _route_statistics(probabilities):
        probabilities = probabilities.clamp_min(1e-8)
        sorted_probability = probabilities.sort(dim=1, descending=True).values
        return {
            "predicted_region": probabilities.argmax(dim=1),
            "max_probability": sorted_probability[:, 0],
            "probability_margin": sorted_probability[:, 0] - sorted_probability[:, 1],
            "entropy": -(probabilities * probabilities.log()).sum(dim=1),
        }

    @classmethod
    def _apply_profile(
        cls,
        actions,
        probabilities,
        model_confidence,
        mapping: Sequence[int],
        profile: Mapping[str, float],
    ):
        stats = cls._route_statistics(probabilities)
        action_indices = action_indices_from_regions(
            stats["predicted_region"], mapping
        )
        values = actions.squeeze(-1)
        selected_values = values.gather(1, action_indices.view(-1, 1)).view(-1)
        anchor = values[:, 0]
        activate = action_indices > 0
        activate &= stats["max_probability"] >= float(
            profile["min_max_probability"]
        )
        activate &= stats["probability_margin"] >= float(
            profile["min_probability_margin"]
        )
        activate &= stats["entropy"] <= float(profile["max_entropy"])
        activate &= model_confidence.view(-1) >= float(
            profile["min_model_confidence"]
        )

        max_count = max(
            0,
            int(np.floor(float(profile["max_activation_rate"]) * len(activate))),
        )
        if int(activate.sum().item()) > max_count:
            limited = torch.zeros_like(activate)
            if max_count > 0:
                score = (
                    stats["max_probability"]
                    + stats["probability_margin"]
                    - stats["entropy"] / np.log(len(REGION_NAMES))
                )
                candidates = torch.where(activate)[0]
                keep_local = torch.topk(
                    score[candidates], k=max_count, largest=True
                ).indices
                limited[candidates[keep_local]] = True
            activate = limited

        beta = float(profile["beta"])
        prediction = anchor + beta * (selected_values - anchor) * activate.float()
        deployed_action = torch.where(
            activate, action_indices, torch.zeros_like(action_indices)
        )
        return {
            "prediction": prediction.view(-1, 1),
            "activated": activate,
            "proposed_action": action_indices,
            "deployed_action": deployed_action,
            **stats,
        }

    @staticmethod
    def _region_probability_mixture(actions, probabilities, mapping):
        table = probabilities.new_tensor(tuple(int(value) for value in mapping))
        mapped_values = actions.squeeze(-1)[:, table]
        return (probabilities * mapped_values).sum(dim=1, keepdim=True)

    def _select_mapping(self, crossfit_output):
        summary = crossfit_output["summary"]
        semantic = float(summary["oof_predicted_semantic_route_mae"])
        empirical = float(summary["oof_predicted_empirical_route_mae"])
        if empirical + 1e-6 < semantic:
            return "empirical", tuple(crossfit_output["final_empirical_map"])
        return "semantic", tuple(SEMANTIC_REGION_ACTION_MAP)

    def _profile_rows(self, crossfit_output, mapping_name, mapping):
        pool = self.crossfitter.pool
        probabilities = crossfit_output["region_probs"]
        confidence = crossfit_output["confidence"]
        labels = self.crossfitter.labels.view(-1)
        anchor = self.crossfitter.anchor.view(-1)
        anchor_error = torch.abs(anchor - labels).numpy()
        fold_values = crossfit_output["coach_fold_index"].numpy()
        rows = []
        for salt, (name, profile) in enumerate(POLICY_PROFILES.items(), start=1):
            route = self._apply_profile(
                self.crossfitter.actions,
                probabilities,
                confidence,
                mapping,
                profile,
            )
            prediction = route["prediction"].view(-1)
            error = torch.abs(prediction - labels).numpy()
            gain_values = anchor_error - error
            fold_gains = []
            for fold in sorted(np.unique(fold_values).tolist()):
                mask = fold_values == int(fold)
                fold_gains.append(float(gain_values[mask].mean()))
            activation = float(route["activated"].float().mean().item())
            gain = float(anchor_error.mean() - error.mean())
            lower = self._bootstrap_lower(
                anchor_error,
                error,
                pool["sample_ids"],
                salt,
            )
            harm = float(np.mean(gain_values < -0.10))
            positive_folds = int(sum(value > 0 for value in fold_gains))
            eligible = (
                gain >= self.minimum_oof_gain
                and lower >= 0.0
                and harm <= self.maximum_harm_rate
                and activation > 0.0
                and positive_folds >= max(3, len(fold_gains) - 1)
            )
            rows.append(
                {
                    "profile": name,
                    "mapping": mapping_name,
                    **profile,
                    "oof_mae": float(error.mean()),
                    "oof_gain": gain,
                    "bootstrap_gain_lower": lower,
                    "harm_over_010_rate": harm,
                    "activation_rate": activation,
                    "positive_coach_folds": positive_folds,
                    "coach_fold_gains": json.dumps(fold_gains),
                    "oof_eligible": bool(eligible),
                }
            )
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v911_oof_policy_profiles.csv", index=False
        )
        return rows

    def _validation_select(self, ensemble, profile_rows, mapping):
        probabilities = ensemble["region_probs"]
        actions = ensemble["actions"]
        confidence = ensemble["confidence"]
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
                actions,
                probabilities,
                confidence,
                mapping,
                profile,
            )
            error = torch.abs(route["prediction"].view(-1) - labels).numpy()
            gain_values = anchor_error - error
            gain = float(anchor_error.mean() - error.mean())
            harm = float(np.mean(gain_values < -0.10))
            activation = float(route["activated"].float().mean().item())
            eligible = (
                gain >= self.minimum_validation_gain
                and harm <= self.maximum_harm_rate
                and activation > 0.0
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
            self.save_dir / "v911_validation_profile_selection.csv", index=False
        )
        eligible = frame[frame["eligible"] == True].sort_values(  # noqa: E712
            ["valid_mae", "activation_rate"], kind="mergesort"
        )
        selected = "anchor" if eligible.empty else str(eligible.iloc[0]["profile"])
        return selected, payload.get(selected), rows

    @staticmethod
    def _true_region(pool, mapping):
        _, actions, _ = pool_action_tensors(pool)
        regions = region_index(pool["labels"].float())
        return route_by_regions(actions, regions, mapping)["prediction"]

    @staticmethod
    def _oracle(pool):
        _, actions, _ = pool_action_tensors(pool)
        labels = pool["labels"].float().view(-1, 1, 1)
        indices = torch.abs(actions - labels).squeeze(-1).argmin(dim=1)
        return actions[torch.arange(len(actions)), indices]

    def train_all(self):
        crossfit_output = self.crossfitter.fit()
        mapping_name, mapping = self._select_mapping(crossfit_output)
        profile_rows = self._profile_rows(crossfit_output, mapping_name, mapping)
        valid_ensemble = self.crossfitter.collect_ensemble(self.valid_pool)
        selected_name, selected_profile, valid_rows = self._validation_select(
            valid_ensemble, profile_rows, mapping
        )
        test_ensemble = self.crossfitter.collect_ensemble(self.test_pool)
        probabilities = test_ensemble["region_probs"]
        actions = test_ensemble["actions"]
        confidence = test_ensemble["confidence"]
        stats = self._route_statistics(probabilities)
        semantic_route = route_by_regions(
            actions, stats["predicted_region"], SEMANTIC_REGION_ACTION_MAP
        )
        empirical_route = route_by_regions(
            actions,
            stats["predicted_region"],
            crossfit_output["final_empirical_map"],
        )
        chosen_route = route_by_regions(actions, stats["predicted_region"], mapping)
        if selected_name == "anchor":
            deploy = {
                "prediction": self.test_pool["anchor"].float().view(-1, 1),
                "activated": torch.zeros(len(actions), dtype=torch.bool),
                "proposed_action": chosen_route["action_indices"],
                "deployed_action": torch.zeros(len(actions), dtype=torch.long),
                **stats,
            }
        else:
            deploy = self._apply_profile(
                actions,
                probabilities,
                confidence,
                mapping,
                selected_profile,
            )

        anchor_regions = region_index(self.test_pool["anchor"].float())
        anchor_threshold = route_by_regions(
            actions, anchor_regions, SEMANTIC_REGION_ACTION_MAP
        )["prediction"]
        named = {
            "anchor": self.test_pool["anchor"].float().view(-1, 1),
            "anchor_threshold_region_route": anchor_threshold,
            "predicted_region_semantic_all": semantic_route["prediction"],
            "predicted_region_empirical_all": empirical_route["prediction"],
            "predicted_region_chosen_mapping_all": chosen_route["prediction"],
            "region_probability_mixture_all": self._region_probability_mixture(
                actions, probabilities, mapping
            ),
            "predicted_region_valid_selected": deploy["prediction"],
            "true_region_semantic_upper_bound": self._true_region(
                self.test_pool, SEMANTIC_REGION_ACTION_MAP
            ),
            "sample_oracle_upper_bound": self._oracle(self.test_pool),
        }
        for name in SPECIALIST_NAMES:
            named[name] = self.test_pool["experts"][name]["prediction"].float()

        labels = self.test_pool["labels"].float().view(-1, 1)
        rows, results = [], {}
        for name, prediction in named.items():
            metrics = safe_metrics(self.metrics_fn, prediction, labels)
            rows.append({"model": name, **metrics})
            results[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v911_test_summary.csv", index=False
        )

        test_region_metrics = region_classification_metrics(probabilities, labels)
        true_regions = region_index(labels)
        predicted_regions = stats["predicted_region"]
        confusion = pd.crosstab(
            pd.Series(
                [REGION_NAMES[index] for index in true_regions.tolist()],
                name="true_region",
            ),
            pd.Series(
                [REGION_NAMES[index] for index in predicted_regions.tolist()],
                name="predicted_region",
            ),
            dropna=False,
        ).reindex(index=REGION_NAMES, columns=REGION_NAMES, fill_value=0)
        confusion.to_csv(self.save_dir / "v911_test_region_confusion.csv")

        frame: Dict[str, object] = {
            "sample_id": self.test_pool["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": named["anchor"].view(-1).tolist(),
            "true_region": [REGION_NAMES[index] for index in true_regions.tolist()],
            "predicted_region": [
                REGION_NAMES[index] for index in predicted_regions.tolist()
            ],
            "activated": deploy["activated"].tolist(),
            "proposed_action": [
                ACTION_NAMES[index] for index in deploy["proposed_action"].tolist()
            ],
            "deployed_action": [
                ACTION_NAMES[index] for index in deploy["deployed_action"].tolist()
            ],
            "max_region_probability": stats["max_probability"].tolist(),
            "region_probability_margin": stats["probability_margin"].tolist(),
            "region_entropy": stats["entropy"].tolist(),
            "model_confidence": confidence.view(-1).tolist(),
            "ensemble_probability_std": test_ensemble[
                "ensemble_probability_std"
            ].view(-1).tolist(),
            "predicted_region_router_prediction": deploy["prediction"].view(-1).tolist(),
        }
        for index, name in enumerate(REGION_NAMES):
            frame[f"p_{name}"] = probabilities[:, index].tolist()
        for index, name in enumerate(ACTION_NAMES):
            frame[f"{name}_prediction"] = actions[:, index, 0].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "predicted_region_router_v911_predictions.csv",
            index=False,
        )

        summary = {
            "method": REGION_ROUTER_VERSION,
            "dataset": str(self.args.dataset_name),
            "seed": int(self.args.seed),
            "region_schema": REGION_NAMES,
            "action_schema": ACTION_NAMES,
            "region_router": crossfit_output["summary"],
            "selected_mapping": mapping_name,
            "selected_region_action_map": [ACTION_NAMES[index] for index in mapping],
            "oof_policy_profiles": profile_rows,
            "validation_profile_rows": valid_rows,
            "selected_profile": selected_name,
            "selected_policy": selected_profile,
            "validation_region_metrics": region_classification_metrics(
                valid_ensemble["region_probs"], self.valid_pool["labels"].float()
            ),
            "test_region_metrics": test_region_metrics,
            "test_activation_rate": float(
                deploy["activated"].float().mean().item()
            ),
            "test_proposed_action_counts": {
                name: int((deploy["proposed_action"] == index).sum().item())
                for index, name in enumerate(ACTION_NAMES)
            },
            "test_deployed_action_counts": {
                name: int((deploy["deployed_action"] == index).sum().item())
                for index, name in enumerate(ACTION_NAMES)
            },
            "test_results": results,
            "selection_protocol": (
                "Five grouped OOF region classifiers predict one of five label "
                "intervals. A Train-only region-to-action map is chosen between "
                "the fixed semantic map and fold-local empirical map. Three "
                "pre-registered confidence gates are screened on OOF and then "
                "Validation; Test labels never choose a map, threshold, or profile."
            ),
        }
        (
            self.save_dir / "predicted_region_router_v911_summary.json"
        ).write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(
            "V9.11 TEST anchor=%.6f predicted=%.6f mixture=%.6f deploy=%.6f "
            "true-region=%.6f oracle=%.6f"
            % (
                results["anchor"]["MAE"],
                results["predicted_region_chosen_mapping_all"]["MAE"],
                results["region_probability_mixture_all"]["MAE"],
                results["predicted_region_valid_selected"]["MAE"],
                results["true_region_semantic_upper_bound"]["MAE"],
                results["sample_oracle_upper_bound"]["MAE"],
            )
        )
        return summary
