"""Conservative category routing over the frozen V9.3 expert pool."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .coach_utils_v93 import region_metrics, safe_metrics
from .model.OrdinalAdvantageCoachV93 import coach_input_features as v93_features
from .model.SelectiveCategoryCoachV95 import (
    REGION_NAMES,
    REGION_TO_SPECIALIST,
    SPECIALIST_NAMES,
    boundary_distance,
    region_index,
    semantic_region_probabilities,
    specialist_center_progress,
)
from .oof_group_splits_v92 import conversation_group_id
from .ordinal_region_coach_system_v93 import OrdinalRegionCoachCrossFitterV93


class SelectiveCategoryCoachTrainerV95:
    """Train an ordinal category model, but deploy only high-confidence routes."""

    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        oof_cache_path,
        valid_pool,
        test_pool,
        hidden_dim=32,
        dropout=0.10,
        residual_max=0.50,
        ordinal_temperature=0.45,
        folds=5,
        max_epochs=40,
        early_stop=7,
        learning_rate=2e-4,
        weight_decay=1e-3,
        batch_size=64,
        minimum_validation_gain=0.0015,
        bootstrap_quantile=0.20,
        bootstrap_repeats=400,
        maximum_activation_rate=0.35,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.valid_pool = valid_pool
        self.test_pool = test_pool
        self.ordinal_temperature = float(ordinal_temperature)
        self.minimum_validation_gain = float(minimum_validation_gain)
        self.bootstrap_quantile = float(bootstrap_quantile)
        self.bootstrap_repeats = int(bootstrap_repeats)
        self.maximum_activation_rate = float(maximum_activation_rate)
        self._bootstrap_draw_cache = {}
        self.region_fitter = OrdinalRegionCoachCrossFitterV93(
            args=args,
            save_dir=self.save_dir,
            oof_cache_path=oof_cache_path,
            hidden_dim=int(hidden_dim),
            dropout=float(dropout),
            residual_max=float(residual_max),
            ordinal_temperature=float(ordinal_temperature),
            folds=int(folds),
            max_epochs=int(max_epochs),
            early_stop=int(early_stop),
            learning_rate=float(learning_rate),
            weight_decay=float(weight_decay),
            batch_size=int(batch_size),
        )

    def _fit_region_model(self):
        model, result = self.region_fitter.fit()
        source = Path(result["checkpoint"])
        target = self.save_dir / "selective_category_calibrator_v95.pth"
        shutil.copy2(source, target)
        oof_frame = pd.read_csv(self.save_dir / "v93_region_oof_predictions.csv")
        oof_frame.rename(
            columns={
                "ordinal_score": "calibrated_score",
                "predicted_region": "calibrated_region",
            }
        ).to_csv(self.save_dir / "v95_region_oof_predictions.csv", index=False)
        shutil.copy2(
            self.save_dir / "v93_region_group_manifest.csv",
            self.save_dir / "v95_region_group_manifest.csv",
        )
        shutil.copy2(
            self.save_dir / "v93_region_fold_history.csv",
            self.save_dir / "v95_region_fold_history.csv",
        )
        return model, {
            "checkpoint": str(target),
            "oof_metrics": result["metrics"],
            "anchor_metrics": result["anchor_threshold_metrics"],
        }

    def _apply_region_model(self, model, pool):
        features = v93_features(pool["function_space"].float(), pool["anchor"].float())
        return self.region_fitter.collect(model, features, pool["anchor"].float())

    @staticmethod
    def _expert(pool, name, key):
        return pool["experts"][name][key].float().view(-1)

    def _bootstrap_gain_lower(self, anchor_error, prediction_error, sample_ids):
        groups = np.asarray([conversation_group_id(value) for value in sample_ids])
        unique = np.unique(groups)
        gains = np.asarray(
            [
                float(anchor_error[groups == group].mean() - prediction_error[groups == group].mean())
                for group in unique
            ],
            dtype=np.float64,
        )
        count = len(gains)
        if count not in self._bootstrap_draw_cache:
            rng = np.random.default_rng(int(self.args.seed) + 55001 + count)
            self._bootstrap_draw_cache[count] = rng.integers(
                0, count, size=(self.bootstrap_repeats, count)
            )
        values = gains[self._bootstrap_draw_cache[count]].mean(axis=1)
        return float(np.quantile(values, self.bootstrap_quantile))

    def _route(self, pool, region_output, policy):
        anchor = pool["anchor"].float().view(-1)
        anchor_probs = semantic_region_probabilities(anchor, policy["temperature"])
        calibrated_score = region_output["score"].float().view(-1)
        calibrated_probs = semantic_region_probabilities(
            calibrated_score, policy["temperature"]
        )
        anchor_region = anchor_probs.argmax(dim=1)
        calibrated_region = calibrated_probs.argmax(dim=1)
        source = policy["source"]
        if source == "anchor_score":
            proposed = anchor_region
            probabilities = anchor_probs
            agreement = torch.ones_like(proposed, dtype=torch.bool)
            score = anchor
        elif source == "calibrated":
            proposed = calibrated_region
            probabilities = calibrated_probs
            agreement = torch.ones_like(proposed, dtype=torch.bool)
            score = calibrated_score
        elif source == "agreement":
            proposed = calibrated_region
            probabilities = calibrated_probs
            agreement = calibrated_region == anchor_region
            score = calibrated_score
        else:
            raise ValueError(f"unknown category source: {source}")

        region_probability = probabilities.gather(1, proposed.view(-1, 1)).view(-1)
        distance = boundary_distance(score)
        chosen = anchor.clone()
        activated = torch.zeros(len(anchor), dtype=torch.bool)
        selected_names = ["anchor"] * len(anchor)
        for region, name in REGION_TO_SPECIALIST.items():
            prediction = self._expert(pool, name, "prediction")
            confidence = self._expert(pool, name, "confidence")
            mask = proposed == int(region)
            mask &= agreement
            mask &= region_probability >= float(policy["min_region_probability"])
            mask &= distance >= float(policy["min_boundary_distance"])
            mask &= confidence >= float(policy["min_expert_confidence"])
            if bool(policy["require_center_progress"]):
                mask &= specialist_center_progress(anchor, prediction, int(region))
            if region == 0:
                mask &= prediction <= anchor
            elif region == 2:
                mask &= prediction.abs() <= anchor.abs()
            else:
                mask &= prediction >= anchor
            chosen[mask] = prediction[mask]
            activated |= mask
            for index in torch.nonzero(mask, as_tuple=False).view(-1).tolist():
                selected_names[index] = name
        beta = float(policy["beta"])
        return {
            "prediction": (anchor + beta * (chosen - anchor)).view(-1, 1),
            "activated": activated,
            "selected_names": selected_names,
            "anchor_region": anchor_region,
            "calibrated_region": calibrated_region,
            "proposed_region": proposed,
            "region_probability": region_probability,
            "boundary_distance": distance,
        }

    def _calibrate(self, valid_region):
        labels = self.valid_pool["labels"].float().view(-1)
        anchor = self.valid_pool["anchor"].float().view(-1)
        anchor_error = torch.abs(anchor - labels).numpy()
        anchor_mae = float(anchor_error.mean())
        anchor_row = {
            "mode": "anchor",
            "source": "anchor_score",
            "temperature": self.ordinal_temperature,
            "min_region_probability": 1.0,
            "min_boundary_distance": 99.0,
            "min_expert_confidence": 1.0,
            "require_center_progress": True,
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
        for source in ("anchor_score", "calibrated", "agreement"):
            for temperature in (0.30, 0.45, 0.60):
                for probability in (0.30, 0.45, 0.60):
                    for distance in (0.00, 0.15, 0.30):
                        for confidence in (0.00, 0.45, 0.60):
                            for center_progress in (False, True):
                                for beta in (0.25, 0.50, 1.00):
                                    policy = {
                                        "source": source,
                                        "temperature": temperature,
                                        "min_region_probability": probability,
                                        "min_boundary_distance": distance,
                                        "min_expert_confidence": confidence,
                                        "require_center_progress": center_progress,
                                        "beta": beta,
                                    }
                                    route = self._route(self.valid_pool, valid_region, policy)
                                    prediction = route["prediction"].view(-1)
                                    error = torch.abs(prediction - labels).numpy()
                                    gain_values = anchor_error - error
                                    mae = float(error.mean())
                                    gain = anchor_mae - mae
                                    activation = float(route["activated"].float().mean().item())
                                    lower = self._bootstrap_gain_lower(
                                        anchor_error, error, self.valid_pool["sample_ids"]
                                    )
                                    harm = float(np.mean(gain_values < -0.10))
                                    eligible = (
                                        gain >= self.minimum_validation_gain
                                        and lower >= 0.0
                                        and 0.0 < activation <= self.maximum_activation_rate
                                    )
                                    rows.append(
                                        {
                                            "mode": "selective_category",
                                            **policy,
                                            "mae": mae,
                                            "gain": gain,
                                            "bootstrap_gain_lower": lower,
                                            "harm_over_010_rate": harm,
                                            "activation_rate": activation,
                                            "eligible": bool(eligible),
                                            "objective": mae + 0.05 * harm + 0.002 * activation
                                            + (0.0 if eligible else 1.0),
                                        }
                                    )
        frame = pd.DataFrame(rows)
        frame.to_csv(self.save_dir / "v95_route_calibration.csv", index=False)
        eligible = frame.loc[frame["eligible"] == True].sort_values(  # noqa: E712
            ["objective", "mae", "activation_rate"], kind="mergesort"
        )
        selected = anchor_row if eligible.empty else eligible.iloc[0].to_dict()
        return {
            key: value.item() if hasattr(value, "item") else value
            for key, value in selected.items()
        }

    def _true_region(self, pool):
        labels = pool["labels"].view(-1)
        regions = region_index(labels)
        prediction = pool["anchor"].float().view(-1).clone()
        for region, name in REGION_TO_SPECIALIST.items():
            mask = regions == int(region)
            prediction[mask] = self._expert(pool, name, "prediction")[mask]
        return prediction.view(-1, 1)

    def _oracle(self, pool):
        labels = pool["labels"].float().view(-1, 1)
        actions = [pool["anchor"].float().view(-1, 1)] + [
            pool["experts"][name]["prediction"].float().view(-1, 1)
            for name in SPECIALIST_NAMES
        ]
        stacked = torch.stack(actions, dim=1)
        best = torch.abs(stacked - labels.unsqueeze(1)).squeeze(-1).argmin(dim=1)
        return stacked[torch.arange(len(labels)), best]

    def train_all(self):
        model, region_summary = self._fit_region_model()
        valid_region = self._apply_region_model(model, self.valid_pool)
        test_region = self._apply_region_model(model, self.test_pool)
        region_summary.update(
            {
                "valid_metrics": region_metrics(
                    valid_region["region_probs"], self.valid_pool["labels"]
                ),
                "test_metrics_diagnostic": region_metrics(
                    test_region["region_probs"], self.test_pool["labels"]
                ),
                "anchor_threshold_valid_metrics": region_metrics(
                    semantic_region_probabilities(
                        self.valid_pool["anchor"], self.ordinal_temperature
                    ),
                    self.valid_pool["labels"],
                ),
                "anchor_threshold_test_metrics_diagnostic": region_metrics(
                    semantic_region_probabilities(
                        self.test_pool["anchor"], self.ordinal_temperature
                    ),
                    self.test_pool["labels"],
                ),
            }
        )
        policy = self._calibrate(valid_region)
        if policy["mode"] == "anchor":
            deploy = {
                "prediction": self.test_pool["anchor"].float().view(-1, 1),
                "activated": torch.zeros(len(self.test_pool["labels"]), dtype=torch.bool),
                "selected_names": ["anchor"] * len(self.test_pool["labels"]),
                "anchor_region": region_index(self.test_pool["anchor"]),
                "calibrated_region": test_region["region_probs"].argmax(dim=1),
                "proposed_region": region_index(self.test_pool["anchor"]),
                "region_probability": torch.ones(len(self.test_pool["labels"])),
                "boundary_distance": boundary_distance(self.test_pool["anchor"]),
            }
        else:
            deploy = self._route(self.test_pool, test_region, policy)

        labels = self.test_pool["labels"].float().view(-1, 1)
        named = {
            "anchor": self.test_pool["anchor"].float().view(-1, 1),
            "selective_category_coach_valid_selected": deploy["prediction"],
            "true_region_expert_policy": self._true_region(self.test_pool),
            "sample_oracle_upper_bound": self._oracle(self.test_pool),
        }
        for name in SPECIALIST_NAMES:
            named[name] = self.test_pool["experts"][name]["prediction"].float().view(-1, 1)
        rows, results = [], {}
        for name, prediction in named.items():
            metrics = safe_metrics(self.metrics_fn, prediction, labels)
            rows.append({"model": name, **metrics})
            results[name] = metrics
        pd.DataFrame(rows).to_csv(self.save_dir / "v95_test_summary.csv", index=False)

        frame = {
            "sample_id": self.test_pool["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": named["anchor"].view(-1).tolist(),
            "selective_category_coach": deploy["prediction"].view(-1).tolist(),
            "true_region_policy": named["true_region_expert_policy"].view(-1).tolist(),
            "sample_oracle_upper_bound": named["sample_oracle_upper_bound"].view(-1).tolist(),
            "selected_expert": deploy["selected_names"],
            "activated": deploy["activated"].tolist(),
            "anchor_region": deploy["anchor_region"].tolist(),
            "calibrated_region": deploy["calibrated_region"].tolist(),
            "proposed_region": deploy["proposed_region"].tolist(),
            "selected_region_probability": deploy["region_probability"].tolist(),
            "boundary_distance": deploy["boundary_distance"].tolist(),
            "calibrated_score": test_region["score"].view(-1).tolist(),
        }
        for index, name in enumerate(REGION_NAMES):
            frame[f"p_{name}"] = test_region["region_probs"][:, index].tolist()
        for name in SPECIALIST_NAMES:
            frame[f"{name}_prediction"] = self._expert(
                self.test_pool, name, "prediction"
            ).tolist()
            frame[f"{name}_confidence"] = self._expert(
                self.test_pool, name, "confidence"
            ).tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "selective_category_coach_v95_predictions.csv", index=False
        )
        summary = {
            "method": "selective_category_coach_v9_5",
            "dataset": str(self.args.dataset_name),
            "seed": int(self.args.seed),
            "expert_pool": "frozen_v9_boundary_positive_plus_v9_2_tail_experts",
            "region_calibrator": region_summary,
            "route_policy": policy,
            "minimum_validation_gain": self.minimum_validation_gain,
            "bootstrap_quantile": self.bootstrap_quantile,
            "maximum_activation_rate": self.maximum_activation_rate,
            "test_activation_rate": float(deploy["activated"].float().mean().item()),
            "test_selected_expert_counts": pd.Series(
                deploy["selected_names"]
            ).value_counts().to_dict(),
            "test_results": results,
            "protocol": (
                "The ordinal model is trained from grouped V9.2 Train OOF features. "
                "Frozen V9.3 experts are used only when category evidence, boundary "
                "distance, expert confidence, correction direction, and a Validation "
                "group-bootstrap safety gate all pass. Test labels are final-only."
            ),
        }
        (self.save_dir / "selective_category_coach_v95_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary
