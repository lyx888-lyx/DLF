"""Validation-only route calibration and frozen Test evaluation for V9.3."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from .coach_utils_v93 import json_scalar, region_metrics, safe_metrics
from .model.OrdinalAdvantageCoachV93 import (
    REGION_NAMES,
    SPECIALIST_NAMES,
    SPECIALIST_TO_REGION,
    region_index,
)


class CoachRouterV93:
    def __init__(self, args, metrics_fn, save_dir, valid_pool, test_pool):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.valid = valid_pool
        self.test = test_pool

    def route(self, pool, region_output, advantage_output, policy):
        anchor = pool["anchor"].clone()
        selected = torch.full((len(anchor),), -1, dtype=torch.long)
        scores = torch.full(
            (len(anchor), len(SPECIALIST_NAMES)),
            -float("inf"),
        )
        for expert_index, name in enumerate(SPECIALIST_NAMES):
            region_probability = region_output["region_probs"][
                :, SPECIALIST_TO_REGION[name]
            ].view(-1)
            predicted_gain = advantage_output[name][
                "predicted_gain"
            ].view(-1)
            win_probability = advantage_output[name][
                "win_probability"
            ].view(-1)
            eligible = (
                region_probability
                >= float(policy["region_probability"])
            ) & (
                predicted_gain >= float(policy["predicted_gain"])
            ) & (
                win_probability >= float(policy["win_probability"])
            )
            scores[:, expert_index] = torch.where(
                eligible,
                predicted_gain,
                scores[:, expert_index],
            )
        best_scores, best_indices = scores.max(dim=1)
        active = torch.isfinite(best_scores)
        selected[active] = best_indices[active]
        prediction = anchor.clone()
        beta = float(policy["beta"])
        for expert_index, name in enumerate(SPECIALIST_NAMES):
            mask = selected == expert_index
            expert_prediction = pool["experts"][name]["prediction"]
            prediction[mask] = (
                (1.0 - beta) * anchor[mask]
                + beta * expert_prediction[mask]
            )
        return prediction, selected

    def hard_ordinal_route(self, pool, region_output, beta):
        predicted_region = region_output["region_probs"].argmax(
            dim=1
        )
        return self._region_index_route(
            pool, predicted_region, beta
        )

    def anchor_score_route(self, pool, beta):
        return self._region_index_route(
            pool, region_index(pool["anchor"]), beta
        )

    def _region_index_route(self, pool, predicted_region, beta):
        prediction = pool["anchor"].clone()
        selected = torch.full(
            (len(prediction),), -1, dtype=torch.long
        )
        for expert_index, name in enumerate(SPECIALIST_NAMES):
            mask = predicted_region == SPECIALIST_TO_REGION[name]
            expert_prediction = pool["experts"][name]["prediction"]
            prediction[mask] = (
                (1.0 - float(beta)) * prediction[mask]
                + float(beta) * expert_prediction[mask]
            )
            selected[mask] = expert_index
        return prediction, selected

    def calibrate(self, advantage_results, valid_region):
        advantage = {
            name: {
                "predicted_gain": advantage_results[name][
                    "oof_predicted_gain"
                ],
                "win_probability": advantage_results[name][
                    "oof_win_probability"
                ],
            }
            for name in SPECIALIST_NAMES
        }
        rows = []
        anchor = self.valid["anchor"]
        labels = self.valid["labels"]
        anchor_mae = float(
            torch.abs(anchor - labels).mean().item()
        )
        rows.append(
            {
                "mode": "anchor",
                "region_probability": 1.0,
                "predicted_gain": 1e9,
                "win_probability": 1.0,
                "beta": 0.0,
                "mae": anchor_mae,
                "harm_over_010_rate": 0.0,
                "activation_rate": 0.0,
                "objective": anchor_mae,
            }
        )
        for region_threshold in (
            0.20,
            0.30,
            0.40,
            0.50,
            0.60,
        ):
            for gain_threshold in (
                -0.02,
                0.00,
                0.02,
                0.04,
                0.08,
            ):
                for win_threshold in (
                    0.45,
                    0.55,
                    0.65,
                    0.75,
                ):
                    for beta in (0.25, 0.50, 0.75, 1.00):
                        policy = {
                            "region_probability": region_threshold,
                            "predicted_gain": gain_threshold,
                            "win_probability": win_threshold,
                            "beta": beta,
                        }
                        prediction, selected = self.route(
                            self.valid,
                            valid_region,
                            advantage,
                            policy,
                        )
                        gain = (
                            torch.abs(anchor - labels)
                            - torch.abs(prediction - labels)
                        )
                        mae = float(
                            torch.abs(prediction - labels)
                            .mean()
                            .item()
                        )
                        harm = float(
                            (gain < -0.10).float().mean().item()
                        )
                        activation = float(
                            (selected >= 0).float().mean().item()
                        )
                        rows.append(
                            {
                                "mode": "ordinal_advantage",
                                **policy,
                                "mae": mae,
                                "harm_over_010_rate": harm,
                                "activation_rate": activation,
                                "objective": mae + 0.05 * harm,
                            }
                        )
        for mode in (
            "anchor_score_region",
            "hard_ordinal",
        ):
            for beta in (0.25, 0.50, 0.75, 1.00):
                if mode == "anchor_score_region":
                    prediction, selected = self.anchor_score_route(
                        self.valid, beta
                    )
                else:
                    prediction, selected = self.hard_ordinal_route(
                        self.valid, valid_region, beta
                    )
                gain = (
                    torch.abs(anchor - labels)
                    - torch.abs(prediction - labels)
                )
                mae = float(
                    torch.abs(prediction - labels).mean().item()
                )
                harm = float(
                    (gain < -0.10).float().mean().item()
                )
                rows.append(
                    {
                        "mode": mode,
                        "region_probability": 0.0,
                        "predicted_gain": -1e9,
                        "win_probability": 0.0,
                        "beta": beta,
                        "mae": mae,
                        "harm_over_010_rate": harm,
                        "activation_rate": float(
                            (selected >= 0).float().mean().item()
                        ),
                        "objective": mae + 0.05 * harm,
                    }
                )
        frame = pd.DataFrame(rows).sort_values(
            ["objective", "mae", "harm_over_010_rate"],
            kind="mergesort",
        )
        frame.to_csv(
            self.save_dir / "v93_route_calibration.csv",
            index=False,
        )
        selected = {
            key: json_scalar(value)
            for key, value in frame.iloc[0].to_dict().items()
        }
        return selected

    def evaluate(
        self,
        region_meta,
        advantage_results,
        valid_region,
        test_region,
        route_policy,
    ):
        test_advantage = {
            name: {
                "predicted_gain": advantage_results[name][
                    "test_predicted_gain"
                ],
                "win_probability": advantage_results[name][
                    "test_win_probability"
                ],
            }
            for name in SPECIALIST_NAMES
        }
        deployable, selected = self.route(
            self.test,
            test_region,
            test_advantage,
            route_policy,
        )
        calibration = pd.read_csv(
            self.save_dir / "v93_route_calibration.csv"
        )
        hard = (
            calibration.loc[
                calibration["mode"] == "hard_ordinal"
            ]
            .sort_values("objective")
            .iloc[0]
        )
        anchor_score = (
            calibration.loc[
                calibration["mode"] == "anchor_score_region"
            ]
            .sort_values("objective")
            .iloc[0]
        )
        hard_prediction, hard_selected = self.hard_ordinal_route(
            self.test,
            test_region,
            float(hard.beta),
        )
        anchor_score_prediction, anchor_score_selected = (
            self.anchor_score_route(
                self.test, float(anchor_score.beta)
            )
        )
        labels = self.test["labels"]
        anchor = self.test["anchor"]
        true_region = anchor.clone()
        true_index = region_index(labels)
        for name in SPECIALIST_NAMES:
            mask = true_index == SPECIALIST_TO_REGION[name]
            true_region[mask] = self.test["experts"][name][
                "prediction"
            ][mask]
        candidates = [anchor] + [
            self.test["experts"][name]["prediction"]
            for name in SPECIALIST_NAMES
        ]
        stacked = torch.stack(candidates, dim=1)
        errors = torch.abs(
            stacked - labels.unsqueeze(1)
        )
        oracle_index = errors.squeeze(-1).argmin(dim=1)
        sample_oracle = stacked[
            torch.arange(len(labels)), oracle_index
        ]
        named = {
            "anchor": anchor,
            "anchor_score_region_valid_selected": (
                anchor_score_prediction
            ),
            "hard_ordinal_valid_selected": hard_prediction,
            "ordinal_advantage_coach_valid_selected": deployable,
            "true_region_expert_policy": true_region,
            "sample_oracle_upper_bound": sample_oracle,
        }
        for name in SPECIALIST_NAMES:
            named[name] = self.test["experts"][name][
                "prediction"
            ]
        rows = []
        result_payload = {}
        for name, prediction in named.items():
            metrics = safe_metrics(
                self.metrics_fn, prediction, labels
            )
            rows.append({"model": name, **metrics})
            result_payload[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v93_test_summary.csv",
            index=False,
        )
        valid_metrics = region_metrics(
            valid_region["region_probs"],
            self.valid["labels"],
        )
        test_metrics = region_metrics(
            test_region["region_probs"],
            self.test["labels"],
        )
        valid_anchor_probability = torch.nn.functional.one_hot(
            region_index(self.valid["anchor"]),
            num_classes=len(REGION_NAMES),
        ).float()
        test_anchor_probability = torch.nn.functional.one_hot(
            region_index(self.test["anchor"]),
            num_classes=len(REGION_NAMES),
        ).float()
        prediction_frame = {
            "sample_id": self.test["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": anchor.view(-1).tolist(),
            "anchor_score_region": (
                anchor_score_prediction.view(-1).tolist()
            ),
            "hard_ordinal": hard_prediction.view(-1).tolist(),
            "ordinal_advantage_coach": deployable.view(-1).tolist(),
            "true_region_policy": true_region.view(-1).tolist(),
            "sample_oracle_upper_bound": (
                sample_oracle.view(-1).tolist()
            ),
            "selected_expert_index": selected.tolist(),
            "anchor_score_selected_expert_index": (
                anchor_score_selected.tolist()
            ),
            "hard_selected_expert_index": hard_selected.tolist(),
            "ordinal_score": test_region["score"].view(-1).tolist(),
        }
        for index, name in enumerate(REGION_NAMES):
            prediction_frame["p_" + name] = (
                test_region["region_probs"][:, index].tolist()
            )
        for name in SPECIALIST_NAMES:
            prediction_frame[name + "_prediction"] = (
                self.test["experts"][name]["prediction"]
                .view(-1)
                .tolist()
            )
            prediction_frame[name + "_predicted_gain"] = (
                test_advantage[name]["predicted_gain"]
                .view(-1)
                .tolist()
            )
            prediction_frame[name + "_win_probability"] = (
                test_advantage[name]["win_probability"]
                .view(-1)
                .tolist()
            )
        pd.DataFrame(prediction_frame).to_csv(
            self.save_dir
            / "ordinal_advantage_coach_v93_predictions.csv",
            index=False,
        )
        summary = {
            "method": (
                "ordinal_region_plus_group_crossfit_"
                "advantage_coach_v9_3"
            ),
            "dataset": str(self.args.dataset_name),
            "seed": int(self.args.seed),
            "expert_mapping": {
                "strong_negative": "V9.2 strong_negative",
                "negative": "CFCompatKD anchor",
                "boundary": "V9 boundary",
                "positive": "V9 positive",
                "strong_positive": "V9.2 strong_positive",
            },
            "region_coach": {
                "checkpoint": str(region_meta["checkpoint"]),
                "oof_metrics": region_meta["metrics"],
                "anchor_threshold_oof_metrics": (
                    region_meta["anchor_threshold_metrics"]
                ),
                "valid_metrics": valid_metrics,
                "anchor_threshold_valid_metrics": region_metrics(
                    valid_anchor_probability,
                    self.valid["labels"],
                ),
                "test_metrics_diagnostic": test_metrics,
                "anchor_threshold_test_metrics_diagnostic": (
                    region_metrics(
                        test_anchor_probability,
                        self.test["labels"],
                    )
                ),
            },
            "advantage_heads": {
                name: {
                    "checkpoint": str(
                        advantage_results[name]["checkpoint"]
                    ),
                    "oof_diagnostics": advantage_results[name][
                        "diagnostics"
                    ],
                }
                for name in SPECIALIST_NAMES
            },
            "route_policy": route_policy,
            "anchor_score_region_beta": float(anchor_score.beta),
            "hard_ordinal_beta": float(hard.beta),
            "test_results": result_payload,
            "protocol": (
                "The ordinal region coach is trained from group-cross-"
                "fitted Train features. Each advantage head and route "
                "policy use video-group cross-fitting inside Validation. "
                "Test labels are read only for final reporting and named "
                "oracle analyses."
            ),
        }
        (
            self.save_dir
            / "ordinal_advantage_coach_v93_summary.json"
        ).write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )
        return summary
