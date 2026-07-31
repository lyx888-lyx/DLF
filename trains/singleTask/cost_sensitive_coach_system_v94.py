"""Final refit, Validation-only calibration, and Test evaluation for V9.4."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .cfcompat_fold_training_v92 import MissingModalityWrapper
from .function_space_features_v92 import collect_wrapper_function_space
from .model.CostSensitiveCoachV94 import (
    ACTION_NAMES,
    REGION_NAMES,
    SPECIALIST_NAMES,
    CostSensitiveCoachV94,
    coach_input_features,
    cost_sensitive_coach_loss,
    exact_specialist_mask,
    region_index,
    stack_action_predictions,
)
from .model.DLF import DLF
from .oof_expert_pool_v94 import (
    CoachConfigV94,
    SpecialistConfigV94,
    collect_specialist_models,
    fit_final_specialists,
    median_selected_coach_epoch,
)
from .oof_tail_residual_system_v92 import load_cfcompat_checkpoint

logger = logging.getLogger("MMSA")


def _cpu_state_dict(model: nn.Module):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def _safe_metrics(metrics_fn, prediction, labels):
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _region_metrics(probabilities: torch.Tensor, labels: torch.Tensor):
    target = region_index(labels).cpu()
    predicted = probabilities.argmax(dim=1).cpu()
    precision_values = []
    recall_values = []
    f1_values = []
    for category in range(len(REGION_NAMES)):
        tp = int(((predicted == category) & (target == category)).sum().item())
        fp = int(((predicted == category) & (target != category)).sum().item())
        fn = int(((predicted != category) & (target == category)).sum().item())
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2.0 * precision * recall / max(1e-12, precision + recall)
        precision_values.append(precision)
        recall_values.append(recall)
        f1_values.append(f1)
    expected = (
        probabilities.cpu() * torch.arange(len(REGION_NAMES)).view(1, -1)
    ).sum(dim=1)
    return {
        "accuracy": float((predicted == target).float().mean().item()),
        "macro_precision": float(np.mean(precision_values)),
        "macro_recall": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "ordinal_index_mae": float(
            torch.abs(expected - target.float()).mean().item()
        ),
    }


def _loader(tensors, batch_size, seed):
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        TensorDataset(*tensors),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
        drop_last=False,
    )


def _action_for_region(region: torch.Tensor) -> torch.Tensor:
    mapping = region.new_tensor((1, 0, 2, 3, 4))
    return mapping[region.long()]


def _gather_actions(actions: torch.Tensor, indices: torch.Tensor):
    return actions[torch.arange(actions.size(0)), indices.long()]


class CostSensitiveOOFCoachTrainerV94:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir: Path,
        nested_pool_path: Path,
        anchor_checkpoint: Path,
        valid_loader,
        test_loader,
        specialist_config: SpecialistConfigV94,
        coach_config: CoachConfigV94,
        min_oof_role_gain: float = 0.0,
    ) -> None:
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.pool = torch.load(nested_pool_path, map_location="cpu")
        self.anchor_checkpoint = Path(anchor_checkpoint)
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.min_oof_role_gain = float(min_oof_role_gain)
        required = {
            "sample_ids", "labels", "anchor", "function_space",
            "expert_predictions", "expert_confidences", "expert_names",
            "action_names", "fold_metadata",
        }
        if not required.issubset(self.pool):
            raise ValueError("nested V9.4 pool is incomplete")
        if tuple(self.pool["expert_names"]) != SPECIALIST_NAMES:
            raise ValueError("nested specialist schema mismatch")
        if tuple(self.pool["action_names"]) != ACTION_NAMES:
            raise ValueError("nested action schema mismatch")
        self.specialist_config = SpecialistConfigV94(
            **self.pool.get("specialist_config", specialist_config.__dict__)
        )
        self.coach_config = CoachConfigV94(
            **self.pool.get("coach_config", coach_config.__dict__)
        )
        if not self.anchor_checkpoint.is_file():
            raise FileNotFoundError(self.anchor_checkpoint)

    def _collect_anchor_split(self, loader):
        backbone = DLF(self.args).to(self.args.device)
        wrapper = MissingModalityWrapper(
            backbone,
            int(self.args.feature_dims[1]),
            int(self.args.feature_dims[2]),
        ).to(self.args.device)
        load_cfcompat_checkpoint(wrapper, self.anchor_checkpoint, self.args.device)
        data = collect_wrapper_function_space(wrapper, loader, self.args.device)
        del wrapper
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return data

    def _oof_specialist_gains(self):
        labels = self.pool["labels"].float()
        anchor = self.pool["anchor"].float()
        experts = self.pool["expert_predictions"].float()
        rows = []
        enabled = {}
        for index, role in enumerate(SPECIALIST_NAMES):
            mask = exact_specialist_mask(labels, role)
            if not mask.any():
                raise RuntimeError(f"OOF pool has no sample for {role}")
            anchor_mae = float(torch.abs(anchor[mask] - labels[mask]).mean().item())
            expert_mae = float(
                torch.abs(experts[mask, index] - labels[mask]).mean().item()
            )
            gain = anchor_mae - expert_mae
            enabled[role] = bool(gain > self.min_oof_role_gain)
            rows.append(
                {
                    "role": role,
                    "count": int(mask.sum().item()),
                    "anchor_mae": anchor_mae,
                    "expert_mae": expert_mae,
                    "oof_role_gain": gain,
                    "enabled": enabled[role],
                }
            )
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v94_nested_oof_specialist_capability.csv", index=False
        )
        return enabled, rows

    @staticmethod
    def _apply_enabled(predictions, confidences, anchor, enabled):
        predictions = predictions.clone()
        confidences = confidences.clone()
        for index, role in enumerate(SPECIALIST_NAMES):
            if not enabled[role]:
                predictions[:, index] = anchor
                confidences[:, index].zero_()
        return predictions, confidences

    def _fit_final_coach(self, expert_predictions, expert_confidences, epochs):
        features = coach_input_features(
            self.pool["function_space"].float(),
            self.pool["anchor"].float(),
            expert_predictions,
            expert_confidences,
        )
        actions = stack_action_predictions(
            self.pool["anchor"].float(), expert_predictions
        )
        torch.manual_seed(int(self.args.seed) + 94001)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(self.args.seed) + 94001)
        model = CostSensitiveCoachV94(
            input_dim=features.size(1),
            hidden_dim=self.coach_config.hidden_dim,
            dropout=self.coach_config.dropout,
            ordinal_residual_max=self.coach_config.ordinal_residual_max,
        ).to(self.args.device)
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.coach_config.learning_rate,
            weight_decay=self.coach_config.weight_decay,
        )
        loader = _loader(
            (
                features.float(),
                self.pool["anchor"].float(),
                actions.float(),
                self.pool["labels"].float(),
            ),
            self.coach_config.batch_size,
            int(self.args.seed) + 94001,
        )
        history = []
        for epoch in range(1, int(epochs) + 1):
            model.train()
            totals: Dict[str, float] = {}
            for batch_features, batch_anchor, batch_actions, batch_labels in loader:
                output = model(
                    batch_features.to(self.args.device),
                    batch_anchor.to(self.args.device),
                )
                losses = cost_sensitive_coach_loss(
                    output,
                    batch_actions.to(self.args.device),
                    batch_labels.to(self.args.device),
                    cost_temperature=self.coach_config.cost_temperature,
                    abstain_margin=self.coach_config.abstain_margin,
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
                    **{key: value / max(1, len(loader)) for key, value in totals.items()},
                }
            )
        model.eval()
        pd.DataFrame(history).to_csv(
            self.save_dir / "v94_final_coach_history.csv", index=False
        )
        checkpoint = self.save_dir / "cost_sensitive_oof_coach_v94.pth"
        torch.save(
            {
                "method": "cost_sensitive_oof_coach_v9_4",
                "epochs": int(epochs),
                "input_dim": int(features.size(1)),
                "config": self.coach_config.__dict__,
                "state_dict": _cpu_state_dict(model),
            },
            checkpoint,
        )
        return model, checkpoint

    @torch.no_grad()
    def _collect_coach(self, model, split, expert_predictions, expert_confidences):
        features = coach_input_features(
            split["feature"].float(),
            split["anchor"].float(),
            expert_predictions,
            expert_confidences,
        )
        output = model(
            features.to(self.args.device), split["anchor"].to(self.args.device)
        )
        return {key: value.detach().cpu() for key, value in output.items()}

    @staticmethod
    def _temperature_probs(logits, temperature):
        return torch.softmax(logits / max(float(temperature), 1e-4), dim=1)

    @staticmethod
    def _route_prediction(actions, probabilities, policy):
        anchor = actions[:, 0]
        prediction = anchor.clone()
        top_two = torch.topk(probabilities, k=2, dim=1)
        selected = top_two.indices[:, 0]
        confidence = top_two.values[:, 0]
        margin = top_two.values[:, 0] - top_two.values[:, 1]
        activate = (
            (selected > 0)
            & (confidence >= float(policy["min_probability"]))
            & (margin >= float(policy["min_margin"]))
        )
        beta = float(policy["beta"])
        chosen = _gather_actions(actions, selected)
        prediction[activate] = (
            (1.0 - beta) * anchor[activate] + beta * chosen[activate]
        )
        return prediction, selected, activate, confidence, margin

    def _calibrate(self, valid, valid_experts, valid_coach):
        labels = valid["labels"]
        actions = stack_action_predictions(valid["anchor"], valid_experts)
        rows = [
            {
                "mode": "anchor",
                "temperature": 1.0,
                "min_probability": 1.0,
                "min_margin": 1.0,
                "beta": 0.0,
                "mae": float(torch.abs(valid["anchor"] - labels).mean().item()),
                "harm_over_010_rate": 0.0,
                "activation_rate": 0.0,
            }
        ]
        best = None
        for temperature in (0.50, 0.75, 1.00, 1.25, 1.50, 2.00):
            probabilities = self._temperature_probs(
                valid_coach["action_logits"], temperature
            )
            for min_probability in (0.0, 0.30, 0.40, 0.50, 0.60, 0.70):
                for min_margin in (0.0, 0.05, 0.10, 0.20, 0.30):
                    for beta in (0.25, 0.50, 0.75, 1.00):
                        policy = {
                            "temperature": temperature,
                            "min_probability": min_probability,
                            "min_margin": min_margin,
                            "beta": beta,
                        }
                        prediction, _, activate, _, _ = self._route_prediction(
                            actions, probabilities, policy
                        )
                        mae = float(torch.abs(prediction - labels).mean().item())
                        gain = torch.abs(valid["anchor"] - labels) - torch.abs(
                            prediction - labels
                        )
                        harm = float((gain < -0.10).float().mean().item())
                        rows.append(
                            {
                                "mode": "cost_sensitive",
                                **policy,
                                "mae": mae,
                                "harm_over_010_rate": harm,
                                "activation_rate": float(activate.float().mean().item()),
                            }
                        )
        for row in rows:
            row["objective"] = row["mae"] + 0.03 * row["harm_over_010_rate"]
            if row["mode"] in ("anchor", "cost_sensitive") and (
                best is None
                or (row["objective"], row["mae"], row["activation_rate"])
                < (best["objective"], best["mae"], best["activation_rate"])
            ):
                best = dict(row)
        if best is None:
            raise RuntimeError("route calibration produced no deployable policy")
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v94_route_calibration.csv", index=False
        )
        return best

    @staticmethod
    def _best_beta_for_indices(actions, labels, indices, anchor):
        best = None
        for beta in (0.25, 0.50, 0.75, 1.00):
            chosen = _gather_actions(actions, indices)
            prediction = anchor.clone()
            mask = indices > 0
            prediction[mask] = (
                (1.0 - beta) * anchor[mask] + beta * chosen[mask]
            )
            mae = float(torch.abs(prediction - labels).mean().item())
            if best is None or mae < best["mae"]:
                best = {"beta": beta, "mae": mae, "prediction": prediction}
        return best

    def run(self):
        enabled, capability_rows = self._oof_specialist_gains()
        oof_experts, oof_confidences = self._apply_enabled(
            self.pool["expert_predictions"].float(),
            self.pool["expert_confidences"].float(),
            self.pool["anchor"].float(),
            enabled,
        )
        final_specialists, _ = fit_final_specialists(
            self.pool["function_space"].float(),
            self.pool["anchor"].float(),
            self.pool["labels"].float(),
            self.specialist_config,
            self.args.device,
            int(self.args.seed) + 95001,
            self.save_dir / "final_specialists",
        )
        selected_epoch = median_selected_coach_epoch(self.pool)
        final_coach, coach_checkpoint = self._fit_final_coach(
            oof_experts, oof_confidences, selected_epoch
        )

        valid = self._collect_anchor_split(self.valid_loader)
        test = self._collect_anchor_split(self.test_loader)
        valid_specialists = collect_specialist_models(
            final_specialists, valid["feature"], valid["anchor"], self.args.device
        )
        test_specialists = collect_specialist_models(
            final_specialists, test["feature"], test["anchor"], self.args.device
        )
        valid_experts, valid_confidences = self._apply_enabled(
            valid_specialists["predictions"],
            valid_specialists["confidences"],
            valid["anchor"],
            enabled,
        )
        test_experts, test_confidences = self._apply_enabled(
            test_specialists["predictions"],
            test_specialists["confidences"],
            test["anchor"],
            enabled,
        )
        valid_coach = self._collect_coach(
            final_coach, valid, valid_experts, valid_confidences
        )
        test_coach = self._collect_coach(
            final_coach, test, test_experts, test_confidences
        )
        route_policy = self._calibrate(valid, valid_experts, valid_coach)

        valid_actions = stack_action_predictions(valid["anchor"], valid_experts)
        test_actions = stack_action_predictions(test["anchor"], test_experts)
        test_probs = self._temperature_probs(
            test_coach["action_logits"], route_policy["temperature"]
        )
        deployable, selected, activate, confidence, margin = self._route_prediction(
            test_actions, test_probs, route_policy
        )

        valid_anchor_action = _action_for_region(region_index(valid["anchor"]))
        anchor_score_policy = self._best_beta_for_indices(
            valid_actions,
            valid["labels"],
            valid_anchor_action,
            valid["anchor"],
        )
        valid_ordinal_action = _action_for_region(
            valid_coach["region_probs"].argmax(dim=1)
        )
        ordinal_policy = self._best_beta_for_indices(
            valid_actions,
            valid["labels"],
            valid_ordinal_action,
            valid["anchor"],
        )

        test_anchor_action = _action_for_region(region_index(test["anchor"]))
        chosen = _gather_actions(test_actions, test_anchor_action)
        anchor_score = test["anchor"].clone()
        mask = test_anchor_action > 0
        anchor_score[mask] = (
            (1.0 - anchor_score_policy["beta"]) * test["anchor"][mask]
            + anchor_score_policy["beta"] * chosen[mask]
        )

        test_ordinal_action = _action_for_region(
            test_coach["region_probs"].argmax(dim=1)
        )
        chosen = _gather_actions(test_actions, test_ordinal_action)
        ordinal_route = test["anchor"].clone()
        mask = test_ordinal_action > 0
        ordinal_route[mask] = (
            (1.0 - ordinal_policy["beta"]) * test["anchor"][mask]
            + ordinal_policy["beta"] * chosen[mask]
        )

        true_action = _action_for_region(region_index(test["labels"]))
        true_region = _gather_actions(test_actions, true_action)
        costs = torch.abs(
            test_actions - test["labels"].view(-1, 1, 1)
        ).squeeze(-1)
        oracle_action = costs.argmin(dim=1)
        sample_oracle = _gather_actions(test_actions, oracle_action)

        named = {
            "anchor": test["anchor"],
            "anchor_score_region_valid_selected": anchor_score,
            "ordinal_region_valid_selected": ordinal_route,
            "cost_sensitive_oof_coach_valid_selected": deployable,
            "true_region_expert_policy": true_region,
            "sample_oracle_upper_bound": sample_oracle,
        }
        for index, role in enumerate(SPECIALIST_NAMES):
            named[role] = test_experts[:, index]
        rows = []
        results = {}
        for name, prediction in named.items():
            metrics = _safe_metrics(self.metrics_fn, prediction, test["labels"])
            rows.append({"model": name, **metrics})
            results[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v94_test_summary.csv", index=False
        )

        region_diagnostics = {
            "nested_oof": _region_metrics(
                self.pool["coach_region_probs"], self.pool["labels"]
            ),
            "valid": _region_metrics(valid_coach["region_probs"], valid["labels"]),
            "test_diagnostic": _region_metrics(
                test_coach["region_probs"], test["labels"]
            ),
        }
        frame = {
            "sample_id": test["sample_ids"],
            "label": test["labels"].view(-1).tolist(),
            "anchor": test["anchor"].view(-1).tolist(),
            "anchor_score_region": anchor_score.view(-1).tolist(),
            "ordinal_region": ordinal_route.view(-1).tolist(),
            "cost_sensitive_coach": deployable.view(-1).tolist(),
            "true_region_policy": true_region.view(-1).tolist(),
            "sample_oracle_upper_bound": sample_oracle.view(-1).tolist(),
            "selected_action": selected.tolist(),
            "activated": activate.tolist(),
            "selected_probability": confidence.tolist(),
            "selected_margin": margin.tolist(),
            "ordinal_score": test_coach["ordinal_score"].view(-1).tolist(),
        }
        for action_index, action_name in enumerate(ACTION_NAMES):
            frame[f"p_action_{action_name}"] = test_probs[:, action_index].tolist()
        for role_index, role in enumerate(SPECIALIST_NAMES):
            frame[f"{role}_prediction"] = test_experts[:, role_index, 0].tolist()
            frame[f"{role}_confidence"] = test_confidences[:, role_index, 0].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "cost_sensitive_oof_coach_v94_predictions.csv",
            index=False,
        )

        summary = {
            "method": "fully_nested_oof_cost_sensitive_coach_v9_4",
            "dataset": str(self.args.dataset_name),
            "seed": int(self.args.seed),
            "anchor_checkpoint": str(self.anchor_checkpoint),
            "nested_pool_version": self.pool.get("version"),
            "nested_pool_protocol": self.pool.get("protocol"),
            "enabled_specialists": enabled,
            "nested_oof_specialist_capability": capability_rows,
            "coach_selected_epoch": int(selected_epoch),
            "coach_checkpoint": str(coach_checkpoint),
            "route_policy": route_policy,
            "anchor_score_beta": float(anchor_score_policy["beta"]),
            "ordinal_region_beta": float(ordinal_policy["beta"]),
            "region_diagnostics": region_diagnostics,
            "test_results": results,
            "test_activation_rate": float(activate.float().mean().item()),
            "test_proposed_action_counts": {
                ACTION_NAMES[index]: int((selected == index).sum().item())
                for index in range(len(ACTION_NAMES))
            },
            "test_deployed_action_counts": {
                ACTION_NAMES[index]: int(
                    (((selected == index) & activate) if index > 0 else (~activate)).sum().item()
                )
                for index in range(len(ACTION_NAMES))
            },
            "selection_protocol": (
                "Every Train action cost comes from the fully nested grouped OOF "
                "expert pool. Final specialists and coach are refit on Train OOF data. "
                "Only route confidence, margin, temperature, and beta use Validation. "
                "Test labels are used only for final metrics and named oracle diagnostics."
            ),
        }
        (self.save_dir / "cost_sensitive_oof_coach_v94_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info(
            "V9.4 TEST anchor=%.6f coach=%.6f true-region=%.6f oracle=%.6f",
            results["anchor"]["MAE"],
            results["cost_sensitive_oof_coach_valid_selected"]["MAE"],
            results["true_region_expert_policy"]["MAE"],
            results["sample_oracle_upper_bound"]["MAE"],
        )
        return summary
