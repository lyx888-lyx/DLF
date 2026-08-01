"""Train and evaluate a low-capacity soft gate on one isolated Router-Train split."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .coach_utils_v93 import cpu_state, safe_metrics
from .model.HoldoutSoftGateV912 import (
    GATE_VERSION,
    HoldoutSoftGateV912,
    holdout_soft_gate_loss,
    soft_gate_prediction,
)
from .model.PredictedRegionRouterV911 import (
    SEMANTIC_REGION_ACTION_MAP,
    region_index,
)
from .model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_DIM,
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    SPECIALIST_NAMES,
    global_context_features,
    stack_action_predictions,
)


@dataclass(frozen=True)
class HoldoutSoftGateConfigV912:
    hidden_dim: int = 48
    action_embedding_dim: int = 8
    dropout: float = 0.10
    temperature: float = 1.0
    anchor_bias: float = 1.50
    ensemble_members: int = 3
    max_epochs: int = 80
    early_stop: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    harm_margin: float = 0.05
    expected_cost_weight: float = 0.10
    harm_weight: float = 0.20
    specialist_mass_weight: float = 0.005
    validation_harm_penalty: float = 0.05


BETA_CANDIDATES = (0.0, 0.25, 0.50, 0.75, 1.0)


def _stack_expert_field(pool, key: str) -> torch.Tensor:
    return torch.stack(
        [pool["experts"][name][key].float() for name in SPECIALIST_NAMES],
        dim=1,
    )


def holdout_split_tensors(pool):
    predictions = _stack_expert_field(pool, "prediction")
    signatures = _stack_expert_field(pool, "signature")
    actions = stack_action_predictions(pool["anchor"].float(), predictions)
    context = global_context_features(
        pool["function_space"].float(), actions
    )
    labels = pool["labels"].float().view(-1, 1)
    return {
        "context": context,
        "signatures": signatures,
        "actions": actions,
        "labels": labels,
    }


def _action_counts(weights: torch.Tensor) -> Dict[str, int]:
    indices = weights.argmax(dim=1)
    return {
        name: int((indices == index).sum().item())
        for index, name in enumerate(ACTION_NAMES)
    }


def _harm_rate(
    prediction: torch.Tensor,
    anchor: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.10,
) -> float:
    gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
    return float((gain < -float(threshold)).float().mean().item())


def _mean_absolute_error(
    prediction: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    return float(torch.abs(prediction - labels).mean().item())


class HoldoutSoftGateTrainerV912:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        expert_pool_path,
        config: HoldoutSoftGateConfigV912,
        minimum_validation_gain: float = 0.0005,
        maximum_validation_harm: float = 0.05,
    ) -> None:
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.expert_pool_path = Path(expert_pool_path)
        self.pool = torch.load(self.expert_pool_path, map_location="cpu")
        self.config = config
        self.minimum_validation_gain = float(minimum_validation_gain)
        self.maximum_validation_harm = float(maximum_validation_harm)

        required_splits = {"router_train", "valid", "test"}
        if not required_splits.issubset(self.pool.get("splits", {})):
            raise ValueError(
                "V9.12 expert pool misses Router/Validation/Test splits"
            )
        provenance = self.pool.get("provenance", {})
        if not provenance.get("same_expert_models_router_valid_test", False):
            raise ValueError("V9.12 requires the same expert stack on all splits")
        if provenance.get("router_labels_used_for_expert_training", True):
            raise ValueError("Router labels leaked into V9.12 expert training")

        self.data = {
            split: holdout_split_tensors(self.pool["splits"][split])
            for split in ("router_train", "valid", "test")
        }
        self.context_dim = int(self.data["router_train"]["context"].size(1))
        self.checkpoints: List[Path] = []

    def new_model(self):
        return HoldoutSoftGateV912(
            context_dim=self.context_dim,
            signature_dim=SIGNATURE_DIM,
            hidden_dim=self.config.hidden_dim,
            action_embedding_dim=self.config.action_embedding_dim,
            dropout=self.config.dropout,
            temperature=self.config.temperature,
            anchor_bias=self.config.anchor_bias,
        ).to(self.args.device)

    def loader(self, split: str, shuffle: bool, seed: int):
        values = self.data[split]
        generator = (
            torch.Generator().manual_seed(int(seed)) if shuffle else None
        )
        return DataLoader(
            TensorDataset(
                values["context"],
                values["signatures"],
                values["actions"],
                values["labels"],
            ),
            batch_size=self.config.batch_size,
            shuffle=bool(shuffle),
            generator=generator,
            drop_last=False,
        )

    def _loss(self, output, actions, labels):
        return holdout_soft_gate_loss(
            output,
            actions,
            labels,
            harm_margin=self.config.harm_margin,
            expected_cost_weight=self.config.expected_cost_weight,
            harm_weight=self.config.harm_weight,
            specialist_mass_weight=self.config.specialist_mass_weight,
        )

    @torch.no_grad()
    def collect(self, model, split: str):
        values = self.data[split]
        model.eval()
        buffers = {
            "weights": [],
            "anchor_weight": [],
            "specialist_mass": [],
            "entropy": [],
        }
        for start in range(0, len(values["context"]), self.config.batch_size):
            output = model(
                values["context"][
                    start : start + self.config.batch_size
                ].to(self.args.device),
                values["signatures"][
                    start : start + self.config.batch_size
                ].to(self.args.device),
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        return {
            key: torch.cat(parts, dim=0)
            for key, parts in buffers.items()
        }

    def train_member(self, member: int):
        seed = int(self.args.seed) + 912700 + 7919 * (int(member) + 1)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model = self.new_model()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        train_loader = self.loader("router_train", True, seed)
        valid = self.data["valid"]
        best = None
        stale = 0
        history = []

        for epoch in range(1, self.config.max_epochs + 1):
            model.train()
            totals: Dict[str, float] = {}
            for context, signatures, actions, labels in train_loader:
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

            valid_output = self.collect(model, "valid")
            valid_prediction = soft_gate_prediction(
                valid_output,
                valid["actions"],
                beta=1.0,
            )["prediction"]
            valid_mae = _mean_absolute_error(
                valid_prediction, valid["labels"]
            )
            valid_harm = _harm_rate(
                valid_prediction,
                valid["actions"][:, 0],
                valid["labels"],
            )
            valid_mass = float(
                valid_output["specialist_mass"].mean().item()
            )
            objective = (
                valid_mae
                + self.config.validation_harm_penalty * valid_harm
                + 0.001 * valid_mass
            )
            row = {
                "member": int(member),
                "epoch": int(epoch),
                "valid_objective": objective,
                "valid_mae": valid_mae,
                "valid_harm_over_010_rate": valid_harm,
                "valid_specialist_mass": valid_mass,
                **{
                    f"train_{key}": value / max(1, len(train_loader))
                    for key, value in totals.items()
                },
            }
            history.append(row)
            if best is None or objective < best["objective"] - 1e-6:
                best = {
                    "objective": objective,
                    "epoch": int(epoch),
                    "state": cpu_state(model),
                    "valid_mae": valid_mae,
                    "valid_harm": valid_harm,
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.config.early_stop:
                break

        if best is None:
            raise RuntimeError(f"V9.12 member {member} selected no epoch")
        checkpoint = self.save_dir / f"holdout_soft_gate_member_{member}_v912.pth"
        torch.save(
            {
                "method": GATE_VERSION,
                "member": int(member),
                "seed": int(seed),
                "selected_epoch": int(best["epoch"]),
                "selected_valid_mae": float(best["valid_mae"]),
                "selected_valid_harm": float(best["valid_harm"]),
                "context_dim": self.context_dim,
                "signature_dim": SIGNATURE_DIM,
                "config": self.config.__dict__,
                "state_dict": best["state"],
            },
            checkpoint,
        )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return checkpoint, best, history

    @torch.no_grad()
    def collect_ensemble(self, split: str):
        if not self.checkpoints:
            raise RuntimeError("V9.12 gate checkpoints are unavailable")
        member_weights = []
        member_entropy = []
        for checkpoint in self.checkpoints:
            payload = torch.load(checkpoint, map_location="cpu")
            model = self.new_model()
            model.load_state_dict(payload["state_dict"], strict=True)
            output = self.collect(model, split)
            member_weights.append(output["weights"])
            member_entropy.append(output["entropy"])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        weight_stack = torch.stack(member_weights, dim=0)
        weights = weight_stack.mean(dim=0)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        disagreement = weight_stack.std(dim=0, unbiased=False).mean(
            dim=1, keepdim=True
        )
        entropy = -(
            weights.clamp_min(1e-8) * weights.clamp_min(1e-8).log()
        ).sum(dim=1, keepdim=True)
        return {
            "weights": weights,
            "anchor_weight": weights[:, :1],
            "specialist_mass": 1.0 - weights[:, :1],
            "entropy": entropy,
            "member_disagreement": disagreement,
            "mean_member_entropy": torch.stack(
                member_entropy, dim=0
            ).mean(dim=0),
        }

    def _candidate_rows(self, split: str, output):
        values = self.data[split]
        anchor = values["actions"][:, 0]
        labels = values["labels"]
        mixture = (
            output["weights"] * values["actions"].squeeze(-1)
        ).sum(dim=1, keepdim=True)
        anchor_mae = _mean_absolute_error(anchor, labels)
        rows = []
        predictions = {}
        for beta in BETA_CANDIDATES:
            prediction = anchor + float(beta) * (mixture - anchor)
            mae = _mean_absolute_error(prediction, labels)
            harm = _harm_rate(prediction, anchor, labels)
            objective = mae + self.config.validation_harm_penalty * harm
            eligible = (
                float(beta) == 0.0
                or (
                    anchor_mae - mae >= self.minimum_validation_gain
                    and harm <= self.maximum_validation_harm
                )
            )
            rows.append(
                {
                    "split": split,
                    "beta": float(beta),
                    "mae": mae,
                    "gain": anchor_mae - mae,
                    "harm_over_010_rate": harm,
                    "objective": objective,
                    "eligible": bool(eligible),
                }
            )
            predictions[float(beta)] = prediction
        return rows, predictions

    def _select_beta(self, valid_rows):
        eligible = [row for row in valid_rows if row["eligible"]]
        if not eligible:
            return 0.0
        best = min(
            eligible,
            key=lambda row: (
                row["objective"],
                row["mae"],
                row["beta"],
            ),
        )
        return float(best["beta"])

    def _upper_bounds(self, split: str):
        values = self.data[split]
        actions = values["actions"]
        labels = values["labels"]
        regions = region_index(labels)
        mapping = torch.tensor(
            SEMANTIC_REGION_ACTION_MAP, dtype=torch.long
        )
        region_actions = mapping[regions]
        true_region = actions.squeeze(-1).gather(
            1, region_actions.view(-1, 1)
        )
        costs = torch.abs(actions.squeeze(-1) - labels)
        oracle_index = costs.argmin(dim=1)
        oracle = actions.squeeze(-1).gather(
            1, oracle_index.view(-1, 1)
        )
        return {
            "true_region_semantic_upper_bound": true_region,
            "sample_oracle_upper_bound": oracle,
            "oracle_index": oracle_index,
        }

    def _metrics_row(self, name, prediction, labels, anchor):
        metrics = safe_metrics(self.metrics_fn, prediction, labels)
        return {
            "model": name,
            **metrics,
            "harm_over_010_rate": _harm_rate(
                prediction, anchor, labels
            ),
        }

    def train_all(self):
        history_rows = []
        member_rows = []
        self.checkpoints = []
        for member in range(self.config.ensemble_members):
            checkpoint, best, history = self.train_member(member)
            self.checkpoints.append(checkpoint)
            history_rows.extend(history)
            member_rows.append(
                {
                    "member": int(member),
                    "checkpoint": str(checkpoint),
                    "selected_epoch": int(best["epoch"]),
                    "selected_valid_mae": float(best["valid_mae"]),
                    "selected_valid_harm": float(best["valid_harm"]),
                }
            )
        pd.DataFrame(history_rows).to_csv(
            self.save_dir / "v912_gate_training_history.csv",
            index=False,
        )
        pd.DataFrame(member_rows).to_csv(
            self.save_dir / "v912_gate_ensemble.csv",
            index=False,
        )

        outputs = {
            split: self.collect_ensemble(split)
            for split in ("router_train", "valid", "test")
        }
        router_rows, router_predictions = self._candidate_rows(
            "router_train", outputs["router_train"]
        )
        valid_rows, valid_predictions = self._candidate_rows(
            "valid", outputs["valid"]
        )
        selected_beta = self._select_beta(valid_rows)
        test_rows, test_predictions = self._candidate_rows(
            "test", outputs["test"]
        )
        for row in valid_rows:
            row["selected"] = bool(row["beta"] == selected_beta)
        for row in test_rows:
            row["selected"] = bool(row["beta"] == selected_beta)

        pd.DataFrame(router_rows).to_csv(
            self.save_dir / "v912_router_train_beta_diagnostics.csv",
            index=False,
        )
        pd.DataFrame(valid_rows).to_csv(
            self.save_dir / "v912_validation_beta_selection.csv",
            index=False,
        )
        pd.DataFrame(test_rows).to_csv(
            self.save_dir / "v912_test_beta_diagnostics.csv",
            index=False,
        )

        test = self.data["test"]
        test_anchor = test["actions"][:, 0]
        test_labels = test["labels"]
        upper = self._upper_bounds("test")
        named = {
            "anchor": test_anchor,
            "uniform_all_actions": test["actions"].mean(dim=1),
            "holdout_soft_gate_beta_1": test_predictions[1.0],
            "holdout_soft_gate_valid_selected": test_predictions[
                selected_beta
            ],
            "true_region_semantic_upper_bound": upper[
                "true_region_semantic_upper_bound"
            ],
            "sample_oracle_upper_bound": upper[
                "sample_oracle_upper_bound"
            ],
        }
        for index, name in enumerate(ACTION_NAMES[1:], start=1):
            named[name] = test["actions"][:, index]

        test_metric_rows = [
            self._metrics_row(
                name,
                prediction,
                test_labels,
                test_anchor,
            )
            for name, prediction in named.items()
        ]
        pd.DataFrame(test_metric_rows).to_csv(
            self.save_dir / "v912_test_summary.csv",
            index=False,
        )

        test_output = outputs["test"]
        frame: Dict[str, object] = {
            "sample_id": self.pool["splits"]["test"]["sample_ids"],
            "label": test_labels.view(-1).tolist(),
            "anchor": test_anchor.view(-1).tolist(),
            "soft_gate_beta_1": test_predictions[1.0].view(-1).tolist(),
            "soft_gate_valid_selected": test_predictions[
                selected_beta
            ].view(-1).tolist(),
            "selected_beta": [selected_beta] * len(test_labels),
            "weight_entropy": test_output["entropy"].view(-1).tolist(),
            "member_disagreement": test_output[
                "member_disagreement"
            ].view(-1).tolist(),
        }
        for index, name in enumerate(ACTION_NAMES):
            frame[f"{name}_weight"] = test_output["weights"][
                :, index
            ].tolist()
            frame[f"{name}_prediction"] = test["actions"][
                :, index, 0
            ].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "holdout_soft_gate_v912_predictions.csv",
            index=False,
        )

        router_anchor = self.data["router_train"]["actions"][:, 0]
        router_labels = self.data["router_train"]["labels"]
        valid_anchor = self.data["valid"]["actions"][:, 0]
        valid_labels = self.data["valid"]["labels"]
        selected_valid = valid_predictions[selected_beta]
        selected_test = test_predictions[selected_beta]

        summary = {
            "method": "holdout_soft_gating_v9_12",
            "gate_version": GATE_VERSION,
            "signature_version": SIGNATURE_VERSION,
            "signature_fields": list(SIGNATURE_FIELDS),
            "expert_pool": str(self.expert_pool_path),
            "expert_pool_method": self.pool.get("method"),
            "source_outer_fold": int(self.pool["outer_fold"]),
            "expert_train_count": int(self.pool["expert_train_count"]),
            "router_train_count": int(self.pool["router_train_count"]),
            "same_expert_models_router_valid_test": True,
            "deep_nested_coach_oof": False,
            "gate_ensemble_checkpoints": [
                str(path) for path in self.checkpoints
            ],
            "gate_selected_epochs": [
                int(row["selected_epoch"]) for row in member_rows
            ],
            "selected_beta": selected_beta,
            "selected_by_validation_only": True,
            "router_train_fit_anchor_mae": _mean_absolute_error(
                router_anchor, router_labels
            ),
            "router_train_fit_selected_mae": _mean_absolute_error(
                router_predictions[selected_beta], router_labels
            ),
            "validation_anchor_mae": _mean_absolute_error(
                valid_anchor, valid_labels
            ),
            "validation_selected_mae": _mean_absolute_error(
                selected_valid, valid_labels
            ),
            "validation_selected_gain": (
                _mean_absolute_error(valid_anchor, valid_labels)
                - _mean_absolute_error(selected_valid, valid_labels)
            ),
            "validation_selected_harm_over_010_rate": _harm_rate(
                selected_valid, valid_anchor, valid_labels
            ),
            "test_anchor_mae": _mean_absolute_error(
                test_anchor, test_labels
            ),
            "test_selected_mae": _mean_absolute_error(
                selected_test, test_labels
            ),
            "test_selected_gain": (
                _mean_absolute_error(test_anchor, test_labels)
                - _mean_absolute_error(selected_test, test_labels)
            ),
            "test_selected_harm_over_010_rate": _harm_rate(
                selected_test, test_anchor, test_labels
            ),
            "test_mean_anchor_weight": float(
                test_output["anchor_weight"].mean().item()
            ),
            "test_mean_specialist_mass": float(
                test_output["specialist_mass"].mean().item()
            ),
            "test_action_counts": _action_counts(
                test_output["weights"]
            ),
            "test_results": {
                row["model"]: {
                    key: value
                    for key, value in row.items()
                    if key != "model"
                }
                for row in test_metric_rows
            },
            "provenance": {
                **self.pool.get("provenance", {}),
                "router_labels_used_for_gate_training": True,
                "validation_labels_used_for_early_stop_and_beta": True,
                "test_labels_used_for_training_or_selection": False,
                "router_train_metrics_are_in_sample_for_gate": True,
                "validation_is_the_primary_model_selection_split": True,
            },
        }
        (self.save_dir / "holdout_soft_gate_v912_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return summary
