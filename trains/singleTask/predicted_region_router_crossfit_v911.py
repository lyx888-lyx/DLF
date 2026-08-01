"""Grouped OOF region classification and fold-ensemble inference for V9.11."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset

from .coach_utils_v93 import cpu_state, safe_group_folds
from .model.PredictedRegionRouterV911 import (
    REGION_NAMES,
    REGION_ROUTER_VERSION,
    SEMANTIC_REGION_ACTION_MAP,
    PredictedRegionRouterV911,
    predicted_region_router_loss,
    region_classification_metrics,
    region_index,
    region_router_features,
    route_by_regions,
)
from .model.SemanticCostCoachV99 import (
    ACTION_NAMES,
    SIGNATURE_DIM,
    SIGNATURE_FIELDS,
    SIGNATURE_VERSION,
    SPECIALIST_NAMES,
    stack_action_predictions,
)


@dataclass(frozen=True)
class PredictedRegionRouterConfigV911:
    hidden_dim: int = 128
    dropout: float = 0.20
    residual_max: float = 2.0
    ordinal_temperature: float = 0.45
    ordinal_blend: float = 0.50
    folds: int = 5
    max_epochs: int = 60
    early_stop: int = 10
    learning_rate: float = 3e-4
    weight_decay: float = 1e-3
    batch_size: int = 64
    empirical_prior_strength: float = 20.0


def _stack_expert_field(pool: Mapping[str, object], key: str) -> torch.Tensor:
    return torch.stack(
        [pool["experts"][name][key].float() for name in SPECIALIST_NAMES],
        dim=1,
    )


def pool_action_tensors(pool: Mapping[str, object]):
    """Normalize strict OOF and frozen Validation/Test pool layouts."""
    if "expert_predictions" in pool and "expert_signatures" in pool:
        expert_predictions = pool["expert_predictions"].float()
        expert_signatures = pool["expert_signatures"].float()
    elif "experts" in pool:
        expert_predictions = _stack_expert_field(pool, "prediction")
        expert_signatures = _stack_expert_field(pool, "signature")
    else:
        raise KeyError(
            "semantic pool must contain expert_predictions/expert_signatures "
            "or experts[name]"
        )
    actions = stack_action_predictions(pool["anchor"].float(), expert_predictions)
    features = region_router_features(
        pool["function_space"].float(), actions, expert_signatures
    )
    return features, actions, expert_signatures


def _action_counts(indices: torch.Tensor) -> Dict[str, int]:
    return {
        name: int((indices.view(-1) == index).sum().item())
        for index, name in enumerate(ACTION_NAMES)
    }


def estimate_empirical_region_map(
    actions: torch.Tensor,
    labels: torch.Tensor,
    indices: Iterable[int],
    prior_strength: float = 20.0,
) -> Tuple[Tuple[int, ...], List[Dict[str, object]]]:
    index = torch.as_tensor(list(indices), dtype=torch.long)
    values = actions[index].squeeze(-1)
    targets = labels[index].view(-1, 1).to(values)
    costs = torch.abs(values - targets)
    regions = region_index(labels[index])
    global_mean = costs.mean(dim=0)
    mapping: List[int] = []
    rows: List[Dict[str, object]] = []
    for region, region_name in enumerate(REGION_NAMES):
        mask = regions == region
        count = int(mask.sum().item())
        if count == 0:
            selected = int(SEMANTIC_REGION_ACTION_MAP[region])
            shrunk = global_mean
            raw = global_mean
        else:
            raw = costs[mask].mean(dim=0)
            shrunk = (
                costs[mask].sum(dim=0)
                + float(prior_strength) * global_mean
            ) / (count + float(prior_strength))
            selected = int(shrunk.argmin().item())
        mapping.append(selected)
        row: Dict[str, object] = {
            "region_index": region,
            "region": region_name,
            "sample_count": count,
            "selected_action_index": selected,
            "selected_action": ACTION_NAMES[selected],
            "semantic_action": ACTION_NAMES[SEMANTIC_REGION_ACTION_MAP[region]],
        }
        for action_index, action_name in enumerate(ACTION_NAMES):
            row[f"raw_mae_{action_name}"] = float(raw[action_index].item())
            row[f"shrunk_mae_{action_name}"] = float(shrunk[action_index].item())
        rows.append(row)
    return tuple(mapping), rows


class PredictedRegionRouterCrossFitterV911:
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
                f"V9.11 strict semantic pool missing {sorted(required - set(self.pool))}"
            )
        if self.pool.get("signature_version") != SIGNATURE_VERSION:
            raise ValueError("V9.11 semantic signature version mismatch")
        if tuple(self.pool.get("signature_fields", ())) != SIGNATURE_FIELDS:
            raise ValueError("V9.11 semantic signature fields mismatch")
        self.features, self.actions, self.signatures = pool_action_tensors(self.pool)
        self.anchor = self.pool["anchor"].float().view(-1, 1)
        self.labels = self.pool["labels"].float().view(-1, 1)
        self.ensemble_checkpoints: List[Path] = []

    def new_model(self):
        return PredictedRegionRouterV911(
            input_dim=self.features.size(1),
            hidden_dim=self.config.hidden_dim,
            dropout=self.config.dropout,
            residual_max=self.config.residual_max,
            ordinal_temperature=self.config.ordinal_temperature,
            ordinal_blend=self.config.ordinal_blend,
        ).to(self.args.device)

    def loader(self, indices: Iterable[int], shuffle: bool, seed: int):
        index = torch.as_tensor(list(indices), dtype=torch.long)
        generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
        return DataLoader(
            TensorDataset(
                self.features[index],
                self.anchor[index],
                self.labels[index],
            ),
            batch_size=self.config.batch_size,
            shuffle=bool(shuffle),
            generator=generator,
            drop_last=False,
        )

    @torch.no_grad()
    def collect(self, model, features, anchor):
        model.eval()
        buffers = {
            "score": [],
            "ordinal_logits": [],
            "class_logits": [],
            "region_probs": [],
            "confidence": [],
        }
        for start in range(0, len(features), self.config.batch_size):
            output = model(
                features[start : start + self.config.batch_size].to(self.args.device),
                anchor[start : start + self.config.batch_size].to(self.args.device),
            )
            for key in buffers:
                buffers[key].append(output[key].detach().cpu())
        return {key: torch.cat(value, dim=0) for key, value in buffers.items()}

    def _class_weights(self, labels: torch.Tensor) -> torch.Tensor:
        target = region_index(labels)
        counts = torch.bincount(target, minlength=len(REGION_NAMES)).float()
        weights = counts.sum() / counts.clamp_min(1.0)
        return weights / weights.mean()

    def _loss(self, output, labels, class_weights):
        return predicted_region_router_loss(
            output,
            labels,
            class_weights=class_weights,
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
        train_loader = self.loader(train_indices, True, seed)
        train_index = torch.as_tensor(train_indices, dtype=torch.long)
        valid_index = torch.as_tensor(valid_indices, dtype=torch.long)
        class_weights = self._class_weights(self.labels[train_index]).to(self.args.device)
        best = None
        stale = 0
        history = []
        for epoch in range(1, self.config.max_epochs + 1):
            model.train()
            totals: Dict[str, float] = {}
            for features, anchor, labels in train_loader:
                output = model(
                    features.to(self.args.device),
                    anchor.to(self.args.device),
                )
                losses = predicted_region_router_loss(
                    output,
                    labels.to(self.args.device),
                    class_weights=class_weights,
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().item())

            valid = self.collect(
                model,
                self.features[valid_index],
                self.anchor[valid_index],
            )
            valid_loss = self._loss(
                valid,
                self.labels[valid_index],
                self._class_weights(self.labels[train_index]),
            )
            predicted_regions = valid["region_probs"].argmax(dim=1)
            route = route_by_regions(
                self.actions[valid_index],
                predicted_regions,
                SEMANTIC_REGION_ACTION_MAP,
            )
            route_mae = float(
                torch.abs(route["prediction"] - self.labels[valid_index]).mean().item()
            )
            metrics = region_classification_metrics(
                valid["region_probs"], self.labels[valid_index]
            )
            objective = (
                float(valid_loss["total"].item())
                + 0.40 * route_mae
                + 0.10 * metrics["ordinal_index_mae"]
            )
            row = {
                "epoch": epoch,
                "valid_objective": objective,
                "valid_semantic_route_mae": route_mae,
                **{f"valid_{key}": value for key, value in metrics.items()},
                **{
                    f"train_{key}": value / max(1, len(train_loader))
                    for key, value in totals.items()
                },
                **{
                    f"valid_loss_{key}": float(value.item())
                    for key, value in valid_loss.items()
                },
            }
            history.append(row)
            if best is None or objective < best["objective"] - 1e-6:
                best = {
                    "objective": objective,
                    "epoch": epoch,
                    "state": cpu_state(model),
                    "route_mae": route_mae,
                    "metrics": metrics,
                }
                stale = 0
            else:
                stale += 1
            if stale >= self.config.early_stop:
                break
        if best is None:
            raise RuntimeError("V9.11 region fold selected no epoch")
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
        index = torch.as_tensor(indices, dtype=torch.long)
        loader = self.loader(indices, True, seed)
        class_weights = self._class_weights(self.labels[index]).to(self.args.device)
        history = []
        for epoch in range(1, int(epochs) + 1):
            model.train()
            totals: Dict[str, float] = {}
            for features, anchor, labels in loader:
                output = model(
                    features.to(self.args.device),
                    anchor.to(self.args.device),
                )
                losses = predicted_region_router_loss(
                    output,
                    labels.to(self.args.device),
                    class_weights=class_weights,
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
    def collect_ensemble(self, pool):
        if not self.ensemble_checkpoints:
            raise RuntimeError("V9.11 ensemble checkpoints are unavailable")
        features, actions, signatures = pool_action_tensors(pool)
        anchor = pool["anchor"].float().view(-1, 1)
        probabilities, scores, confidences = [], [], []
        for checkpoint in self.ensemble_checkpoints:
            payload = torch.load(checkpoint, map_location="cpu")
            model = self.new_model()
            model.load_state_dict(payload["state_dict"], strict=True)
            output = self.collect(model, features, anchor)
            probabilities.append(output["region_probs"])
            scores.append(output["score"])
            confidences.append(output["confidence"])
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        probability_stack = torch.stack(probabilities, dim=0)
        score_stack = torch.stack(scores, dim=0)
        confidence_stack = torch.stack(confidences, dim=0)
        mean_probability = probability_stack.mean(dim=0)
        mean_probability = mean_probability / mean_probability.sum(
            dim=1, keepdim=True
        ).clamp_min(1e-8)
        return {
            "features": features,
            "actions": actions,
            "signatures": signatures,
            "score": score_stack.mean(dim=0),
            "region_probs": mean_probability,
            "confidence": confidence_stack.mean(dim=0),
            "ensemble_probability_std": probability_stack.std(
                dim=0, unbiased=False
            ).mean(dim=1, keepdim=True),
        }

    def fit(self):
        specs, manifest = safe_group_folds(
            self.pool["sample_ids"],
            self.labels.view(-1).tolist(),
            self.config.folds,
            int(self.args.seed) + 911001,
        )
        manifest.to_csv(
            self.save_dir / "v911_region_group_manifest.csv", index=False
        )
        n = len(self.labels)
        probabilities = torch.full((n, len(REGION_NAMES)), float("nan"))
        scores = torch.full((n, 1), float("nan"))
        confidences = torch.full((n, 1), float("nan"))
        coach_fold_index = torch.full((n,), -1, dtype=torch.long)
        semantic_actions = torch.full((n,), -1, dtype=torch.long)
        empirical_actions = torch.full((n,), -1, dtype=torch.long)
        best_epochs: List[int] = []
        selection_history: List[Dict[str, object]] = []
        refit_history: List[Dict[str, object]] = []
        map_rows: List[Dict[str, object]] = []
        checkpoint_rows: List[Dict[str, object]] = []
        self.ensemble_checkpoints = []

        for spec in specs:
            fold = int(spec.outer_fold)
            selection_seed = int(self.args.seed) + 21101 * (fold + 1)
            best, history = self.train_for_selection(
                spec.inner_train_indices,
                spec.inner_valid_indices,
                selection_seed,
            )
            best_epochs.append(int(best["epoch"]))
            selection_history.extend({"fold": fold, **row} for row in history)
            development = sorted(
                set(spec.inner_train_indices) | set(spec.inner_valid_indices)
            )
            empirical_map, empirical_rows = estimate_empirical_region_map(
                self.actions,
                self.labels,
                development,
                self.config.empirical_prior_strength,
            )
            map_rows.extend(
                {
                    "scope": f"coach_fold_{fold}_development",
                    **row,
                }
                for row in empirical_rows
            )
            refit_seed = int(self.args.seed) + 23003 * (fold + 1)
            model, fixed_history = self.fit_fixed_epochs(
                development,
                int(best["epoch"]),
                refit_seed,
            )
            refit_history.extend({"fold": fold, **row} for row in fixed_history)
            holdout = torch.as_tensor(spec.outer_holdout_indices, dtype=torch.long)
            output = self.collect(
                model,
                self.features[holdout],
                self.anchor[holdout],
            )
            probabilities[holdout] = output["region_probs"]
            scores[holdout] = output["score"]
            confidences[holdout] = output["confidence"]
            coach_fold_index[holdout] = fold
            predicted_region = output["region_probs"].argmax(dim=1)
            semantic_actions[holdout] = route_by_regions(
                self.actions[holdout],
                predicted_region,
                SEMANTIC_REGION_ACTION_MAP,
            )["action_indices"]
            empirical_actions[holdout] = route_by_regions(
                self.actions[holdout],
                predicted_region,
                empirical_map,
            )["action_indices"]
            checkpoint = self.save_dir / f"predicted_region_fold_{fold}_v911.pth"
            torch.save(
                {
                    "method": REGION_ROUTER_VERSION,
                    "fold": fold,
                    "feature_dim": int(self.features.size(1)),
                    "signature_version": SIGNATURE_VERSION,
                    "config": self.config.__dict__,
                    "selected_epoch": int(best["epoch"]),
                    "development_count": len(development),
                    "holdout_count": len(holdout),
                    "empirical_region_action_map": list(empirical_map),
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
                    "empirical_region_action_map": json.dumps(
                        [ACTION_NAMES[index] for index in empirical_map]
                    ),
                }
            )
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if not torch.isfinite(probabilities).all() or (coach_fold_index < 0).any():
            raise FloatingPointError("V9.11 region OOF predictions are incomplete")
        pd.DataFrame(selection_history).to_csv(
            self.save_dir / "v911_region_selection_history.csv", index=False
        )
        pd.DataFrame(refit_history).to_csv(
            self.save_dir / "v911_region_refit_history.csv", index=False
        )
        pd.DataFrame(checkpoint_rows).to_csv(
            self.save_dir / "v911_region_ensemble.csv", index=False
        )

        final_empirical_map, final_map_rows = estimate_empirical_region_map(
            self.actions,
            self.labels,
            range(n),
            self.config.empirical_prior_strength,
        )
        map_rows.extend({"scope": "full_train", **row} for row in final_map_rows)
        pd.DataFrame(map_rows).to_csv(
            self.save_dir / "v911_region_action_maps.csv", index=False
        )

        true_regions = region_index(self.labels)
        predicted_regions = probabilities.argmax(dim=1)
        semantic_prediction = self.actions.squeeze(-1).gather(
            1, semantic_actions.view(-1, 1)
        )
        empirical_prediction = self.actions.squeeze(-1).gather(
            1, empirical_actions.view(-1, 1)
        )
        true_region_route = route_by_regions(
            self.actions, true_regions, SEMANTIC_REGION_ACTION_MAP
        )["prediction"]
        anchor_regions = region_index(self.anchor)
        anchor_threshold_route = route_by_regions(
            self.actions, anchor_regions, SEMANTIC_REGION_ACTION_MAP
        )["prediction"]
        costs = torch.abs(
            self.actions.squeeze(-1) - self.labels.view(-1, 1)
        )
        oracle_indices = costs.argmin(dim=1)
        oracle_prediction = self.actions.squeeze(-1).gather(
            1, oracle_indices.view(-1, 1)
        )

        max_probability, _ = probabilities.max(dim=1)
        sorted_probability = probabilities.sort(dim=1, descending=True).values
        probability_margin = sorted_probability[:, 0] - sorted_probability[:, 1]
        entropy = -(
            probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()
        ).sum(dim=1)
        frame: Dict[str, object] = {
            "sample_id": self.pool["sample_ids"],
            "group_id": self.pool["group_ids"],
            "coach_fold": coach_fold_index.tolist(),
            "label": self.labels.view(-1).tolist(),
            "anchor": self.anchor.view(-1).tolist(),
            "true_region": [REGION_NAMES[index] for index in true_regions.tolist()],
            "predicted_region": [
                REGION_NAMES[index] for index in predicted_regions.tolist()
            ],
            "ordinal_score": scores.view(-1).tolist(),
            "model_confidence": confidences.view(-1).tolist(),
            "max_region_probability": max_probability.tolist(),
            "region_probability_margin": probability_margin.tolist(),
            "region_entropy": entropy.tolist(),
            "semantic_action": [
                ACTION_NAMES[index] for index in semantic_actions.tolist()
            ],
            "empirical_action": [
                ACTION_NAMES[index] for index in empirical_actions.tolist()
            ],
            "semantic_prediction": semantic_prediction.view(-1).tolist(),
            "empirical_prediction": empirical_prediction.view(-1).tolist(),
        }
        for index, name in enumerate(REGION_NAMES):
            frame[f"p_{name}"] = probabilities[:, index].tolist()
        pd.DataFrame(frame).to_csv(
            self.save_dir / "v911_region_oof_predictions.csv", index=False
        )
        confusion = pd.crosstab(
            pd.Series(frame["true_region"], name="true_region"),
            pd.Series(frame["predicted_region"], name="predicted_region"),
            dropna=False,
        ).reindex(index=REGION_NAMES, columns=REGION_NAMES, fill_value=0)
        confusion.to_csv(self.save_dir / "v911_region_oof_confusion.csv")

        metrics = region_classification_metrics(probabilities, self.labels)
        summary = {
            "method": REGION_ROUTER_VERSION,
            "checkpoint_count": len(self.ensemble_checkpoints),
            "ensemble_checkpoints": [str(path) for path in self.ensemble_checkpoints],
            "crossfit_best_epochs": best_epochs,
            "region_metrics": metrics,
            "semantic_region_action_map": [
                ACTION_NAMES[index] for index in SEMANTIC_REGION_ACTION_MAP
            ],
            "empirical_region_action_map": [
                ACTION_NAMES[index] for index in final_empirical_map
            ],
            "oof_anchor_mae": float(
                torch.abs(self.anchor - self.labels).mean().item()
            ),
            "oof_predicted_semantic_route_mae": float(
                torch.abs(semantic_prediction - self.labels).mean().item()
            ),
            "oof_predicted_empirical_route_mae": float(
                torch.abs(empirical_prediction - self.labels).mean().item()
            ),
            "oof_anchor_threshold_semantic_route_mae": float(
                torch.abs(anchor_threshold_route - self.labels).mean().item()
            ),
            "oof_true_region_semantic_route_mae": float(
                torch.abs(true_region_route - self.labels).mean().item()
            ),
            "oof_sample_oracle_mae": float(
                torch.abs(oracle_prediction - self.labels).mean().item()
            ),
            "oof_predicted_region_counts": {
                name: int((predicted_regions == index).sum().item())
                for index, name in enumerate(REGION_NAMES)
            },
            "oof_semantic_action_counts": _action_counts(semantic_actions),
            "oof_empirical_action_counts": _action_counts(empirical_actions),
            "signature_version": SIGNATURE_VERSION,
            "pool_version": self.pool.get("version"),
            "provenance": self.pool.get("provenance"),
        }
        return {
            "region_probs": probabilities,
            "score": scores,
            "confidence": confidences,
            "coach_fold_index": coach_fold_index,
            "semantic_actions": semantic_actions,
            "empirical_actions": empirical_actions,
            "final_empirical_map": final_empirical_map,
            "summary": summary,
        }
