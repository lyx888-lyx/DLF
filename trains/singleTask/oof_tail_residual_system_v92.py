"""Train and evaluate V9.2 tail experts from grouped OOF CFCompatKD caches."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

from .cfcompat_fold_training_v92 import MissingModalityWrapper, mode_to_mask
from .oof_group_splits_v92 import canonical_sample_id
from .model.CachedTailResidualHeadV92 import CachedTailResidualHeadV92
from .model.DLF import DLF
from .model.FSC_DLF import _FusionFeatureCapture, load_dlf_checkpoint
from .oof_tail_residual_v92 import (
    MECHANISM_NAMES,
    TAIL_ROLE_NAMES,
    OOFTailLossWeights,
    direction_agreement_summary,
    exact_tail_mask,
    oof_tail_residual_loss,
    residual_diagnostic_rows,
    residual_mechanism_index,
    tail_capability_metrics,
)

logger = logging.getLogger("MMSA")


def _extract_state(payload):
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model", "student"):
            if isinstance(payload.get(key), dict):
                payload = payload[key]
                break
    if not isinstance(payload, dict):
        raise TypeError("checkpoint does not contain a state dictionary")
    result = {}
    for key, value in payload.items():
        name = str(key)
        if name.startswith("module."):
            name = name[len("module.") :]
        result[name] = value
    return result


def load_cfcompat_checkpoint(model: MissingModalityWrapper, path, device):
    state = _extract_state(torch.load(path, map_location=device))
    try:
        return model.load_state_dict(state, strict=True), "wrapper_strict"
    except RuntimeError:
        incompatible = load_dlf_checkpoint(model.backbone, path, map_location=device)
        return incompatible, "backbone_compatible"


@torch.no_grad()
def collect_cfcompat_features(model, dataloader, device):
    model.eval()
    sample_count = len(dataloader.dataset)
    rows = []
    for batch in dataloader:
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["labels"]["M"].to(device).view(-1, 1)
        mask = mode_to_mask("LAV", labels.size(0), device, audio.dtype)
        capture = _FusionFeatureCapture(model.backbone)
        try:
            output = model(text, audio, vision, mask)
            if capture.value is None:
                raise RuntimeError("failed to capture CFCompatKD fusion feature")
            feature = capture.value.detach().cpu()
        finally:
            capture.close()
        indices = batch["index"].view(-1).cpu().tolist()
        ids = [canonical_sample_id(value) for value in list(batch["id"])]
        for offset, index in enumerate(indices):
            rows.append(
                {
                    "sample_index": int(index),
                    "sample_id": ids[offset],
                    "label": float(labels[offset].item()),
                    "prediction": float(output["output_logit"][offset].item()),
                    "feature": feature[offset].clone(),
                }
            )
    rows.sort(key=lambda row: row["sample_index"])
    if len(rows) != sample_count:
        raise RuntimeError("feature collection sample count mismatch")
    if [row["sample_index"] for row in rows] != list(range(sample_count)):
        raise RuntimeError("feature collection indices are not contiguous")
    return {
        "sample_ids": [row["sample_id"] for row in rows],
        "labels": torch.tensor([row["label"] for row in rows]).view(-1, 1),
        "anchor": torch.tensor([row["prediction"] for row in rows]).view(-1, 1),
        "feature": torch.stack([row["feature"] for row in rows], dim=0),
    }


def _safe_metrics(metrics_fn, prediction, labels):
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


class OOFTailResidualTrainerV92:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        oof_cache_path,
        teacher_paths: Sequence[Path],
        valid_loader,
        test_loader,
        hidden_dim: int = 96,
        dropout: float = 0.15,
        residual_max: float = 1.50,
        max_epochs: int = 40,
        early_stop: int = 8,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-3,
        membership_temperature: float = 0.25,
        gain_margin: float = 0.08,
        gain_fraction: float = 0.20,
        global_tolerance: float = 0.012,
        max_harm_rate: float = 0.15,
        batch_size: int = 64,
        loss_weights: OOFTailLossWeights = OOFTailLossWeights(),
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.teacher_paths = [Path(value) for value in teacher_paths]
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.residual_max = float(residual_max)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.membership_temperature = float(membership_temperature)
        self.gain_margin = float(gain_margin)
        self.gain_fraction = float(gain_fraction)
        self.global_tolerance = float(global_tolerance)
        self.max_harm_rate = float(max_harm_rate)
        self.batch_size = int(batch_size)
        self.loss_weights = loss_weights

        self.oof = torch.load(oof_cache_path, map_location="cpu")
        required = {
            "sample_ids",
            "labels",
            "oof_prediction",
            "oof_feature",
            "fold_index",
        }
        if not required.issubset(self.oof):
            raise ValueError("OOF cache is missing required keys")
        if not torch.isfinite(self.oof["oof_prediction"]).all():
            raise ValueError("OOF predictions are non-finite")
        if not torch.isfinite(self.oof["oof_feature"]).all():
            raise ValueError("OOF features are non-finite")

        self.anchor_index, self.valid, self.test, self.anchor_load_mode = (
            self._select_and_collect_anchor(valid_loader, test_loader)
        )
        if self.oof["oof_feature"].size(1) != self.valid["feature"].size(1):
            raise RuntimeError("OOF and full-anchor feature dimensions differ")
        self.feature_dim = int(self.oof["oof_feature"].size(1))
        self._write_preflight_diagnostics()

    def _new_wrapper(self):
        backbone = DLF(self.args).to(self.args.device)
        return MissingModalityWrapper(
            backbone,
            int(self.args.feature_dims[1]),
            int(self.args.feature_dims[2]),
        ).to(self.args.device)

    def _select_and_collect_anchor(self, valid_loader, test_loader):
        candidates = []
        for index, path in enumerate(self.teacher_paths):
            model = self._new_wrapper()
            _, mode = load_cfcompat_checkpoint(model, path, self.args.device)
            valid = collect_cfcompat_features(model, valid_loader, self.args.device)
            mae = float(torch.abs(valid["anchor"] - valid["labels"]).mean().item())
            candidates.append((mae, index, valid, mode))
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        _, anchor_index, valid, load_mode = min(candidates, key=lambda row: row[0])
        model = self._new_wrapper()
        load_cfcompat_checkpoint(
            model, self.teacher_paths[anchor_index], self.args.device
        )
        test = collect_cfcompat_features(model, test_loader, self.args.device)
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(
            "V9.2 full anchor index=%d valid_mae=%.6f path=%s mode=%s",
            anchor_index,
            float(torch.abs(valid["anchor"] - valid["labels"]).mean().item()),
            self.teacher_paths[anchor_index],
            load_mode,
        )
        return anchor_index, valid, test, load_mode

    def _write_preflight_diagnostics(self):
        rows = []
        rows.extend(
            residual_diagnostic_rows(
                self.oof["oof_prediction"],
                self.oof["labels"],
                "oof_train",
                self.oof["fold_index"],
            )
        )
        rows.extend(
            residual_diagnostic_rows(
                self.valid["anchor"], self.valid["labels"], "valid"
            )
        )
        rows.extend(
            residual_diagnostic_rows(
                self.test["anchor"], self.test["labels"], "test"
            )
        )
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v92_oof_anchor_residual_diagnostics.csv",
            index=False,
        )
        direction = direction_agreement_summary(
            self.oof["oof_prediction"],
            self.oof["labels"],
            self.valid["anchor"],
            self.valid["labels"],
        )
        pd.DataFrame(direction).to_csv(
            self.save_dir / "v92_oof_valid_direction_agreement.csv",
            index=False,
        )

    def _new_head(self):
        return CachedTailResidualHeadV92(
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            residual_max=self.residual_max,
        ).to(self.args.device)

    def _train_loader(self, role_seed):
        dataset = TensorDataset(
            self.oof["oof_feature"].float(),
            self.oof["oof_prediction"].float(),
            self.oof["labels"].float(),
            self.oof["fold_index"].long(),
        )
        generator = torch.Generator().manual_seed(int(role_seed))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
            drop_last=False,
        )

    @torch.no_grad()
    def _collect_head(self, model, feature, anchor, labels):
        model.eval()
        outputs = {
            "prediction": [],
            "correction": [],
            "raw_correction": [],
            "applicability_prob": [],
            "mechanism_probs": [],
        }
        for start in range(0, feature.size(0), self.batch_size):
            end = min(start + self.batch_size, feature.size(0))
            batch = model(
                feature[start:end].to(self.args.device),
                anchor[start:end].to(self.args.device),
            )
            for key in outputs:
                outputs[key].append(batch[key].detach().cpu())
        return {
            **{key: torch.cat(values, dim=0) for key, values in outputs.items()},
            "anchor": anchor.detach().cpu().clone(),
            "labels": labels.detach().cpu().clone(),
        }

    def _validation_row(self, collected, role):
        labels = collected["labels"]
        anchor = collected["anchor"]
        prediction = collected["prediction"]
        mask = exact_tail_mask(labels, role)
        anchor_global = float(torch.abs(anchor - labels).mean().item())
        global_mae = float(torch.abs(prediction - labels).mean().item())
        anchor_tail = float(torch.abs(anchor[mask] - labels[mask]).mean().item())
        tail_mae = float(torch.abs(prediction[mask] - labels[mask]).mean().item())
        gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
        harm = float((gain < -0.10).float().mean().item())
        global_delta = global_mae - anchor_global
        tail_gain = anchor_tail - tail_mae
        objective = (
            0.80 * tail_mae
            + 0.20 * global_mae
            + 3.0 * max(0.0, global_delta - self.global_tolerance)
            + 0.20 * max(0.0, harm - self.max_harm_rate)
        )
        gate_target = exact_tail_mask(labels, role)
        gate_prediction = collected["applicability_prob"].view(-1) >= 0.5
        mechanism_target = residual_mechanism_index(anchor, labels)
        mechanism_prediction = collected["mechanism_probs"].argmax(dim=1)
        eligible = bool(
            tail_gain > 0.0
            and global_delta <= self.global_tolerance
            and harm <= self.max_harm_rate
        )
        return {
            "objective": float(objective),
            "eligible": eligible,
            "global_mae": global_mae,
            "tail_mae": tail_mae,
            "anchor_global_mae": anchor_global,
            "anchor_tail_mae": anchor_tail,
            "tail_gain": tail_gain,
            "global_delta": global_delta,
            "harm_over_010_rate": harm,
            "gate_accuracy": float(
                (gate_prediction == gate_target).float().mean().item()
            ),
            "mechanism_accuracy": float(
                (mechanism_prediction == mechanism_target).float().mean().item()
            ),
            "mean_abs_correction": float(
                torch.abs(collected["correction"]).mean().item()
            ),
        }

    def train_candidate(self, role):
        role_dir = self.save_dir / role
        role_dir.mkdir(parents=True, exist_ok=True)
        role_seed = int(getattr(self.args, "seed", 0))
        torch.manual_seed(role_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(role_seed)
        model = self._new_head()
        optimizer = optim.AdamW(
            model.parameter_groups(self.learning_rate),
            weight_decay=self.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2, min_lr=1e-7
        )
        loader = self._train_loader(role_seed)

        initial = self._collect_head(
            model,
            self.valid["feature"],
            self.valid["anchor"],
            self.valid["labels"],
        )
        initial_row = self._validation_row(initial, role)
        best = {
            "epoch": 0,
            "objective": initial_row["objective"],
            "valid": initial_row,
            "state": _cpu_state_dict(model),
            "anchor_fallback": True,
        }
        history: List[Dict[str, object]] = [
            {
                "epoch": 0,
                "stage": "anchor",
                **{"valid_" + key: value for key, value in initial_row.items()},
            }
        ]
        no_improvement = 0
        best_training_objective = float("inf")

        for epoch in range(1, self.max_epochs + 1):
            model.train()
            totals: Dict[str, float] = {}
            for feature, anchor, labels, fold_index in loader:
                del fold_index
                outputs = model(
                    feature.to(self.args.device), anchor.to(self.args.device)
                )
                losses = oof_tail_residual_loss(
                    outputs,
                    labels.to(self.args.device),
                    role=role,
                    membership_temperature=self.membership_temperature,
                    gain_margin=self.gain_margin,
                    gain_fraction=self.gain_fraction,
                    weights=self.loss_weights,
                )
                optimizer.zero_grad()
                losses["total"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
                optimizer.step()
                for key, value in losses.items():
                    totals[key] = totals.get(key, 0.0) + float(value.detach().item())
            train_row = {
                key: value / max(1, len(loader)) for key, value in totals.items()
            }
            valid = self._collect_head(
                model,
                self.valid["feature"],
                self.valid["anchor"],
                self.valid["labels"],
            )
            valid_row = self._validation_row(valid, role)
            scheduler.step(valid_row["objective"])
            history.append(
                {
                    "epoch": epoch,
                    "stage": "oof_residual",
                    **{"train_" + key: value for key, value in train_row.items()},
                    **{"valid_" + key: value for key, value in valid_row.items()},
                }
            )
            logger.info(
                "V9.2 role=%s epoch=%d tail_gain=%+.6f global_delta=%+.6f "
                "harm=%.4f eligible=%s",
                role,
                epoch,
                valid_row["tail_gain"],
                valid_row["global_delta"],
                valid_row["harm_over_010_rate"],
                valid_row["eligible"],
            )
            if valid_row["objective"] < best_training_objective - 1e-6:
                best_training_objective = valid_row["objective"]
                no_improvement = 0
            else:
                no_improvement += 1
            if valid_row["eligible"] and (
                best["anchor_fallback"]
                or valid_row["objective"] < best["objective"] - 1e-6
            ):
                best = {
                    "epoch": epoch,
                    "objective": valid_row["objective"],
                    "valid": dict(valid_row),
                    "state": _cpu_state_dict(model),
                    "anchor_fallback": False,
                }
            if no_improvement >= self.early_stop:
                break

        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self._collect_head(
            model,
            self.valid["feature"],
            self.valid["anchor"],
            self.valid["labels"],
        )
        test = self._collect_head(
            model,
            self.test["feature"],
            self.test["anchor"],
            self.test["labels"],
        )
        pd.DataFrame(history).to_csv(role_dir / "training_history.csv", index=False)
        torch.save(
            {
                "method": "oof_tail_residual_expert_v9_2",
                "role": role,
                "selected_epoch": int(best["epoch"]),
                "anchor_fallback": bool(best["anchor_fallback"]),
                "selected_valid": dict(best["valid"]),
                "anchor_index": self.anchor_index,
                "anchor_checkpoint": str(self.teacher_paths[self.anchor_index]),
                "state_dict": best["state"],
            },
            role_dir / "oof_tail_residual_expert_v92_best.pth",
        )
        return {"role": role, "best": best, "valid": valid, "test": test}

    def _capability_rows(self, results, split):
        base = self.valid if split == "valid" else self.test
        rows = [
            {
                "expert": "anchor",
                **tail_capability_metrics(base["anchor"], base["labels"]),
            }
        ]
        for result in results:
            rows.append(
                {
                    "expert": result["role"],
                    **tail_capability_metrics(
                        result[split]["prediction"], base["labels"]
                    ),
                }
            )
        return rows

    def _select_policy(self, valid_rows):
        anchor = next(row for row in valid_rows if row["expert"] == "anchor")
        policy = {}
        for tail in ("strong_negative", "strong_positive"):
            column = tail + "_mae"
            eligible = [anchor]
            for row in valid_rows:
                if row["expert"] == "anchor":
                    continue
                if row["global_mae"] - anchor["global_mae"] <= self.global_tolerance:
                    eligible.append(row)
            best = min(eligible, key=lambda row: row[column])
            if best[column] >= anchor[column] - 1e-8:
                best = anchor
            policy[tail] = {
                "expert": best["expert"],
                "valid_mae": float(best[column]),
                "anchor_valid_mae": float(anchor[column]),
                "valid_gain": float(anchor[column] - best[column]),
                "global_delta": float(best["global_mae"] - anchor["global_mae"]),
            }
        return policy

    @staticmethod
    def _by_name(results):
        return {result["role"]: result for result in results}

    def _selected_output(self, results_by_name, split, expert):
        base = self.valid if split == "valid" else self.test
        if expert == "anchor":
            return {
                "prediction": base["anchor"],
                "applicability_prob": torch.zeros_like(base["anchor"]),
            }
        return results_by_name[expert][split]

    def _calibrate_gate(self, results, policy):
        by_name = self._by_name(results)
        anchor = self.valid["anchor"]
        labels = self.valid["labels"]
        negative = self._selected_output(
            by_name, "valid", policy["strong_negative"]["expert"]
        )
        positive = self._selected_output(
            by_name, "valid", policy["strong_positive"]["expert"]
        )
        rows = []
        best = None
        for negative_probability in (0.40, 0.50, 0.60, 0.70, 0.80):
            for positive_probability in (0.40, 0.50, 0.60, 0.70, 0.80):
                for beta in (0.25, 0.50, 0.75, 1.00):
                    prediction = anchor.clone()
                    negative_mask = (
                        (anchor.view(-1) < 0)
                        & (
                            negative["applicability_prob"].view(-1)
                            >= negative_probability
                        )
                    )
                    positive_mask = (
                        (anchor.view(-1) > 0)
                        & (
                            positive["applicability_prob"].view(-1)
                            >= positive_probability
                        )
                    )
                    prediction[negative_mask] = (
                        (1.0 - beta) * anchor[negative_mask]
                        + beta * negative["prediction"][negative_mask]
                    )
                    prediction[positive_mask] = (
                        (1.0 - beta) * anchor[positive_mask]
                        + beta * positive["prediction"][positive_mask]
                    )
                    mae = float(torch.abs(prediction - labels).mean().item())
                    gain = torch.abs(anchor - labels) - torch.abs(prediction - labels)
                    harm = float((gain < -0.10).float().mean().item())
                    objective = mae + 0.03 * harm
                    row = {
                        "negative_probability": negative_probability,
                        "positive_probability": positive_probability,
                        "beta": beta,
                        "mae": mae,
                        "harm_over_010_rate": harm,
                        "objective": objective,
                    }
                    rows.append(row)
                    if best is None or (objective, mae) < (
                        best["objective"],
                        best["mae"],
                    ):
                        best = row
        if best is None:
            raise RuntimeError("gate calibration generated no candidate")
        return best, rows

    def _apply_gate(self, results, split, policy, gate_policy):
        by_name = self._by_name(results)
        base = self.valid if split == "valid" else self.test
        anchor = base["anchor"]
        negative = self._selected_output(
            by_name, split, policy["strong_negative"]["expert"]
        )
        positive = self._selected_output(
            by_name, split, policy["strong_positive"]["expert"]
        )
        prediction = anchor.clone()
        negative_mask = (
            (anchor.view(-1) < 0)
            & (
                negative["applicability_prob"].view(-1)
                >= gate_policy["negative_probability"]
            )
        )
        positive_mask = (
            (anchor.view(-1) > 0)
            & (
                positive["applicability_prob"].view(-1)
                >= gate_policy["positive_probability"]
            )
        )
        beta = float(gate_policy["beta"])
        prediction[negative_mask] = (
            (1.0 - beta) * anchor[negative_mask]
            + beta * negative["prediction"][negative_mask]
        )
        prediction[positive_mask] = (
            (1.0 - beta) * anchor[positive_mask]
            + beta * positive["prediction"][positive_mask]
        )
        return prediction

    def train_all(self):
        results = [self.train_candidate(role) for role in TAIL_ROLE_NAMES]
        valid_rows = self._capability_rows(results, "valid")
        test_rows = self._capability_rows(results, "test")
        pd.DataFrame(valid_rows).to_csv(
            self.save_dir / "v92_valid_tail_capability_matrix.csv", index=False
        )
        pd.DataFrame(test_rows).to_csv(
            self.save_dir / "v92_test_tail_capability_matrix.csv", index=False
        )
        policy = self._select_policy(valid_rows)
        gate_policy, gate_rows = self._calibrate_gate(results, policy)
        pd.DataFrame(gate_rows).to_csv(
            self.save_dir / "v92_gate_calibration.csv", index=False
        )
        return self._evaluate_and_save(results, policy, gate_policy)

    def _evaluate_and_save(self, results, policy, gate_policy):
        by_name = self._by_name(results)
        anchor = self.test["anchor"]
        labels = self.test["labels"]
        deployable = self._apply_gate(results, "test", policy, gate_policy)

        true_region = anchor.clone()
        for tail in ("strong_negative", "strong_positive"):
            mask = exact_tail_mask(labels, tail)
            selected = self._selected_output(
                by_name, "test", policy[tail]["expert"]
            )
            true_region[mask] = selected["prediction"][mask]

        candidate_predictions = [anchor] + [
            result["test"]["prediction"] for result in results
        ]
        stacked = torch.stack(candidate_predictions, dim=1)
        errors = torch.abs(stacked - labels.unsqueeze(1))
        indices = errors.squeeze(-1).argmin(dim=1)
        sample_oracle = stacked[torch.arange(stacked.size(0)), indices]

        named = {
            "anchor": anchor,
            **{result["role"]: result["test"]["prediction"] for result in results},
            "oof_gate_valid_selected": deployable,
            "true_region_valid_selected_tail_policy": true_region,
            "sample_oracle_upper_bound": sample_oracle,
        }
        rows = []
        payload = {}
        for name, prediction in named.items():
            metrics = _safe_metrics(self.metrics_fn, prediction, labels)
            metrics.update(tail_capability_metrics(prediction, labels))
            rows.append({"model": name, **metrics})
            payload[name] = metrics
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v92_test_tail_summary.csv", index=False
        )

        frame: Dict[str, object] = {
            "sample_id": self.test["sample_ids"],
            "label": labels.view(-1).tolist(),
            "anchor": anchor.view(-1).tolist(),
            "oof_gate": deployable.view(-1).tolist(),
            "true_region_tail_policy": true_region.view(-1).tolist(),
            "sample_oracle_upper_bound": sample_oracle.view(-1).tolist(),
        }
        for result in results:
            name = result["role"]
            frame[name + "_prediction"] = (
                result["test"]["prediction"].view(-1).tolist()
            )
            frame[name + "_correction"] = (
                result["test"]["correction"].view(-1).tolist()
            )
            frame[name + "_applicability"] = (
                result["test"]["applicability_prob"].view(-1).tolist()
            )
            for index, mechanism in enumerate(MECHANISM_NAMES):
                frame[name + "_p_" + mechanism] = (
                    result["test"]["mechanism_probs"][:, index].tolist()
                )
        pd.DataFrame(frame).to_csv(
            self.save_dir / "oof_tail_residual_experts_v92_predictions.csv",
            index=False,
        )

        summary = {
            "method": "nested_grouped_oof_cfcompat_tail_residual_v9_2",
            "dataset": str(self.args.dataset_name),
            "seed": int(getattr(self.args, "seed", 0)),
            "anchor_index": int(self.anchor_index),
            "anchor_checkpoint": str(self.teacher_paths[self.anchor_index]),
            "anchor_load_mode": self.anchor_load_mode,
            "oof_cache_version": self.oof.get("version"),
            "oof_outer_folds": int(self.oof.get("outer_folds", -1)),
            "oof_protocol": self.oof.get("protocol"),
            "candidates": {
                result["role"]: {
                    "selected_epoch": int(result["best"]["epoch"]),
                    "anchor_fallback": bool(result["best"]["anchor_fallback"]),
                    "selected_valid": dict(result["best"]["valid"]),
                }
                for result in results
            },
            "validation_selected_tail_policy": policy,
            "gate_policy": gate_policy,
            "test_results": payload,
            "loss_weights": self.loss_weights.__dict__,
            "selection_protocol": (
                "Outer OOF groups are excluded from every fold training and inner "
                "selection stage. Tail-head checkpoints, expert assignment, gate "
                "thresholds, and beta use Validation only. Test labels are used "
                "only for final reporting and named oracle analyses."
            ),
        }
        (self.save_dir / "oof_tail_residual_experts_v92_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        logger.info(
            "V9.2 TEST anchor=%.6f gate=%.6f true-region=%.6f",
            payload["anchor"]["MAE"],
            payload["oof_gate_valid_selected"]["MAE"],
            payload["true_region_valid_selected_tail_policy"]["MAE"],
        )
        return summary
