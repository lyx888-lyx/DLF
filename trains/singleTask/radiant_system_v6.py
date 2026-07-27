"""Training and evaluation system for RADIANT-DLF V6."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids
from .model.RADIANT_DLF import MODE_NAMES
from .radiant_losses_v6 import (
    REGION_NAMES,
    posterior_diagnostics,
    radiant_loss,
    region_index,
    safe_corr,
)


logger = logging.getLogger("MMSA")


def _cpu_state_dict(model: nn.Module):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _safe_metrics(metrics_fn, prediction, labels):
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _selection_stats(anchor, prediction, labels):
    gain = torch.abs(anchor - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = torch.abs(prediction - anchor).view(-1) > 1e-8
    count = int(selected.sum().item())
    return {
        "mae": float(torch.abs(prediction - labels).mean().item()),
        "mean_realized_gain": float(gain.mean().item()),
        "selected_mean_gain": float(gain[selected].mean().item()) if count else None,
        "correction_rate": float(selected.float().mean().item()),
        "correction_count": count,
        "correction_precision": float((gain[selected] > 0).float().mean().item()) if count else None,
        "harm_over_005_rate": float((gain < -0.05).float().mean().item()),
        "harm_over_010_rate": float((gain < -0.10).float().mean().item()),
        "mean_abs_correction": float(torch.abs(prediction - anchor).mean().item()),
    }


def _apply_policy(raw, policy):
    source = str(policy.get("source", "mean"))
    alpha = float(policy.get("alpha", 1.0))
    target = raw["mean_prediction"] if source == "mean" else raw["median_prediction"]
    return raw["anchor"] + alpha * (target - raw["anchor"])


def _calibrate_policies(raw_by_mode):
    rows = []
    policies = {}
    for mode, raw in raw_by_mode.items():
        best = None
        for source in ("mean", "median"):
            for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
                policy = {"source": source, "alpha": alpha}
                prediction = _apply_policy(raw, policy)
                stats = _selection_stats(raw["anchor"], prediction, raw["labels"])
                objective = (
                    stats["mae"]
                    + 0.10 * stats["harm_over_010_rate"]
                    + 0.01 * stats["mean_abs_correction"]
                )
                row = {"mode": mode, **policy, "objective": objective, **stats}
                rows.append(row)
                if best is None or (row["mae"], row["harm_over_010_rate"]) < (
                    best["mae"], best["harm_over_010_rate"]
                ):
                    best = row
        policies[mode] = {"source": best["source"], "alpha": best["alpha"]}
    return policies, rows


def _region_rows(model_name, mode, anchor, prediction, labels):
    regions = region_index(labels)
    rows = []
    for index, name in enumerate(REGION_NAMES):
        mask = regions == index
        if mask.any():
            rows.append({
                "model": model_name,
                "mode": mode,
                "region": name,
                "count": int(mask.sum().item()),
                "anchor_mae": float(torch.abs(anchor[mask] - labels[mask]).mean().item()),
                "final_mae": float(torch.abs(prediction[mask] - labels[mask]).mean().item()),
                "anchor_bias": float((labels[mask] - anchor[mask]).mean().item()),
                "final_bias": float((labels[mask] - prediction[mask]).mean().item()),
            })
    return rows


class RadiantTrainerV6:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        head_epochs=8,
        max_epochs=40,
        early_stop=10,
        head_lr=3e-4,
        tail_lr=1e-5,
        weight_decay=1e-3,
        loss_weights=None,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.head_epochs = int(head_epochs)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.head_lr = float(head_lr)
        self.tail_lr = float(tail_lr)
        self.weight_decay = float(weight_decay)
        self.loss_weights = loss_weights or {
            "posterior_energy": 1.00,
            "posterior_mean": 0.75,
            "latent_energy": 0.20,
            "latent_mean": 0.20,
            "support": 0.05,
            "sign": 0.08,
            "scale": 0.05,
            "region": 0.25,
            "bias": 0.08,
            "monotonic": 0.08,
            "voi": 0.15,
            "backbone": 0.25,
        }

    def _optimizer(self, model, stage):
        head_parameters = [
            parameter for parameter in model.posterior.parameters() if parameter.requires_grad
        ]
        groups = [{"params": head_parameters, "lr": self.head_lr}]
        if stage == "tail":
            tail_parameters = [
                parameter for parameter in model.backbone.parameters() if parameter.requires_grad
            ]
            if tail_parameters:
                groups.append({"params": tail_parameters, "lr": self.tail_lr})
        return optim.AdamW(groups, weight_decay=self.weight_decay)

    def _train_epoch(self, model, dataloader, optimizer, include_backbone_loss):
        model.set_train_mode()
        totals = {}
        predictions, targets = [], []
        optimizer.zero_grad()
        accumulation = max(1, int(getattr(self.args, "update_epochs", 1)))
        for step, batch in enumerate(tqdm(dataloader, leave=False), start=1):
            text = batch["text"].to(self.args.device)
            audio = batch["audio"].to(self.args.device)
            vision = batch["vision"].to(self.args.device)
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            views = model(text, audio, vision)
            losses = radiant_loss(
                views,
                labels,
                dataset=self.args.dataset_name,
                weights=self.loss_weights,
                include_backbone_loss=include_backbone_loss,
            )
            (losses["total"] / accumulation).backward()
            if step % accumulation == 0 or step == len(dataloader):
                nn.utils.clip_grad_norm_(
                    [parameter for parameter in model.parameters() if parameter.requires_grad],
                    max_norm=2.0,
                )
                optimizer.step()
                optimizer.zero_grad()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
            predictions.append(views["lav"]["mean_prediction"].detach().cpu())
            targets.append(labels.detach().cpu())
        count = max(1, len(dataloader))
        row = {name: value / count for name, value in totals.items()}
        row.update({
            f"train_{name}": value
            for name, value in _safe_metrics(
                self.metrics_fn, torch.cat(predictions), torch.cat(targets)
            ).items()
        })
        return row

    @torch.no_grad()
    def collect(self, model, dataloader):
        model.eval()
        buffers = {
            mode: {
                "anchor": [],
                "mean_prediction": [],
                "median_prediction": [],
                "candidate_predictions": [],
                "component_weights": [],
                "entropy": [],
                "spread": [],
                "predicted_scale": [],
                "voi": [],
                "labels": [],
                "sample_ids": [],
            }
            for mode in MODE_NAMES
        }
        for batch in tqdm(dataloader, leave=False):
            text = batch["text"].to(self.args.device)
            audio = batch["audio"].to(self.args.device)
            vision = batch["vision"].to(self.args.device)
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            views = model(text, audio, vision)
            ids = normalize_batch_ids(batch.get("id"))
            for mode, view in views.items():
                target = buffers[mode]
                for key in (
                    "anchor",
                    "mean_prediction",
                    "median_prediction",
                    "candidate_predictions",
                    "component_weights",
                    "entropy",
                    "spread",
                    "predicted_scale",
                    "voi",
                ):
                    target[key].append(view[key].detach().cpu())
                target["labels"].append(labels.detach().cpu())
                target["sample_ids"].extend(ids)
        result = {}
        for mode, buffer in buffers.items():
            result[mode] = {
                key: torch.cat(value, dim=0) if key != "sample_ids" else value
                for key, value in buffer.items()
            }
        return result

    def _valid_objectives(self, raw):
        lav = float(torch.abs(raw["lav"]["mean_prediction"] - raw["lav"]["labels"]).mean().item())
        missing = np.mean([
            float(torch.abs(raw[mode]["mean_prediction"] - raw[mode]["labels"]).mean().item())
            for mode in ("la", "lv", "l")
        ])
        return lav, float(missing), float(0.5 * (lav + missing))

    def train(self, model, dataloaders):
        history = []
        best_lav = {"value": float("inf"), "epoch": 0, "state": None}
        best_j = {"value": float("inf"), "epoch": 0, "state": None}
        global_epoch = 0

        stages = (
            ("head", self.head_epochs),
            ("tail", max(0, self.max_epochs - self.head_epochs)),
        )
        for stage, stage_epochs in stages:
            if stage_epochs <= 0:
                continue
            if stage == "head":
                model.freeze_backbone()
            else:
                model.unfreeze_backbone_tail()
            optimizer = self._optimizer(model, stage)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
            )
            stage_best_epoch = global_epoch
            for _ in range(stage_epochs):
                global_epoch += 1
                train_row = self._train_epoch(
                    model,
                    dataloaders["train"],
                    optimizer,
                    include_backbone_loss=(stage == "tail"),
                )
                valid_raw = self.collect(model, dataloaders["valid"])
                lav, missing, joint = self._valid_objectives(valid_raw)
                scheduler.step(joint)
                row = {
                    "epoch": global_epoch,
                    "stage": stage,
                    **train_row,
                    "valid_lav_mae": lav,
                    "valid_missing_macro_mae": missing,
                    "valid_joint": joint,
                }
                for mode in MODE_NAMES:
                    diag = posterior_diagnostics(valid_raw[mode], valid_raw[mode]["labels"])
                    for name, value in diag.items():
                        row[f"valid_{mode}_{name}"] = value
                history.append(row)
                logger.info(
                    "V6 epoch=%d stage=%s train=%.4f valid_lav=%.4f "
                    "valid_missing=%.4f J=%.4f support=%.4f",
                    global_epoch,
                    stage,
                    train_row.get("total", float("nan")),
                    lav,
                    missing,
                    joint,
                    row["valid_lav_posterior_support_mae"],
                )
                improved = False
                if lav < best_lav["value"] - 1e-6:
                    best_lav = {
                        "value": lav,
                        "epoch": global_epoch,
                        "state": _cpu_state_dict(model),
                    }
                    improved = True
                if joint < best_j["value"] - 1e-6:
                    best_j = {
                        "value": joint,
                        "epoch": global_epoch,
                        "state": _cpu_state_dict(model),
                    }
                    improved = True
                if improved:
                    stage_best_epoch = global_epoch
                if stage == "tail" and global_epoch - stage_best_epoch >= self.early_stop:
                    break

        if best_lav["state"] is None or best_j["state"] is None:
            raise RuntimeError("RADIANT V6 failed to select a validation checkpoint")
        pd.DataFrame(history).to_csv(self.save_dir / "v6_training_history.csv", index=False)
        torch.save(best_lav, self.save_dir / "radiant_v6_best_lav.pth")
        torch.save(best_j, self.save_dir / "radiant_v6_best_joint.pth")
        return best_lav, best_j

    def _evaluate_checkpoint(self, model, state, dataloaders, name):
        model.load_state_dict(state["state"])
        model.to(self.args.device)
        valid_raw = self.collect(model, dataloaders["valid"])
        policies, calibration_rows = _calibrate_policies(valid_raw)
        test_raw = self.collect(model, dataloaders["test"])
        mode_results = {}
        prediction_columns = {
            "sample_id": test_raw["lav"]["sample_ids"],
            "label": test_raw["lav"]["labels"].view(-1).numpy(),
        }
        region_rows = []
        for mode in MODE_NAMES:
            raw = test_raw[mode]
            prediction = _apply_policy(raw, policies[mode])
            anchor_metrics = _safe_metrics(self.metrics_fn, raw["anchor"], raw["labels"])
            final_metrics = _safe_metrics(self.metrics_fn, prediction, raw["labels"])
            diag = posterior_diagnostics(raw, raw["labels"])
            stats = _selection_stats(raw["anchor"], prediction, raw["labels"])
            mode_results[mode] = {
                "policy": policies[mode],
                "anchor_metrics": anchor_metrics,
                "metrics": final_metrics,
                "posterior": diag,
                "selection": stats,
            }
            prediction_columns[f"{mode}_anchor"] = raw["anchor"].view(-1).numpy()
            prediction_columns[f"{mode}_prediction"] = prediction.view(-1).numpy()
            prediction_columns[f"{mode}_posterior_mean"] = raw["mean_prediction"].view(-1).numpy()
            prediction_columns[f"{mode}_posterior_median"] = raw["median_prediction"].view(-1).numpy()
            prediction_columns[f"{mode}_entropy"] = raw["entropy"].view(-1).numpy()
            prediction_columns[f"{mode}_spread"] = raw["spread"].view(-1).numpy()
            prediction_columns[f"{mode}_predicted_scale"] = raw["predicted_scale"].view(-1).numpy()
            region_rows.extend(
                _region_rows(name, mode, raw["anchor"], prediction, raw["labels"])
            )
        calibration_frame = pd.DataFrame(calibration_rows)
        calibration_frame.to_csv(
            self.save_dir / f"v6_{name}_policy_calibration.csv", index=False
        )
        pd.DataFrame(prediction_columns).to_csv(
            self.save_dir / f"v6_{name}_predictions.csv", index=False
        )
        pd.DataFrame(region_rows).to_csv(
            self.save_dir / f"v6_{name}_region_diagnostics.csv", index=False
        )
        acquisition = self._active_acquisition_curves(test_raw, policies)
        pd.DataFrame(acquisition).to_csv(
            self.save_dir / f"v6_{name}_active_acquisition.csv", index=False
        )
        return {
            "selected_epoch": int(state["epoch"]),
            "selected_valid_value": float(state["value"]),
            "policies": policies,
            "modes": mode_results,
            "active_acquisition": acquisition,
        }

    def _active_acquisition_curves(self, raw, policies):
        policy_predictions = {
            mode: _apply_policy(raw[mode], policies[mode]) for mode in MODE_NAMES
        }
        rows = []
        specifications = {
            "l": ((0, "la", "add_audio"), (1, "lv", "add_visual"), (2, "lav", "add_audio_visual")),
            "la": ((1, "lav", "add_visual"),),
            "lv": ((0, "lav", "add_audio"),),
        }
        for initial, actions in specifications.items():
            labels = raw[initial]["labels"]
            base = policy_predictions[initial]
            voi = raw[initial]["voi"]
            action_scores = torch.stack([voi[:, index] for index, _, _ in actions], dim=1)
            best_score, best_action = action_scores.max(dim=1)
            ranking = torch.argsort(best_score, descending=True)
            for budget in (0.0, 0.10, 0.25, 0.50, 0.75, 1.0):
                prediction = base.clone()
                acquired = torch.zeros(len(labels), dtype=torch.bool)
                if budget > 0:
                    count = min(len(labels), max(1, int(round(len(labels) * budget))))
                    selected = ranking[:count]
                    positive = best_score[selected] > 0
                    selected = selected[positive]
                    acquired[selected] = True
                    for action_index, (_, target_mode, _) in enumerate(actions):
                        mask = selected[best_action[selected] == action_index]
                        if len(mask):
                            prediction[mask] = policy_predictions[target_mode][mask]
                rows.append({
                    "initial_mode": initial,
                    "budget": budget,
                    "acquisition_rate": float(acquired.float().mean().item()),
                    "mae": float(torch.abs(prediction - labels).mean().item()),
                    "corr": safe_corr(prediction, labels),
                    "baseline_mae": float(torch.abs(base - labels).mean().item()),
                })
        return rows

    def evaluate_and_save(self, model, dataloaders, best_lav, best_j):
        lav_result = self._evaluate_checkpoint(model, best_lav, dataloaders, "best_lav")
        joint_result = self._evaluate_checkpoint(model, best_j, dataloaders, "best_joint")
        summary = {
            "method": "radiant_recoverable_posterior_acquisition_v6",
            "seed": int(self.args.seed),
            "backbone": "DLF",
            "modalities": list(MODE_NAMES),
            "best_lav": lav_result,
            "best_joint": joint_result,
            "loss_weights": self.loss_weights,
        }
        (self.save_dir / "radiant_v6_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return summary
