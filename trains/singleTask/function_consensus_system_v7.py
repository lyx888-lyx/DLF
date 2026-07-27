"""Teacher caching, training and evaluation for V7 function-space consensus."""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Sequence

import pandas as pd
import torch
import torch.nn as nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids
from .function_consensus_v7 import REGION_NAMES, consensus_distillation_loss, region_index, safe_corr
from .model.FSC_DLF import FrozenTeacherDLF, robust_committee_summary

logger = logging.getLogger("MMSA")


def _safe_metrics(metrics_fn, prediction, labels):
    return {
        key: float(value)
        for key, value in metrics_fn(
            prediction.detach().cpu(), labels.detach().cpu()
        ).items()
    }


def _cpu_state_dict(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _checkpoint_signature(paths: Sequence[Path]):
    digest = hashlib.sha256()
    for path in paths:
        stat = path.stat()
        digest.update(str(path.resolve()).encode("utf-8"))
        digest.update(str(stat.st_size).encode("utf-8"))
        digest.update(str(stat.st_mtime_ns).encode("utf-8"))
    return digest.hexdigest()


def _cache_index(sample_ids):
    mapping = {}
    for index, value in enumerate(sample_ids):
        key = str(value)
        if key in mapping:
            raise RuntimeError(f"Duplicate sample id in teacher cache: {key}")
        mapping[key] = index
    return mapping


def _batch_cache_indices(batch_ids, mapping):
    values = normalize_batch_ids(batch_ids)
    try:
        return torch.tensor([mapping[str(value)] for value in values], dtype=torch.long)
    except KeyError as error:
        raise KeyError(f"Sample id missing from teacher cache: {error}") from error


@torch.no_grad()
def _collect_teacher(teacher, dataloader, device):
    predictions, features, labels, sample_ids = [], [], [], []
    teacher.eval()
    for batch in tqdm(dataloader, leave=False):
        text = batch["text"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        output = teacher(text, audio, vision)
        predictions.append(output["prediction"].detach().cpu())
        features.append(output["feature"].detach().cpu())
        labels.append(batch["labels"]["M"].view(-1, 1).cpu())
        sample_ids.extend(normalize_batch_ids(batch.get("id")))
    prediction = torch.cat(predictions, dim=0)
    feature = torch.cat(features, dim=0)
    target = torch.cat(labels, dim=0)
    order = sorted(range(len(sample_ids)), key=lambda index: str(sample_ids[index]))
    order_tensor = torch.tensor(order, dtype=torch.long)
    return {
        "prediction": prediction[order_tensor],
        "feature": feature[order_tensor],
        "labels": target[order_tensor],
        "sample_ids": [sample_ids[index] for index in order],
    }


def build_teacher_cache(
    args,
    dataloaders,
    teacher_paths,
    device,
    cache_path,
    rebuild=False,
):
    cache_path = Path(cache_path)
    signature = _checkpoint_signature(teacher_paths)
    if cache_path.is_file() and not rebuild:
        payload = torch.load(cache_path, map_location="cpu")
        if payload.get("checkpoint_signature") == signature:
            logger.info("Reusing V7 teacher cache: %s", cache_path)
            return payload
        logger.warning("Teacher checkpoint signature changed; rebuilding cache.")

    split_buffers = {
        split: {"predictions": [], "features": [], "labels": None, "sample_ids": None}
        for split in ("train", "valid", "test")
    }
    diagnostics = []
    for teacher_index, path in enumerate(teacher_paths):
        logger.info("Caching teacher %d/%d: %s", teacher_index + 1, len(teacher_paths), path)
        teacher = FrozenTeacherDLF(args).to(device)
        incompatible = teacher.load_checkpoint(path, map_location=device)
        logger.info(
            "Loaded teacher %s (missing=%d unexpected=%d)",
            path,
            len(incompatible.missing_keys),
            len(incompatible.unexpected_keys),
        )
        teacher.freeze()
        for split in ("train", "valid", "test"):
            collected = _collect_teacher(teacher, dataloaders[split], device)
            target = split_buffers[split]
            if target["sample_ids"] is None:
                target["sample_ids"] = collected["sample_ids"]
                target["labels"] = collected["labels"]
            elif target["sample_ids"] != collected["sample_ids"]:
                raise RuntimeError(f"Teacher order mismatch for split={split}.")
            target["predictions"].append(collected["prediction"])
            target["features"].append(collected["feature"])
            diagnostics.append({
                "teacher_index": teacher_index,
                "checkpoint": str(path),
                "split": split,
                "mae": float(torch.abs(collected["prediction"] - collected["labels"]).mean().item()),
                "corr": safe_corr(collected["prediction"], collected["labels"]),
            })
        del teacher
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    splits = {}
    for split, buffer in split_buffers.items():
        splits[split] = {
            "predictions": torch.stack(buffer["predictions"], dim=1),
            "features": torch.stack(buffer["features"], dim=0),
            "labels": buffer["labels"],
            "sample_ids": buffer["sample_ids"],
        }
    payload = {
        "checkpoint_signature": signature,
        "teacher_paths": [str(path) for path in teacher_paths],
        "splits": splits,
        "diagnostics": diagnostics,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    pd.DataFrame(diagnostics).to_csv(
        cache_path.parent / "v7_teacher_cache_diagnostics.csv", index=False
    )
    return payload


def _committee_from_split(split, indices, device, temperature):
    predictions = split["predictions"][indices].to(device)
    features = split["features"][:, indices].to(device)
    return robust_committee_summary(predictions, features, temperature)


def _selection_stats(anchor, prediction, labels):
    gain = torch.abs(anchor - labels).view(-1) - torch.abs(prediction - labels).view(-1)
    selected = torch.abs(prediction - anchor).view(-1) > 1e-8
    count = int(selected.sum().item())
    return {
        "mae": float(torch.abs(prediction - labels).mean().item()),
        "mean_realized_gain": float(gain.mean().item()),
        "correction_precision": float((gain[selected] > 0).float().mean().item()) if count else None,
        "harm_over_005_rate": float((gain < -0.05).float().mean().item()),
        "harm_over_010_rate": float((gain < -0.10).float().mean().item()),
        "mean_abs_change": float(torch.abs(prediction - anchor).mean().item()),
    }


def _fit_robust_teacher_weights(predictions, labels, sample_ids, steps=500):
    teacher_count = predictions.size(1)
    logits = torch.zeros(teacher_count, dtype=torch.float32, requires_grad=True)
    optimizer = optim.Adam([logits], lr=0.05)
    groups = torch.tensor([
        int(hashlib.sha1(str(value).encode("utf-8")).hexdigest(), 16) % 3
        for value in sample_ids
    ])
    uniform = torch.full((teacher_count,), 1.0 / teacher_count)
    best = None
    for _ in range(int(steps)):
        weights = torch.softmax(logits, dim=0)
        prediction = (predictions * weights.view(1, -1, 1)).sum(dim=1)
        overall = torch.abs(prediction - labels).mean()
        group_losses = []
        for group in range(3):
            mask = groups == group
            if mask.any():
                group_losses.append(torch.abs(prediction[mask] - labels[mask]).mean())
        group_tensor = torch.stack(group_losses)
        stability = group_tensor.std(unbiased=False)
        kl = (weights * (weights.clamp_min(1e-8) / uniform).log()).sum()
        objective = overall + 0.35 * stability + 0.01 * kl
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        row = (float(objective.item()), weights.detach().clone())
        if best is None or row[0] < best[0]:
            best = row
    return best[1]


def _committee_baselines(split, weights=None):
    predictions = split["predictions"]
    sorted_values = predictions.sort(dim=1).values
    teacher_count = predictions.size(1)
    if teacher_count >= 5:
        trim = max(1, int(round(teacher_count * 0.20)))
        trimmed = sorted_values[:, trim:teacher_count - trim].mean(dim=1)
    elif teacher_count >= 3:
        trimmed = sorted_values[:, 1:-1].mean(dim=1)
    else:
        trimmed = predictions.mean(dim=1)
    result = {
        "uniform": predictions.mean(dim=1),
        "median": predictions.median(dim=1).values,
        "trimmed": trimmed,
        "robust_consensus": 0.50 * trimmed + 0.50 * predictions.mean(dim=1),
    }
    if weights is not None:
        result["robust_simplex"] = (
            predictions * weights.view(1, -1, 1)
        ).sum(dim=1)
    return result


class FunctionConsensusTrainerV7:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths,
        dispersion_temperature=0.12,
        head_epochs=5,
        max_epochs=30,
        early_stop=8,
        head_lr=3e-4,
        tail_lr=1e-5,
        weight_decay=1e-3,
        loss_weights=None,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache = teacher_cache
        self.teacher_paths = list(teacher_paths)
        self.dispersion_temperature = float(dispersion_temperature)
        self.head_epochs = int(head_epochs)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.head_lr = float(head_lr)
        self.tail_lr = float(tail_lr)
        self.weight_decay = float(weight_decay)
        self.loss_weights = loss_weights or {
            "supervised": 1.00,
            "base_supervised": 0.35,
            "distill": 0.55,
            "pairwise": 0.18,
            "kernel": 0.10,
            "ordinal": 0.12,
            "uncertainty": 0.06,
            "region": 0.28,
            "bias": 0.08,
            "disagreement_shrink": 0.10,
            "correction_shrink": 0.01,
        }
        self.index = {
            split: _cache_index(self.cache["splits"][split]["sample_ids"])
            for split in ("train", "valid", "test")
        }

    def _optimizer(self, model, stage):
        head_parameters = [
            parameter
            for module in (
                model.adapter,
                model.residual_head,
                model.log_scale_head,
                model.ordinal_head,
            )
            for parameter in module.parameters()
            if parameter.requires_grad
        ]
        groups = [{"params": head_parameters, "lr": self.head_lr}]
        if stage == "tail":
            tail = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
            if tail:
                groups.append({"params": tail, "lr": self.tail_lr})
        return optim.AdamW(groups, weight_decay=self.weight_decay)

    def _train_epoch(self, model, dataloader, optimizer):
        model.set_train_mode()
        totals = {}
        optimizer.zero_grad()
        accumulation = max(1, int(getattr(self.args, "update_epochs", 1)))
        split = self.cache["splits"]["train"]
        for step, batch in enumerate(tqdm(dataloader, leave=False), start=1):
            indices = _batch_cache_indices(batch.get("id"), self.index["train"])
            committee = _committee_from_split(
                split, indices, self.args.device, self.dispersion_temperature
            )
            text = batch["text"].to(self.args.device)
            audio = batch["audio"].to(self.args.device)
            vision = batch["vision"].to(self.args.device)
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            student = model(text, audio, vision)
            losses = consensus_distillation_loss(student, committee, labels, self.loss_weights)
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
        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    @torch.no_grad()
    def collect(self, model, dataloader, split_name):
        model.eval()
        output = {
            "sample_ids": [], "labels": [], "base_prediction": [],
            "prediction": [], "correction": [], "log_scale": [],
            "feature": [], "consensus": [], "dispersion": [], "agreement": [],
        }
        split = self.cache["splits"][split_name]
        for batch in tqdm(dataloader, leave=False):
            indices = _batch_cache_indices(batch.get("id"), self.index[split_name])
            committee = _committee_from_split(
                split, indices, self.args.device, self.dispersion_temperature
            )
            text = batch["text"].to(self.args.device)
            audio = batch["audio"].to(self.args.device)
            vision = batch["vision"].to(self.args.device)
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            student = model(text, audio, vision)
            output["sample_ids"].extend(normalize_batch_ids(batch.get("id")))
            output["labels"].append(labels.cpu())
            for key in ("base_prediction", "prediction", "correction", "log_scale", "feature"):
                output[key].append(student[key].detach().cpu())
            for key in ("consensus", "dispersion", "agreement"):
                output[key].append(committee[key].detach().cpu())
        return {
            key: value if key == "sample_ids" else torch.cat(value, dim=0)
            for key, value in output.items()
        }

    def train(self, model, dataloaders):
        history = []
        best = {"value": float("inf"), "epoch": 0, "state": None}
        global_epoch = 0
        stages = (("head", self.head_epochs), ("tail", max(0, self.max_epochs - self.head_epochs)))
        for stage, epochs in stages:
            if epochs <= 0:
                continue
            if stage == "head":
                model.freeze_backbone()
            else:
                model.unfreeze_backbone_tail()
            optimizer = self._optimizer(model, stage)
            scheduler = ReduceLROnPlateau(
                optimizer, mode="min", factor=0.5, patience=3, min_lr=1e-7
            )
            stage_best = global_epoch
            for _ in range(epochs):
                global_epoch += 1
                train_row = self._train_epoch(model, dataloaders["train"], optimizer)
                valid = self.collect(model, dataloaders["valid"], "valid")
                valid_mae = float(torch.abs(valid["prediction"] - valid["labels"]).mean().item())
                valid_base = float(torch.abs(valid["base_prediction"] - valid["labels"]).mean().item())
                valid_teacher = float(torch.abs(valid["consensus"] - valid["labels"]).mean().item())
                scheduler.step(valid_mae)
                row = {
                    "epoch": global_epoch,
                    "stage": stage,
                    **train_row,
                    "valid_mae": valid_mae,
                    "valid_base_mae": valid_base,
                    "valid_teacher_mae": valid_teacher,
                    "valid_residual_corr": safe_corr(
                        valid["prediction"] - valid["base_prediction"],
                        valid["labels"] - valid["base_prediction"],
                    ),
                    "valid_scale_error_corr": safe_corr(
                        torch.nn.functional.softplus(valid["log_scale"]),
                        torch.abs(valid["prediction"] - valid["labels"]),
                    ),
                    "valid_teacher_dispersion": float(valid["dispersion"].mean().item()),
                }
                history.append(row)
                logger.info(
                    "V7 epoch=%d stage=%s train=%.4f valid=%.4f base=%.4f teacher=%.4f res_corr=%.4f",
                    global_epoch, stage, train_row.get("total", float("nan")),
                    valid_mae, valid_base, valid_teacher, row["valid_residual_corr"],
                )
                if valid_mae < best["value"] - 1e-6:
                    best = {"value": valid_mae, "epoch": global_epoch, "state": _cpu_state_dict(model)}
                    stage_best = global_epoch
                if stage == "tail" and global_epoch - stage_best >= self.early_stop:
                    break
        if best["state"] is None:
            raise RuntimeError("V7 failed to select a validation checkpoint.")
        pd.DataFrame(history).to_csv(self.save_dir / "v7_training_history.csv", index=False)
        torch.save(best, self.save_dir / "function_consensus_v7_best.pth")
        return best

    def _calibrate_student(self, valid):
        rows = []
        best = None
        for alpha in (0.0, 0.25, 0.50, 0.75, 1.0):
            prediction = valid["base_prediction"] + alpha * (
                valid["prediction"] - valid["base_prediction"]
            )
            stats = _selection_stats(valid["base_prediction"], prediction, valid["labels"])
            objective = stats["mae"] + 0.10 * stats["harm_over_010_rate"]
            row = {"alpha": alpha, "objective": objective, **stats}
            rows.append(row)
            if best is None or (row["objective"], row["mae"]) < (best["objective"], best["mae"]):
                best = row
        return best, rows

    def _region_rows(self, name, anchor, prediction, labels):
        regions = region_index(labels)
        rows = []
        for index, region_name in enumerate(REGION_NAMES):
            mask = regions == index
            if mask.any():
                rows.append({
                    "model": name,
                    "region": region_name,
                    "count": int(mask.sum().item()),
                    "anchor_mae": float(torch.abs(anchor[mask] - labels[mask]).mean().item()),
                    "final_mae": float(torch.abs(prediction[mask] - labels[mask]).mean().item()),
                    "anchor_bias": float((labels[mask] - anchor[mask]).mean().item()),
                    "final_bias": float((labels[mask] - prediction[mask]).mean().item()),
                })
        return rows

    def evaluate_and_save(self, model, dataloaders, best):
        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        test = self.collect(model, dataloaders["test"], "test")
        student_policy, policy_rows = self._calibrate_student(valid)
        alpha = float(student_policy["alpha"])
        valid_student = valid["base_prediction"] + alpha * (
            valid["prediction"] - valid["base_prediction"]
        )
        test_student = test["base_prediction"] + alpha * (
            test["prediction"] - test["base_prediction"]
        )

        valid_split = self.cache["splits"]["valid"]
        test_split = self.cache["splits"]["test"]
        weights = _fit_robust_teacher_weights(
            valid_split["predictions"], valid_split["labels"], valid_split["sample_ids"]
        )
        valid_committees = _committee_baselines(valid_split, weights)
        test_committees = _committee_baselines(test_split, weights)

        hybrid_rows, best_hybrid = [], None
        for committee_name, valid_committee in valid_committees.items():
            for beta in (0.0, 0.25, 0.50, 0.75, 1.0):
                prediction = beta * valid_committee + (1.0 - beta) * valid_student
                mae = float(torch.abs(prediction - valid["labels"]).mean().item())
                row = {"committee": committee_name, "beta": beta, "valid_mae": mae}
                hybrid_rows.append(row)
                if best_hybrid is None or mae < best_hybrid["valid_mae"]:
                    best_hybrid = row
        test_hybrid = (
            float(best_hybrid["beta"]) * test_committees[best_hybrid["committee"]]
            + (1.0 - float(best_hybrid["beta"])) * test_student
        )

        result_rows = []
        named_predictions = {
            "student_base": test["base_prediction"],
            "student_raw": test["prediction"],
            "student_calibrated": test_student,
            **{f"committee_{name}": value for name, value in test_committees.items()},
            "hybrid_valid_selected": test_hybrid,
        }
        metrics = {}
        region_rows = []
        for name, prediction in named_predictions.items():
            metric = _safe_metrics(self.metrics_fn, prediction, test["labels"])
            stats = _selection_stats(test["base_prediction"], prediction, test["labels"])
            metrics[name] = {"metrics": metric, "selection": stats}
            result_rows.append({"model": name, **metric, **stats})
            region_rows.extend(self._region_rows(
                name, test["base_prediction"], prediction, test["labels"]
            ))

        teacher_rows = []
        for index, path in enumerate(self.teacher_paths):
            prediction = test_split["predictions"][:, index]
            teacher_rows.append({
                "teacher_index": index,
                "checkpoint": str(path),
                **_safe_metrics(self.metrics_fn, prediction, test_split["labels"]),
            })

        pd.DataFrame(policy_rows).to_csv(self.save_dir / "v7_student_policy_calibration.csv", index=False)
        pd.DataFrame(hybrid_rows).to_csv(self.save_dir / "v7_hybrid_calibration.csv", index=False)
        pd.DataFrame(result_rows).to_csv(self.save_dir / "v7_test_baseline_comparison.csv", index=False)
        pd.DataFrame(region_rows).to_csv(self.save_dir / "v7_test_region_diagnostics.csv", index=False)
        pd.DataFrame(teacher_rows).to_csv(self.save_dir / "v7_test_teacher_metrics.csv", index=False)

        predictions = {
            "sample_id": test["sample_ids"],
            "label": test["labels"].view(-1).numpy(),
        }
        for name, prediction in named_predictions.items():
            predictions[name] = prediction.view(-1).numpy()
        pd.DataFrame(predictions).to_csv(
            self.save_dir / "function_consensus_v7_predictions.csv", index=False
        )

        summary = {
            "method": "stable_function_space_consensus_distillation_v7",
            "seed": int(self.args.seed),
            "teacher_count": len(self.teacher_paths),
            "teacher_paths": [str(path) for path in self.teacher_paths],
            "selected_epoch": int(best["epoch"]),
            "selected_valid_mae": float(best["value"]),
            "student_policy": {"alpha": alpha},
            "robust_teacher_weights": weights.tolist(),
            "hybrid_policy": best_hybrid,
            "results": metrics,
            "loss_weights": self.loss_weights,
        }
        (self.save_dir / "function_consensus_v7_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        logger.info(
            "V7 TEST base=%.4f student=%.4f uniform=%.4f robust=%.4f hybrid=%.4f",
            metrics["student_base"]["metrics"]["MAE"],
            metrics["student_calibrated"]["metrics"]["MAE"],
            metrics["committee_uniform"]["metrics"]["MAE"],
            metrics["committee_robust_simplex"]["metrics"]["MAE"],
            metrics["hybrid_valid_selected"]["metrics"]["MAE"],
        )
        return summary
