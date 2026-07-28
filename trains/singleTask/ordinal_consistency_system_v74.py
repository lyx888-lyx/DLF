"""Training and evaluation for ordinal-consistent complementarity V7.4."""

from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd
import torch
import torch.nn.functional as F
from torch import nn, optim
from tqdm import tqdm

from .complementarity_v71 import (
    apply_global_committee,
    apply_region_committee,
    fit_committee_cv,
)
from .expert_analysis import normalize_batch_ids


LOGGER = logging.getLogger("MMSA")
THRESHOLDS_7 = (-2.5, -1.5, -0.5, 0.5, 1.5, 2.5)
THRESHOLDS_5 = (-1.5, -0.5, 0.5, 1.5)


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


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
        return torch.tensor(
            [mapping[str(value)] for value in values], dtype=torch.long
        )
    except KeyError as error:
        raise KeyError(f"Sample id missing from teacher cache: {error}") from error


def _group_ids(sample_ids, groups=3):
    return torch.tensor([
        int(hashlib.sha1(str(value).encode("utf-8")).hexdigest(), 16) % groups
        for value in sample_ids
    ], dtype=torch.long)


def _fast_metrics(prediction, labels):
    prediction = prediction.view(-1).detach().cpu()
    labels = labels.view(-1).detach().cpu()
    acc7 = (
        torch.round(prediction.clamp(-3.0, 3.0))
        == torch.round(labels.clamp(-3.0, 3.0))
    ).float().mean()
    acc5 = (
        torch.round(prediction.clamp(-2.0, 2.0))
        == torch.round(labels.clamp(-2.0, 2.0))
    ).float().mean()
    return {
        "mae": float(torch.abs(prediction - labels).mean().item()),
        "acc_7": float(acc7.item()),
        "acc_5": float(acc5.item()),
    }


def _fold_metrics(prediction, labels, sample_ids, folds=3):
    groups = _group_ids(sample_ids, folds)
    rows = []
    for fold in range(folds):
        mask = groups == fold
        if mask.any():
            rows.append(_fast_metrics(prediction[mask], labels[mask]))
    if not rows:
        raise RuntimeError("No validation folds were created.")
    mae = torch.tensor([row["mae"] for row in rows])
    acc7 = torch.tensor([row["acc_7"] for row in rows])
    acc5 = torch.tensor([row["acc_5"] for row in rows])
    return {
        "fold_rows": rows,
        "fold_mae_mean": float(mae.mean().item()),
        "fold_maes": [float(value) for value in mae.tolist()],
        "fold_acc7_std": float(acc7.std(unbiased=False).item()),
        "fold_acc5_std": float(acc5.std(unbiased=False).item()),
    }


def ordinal_consistency_loss(
    output: Mapping[str, torch.Tensor],
    labels: torch.Tensor,
    weights: Mapping[str, float],
    boundary_temperature: float = 4.0,
):
    labels = labels.view(-1, 1)
    thresholds7 = labels.new_tensor(THRESHOLDS_7).view(1, -1)
    thresholds5 = labels.new_tensor(THRESHOLDS_5).view(1, -1)
    target7 = (labels > thresholds7).float()
    clipped5 = labels.clamp(-2.0, 2.0)
    target5 = (clipped5 > thresholds5).float()

    prediction = output["prediction"]
    legacy = output["legacy_prediction"].detach()
    ordinal7 = output["ordinal7_value"]
    ordinal5 = output["ordinal5_value"]
    ordinal_consensus = output["ordinal_consensus"]

    losses = {
        "supervised": F.l1_loss(prediction, labels),
        "ordinal7": F.binary_cross_entropy_with_logits(
            output["ordinal7_logits"], target7
        ),
        "ordinal5": F.binary_cross_entropy_with_logits(
            output["ordinal5_logits"], target5
        ),
        "boundary7": F.binary_cross_entropy_with_logits(
            float(boundary_temperature) * (prediction - thresholds7), target7
        ),
        "boundary5": F.binary_cross_entropy_with_logits(
            float(boundary_temperature) * (prediction - thresholds5), target5
        ),
        "ordinal_mae": F.l1_loss(ordinal_consensus, labels.clamp(-3.0, 3.0)),
        "cross_scale_consistency": F.smooth_l1_loss(
            ordinal7.clamp(-2.0, 2.0), ordinal5
        ),
        "legacy_preservation": F.smooth_l1_loss(prediction, legacy),
        "blend_regularization": output["ordinal_blend"].square().mean(),
    }
    total = prediction.new_zeros(())
    for name, value in losses.items():
        total = total + float(weights.get(name, 0.0)) * value
    losses["total"] = total
    return losses


class OrdinalConsistencyTrainerV74:
    def __init__(
        self,
        args,
        metrics_fn,
        save_dir,
        teacher_cache,
        teacher_paths: Sequence[Path],
        init_index: int,
        source_summary: Mapping[str, object],
        max_epochs: int = 12,
        early_stop: int = 4,
        learning_rate: float = 3e-4,
        weight_decay: float = 1e-3,
        valid_mae_tolerance: float = 0.001,
        fold_mae_tolerance: float = 0.010,
        min_accuracy_gain: float = 0.002,
        folds: int = 3,
        boundary_temperature: float = 4.0,
        loss_weights=None,
    ):
        self.args = args
        self.metrics_fn = metrics_fn
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cache = teacher_cache
        self.teacher_paths = [Path(value) for value in teacher_paths]
        self.init_index = int(init_index)
        self.source_summary = dict(source_summary)
        self.max_epochs = int(max_epochs)
        self.early_stop = int(early_stop)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.valid_mae_tolerance = float(valid_mae_tolerance)
        self.fold_mae_tolerance = float(fold_mae_tolerance)
        self.min_accuracy_gain = float(min_accuracy_gain)
        self.folds = int(folds)
        self.boundary_temperature = float(boundary_temperature)
        self.loss_weights = loss_weights or {
            "supervised": 1.00,
            "ordinal7": 0.25,
            "ordinal5": 0.18,
            "boundary7": 0.16,
            "boundary5": 0.12,
            "ordinal_mae": 0.10,
            "cross_scale_consistency": 0.05,
            "legacy_preservation": 0.05,
            "blend_regularization": 0.01,
        }
        self.index = {
            split: _cache_index(self.cache["splits"][split]["sample_ids"])
            for split in ("train", "valid", "test")
        }
        self.committee = self._fit_committee()

    def _fit_committee(self):
        valid = self.cache["splits"]["valid"]
        anchor = valid["predictions"][:, self.init_index]
        fitted = fit_committee_cv(
            valid["predictions"],
            valid["labels"],
            anchor,
            valid["sample_ids"],
            temperature=0.55,
            steps=600,
        )
        pd.DataFrame(fitted["cv_rows"]).to_csv(
            self.save_dir / "v74_committee_cv.csv", index=False
        )
        return fitted

    def _committee_predictions(self, split_name):
        split = self.cache["splits"][split_name]
        predictions = split["predictions"].float()
        anchor = predictions[:, self.init_index]
        global_prediction = apply_global_committee(
            predictions, self.committee["global_weights"]
        )
        region_prediction = apply_region_committee(
            predictions,
            anchor,
            self.committee["region_weights"],
            0.55,
        )
        return {
            "global_simplex": global_prediction,
            "region_simplex": region_prediction,
            "anchor": anchor,
        }

    def _optimizer(self, model):
        parameters = model.trainable_parameters()
        if not parameters:
            raise RuntimeError("V7.4 has no trainable ordinal parameters.")
        return optim.AdamW(
            parameters,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def _train_epoch(self, model, dataloader, optimizer):
        model.set_train_mode()
        totals = {}
        optimizer.zero_grad()
        for batch in tqdm(dataloader, leave=False):
            text = batch["text"].to(self.args.device)
            audio = batch["audio"].to(self.args.device)
            vision = batch["vision"].to(self.args.device)
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            output = model(text, audio, vision)
            losses = ordinal_consistency_loss(
                output,
                labels,
                self.loss_weights,
                boundary_temperature=self.boundary_temperature,
            )
            losses["total"].backward()
            nn.utils.clip_grad_norm_(model.trainable_parameters(), 2.0)
            optimizer.step()
            optimizer.zero_grad()
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + float(value.detach().item())
        count = max(1, len(dataloader))
        return {name: value / count for name, value in totals.items()}

    @torch.no_grad()
    def collect(self, model, dataloader, split_name):
        model.eval()
        mapping = self.index[split_name]
        split = self.cache["splits"][split_name]
        count = len(split["sample_ids"])
        keys = (
            "base_prediction",
            "legacy_prediction",
            "prediction",
            "ordinal7_value",
            "ordinal5_value",
            "ordinal_consensus",
        )
        buffers = None
        seen = torch.zeros(count, dtype=torch.bool)
        for batch in tqdm(dataloader, leave=False):
            indices = _batch_cache_indices(batch.get("id"), mapping)
            output = model(
                batch["text"].to(self.args.device),
                batch["audio"].to(self.args.device),
                batch["vision"].to(self.args.device),
            )
            if buffers is None:
                buffers = {
                    key: torch.empty(
                        (count,) + tuple(output[key].shape[1:]),
                        dtype=output[key].dtype,
                    )
                    for key in keys
                }
            for key in keys:
                buffers[key][indices] = output[key].detach().cpu()
            seen[indices] = True
        if buffers is None or not bool(seen.all()):
            raise RuntimeError(
                f"Incomplete V7.4 collection for {split_name}: "
                f"missing={int((~seen).sum().item())}"
            )
        return {
            "sample_ids": list(split["sample_ids"]),
            "labels": split["labels"].clone(),
            **buffers,
        }

    def _reference_prediction(self, collected, committees):
        policy = self.source_summary["hybrid_policy"]
        alpha = float(self.source_summary["student_policy"]["alpha"])
        legacy = collected["base_prediction"] + alpha * (
            collected["legacy_prediction"] - collected["base_prediction"]
        )
        beta = float(policy["beta"])
        committee_name = str(policy["committee"])
        return beta * committees[committee_name] + (1.0 - beta) * legacy

    def _select_policy(self, collected, committees):
        labels = collected["labels"]
        sample_ids = collected["sample_ids"]
        reference = self._reference_prediction(collected, committees)
        reference_metrics = _fast_metrics(reference, labels)
        reference_folds = _fold_metrics(
            reference, labels, sample_ids, self.folds
        )
        reference_score = (
            reference_metrics["acc_7"] + reference_metrics["acc_5"]
            - 0.25 * (
                reference_folds["fold_acc7_std"]
                + reference_folds["fold_acc5_std"]
            )
        )
        reference_row = {
            "source": "legacy_reference",
            "committee": str(self.source_summary["hybrid_policy"]["committee"]),
            "beta": float(self.source_summary["hybrid_policy"]["beta"]),
            "alpha": float(self.source_summary["student_policy"]["alpha"]),
            "score": float(reference_score),
            "feasible": True,
            **reference_metrics,
            **{key: value for key, value in reference_folds.items() if key != "fold_rows"},
        }
        rows = [reference_row]
        feasible = [reference_row]
        mae_limit = reference_metrics["mae"] + self.valid_mae_tolerance
        fold_limits = [
            value + self.fold_mae_tolerance
            for value in reference_folds["fold_maes"]
        ]
        required_accuracy = (
            reference_metrics["acc_7"]
            + reference_metrics["acc_5"]
            + self.min_accuracy_gain
        )

        committee_names = tuple(dict.fromkeys((
            "global_simplex",
            "region_simplex",
            str(self.source_summary["hybrid_policy"]["committee"]),
        )))
        for committee_name in committee_names:
            if committee_name not in committees:
                continue
            for beta in (0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
                for alpha in (0.50, 0.75, 1.00):
                    ordinal_student = collected["base_prediction"] + float(alpha) * (
                        collected["prediction"] - collected["base_prediction"]
                    )
                    prediction = float(beta) * committees[committee_name] + (
                        1.0 - float(beta)
                    ) * ordinal_student
                    metrics = _fast_metrics(prediction, labels)
                    folds = _fold_metrics(
                        prediction, labels, sample_ids, self.folds
                    )
                    accuracy_sum = metrics["acc_7"] + metrics["acc_5"]
                    stability = 0.25 * (
                        folds["fold_acc7_std"] + folds["fold_acc5_std"]
                    )
                    score = accuracy_sum - stability
                    is_feasible = (
                        metrics["mae"] <= mae_limit + 1e-12
                        and accuracy_sum >= required_accuracy - 1e-12
                        and all(
                            folds["fold_maes"][index] <= fold_limits[index] + 1e-12
                            for index in range(self.folds)
                        )
                    )
                    row = {
                        "source": "ordinal_v74",
                        "committee": committee_name,
                        "beta": float(beta),
                        "alpha": float(alpha),
                        "score": float(score),
                        "feasible": bool(is_feasible),
                        **metrics,
                        **{key: value for key, value in folds.items() if key != "fold_rows"},
                    }
                    rows.append(row)
                    if is_feasible:
                        feasible.append(row)

        selected = max(
            feasible,
            key=lambda row: (
                row["score"],
                row["acc_7"] + row["acc_5"],
                min(row["acc_7"], row["acc_5"]),
                -row["mae"],
                row["source"] == "legacy_reference",
            ),
        )
        return selected, rows, reference, reference_metrics

    def train(self, model, dataloaders):
        model.freeze_legacy()
        optimizer = self._optimizer(model)
        history = []
        policy_rows = []

        initial_valid = self.collect(model, dataloaders["valid"], "valid")
        valid_committees = self._committee_predictions("valid")
        initial_policy, initial_rows, _, _ = self._select_policy(
            initial_valid, valid_committees
        )
        policy_rows.extend({"epoch": 0, **row} for row in initial_rows)
        best = {
            "epoch": 0,
            "score": float(initial_policy["score"]),
            "state": _cpu_state_dict(model),
            "policy": dict(initial_policy),
        }
        last_improvement = 0

        for epoch in range(1, self.max_epochs + 1):
            train_row = self._train_epoch(model, dataloaders["train"], optimizer)
            valid = self.collect(model, dataloaders["valid"], "valid")
            selected, rows, _, reference_metrics = self._select_policy(
                valid, valid_committees
            )
            policy_rows.extend({"epoch": epoch, **row} for row in rows)
            row = {
                "epoch": epoch,
                **train_row,
                "selected_source": selected["source"],
                "selected_score": selected["score"],
                "selected_mae": selected["mae"],
                "selected_acc_7": selected["acc_7"],
                "selected_acc_5": selected["acc_5"],
                "reference_mae": reference_metrics["mae"],
                "ordinal_blend": float(
                    model.max_ordinal_blend
                    * torch.sigmoid(model.ordinal_blend_logit.detach()).item()
                ),
            }
            history.append(row)
            LOGGER.info(
                "V7.4 epoch=%d source=%s Valid(MAE=%.4f Acc7=%.4f Acc5=%.4f) "
                "blend=%.4f",
                epoch,
                selected["source"],
                selected["mae"],
                selected["acc_7"],
                selected["acc_5"],
                row["ordinal_blend"],
            )
            improved = (
                selected["score"] > best["score"] + 1e-6
                or (
                    abs(selected["score"] - best["score"]) <= 1e-6
                    and selected["mae"] < best["policy"]["mae"] - 1e-6
                )
            )
            if improved:
                best = {
                    "epoch": epoch,
                    "score": float(selected["score"]),
                    "state": _cpu_state_dict(model),
                    "policy": dict(selected),
                }
                last_improvement = epoch
            if epoch - last_improvement >= self.early_stop:
                break

        pd.DataFrame(history).to_csv(
            self.save_dir / "v74_training_history.csv", index=False
        )
        pd.DataFrame(policy_rows).to_csv(
            self.save_dir / "v74_valid_policy_search.csv", index=False
        )
        torch.save(best, self.save_dir / "ordinal_consistency_v74_best.pth")
        return best

    def _apply_policy(self, collected, committees, policy):
        if policy["source"] == "legacy_reference":
            return self._reference_prediction(collected, committees)
        ordinal_student = collected["base_prediction"] + float(policy["alpha"]) * (
            collected["prediction"] - collected["base_prediction"]
        )
        return float(policy["beta"]) * committees[policy["committee"]] + (
            1.0 - float(policy["beta"])
        ) * ordinal_student

    def _bootstrap(self, reference, candidate, labels, samples=2000, seed=1184):
        generator = torch.Generator().manual_seed(int(seed))
        n = labels.numel()
        values = {name: torch.empty(samples) for name in ("mae", "acc7", "acc5")}
        for index in range(samples):
            draw = torch.randint(0, n, (n,), generator=generator)
            ref_metrics = _fast_metrics(reference.view(-1)[draw], labels.view(-1)[draw])
            cand_metrics = _fast_metrics(candidate.view(-1)[draw], labels.view(-1)[draw])
            values["mae"][index] = ref_metrics["mae"] - cand_metrics["mae"]
            values["acc7"][index] = cand_metrics["acc_7"] - ref_metrics["acc_7"]
            values["acc5"][index] = cand_metrics["acc_5"] - ref_metrics["acc_5"]

        def summarize(tensor):
            ordered = tensor.sort().values
            low = ordered[max(0, int(math.floor(0.025 * samples)))]
            high = ordered[min(samples - 1, int(math.ceil(0.975 * samples)) - 1)]
            return {
                "mean": float(tensor.mean().item()),
                "ci95_low": float(low.item()),
                "ci95_high": float(high.item()),
                "probability_positive": float((tensor > 0).float().mean().item()),
            }

        return {
            "mae_gain_positive_is_better": summarize(values["mae"]),
            "acc7_gain_positive_is_better": summarize(values["acc7"]),
            "acc5_gain_positive_is_better": summarize(values["acc5"]),
        }

    def evaluate_and_save(self, model, dataloaders, best, bootstrap_samples=2000):
        model.load_state_dict(best["state"])
        model.to(self.args.device)
        valid = self.collect(model, dataloaders["valid"], "valid")
        test = self.collect(model, dataloaders["test"], "test")
        valid_committees = self._committee_predictions("valid")
        test_committees = self._committee_predictions("test")

        # Re-run the frozen Valid selector for an auditable final policy. It must
        # agree with the checkpoint policy unless floating point ties occur.
        selected, rows, valid_reference, valid_reference_metrics = self._select_policy(
            valid, valid_committees
        )
        pd.DataFrame(rows).to_csv(
            self.save_dir / "v74_final_valid_policy.csv", index=False
        )
        test_reference = self._reference_prediction(test, test_committees)
        test_selected = self._apply_policy(test, test_committees, selected)

        named = {
            "anchor": test["base_prediction"],
            "global_simplex": test_committees["global_simplex"],
            "original_v71_hybrid": test_reference,
            "ordinal_v74_selected": test_selected,
        }
        results = {}
        rows_out = []
        for name, prediction in named.items():
            metric = {
                key: float(value)
                for key, value in self.metrics_fn(
                    prediction.detach().cpu(), test["labels"].detach().cpu()
                ).items()
            }
            results[name] = metric
            rows_out.append({"model": name, **metric})
        pd.DataFrame(rows_out).to_csv(
            self.save_dir / "v74_test_comparison.csv", index=False
        )
        pd.DataFrame({
            "sample_id": test["sample_ids"],
            "label": test["labels"].view(-1).tolist(),
            "original_v71_hybrid": test_reference.view(-1).tolist(),
            "ordinal_v74_selected": test_selected.view(-1).tolist(),
            "ordinal7_value": test["ordinal7_value"].view(-1).tolist(),
            "ordinal5_value": test["ordinal5_value"].view(-1).tolist(),
        }).to_csv(self.save_dir / "v74_test_predictions.csv", index=False)

        summary = {
            "method": "ordinal_consistent_complementarity_v7_4",
            "selection_protocol": (
                "Ordinal heads are trained on Train. Epoch and final hybrid policy "
                "are selected only on Valid under MAE and fold-stability constraints. "
                "Test is evaluated once after freezing the policy."
            ),
            "selected_epoch": int(best["epoch"]),
            "selected_policy": selected,
            "valid_reference_metrics": valid_reference_metrics,
            "valid_selected_metrics": _fast_metrics(
                self._apply_policy(valid, valid_committees, selected),
                valid["labels"],
            ),
            "test_results": results,
            "test_mae_below_070": bool(
                results["ordinal_v74_selected"]["MAE"] < 0.70
            ),
            "bootstrap_vs_original_v71": self._bootstrap(
                test_reference,
                test_selected,
                test["labels"],
                samples=int(bootstrap_samples),
                seed=int(getattr(self.args, "seed", 1111)) + 74,
            ),
            "loss_weights": dict(self.loss_weights),
            "committee": {
                "global_weights": self.committee["global_weights"].tolist(),
                "region_weights": self.committee["region_weights"].tolist(),
                "global_cv_score": self.committee["global_cv_score"],
                "region_cv_score": self.committee["region_cv_score"],
            },
        }
        (self.save_dir / "ordinal_consistency_v74_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        LOGGER.info(
            "V7.4 final source=%s Test(MAE=%.4f Acc7=%.4f Acc5=%.4f) "
            "reference_MAE=%.4f",
            selected["source"],
            results["ordinal_v74_selected"]["MAE"],
            results["ordinal_v74_selected"]["acc_7"],
            results["ordinal_v74_selected"]["acc_5"],
            results["original_v71_hybrid"]["MAE"],
        )
        return summary
