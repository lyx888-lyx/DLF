"""Safety corrections for V8 unimodal expert training."""
from __future__ import annotations

import json
from typing import Dict
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch import nn
from tqdm import tqdm

from .expert_analysis import normalize_batch_ids
from .unimodal_expert_system_v8 import (
    ACCEPTANCE_THRESHOLDS, UnimodalExpertTrainerV8, _cpu_state_dict,
    _safe_float, _spearman, dataset_diagnostics, fit_train_normalizer,
)


def _auc(y, score):
    try:
        return float(roc_auc_score(y, score))
    except ValueError:
        return float("nan")


def uncertainty_diagnostics_safe(prediction, raw_score, labels, scale, orientation):
    """Q1 is lowest risk and Q4 is highest risk; high values always mean error."""
    p = prediction.view(-1).detach().cpu().numpy().astype(np.float64)
    r = raw_score.view(-1).detach().cpu().numpy().astype(np.float64)
    y = labels.view(-1).detach().cpu().numpy().astype(np.float64)
    if not (len(p) == len(r) == len(y)):
        raise RuntimeError("prediction/uncertainty/label length mismatch")
    error = np.abs(p - y)
    target = np.tanh(error / max(float(scale), 1e-4))
    risk = r if orientation == "direct" else 1.0 - r
    high = (error >= np.quantile(error, 0.70)).astype(np.int64)
    groups = np.array_split(np.argsort(risk, kind="mergesort"), 4)
    qrows = [{
        "quartile": f"Q{i+1}", "count": int(len(g)),
        "mean_predicted_risk": float(risk[g].mean()),
        "true_mae": float(error[g].mean()),
    } for i, g in enumerate(groups)]
    q = [row["true_mae"] for row in qrows]
    return {
        "score_orientation": orientation,
        "error_spearman": _spearman(risk, error),
        "raw_error_spearman": _spearman(r, error),
        "target_abs_error_spearman": _spearman(target, error),
        "raw_target_spearman": _spearman(r, target),
        "high_error_auroc": _auc(high, risk),
        "raw_high_error_auroc": _auc(high, r),
        "oriented_target_l1": float(np.abs(risk - target).mean()),
        "quartile_rows": qrows, "quartile_mae": q,
        "q4_q1_ratio": float(q[-1] / q[0]) if q[0] > 1e-12 else float("nan"),
        "quartile_monotonic": bool(all(q[i] <= q[i+1] for i in range(3))),
        "target_distribution": {
            "min": float(target.min()), "p10": float(np.quantile(target, .1)),
            "p25": float(np.quantile(target, .25)), "median": float(np.median(target)),
            "p75": float(np.quantile(target, .75)), "p90": float(np.quantile(target, .9)),
            "max": float(target.max()), "mean": float(target.mean()),
            "std": float(target.std()),
        },
    }


def choose_orientation(prediction, raw_score, labels, scale):
    direct = uncertainty_diagnostics_safe(prediction, raw_score, labels, scale, "direct")
    inverse = uncertainty_diagnostics_safe(prediction, raw_score, labels, scale, "inverted")
    key = lambda d: (_safe_float(d["error_spearman"], -9),
                     _safe_float(d["high_error_auroc"], -9),
                     -_safe_float(d["oriented_target_l1"], 9))
    orientation = "direct" if key(direct) >= key(inverse) else "inverted"
    return orientation, direct, inverse


class SafeUnimodalExpertTrainerV8(UnimodalExpertTrainerV8):
    def __init__(self, *args, prediction_loss="mae", joint_mode="disabled", **kwargs):
        super().__init__(*args, **kwargs)
        if prediction_loss not in ("mae", "mse", "huber"):
            raise ValueError(prediction_loss)
        if joint_mode not in ("disabled", "detached", "shared"):
            raise ValueError(joint_mode)
        self.prediction_loss_name = prediction_loss
        self.joint_mode = joint_mode

    def _pred_loss(self, pred, labels):
        if self.prediction_loss_name == "mae": return F.l1_loss(pred, labels)
        if self.prediction_loss_name == "mse": return F.mse_loss(pred, labels)
        return F.smooth_l1_loss(pred, labels)

    @torch.no_grad()
    def collect(self, model, loader):
        model.eval(); buffers = {k: [] for k in ("prediction", "raw_score", "labels", "missing")}; ids = []
        for batch in tqdm(loader, leave=False):
            out = model(batch[self.batch_key].to(self.args.device))
            buffers["prediction"].append(out["prediction"].cpu())
            buffers["raw_score"].append(out["uncertainty"].cpu())
            buffers["labels"].append(batch["labels"]["M"].view(-1, 1).cpu())
            buffers["missing"].append(out["all_missing"].cpu())
            ids.extend(str(x) for x in normalize_batch_ids(batch.get("id")))
        if len(ids) != len(set(ids)): raise RuntimeError("duplicate sample ids")
        order = torch.tensor(sorted(range(len(ids)), key=lambda i: ids[i]), dtype=torch.long)
        return {"sample_ids": [ids[i] for i in order.tolist()],
                **{k: torch.cat(v)[order] for k, v in buffers.items()}}

    def evaluate(self, model, loader, orientation="direct"):
        c = self.collect(model, loader)
        metrics = self._metric_dict(self.metrics_fn, c["prediction"], c["labels"])
        diag = uncertainty_diagnostics_safe(c["prediction"], c["raw_score"], c["labels"],
                                            float(model.error_scale), orientation)
        c["risk"] = c["raw_score"] if orientation == "direct" else 1-c["raw_score"]
        return {"metrics": metrics, "uncertainty": diag, "collected": c,
                "missing_sample_rate": float(c["missing"].float().mean())}

    def _train_epoch(self, model, loader, optimizer, stage):
        model.set_stage_mode(stage); sums = dict(prediction=0., uncertainty=0., total=0.); n = 0
        detach = stage in ("uncertainty", "joint_detached")
        for batch in tqdm(loader, leave=False):
            labels = batch["labels"]["M"].to(self.args.device).view(-1, 1)
            out = model(batch[self.batch_key].to(self.args.device), detach_uncertainty_features=detach)
            lp = self._pred_loss(out["prediction"], labels)
            lu = F.smooth_l1_loss(out["uncertainty"], model.error_target(out["prediction"], labels))
            loss = lp if stage == "prediction" else lu if stage == "uncertainty" else lp+self.uncertainty_weight*lu
            optimizer.zero_grad(set_to_none=True); loss.backward()
            nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], self.grad_clip)
            optimizer.step(); n += 1
            for key, value in (("prediction", lp), ("uncertainty", lu), ("total", loss)):
                sums[key] += float(value.detach())
        return {k: v/max(1, n) for k, v in sums.items()}

    def _fit_uncertainty_stage(self, model, loaders):
        opt = self._optimizer(model, "uncertainty"); best = None; history = []
        for epoch in range(1, self.uncertainty_epochs+1):
            losses = self._train_epoch(model, loaders["train"], opt, "uncertainty")
            c = self.collect(model, loaders["valid"])
            orient, direct, inverse = choose_orientation(c["prediction"], c["raw_score"], c["labels"], float(model.error_scale))
            selected = direct if orient == "direct" else inverse
            history.append({"stage":"uncertainty","epoch":epoch,**{f"train_{k}":v for k,v in losses.items()},
                            "valid_score_orientation":orient,"valid_error_spearman":selected["error_spearman"],
                            "valid_high_error_auroc":selected["high_error_auroc"],
                            "valid_target_abs_error_spearman":selected["target_abs_error_spearman"]})
            key = (_safe_float(selected["error_spearman"],-9), _safe_float(selected["high_error_auroc"],-9),
                   -_safe_float(selected["oriented_target_l1"],9))
            if best is None or key > best["key"]:
                best = {"epoch":epoch,"key":key,"orientation":orient,"diagnostics":selected,
                        "direct_diagnostics":direct,"inverted_diagnostics":inverse,"state":_cpu_state_dict(model)}
        model.load_state_dict(best["state"]); torch.save(best, self.save_dir/"uncertainty_stage_best.pth")
        return best, history

    def _fit_joint_safe(self, model, loaders, orientation):
        if self.joint_mode == "disabled" or self.joint_epochs <= 0: return None, []
        stage = "joint_detached" if self.joint_mode == "detached" else "joint_shared"
        opt = self._optimizer(model, stage); initial = self.evaluate(model, loaders["valid"], orientation)
        best = {"epoch":0,"metrics":initial["metrics"],"uncertainty":initial["uncertainty"],"state":_cpu_state_dict(model)}; hist=[]
        for epoch in range(1, self.joint_epochs+1):
            losses = self._train_epoch(model, loaders["train"], opt, stage)
            val = self.evaluate(model, loaders["valid"], orientation); metrics = val["metrics"]
            hist.append({"stage":stage,"epoch":epoch,**{f"train_{k}":v for k,v in losses.items()},
                         **{f"valid_{k}":v for k,v in metrics.items()}})
            if self._prediction_is_better(metrics, best["metrics"]):
                best = {"epoch":epoch,"metrics":metrics,"uncertainty":val["uncertainty"],"state":_cpu_state_dict(model)}
        return best, hist

    @staticmethod
    def _invariance(a, b):
        if a["sample_ids"] != b["sample_ids"] or not torch.equal(a["labels"], b["labels"]):
            raise RuntimeError("Stage A/B alignment mismatch")
        diff = (a["prediction"]-b["prediction"]).abs()
        return {"max_abs_prediction_difference":float(diff.max()),
                "mean_abs_prediction_difference":float(diff.mean()),
                "within_1e_7":bool(float(diff.max()) <= 1e-7)}

    def train_and_evaluate(self, model, loaders, normalizer_info=None):
        a, ha = self._fit_prediction_stage(model, loaders); model.load_state_dict(a["state"])
        aval = self.evaluate(model, loaders["valid"]); scale_loader = loaders.get("train_eval", loaders["train"])
        scale = self._fit_error_scale(model, scale_loader)
        b, hb = self._fit_uncertainty_stage(model, loaders); model.load_state_dict(b["state"])
        orient = b["orientation"]; bval = self.evaluate(model, loaders["valid"], orient)
        invariant = self._invariance(aval["collected"], bval["collected"])
        if not invariant["within_1e_7"]: raise RuntimeError(f"Stage B changed predictions: {invariant}")
        c, hc = self._fit_joint_safe(model, loaders, orient); accepted = False
        if c is not None and c["epoch"] > 0:
            m, u = c["metrics"], c["uncertainty"]
            accepted = bool(m["MAE"] <= a["metrics"]["MAE"]+self.max_prediction_degradation and
                            m["Corr"] >= a["metrics"]["Corr"]-self.max_corr_degradation and
                            _safe_float(u["error_spearman"],-1)>0 and _safe_float(u["high_error_auroc"],0)>.5 and
                            self._prediction_is_better(m, a["metrics"]))
        model.load_state_dict(c["state"] if accepted else b["state"])
        selected = f"joint_{self.joint_mode}" if accepted else "uncertainty_head_only"
        valid = self.evaluate(model, loaders["valid"], orient); test = self.evaluate(model, loaders["test"], orient)
        checks = {"mae":test["metrics"]["MAE"]<=ACCEPTANCE_THRESHOLDS[self.modality]["mae"],
                  "corr":test["metrics"]["Corr"]>=ACCEPTANCE_THRESHOLDS[self.modality]["corr"],
                  "uncertainty_direction":test["uncertainty"]["error_spearman"]>0 and test["uncertainty"]["high_error_auroc"]>.5 and test["uncertainty"]["q4_q1_ratio"]>1,
                  "target_direction_sanity":test["uncertainty"]["target_abs_error_spearman"]>.99}
        pd.DataFrame(ha+hb+hc).to_csv(self.save_dir/"unimodal_expert_v8_history.csv",index=False)
        pd.DataFrame([{"stage":"prediction_only",**a["metrics"]},{"stage":"error_head_only",**bval["metrics"]},
                      {"stage":"final_selected",**valid["metrics"]}]).to_csv(self.save_dir/"unimodal_expert_v8_stage_valid_comparison.csv",index=False)
        d=test["collected"]; pred=d["prediction"].view(-1); labels=d["labels"].view(-1)
        frame=pd.DataFrame({"sample_id":d["sample_ids"],"label":labels.tolist(),"prediction":pred.tolist(),
                            "absolute_error":(pred-labels).abs().tolist(),
                            "uncertainty_target":torch.tanh((pred-labels).abs()/max(scale,1e-4)).tolist(),
                            "raw_uncertainty":d["raw_score"].view(-1).tolist(),"uncertainty_risk":d["risk"].view(-1).tolist(),
                            "score_orientation":orient})
        frame.to_csv(self.save_dir/"unimodal_expert_v8_test_predictions.csv",index=False)
        frame.sort_values("absolute_error",ascending=False).head(50).to_csv(self.save_dir/"unimodal_expert_v8_error_audit_top50.csv",index=False)
        pd.DataFrame(test["uncertainty"]["quartile_rows"]).to_csv(self.save_dir/"unimodal_expert_v8_uncertainty_quartiles.csv",index=False)
        summary={"method":"safe_staged_unimodal_expert_v8","modality":self.modality,"selected_stage":selected,
                 "joint_mode":self.joint_mode,"joint_accepted":accepted,"score_orientation":orient,
                 "prediction_loss":self.prediction_loss_name,"error_scale_q75_train":scale,"normalizer":normalizer_info or {},
                 "model_profile":model.model_profile(),"stage_a":{"selected_epoch":a["epoch"],"valid_metrics":a["metrics"]},
                 "stage_b":{"selected_epoch":b["epoch"],"prediction_invariance":invariant,
                            "direct_diagnostics":b["direct_diagnostics"],"inverted_diagnostics":b["inverted_diagnostics"]},
                 "stage_c":None if c is None else {"selected_epoch":c["epoch"],"valid_metrics":c["metrics"]},
                 "valid":{"metrics":valid["metrics"],"uncertainty":valid["uncertainty"]},
                 "test":{"metrics":test["metrics"],"uncertainty":test["uncertainty"],"missing_sample_rate":test["missing_sample_rate"]},
                 "acceptance":{"checks":checks,"all_pass":bool(all(checks.values()))}}
        torch.save({"state":_cpu_state_dict(model),"summary":summary},self.save_dir/"unimodal_expert_v8_best.pth")
        (self.save_dir/"unimodal_expert_v8_summary.json").write_text(json.dumps(summary,indent=2,allow_nan=True),encoding="utf-8")
        return summary
