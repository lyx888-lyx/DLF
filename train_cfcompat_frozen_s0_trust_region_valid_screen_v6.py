"""CFCompatKD v6: frozen-S0 function-level trust-region, Seed1113 Valid-only.

One trajectory only.  v4 DISTILL/PRESERVE/ABSTAIN is unchanged; v6 adds a
one-sided safe loss toward cached pre-training Student (S0) predictions on
Train events where S0 already beats the frozen ModDrop baseline by 0.02.
Official Test is never constructed or accessed.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_cfcompat_regret_preserve_valid_screen as v4
import train_cfcompat_safe_projection_valid_screen as base
from train_cf_compat_kd import batch_to_device
from trains.singleTask.cf_compat_kd_utils import compatibility_for_modes, gated_kd_loss
from trains.singleTask.cfcompat_regret_preserve_utils import (
    DISTILL_MARGIN, LAMBDA_PRESERVE, MILD_CFCOMPAT_BASE, MILD_CFCOMPAT_SCALE,
    PRESERVE_MARGIN, regret_preserve_decision,
)
from trains.singleTask.cfcompat_s0_trust_region_utils import (
    BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED, DEV_SEED,
    J_MAX_DEGRADATION_VS_V4, LAMBDA_S0_TRUST, METHOD,
    NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION, OUTPUT_TAG,
    OVERALL_NTR_MAX_DEGRADATION, RUN, RUNS, S0_PROTECT_MARGIN, VERSION,
    development_signal_gate, jsonable, mechanism_transfer_summary,
    s0_regret_projection_summary, s0_trust_decision,
)
from trains.singleTask.cfcompat_safe_projection_utils import derive_valid_events
from trains.singleTask.cfcompat_stability_utils import preserve_rng_state
from trains.singleTask.fixed_kd_utils import checkpoint_sha256, teacher_lav_prediction
from trains.singleTask.missing_utils import MISSING_MODES, compute_full_dlf_loss, compute_task_loss, mode_to_mask

_ORIGINAL_TRAIN_TRAJECTORY = base.train_trajectory
_ACTIVE_DECISIONS = None
_ACTIVE_S0_TRAIN = None
_ACTIVE_S0_VALID = None
_ACTIVE_VALID_REFERENCE = None


class FrozenS0Bundle:
    def __init__(self, v4_bundle, s0_train, s0_valid):
        self.valid_reference = v4_bundle.valid_reference.copy()
        self.train_baseline = v4_bundle.train_baseline.copy()
        self.train_by_index = dict(v4_bundle.train_by_index)
        self.s0_train = s0_train.copy()
        self.s0_valid = s0_valid.copy()
        self.s0_by_index = {int(r.sample_index): r._asdict() for r in s0_train.itertuples(index=False)}

    def parameters(self):
        return iter(())


def parse_args():
    p = argparse.ArgumentParser(description="Frozen-S0 trust-region CFCompatKD v6")
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--num-workers", type=int, default=1)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--result-root", default="result")
    p.add_argument("--log-dir", default="log/missing_baseline")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.num_workers != 1:
        p.error("v6 fixes num_workers=1")
    a.seeds = [DEV_SEED]
    a.max_epochs = 2 if a.smoke_test else None
    return a


def result_paths(cli):
    out = Path(cli.result_root) / "missing_baseline" / OUTPUT_TAG / cli.dataset / "valid_screen" / "seed1113_dev"
    model = Path(cli.model_save_dir) / "missing_baseline" / OUTPUT_TAG / cli.dataset / "valid_screen" / "seed1113_dev"
    if cli.smoke_test:
        out, model = out / "smoke", model / "smoke"
    if out.exists() or model.exists():
        if not cli.overwrite:
            raise FileExistsError("v6 output exists; use --overwrite: {} / {}".format(out, model))
        if out.exists(): shutil.rmtree(out)
        if model.exists(): shutil.rmtree(model)
    out.mkdir(parents=True, exist_ok=True); model.mkdir(parents=True, exist_ok=True)
    return out, model


def create_logger(cli):
    d = Path(cli.log_dir); d.mkdir(parents=True, exist_ok=True)
    kind = "smoke" if cli.smoke_test else "formal"
    path = d / "DLF-mosi-frozen-s0-trust-v6-{}-{}.log".format(kind, datetime.now().strftime("%Y%m%d-%H%M%S"))
    log = logging.getLogger("frozen_s0_trust_v6"); log.handlers.clear(); log.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for h in (logging.FileHandler(path), logging.StreamHandler()): h.setFormatter(fmt); log.addHandler(h)
    return log, path


def cache_s0(student, loader, device, expected):
    gen = getattr(loader, "generator", None)
    state = gen.get_state().clone() if gen is not None else None
    try:
        with preserve_rng_state():
            frame = base.prediction_rows(student, loader, device)
    finally:
        if state is not None: gen.set_state(state)
    if state is not None and not torch.equal(gen.get_state(), state):
        raise RuntimeError("loader generator changed during S0 cache")
    if len(frame) != expected or frame.sample_index.nunique() != expected:
        raise RuntimeError("S0 cache count mismatch")
    cols = ["label"] + ["{}_pred".format(m) for m in ("LAV",) + MISSING_MODES]
    if not np.isfinite(frame[cols].to_numpy(dtype=np.float64)).all():
        raise FloatingPointError("S0 cache contains NaN/Inf")
    return frame.sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def load_assets(cli, args, loaders, seed):
    global _ACTIVE_S0_TRAIN, _ACTIVE_S0_VALID, _ACTIVE_VALID_REFERENCE
    if int(seed) != DEV_SEED: raise RuntimeError("v6 is locked to Seed1113")
    teacher, student, v4_bundle, assets = v4.load_assets(cli, args, loaders, seed)
    s0_train = cache_s0(student, loaders["train"], args.device, 1284)
    s0_valid = cache_s0(student, loaders["valid"], args.device, 229)
    bundle = FrozenS0Bundle(v4_bundle, s0_train, s0_valid)
    _ACTIVE_S0_TRAIN, _ACTIVE_S0_VALID = s0_train.copy(), s0_valid.copy()
    _ACTIVE_VALID_REFERENCE = bundle.valid_reference.copy()
    assets.update({"s0_anchor": "cached_pre_training_student", "s0_protect_margin": S0_PROTECT_MARGIN,
                   "lambda_s0_trust": LAMBDA_S0_TRUST, "s0_live_model_retained": False})
    return teacher, student, bundle, assets


def s0_for_modes(bundle, indices, modes, labels, device, dtype):
    vals = []
    for off, (idx, mode) in enumerate(zip(indices, modes)):
        rec = bundle.s0_by_index[int(idx)]
        if abs(float(rec["label"]) - float(labels[off].detach().cpu())) > 1e-6:
            raise RuntimeError("S0 label binding changed at sample {}".format(idx))
        vals.append(float(rec["{}_pred".format(mode)]))
    out = torch.as_tensor(vals, device=device, dtype=dtype).view(-1, 1)
    if not torch.isfinite(out).all(): raise FloatingPointError("non-finite S0 binding")
    return out


def forward_objective(run, batch, missing_mask, modes, args, teacher, bundle, student,
                      cache_by_index, criterion, cosine, hinge):
    global _ACTIVE_DECISIONS
    if run != RUN or _ACTIVE_DECISIONS is None: raise RuntimeError("invalid v6 recorder/run")
    text, audio, vision, labels = batch_to_device(batch, args.device)
    missing_mask = missing_mask.to(device=args.device, dtype=audio.dtype)
    full_mask = mode_to_mask("LAV", labels.size(0), args.device, audio.dtype)
    full_output = student(text, audio, vision, full_mask)
    full_loss, _ = compute_full_dlf_loss(full_output, labels, criterion, cosine, hinge)
    missing_output = student(text, audio, vision, missing_mask)
    missing_loss, _ = compute_task_loss(missing_output, labels, criterion)
    current = missing_output["output_logit"].view(-1, 1)
    teacher_pred = teacher_lav_prediction(teacher, text, audio, vision).view(-1, 1)
    indices = batch["index"].view(-1).cpu().numpy().astype(int).tolist(); modes = list(modes)
    compat = compatibility_for_modes(cache_by_index, indices, modes, args.device, labels.dtype).view(-1)
    baseline = v4.baseline_for_modes(bundle, indices, modes, labels, args.device, labels.dtype)
    s0 = s0_for_modes(bundle, indices, modes, labels, args.device, labels.dtype)

    dec = regret_preserve_decision(current.detach(), teacher_pred, baseline, labels, compat)
    kd_loss, each_kd = gated_kd_loss(current, dec["teacher_safe_target"], dec["distill_gate"])
    each_pres = F.smooth_l1_loss(current.view(-1), dec["preserve_safe_target"].view(-1), reduction="none")
    pw = dec["preserve_gate"].to(each_pres); pres_loss = torch.sum(pw * each_pres) / (pw.sum() + 1e-8)
    trust = s0_trust_decision(current.detach(), s0, baseline, labels)
    each_trust = F.smooth_l1_loss(current.view(-1), trust["s0_safe_target"].view(-1), reduction="none")
    tw = trust["s0_trust_gate"].to(each_trust); trust_loss = torch.sum(tw * each_trust) / (tw.sum() + 1e-8)
    total = full_loss + missing_loss + kd_loss + LAMBDA_PRESERVE * pres_loss + LAMBDA_S0_TRUST * trust_loss
    if not torch.isfinite(total): raise FloatingPointError("NaN/Inf in v6 objective")

    ids = list(batch["id"]); records = []; proj = dec["teacher_projection"]
    for off in range(labels.size(0)):
        r = {k: (bool(v[off].detach().cpu()) if v.dtype == torch.bool else float(v[off].detach().cpu())) for k, v in proj.items()}
        r.update({
            "sample_index": int(indices[off]), "sample_id": str(ids[off]), "mode": str(modes[off]),
            "label": float(labels[off].detach().cpu()), "student_prediction": float(current[off].detach().cpu()),
            "baseline_prediction": float(baseline[off].detach().cpu()), "teacher_prediction": float(teacher_pred[off].detach().cpu()),
            "s0_prediction": float(s0[off].detach().cpu()), "teacher_safe_target": float(dec["teacher_safe_target"][off].detach().cpu()),
            "preserve_safe_target": float(dec["preserve_safe_target"][off].detach().cpu()), "distill": bool(dec["distill"][off].detach().cpu()),
            "preserve": bool(dec["preserve"][off].detach().cpu()), "decision_abstain": bool(dec["abstain"][off].detach().cpu()),
            "teacher_beneficial": bool(dec["teacher_beneficial"][off].detach().cpu()), "current_regressed": bool(dec["current_regressed"][off].detach().cpu()),
            "baseline_error": float(dec["baseline_error"][off].detach().cpu()), "teacher_error": float(dec["teacher_error"][off].detach().cpu()),
            "current_error": float(dec["current_error"][off].detach().cpu()), "teacher_advantage_vs_baseline": float(dec["teacher_advantage_vs_baseline"][off].detach().cpu()),
            "current_regret_vs_baseline": float(dec["current_regret_vs_baseline"][off].detach().cpu()), "compatibility": float(compat[off].detach().cpu()),
            "mild_compatibility": float(dec["mild_compatibility"][off].detach().cpu()), "distill_gate": float(dec["distill_gate"][off].detach().cpu()),
            "preserve_gate": float(dec["preserve_gate"][off].detach().cpu()), "distill_loss_each": float(each_kd[off].detach().cpu()),
            "preserve_loss_each": float(each_pres[off].detach().cpu()), "s0_safe_target": float(trust["s0_safe_target"][off].detach().cpu()),
            "s0_error": float(trust["s0_error"][off].detach().cpu()), "s0_advantage_vs_baseline": float(trust["s0_advantage_vs_baseline"][off].detach().cpu()),
            "current_regret_vs_s0": float(trust["current_regret_vs_s0"][off].detach().cpu()), "s0_protected": bool(trust["s0_protected"][off].detach().cpu()),
            "s0_trust_active": bool(trust["s0_trust_active"][off].detach().cpu()), "s0_trust_gate": float(trust["s0_trust_gate"][off].detach().cpu()),
            "s0_trust_loss_each": float(each_trust[off].detach().cpu()), "crossed_baseline_from_s0": bool(trust["crossed_baseline_from_s0"][off].detach().cpu()),
        })
        r["event_ordinal"] = len(_ACTIVE_DECISIONS) + 1; records.append(r); _ACTIVE_DECISIONS.append(dict(r))
    diag = {"full_loss": float(full_loss.detach().cpu()), "missing_loss": float(missing_loss.detach().cpu()),
            "kd_loss": float(kd_loss.detach().cpu()), "mean_gate": float(dec["distill_gate"].mean().cpu()),
            "weighted_kd": float(kd_loss.detach().cpu()),
            "baseline_missing_MAE": float(torch.abs(baseline.view(-1)-labels.view(-1)).mean().cpu()),
            "safe_target_MAE": float(torch.abs(dec["teacher_safe_target"].view(-1)-labels.view(-1)).mean().cpu())}
    return total, diag, records


def reference_prediction_rows(bundle, teacher, loader, device):
    del teacher, loader, device
    return bundle.valid_reference.copy()


def train_trajectory(cli, logger, output_root, model_root):
    global _ACTIVE_DECISIONS, _ACTIVE_S0_TRAIN, _ACTIVE_S0_VALID, _ACTIVE_VALID_REFERENCE
    v4._ACTIVE_TRAIN_BASELINE_FRAME = None; _ACTIVE_DECISIONS = []
    _ACTIVE_S0_TRAIN = _ACTIVE_S0_VALID = _ACTIVE_VALID_REFERENCE = None
    try:
        result, epochs, raw = _ORIGINAL_TRAIN_TRAJECTORY(cli, logger, output_root, model_root, DEV_SEED, RUN)
        baseline = v4._ACTIVE_TRAIN_BASELINE_FRAME.copy(); decisions = pd.DataFrame(_ACTIVE_DECISIONS)
        s0_train, s0_valid, valid_ref = _ACTIVE_S0_TRAIN.copy(), _ACTIVE_S0_VALID.copy(), _ACTIVE_VALID_REFERENCE.copy()
    finally:
        v4._ACTIVE_TRAIN_BASELINE_FRAME = None; _ACTIVE_DECISIONS = None
        _ACTIVE_S0_TRAIN = _ACTIVE_S0_VALID = _ACTIVE_VALID_REFERENCE = None
    expected = 1284 * int(result["TrainEpochCount"])
    if len(decisions) != expected: raise RuntimeError("v6 decision count mismatch")
    decisions["Epoch"] = ((decisions.event_ordinal.astype(int)-1)//1284)+1
    run_dir = output_root / "seed1113" / RUN; run_dir.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(run_dir/"train_frozen_moddrop_baseline_cache.csv", index=False)
    s0_train.to_csv(run_dir/"train_frozen_s0_prediction_cache.csv", index=False)
    s0_valid.to_csv(run_dir/"valid_frozen_s0_prediction_cache.csv", index=False)
    decisions.to_csv(run_dir/"train_s0_trust_decisions.csv", index=False)
    result.update({"Method": METHOD, "S0ProtectMargin": S0_PROTECT_MARGIN, "LambdaS0Trust": LAMBDA_S0_TRUST})
    return result, epochs, raw, baseline, decisions, s0_train, s0_valid, valid_ref


def load_v4_reference(cli):
    root = Path(cli.result_root)/"missing_baseline"/"cfcompat_regret_preserve_v4"/cli.dataset/"valid_screen"/"seed1113_dev"
    grid_p = root/"regret_preserve_v4_candidate_grid.csv"; raw_p = root/"regret_preserve_v4_candidate_raw_valid_events.csv"
    if not grid_p.is_file() or not raw_p.is_file(): raise FileNotFoundError("missing frozen v4 reference")
    grid, raw = pd.read_csv(grid_p), pd.read_csv(raw_p)
    if len(grid) != 1 or int(grid.iloc[0].Seed) != DEV_SEED or set(raw.Split.astype(str)) != {"valid"}:
        raise RuntimeError("invalid frozen v4 reference")
    return grid.iloc[0].to_dict(), raw


def build_drift(events, s0_valid):
    rows = []
    for r in s0_valid.itertuples(index=False):
        for mode in MISSING_MODES: rows.append({"sample_index": int(r.sample_index), "Mode": mode, "s0_prediction": float(getattr(r, mode+"_pred"))})
    local = events.loc[events.Mode.astype(str).isin(MISSING_MODES)].merge(pd.DataFrame(rows), on=["sample_index","Mode"], validate="one_to_one")
    local["s0_error"] = np.abs(local.s0_prediction-local.label); local["s0_gain_vs_baseline"] = local.baseline_error-local.s0_error
    local["final_abs_drift_from_s0"] = np.abs(local.candidate_prediction-local.s0_prediction)
    local["s0_beneficial"] = local.s0_gain_vs_baseline >= S0_PROTECT_MARGIN
    local["teacher_beneficial_margin"] = local.teacher_advantage >= DISTILL_MARGIN
    local["final_crossed_baseline_from_s0"] = (local.s0_prediction-local.baseline_prediction)*(local.candidate_prediction-local.baseline_prediction) < 0
    local["final_worse_than_s0_by_margin"] = local.candidate_error-local.s0_error >= S0_PROTECT_MARGIN
    return local


def drift_summary(frame):
    specs = {"ALL": np.ones(len(frame), bool), "S0_BENEFICIAL": frame.s0_beneficial.to_numpy(bool),
             "TEACHER_BENEFICIAL": frame.teacher_beneficial_margin.to_numpy(bool),
             "S0_AND_TEACHER_BENEFICIAL": frame.s0_beneficial.to_numpy(bool)&frame.teacher_beneficial_margin.to_numpy(bool)}
    rows=[]
    for name, mask in specs.items():
        x=frame.loc[mask]
        if x.empty: continue
        rows.append({"group":name,"N":len(x),"mean_abs_final_drift_from_s0":float(x.final_abs_drift_from_s0.mean()),
                     "final_crossed_baseline_from_s0_rate":float(x.final_crossed_baseline_from_s0.mean()),
                     "final_worse_than_s0_by_margin_rate":float(x.final_worse_than_s0_by_margin.mean()),
                     "s0_mean_gain_vs_baseline":float(x.s0_gain_vs_baseline.mean()),"final_mean_gain_vs_baseline":float(x.gain_vs_dlf.mean()),
                     "final_negative_transfer_rate":float((x.gain_vs_dlf < -0.02).mean()),"final_severe_negative_transfer_rate":float((x.gain_vs_dlf < -0.10).mean())})
    return pd.DataFrame(rows)


def bind_hooks():
    base.RUNS=RUNS; base.METHOD=METHOD; base.load_assets=load_assets; base.forward_objective=forward_objective
    base.reference_prediction_rows=reference_prediction_rows; base.projection_summary=s0_regret_projection_summary


def main():
    cli=parse_args(); bind_hooks(); out, model=result_paths(cli); logger, log_path=create_logger(cli); v4_grid, v4_raw=load_v4_reference(cli)
    result, epochs, raw, baseline, decisions, s0_train, s0_valid, valid_ref=train_trajectory(cli, logger, out, model)
    raw=pd.DataFrame(raw); events=derive_valid_events(raw); v4_events=derive_valid_events(v4_raw)
    cand_transfer=mechanism_transfer_summary(events,RUN); v4_run=str(v4_raw.Run.iloc[0]); v4_transfer=mechanism_transfer_summary(v4_events,v4_run)
    drift=build_drift(events,s0_valid); drift_sum=drift_summary(drift)
    gate=None if cli.smoke_test else development_signal_gate(float(result["J_valid"]),float(v4_grid["J_valid"]),cand_transfer,v4_transfer)
    verdict="SMOKE_ONLY_NO_MECHANISM_DECISION" if gate is None else ("MECHANISM_SIGNAL_POSITIVE_FROZEN_S0_TRUST" if gate["passed"] else "MECHANISM_SIGNAL_NEGATIVE_OR_MIXED_FROZEN_S0_TRUST")
    transfers=[]
    for src, sm in (("v6_candidate",cand_transfer),("v4_frozen",v4_transfer)):
        for group in ("all_missing","teacher_beneficial","teacher_nonbeneficial"): transfers.append({"source":src,"group":group,**sm[group]})
    artifacts={"frozen_s0_v6_candidate_grid.csv":pd.DataFrame([result]),"frozen_s0_v6_epoch_metrics.csv":pd.DataFrame(epochs),
               "frozen_s0_v6_candidate_raw_valid_events.csv":raw,"frozen_s0_v6_candidate_valid_events.csv":events,
               "frozen_s0_v6_valid_s0_drift_events.csv":drift,"frozen_s0_v6_valid_s0_drift_summary.csv":drift_sum,
               "frozen_s0_v6_transfer_summary.csv":pd.DataFrame(transfers),"frozen_s0_v6_train_baseline_cache.csv":baseline,
               "frozen_s0_v6_train_s0_cache.csv":s0_train,"frozen_s0_v6_valid_s0_cache.csv":s0_valid,
               "frozen_s0_v6_train_decisions.csv":decisions,"frozen_s0_v6_valid_reference_cache.csv":valid_ref,
               "frozen_s0_v6_v4_reference_raw_valid_events.csv":v4_raw}
    for name, frame in artifacts.items(): frame.to_csv(out/name,index=False)
    summary={"version":VERSION,"method":METHOD,"verdict":verdict,"mechanism_signal_gate":jsonable(gate) if gate else None,
             "candidate_transfer":jsonable(cand_transfer),"frozen_v4_transfer":jsonable(v4_transfer),"valid_s0_drift_summary":jsonable(drift_sum.to_dict("records")),
             "protocol":{"development_seed":DEV_SEED,"new_trajectories_trained":1,"base_objective":"v4_unchanged_plus_s0_trust",
                         "s0_anchor":"cached_pre_training_student","s0_protect_margin":S0_PROTECT_MARGIN,"lambda_s0_trust":LAMBDA_S0_TRUST,
                         "distill_margin":DISTILL_MARGIN,"preserve_margin":PRESERVE_MARGIN,"lambda_preserve":LAMBDA_PRESERVE,
                         "mild_cfcompat":"{:.2f}+{:.2f}*compatibility".format(MILD_CFCOMPAT_BASE,MILD_CFCOMPAT_SCALE),
                         "checkpoint_selection":"minimum_official_valid_J","official_test_constructed":False,"official_test_accessed":False,"no_gate_classifier_tuning":True},
             "frozen_signal_thresholds":{"J_max_degradation_vs_v4":J_MAX_DEGRADATION_VS_V4,
                "beneficial_teacher_NTR_reduction_required":BENEFICIAL_TEACHER_NTR_REDUCTION_REQUIRED,
                "nonbeneficial_teacher_NTR_max_degradation":NONBENEFICIAL_TEACHER_NTR_MAX_DEGRADATION,
                "overall_NTR_max_degradation":OVERALL_NTR_MAX_DEGRADATION}}
    summary_path=out/"frozen_s0_v6_valid_screen_summary.json"; summary_path.write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    logger.info("complete verdict=%s output=%s log=%s",verdict,out,log_path)
    print("Frozen-S0 Trust-Region CFCompatKD v6 complete"); print("candidate J:",result["J_valid"]); print("frozen v4 J:",v4_grid["J_valid"])
    if gate: print("beneficial-Teacher NTR reduction vs v4:",gate["beneficial_teacher_NTR_reduction_vs_v4"])
    print("verdict:",verdict); print("official Test was not constructed or accessed"); print("summary:",summary_path)


if __name__ == "__main__": main()
