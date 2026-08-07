"""CFCompatKD v5.2: frozen Valid Student-transfer failure diagnostic.

Read-only analysis. Joins v4/v2 validation-best Student predictions with the
frozen v5 Valid gate diagnostic. It performs no training/checkpoint selection,
constructs no DataLoader, and never reads Test.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

VERSION = "cfcompat_student_transfer_failure_diagnostic_v5p2"
SEED = 1113
MODES = ("LA", "LV", "L")
V4_RUN = "regret_preserve_cfcompat"
REF_RUNS = ("cfcompat_replay", "student_safe_uniform")
BENEFIT_MARGIN = 0.02
NEG_MARGIN = 0.02
SEVERE_MARGIN = 0.10

DEFAULT_CANDIDATE = Path("result/missing_baseline/cfcompat_regret_preserve_v4/mosi/valid_screen/seed1113_dev/regret_preserve_v4_candidate_raw_valid_events.csv")
DEFAULT_REFERENCE = Path("result/missing_baseline/cfcompat_regret_preserve_v4/mosi/valid_screen/seed1113_dev/regret_preserve_v4_reference_raw_valid_events.csv")
DEFAULT_GATE = Path("result/missing_baseline/cfcompat_crossfit_transfer_risk_v5/mosi/valid_screen/seed1113_dev/gate_only/crossfit_transfer_risk_v5_valid_gate_diagnostic.csv")
DEFAULT_OUTPUT = Path("result/missing_baseline/cfcompat_student_transfer_failure_diagnostic_v5p2/mosi/seed1113_valid")

FEATURES = (
    "baseline_missing_prediction", "baseline_full_prediction", "teacher_full_prediction",
    "initial_student_missing_prediction", "teacher_minus_baseline_missing",
    "student0_minus_baseline_missing", "teacher_minus_student0",
    "baseline_full_minus_missing", "teacher_minus_baseline_full",
    "abs_teacher_minus_baseline_missing", "abs_student0_minus_baseline_missing",
    "abs_teacher_minus_student0", "abs_baseline_full_minus_missing",
    "abs_teacher_minus_baseline_full", "abs_baseline_missing_prediction",
    "abs_teacher_full_prediction", "abs_initial_student_missing_prediction",
)
RULE_COLS = (
    "mode", "baseline_difficulty_quartile", "teacher_gap_quartile",
    "initial_student_gap_quartile", "label_region",
    "teacher_baseline_sign_relation", "teacher_crosses_label_relative_baseline",
    "v5_gate_error_type",
)


def parse_args():
    p = argparse.ArgumentParser(description="CFCompatKD v5.2 Student-transfer failure diagnostic")
    p.add_argument("--v4-candidate-csv", default=str(DEFAULT_CANDIDATE))
    p.add_argument("--v4-reference-csv", default=str(DEFAULT_REFERENCE))
    p.add_argument("--v5-valid-gate-csv", default=str(DEFAULT_GATE))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    p.add_argument("--near-zero-label-threshold", type=float, default=0.5)
    p.add_argument("--min-rule-support", type=int, default=20)
    p.add_argument("--top-k", type=int, default=120)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    if a.near_zero_label_threshold < 0 or a.min_rule_support < 1 or a.top_k < 1:
        p.error("invalid diagnostic threshold/support argument")
    return a


def _finite(df, cols, name):
    if not np.isfinite(df.loc[:, cols].to_numpy(float)).all():
        raise FloatingPointError(f"{name} contains NaN/Inf")


def validate_raw(df, allowed_runs, name):
    req = {"Seed", "Run", "Mode", "sample_index", "sample_id", "label", "baseline_prediction", "candidate_prediction", "teacher_prediction", "Split"}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"{name} lacks {sorted(missing)}")
    out = df.copy()
    if set(out["Seed"].astype(int)) != {SEED} or set(out["Split"].astype(str).str.lower()) != {"valid"}:
        raise RuntimeError(f"{name} must be Seed1113 official Valid only")
    runs = set(out["Run"].astype(str))
    if not runs or not runs.issubset(set(allowed_runs)):
        raise RuntimeError(f"{name} unexpected runs: {sorted(runs)}")
    if out.duplicated(["Run", "Mode", "sample_index"]).any():
        raise RuntimeError(f"{name} has duplicate run/mode/sample rows")
    _finite(out, ["label", "baseline_prediction", "candidate_prediction", "teacher_prediction"], name)
    return out


def validate_gate(df):
    req = {"sample_index", "sample_id", "mode", "split", "label", "teacher_advantage_vs_baseline", "beneficial_label", "full_train_benefit_probability", *FEATURES}
    missing = req.difference(df.columns)
    if missing:
        raise ValueError(f"v5 gate lacks {sorted(missing)}")
    out = df.copy()
    if set(out["split"].astype(str).str.lower()) != {"valid"}:
        raise RuntimeError("v5 gate must be Valid only")
    if not set(out["mode"].astype(str)).issubset(set(MODES)) or out.duplicated(["sample_index", "mode"]).any():
        raise RuntimeError("v5 gate has invalid mode/sample grid")
    p = out["full_train_benefit_probability"].to_numpy(float)
    if not np.isfinite(p).all() or np.any(p <= 0) or np.any(p >= 1):
        raise FloatingPointError("v5 probabilities must lie strictly inside (0,1)")
    _finite(out, ["label", "teacher_advantage_vs_baseline", *FEATURES], "v5 gate")
    return out


def prepare_events(raw):
    e = raw.loc[raw["Mode"].astype(str).isin(MODES)].copy().rename(columns={"Mode": "mode"})
    e["baseline_error"] = np.abs(e.baseline_prediction - e.label)
    e["student_error"] = np.abs(e.candidate_prediction - e.label)
    e["teacher_error"] = np.abs(e.teacher_prediction - e.label)
    e["teacher_advantage"] = e.baseline_error - e.teacher_error
    e["student_gain_vs_baseline"] = e.baseline_error - e.student_error
    e["student_regret_vs_baseline"] = e.student_error - e.baseline_error
    e["teacher_beneficial"] = e.teacher_advantage >= BENEFIT_MARGIN
    e["student_improved_any"] = e.student_gain_vs_baseline > 0
    e["positive_transfer"] = e.student_gain_vs_baseline > NEG_MARGIN
    e["negative_transfer"] = e.student_regret_vs_baseline > NEG_MARGIN
    e["severe_negative_transfer"] = e.student_regret_vs_baseline > SEVERE_MARGIN
    e["transfer_outcome"] = np.select(
        [e.severe_negative_transfer, e.negative_transfer, e.positive_transfer],
        ["severe_negative", "negative", "positive"], default="neutral")
    e["teacher_student_quadrant"] = np.select(
        [e.teacher_beneficial & e.student_improved_any,
         e.teacher_beneficial & ~e.student_improved_any,
         ~e.teacher_beneficial & e.student_improved_any],
        ["teacher_helpful_student_improved", "teacher_helpful_student_not_improved",
         "teacher_not_helpful_student_improved"],
        default="teacher_not_helpful_student_not_improved")
    e["mechanism_failure_class"] = np.select(
        [e.teacher_beneficial & e.positive_transfer,
         e.teacher_beneficial & e.severe_negative_transfer,
         e.teacher_beneficial & e.negative_transfer,
         e.teacher_beneficial,
         ~e.teacher_beneficial & e.negative_transfer,
         ~e.teacher_beneficial & e.positive_transfer],
        ["successful_teacher_transfer", "beneficial_teacher_severe_student_regression",
         "beneficial_teacher_student_regression", "beneficial_teacher_not_realized",
         "nonbeneficial_teacher_student_regression", "student_improved_without_beneficial_teacher"],
        default="neutral_without_beneficial_teacher")
    return e


def join_gate(candidate, gate, tol=1e-5):
    cols = ["sample_index", "sample_id", "mode", "label", "teacher_advantage_vs_baseline",
            "beneficial_label", "full_train_benefit_probability", *FEATURES]
    g = gate.loc[:, cols].loc[:, lambda x: ~x.columns.duplicated()].copy()
    m = candidate.merge(g, on=["sample_index", "mode"], how="inner", suffixes=("", "_gate"), validate="one_to_one")
    if len(m) != len(candidate):
        raise RuntimeError("v4 candidate/v5 gate binding is incomplete")
    checks = {
        "label": np.abs(m.label - m.label_gate),
        "baseline": np.abs(m.baseline_prediction - m.baseline_missing_prediction),
        "teacher": np.abs(m.teacher_prediction - m.teacher_full_prediction),
        "teacher_advantage": np.abs(m.teacher_advantage - m.teacher_advantage_vs_baseline),
    }
    for name, delta in checks.items():
        if float(np.max(delta)) > tol:
            raise RuntimeError(f"frozen source mismatch {name}: {float(np.max(delta)):.8g}")
    if not np.array_equal(m.teacher_beneficial.astype(bool).to_numpy(), m.beneficial_label.astype(bool).to_numpy()):
        raise RuntimeError("v4 recomputed/v5 stored beneficial labels disagree")
    m["initial_student_reference_prediction"] = m.initial_student_missing_prediction
    m["initial_student_reference_error"] = np.abs(m.initial_student_reference_prediction - m.label)
    m["final_minus_initial_error"] = m.student_error - m.initial_student_reference_error
    m["v5_gate_probability"] = m.full_train_benefit_probability.astype(float)
    m["v5_gate_predicted_beneficial"] = m.v5_gate_probability >= 0.5
    truth, pred = m.teacher_beneficial.astype(bool), m.v5_gate_predicted_beneficial.astype(bool)
    m["v5_gate_error_type"] = np.select(
        [truth & pred, truth & ~pred, ~truth & pred],
        ["true_positive", "false_negative", "false_positive"], default="true_negative")
    m["v5_gate_misclassified"] = truth.ne(pred)
    m["joint_gate_student_class"] = m["v5_gate_error_type"].astype(str) + "__" + m["mechanism_failure_class"].astype(str)
    return m


def _quartile(s):
    pct = s.rank(method="average", pct=True)
    return pd.cut(pct, [0, .25, .5, .75, 1], labels=["Q1_low", "Q2", "Q3", "Q4_high"], include_lowest=True).astype(str)


def add_characteristics(df, zero_thr):
    x = df.copy()
    x["abs_teacher_baseline_gap"] = np.abs(x.teacher_prediction - x.baseline_prediction)
    x["abs_initial_student_baseline_gap"] = np.abs(x.initial_student_reference_prediction - x.baseline_prediction)
    x["label_region"] = np.select([np.abs(x.label) <= zero_thr, x.label < 0], ["near_zero", "negative"], default="positive")
    prod = np.sign(x.teacher_prediction) * np.sign(x.baseline_prediction)
    x["teacher_baseline_sign_relation"] = np.select([prod > 0, prod < 0], ["same_nonzero", "opposite_nonzero"], default="zero_involved")
    x["teacher_crosses_label_relative_baseline"] = (((x.baseline_prediction-x.label)*(x.teacher_prediction-x.label)) < 0).astype(str)
    for src, dst in [("baseline_error", "baseline_difficulty_quartile"),
                     ("abs_teacher_baseline_gap", "teacher_gap_quartile"),
                     ("abs_initial_student_baseline_gap", "initial_student_gap_quartile")]:
        x[dst] = x.groupby("mode", group_keys=False)[src].transform(_quartile)
    return x


def run_summary(events):
    rows = []
    for run, r in events.groupby("Run", sort=True):
        for mode in ("ALL",) + MODES:
            z = r if mode == "ALL" else r.loc[r["mode"].astype(str).eq(mode)]
            if z.empty:
                continue
            rows.append({"Run": run, "mode": mode, "N": len(z),
                         "teacher_beneficial_prevalence": z.teacher_beneficial.mean(),
                         "student_improved_any_rate": z.student_improved_any.mean(),
                         "positive_transfer_rate": z.positive_transfer.mean(),
                         "negative_transfer_rate": z.negative_transfer.mean(),
                         "severe_negative_transfer_rate": z.severe_negative_transfer.mean(),
                         "mean_student_gain_vs_baseline": z.student_gain_vs_baseline.mean(),
                         "beneficial_teacher_not_improved_rate": (z.teacher_beneficial & ~z.student_improved_any).mean(),
                         "beneficial_teacher_negative_transfer_rate": (z.teacher_beneficial & z.negative_transfer).mean()})
    return pd.DataFrame(rows)


def quadrant_summary(events):
    rows = []
    both = pd.concat([events.assign(_mode="ALL"), events.assign(_mode=events["mode"].astype(str))], ignore_index=True)
    for (run, mode, q), z in both.groupby(["Run", "_mode", "teacher_student_quadrant"], sort=True):
        denom = len(both.loc[(both.Run == run) & (both._mode == mode)])
        rows.append({"Run": run, "mode": mode, "quadrant": q, "N": len(z), "fraction": len(z)/denom,
                     "mean_teacher_advantage": z.teacher_advantage.mean(),
                     "mean_student_gain_vs_baseline": z.student_gain_vs_baseline.mean(),
                     "negative_transfer_rate": z.negative_transfer.mean(),
                     "severe_negative_transfer_rate": z.severe_negative_transfer.mean()})
    return pd.DataFrame(rows)


def compare_runs(candidate, refs):
    out = candidate[["sample_index", "sample_id", "mode", "label", "baseline_prediction", "teacher_prediction",
                     "candidate_prediction", "student_error", "student_gain_vs_baseline", "mechanism_failure_class"]].copy()
    out = out.rename(columns={"candidate_prediction":"v4_prediction", "student_error":"v4_error",
                              "student_gain_vs_baseline":"v4_gain_vs_baseline", "mechanism_failure_class":"v4_failure_class"})
    for run in REF_RUNS:
        z = refs.loc[refs.Run.astype(str).eq(run), ["sample_index", "mode", "baseline_prediction", "teacher_prediction",
                                                    "candidate_prediction", "student_error", "student_gain_vs_baseline", "mechanism_failure_class"]].copy()
        if len(z) != len(candidate):
            raise RuntimeError(f"reference {run} sample grid is incomplete")
        z = z.rename(columns={"baseline_prediction":f"{run}_baseline", "teacher_prediction":f"{run}_teacher",
                              "candidate_prediction":f"{run}_prediction", "student_error":f"{run}_error",
                              "student_gain_vs_baseline":f"{run}_gain_vs_baseline", "mechanism_failure_class":f"{run}_failure_class"})
        out = out.merge(z, on=["sample_index", "mode"], how="left", validate="one_to_one")
        if max((out.baseline_prediction-out[f"{run}_baseline"]).abs().max(), (out.teacher_prediction-out[f"{run}_teacher"]).abs().max()) > 1e-5:
            raise RuntimeError(f"reference {run} baseline/Teacher binding changed")
        out[f"v4_error_delta_vs_{run}"] = out.v4_error - out[f"{run}_error"]
        out[f"v4_improved_vs_{run}_by_margin"] = out[f"v4_error_delta_vs_{run}"] < -NEG_MARGIN
        out[f"v4_worsened_vs_{run}_by_margin"] = out[f"v4_error_delta_vs_{run}"] > NEG_MARGIN
    return out


def _smd(a, b):
    var = (np.var(a, ddof=1) + np.var(b, ddof=1))/2
    return 0.0 if not math.isfinite(var) or var <= 1e-18 else float((np.mean(b)-np.mean(a))/math.sqrt(var))


def feature_summary(candidate):
    success = candidate.loc[candidate.mechanism_failure_class.astype(str).eq("successful_teacher_transfer")]
    fail_names = {"beneficial_teacher_not_realized", "beneficial_teacher_student_regression", "beneficial_teacher_severe_student_regression"}
    failed = candidate.loc[candidate.mechanism_failure_class.astype(str).isin(fail_names)]
    rows = []
    cols = list(dict.fromkeys([*FEATURES, "baseline_error", "teacher_advantage", "abs_teacher_baseline_gap",
                               "initial_student_reference_error", "abs_initial_student_baseline_gap"]))
    for mode in ("ALL",) + MODES:
        s = success if mode == "ALL" else success.loc[success["mode"].astype(str).eq(mode)]
        f = failed if mode == "ALL" else failed.loc[failed["mode"].astype(str).eq(mode)]
        if len(s) < 5 or len(f) < 5:
            continue
        for col in cols:
            if col not in candidate:
                continue
            a, b = s[col].to_numpy(float), f[col].to_numpy(float)
            ks = ks_2samp(a, b)
            rows.append({"mode":mode, "feature":col, "success_N":len(a), "failure_N":len(b),
                         "success_mean":np.mean(a), "failure_mean":np.mean(b),
                         "failure_minus_success_smd":_smd(a,b), "ks_statistic":ks.statistic, "ks_pvalue":ks.pvalue})
    columns = ["mode","feature","success_N","failure_N","success_mean","failure_mean","failure_minus_success_smd","ks_statistic","ks_pvalue","priority_score"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns)
    df["priority_score"] = df.failure_minus_success_smd.abs() + df.ks_statistic
    return df.sort_values("priority_score", ascending=False, kind="mergesort")[columns]


def failure_rules(candidate, min_support):
    fail_names = {"beneficial_teacher_not_realized", "beneficial_teacher_student_regression", "beneficial_teacher_severe_student_regression"}
    target = candidate.mechanism_failure_class.astype(str).isin(fail_names)
    helpful = candidate.teacher_beneficial.astype(bool)
    base_rate = float(target[helpful].mean())
    rows = []
    specs = []
    for col in RULE_COLS:
        specs.extend([(f"{col}={v}", candidate[col].astype(str).eq(v), 1) for v in sorted(candidate[col].astype(str).unique())])
    for left, right in combinations(RULE_COLS, 2):
        for lv in sorted(candidate[left].astype(str).unique()):
            for rv in sorted(candidate[right].astype(str).unique()):
                specs.append((f"{left}={lv} & {right}={rv}", candidate[left].astype(str).eq(lv) & candidate[right].astype(str).eq(rv), 2))
    for rule, mask, order in specs:
        m = mask & helpful
        n = int(m.sum())
        if n < min_support:
            continue
        rate = float(target[m].mean())
        rows.append({"rule":rule, "rule_order":order, "support":n, "support_fraction_of_teacher_helpful":n/int(helpful.sum()),
                     "beneficial_teacher_failure_rate":rate, "lift_vs_teacher_helpful":rate/base_rate if base_rate>0 else np.nan,
                     "mean_teacher_advantage":candidate.loc[m,"teacher_advantage"].mean(),
                     "mean_student_gain_vs_baseline":candidate.loc[m,"student_gain_vs_baseline"].mean(),
                     "negative_transfer_rate":candidate.loc[m,"negative_transfer"].mean()})
    columns = ["rule","rule_order","support","support_fraction_of_teacher_helpful","beneficial_teacher_failure_rate","lift_vs_teacher_helpful","mean_teacher_advantage","mean_student_gain_vs_baseline","negative_transfer_rate"]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=columns)
    return df.sort_values(["beneficial_teacher_failure_rate","lift_vs_teacher_helpful","support"], ascending=[False,False,False], kind="mergesort")[columns]


def joint_summary(candidate):
    rows=[]
    for (g,f), z in candidate.groupby(["v5_gate_error_type","mechanism_failure_class"], sort=True):
        rows.append({"v5_gate_error_type":g, "mechanism_failure_class":f, "N":len(z), "fraction_of_all":len(z)/len(candidate),
                     "mean_gate_probability":z.v5_gate_probability.mean(), "mean_teacher_advantage":z.teacher_advantage.mean(),
                     "mean_student_gain_vs_baseline":z.student_gain_vs_baseline.mean(), "negative_transfer_rate":z.negative_transfer.mean(),
                     "severe_negative_transfer_rate":z.severe_negative_transfer.mean()})
    return pd.DataFrame(rows)


def top_failures(candidate, k):
    x=candidate.copy()
    sev=np.select([x.mechanism_failure_class.eq("beneficial_teacher_severe_student_regression"),
                   x.mechanism_failure_class.eq("beneficial_teacher_student_regression"),
                   x.mechanism_failure_class.eq("beneficial_teacher_not_realized"),
                   x.mechanism_failure_class.eq("nonbeneficial_teacher_student_regression")], [4,3,2,1], default=0)
    x["failure_priority"] = sev + np.maximum(x.student_regret_vs_baseline,0) + np.maximum(x.teacher_advantage,0) + .25*x.v5_gate_misclassified.astype(float)
    keep=["sample_index","sample_id","mode","label","baseline_prediction","teacher_prediction","initial_student_reference_prediction","candidate_prediction",
          "baseline_error","teacher_error","initial_student_reference_error","student_error","teacher_advantage","student_gain_vs_baseline","student_regret_vs_baseline",
          "transfer_outcome","teacher_student_quadrant","mechanism_failure_class","v5_gate_probability","v5_gate_error_type",
          "baseline_difficulty_quartile","teacher_gap_quartile","initial_student_gap_quartile","label_region","failure_priority"]
    return x.sort_values("failure_priority", ascending=False, kind="mergesort").head(k)[keep]


def main():
    a=parse_args()
    paths=[Path(a.v4_candidate_csv),Path(a.v4_reference_csv),Path(a.v5_valid_gate_csv)]
    for p in paths:
        if not p.is_file(): raise FileNotFoundError(p)
    out=Path(a.output_dir)
    if out.exists():
        if not a.overwrite: raise FileExistsError(f"output exists; use --overwrite: {out}")
        shutil.rmtree(out)
    out.mkdir(parents=True)

    cand=prepare_events(validate_raw(pd.read_csv(paths[0]),[V4_RUN],"v4 candidate"))
    refs=prepare_events(validate_raw(pd.read_csv(paths[1]),REF_RUNS,"v4 references"))
    gate=validate_gate(pd.read_csv(paths[2]))
    if len(cand)!=229*3 or cand.sample_index.nunique()!=229 or len(gate)!=229*3 or gate.sample_index.nunique()!=229:
        raise RuntimeError("candidate/gate must each contain exactly 229 x 3 missing Valid events")
    for run in REF_RUNS:
        z=refs.loc[refs.Run.astype(str).eq(run)]
        if len(z)!=229*3 or z.sample_index.nunique()!=229: raise RuntimeError(f"reference {run} is incomplete")

    cand=add_characteristics(join_gate(cand,gate),a.near_zero_label_threshold)
    all_events=pd.concat([cand,refs],ignore_index=True,sort=False)
    runs=run_summary(all_events)
    quadrants=quadrant_summary(all_events)
    compare=compare_runs(cand,refs)
    features=feature_summary(cand)
    rules=failure_rules(cand,a.min_rule_support)
    joint=joint_summary(cand)
    top=top_failures(cand,a.top_k)

    helpful=cand.teacher_beneficial.astype(bool)
    not_improved=helpful & ~cand.student_improved_any.astype(bool)
    neg=helpful & cand.negative_transfer.astype(bool)
    severe=helpful & cand.severe_negative_transfer.astype(bool)
    gate_tp_fail=helpful & cand.v5_gate_predicted_beneficial.astype(bool) & ~cand.student_improved_any.astype(bool)
    summary={
        "version":VERSION,"scope":"Seed1113 official Valid only","student_training_performed":False,
        "checkpoint_selection_performed":False,"test_accessed":False,"test_constructed":False,
        "teacher_benefit_margin":BENEFIT_MARGIN,"negative_transfer_margin":NEG_MARGIN,"severe_negative_transfer_margin":SEVERE_MARGIN,
        "counts":{"candidate_missing_events":len(cand),"candidate_samples":cand.sample_index.nunique(),"teacher_beneficial_events":int(helpful.sum()),
                  "beneficial_teacher_not_improved_events":int(not_improved.sum()),"beneficial_teacher_negative_transfer_events":int(neg.sum()),
                  "beneficial_teacher_severe_negative_events":int(severe.sum()),"v5_gate_true_positive_but_student_not_improved_events":int(gate_tp_fail.sum())},
        "rates":{"teacher_beneficial_prevalence":float(helpful.mean()),"failure_rate_given_teacher_beneficial":float(not_improved.sum()/max(int(helpful.sum()),1)),
                 "negative_transfer_rate_given_teacher_beneficial":float(neg.sum()/max(int(helpful.sum()),1)),
                 "severe_negative_rate_given_teacher_beneficial":float(severe.sum()/max(int(helpful.sum()),1))},
        "historical_artifact_limits":{"valid_epoch_trajectory_available":False,"valid_per_sample_train_route_available":False,
            "initial_student_reference_source":"frozen v5 gate prepass; common Seed1113 initial reference, not a historical v4 per-epoch artifact",
            "interpretation":"Valid route/KD-active fields are intentionally not fabricated; v4 routing occurred on Train events only."},
        "interpretation_guardrails":[
            "Teacher beneficial uses frozen Valid labels for diagnosis only.",
            "Beneficial-Teacher Student regression suggests gating alone may be insufficient but does not prove gradient interference causally.",
            "v5 gate probabilities are joined diagnostically; v5 did not train the v4 Student.",
            "No Test sample-level claim is permitted."],
        "reference_comparison_counts":{k:int(compare[k].sum()) for k in compare.columns if k.startswith("v4_") and ("improved_vs_" in k or "worsened_vs_" in k)},
        "top_beneficial_teacher_failure_features":features.head(15).to_dict("records"),
        "top_beneficial_teacher_failure_rules":rules.head(15).to_dict("records"),
        "inputs":{"v4_candidate_csv":str(paths[0]),"v4_reference_csv":str(paths[1]),"v5_valid_gate_csv":str(paths[2])},
    }
    outputs={"student_transfer_failure_table.csv":cand,"reference_run_transfer_table.csv":refs,"student_transfer_summary.csv":runs,
             "teacher_student_quadrant_summary.csv":quadrants,"v4_vs_reference_sample_deltas.csv":compare,
             "beneficial_teacher_failure_feature_summary.csv":features,"beneficial_teacher_failure_rules.csv":rules,
             "joint_gate_student_failure_summary.csv":joint,"top_student_failure_samples.csv":top}
    for name,df in outputs.items(): df.to_csv(out/name,index=False)
    (out/"student_transfer_failure_summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    print("================ CFCompatKD v5.2 Student-transfer diagnostic ================")
    print("candidate missing events:",len(cand)); print("teacher beneficial:",int(helpful.sum()))
    print("beneficial Teacher but Student not improved:",int(not_improved.sum()))
    print("beneficial Teacher + negative transfer:",int(neg.sum())); print("beneficial Teacher + severe negative:",int(severe.sum()))
    print("Student training: False"); print("Test accessed: False"); print("output:",out)


if __name__ == "__main__":
    main()
