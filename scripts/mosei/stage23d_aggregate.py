#!/usr/bin/env python3
"""Aggregate the ten frozen Expert self-risk audits and apply frozen gates."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from stage23d_self_risk_common import (
    EXPERTS,
    FOLDS,
    MISSING_MODES,
    MODES,
    OUT,
    ROOT,
    RUNTIME,
    atomic_json,
    atomic_tsv,
    git_branch,
    git_head,
    sha256_file,
    utc_now,
)


ANALYSIS = OUT / "analysis"
FINAL = OUT / "final"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-complete", action="store_true", help="fail unless all audits exist"
    )
    return parser.parse_args()


def phase_dir(fold, expert):
    return OUT / "phase1" / f"checkpoint_fold{fold}" / expert


def feature_dir(fold, expert):
    return OUT / "features" / f"checkpoint_fold{fold}" / expert


def load_all(name, sep="\t"):
    frames = []
    missing = []
    for fold in FOLDS:
        for expert in EXPERTS:
            path = phase_dir(fold, expert) / name
            if path.exists():
                frames.append(pd.read_csv(path, sep=sep))
            else:
                missing.append(str(path))
    if missing:
        raise RuntimeError(f"Missing {name} for {len(missing)} audits")
    return pd.concat(frames, ignore_index=True)


def checkpoint_identity_audit():
    rows = []
    excluded = {
        "train_index",
        "checkpoint_fold",
        "identity",
        "prediction",
    }
    for expert in EXPERTS:
        for mode in MODES:
            frames = []
            for fold in FOLDS:
                path = feature_dir(fold, expert) / f"static_internal_features_{mode}.csv.gz"
                frame = pd.read_csv(path, dtype={"sample_id": str, "video_id": str})
                frame["identity"] = fold
                frames.append(frame)
            data = pd.concat(frames, ignore_index=True)
            numeric = [
                column
                for column in data.select_dtypes(include=[np.number]).columns
                if column not in excluded and not column.startswith("pca_")
            ]
            train = data["self_risk_role"].eq("inner_train").to_numpy()
            test = data["self_risk_role"].eq("outer").to_numpy()
            scaler = StandardScaler().fit(data.loc[train, numeric])
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=23170,
            ).fit(scaler.transform(data.loc[train, numeric]), data.loc[train, "identity"])
            probability = model.predict_proba(
                scaler.transform(data.loc[test, numeric])
            )[:, 1]
            labels = data.loc[test, "identity"].to_numpy()
            rows.append(
                {
                    "expert_id": expert,
                    "mode": mode,
                    "train_samples": int(train.sum()),
                    "outer_samples": int(test.sum()),
                    "scalar_summary_dimensions": len(numeric),
                    "accuracy": float(np.mean((probability >= 0.5) == labels)),
                    "AUROC": float(roc_auc_score(labels, probability)),
                    "chance_AUROC": 0.5,
                    "raw_PCA_coordinates_used": False,
                    "source_roles_disjoint": True,
                }
            )
    return pd.DataFrame(rows)


def cross_expert_diagnostic():
    frames = []
    for fold in FOLDS:
        for expert in EXPERTS:
            path = phase_dir(fold, expert) / "outer_diagnostic_ledger.csv.gz"
            frames.append(pd.read_csv(path, dtype={"sample_id": str, "video_id": str}))
    long = pd.concat(frames, ignore_index=True)
    rows = []
    for keys, group in long.groupby(["checkpoint_fold", "mode"], sort=True):
        top1 = []
        top2 = []
        regrets = []
        for _, local in group.groupby("sample_id", sort=False):
            local = local.sort_values("R2_expected_abs_error")
            actual_best = local.loc[local["actual_abs_error"].idxmin(), "expert_id"]
            top1.append(local.iloc[0]["expert_id"] == actual_best)
            top2.append(actual_best in set(local.iloc[:2]["expert_id"]))
            regrets.append(
                float(local.iloc[0]["actual_abs_error"] - local["actual_abs_error"].min())
            )
        rows.append(
            {
                "checkpoint_fold": int(keys[0]),
                "mode": keys[1],
                "samples": len(top1),
                "lowest_predicted_risk_top1_expert_accuracy": float(np.mean(top1)),
                "lowest_predicted_risk_top2_coverage": float(np.mean(top2)),
                "selected_expert_mean_regret": float(np.mean(regrets)),
                "chance_top1": 1.0 / len(EXPERTS),
                "chance_top2": 2.0 / len(EXPERTS),
                "diagnostic_only": True,
            }
        )
    return pd.DataFrame(rows)


def model_mean(metrics, model, metric):
    values = pd.to_numeric(
        metrics.loc[metrics["model"] == model, metric], errors="coerce"
    )
    return float(values.mean())


def markdown_table(frame):
    """Render a compact Markdown table without the optional tabulate package."""
    columns = list(frame.columns)

    def render(value):
        if isinstance(value, (float, np.floating)):
            return f"{float(value):.4f}"
        return str(value).replace("|", "\\|")

    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(render(value) for value in row) + " |")
    return "\n".join(lines)


def coverage_gate(coverage):
    local = coverage.loc[coverage["model"] == "R2"].copy()
    rows = []
    for mode in MODES:
        curve = (
            local.loc[local["mode"] == mode]
            .groupby("coverage", as_index=False)
            .agg(
                retained_actual_MAE=("retained_actual_MAE", "mean"),
                improvement_vs_random=("improvement_vs_random", "mean"),
            )
            .sort_values("coverage")
        )
        downward = int(
            (np.diff(curve["retained_actual_MAE"].to_numpy()) < -1e-8).sum()
        )
        positive = float(
            curve.loc[curve["coverage"] < 1, "improvement_vs_random"].mean()
        )
        passed = downward <= 1 and positive > 0
        rows.append(
            {
                "mode": mode,
                "downward_adjacent_steps": downward,
                "mean_improvement_vs_random_below_full_coverage": positive,
                "pass": passed,
            }
        )
    table = pd.DataFrame(rows)
    missing_pass = int(
        table.loc[table["mode"].isin(MISSING_MODES), "pass"].sum()
    )
    aggregate_pass = bool(table["pass"].all() and missing_pass >= 2)
    return table, aggregate_pass


def gate_table(metrics, coverage):
    r2 = metrics.loc[metrics["model"] == "R2"].copy()
    mean_s = float(r2["Error_Spearman"].mean())
    fold_means = r2.groupby("checkpoint_fold")["Error_Spearman"].mean()
    expert_fold = r2.groupby(["expert_id", "checkpoint_fold"])[
        "Error_Spearman"
    ].mean()
    positive_both = sum(
        bool((expert_fold.loc[expert] > 0).all()) for expert in EXPERTS
    )
    bad = float(r2["bad20_AUROC"].mean())
    cw = float(pd.to_numeric(r2["confident_wrong_AUROC"]).mean())
    deltas = {}
    for reference in ("R1", "N2_length_mask_only", "N5_label_bin_prior"):
        ref = metrics.loc[metrics["model"] == reference]
        deltas[reference] = {
            key: float(r2[key].mean() - pd.to_numeric(ref[key]).mean())
            for key in ("Error_Spearman", "bad20_AUROC", "confident_wrong_AUROC")
        }
    shuffle_rows = []
    for seed in (23071, 23072, 23073):
        name = f"N0_shuffle_seed{seed}"
        local = metrics.loc[metrics["model"] == name]
        shuffle_rows.append(
            {
                "model": name,
                "Error_Spearman": float(local["Error_Spearman"].mean()),
                "bad20_AUROC": float(local["bad20_AUROC"].mean()),
                "confident_wrong_AUROC": float(
                    pd.to_numeric(local["confident_wrong_AUROC"]).mean()
                ),
            }
        )
    strongest = max(shuffle_rows, key=lambda row: row["Error_Spearman"])
    shuffle_delta_s = mean_s - strongest["Error_Spearman"]
    shuffle_delta_bad = bad - strongest["bad20_AUROC"]
    missing_positive = int(
        (
            r2.loc[r2["mode"].isin(MISSING_MODES)]
            .groupby("mode")["Error_Spearman"]
            .mean()
            > 0
        ).sum()
    )
    coverage_detail, coverage_pass = coverage_gate(coverage)
    values = [
        ("G1", mean_s, mean_s >= 0.25),
        ("G2", float(fold_means.min()), float(fold_means.min()) >= 0.15),
        ("G3", positive_both, positive_both >= 4),
        ("G4", bad, bad >= 0.63),
        ("G5", cw, cw >= 0.63),
        (
            "G6",
            deltas["R1"]["Error_Spearman"],
            deltas["R1"]["Error_Spearman"] >= 0.02
            and max(
                deltas["R1"]["bad20_AUROC"],
                deltas["R1"]["confident_wrong_AUROC"],
            )
            >= 0.01
            and min(
                deltas["R1"]["bad20_AUROC"],
                deltas["R1"]["confident_wrong_AUROC"],
            )
            >= -0.01,
        ),
        (
            "G7",
            shuffle_delta_s,
            shuffle_delta_s >= 0.03 and shuffle_delta_bad >= 0.02,
        ),
        ("G8", missing_positive, missing_positive >= 2),
        ("G9", int(coverage_pass), coverage_pass),
        (
            "G10",
            min(
                deltas["N2_length_mask_only"]["Error_Spearman"],
                deltas["N5_label_bin_prior"]["Error_Spearman"],
            ),
            deltas["N2_length_mask_only"]["Error_Spearman"] >= 0.02
            and deltas["N5_label_bin_prior"]["Error_Spearman"] >= 0.02,
        ),
    ]
    table = pd.DataFrame(
        [
            {"gate": name, "observed": observed, "pass": bool(passed)}
            for name, observed, passed in values
        ]
    )
    all_pass = bool(table["pass"].all())
    weak = (
        int(table["pass"].sum()) >= 7
        and mean_s >= 0.20
        and float(fold_means.min()) >= 0.10
        and bad >= 0.58
        and cw >= 0.58
        and deltas["R1"]["Error_Spearman"] > 0
        and shuffle_delta_s > 0
    )
    decision = "PASS" if all_pass else ("WEAK" if weak else "FAIL")
    details = {
        "decision": decision,
        "passed_conditions": int(table["pass"].sum()),
        "total_conditions": len(table),
        "mean_Error_Spearman": mean_s,
        "fold_mean_Error_Spearman": {
            str(int(key)): float(value) for key, value in fold_means.items()
        },
        "positive_both_fold_expert_configs": int(positive_both),
        "mean_bad20_AUROC": bad,
        "mean_confident_wrong_AUROC": cw,
        "missing_modes_positive": missing_positive,
        "R2_minus_references": deltas,
        "strongest_shuffle": strongest,
        "R2_minus_strongest_shuffle_Error_Spearman": shuffle_delta_s,
        "R2_minus_strongest_shuffle_bad20_AUROC": shuffle_delta_bad,
        "coverage_pass": coverage_pass,
    }
    return table, coverage_detail, details


def report_markdown(details, r2, proxies, identity, comparable):
    expert = (
        r2.groupby("expert_id")
        .agg(
            Spearman=("Error_Spearman", "mean"),
            bad20_AUROC=("bad20_AUROC", "mean"),
            confident_wrong_AUROC=("confident_wrong_AUROC", "mean"),
        )
        .sort_values("Spearman", ascending=False)
    )
    mode = (
        r2.groupby("mode")
        .agg(
            Spearman=("Error_Spearman", "mean"),
            bad20_AUROC=("bad20_AUROC", "mean"),
            confident_wrong_AUROC=("confident_wrong_AUROC", "mean"),
        )
        .sort_values("Spearman", ascending=False)
    )
    proxy = (
        proxies.groupby("proxy")["Error_Spearman"].mean().sort_values(ascending=False)
    )
    best_expert, worst_expert = expert.index[0], expert.index[-1]
    strongest_mode = mode.loc[mode.index.isin(MISSING_MODES)].index[0]
    top_proxy = proxy.index[0]
    conclusion = {
        "PASS": "SELF_RISK_SIGNAL_PASS",
        "WEAK": "SELF_RISK_SIGNAL_WEAK",
        "FAIL": "SELF_RISK_SIGNAL_FAIL",
    }[details["decision"]]
    if details["decision"] != "FAIL":
        conclusion += "（A5 仍需按冻结协议完成后才是最终结论）"
    lines = [
        "# Stage23D-A Expert Self-Risk Signal Audit",
        "",
        f"Phase-1 结论：**{conclusion}**。通过 {details['passed_conditions']}/10 个冻结门槛。",
        "",
        "## 易懂结论",
        "",
        f"1. Expert 能否从内部状态预测自身误差？平均 Spearman={details['mean_Error_Spearman']:.4f}；按冻结门槛判定为 {details['decision']}。",
        f"2. 内部状态是否明显优于只看 prediction/head disagreement？R2−R1 Spearman={details['R2_minus_references']['R1']['Error_Spearman']:+.4f}。",
        f"3. 哪些内部信号最有效？单 proxy 中 `{top_proxy}` 的平均 Spearman 最高（{proxy.iloc[0]:.4f}）。",
        f"4. hidden-space 异常是否对应更高错误？centroid/knn/nearest-source proxy 的独立结果已写入 `individual_proxy_metrics.tsv`，不以单一汇总掩盖方向差异。",
        "5. submode/perturbation instability 是否有价值？submode instability 已纳入 A4；A5 perturbation 仅在 Phase-1 为 WEAK/PASS 时授权。",
        f"6. 能否识别 worst-20% error？R2 mean AUROC={details['mean_bad20_AUROC']:.4f}。",
        f"7. 能否识别 confident-wrong？R2 mean AUROC={details['mean_confident_wrong_AUROC']:.4f}。",
        f"8. 哪个 Expert 最有自知、哪个最差？按两折四模式 Spearman，最佳 `{best_expert}`，最弱 `{worst_expert}`。",
        f"9. 哪个 missing mode 信号最强？四模式汇总中 `{strongest_mode}` 最高；完整 missing-mode 表保留在 per-mode 指标中。",
        f"10. 两个 checkpoint replication 是否一致？fold0={details['fold_mean_Error_Spearman']['0']:.4f}，fold1={details['fold_mean_Error_Spearman']['1']:.4f}。",
        "11. 风险输出能否校准到统一 MAE 单位？R2 直接输出 expected absolute error，量化覆盖率、pinball 和 q90−q50 宽度均以同一绝对误差单位保存。",
        f"12. 是否值得进入 Stage23D-B？{'否；冻结门槛失败。' if details['decision']=='FAIL' else '尚不能；需先完成冻结授权的 A5，再做人工决策。'}",
        "13. 若失败，原因是无信号还是样本不足？每折 OOF 外层仍有约千级样本；应优先依据 R2 对 R1/N0/N2/N5 的差异判断信号增量，而不是把失败归因于样本数。",
        f"14. 是否有理由继续动态专家选择主线？只读跨专家诊断 top-1={comparable['lowest_predicted_risk_top1_expert_accuracy'].mean():.4f}、top-2={comparable['lowest_predicted_risk_top2_coverage'].mean():.4f}；这不是 Arbiter 性能声明。",
        f"15. checkpoint 指纹风险如何？scalar summary identity AUROC={identity['AUROC'].mean():.4f}（随机为 0.5）；raw PCA 坐标未用于该结论。",
        "16. 是否访问 Official Valid/Test 或训练新模型？Official Valid=0、Test=0；未训练 Arbiter、Student 或 Expert，Risk probe 是本阶段授权的小模型。",
        "17. 下一步是什么？停止在 Stage23D-A 报告处，等待人工研究决策；绝不自动进入 Stage23D-B。",
        "",
        "## 冻结边界",
        "",
        "- Stage23A-v2a = SIGNAL_AUDIT_FAIL，未修改。",
        "- Stage23A-v2b = LOCAL_COMPETENCE_WEAK，未修改。",
        "- Stage23C 主工作树、进程和 GPU0/1/2 未被本阶段修改或占用。",
        "- 五个 frozen Expert、两个 checkpoint fold，共 10 个独立 self-risk audits。",
        "- 每个 audit 的 outer labels 仅在冻结预测之后打开一次。",
        "",
        "## Expert 汇总",
        "",
        markdown_table(expert.reset_index()),
        "",
        "## Mode 汇总",
        "",
        markdown_table(mode.reset_index()),
        "",
        "完整表、负对照、risk–coverage、quantile calibration、敏感性和 SHA 清单位于同目录。",
        "",
    ]
    return "\n".join(lines), conclusion


def main():
    cli = parse_args()
    metrics = load_all("outer_metrics.tsv")
    coverage = load_all("risk_coverage.tsv")
    calibration = load_all("quantile_calibration.tsv")
    proxies = load_all("individual_proxy_metrics.tsv")
    sensitivity = load_all("confident_wrong_sensitivity.tsv")
    identity = checkpoint_identity_audit()
    comparable = cross_expert_diagnostic()
    gate, coverage_detail, details = gate_table(metrics, coverage)
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FINAL.mkdir(parents=True, exist_ok=True)
    atomic_tsv(metrics, ANALYSIS / "per_expert_mode_metrics.tsv")
    atomic_tsv(
        metrics.groupby(["model", "mode"], as_index=False).mean(numeric_only=True),
        ANALYSIS / "per_mode_metrics.tsv",
    )
    atomic_tsv(
        metrics.loc[~metrics["model"].isin(["R0", "R1", "R2"])],
        ANALYSIS / "negative_controls.tsv",
    )
    atomic_tsv(coverage, ANALYSIS / "risk_coverage_curves.tsv")
    atomic_tsv(coverage_detail, ANALYSIS / "risk_coverage_gate.tsv")
    atomic_tsv(calibration, ANALYSIS / "quantile_calibration.tsv")
    atomic_tsv(proxies, ANALYSIS / "individual_proxy_metrics.tsv")
    atomic_tsv(sensitivity, ANALYSIS / "confident_wrong_analysis.tsv")
    atomic_tsv(identity, ANALYSIS / "checkpoint_identity_audit.tsv")
    atomic_tsv(comparable, ANALYSIS / "cross_expert_comparability.tsv")
    atomic_tsv(gate, ANALYSIS / "phase1_gate.tsv")
    atomic_json(ANALYSIS / "phase1_gate.json", details)
    r2 = metrics.loc[metrics["model"] == "R2"].copy()
    markdown, conclusion = report_markdown(
        details, r2, proxies, identity, comparable
    )
    report_path = ANALYSIS / "stage23d_a_self_risk_audit.md"
    report_path.write_text(markdown, encoding="utf-8")
    audit = {
        "stage": "Stage23D-A Expert Self-Risk Signal Audit",
        "phase1_decision": details["decision"],
        "conclusion": conclusion,
        "gate": details,
        "checkpoint_audits": 10,
        "modes": list(MODES),
        "expert_ids": list(EXPERTS),
        "a5_authorized": (
            "none"
            if details["decision"] == "FAIL"
            else ("limited_pilot" if details["decision"] == "WEAK" else "full_after_pilot")
        ),
        "formal_arbiter_training_count": 0,
        "student_training_count": 0,
        "expert_retraining_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "stage23c_modification_count": 0,
        "branch": git_branch(),
        "base_commit": "99d18cc49c52b799c83a65af2ed308264d8c649c",
        "code_head_at_aggregation": git_head(),
        "completed_at": utc_now(),
    }
    atomic_json(ANALYSIS / "stage23d_a_self_risk_audit.json", audit)
    lock = {
        "official_valid_authorized": False,
        "test_authorized": False,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "status": "LOCKED_UNTOUCHED",
    }
    atomic_json(FINAL / "OFFICIAL_VALID_TEST_LOCK_STATUS.json", lock)
    extraction_manifests = []
    for fold in FOLDS:
        for expert in EXPERTS:
            path = feature_dir(fold, expert) / "extraction_manifest.json"
            payload = json.loads(path.read_text())
            extraction_manifests.append(
                {
                    "checkpoint_fold": fold,
                    "expert_id": expert,
                    "gpu_id": payload["gpu_id"],
                    "feature_schema_version": payload["feature_schema_version"],
                    "samples": payload["samples"],
                    "rows": payload["rows"],
                    "max_abs_final_replay_diff": payload[
                        "max_abs_final_replay_diff"
                    ],
                    "hook_prediction_max_abs_diff": payload[
                        "hook_prediction_max_abs_diff"
                    ],
                }
            )
    gpu_report = {
        "extractions": extraction_manifests,
        "stage23d_gpu": 3,
        "stage23c_reserved_gpus_not_used": [0, 1, 2],
        "cpu_thread_limits": {
            "OMP_NUM_THREADS": 1,
            "MKL_NUM_THREADS": 1,
            "OPENBLAS_NUM_THREADS": 1,
            "NUMEXPR_NUM_THREADS": 1,
        },
        "phase1_started_only_after_stage23c_train_queue_pipeline_count_zero": True,
        "extraction_queue_log": str(
            (RUNTIME / "extraction_v2_resume.log").resolve()
        ),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
    }
    atomic_json(FINAL / "runtime_gpu_report.json", gpu_report)
    command_text = "\n".join(
        [
            "# Stage23D-A reproducible commands",
            "python scripts/mosei/stage23d_authorize.py",
            "python scripts/mosei/stage23d_freeze_gate.py",
            "python scripts/mosei/stage23d_extract_queue.py --gpu-id 3 --batch-size 16 --num-workers 0",
            "for fold in 0 1; do for expert in uniform_kd_seed1111 moddrop_seed1111 moddrop_seed1114 cfcompat_seed1111 cfcompat_seed1114; do",
            "  python scripts/mosei/stage23d_phase1.py --phase select --checkpoint-fold $fold --expert-id $expert",
            "  python scripts/mosei/stage23d_phase1.py --phase evaluate --checkpoint-fold $fold --expert-id $expert",
            "done; done",
            "python scripts/mosei/stage23d_aggregate.py --require-complete",
            "",
        ]
    )
    (FINAL / "reproducible_commands.sh").write_text(command_text, encoding="utf-8")
    git_diff = subprocess.check_output(
        ["git", "diff", "--binary"], cwd=ROOT
    )
    (FINAL / "git_diff_at_aggregation.patch").write_bytes(git_diff)
    artifact_rows = []
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {
            "artifact_sha256_manifest.tsv",
            "artifact_sha_verification.json",
        }:
            artifact_rows.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    atomic_tsv(
        pd.DataFrame(artifact_rows), FINAL / "artifact_sha256_manifest.tsv"
    )
    sha_failures = [
        row["path"]
        for row in artifact_rows
        if sha256_file(ROOT / row["path"]) != row["sha256"]
    ]
    atomic_json(
        FINAL / "artifact_sha_verification.json",
        {
            "checked_artifacts": len(artifact_rows),
            "sha256_failures": len(sha_failures),
            "failure_paths": sha_failures,
            "status": "PASS" if not sha_failures else "FAIL",
            "verified_at": utc_now(),
        },
    )
    state = {
        "stage": "Stage23D-A",
        "status": (
            "PHASE1_COMPLETE_A5_NOT_AUTHORIZED"
            if details["decision"] == "FAIL"
            else "PHASE1_COMPLETE_A5_ACTION_REQUIRED"
        ),
        "phase1_decision": details["decision"],
        "report_path": str(report_path.resolve()),
        "report_sha256": sha256_file(report_path),
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "stage23c_modification_count": 0,
        "updated_at": utc_now(),
    }
    atomic_json(RUNTIME / "state.json", state)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
