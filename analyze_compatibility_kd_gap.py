"""Validation-only Compatibility--Distillation Discrepancy analysis.

Question:
Does the train-calibrated compatibility score reflect an *independent*
teacher--student distillation mismatch?

Compatibility is estimated from the frozen ModDrop evaluator:
    delta_eval = |evaluator_LAV - evaluator_m|
    C = 1 - F_train_mid(delta_eval)

The validation target diagnostic is deliberately different:
    gap_abs = |clean_teacher_LAV - FixedKD_student_m|

The clean teacher and FixedKD student are not used to construct C.  We also
report the SmoothL1 discrepancy used by the KD objective.

No test split is constructed or read.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data_loader import MMDataLoader
from train_cf_compat_kd import build_config, prediction_rows
from trains.singleTask.fixed_kd_utils import build_frozen_teacher, teacher_lav_prediction
from trains.singleTask.missing_utils import MissingModalityWrapper
from trains.singleTask.model.DLF import DLF


MODES = ("LA", "LV", "L")
QUARTILE_LABELS = ("Q1_low", "Q2", "Q3", "Q4_high")


def parse_args():
    p = argparse.ArgumentParser(
        description="Validation-only compatibility vs independent distillation discrepancy."
    )
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--result-root", default="result")
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--evaluator-valid-pred")
    p.add_argument("--fixedkd-pred")
    p.add_argument("--fixedkd-checkpoint")
    p.add_argument("--teacher-pred")
    p.add_argument("--teacher-checkpoint")
    p.add_argument("--train-cache")
    p.add_argument("--output-dir")
    p.add_argument("--bootstrap-reps", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=20261001)
    p.add_argument("--no-plot", action="store_true")
    return p.parse_args()


def default_paths(args):
    root = Path(args.result_root)
    evaluator = Path(args.evaluator_valid_pred) if args.evaluator_valid_pred else (
        root / "missing_baseline" / "moddrop_benchmark_multiseed_v1"
        / f"seed{args.seed}" / f"{args.dataset}_best_valid_predictions.csv"
    )
    fixed_pred = Path(args.fixedkd_pred) if args.fixedkd_pred else None
    fixed_ckpt = Path(args.fixedkd_checkpoint) if args.fixedkd_checkpoint else (
        Path(args.model_save_dir) / "missing_baseline" / "fixed_kd"
        / f"DLF_{args.dataset}_seed{args.seed}_best.pth"
    )
    teacher_pred = Path(args.teacher_pred) if args.teacher_pred else None
    teacher_ckpt = Path(args.teacher_checkpoint) if args.teacher_checkpoint else (
        Path(args.model_save_dir) / f"DLF_{args.dataset}_seed{args.seed}_best.pth"
    )
    cache = Path(args.train_cache) if args.train_cache else (
        root / "counterfactual_compatibility" / "cf_compat_v1_multiseed"
        / args.dataset / f"seed{args.seed}" / "train_counterfactual_compatibility.csv"
    )
    output = Path(args.output_dir) if args.output_dir else (
        root / "analysis" / "compatibility_kd_gap_v1" / args.dataset / f"seed{args.seed}"
    )
    return evaluator, fixed_pred, fixed_ckpt, teacher_pred, teacher_ckpt, cache, output


def forbid_test_path(path):
    if path is None:
        return
    text = str(path).replace("\\", "/").lower()
    if "test" in Path(path).name.lower() or "/test/" in text:
        raise ValueError(f"Validation-only analysis refuses test input: {path}")


def require_columns(frame, columns, source):
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{source} missing required columns: {missing}")


def build_cli(args):
    return SimpleNamespace(
        dataset=args.dataset,
        config_file=args.config_file,
        gpu_ids=args.gpu_ids,
    )


def generate_fixedkd_valid_predictions(args, checkpoint, output_path):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"FixedKD checkpoint absent: {checkpoint}")
    cfg = build_config(build_cli(args), args.seed)
    loaders = MMDataLoader(cfg, args.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Expected train/valid loaders only.")

    backbone = DLF(cfg).to(cfg.device)
    model = MissingModalityWrapper(
        backbone, cfg.feature_dims[1], cfg.feature_dims[2]
    ).to(cfg.device)
    model.load_state_dict(torch.load(checkpoint, map_location=cfg.device), strict=True)
    model.eval()
    frame = prediction_rows(model, loaders["valid"], cfg.device)
    frame["selected_by"] = "validation"
    frame["diagnostic_only"] = False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def generate_teacher_valid_predictions(args, checkpoint, output_path):
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Clean teacher checkpoint absent: {checkpoint}")
    cfg = build_config(build_cli(args), args.seed)
    loaders = MMDataLoader(cfg, args.num_workers)
    if set(loaders) != {"train", "valid"}:
        raise RuntimeError("Expected train/valid loaders only.")

    teacher = build_frozen_teacher(DLF, cfg, checkpoint)
    rows = []
    teacher.eval()
    for batch in loaders["valid"]:
        text = batch["text"].to(cfg.device)
        audio = batch["audio"].to(cfg.device)
        vision = batch["vision"].to(cfg.device)
        labels = batch["labels"]["M"].view(-1).cpu().numpy()
        indices = batch["index"].view(-1).cpu().numpy().astype(int)
        ids = list(batch["id"])
        pred = teacher_lav_prediction(teacher, text, audio, vision).view(-1).cpu().numpy()
        for j, index in enumerate(indices):
            rows.append({
                "sample_index": int(index),
                "sample_id": str(ids[j]),
                "label": float(labels[j]),
                "teacher_LAV_pred": float(pred[j]),
            })

    frame = pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    if frame.sample_index.duplicated().any():
        raise RuntimeError("Teacher validation predictions contain duplicate sample_index.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)
    return frame


def load_inputs(args, evaluator_path, fixed_pred_path, fixed_ckpt,
                teacher_pred_path, teacher_ckpt, cache_path, output_dir):
    for path in (evaluator_path, cache_path):
        if not path.is_file():
            raise FileNotFoundError(path)
        forbid_test_path(path)
    forbid_test_path(fixed_pred_path)
    forbid_test_path(fixed_ckpt)
    forbid_test_path(teacher_pred_path)
    forbid_test_path(teacher_ckpt)

    evaluator = pd.read_csv(evaluator_path)
    cache = pd.read_csv(cache_path)

    if fixed_pred_path is not None:
        if not fixed_pred_path.is_file():
            raise FileNotFoundError(fixed_pred_path)
        fixedkd = pd.read_csv(fixed_pred_path)
        fixed_source = str(fixed_pred_path)
    else:
        generated = output_dir / f"{args.dataset}_seed{args.seed}_fixedkd_valid_predictions.csv"
        if generated.is_file():
            fixedkd = pd.read_csv(generated)
        else:
            fixedkd = generate_fixedkd_valid_predictions(args, fixed_ckpt, generated)
        fixed_source = str(generated)

    if teacher_pred_path is not None:
        if not teacher_pred_path.is_file():
            raise FileNotFoundError(teacher_pred_path)
        teacher = pd.read_csv(teacher_pred_path)
        teacher_source = str(teacher_pred_path)
    else:
        generated = output_dir / f"{args.dataset}_seed{args.seed}_clean_teacher_valid_predictions.csv"
        if generated.is_file():
            teacher = pd.read_csv(generated)
        else:
            teacher = generate_teacher_valid_predictions(args, teacher_ckpt, generated)
        teacher_source = str(generated)

    pred_cols = ["sample_index", "sample_id", "label", "LAV_pred"] + [f"{m}_pred" for m in MODES]
    require_columns(evaluator, pred_cols, evaluator_path)
    require_columns(fixedkd, pred_cols, fixed_source)
    require_columns(teacher, ["sample_index", "sample_id", "label", "teacher_LAV_pred"], teacher_source)
    require_columns(cache, ["sample_index"] + [f"delta_{m}" for m in MODES], cache_path)

    evaluator = evaluator.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    fixedkd = fixedkd.sort_values("sample_index", kind="mergesort").reset_index(drop=True)
    teacher = teacher.sort_values("sample_index", kind="mergesort").reset_index(drop=True)

    ref = evaluator
    for name, frame in (("FixedKD", fixedkd), ("Teacher", teacher)):
        if len(frame) != len(ref):
            raise ValueError(f"{name} validation row count differs from evaluator.")
        for col in ("sample_index", "sample_id"):
            if not np.array_equal(ref[col].astype(str).to_numpy(), frame[col].astype(str).to_numpy()):
                raise ValueError(f"{name} is not paired with evaluator by {col}.")
        if not np.allclose(ref["label"].to_numpy(float), frame["label"].to_numpy(float),
                           atol=1e-8, rtol=0):
            raise ValueError(f"{name} labels differ from evaluator.")

    return evaluator, fixedkd, teacher, cache, fixed_source, teacher_source


def train_calibrated_compatibility(train_delta, target_delta):
    train = np.sort(np.asarray(train_delta, dtype=np.float64))
    target = np.asarray(target_delta, dtype=np.float64)
    if len(train) == 0 or not np.isfinite(train).all() or np.any(train < 0):
        raise ValueError("Training deltas must be finite, non-negative, and non-empty.")
    if not np.isfinite(target).all() or np.any(target < 0):
        raise ValueError("Target deltas must be finite and non-negative.")

    left = np.searchsorted(train, target, side="left").astype(np.float64)
    right = np.searchsorted(train, target, side="right").astype(np.float64)
    q = (left + 0.5 * (right - left)) / float(len(train))
    lo = 0.5 / float(len(train))
    hi = 1.0 - lo
    q = np.clip(q, lo, hi)
    return 1.0 - q


def smooth_l1_numpy(student, teacher):
    student_t = torch.as_tensor(student, dtype=torch.float64)
    teacher_t = torch.as_tensor(teacher, dtype=torch.float64)
    return F.smooth_l1_loss(student_t, teacher_t, reduction="none").numpy()


def build_long_frame(evaluator, fixedkd, teacher, cache):
    rows = []
    teacher_pred = teacher["teacher_LAV_pred"].to_numpy(dtype=np.float64)

    for mode in MODES:
        evaluator_delta = np.abs(
            evaluator["LAV_pred"].to_numpy(dtype=np.float64)
            - evaluator[f"{mode}_pred"].to_numpy(dtype=np.float64)
        )
        compatibility = train_calibrated_compatibility(
            cache[f"delta_{mode}"].to_numpy(dtype=np.float64),
            evaluator_delta,
        )
        student_pred = fixedkd[f"{mode}_pred"].to_numpy(dtype=np.float64)
        abs_gap = np.abs(teacher_pred - student_pred)
        smooth_l1_gap = smooth_l1_numpy(student_pred, teacher_pred)

        rows.append(pd.DataFrame({
            "sample_index": evaluator["sample_index"].to_numpy(),
            "sample_id": evaluator["sample_id"].astype(str).to_numpy(),
            "label": evaluator["label"].to_numpy(dtype=np.float64),
            "mode": mode,
            "evaluator_delta": evaluator_delta,
            "compatibility": compatibility,
            "teacher_LAV_pred": teacher_pred,
            "fixedkd_student_pred": student_pred,
            "independent_abs_gap": abs_gap,
            "independent_smoothl1_gap": smooth_l1_gap,
        }))

    frame = pd.concat(rows, ignore_index=True)
    frame["compat_quartile"] = pd.cut(
        frame["compatibility"],
        bins=[0.0, 0.25, 0.50, 0.75, 1.0],
        labels=QUARTILE_LABELS,
        include_lowest=True,
        right=True,
    )
    if frame.compat_quartile.isna().any():
        raise RuntimeError("Some compatibility values were not assigned to quartiles.")
    return frame


def cluster_bootstrap_ci(frame, value_col, reps, seed):
    ids = frame["sample_index"].drop_duplicates().to_numpy()
    if len(ids) < 2:
        return np.nan, np.nan
    grouped = {
        idx: frame.loc[frame.sample_index == idx, value_col].to_numpy(float)
        for idx in ids
    }
    rng = np.random.default_rng(seed)
    means = np.empty(reps, dtype=np.float64)
    for r in range(reps):
        sampled = rng.choice(ids, size=len(ids), replace=True)
        values = np.concatenate([grouped[idx] for idx in sampled])
        means[r] = float(values.mean())
    return tuple(np.quantile(means, [0.025, 0.975]))


def aggregate(frame, group_type, group, reps, seed):
    a_lo, a_hi = cluster_bootstrap_ci(frame, "independent_abs_gap", reps, seed)
    s_lo, s_hi = cluster_bootstrap_ci(frame, "independent_smoothl1_gap", reps, seed + 1)
    return {
        "group_type": group_type,
        "group": group,
        "count_rows": int(len(frame)),
        "count_samples": int(frame.sample_index.nunique()),
        "mean_compatibility": float(frame.compatibility.mean()),
        "mean_evaluator_delta": float(frame.evaluator_delta.mean()),
        "mean_abs_gap": float(frame.independent_abs_gap.mean()),
        "median_abs_gap": float(frame.independent_abs_gap.median()),
        "abs_gap_ci95_low": float(a_lo),
        "abs_gap_ci95_high": float(a_hi),
        "mean_smoothl1_gap": float(frame.independent_smoothl1_gap.mean()),
        "median_smoothl1_gap": float(frame.independent_smoothl1_gap.median()),
        "smoothl1_gap_ci95_low": float(s_lo),
        "smoothl1_gap_ci95_high": float(s_hi),
    }


def summarize(frame, reps, seed):
    rows = [aggregate(frame, "overall", "all", reps, seed)]
    for j, label in enumerate(QUARTILE_LABELS):
        local = frame.loc[frame.compat_quartile.astype(str) == label]
        rows.append(aggregate(local, "compat_quartile", label, reps, seed + 100 + j * 5))
    for j, mode in enumerate(MODES):
        local = frame.loc[frame["mode"] == mode]
        rows.append(aggregate(local, "mode", mode, reps, seed + 200 + j * 20))
        for k, label in enumerate(QUARTILE_LABELS):
            cell = local.loc[local.compat_quartile.astype(str) == label]
            rows.append(aggregate(
                cell, f"mode_{mode}_quartile", label, reps, seed + 300 + j * 50 + k * 5
            ))
    summary = pd.DataFrame(rows)

    def corr(x, y):
        pearson = float(frame[x].corr(frame[y], method="pearson"))
        spearman = float(
            frame[x].rank(method="average").corr(
                frame[y].rank(method="average"), method="pearson"
            )
        )
        return pearson, spearman

    p_abs, s_abs = corr("compatibility", "independent_abs_gap")
    p_sl1, s_sl1 = corr("compatibility", "independent_smoothl1_gap")

    q = summary.loc[summary.group_type == "compat_quartile"].set_index("group")
    diagnostics = {
        "split": "validation_only",
        "compatibility_source":
            "ModDrop evaluator validation LAV-vs-condition delta calibrated by the same-mode train-only delta distribution",
        "independent_gap_source":
            "clean DLF teacher LAV prediction vs FixedKD student condition prediction",
        "pearson_compatibility_abs_gap": p_abs,
        "spearman_compatibility_abs_gap": s_abs,
        "pearson_compatibility_smoothl1_gap": p_sl1,
        "spearman_compatibility_smoothl1_gap": s_sl1,
        "q1_mean_abs_gap": float(q.loc["Q1_low", "mean_abs_gap"]),
        "q4_mean_abs_gap": float(q.loc["Q4_high", "mean_abs_gap"]),
        "q1_minus_q4_abs_gap": float(
            q.loc["Q1_low", "mean_abs_gap"] - q.loc["Q4_high", "mean_abs_gap"]
        ),
        "q1_mean_smoothl1_gap": float(q.loc["Q1_low", "mean_smoothl1_gap"]),
        "q4_mean_smoothl1_gap": float(q.loc["Q4_high", "mean_smoothl1_gap"]),
        "q1_minus_q4_smoothl1_gap": float(
            q.loc["Q1_low", "mean_smoothl1_gap"] - q.loc["Q4_high", "mean_smoothl1_gap"]
        ),
    }
    return summary, diagnostics


def make_plot(summary, output_path):
    import matplotlib.pyplot as plt

    q = summary.loc[summary.group_type == "compat_quartile"].copy()
    order = {name: i for i, name in enumerate(QUARTILE_LABELS)}
    q["order"] = q["group"].map(order)
    q = q.sort_values("order")

    x = np.arange(len(q))
    y = q["mean_abs_gap"].to_numpy(float)
    lower = y - q["abs_gap_ci95_low"].to_numpy(float)
    upper = q["abs_gap_ci95_high"].to_numpy(float) - y

    fig, ax = plt.subplots(figsize=(5.3, 3.5))
    ax.errorbar(x, y, yerr=np.vstack([lower, upper]), marker="o", capsize=4, linewidth=1.8)
    ax.set_xticks(x)
    ax.set_xticklabels(["Low", "Q2", "Q3", "High"])
    ax.set_xlabel("Compatibility quartile")
    ax.set_ylabel("Teacher--student prediction gap")
    ax.set_title("Compatibility vs. Independent Distillation Discrepancy")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    args = parse_args()
    evaluator_path, fixed_pred_path, fixed_ckpt, teacher_pred_path, teacher_ckpt, cache_path, output_dir = default_paths(args)

    evaluator, fixedkd, teacher, cache, fixed_source, teacher_source = load_inputs(
        args, evaluator_path, fixed_pred_path, fixed_ckpt,
        teacher_pred_path, teacher_ckpt, cache_path, output_dir
    )
    frame = build_long_frame(evaluator, fixedkd, teacher, cache)
    summary, diagnostics = summarize(frame, args.bootstrap_reps, args.bootstrap_seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    samples_path = output_dir / f"{args.dataset}_seed{args.seed}_compatibility_kd_gap_samples.csv"
    summary_path = output_dir / f"{args.dataset}_seed{args.seed}_compatibility_kd_gap_summary.csv"
    diagnostics_path = output_dir / f"{args.dataset}_seed{args.seed}_compatibility_kd_gap_diagnostics.json"
    plot_path = output_dir / f"{args.dataset}_seed{args.seed}_compatibility_kd_gap.png"

    frame.to_csv(samples_path, index=False)
    summary.to_csv(summary_path, index=False)
    diagnostics.update({
        "evaluator_valid_predictions": str(evaluator_path),
        "fixedkd_valid_predictions": fixed_source,
        "clean_teacher_valid_predictions": teacher_source,
        "train_cache": str(cache_path),
        "fixedkd_checkpoint": str(fixed_ckpt),
        "clean_teacher_checkpoint": str(teacher_ckpt),
    })
    diagnostics_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not args.no_plot:
        make_plot(summary, plot_path)

    cols = [
        "group_type", "group", "count_rows", "mean_compatibility",
        "mean_abs_gap", "median_abs_gap", "abs_gap_ci95_low", "abs_gap_ci95_high",
        "mean_smoothl1_gap"
    ]
    print("Validation-only Compatibility--Distillation Discrepancy analysis complete.")
    print(f"summary={summary_path}")
    print(f"diagnostics={diagnostics_path}")
    if not args.no_plot:
        print(f"plot={plot_path}")
    print()
    print(summary.loc[
        summary.group_type.isin(["overall", "compat_quartile"]), cols
    ].to_string(index=False))
    print()
    print("Mode-wise:")
    print(summary.loc[summary.group_type == "mode", cols].to_string(index=False))
    print()
    print(json.dumps(diagnostics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
