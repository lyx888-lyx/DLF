#!/usr/bin/env python
"""Build and audit Stage23C development-only Soft-Oracle targets.

Only Train OOF Expert ledgers and Direction inner-train/inner-valid labels are
used.  No outer Oracle responsibility or target is constructed.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr

from stage23a_v2_common import EXPERTS, MODES, ROOT, regression_metrics


V1 = ROOT / "result" / "arbiter_audit_v1" / "mosei"
V2 = ROOT / "result" / "arbiter_audit_v2" / "mosei"
V2B = ROOT / "result" / "arbiter_audit_v2b" / "mosei"
OUT = ROOT / "result" / "oracle_distillation_v1" / "mosei"
RUNTIME = ROOT / "runtime" / "stage23c"
TAUS = (0.02, 0.05, 0.10, 0.20)
FIXED_L2 = (0.0, 0.001, 0.01)
SHUFFLE_SEEDS = (23611, 23612, 23613)
HEADS = (
    "output_logit",
    "logits_c",
    "logits_l_hetero",
    "logits_a_hetero",
    "logits_v_hetero",
)
EXPECTED = {
    "A": {
        "fold": 0,
        "inner_train": (29752, 7438, 1007),
        "inner_valid": (4376, 1094, 130),
        "outer_evaluation": (31176, 7794, 1112),
    },
    "B": {
        "fold": 1,
        "inner_train": (27732, 6933, 980),
        "inner_valid": (3444, 861, 132),
        "outer_evaluation": (34128, 8532, 1137),
    },
}


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value):
    payload = json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    os.replace(str(temporary), str(path))


def write_tsv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, sep="\t", index=False, float_format="%.10g")
    os.replace(str(temporary), str(path))


def write_gzip_csv(frame, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as raw:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6
        ) as compressed:
            frame.to_csv(compressed, index=False, float_format="%.10g")
    os.replace(str(temporary), str(path))


def softmax(values):
    values = np.asarray(values, dtype=np.float64)
    shifted = values - values.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def overall_j(frame, prediction):
    maes = {}
    values = np.asarray(prediction, dtype=np.float64)
    for mode in MODES:
        mask = frame["mode"].to_numpy() == mode
        maes[mode] = float(
            np.mean(
                np.abs(
                    values[mask]
                    - frame.loc[mask, "label"].to_numpy(dtype=np.float64)
                )
            )
        )
    return 0.5 * maes["LAV"] + 0.5 * np.mean(
        [maes[mode] for mode in ("LA", "LV", "L")]
    )


def load_roles(direction):
    root = V2 / "features" / "meta57" / f"direction_{direction}"
    roles = {}
    for role in ("inner_train", "inner_valid", "outer_evaluation"):
        frame = pd.read_csv(
            root / f"{role}_targets_and_audit.csv",
            usecols=[
                "meta_row_id",
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "expert_fold",
                "role",
                "label",
            ],
            dtype={"meta_row_id": str, "sample_id": str, "video_id": str},
        )
        expected = EXPECTED[direction][role]
        observed = (
            len(frame),
            frame["sample_id"].nunique(),
            frame["video_id"].nunique(),
        )
        if observed != expected:
            raise RuntimeError(
                f"{direction} {role} count mismatch {observed} != {expected}"
            )
        if frame["meta_row_id"].duplicated().any():
            raise RuntimeError(f"{direction} {role} duplicate meta row")
        roles[role] = frame
    source_sets = {
        role: set(frame["video_id"]) for role, frame in roles.items()
    }
    if (
        source_sets["inner_train"] & source_sets["inner_valid"]
        or source_sets["inner_train"] & source_sets["outer_evaluation"]
        or source_sets["inner_valid"] & source_sets["outer_evaluation"]
    ):
        raise RuntimeError(f"{direction} source leakage")
    return roles


def load_hierarchical(fold):
    path = (
        V2
        / "features"
        / "hierarchical"
        / f"hierarchical_logits_fold{fold}.csv.gz"
    )
    frame = pd.read_csv(
        path,
        compression="gzip",
        usecols=[
            "sample_id",
            "video_id",
            "train_index",
            "outer_fold",
            "mode",
            "expert_id",
            "label",
            *HEADS,
            "sample_binding_sha256",
            "row_binding_sha256",
        ],
        dtype={"sample_id": str, "video_id": str},
    )
    if set(frame["expert_id"]) != set(EXPERTS):
        raise RuntimeError("Frozen Expert pool changed")
    if frame.duplicated(["sample_id", "mode", "expert_id"]).any():
        raise RuntimeError("Duplicate hierarchical Expert row")
    expected = frame["sample_id"].nunique() * len(MODES) * len(EXPERTS)
    if len(frame) != expected:
        raise RuntimeError("Incomplete hierarchical ledger")
    if not np.isfinite(frame[list(HEADS)].to_numpy(dtype=float)).all():
        raise RuntimeError("Non-finite hierarchical prediction")
    return frame, path


def binding_audit(fold, hierarchical):
    old_path = (
        V1 / "expert_oof" / f"outer_fold{fold}" / "oof_predictions.csv"
    )
    old = pd.read_csv(
        old_path, dtype={"sample_id": str, "video_id": str}
    )
    joined = old.merge(
        hierarchical[
            [
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "expert_id",
                "label",
                "output_logit",
            ]
        ],
        on=["sample_id", "video_id", "train_index", "mode", "expert_id"],
        how="outer",
        suffixes=("_oof", "_hier"),
        indicator=True,
        validate="one_to_one",
    )
    prediction_diff = np.abs(
        joined["prediction"].to_numpy(dtype=float)
        - joined["output_logit"].to_numpy(dtype=float)
    )
    label_diff = np.abs(
        joined["label_oof"].to_numpy(dtype=float)
        - joined["label_hier"].to_numpy(dtype=float)
    )
    return {
        "fold": fold,
        "oof_path": str(old_path.resolve()),
        "oof_sha256": sha256(old_path),
        "hierarchical_rows": len(hierarchical),
        "oof_rows": len(old),
        "join_non_both_rows": int((joined["_merge"] != "both").sum()),
        "max_abs_final_prediction_diff": float(np.max(prediction_diff)),
        "max_abs_label_diff": float(np.max(label_diff)),
        "binding_pass": bool(
            (joined["_merge"] == "both").all()
            and np.max(prediction_diff) <= 1e-6
            and np.max(label_diff) <= 1e-8
        ),
    }


def pivot_experts(meta, hierarchical):
    keys = ["sample_id", "video_id", "train_index", "mode"]
    expected = meta[keys + ["label", "role", "meta_row_id"]].copy()
    local = hierarchical.merge(
        expected[keys],
        on=keys,
        how="inner",
        validate="many_to_one",
    )
    if len(local) != len(meta) * len(EXPERTS):
        raise RuntimeError("Hierarchical/meta binding is incomplete")
    output = expected.copy()
    for head in HEADS:
        pivot = local.pivot(
            index=keys, columns="expert_id", values=head
        ).reindex(columns=EXPERTS)
        pivot = pivot.reset_index()
        renamed = pivot.rename(
            columns={expert: f"{head}__{expert}" for expert in EXPERTS}
        )
        output = output.merge(
            renamed, on=keys, how="left", validate="one_to_one"
        )
    if output.isna().any().any():
        raise RuntimeError("Missing pivot value")
    return output


def prediction_matrix(frame, head="output_logit"):
    return frame[
        [f"{head}__{expert}" for expert in EXPERTS]
    ].to_numpy(dtype=np.float64)


def responsibilities(frame, tau):
    predictions = prediction_matrix(frame)
    label = frame["label"].to_numpy(dtype=np.float64)[:, None]
    errors = np.abs(predictions - label)
    regret = errors - errors.min(axis=1, keepdims=True)
    q = softmax(-regret / float(tau))
    return q, errors, regret


def fit_simplex(predictions, label, l2):
    predictions = np.asarray(predictions, dtype=np.float64)
    label = np.asarray(label, dtype=np.float64)
    center = np.full(predictions.shape[1], 1.0 / predictions.shape[1])

    def objective(weights):
        residual = predictions @ weights - label
        return float(np.mean(np.square(residual)) + l2 * np.sum(np.square(weights - center)))

    result = minimize(
        objective,
        center,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * predictions.shape[1],
        constraints={"type": "eq", "fun": lambda weights: weights.sum() - 1.0},
        options={"maxiter": 2000, "ftol": 1e-12},
    )
    if not result.success:
        raise RuntimeError(f"Simplex fit failed: {result.message}")
    weights = np.maximum(result.x, 0)
    return weights / weights.sum()


def fit_fixed_weights(frame, fit_mask, l2):
    weights = {}
    for mode in MODES:
        mask = fit_mask & (frame["mode"].to_numpy() == mode)
        weights[mode] = fit_simplex(
            prediction_matrix(frame)[mask],
            frame.loc[mask, "label"].to_numpy(dtype=float),
            float(l2),
        )
    return weights


def apply_fixed(frame, weights):
    predictions = prediction_matrix(frame)
    result = np.zeros(len(frame), dtype=np.float64)
    for mode in MODES:
        mask = frame["mode"].to_numpy() == mode
        result[mask] = predictions[mask] @ weights[mode]
    return result


def derange_by_source(sources, seed):
    rng = np.random.RandomState(int(seed))
    sources = np.asarray(sources, dtype=str)
    unique = np.unique(sources)
    rng.shuffle(unique)
    groups = []
    for source in unique:
        group = np.flatnonzero(sources == source)
        rng.shuffle(group)
        groups.append(group)
    ordered = np.concatenate(groups)
    maximum = max(len(group) for group in groups)
    shifts = np.arange(maximum, len(sources) - maximum + 1)
    rng.shuffle(shifts)
    for shift in shifts:
        permutation = np.empty(len(sources), dtype=np.int64)
        permutation[ordered] = np.roll(ordered, int(shift))
        if not np.any(sources == sources[permutation]):
            return permutation
    raise RuntimeError("Could not construct cross-source q derangement")


def shuffled_q(frame, q, scope_mask, seed):
    output = q.copy()
    modes = frame["mode"].to_numpy()
    sources = frame["video_id"].to_numpy(dtype=str)
    for mode_index, mode in enumerate(MODES):
        indices = np.flatnonzero(scope_mask & (modes == mode))
        permutation = derange_by_source(
            sources[indices], int(seed) + mode_index * 10000
        )
        output[indices] = q[indices[permutation]]
        if np.any(sources[indices] == sources[indices[permutation]]):
            raise RuntimeError("Shuffled q retained a source pairing")
    return output


def weighted_heads(frame, q):
    return {
        head: np.sum(q * prediction_matrix(frame, head), axis=1)
        for head in HEADS
    }


def sanity_rows(direction, frame, tau_targets, equal, fixed_candidates):
    rows = []
    role_masks = {
        role: frame["role"].to_numpy() == role
        for role in ("inner_train", "inner_valid")
    }
    hard = np.min(
        np.abs(
            prediction_matrix(frame)
            - frame["label"].to_numpy(dtype=float)[:, None]
        ),
        axis=1,
    )
    for role, role_mask in role_masks.items():
        for mode in (*MODES, "Overall"):
            mask = (
                role_mask
                if mode == "Overall"
                else role_mask & (frame["mode"].to_numpy() == mode)
            )
            label = frame.loc[mask, "label"].to_numpy(dtype=float)
            methods = {
                "Equal Teacher": equal,
                "Hard select-one Oracle": (
                    frame["label"].to_numpy(dtype=float)
                    + hard
                ),
            }
            for l2, values in fixed_candidates.items():
                methods[f"Fixed Teacher l2={l2}"] = values
            for tau, values in tau_targets.items():
                methods[f"Soft-Oracle final tau={tau}"] = values
            for name, values in methods.items():
                if name == "Hard select-one Oracle":
                    mae = float(hard[mask].mean())
                    metrics = {"MAE": mae, "Corr": "NA"}
                else:
                    metrics = regression_metrics(values[mask], label)
                rows.append(
                    {
                        "direction": direction,
                        "role": role,
                        "mode": mode,
                        "teacher": name,
                        "MAE": metrics["MAE"],
                        "Corr": metrics["Corr"],
                    }
                )
    return rows


def main():
    protocol_path = OUT / "protocol" / "frozen_protocol_manifest.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if not protocol["oracle_target_audit_authorized"]:
        raise RuntimeError("Oracle target audit is not authorized")
    if (
        protocol["outer_evaluation_access_count"]
        or protocol["official_valid_access_count"]
        or protocol["locked_test_access_count"]
    ):
        raise RuntimeError("Protected access count is nonzero")

    v2a = json.loads(
        (V2 / "analysis_v2a" / "stage23a_v2a_signal_audit.json").read_text(
            encoding="utf-8"
        )
    )
    v2b = json.loads(
        (V2B / "final" / "stage23a_v2b_audit.json").read_text(
            encoding="utf-8"
        )
    )
    if v2a["status"] != "SIGNAL_AUDIT_FAIL":
        raise RuntimeError("v2a conclusion changed")
    if v2b["status"] != "LOCAL_COMPETENCE_WEAK":
        raise RuntimeError("v2b conclusion changed")

    hierarchical = {}
    hierarchical_paths = {}
    binding = []
    for fold in (0, 1):
        hierarchical[fold], hierarchical_paths[fold] = load_hierarchical(fold)
        local_binding = binding_audit(fold, hierarchical[fold])
        binding.append(local_binding)
        if not local_binding["binding_pass"]:
            raise RuntimeError(f"Fold {fold} OOF/hierarchical binding failed")

    all_sanity = []
    all_responsibility = []
    selection = {
        "stage": "Stage23C development target selection",
        "selection_roles": ["inner_train", "inner_valid"],
        "outer_target_constructed": False,
        "directions": {},
    }
    asset_roles = []
    for direction in ("A", "B"):
        roles = load_roles(direction)
        for role, role_frame in roles.items():
            asset_roles.append(
                {
                    "direction": direction,
                    "role": role,
                    "meta_rows": len(role_frame),
                    "samples": role_frame["sample_id"].nunique(),
                    "sources": role_frame["video_id"].nunique(),
                    "modes": "|".join(sorted(role_frame["mode"].unique())),
                }
            )
        development = pd.concat(
            [roles["inner_train"], roles["inner_valid"]], ignore_index=True
        )
        fold = EXPECTED[direction]["fold"]
        frame = pivot_experts(development, hierarchical[fold])
        frame = frame.sort_values(
            ["role", "train_index", "mode"], kind="mergesort"
        ).reset_index(drop=True)
        train_mask = frame["role"].to_numpy() == "inner_train"
        valid_mask = frame["role"].to_numpy() == "inner_valid"
        predictions = prediction_matrix(frame)
        labels = frame["label"].to_numpy(dtype=np.float64)
        equal = predictions.mean(axis=1)

        fixed_candidates = {}
        fixed_screen_weights = {}
        fixed_rows = []
        for l2 in FIXED_L2:
            weights = fit_fixed_weights(frame, train_mask, l2)
            values = apply_fixed(frame, weights)
            fixed_candidates[l2] = values
            fixed_screen_weights[l2] = weights
            fixed_rows.append(
                {
                    "direction": direction,
                    "l2": l2,
                    "inner_valid_J": overall_j(
                        frame.loc[valid_mask].reset_index(drop=True),
                        values[valid_mask],
                    ),
                    **{
                        f"{mode}__{expert}": float(weights[mode][expert_index])
                        for mode in MODES
                        for expert_index, expert in enumerate(EXPERTS)
                    },
                }
            )
        fixed_table = pd.DataFrame(fixed_rows).sort_values(
            ["inner_valid_J", "l2"]
        )
        selected_l2 = float(fixed_table.iloc[0]["l2"])
        screen_weights = fixed_screen_weights[selected_l2]
        final_weights = fit_fixed_weights(
            frame, np.ones(len(frame), dtype=bool), selected_l2
        )
        screen_fixed = apply_fixed(frame, screen_weights)
        final_fixed = apply_fixed(frame, final_weights)
        write_tsv(
            fixed_table,
            OUT / "targets" / f"direction_{direction}_fixed_weight_candidates.tsv",
        )
        weight_rows = []
        for fit_scope, values in (
            ("inner_train", screen_weights),
            ("full_development", final_weights),
        ):
            for mode in MODES:
                for expert_index, expert in enumerate(EXPERTS):
                    weight_rows.append(
                        {
                            "direction": direction,
                            "fit_scope": fit_scope,
                            "selected_l2": selected_l2,
                            "mode": mode,
                            "expert_id": expert,
                            "weight": values[mode][expert_index],
                        }
                    )
        write_tsv(
            pd.DataFrame(weight_rows),
            OUT / "targets" / f"direction_{direction}_fixed_weights.tsv",
        )

        tau_targets = {}
        tau_rows = []
        q_by_tau = {}
        error = regret = None
        for tau in TAUS:
            q, error, regret = responsibilities(frame, tau)
            values = np.sum(q * predictions, axis=1)
            q_by_tau[tau] = q
            tau_targets[tau] = values
            tau_rows.append(
                {
                    "direction": direction,
                    "tau": tau,
                    "inner_train_J": overall_j(
                        frame.loc[train_mask].reset_index(drop=True),
                        values[train_mask],
                    ),
                    "inner_valid_J": overall_j(
                        frame.loc[valid_mask].reset_index(drop=True),
                        values[valid_mask],
                    ),
                    "inner_valid_mean_entropy": float(
                        np.mean(
                            -np.sum(
                                q[valid_mask]
                                * np.log(np.clip(q[valid_mask], 1e-12, 1.0)),
                                axis=1,
                            )
                        )
                    ),
                }
            )
        tau_table = pd.DataFrame(tau_rows).sort_values(
            ["inner_valid_J", "tau"]
        )
        selected_tau = float(tau_table.iloc[0]["tau"])
        q = q_by_tau[selected_tau]
        oracle_heads = weighted_heads(frame, q)
        oracle_final = oracle_heads["output_logit"]
        write_tsv(
            tau_table,
            OUT / "targets" / f"direction_{direction}_tau_candidates.tsv",
        )

        entropy = -np.sum(q * np.log(np.clip(q, 1e-12, 1.0)), axis=1)
        ordered_q = np.sort(q, axis=1)
        ordered_error = np.sort(error, axis=1)
        responsibility = frame[
            [
                "meta_row_id",
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "role",
                "label",
            ]
        ].copy()
        responsibility["selected_tau"] = selected_tau
        responsibility["q_entropy"] = entropy
        responsibility["top1_responsibility"] = ordered_q[:, -1]
        responsibility["top2_responsibility_sum"] = (
            ordered_q[:, -1] + ordered_q[:, -2]
        )
        responsibility["best_second_error_margin"] = (
            ordered_error[:, 1] - ordered_error[:, 0]
        )
        responsibility["hard_selection_indicator"] = (
            ordered_q[:, -1] >= 1.0 - 1e-8
        ).astype(int)
        for expert_index, expert in enumerate(EXPERTS):
            responsibility[f"error__{expert}"] = error[:, expert_index]
            responsibility[f"regret__{expert}"] = regret[:, expert_index]
            responsibility[f"q__{expert}"] = q[:, expert_index]
        responsibility_path = (
            OUT
            / "targets"
            / f"direction_{direction}_soft_oracle_responsibility.csv.gz"
        )
        write_gzip_csv(responsibility, responsibility_path)
        all_responsibility.append(responsibility.assign(direction=direction))

        target = frame[
            [
                "meta_row_id",
                "sample_id",
                "video_id",
                "train_index",
                "mode",
                "role",
                "label",
            ]
        ].copy()
        target["equal_final"] = equal
        target["screen_fixed_final"] = screen_fixed
        target["final_fixed_final"] = final_fixed
        for head, values in oracle_heads.items():
            target[f"oracle__{head}"] = values
        lower = predictions.min(axis=1)
        upper = predictions.max(axis=1)
        convex_violation = (oracle_final < lower - 1e-10) | (
            oracle_final > upper + 1e-10
        )
        target["oracle_final_convex_hull_violation"] = convex_violation.astype(
            int
        )
        if convex_violation.any():
            raise RuntimeError("Soft-Oracle final target left convex hull")

        npz_payload = {
            "meta_row_id": target["meta_row_id"].to_numpy(dtype=str),
            "train_index": target["train_index"].to_numpy(dtype=np.int64),
            "mode": target["mode"].to_numpy(dtype=str),
            "role": target["role"].to_numpy(dtype=str),
            "equal_final": equal.astype(np.float32),
            "screen_fixed_final": screen_fixed.astype(np.float32),
            "final_fixed_final": final_fixed.astype(np.float32),
        }
        for head, values in oracle_heads.items():
            npz_payload[f"oracle__{head}"] = values.astype(np.float32)

        for seed in SHUFFLE_SEEDS:
            screen_q = shuffled_q(frame, q, train_mask, seed)
            final_q = shuffled_q(
                frame, q, np.ones(len(frame), dtype=bool), seed
            )
            for scope, values in (
                ("screen", weighted_heads(frame, screen_q)),
                ("final", weighted_heads(frame, final_q)),
            ):
                for head, head_values in values.items():
                    npz_payload[
                        f"s6_{scope}_seed{seed}__{head}"
                    ] = head_values.astype(np.float32)

        npz_path = OUT / "targets" / f"direction_{direction}_training_targets.npz"
        temporary = npz_path.with_suffix(".npz.tmp")
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **npz_payload)
        os.replace(str(temporary), str(npz_path))
        target_path = (
            OUT
            / "targets"
            / f"direction_{direction}_teacher_targets.csv.gz"
        )
        write_gzip_csv(target, target_path)

        sanity = sanity_rows(
            direction, frame, tau_targets, equal, fixed_candidates
        )
        all_sanity.extend(sanity)
        q_summary = []
        for role in ("inner_train", "inner_valid"):
            role_mask = frame["role"].to_numpy() == role
            for mode in MODES:
                mask = role_mask & (frame["mode"].to_numpy() == mode)
                row = {
                    "direction": direction,
                    "role": role,
                    "mode": mode,
                    "samples": int(mask.sum()),
                    "mean_entropy": float(entropy[mask].mean()),
                    "mean_top1": float(ordered_q[mask, -1].mean()),
                    "mean_top2_sum": float(
                        (ordered_q[mask, -1] + ordered_q[mask, -2]).mean()
                    ),
                    "hard_fraction": float(
                        (ordered_q[mask, -1] >= 1.0 - 1e-8).mean()
                    ),
                    "entropy_margin_spearman": float(
                        spearmanr(
                            entropy[mask],
                            ordered_error[mask, 1]
                            - ordered_error[mask, 0],
                        ).correlation
                    ),
                }
                for expert_index, expert in enumerate(EXPERTS):
                    row[f"mean_q__{expert}"] = float(
                        q[mask, expert_index].mean()
                    )
                q_summary.append(row)
        write_tsv(
            pd.DataFrame(q_summary),
            OUT / "targets" / f"direction_{direction}_responsibility_summary.tsv",
        )

        selection["directions"][direction] = {
            "development_fold": fold,
            "selected_tau": selected_tau,
            "selected_fixed_l2": selected_l2,
            "responsibility_path": str(responsibility_path.resolve()),
            "responsibility_sha256": sha256(responsibility_path),
            "teacher_target_path": str(target_path.resolve()),
            "teacher_target_sha256": sha256(target_path),
            "training_target_npz_path": str(npz_path.resolve()),
            "training_target_npz_sha256": sha256(npz_path),
            "outer_target_constructed": False,
            "convex_hull_violations": int(convex_violation.sum()),
        }

    sanity_frame = pd.DataFrame(all_sanity)
    write_tsv(sanity_frame, OUT / "targets" / "teacher_target_sanity.tsv")
    write_tsv(
        pd.concat(all_responsibility, ignore_index=True)
        .groupby(["direction", "role", "mode"], as_index=False)
        .agg(
            rows=("meta_row_id", "size"),
            mean_entropy=("q_entropy", "mean"),
            mean_top1=("top1_responsibility", "mean"),
            mean_top2=("top2_responsibility_sum", "mean"),
            hard_fraction=("hard_selection_indicator", "mean"),
            mean_margin=("best_second_error_margin", "mean"),
        ),
        OUT / "targets" / "responsibility_overview.tsv",
    )
    write_json(
        OUT / "protocol" / "frozen_target_selection.json", selection
    )

    asset_audit = {
        "stage": "Stage23C read-only asset and target audit",
        "status": "PASS",
        "completed_at": utc_now(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=str(ROOT), text=True
        ).strip(),
        "head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip(),
        "frozen_conclusions": {
            "Stage23A-v2a": v2a["status"],
            "Stage23A-v2b": v2b["status"],
        },
        "protocol_manifest_sha256": sha256(protocol_path),
        "total_meta_rows": 65304,
        "total_samples": 16326,
        "experts": list(EXPERTS),
        "hierarchical_heads": list(HEADS),
        "role_counts": asset_roles,
        "binding_audit": binding,
        "hierarchical_assets": {
            str(fold): {
                "path": str(hierarchical_paths[fold].resolve()),
                "sha256": sha256(hierarchical_paths[fold]),
                "rows": len(hierarchical[fold]),
            }
            for fold in (0, 1)
        },
        "sample_binding_pass": all(row["binding_pass"] for row in binding),
        "all_experts_answer_same_sample_mode": True,
        "student_modality_advantage_over_experts": False,
        "outer_oracle_target_constructed": False,
        "outer_evaluation_access_count": 0,
        "official_valid_access_count": 0,
        "locked_test_access_count": 0,
        "formal_training_processes_at_audit": [],
    }
    write_json(OUT / "audit" / "asset_audit.json", asset_audit)

    state_path = RUNTIME / "state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    state.update(
        {
            "status": "TARGETS_FROZEN_STUDENT_SCREEN_NOT_STARTED",
            "target_selection_path": str(
                (OUT / "protocol" / "frozen_target_selection.json").resolve()
            ),
            "target_selection_sha256": sha256(
                OUT / "protocol" / "frozen_target_selection.json"
            ),
            "outer_evaluation_access_count": 0,
            "official_valid_access_count": 0,
            "locked_test_access_count": 0,
            "updated_at": utc_now(),
        }
    )
    write_json(state_path, state)
    print(json.dumps(state, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
