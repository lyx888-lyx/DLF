"""Stage 5A utilities for compatibility-routed dual-teacher prediction KD."""
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from trains.singleTask.cf_compat_kd_utils import CACHE_COLUMNS, cache_paths
from trains.singleTask.fixed_kd_utils import checkpoint_sha256, teacher_lav_prediction
from trains.singleTask.missing_utils import MISSING_MODES, mode_to_mask


AUDIT_VERSION = "crdtd_v1"
TEACHER_ROUTES = {
    "mode_only": ("DLF-ModeTeacherKD-v1", "mode_teacher_kd_v1", "mode-teacher"),
    "uniform_dual": ("DLF-UniformDualTeacherKD-v1", "uniform_dual_teacher_kd_v1", "uniform-dual-teacher"),
    "compatibility_routed": ("DLF-CRDTD-v1", "crdtd_v1", "crdtd"),
}


def capture_rng_state():
    return {
        "python": random.getstate(), "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def preserve_rng_and_modes(*models):
    state = capture_rng_state()
    modes = [model.training for model in models]
    try:
        yield
    finally:
        restore_rng_state(state)
        for model, training in zip(models, modes):
            model.train(training)


def dual_teacher_audit_paths(root="result", dataset="mosi", seed=1111):
    directory = Path(root) / "dual_teacher" / AUDIT_VERSION / dataset / "seed{}".format(int(seed))
    return {
        "directory": directory,
        "targets": directory / "train_dual_teacher_targets.csv",
        "samples": directory / "dual_teacher_suitability_samples.csv",
        "quartiles": directory / "dual_teacher_suitability_quartiles.csv",
        "summary": directory / "dual_teacher_suitability_summary.json",
        "manifest": directory / "dual_teacher_audit_manifest.json",
    }


def locate_stage3_cache(root="result", dataset="mosi"):
    """Locate the locked Stage 3 cache through its config manifest and verify it."""
    paths = cache_paths(root, dataset)
    config_path = paths["config"]
    if not config_path.is_file():
        raise FileNotFoundError("Stage 3 compatibility manifest is absent: {}".format(config_path))
    config = json.loads(config_path.read_text())
    csv_path = paths["csv"]
    if config.get("version") != "cf_compat_v1" or config.get("source") != "train_only":
        raise ValueError("Stage 3 compatibility manifest is not the locked train-only cache.")
    if not csv_path.is_file():
        raise FileNotFoundError("Manifest-bound compatibility CSV is absent: {}".format(csv_path))
    frame = pd.read_csv(csv_path)
    if list(frame.columns) != list(CACHE_COLUMNS):
        raise ValueError("Compatibility cache schema differs from frozen Stage 3.")
    validate_train_cache(frame)
    return csv_path, config_path, config, frame


def validate_train_cache(frame, expected_count=1284):
    required = {
        "sample_index", "sample_id", "label", "evaluator_LAV_pred",
        "evaluator_LA_pred", "evaluator_LV_pred", "evaluator_L_pred",
        "delta_LA", "delta_LV", "delta_L", "compat_LA", "compat_LV", "compat_L",
    }
    if not required.issubset(frame.columns):
        raise ValueError("Compatibility cache lacks required dual-teacher fields.")
    if len(frame) != expected_count or frame.sample_index.nunique() != expected_count:
        raise ValueError("Compatibility cache must contain exactly {} unique train samples.".format(expected_count))
    expected = np.arange(expected_count)
    if not np.array_equal(np.sort(frame.sample_index.astype(int).to_numpy()), expected):
        raise ValueError("Compatibility cache has missing or non-contiguous sample indices.")
    for mode in MISSING_MODES:
        values = frame["compat_{}".format(mode)].to_numpy(float)
        if not np.isfinite(values).all() or not ((values > 0) & (values < 1)).all():
            raise ValueError("Compatibility must be finite and strictly inside (0,1).")


def cache_table(frame):
    validate_train_cache(frame, len(frame))
    return {int(row.sample_index): row._asdict() for row in frame.itertuples(index=False)}


def lookup_mode_values(table, indices, modes, stem, device, dtype):
    if len(indices) != len(modes):
        raise ValueError("sample_index and missing mode lengths differ.")
    values = []
    for index, mode in zip(indices, modes):
        if mode not in MISSING_MODES:
            raise ValueError("Mode target lookup permits only LA/LV/L.")
        if int(index) not in table:
            raise KeyError("Missing train cache binding for sample_index {}.".format(index))
        key = "evaluator_{}_pred".format(mode) if stem == "evaluator" else "{}_{}".format(stem, mode)
        if key == "evaluator_LAV_pred":
            raise ValueError("Mode target must never use evaluator_LAV_pred.")
        if key not in table[int(index)]:
            raise KeyError("Cache field {} is absent.".format(key))
        values.append(float(table[int(index)][key]))
    tensor = torch.tensor(values, device=device, dtype=dtype).view(-1)
    if not torch.isfinite(tensor).all():
        raise ValueError("Non-finite cache lookup.")
    return tensor.detach()


def mode_teacher_targets(table, indices, modes, device, dtype):
    return lookup_mode_values(table, indices, modes, "evaluator", device, dtype)


def compatibility_targets(table, indices, modes, device, dtype):
    values = lookup_mode_values(table, indices, modes, "compat", device, dtype)
    if not torch.all((values > 0) & (values < 1)):
        raise ValueError("Compatibility route is outside (0,1).")
    return values


def route_alpha(route, compatibility):
    compatibility = compatibility.detach().view(-1)
    if route == "mode_only":
        alpha = torch.zeros_like(compatibility)
    elif route == "uniform_dual":
        alpha = torch.full_like(compatibility, .5)
    elif route == "compatibility_routed":
        if not torch.all((compatibility > 0) & (compatibility < 1)):
            raise ValueError("Compatibility route requires C strictly in (0,1).")
        alpha = compatibility
    else:
        raise ValueError("Unknown teacher route: {}".format(route))
    if not torch.isfinite(alpha).all() or not torch.equal(alpha + (1 - alpha), torch.ones_like(alpha)):
        raise ValueError("Per-sample route weights must sum exactly to one.")
    return alpha.detach()


def routed_dual_teacher_loss(student_prediction, full_target, mode_target, alpha):
    student = student_prediction.reshape(-1)
    full = full_target.detach().to(student).reshape(-1)
    mode = mode_target.detach().to(student).reshape(-1)
    alpha = alpha.detach().to(student).reshape(-1)
    if not (student.shape == full.shape == mode.shape == alpha.shape):
        raise ValueError("Dual-teacher loss inputs must all have shape [batch].")
    d_full = F.smooth_l1_loss(student, full, reduction="none").reshape(-1)
    d_mode = F.smooth_l1_loss(student, mode, reduction="none").reshape(-1)
    routed = alpha * d_full + (1 - alpha) * d_mode
    if routed.shape != student.shape:
        raise RuntimeError("Dual-teacher route broadcast unexpectedly.")
    return routed.mean(), d_full, d_mode, routed


def distribution_stats(values):
    x = np.asarray(values, dtype=float)
    if x.size == 0 or not np.isfinite(x).all():
        raise ValueError("Diagnostic values must be non-empty and finite.")
    return {"mean": float(x.mean()), "std": float(x.std()), "min": float(x.min()),
            "p10": float(np.quantile(x, .10)), "p25": float(np.quantile(x, .25)),
            "median": float(np.quantile(x, .50)), "p75": float(np.quantile(x, .75)),
            "p90": float(np.quantile(x, .90)), "p95": float(np.quantile(x, .95)),
            "max": float(x.max())}


def _student_predictions(student, text, audio, vision):
    result = {}
    for mode in MISSING_MODES:
        mask = mode_to_mask(mode, text.size(0), audio.device, audio.dtype)
        result[mode] = student(text, audio, vision, mask)["output_logit"].view(-1).detach().cpu().numpy()
    return result


def build_dual_teacher_suitability_audit(student, full_teacher, train_loader, device,
                                         compatibility_frame, full_teacher_checkpoint,
                                         mode_teacher_checkpoint, compatibility_config,
                                         root="result", dataset="mosi", seed=1111):
    """Build a shuffle=False, train-only, RNG/parameter-preserving suitability audit."""
    validate_train_cache(compatibility_frame)
    table = cache_table(compatibility_frame)
    before = {k: v.detach().cpu().clone() for k, v in student.state_dict().items()}
    records = []
    with preserve_rng_and_modes(student, full_teacher):
        student.eval(); full_teacher.eval()
        with torch.inference_mode():
            for batch in train_loader:
                text = batch["text"].to(device); audio = batch["audio"].to(device); vision = batch["vision"].to(device)
                labels = batch["labels"]["M"].view(-1).cpu().numpy()
                indices = batch["index"].view(-1).cpu().numpy().astype(int)
                ids = list(batch["id"])
                full = teacher_lav_prediction(full_teacher, text, audio, vision).view(-1).cpu().numpy()
                initial = _student_predictions(student, text, audio, vision)
                for pos, index in enumerate(indices):
                    if int(index) not in table:
                        raise KeyError("Audit sample missing from Stage 3 cache: {}".format(index))
                    source = table[int(index)]
                    base = {"sample_index": int(index), "sample_id": str(ids[pos]), "label": float(labels[pos]),
                            "full_teacher_LAV_pred": float(full[pos])}
                    if str(source["sample_id"]) != base["sample_id"]:
                        raise ValueError("sample_index/sample_id audit binding mismatch.")
                    for mode in MISSING_MODES:
                        mode_pred = float(source["evaluator_{}_pred".format(mode)])
                        compat = float(source["compat_{}".format(mode)])
                        s0 = float(initial[mode][pos]); label = base["label"]; tf = base["full_teacher_LAV_pred"]
                        base["mode_teacher_{}_pred".format(mode)] = mode_pred
                        base["compat_{}".format(mode)] = compat
                        base["full_mode_gap_{}".format(mode)] = abs(tf - mode_pred)
                        records.append({**base, "mode": mode, "compatibility": compat,
                                        "full_teacher_error": abs(tf-label), "mode_teacher_error": abs(mode_pred-label),
                                        "initial_student_full_gap": abs(s0-tf), "initial_student_mode_gap": abs(s0-mode_pred),
                                        "teacher_disagreement": abs(tf-mode_pred),
                                        "accuracy_advantage_mode": abs(tf-label)-abs(mode_pred-label),
                                        "attainability_advantage_mode": abs(s0-tf)-abs(s0-mode_pred),
                                        "initial_student_prediction": s0})
    if any(not torch.equal(before[k], v.detach().cpu()) for k, v in student.state_dict().items()):
        raise RuntimeError("Suitability audit changed Student parameters.")
    samples = pd.DataFrame(records).sort_values(["mode", "compatibility", "sample_index"], kind="mergesort")
    if len(samples) != 1284 * 3 or samples.sample_index.nunique() != 1284:
        raise RuntimeError("Suitability audit must cover 1284 unique train samples x three modes.")
    samples["compatibility_quartile"] = samples.groupby("mode", group_keys=False)["compatibility"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]))
    group_rows = []
    for (mode, quartile), local in samples.groupby(["mode", "compatibility_quartile"], observed=False):
        group_rows.append({"Mode": mode, "Quartile": str(quartile), "count": len(local),
                           "mean_compatibility": float(local.compatibility.mean()),
                           "mean_full_teacher_error": float(local.full_teacher_error.mean()),
                           "mean_mode_teacher_error": float(local.mode_teacher_error.mean()),
                           "fraction_mode_teacher_more_accurate": float((local.mode_teacher_error < local.full_teacher_error).mean()),
                           "mean_initial_student_full_gap": float(local.initial_student_full_gap.mean()),
                           "mean_initial_student_mode_gap": float(local.initial_student_mode_gap.mean()),
                           "fraction_mode_teacher_closer_to_student": float((local.initial_student_mode_gap < local.initial_student_full_gap).mean()),
                           "mean_teacher_disagreement": float(local.teacher_disagreement.mean()),
                           "mean_accuracy_advantage_mode": float(local.accuracy_advantage_mode.mean()),
                           "mean_attainability_advantage_mode": float(local.attainability_advantage_mode.mean())})
    quartiles = pd.DataFrame(group_rows)
    targets = compatibility_frame[["sample_index", "sample_id", "label"]].copy()
    full_by_index = samples.drop_duplicates("sample_index").set_index("sample_index")["full_teacher_LAV_pred"]
    targets["full_teacher_LAV_pred"] = targets.sample_index.map(full_by_index)
    for mode in MISSING_MODES:
        targets["mode_teacher_{}_pred".format(mode)] = compatibility_frame["evaluator_{}_pred".format(mode)]
        targets["compat_{}".format(mode)] = compatibility_frame["compat_{}".format(mode)]
        targets["full_mode_gap_{}".format(mode)] = np.abs(targets.full_teacher_LAV_pred-targets["mode_teacher_{}_pred".format(mode)])
    paths = dual_teacher_audit_paths(root, dataset, seed); paths["directory"].mkdir(parents=True, exist_ok=True)
    targets.to_csv(paths["targets"], index=False); samples.to_csv(paths["samples"], index=False); quartiles.to_csv(paths["quartiles"], index=False)
    summary = {"dataset": dataset, "seed": int(seed), "source_split": "train", "train_sample_count": 1284,
               "audit_row_count": int(len(samples)), "rng_state_preserved": True, "model_parameters_preserved": True,
               "full_teacher_checkpoint": str(full_teacher_checkpoint), "full_teacher_sha256": checkpoint_sha256(full_teacher_checkpoint),
               "mode_teacher_checkpoint": str(mode_teacher_checkpoint), "mode_teacher_sha256": checkpoint_sha256(mode_teacher_checkpoint),
               "compatibility_config_sha256": compatibility_config.get("config_sha256"),
               "quartiles": group_rows}
    paths["summary"].write_text(json.dumps(summary, indent=2, sort_keys=True)+"\n")
    manifest = dict(summary)
    manifest.update({"targets_sha256": checkpoint_sha256(paths["targets"]), "samples_sha256": checkpoint_sha256(paths["samples"]),
                     "quartiles_sha256": checkpoint_sha256(paths["quartiles"])})
    paths["manifest"].write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return paths, summary
