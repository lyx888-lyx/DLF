"""Figure-4-style cross-modal interaction visualization for CFCompat.

Why this script exists
----------------------
The DLF L<-V attention uses a future mask. On aligned MOSI, BERT tokens occupy
the beginning of the 50-step sequence, so plotting the *post-mask* attention
directly is structurally front-loaded and visually uninformative.

This script therefore uses a model-faithful post-hoc interaction diagnostic:
pairwise occlusion interaction between one text word and one visual time
window. For prediction f, text word i and visual window j,

    I_ij = | f(x_{-i,-j}) - f(x_{-i}) - f(x_{-j}) + f(x) |.

This finite-difference interaction is zero when the two perturbations act
additively and is larger when their joint effect is non-additive. No parameter
is trained or changed for this analysis.

The presentation follows the qualitative style of PMR Figure 4:
  * visual windows on top,
  * text words on the y-axis,
  * one heatmap for Uniform KD,
  * one heatmap for CFCompat,
  * shared color scale.

Automatic qualitative case selection is validation-only.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle
from matplotlib.colors import PowerNorm

from data_loader import MMDataset
from train_cf_compat_kd import build_config
from trains.singleTask.fixed_kd_utils import fixed_kd_checkpoint_path
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


DEFAULT_OUTPUT = Path("result/analysis/crossmodal_interaction_v1")


def parse_args():
    p = argparse.ArgumentParser(
        description="Figure-4-style text--vision interaction visualization."
    )
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--split", choices=("valid", "test"), default="valid")
    p.add_argument(
        "--condition",
        choices=("LV", "LAV"),
        default="LV",
        help="LV is recommended: shifted target view while text and vision remain observable.",
    )
    p.add_argument(
        "--baseline",
        choices=("fixedkd", "moddrop"),
        default="fixedkd",
        help="fixedkd corresponds to Uniform KD.",
    )
    p.add_argument("--sample-index", type=int, default=None)
    p.add_argument(
        "--selection",
        choices=("representative", "largest_gain"),
        default="representative",
    )
    p.add_argument("--min-abs-label", type=float, default=1.0)
    p.add_argument("--min-raw-words", type=int, default=10)
    p.add_argument("--max-raw-words", type=int, default=24)
    p.add_argument(
        "--min-visual-steps",
        type=int,
        default=10,
        help="Minimum number of non-padding visual steps for automatic case selection.",
    )
    p.add_argument("--max-words", type=int, default=16)
    p.add_argument("--visual-bins", type=int, default=10)
    p.add_argument(
        "--display-gamma",
        type=float,
        default=0.45,
        help="Shared PowerNorm gamma; <1 reveals weaker but non-zero interactions.",
    )
    p.add_argument("--occlusion-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--baseline-checkpoint")
    p.add_argument("--cfcompat-checkpoint")
    p.add_argument("--video-file")
    p.add_argument("--output-dir")
    return p.parse_args()


def _torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def _existing_checkpoint(candidates, label):
    checked = []
    for candidate in candidates:
        path = Path(candidate)
        checked.append(str(path))
        if path.is_file():
            return path
    raise FileNotFoundError(
        "{} checkpoint was not found. Checked:\n  {}".format(
            label, "\n  ".join(checked)
        )
    )


def resolve_baseline_checkpoint(cli):
    if cli.baseline_checkpoint:
        return _existing_checkpoint([cli.baseline_checkpoint], "baseline")

    root = Path(cli.model_save_dir)
    if cli.baseline == "fixedkd":
        candidates = [
            fixed_kd_checkpoint_path(root, cli.dataset, cli.seed),
            root / "missing_baseline" / "fixed_kd" / "train"
            / "DLF_{}_seed{}_best.pth".format(cli.dataset, cli.seed),
        ]
    else:
        candidates = [
            root / "missing_baseline" / "moddrop_benchmark_multiseed_v1"
            / "seed{}".format(cli.seed)
            / "DLF_{}_seed{}_best_valid.pth".format(cli.dataset, cli.seed),
            root / "missing_baseline" / "moddrop"
            / "DLF_{}_seed{}_best.pth".format(cli.dataset, cli.seed),
        ]
    return _existing_checkpoint(candidates, cli.baseline)


def resolve_cfcompat_checkpoint(cli):
    if cli.cfcompat_checkpoint:
        return _existing_checkpoint([cli.cfcompat_checkpoint], "CFCompat")

    root = Path(cli.model_save_dir)
    candidates = [
        root / "missing_baseline" / "cf_compat_kd_v1" / "benchmark_multiseed"
        / "seed{}".format(cli.seed)
        / "DLF_{}_seed{}_best_valid.pth".format(cli.dataset, cli.seed),
        root / "missing_baseline" / "cf_compat_kd_v1"
        / "DLF_{}_seed{}_best_valid.pth".format(cli.dataset, cli.seed),
    ]
    return _existing_checkpoint(candidates, "CFCompat")


def build_models(cli):
    cfg_cli = SimpleNamespace(
        dataset=cli.dataset,
        config_file=cli.config_file,
        gpu_ids=cli.gpu_ids,
    )
    cfg = build_config(cfg_cli, cli.seed)
    setup_seed(cli.seed)

    baseline_checkpoint = resolve_baseline_checkpoint(cli)
    cfcompat_checkpoint = resolve_cfcompat_checkpoint(cli)

    def make_model(checkpoint):
        backbone = DLF(cfg).to(cfg.device)
        model = MissingModalityWrapper(
            backbone,
            cfg.feature_dims[1],
            cfg.feature_dims[2],
        ).to(cfg.device)
        model.load_state_dict(
            _torch_load(checkpoint, cfg.device),
            strict=True,
        )
        model.eval()
        return model

    return (
        cfg,
        make_model(baseline_checkpoint),
        make_model(cfcompat_checkpoint),
        baseline_checkpoint,
        cfcompat_checkpoint,
    )


def batch_to_device(batch, device):
    return (
        batch["text"].to(device),
        batch["audio"].to(device),
        batch["vision"].to(device),
        batch["labels"]["M"].to(device).view(-1, 1),
    )


@torch.no_grad()
def predict_condition(model, text, audio, vision, condition):
    mask = mode_to_mask(
        condition,
        batch_size=text.size(0),
        device=text.device,
        dtype=audio.dtype,
    )
    output = model(text, audio, vision, mask)
    return output["output_logit"].view(-1)


def scan_candidates(cli, cfg, dataset, baseline, cfcompat):
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cli.num_workers,
        drop_last=False,
    )
    rows = []
    for batch in loader:
        text, audio, vision, labels = batch_to_device(batch, cfg.device)
        base_pred = predict_condition(
            baseline, text, audio, vision, cli.condition
        ).cpu().numpy()
        ours_pred = predict_condition(
            cfcompat, text, audio, vision, cli.condition
        ).cpu().numpy()
        labels_np = labels.view(-1).cpu().numpy()
        indices = batch["index"].view(-1).cpu().numpy().astype(int)
        raw_texts = list(batch["raw_text"])
        ids = list(batch["id"])
        visual_active = (
            torch.linalg.vector_norm(vision.detach(), dim=-1) > 1e-8
        ).sum(dim=1).cpu().numpy().astype(int)

        for offset, index in enumerate(indices):
            truth = float(labels_np[offset])
            bp = float(base_pred[offset])
            op = float(ours_pred[offset])
            be = abs(bp - truth)
            oe = abs(op - truth)
            raw = str(raw_texts[offset])
            rows.append({
                "sample_index": int(index),
                "sample_id": str(ids[offset]),
                "raw_text": raw,
                "label": truth,
                "baseline_pred": bp,
                "cfcompat_pred": op,
                "baseline_abs_error": be,
                "cfcompat_abs_error": oe,
                "error_gain": be - oe,
                "abs_label": abs(truth),
                "raw_word_count": len(raw.strip().split()),
                "visual_active_steps": int(visual_active[offset]),
            })

    return pd.DataFrame(rows).sort_values(
        "sample_index", kind="mergesort"
    ).reset_index(drop=True)


def select_sample(cli, candidates):
    if cli.sample_index is not None:
        hit = candidates.loc[
            candidates.sample_index.astype(int) == int(cli.sample_index)
        ]
        if len(hit) != 1:
            raise ValueError(
                "--sample-index={} was not found uniquely.".format(
                    cli.sample_index
                )
            )
        return hit.iloc[0], "manual_sample_index"

    eligible = candidates.loc[
        (candidates.error_gain > 0)
        & (candidates.abs_label >= float(cli.min_abs_label))
        & (candidates.raw_word_count >= int(cli.min_raw_words))
        & (candidates.raw_word_count <= int(cli.max_raw_words))
        & (candidates.visual_active_steps >= int(cli.min_visual_steps))
    ].copy()

    if eligible.empty:
        eligible = candidates.loc[
            (candidates.error_gain > 0)
            & (candidates.raw_word_count >= 6)
            & (candidates.visual_active_steps >= 6)
        ].copy()
    if eligible.empty:
        eligible = candidates.copy()

    if cli.selection == "largest_gain":
        row = eligible.sort_values(
            ["error_gain", "visual_active_steps", "raw_word_count"],
            ascending=[False, False, False],
            kind="mergesort",
        ).iloc[0]
        return row, "largest_positive_error_gain"

    positive = eligible.loc[eligible.error_gain > 0].copy()
    if positive.empty:
        positive = eligible.copy()

    # Keep the gain itself representative (middle 50%), then prefer a case
    # with richer temporal support. This avoids both cherry-picking the largest
    # gain and selecting a visually degenerate padded utterance.
    q1 = float(positive.error_gain.quantile(0.25))
    q3 = float(positive.error_gain.quantile(0.75))
    middle = positive.loc[
        (positive.error_gain >= q1) & (positive.error_gain <= q3)
    ].copy()
    if middle.empty:
        middle = positive.copy()

    median_gain = float(positive.error_gain.median())
    middle["distance_to_median_gain"] = np.abs(
        middle.error_gain - median_gain
    )
    row = middle.sort_values(
        [
            "visual_active_steps",
            "raw_word_count",
            "distance_to_median_gain",
            "abs_label",
            "sample_index",
        ],
        ascending=[False, False, True, False, True],
        kind="mergesort",
    ).iloc[0]
    return row, "representative_middle_gain_with_temporal_coverage"


def bert_word_groups(model, text_tensor):
    """Merge BERT WordPiece tokens into human-readable word groups."""
    tokenizer = model.backbone.text_model.get_tokenizer()
    array = text_tensor.detach().cpu().numpy()
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError(
            "Expected BERT tensor [3,T], got {}".format(array.shape)
        )

    ids = array[0].astype(np.int64)
    mask = array[1].astype(np.float64)
    tokens = tokenizer.convert_ids_to_tokens(ids.tolist())
    special = set(getattr(tokenizer, "all_special_tokens", []))

    groups = []
    labels = []
    for pos, (token, valid) in enumerate(zip(tokens, mask)):
        token = str(token)
        if valid <= 0 or token in special:
            continue

        if token.startswith("##") and groups:
            labels[-1] += token[2:]
            groups[-1].append(int(pos))
        else:
            groups.append([int(pos)])
            labels.append(token.replace("Ġ", ""))

    if not groups:
        raise RuntimeError("No non-special BERT word groups were found.")
    return groups, labels, int(tokenizer.mask_token_id)


def active_visual_positions(vision_tensor, eps=1e-8):
    """Return non-padding visual positions from the aligned feature sequence."""
    vision = vision_tensor.detach().cpu().numpy()
    if vision.ndim != 2:
        raise ValueError(
            "Expected visual tensor [T,D], got {}".format(vision.shape)
        )
    norms = np.linalg.norm(vision, axis=1)
    active = np.flatnonzero(norms > float(eps)).astype(np.int64)
    if active.size == 0:
        active = np.arange(vision.shape[0], dtype=np.int64)
    return active


def visual_windows(vision_tensor, count):
    """Partition only the non-padding visual support into display windows."""
    active = active_visual_positions(vision_tensor)
    count = min(int(count), int(len(active)))
    chunks = np.array_split(active, count)

    windows = []
    labels = []
    centers = []
    for chunk in chunks:
        chunk = np.asarray(chunk, dtype=np.int64)
        windows.append(chunk)
        centers.append(int(round(float(chunk.mean()))))

        start = int(chunk[0]) + 1
        stop = int(chunk[-1]) + 1
        if len(chunk) == 1:
            labels.append("V{}".format(start))
        elif np.all(np.diff(chunk) == 1):
            labels.append("V{}–{}".format(start, stop))
        else:
            labels.append("V{}".format(int(round(float(chunk.mean()))) + 1))

    return windows, labels, np.asarray(centers, dtype=np.int64), active


def build_variant_specs(word_count, visual_count):
    specs = [("base", None, None)]
    specs.extend(("text", i, None) for i in range(word_count))
    specs.extend(("vision", None, j) for j in range(visual_count))
    specs.extend(
        ("pair", i, j)
        for i in range(word_count)
        for j in range(visual_count)
    )
    return specs


@torch.no_grad()
def run_occlusion_variants(
    model,
    sample,
    condition,
    word_groups,
    mask_token_id,
    windows,
    batch_size,
    device,
):
    specs = build_variant_specs(len(word_groups), len(windows))
    predictions = {}

    base_text = sample["text"].detach().cpu()
    base_audio = sample["audio"].detach().cpu()
    base_vision = sample["vision"].detach().cpu()

    for begin in range(0, len(specs), int(batch_size)):
        chunk = specs[begin: begin + int(batch_size)]
        n = len(chunk)

        text = base_text.unsqueeze(0).repeat(n, 1, 1)
        audio = base_audio.unsqueeze(0).repeat(n, 1, 1)
        vision = base_vision.unsqueeze(0).repeat(n, 1, 1)

        for row, (_, word_index, visual_index) in enumerate(chunk):
            if word_index is not None:
                positions = word_groups[int(word_index)]
                for pos in positions:
                    text[row, 0, int(pos)] = int(mask_token_id)
                    # Keep the token visible to BERT; only replace its identity.
                    text[row, 1, int(pos)] = 1.0
            if visual_index is not None:
                positions = windows[int(visual_index)]
                vision[row, positions, :] = 0.0

        pred = predict_condition(
            model,
            text.to(device),
            audio.to(device),
            vision.to(device),
            condition,
        ).detach().cpu().numpy()

        for local, spec in enumerate(chunk):
            predictions[spec] = float(pred[local])

    return predictions


def interaction_from_predictions(predictions, word_count, visual_count):
    base = predictions[("base", None, None)]

    text_effect = np.asarray([
        abs(base - predictions[("text", i, None)])
        for i in range(word_count)
    ], dtype=np.float64)

    visual_effect = np.asarray([
        abs(base - predictions[("vision", None, j)])
        for j in range(visual_count)
    ], dtype=np.float64)

    matrix = np.zeros((word_count, visual_count), dtype=np.float64)
    signed = np.zeros_like(matrix)
    for i in range(word_count):
        text_only = predictions[("text", i, None)]
        for j in range(visual_count):
            vision_only = predictions[("vision", None, j)]
            both = predictions[("pair", i, j)]
            value = both - text_only - vision_only + base
            signed[i, j] = value
            matrix[i, j] = abs(value)

    return {
        "base_prediction": float(base),
        "text_effect": text_effect,
        "visual_effect": visual_effect,
        "interaction": matrix,
        "signed_interaction": signed,
    }


def choose_display_words(
    labels,
    baseline_pack,
    cfcompat_pack,
    max_words,
):
    n = len(labels)
    if n <= int(max_words):
        return np.arange(n, dtype=np.int64)

    # Selection is method-symmetric: average marginal text influence.
    score = 0.5 * (
        baseline_pack["text_effect"] + cfcompat_pack["text_effect"]
    )
    chosen = np.argsort(-score, kind="mergesort")[: int(max_words)]
    return np.sort(chosen.astype(np.int64))


def load_video_frames(video_path, centers, input_length):
    if video_path is None:
        return None
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "--video-file requires opencv-python."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(path))

    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for center in centers:
        frac = (float(center) + 0.5) / float(input_length)
        index = int(round(frac * max(count - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            frames.append(None)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    cap.release()
    return frames


def draw_visual_header(
    ax,
    window_labels,
    importance,
    frames=None,
):
    n = len(window_labels)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    importance = np.asarray(importance, dtype=np.float64)
    if importance.size and importance.max() > 0:
        importance = importance / importance.max()

    cmap = plt.get_cmap("viridis")

    for j, label in enumerate(window_labels):
        strength = float(importance[j])
        edge = cmap(0.20 + 0.75 * strength)
        fill = cmap(0.08 + 0.72 * strength)
        if frames is not None and frames[j] is not None:
            ax.imshow(
                frames[j],
                extent=(j + 0.05, j + 0.95, 0.14, 0.90),
                aspect="auto",
                interpolation="bilinear",
            )
            rect = Rectangle(
                (j + 0.05, 0.14),
                0.90,
                0.76,
                facecolor="none",
                edgecolor=edge,
                linewidth=1.8,
            )
            ax.add_patch(rect)
        else:
            rect = Rectangle(
                (j + 0.05, 0.14),
                0.90,
                0.76,
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.6,
            )
            ax.add_patch(rect)
            text_color = "white" if strength > 0.58 else "#222222"
            ax.text(
                j + 0.5,
                0.52,
                label,
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
                fontweight="semibold" if strength > 0.58 else "normal",
            )

    ax.text(
        -0.12,
        0.52,
        "Visual\nwindows",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="semibold",
    )


def plot_figure(
    cli,
    selected,
    sample,
    labels,
    windows,
    window_labels,
    centers,
    baseline_pack,
    cfcompat_pack,
    output_dir,
):
    display_rows = choose_display_words(
        labels,
        baseline_pack,
        cfcompat_pack,
        cli.max_words,
    )
    display_labels = [labels[i] for i in display_rows]

    base = baseline_pack["interaction"][display_rows]
    ours = cfcompat_pack["interaction"][display_rows]

    combined = np.concatenate([base.ravel(), ours.ravel()])
    positive = combined[combined > 0]
    if positive.size:
        vmax = float(np.quantile(positive, 0.97))
        vmax = max(vmax, float(positive.max()) * 0.20, 1e-8)
    else:
        vmax = 1.0

    # Normalize both methods with ONE shared raw scale. PowerNorm is then used
    # only as a display transform to reveal weak-but-nonzero structure.
    base_disp = np.clip(base / vmax, 0.0, 1.0)
    ours_disp = np.clip(ours / vmax, 0.0, 1.0)
    display_norm = PowerNorm(
        gamma=float(cli.display_gamma),
        vmin=0.0,
        vmax=1.0,
    )

    visual_importance = 0.5 * (
        baseline_pack["visual_effect"] + cfcompat_pack["visual_effect"]
    )
    frames = load_video_frames(
        cli.video_file,
        centers,
        sample["vision"].shape[0],
    )

    truth = float(selected["label"])
    bp = float(baseline_pack["base_prediction"])
    op = float(cfcompat_pack["base_prediction"])
    be = abs(bp - truth)
    oe = abs(op - truth)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.7,
        "ytick.labelsize": 10.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(10.4, 6.8))
    gs = gridspec.GridSpec(
        3,
        2,
        width_ratios=[1.55, 6.00],
        height_ratios=[0.78, 2.55, 2.55],
        hspace=0.18,
        wspace=0.06,
    )

    ax_header_label = fig.add_subplot(gs[0, 0])
    ax_header = fig.add_subplot(gs[0, 1])
    ax_left1 = fig.add_subplot(gs[1, 0])
    ax_map1 = fig.add_subplot(gs[1, 1])
    ax_left2 = fig.add_subplot(gs[2, 0])
    ax_map2 = fig.add_subplot(gs[2, 1])

    ax_header_label.axis("off")
    ax_left1.axis("off")
    ax_left2.axis("off")

    draw_visual_header(
        ax_header,
        window_labels,
        visual_importance,
        frames,
    )

    im1 = ax_map1.imshow(
        base_disp,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=display_norm,
    )
    im2 = ax_map2.imshow(
        ours_disp,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        norm=display_norm,
    )

    for panel, show_x in ((ax_map1, False), (ax_map2, True)):
        panel.set_yticks(np.arange(len(display_labels)))
        panel.set_yticklabels(display_labels)
        panel.tick_params(axis="y", length=0, pad=5)
        panel.set_xticks(np.arange(len(window_labels)))
        panel.set_xticklabels(
            window_labels if show_x else [""] * len(window_labels),
            rotation=0,
        )
        panel.tick_params(axis="x", length=0 if not show_x else 3)
        if show_x:
            panel.set_xlabel("Visual time window")
        for spine in panel.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

    # Keep token labels neutral by default. PMR manually marks emotion words;
    # we avoid automatic red highlighting because strong interaction is not
    # equivalent to sentiment-bearing semantics.
    baseline_name = "Uniform KD" if cli.baseline == "fixedkd" else "DLF-ModDrop"
    ax_left1.text(
        0.98, 0.64,
        "(a) {}".format(baseline_name),
        ha="right", va="center",
        fontsize=12.2, fontweight="bold",
    )
    ax_left1.text(
        0.98, 0.43,
        "Pred {:.2f}\nAE {:.2f}".format(bp, be),
        ha="right", va="center",
        fontsize=10.2,
    )

    ax_left2.text(
        0.98, 0.64,
        "(b) CFCompat",
        ha="right", va="center",
        fontsize=12.2, fontweight="bold",
    )
    ax_left2.text(
        0.98, 0.43,
        "Pred {:.2f}\nAE {:.2f}".format(op, oe),
        ha="right", va="center",
        fontsize=10.2,
    )

    raw = str(selected["raw_text"]).strip()
    if len(raw) > 125:
        raw = raw[:122] + "..."

    fig.suptitle(
        "Cross-modal Interaction on CMU-MOSI ({})\n"
        "Truth = {:.2f}   |   {}".format(cli.condition, truth, raw),
        fontsize=12.8,
        y=0.985,
    )

    cax = fig.add_axes([0.925, 0.205, 0.014, 0.58])
    cb = fig.colorbar(im2, cax=cax)
    cb.set_label("Relative interaction strength", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    output_dir.mkdir(parents=True, exist_ok=True)
    stem = "{}_seed{}_{}_{}_idx{}".format(
        cli.dataset,
        cli.seed,
        cli.split,
        cli.condition,
        int(selected["sample_index"]),
    )
    png = output_dir / "{}.png".format(stem)
    pdf = output_dir / "{}.pdf".format(stem)
    npz = output_dir / "{}_interaction.npz".format(stem)

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        npz,
        baseline_interaction_raw=baseline_pack["interaction"],
        cfcompat_interaction_raw=cfcompat_pack["interaction"],
        baseline_signed_interaction=baseline_pack["signed_interaction"],
        cfcompat_signed_interaction=cfcompat_pack["signed_interaction"],
        baseline_text_effect=baseline_pack["text_effect"],
        cfcompat_text_effect=cfcompat_pack["text_effect"],
        baseline_visual_effect=baseline_pack["visual_effect"],
        cfcompat_visual_effect=cfcompat_pack["visual_effect"],
        displayed_word_indices=display_rows,
        word_labels=np.asarray(labels, dtype=object),
        visual_window_labels=np.asarray(window_labels, dtype=object),
        visual_centers=centers,
        shared_display_vmax_raw=np.asarray([vmax], dtype=np.float64),
        display_gamma=np.asarray([float(cli.display_gamma)], dtype=np.float64),
    )

    return png, pdf, npz, {
        "truth": truth,
        "baseline_prediction": bp,
        "cfcompat_prediction": op,
        "baseline_abs_error": be,
        "cfcompat_abs_error": oe,
        "displayed_words": display_labels,
        "visual_window_labels": window_labels,
        "shared_display_vmax_raw": vmax,
    }


def main():
    cli = parse_args()

    if cli.split == "test" and cli.sample_index is None:
        raise ValueError(
            "Automatic case selection is validation-only. "
            "For --split test, explicitly provide --sample-index."
        )
    if cli.condition not in ("LV", "LAV"):
        raise ValueError(
            "Text--vision interaction visualization requires vision to be present."
        )

    cfg, baseline, cfcompat, baseline_ckpt, cfcompat_ckpt = build_models(cli)
    dataset = MMDataset(cfg, mode=cli.split)

    candidates = scan_candidates(
        cli,
        cfg,
        dataset,
        baseline,
        cfcompat,
    )
    selected, selection_reason = select_sample(cli, candidates)
    sample_index = int(selected["sample_index"])
    sample = dataset[sample_index]

    word_groups, word_labels, mask_token_id = bert_word_groups(
        baseline,
        sample["text"],
    )
    windows, window_labels, centers, active_visual = visual_windows(
        sample["vision"],
        cli.visual_bins,
    )

    print("Computing Uniform-KD/ModDrop occlusion interaction ...")
    baseline_predictions = run_occlusion_variants(
        baseline,
        sample,
        cli.condition,
        word_groups,
        mask_token_id,
        windows,
        cli.occlusion_batch_size,
        cfg.device,
    )
    baseline_pack = interaction_from_predictions(
        baseline_predictions,
        len(word_groups),
        len(windows),
    )

    print("Computing CFCompat occlusion interaction ...")
    cfcompat_predictions = run_occlusion_variants(
        cfcompat,
        sample,
        cli.condition,
        word_groups,
        mask_token_id,
        windows,
        cli.occlusion_batch_size,
        cfg.device,
    )
    cfcompat_pack = interaction_from_predictions(
        cfcompat_predictions,
        len(word_groups),
        len(windows),
    )

    output_dir = (
        Path(cli.output_dir)
        if cli.output_dir
        else DEFAULT_OUTPUT / cli.dataset / "seed{}".format(cli.seed)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = candidates.copy()
    candidates["selected"] = (
        candidates.sample_index.astype(int) == sample_index
    )
    candidates["selection_reason"] = np.where(
        candidates["selected"],
        selection_reason,
        "",
    )
    candidate_csv = output_dir / "{}_seed{}_{}_candidate_summary.csv".format(
        cli.dataset,
        cli.seed,
        cli.split,
    )
    candidates.to_csv(candidate_csv, index=False)

    png, pdf, npz, plot_meta = plot_figure(
        cli,
        selected,
        sample,
        word_labels,
        windows,
        window_labels,
        centers,
        baseline_pack,
        cfcompat_pack,
        output_dir,
    )

    metadata = {
        "dataset": cli.dataset,
        "seed": int(cli.seed),
        "split": cli.split,
        "condition": cli.condition,
        "baseline": cli.baseline,
        "selection": cli.selection,
        "selection_reason": selection_reason,
        "sample_index": sample_index,
        "sample_id": str(selected["sample_id"]),
        "raw_text": str(selected["raw_text"]),
        "baseline_checkpoint": str(baseline_ckpt),
        "cfcompat_checkpoint": str(cfcompat_ckpt),
        "diagnostic": "pairwise_text_visual_occlusion_interaction",
        "formula": "abs(f(x_-i,-j)-f(x_-i)-f(x_-j)+f(x))",
        "text_perturbation": "replace all WordPiece pieces of one word by [MASK]",
        "vision_perturbation": "zero one contiguous visual time window",
        "visual_active_positions": [int(v) for v in active_visual.tolist()],
        "visual_active_steps": int(len(active_visual)),
        "display_gamma": float(cli.display_gamma),
        "note": (
            "Padding-only visual steps are excluded before binning. No model "
            "parameter is changed. Both heatmaps use one shared raw scale and "
            "the same monotonic PowerNorm display transform. Raw interaction "
            "values are saved in NPZ."
        ),
        **plot_meta,
    }
    meta_path = output_dir / "{}_seed{}_{}_{}_idx{}_metadata.json".format(
        cli.dataset,
        cli.seed,
        cli.split,
        cli.condition,
        sample_index,
    )
    meta_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=" * 88)
    print("Figure-4-style cross-modal interaction visualization complete.")
    print("baseline checkpoint : {}".format(baseline_ckpt))
    print("CFCompat checkpoint : {}".format(cfcompat_ckpt))
    print("split / condition   : {} / {}".format(cli.split, cli.condition))
    print("selection            : {} ({})".format(cli.selection, selection_reason))
    print("sample_index         : {}".format(sample_index))
    print("sample_id            : {}".format(selected["sample_id"]))
    print("truth                : {:.6f}".format(float(selected["label"])))
    print("baseline pred / AE   : {:.6f} / {:.6f}".format(
        float(baseline_pack["base_prediction"]),
        abs(float(baseline_pack["base_prediction"]) - float(selected["label"])),
    ))
    print("CFCompat pred / AE   : {:.6f} / {:.6f}".format(
        float(cfcompat_pack["base_prediction"]),
        abs(float(cfcompat_pack["base_prediction"]) - float(selected["label"])),
    ))
    print("visual active steps  : {}".format(len(active_visual)))
    print("visual windows       : {}".format(", ".join(window_labels)))
    print("raw text             : {}".format(selected["raw_text"]))
    print("-" * 88)
    print("PNG                  : {}".format(png))
    print("PDF                  : {}".format(pdf))
    print("NPZ                  : {}".format(npz))
    print("candidates           : {}".format(candidate_csv))
    print("metadata             : {}".format(meta_path))
    print("=" * 88)


if __name__ == "__main__":
    main()
