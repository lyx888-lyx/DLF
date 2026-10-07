"""Figure-4-style text--vision attention visualization for CFCompat.

This script uses the *actual* DLF cross-modal attention weights from the last
L<-V Transformer layer. It compares Uniform KD (FixedKD) against CFCompat on
the same MOSI sample and the same target condition.

Default protocol:
  * dataset: MOSI
  * split: validation (avoids test-driven case selection)
  * target condition: LV (text + vision are present, audio is shifted/missing)
  * baseline: FixedKD / Uniform KD
  * comparison: CFCompat
  * sample selection: a representative positive-improvement validation sample,
    not the maximum-improvement sample.

Outputs:
  * publication-style PNG/PDF
  * candidate CSV with transparent sample-selection diagnostics
  * NPZ containing the raw attention matrices
  * JSON metadata

Optional:
  --video-file /path/to/exact_utterance_clip.mp4
      If an utterance-level clip is available, sampled video frames are shown
      above the attention heatmaps. Otherwise visual-window placeholders are
      rendered so the figure remains reproducible from aligned_50.pkl alone.
"""

import argparse
import json
import math
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

from data_loader import MMDataset
from train_cf_compat_kd import build_config
from trains.singleTask.fixed_kd_utils import fixed_kd_checkpoint_path
from trains.singleTask.missing_utils import (
    MissingModalityWrapper,
    mode_to_mask,
)
from trains.singleTask.model.DLF import DLF
from utils.functions import setup_seed


DEFAULT_OUTPUT = Path("result/analysis/crossmodal_attention_v1")


def parse_args():
    p = argparse.ArgumentParser(
        description="Figure-4-style DLF cross-modal attention visualization."
    )
    p.add_argument("--dataset", choices=("mosi",), default="mosi")
    p.add_argument("--seed", type=int, default=1114)
    p.add_argument("--split", choices=("valid", "test"), default="valid")
    p.add_argument(
        "--condition",
        choices=("LV", "LAV"),
        default="LV",
        help="Use LV for the shifted target view while keeping vision observable.",
    )
    p.add_argument(
        "--baseline",
        choices=("fixedkd", "moddrop"),
        default="fixedkd",
        help="fixedkd gives the cleanest Uniform-KD-vs-CFCompat mechanism comparison.",
    )
    p.add_argument("--sample-index", type=int, default=None)
    p.add_argument(
        "--selection",
        choices=("representative", "largest_gain"),
        default="representative",
        help="representative selects the median positive gain after simple quality filters.",
    )
    p.add_argument("--min-abs-label", type=float, default=1.0)
    p.add_argument("--max-raw-words", type=int, default=22)
    p.add_argument("--max-tokens", type=int, default=18)
    p.add_argument("--visual-bins", type=int, default=10)
    p.add_argument("--layer-index", type=int, default=-1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--result-root", default="result")
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
    for path in candidates:
        path = Path(path)
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

    baseline_ckpt = resolve_baseline_checkpoint(cli)
    cfcompat_ckpt = resolve_cfcompat_checkpoint(cli)

    def make_model(checkpoint):
        backbone = DLF(cfg).to(cfg.device)
        model = MissingModalityWrapper(
            backbone,
            cfg.feature_dims[1],
            cfg.feature_dims[2],
        ).to(cfg.device)
        state = _torch_load(checkpoint, cfg.device)
        model.load_state_dict(state, strict=True)
        model.eval()
        return model

    baseline = make_model(baseline_ckpt)
    cfcompat = make_model(cfcompat_ckpt)
    return cfg, baseline, cfcompat, baseline_ckpt, cfcompat_ckpt


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
    return model(text, audio, vision, mask)["output_logit"].view(-1)


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
        ).detach().cpu().numpy()
        ours_pred = predict_condition(
            cfcompat, text, audio, vision, cli.condition
        ).detach().cpu().numpy()
        labels_np = labels.view(-1).detach().cpu().numpy()
        indices = batch["index"].view(-1).cpu().numpy().astype(int)
        raw_texts = list(batch["raw_text"])
        ids = list(batch["id"])

        for off, idx in enumerate(indices):
            truth = float(labels_np[off])
            bp = float(base_pred[off])
            op = float(ours_pred[off])
            base_err = abs(bp - truth)
            ours_err = abs(op - truth)
            raw = str(raw_texts[off])
            word_count = len(raw.strip().split())
            rows.append({
                "sample_index": int(idx),
                "sample_id": str(ids[off]),
                "raw_text": raw,
                "label": truth,
                "baseline_pred": bp,
                "cfcompat_pred": op,
                "baseline_abs_error": base_err,
                "cfcompat_abs_error": ours_err,
                "error_gain": base_err - ours_err,
                "abs_label": abs(truth),
                "raw_word_count": word_count,
            })
    return pd.DataFrame(rows).sort_values("sample_index", kind="mergesort").reset_index(drop=True)


def select_sample(cli, candidates):
    if cli.sample_index is not None:
        hit = candidates.loc[candidates.sample_index == int(cli.sample_index)]
        if len(hit) != 1:
            raise ValueError(
                "--sample-index={} was not found uniquely in {} split.".format(
                    cli.sample_index, cli.split
                )
            )
        return hit.iloc[0], "manual_sample_index"

    eligible = candidates.loc[
        (candidates.error_gain > 0)
        & (candidates.abs_label >= float(cli.min_abs_label))
        & (candidates.raw_word_count >= 4)
        & (candidates.raw_word_count <= int(cli.max_raw_words))
    ].copy()

    if eligible.empty:
        eligible = candidates.loc[candidates.error_gain > 0].copy()
    if eligible.empty:
        eligible = candidates.copy()

    if cli.selection == "largest_gain":
        row = eligible.sort_values(
            ["error_gain", "abs_label"],
            ascending=[False, False],
            kind="mergesort",
        ).iloc[0]
        return row, "largest_positive_error_gain"

    positive = eligible.loc[eligible.error_gain > 0].copy()
    if positive.empty:
        row = eligible.iloc[len(eligible) // 2]
        return row, "fallback_middle_candidate"

    median_gain = float(positive.error_gain.median())
    positive["distance_to_median_gain"] = np.abs(
        positive.error_gain - median_gain
    )
    row = positive.sort_values(
        ["distance_to_median_gain", "abs_label", "sample_index"],
        ascending=[True, False, True],
        kind="mergesort",
    ).iloc[0]
    return row, "representative_median_positive_gain"


def capture_text_to_vision_attention(
    model,
    sample,
    cfg,
    condition,
    layer_index,
):
    if not hasattr(model, "backbone"):
        raise ValueError("Expected MissingModalityWrapper with .backbone.")
    layers = model.backbone.trans_l_with_v.layers
    if not (-len(layers) <= layer_index < len(layers)):
        raise IndexError(
            "layer_index={} invalid for {} L<-V layers.".format(
                layer_index, len(layers)
            )
        )
    attention_module = layers[layer_index].self_attn
    capture = {}

    def hook(_module, _inputs, output):
        if not isinstance(output, (tuple, list)) or len(output) != 2:
            raise RuntimeError("Unexpected MultiheadAttention output.")
        capture["weights"] = output[1].detach().cpu()

    handle = attention_module.register_forward_hook(hook)
    try:
        text = sample["text"].unsqueeze(0).to(cfg.device)
        audio = sample["audio"].unsqueeze(0).to(cfg.device)
        vision = sample["vision"].unsqueeze(0).to(cfg.device)
        labels = sample["labels"]["M"].view(-1).to(cfg.device)
        with torch.no_grad():
            pred = predict_condition(
                model, text, audio, vision, condition
            )
    finally:
        handle.remove()

    if "weights" not in capture:
        raise RuntimeError("L<-V attention hook did not capture any weights.")
    weights = capture["weights"]
    if weights.ndim != 3 or weights.size(0) != 1:
        raise RuntimeError(
            "Expected attention [1,T_text,T_visual], got {}".format(
                tuple(weights.shape)
            )
        )
    return {
        "prediction": float(pred[0].item()),
        "label": float(labels[0].item()),
        "attention": weights[0].numpy().astype(np.float64),
        "text_input": sample["text"].detach().cpu().numpy(),
        "vision_input": sample["vision"].detach().cpu().numpy(),
        "raw_text": str(sample["raw_text"]),
        "sample_id": str(sample["id"]),
        "sample_index": int(sample["index"]),
    }


def _conv_centers(length_after_conv, kernel_size, input_length):
    offset = (int(kernel_size) - 1) // 2
    centers = np.arange(length_after_conv, dtype=np.int64) + offset
    return np.clip(centers, 0, int(input_length) - 1)


def build_text_rows(pack, model, cfg, max_tokens):
    attn = pack["attention"]
    text_input = pack["text_input"]

    if text_input.ndim != 2 or text_input.shape[0] < 2:
        raise ValueError(
            "Expected BERT input with shape [3,T], got {}".format(
                text_input.shape
            )
        )
    input_ids = text_input[0].astype(np.int64)
    input_mask = text_input[1].astype(np.float64)
    tokenizer = model.backbone.text_model.get_tokenizer()
    tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

    q_len = attn.shape[0]
    centers = _conv_centers(
        q_len,
        cfg.conv1d_kernel_size_l,
        len(input_ids),
    )

    special = set(getattr(tokenizer, "all_special_tokens", []))
    valid_rows = []
    labels = []
    for q, center in enumerate(centers):
        token = str(tokens[int(center)])
        if input_mask[int(center)] <= 0:
            continue
        if token in special:
            continue
        valid_rows.append(q)
        labels.append(token)

    if not valid_rows:
        valid_rows = list(range(q_len))
        labels = ["T{}".format(i + 1) for i in valid_rows]

    # If a sample is long, keep the most visually informative query positions
    # but preserve their original order in the final figure.
    if len(valid_rows) > int(max_tokens):
        row_strength = np.asarray(
            [np.max(attn[q]) for q in valid_rows],
            dtype=np.float64,
        )
        chosen_local = np.argsort(-row_strength, kind="mergesort")[
            : int(max_tokens)
        ]
        chosen_local = np.sort(chosen_local)
        valid_rows = [valid_rows[i] for i in chosen_local]
        labels = [labels[i] for i in chosen_local]

    return np.asarray(valid_rows, dtype=np.int64), labels


def aggregate_visual_bins(attn, cfg, visual_bins, vision_input_length):
    src_len = attn.shape[1]
    centers = _conv_centers(
        src_len,
        cfg.conv1d_kernel_size_v,
        vision_input_length,
    )
    visual_bins = min(int(visual_bins), src_len)
    chunks = np.array_split(np.arange(src_len), visual_bins)
    agg = np.stack(
        [attn[:, chunk].sum(axis=1) for chunk in chunks],
        axis=1,
    )
    bin_centers = np.asarray(
        [int(round(float(centers[chunk].mean()))) for chunk in chunks],
        dtype=np.int64,
    )
    return agg, bin_centers, chunks


def normalize_rows_for_display(matrix):
    matrix = np.asarray(matrix, dtype=np.float64)
    denominator = matrix.sum(axis=1, keepdims=True)
    denominator[denominator <= 0] = 1.0
    return matrix / denominator


def load_video_frames(video_path, fractions):
    if video_path is None:
        return None
    path = Path(video_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "--video-file requires opencv-python (cv2)."
        ) from exc

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError("Could not open video: {}".format(path))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for frac in fractions:
        index = int(round(float(frac) * max(count - 1, 0)))
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = cap.read()
        if not ok:
            frames.append(None)
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()
    return frames


def _draw_visual_header(ax, bin_centers, input_len, frames=None):
    n = len(bin_centers)
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.axis("off")

    if frames is None:
        for j, center in enumerate(bin_centers):
            rect = Rectangle(
                (j + 0.05, 0.12),
                0.90,
                0.72,
                facecolor="#F2F2F2",
                edgecolor="#777777",
                linewidth=0.7,
            )
            ax.add_patch(rect)
            ax.text(
                j + 0.5,
                0.48,
                "V{}".format(int(center) + 1),
                ha="center",
                va="center",
                fontsize=8.5,
            )
    else:
        for j, frame in enumerate(frames):
            if frame is None:
                rect = Rectangle(
                    (j + 0.05, 0.12),
                    0.90,
                    0.72,
                    facecolor="#F2F2F2",
                    edgecolor="#777777",
                    linewidth=0.7,
                )
                ax.add_patch(rect)
            else:
                ax.imshow(
                    frame,
                    extent=(j + 0.04, j + 0.96, 0.10, 0.88),
                    aspect="auto",
                    interpolation="bilinear",
                )
            ax.text(
                j + 0.5,
                0.02,
                "V{}".format(int(bin_centers[j]) + 1),
                ha="center",
                va="bottom",
                fontsize=7.5,
            )
    ax.text(
        -0.15,
        0.5,
        "Visual\nwindows",
        ha="right",
        va="center",
        fontsize=9.5,
        fontweight="semibold",
    )


def _clean_token(token):
    token = str(token)
    if token.startswith("##"):
        return token[2:]
    return token


def plot_figure(
    cli,
    selected,
    baseline_pack,
    cfcompat_pack,
    baseline_model,
    cfg,
    output_dir,
):
    base_attn = baseline_pack["attention"]
    ours_attn = cfcompat_pack["attention"]

    rows_base, labels_base = build_text_rows(
        baseline_pack, baseline_model, cfg, cli.max_tokens
    )
    rows_ours, labels_ours = build_text_rows(
        cfcompat_pack, baseline_model, cfg, cli.max_tokens
    )

    # Same input => same query positions. Use the intersection to guarantee a
    # one-to-one comparison even if a tokenizer edge case occurs.
    common_rows = [
        int(row) for row in rows_base
        if int(row) in set(rows_ours.tolist())
    ]
    if not common_rows:
        raise RuntimeError("No common text query positions were available.")

    label_by_row = {
        int(row): label
        for row, label in zip(rows_base.tolist(), labels_base)
    }
    labels = [_clean_token(label_by_row[row]) for row in common_rows]

    base = base_attn[np.asarray(common_rows)]
    ours = ours_attn[np.asarray(common_rows)]

    base, bin_centers, chunks = aggregate_visual_bins(
        base,
        cfg,
        cli.visual_bins,
        baseline_pack["vision_input"].shape[0],
    )
    ours, ours_centers, _ = aggregate_visual_bins(
        ours,
        cfg,
        cli.visual_bins,
        cfcompat_pack["vision_input"].shape[0],
    )
    if not np.array_equal(bin_centers, ours_centers):
        raise RuntimeError("Baseline/CFCompat visual bin centers differ.")

    base_disp = normalize_rows_for_display(base)
    ours_disp = normalize_rows_for_display(ours)

    fractions = (
        (bin_centers.astype(np.float64) + 0.5)
        / float(baseline_pack["vision_input"].shape[0])
    )
    frames = load_video_frames(cli.video_file, fractions)

    base_pred = float(baseline_pack["prediction"])
    ours_pred = float(cfcompat_pack["prediction"])
    truth = float(baseline_pack["label"])
    base_ae = abs(base_pred - truth)
    ours_ae = abs(ours_pred - truth)

    plt.rcParams.update({
        "font.family": "Times New Roman",
        "mathtext.fontset": "stix",
        "font.size": 10.5,
        "axes.titlesize": 11.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 9.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    fig = plt.figure(figsize=(10.2, 7.1))
    gs = gridspec.GridSpec(
        4,
        2,
        width_ratios=[1.35, 5.8],
        height_ratios=[0.75, 2.45, 0.75, 2.45],
        hspace=0.16,
        wspace=0.04,
    )

    ax_label1 = fig.add_subplot(gs[0:2, 0])
    ax_head1 = fig.add_subplot(gs[0, 1])
    ax_map1 = fig.add_subplot(gs[1, 1])

    ax_label2 = fig.add_subplot(gs[2:4, 0])
    ax_head2 = fig.add_subplot(gs[2, 1])
    ax_map2 = fig.add_subplot(gs[3, 1])

    for ax in (ax_label1, ax_label2):
        ax.axis("off")

    _draw_visual_header(
        ax_head1,
        bin_centers,
        baseline_pack["vision_input"].shape[0],
        frames,
    )
    _draw_visual_header(
        ax_head2,
        bin_centers,
        baseline_pack["vision_input"].shape[0],
        frames,
    )

    vmax = max(
        float(base_disp.max()),
        float(ours_disp.max()),
        1e-8,
    )
    im1 = ax_map1.imshow(
        base_disp,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
    )
    im2 = ax_map2.imshow(
        ours_disp,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0.0,
        vmax=vmax,
    )

    for ax in (ax_map1, ax_map2):
        ax.set_yticks(np.arange(len(labels)))
        ax.set_yticklabels(labels)
        ax.set_xticks(np.arange(len(bin_centers)))
        ax.set_xticklabels(
            ["V{}".format(int(v) + 1) for v in bin_centers],
            rotation=0,
        )
        ax.set_xlabel("Visual time window")
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color("#555555")

    # Emphasize the three strongest CFCompat query rows by label color only.
    ours_strength = ours_disp.max(axis=1)
    top = set(
        np.argsort(-ours_strength, kind="mergesort")[
            : min(3, len(ours_strength))
        ].tolist()
    )
    for i, tick in enumerate(ax_map2.get_yticklabels()):
        if i in top:
            tick.set_color("#B22222")
            tick.set_fontweight("bold")

    baseline_name = "Uniform KD" if cli.baseline == "fixedkd" else "DLF-ModDrop"
    ax_label1.text(
        0.98,
        0.64,
        "(a) {}".format(baseline_name),
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax_label1.text(
        0.98,
        0.45,
        "Pred = {:.3f}\nAE = {:.3f}".format(base_pred, base_ae),
        ha="right",
        va="center",
        fontsize=10,
    )

    ax_label2.text(
        0.98,
        0.64,
        "(b) CFCompat",
        ha="right",
        va="center",
        fontsize=12,
        fontweight="bold",
    )
    ax_label2.text(
        0.98,
        0.45,
        "Pred = {:.3f}\nAE = {:.3f}".format(ours_pred, ours_ae),
        ha="right",
        va="center",
        fontsize=10,
    )

    raw = str(baseline_pack["raw_text"]).strip()
    if len(raw) > 145:
        raw = raw[:142] + "..."

    fig.suptitle(
        "Text--Vision Cross-modal Attention under the {} Condition\n"
        "Truth = {:.3f}   |   {}   |   sample {}".format(
            cli.condition,
            truth,
            raw,
            int(selected["sample_index"]),
        ),
        fontsize=12.5,
        y=0.985,
    )

    cax = fig.add_axes([0.925, 0.20, 0.014, 0.60])
    cb = fig.colorbar(im2, cax=cax)
    cb.set_label("Attention weight", fontsize=10)
    cb.ax.tick_params(labelsize=8)

    fig.text(
        0.57,
        0.025,
        "Rows are text-query positions and columns are aggregated visual-key windows; "
        "both panels share the same color scale.",
        ha="center",
        va="bottom",
        fontsize=9.2,
    )

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
    npz = output_dir / "{}_attention.npz".format(stem)

    fig.savefig(png, dpi=600, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    np.savez_compressed(
        npz,
        baseline_attention_raw=base_attn,
        cfcompat_attention_raw=ours_attn,
        baseline_attention_display=base_disp,
        cfcompat_attention_display=ours_disp,
        query_rows=np.asarray(common_rows, dtype=np.int64),
        text_labels=np.asarray(labels, dtype=object),
        visual_bin_centers=bin_centers,
    )

    return png, pdf, npz, {
        "truth": truth,
        "baseline_prediction": base_pred,
        "cfcompat_prediction": ours_pred,
        "baseline_abs_error": base_ae,
        "cfcompat_abs_error": ours_ae,
        "display_tokens": labels,
        "visual_bin_centers": [int(v) for v in bin_centers],
    }


def main():
    cli = parse_args()
    if cli.split == "test" and cli.sample_index is None:
        raise ValueError(
            "Automatic qualitative case selection is validation-only. "
            "For --split test, provide an explicit --sample-index."
        )
    if cli.condition not in ("LV", "LAV"):
        raise ValueError(
            "Figure-4-style text--vision attention requires vision to be present."
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

    baseline_pack = capture_text_to_vision_attention(
        baseline,
        sample,
        cfg,
        cli.condition,
        cli.layer_index,
    )
    cfcompat_pack = capture_text_to_vision_attention(
        cfcompat,
        sample,
        cfg,
        cli.condition,
        cli.layer_index,
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
        cli.dataset, cli.seed, cli.split
    )
    candidates.to_csv(candidate_csv, index=False)

    png, pdf, npz, plot_meta = plot_figure(
        cli,
        selected,
        baseline_pack,
        cfcompat_pack,
        baseline,
        cfg,
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
        "layer_index": int(cli.layer_index),
        "attention_source": "DLF trans_l_with_v final/selected Transformer layer, head-averaged",
        "note": (
            "This is true model attention, not a gradient saliency proxy. "
            "Visual columns are contiguous attention-key bins."
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
    print("Figure-4-style cross-modal attention visualization complete.")
    print("baseline checkpoint : {}".format(baseline_ckpt))
    print("CFCompat checkpoint : {}".format(cfcompat_ckpt))
    print("split / condition   : {} / {}".format(cli.split, cli.condition))
    print("selection            : {} ({})".format(cli.selection, selection_reason))
    print("sample_index         : {}".format(sample_index))
    print("sample_id            : {}".format(selected["sample_id"]))
    print("truth                : {:.6f}".format(float(selected["label"])))
    print("baseline pred / AE   : {:.6f} / {:.6f}".format(
        float(selected["baseline_pred"]),
        float(selected["baseline_abs_error"]),
    ))
    print("CFCompat pred / AE   : {:.6f} / {:.6f}".format(
        float(selected["cfcompat_pred"]),
        float(selected["cfcompat_abs_error"]),
    ))
    print("error gain           : {:.6f}".format(float(selected["error_gain"])))
    print("raw text             : {}".format(selected["raw_text"]))
    print("-" * 88)
    print("PNG                   : {}".format(png))
    print("PDF                   : {}".format(pdf))
    print("NPZ                   : {}".format(npz))
    print("candidates            : {}".format(candidate_csv))
    print("metadata              : {}".format(meta_path))
    print("=" * 88)


if __name__ == "__main__":
    main()
