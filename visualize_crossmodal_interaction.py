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
import ast
import io
import json
import re
import subprocess
import textwrap
import warnings
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
from matplotlib.colors import Normalize

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


# Visualization-only lexical filtering. These words are not removed from the
# model input or from the interaction computation; they are hidden only in the
# final qualitative figure to reduce clutter, following the presentation style
# of PMR Figure 4.
DISPLAY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
    "by", "do", "does", "did", "for", "from", "he", "her", "hers", "him",
    "his", "i", "if", "in", "is", "it", "its", "me", "mean", "my", "of",
    "on", "or", "other", "our", "ours", "she", "so", "that", "the", "their",
    "theirs", "them", "then", "these", "they", "this", "those", "to", "too",
    "us", "very", "was", "we", "were", "with", "you", "your", "yours",
}

# A compact visualization lexicon used only to color clearly sentiment-bearing
# words in red, analogous to the qualitative annotation in PMR Figure 4.
DISPLAY_SENTIMENT_WORDS = {
    "amazing", "awful", "bad", "beautiful", "boring", "comedy", "disappointed",
    "disappointing", "excellent", "fantastic", "funny", "good", "great", "hate",
    "hated", "horrible", "horror", "love", "loved", "poor", "successful",
    "terrible", "terrific", "atrocious", "wonderful", "worst",
}


def _display_token(token):
    return str(token).lower().strip().replace("##", "")


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
    p.add_argument(
        "--max-words",
        type=int,
        default=8,
        help="Maximum number of content words displayed after stopword filtering.",
    )
    p.add_argument("--visual-bins", type=int, default=10)
    p.add_argument(
        "--display-gamma",
        type=float,
        default=0.65,
        help=(
            "Shared global contrast exponent applied after robust normalization. "
            "Values below 1 reveal weaker interactions without row-wise "
            "renormalization. Both methods use the same transform."
        ),
    )
    p.add_argument("--occlusion-batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--gpu-ids", nargs="*", type=int, default=[0])
    p.add_argument("--model-save-dir", default="pt")
    p.add_argument("--config-file", default="config/config.json")
    p.add_argument("--baseline-checkpoint")
    p.add_argument("--cfcompat-checkpoint")
    p.add_argument(
        "--mosi-raw-root",
        default="/sharefile/lyx_model/MMSA_new/MOSI/Raw/Raw",
        help=(
            "Root containing MOSI utterance clips as <video_id>/<segment_id>.mp4. "
            "Used automatically unless --video-file is supplied."
        ),
    )
    p.add_argument(
        "--video-file",
        help="Optional explicit utterance-level mp4; overrides --mosi-raw-root lookup.",
    )
    p.add_argument(
        "--no-video-frames",
        action="store_true",
        help="Disable raw-video frame extraction and keep visual-window placeholders.",
    )
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
    """Choose a compact, method-symmetric set of content words for display.

    The interaction is still computed for every word.  Stopword removal is a
    visualization-only operation. Among the remaining content words, ranking
    uses the maximum cross-modal interaction observed in either model, so the
    selection does not favor Uniform KD or CFCompat.
    """
    n = len(labels)
    if n == 0:
        return np.asarray([], dtype=np.int64)

    content = []
    for i, label in enumerate(labels):
        token = _display_token(label)
        if not token:
            continue
        if token in DISPLAY_STOPWORDS:
            continue
        if len(token) <= 1:
            continue
        content.append(i)

    # Never fail on an unusual utterance consisting almost entirely of
    # function words.
    if not content:
        content = list(range(n))

    base_strength = np.max(
        baseline_pack["interaction"],
        axis=1,
    )
    ours_strength = np.max(
        cfcompat_pack["interaction"],
        axis=1,
    )
    score = np.maximum(base_strength, ours_strength)

    limit = max(1, int(max_words))
    if len(content) > limit:
        ranked = sorted(
            content,
            key=lambda i: (-float(score[i]), int(i)),
        )
        chosen = ranked[:limit]

        # Clearly sentiment-bearing words are useful anchors in the qualitative
        # figure. If one was pushed out by a near-tied interaction score, keep
        # it by replacing the weakest currently selected non-sentiment word.
        sentiment_candidates = [
            i for i in content
            if _display_token(labels[i]) in DISPLAY_SENTIMENT_WORDS
        ]
        for i in sentiment_candidates:
            if i in chosen:
                continue
            replaceable = [
                j for j in chosen
                if _display_token(labels[j]) not in DISPLAY_SENTIMENT_WORDS
            ]
            if not replaceable:
                continue
            weakest = min(replaceable, key=lambda j: float(score[j]))
            chosen.remove(weakest)
            chosen.append(i)
    else:
        chosen = content

    # Restore sentence order for readability.
    return np.asarray(sorted(set(chosen)), dtype=np.int64)


def _sample_id_to_python(sample_id):
    """Convert bytes/tensor/ndarray wrappers while preserving structured IDs."""
    if isinstance(sample_id, bytes):
        return sample_id.decode("utf-8", errors="replace")

    if torch.is_tensor(sample_id):
        value = sample_id.detach().cpu().numpy()
        if value.ndim == 0:
            return _sample_id_to_python(value.item())
        return [_sample_id_to_python(v) for v in value.tolist()]

    if isinstance(sample_id, np.ndarray):
        if sample_id.ndim == 0:
            return _sample_id_to_python(sample_id.item())
        return [_sample_id_to_python(v) for v in sample_id.tolist()]

    if isinstance(sample_id, (list, tuple)):
        return [_sample_id_to_python(v) for v in sample_id]

    return sample_id


def _try_literal_structured_id(value):
    """Parse stringified list/tuple IDs such as "['video', '7']"."""
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[(":
        return value
    try:
        parsed = ast.literal_eval(stripped)
    except Exception:
        return value
    return _sample_id_to_python(parsed)


def _find_segment_file(raw_root, video_id, segment_id):
    """Return an exact clip, then a safe MOSI 0-based->1-based fallback.

    Many processed MOSI feature files encode segment indices from 0, while
    utterance clips on disk are named 1.mp4, 2.mp4, ... . We only apply +1
    when the exact file is absent and the directory itself has no 0.mp4,
    which makes the convention mismatch explicit rather than arbitrary.
    """
    folder = Path(raw_root) / str(video_id)
    if not folder.is_dir():
        return None, None, None

    try:
        segment_int = int(float(str(segment_id).strip()))
    except Exception:
        return None, None, None

    exact = folder / (str(segment_int) + ".mp4")
    if exact.is_file():
        return exact, str(segment_int), 0

    plus_one = folder / (str(segment_int + 1) + ".mp4")
    zero_file = folder / "0.mp4"
    one_file = folder / "1.mp4"

    if (not zero_file.exists()) and one_file.is_file() and plus_one.is_file():
        return plus_one, str(segment_int + 1), 1

    return None, None, None


def resolve_mosi_clip_path(sample_id, raw_root):
    """Resolve a MOSI utterance ID to raw_root/video_id/segment.mp4.

    Returns
    -------
    clip_path, video_id, disk_segment_id, segment_offset, normalized_id
    """
    raw_root = Path(raw_root)
    normalized = _sample_id_to_python(sample_id)
    normalized = _try_literal_structured_id(normalized)

    video_id = None
    segment_id = None

    if isinstance(normalized, dict):
        for key in ("video_id", "video", "vid"):
            if key in normalized:
                video_id = str(normalized[key])
                break
        for key in ("segment_id", "segment", "clip_id", "clip", "sid"):
            if key in normalized:
                segment_id = str(normalized[key])
                break

    elif isinstance(normalized, (list, tuple)) and len(normalized) >= 2:
        video_id = str(normalized[0]).strip()
        segment_id = str(normalized[1]).strip()

    else:
        sid = str(normalized).strip()

        # Match actual MOSI video-directory names anywhere inside the serialized
        # ID. This also handles strings such as "['1DmNV9C1hbY', '7']".
        matched_dir = None
        if raw_root.is_dir():
            dirs = [p.name for p in raw_root.iterdir() if p.is_dir()]
            dirs.sort(key=len, reverse=True)
            for candidate in dirs:
                if candidate in sid:
                    matched_dir = candidate
                    remainder = sid.split(candidate, 1)[1]
                    numbers = re.findall(r"\d+", remainder)
                    if numbers:
                        video_id = candidate
                        segment_id = numbers[0]
                        break

        if video_id is None:
            patterns = (
                r"^(.+?)\$_\$(\d+)$",
                r"^(.+?)\[(\d+)\]$",
                r"^(.+?)[/#,:](\d+)$",
                r"^(.+?)::(\d+)$",
                r"^(.+?)\s+(\d+)$",
            )
            for pattern in patterns:
                match = re.match(pattern, sid)
                if match:
                    video_id, segment_id = match.group(1), match.group(2)
                    break

        if video_id is None:
            match = re.match(r"^(.+)_(\d+)$", sid)
            if match:
                video_id, segment_id = match.group(1), match.group(2)

    if video_id is None or segment_id is None:
        return None, video_id, segment_id, None, normalized

    clip, disk_segment, offset = _find_segment_file(
        raw_root,
        video_id,
        segment_id,
    )
    return clip, str(video_id), disk_segment, offset, normalized


def _sample_decoded_frames(all_frames, relative_positions):
    if not all_frames:
        return None
    nframes = len(all_frames)
    output = []
    for rel in relative_positions:
        index = int(round(float(rel) * max(nframes - 1, 0)))
        index = min(max(index, 0), max(nframes - 1, 0))
        output.append(all_frames[index])
    return output


def _read_video_frames_cv2(video_path, relative_positions):
    """Decode the whole short MOSI utterance sequentially with OpenCV.

    Sequential decoding is used instead of repeated random seeking because
    some MOSI mp4 files have sparse keyframes: OpenCV can open the file while
    random seek followed by read still fails.
    """
    try:
        import cv2
    except ImportError:
        return None

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None

    decoded = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        decoded.append(frame)

    cap.release()
    return _sample_decoded_frames(decoded, relative_positions)


def _read_video_frames_imageio(video_path, relative_positions):
    """Fallback: decode the complete short utterance with imageio."""
    try:
        import imageio.v3 as iio
    except Exception:
        return None

    try:
        all_frames = iio.imread(video_path)
        if all_frames is None or len(all_frames) == 0:
            return None
        return _sample_decoded_frames(list(all_frames), relative_positions)
    except Exception:
        return None


def _ffprobe_duration(video_path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
            text=True,
        )
        duration = float(result.stdout.strip())
        if duration > 0:
            return duration
    except Exception:
        pass
    return None


def _read_video_frames_ffmpeg(video_path, relative_positions):
    """Final fallback using the ffmpeg executable directly."""
    try:
        from PIL import Image
    except Exception:
        return None

    duration = _ffprobe_duration(video_path)
    if duration is None:
        return None

    frames = []
    for rel in relative_positions:
        sec = min(max(float(rel), 0.0), 0.995) * duration
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-v", "error",
                    "-ss", "{:.6f}".format(sec),
                    "-i", str(video_path),
                    "-frames:v", "1",
                    "-f", "image2pipe",
                    "-vcodec", "png",
                    "pipe:1",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
            )
            if not result.stdout:
                frames.append(None)
                continue
            image = Image.open(io.BytesIO(result.stdout)).convert("RGB")
            frames.append(np.asarray(image))
        except Exception:
            frames.append(None)

    if not frames or all(frame is None for frame in frames):
        return None
    return frames


def load_video_frames(video_path, centers, active_positions):
    """Extract one real frame per displayed visual window.

    Returns frames plus the decoder backend used.
    """
    if video_path is None:
        return None, "placeholder"

    path = Path(video_path)
    if not path.is_file():
        return None, "placeholder"

    active_positions = np.asarray(active_positions, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if active_positions.size == 0:
        return None, "placeholder"

    lo = float(active_positions.min())
    hi = float(active_positions.max())
    span = max(hi - lo, 1.0)
    relative = np.clip((centers - lo) / span, 0.0, 1.0)

    readers = (
        ("opencv", _read_video_frames_cv2),
        ("imageio", _read_video_frames_imageio),
        ("ffmpeg", _read_video_frames_ffmpeg),
    )

    for backend, reader in readers:
        frames = reader(path, relative)
        if frames is None:
            continue

        valid_count = sum(frame is not None for frame in frames)
        if valid_count == len(relative):
            return frames, backend

        for fallback_name, fallback_reader in readers:
            if fallback_name == backend:
                continue
            fallback = fallback_reader(path, relative)
            if fallback is None:
                continue
            merged = [
                frame if frame is not None else fallback[i]
                for i, frame in enumerate(frames)
            ]
            if all(frame is not None for frame in merged):
                return merged, backend + "+" + fallback_name

        if valid_count > 0:
            return frames, backend + "_partial"

    warnings.warn(
        "The raw clip exists but no decoder could extract frames from {}. "
        "Tried OpenCV sequential decoding, imageio, and ffmpeg. "
        "Using placeholders instead.".format(path)
    )
    return None, "placeholder"


def _crop_frame_for_strip(frame):
    """Center-crop a frame to a compact 4:3 thumbnail."""
    if frame is None:
        return None
    frame = np.asarray(frame)
    if frame.ndim != 3 or frame.shape[0] == 0 or frame.shape[1] == 0:
        return frame

    h, w = frame.shape[:2]
    target_ratio = 4.0 / 3.0
    current_ratio = float(w) / float(h)

    if current_ratio > target_ratio:
        new_w = max(1, int(round(h * target_ratio)))
        x0 = max(0, (w - new_w) // 2)
        frame = frame[:, x0:x0 + new_w]
    else:
        new_h = max(1, int(round(w / target_ratio)))
        y0 = max(0, (h - new_h) // 2)
        frame = frame[y0:y0 + new_h, :]
    return frame


def draw_visual_header(
    ax,
    window_labels,
    importance,
    frames=None,
):
    """PMR-style strip of real frames or explicit placeholders."""
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
        edge = cmap(0.18 + 0.78 * strength)

        x0, x1 = j + 0.06, j + 0.94
        y0, y1 = 0.20, 0.93

        frame = None if frames is None else _crop_frame_for_strip(frames[j])
        if frame is not None:
            ax.imshow(
                frame,
                extent=(x0, x1, y0, y1),
                aspect="auto",
                interpolation="bilinear",
            )
        else:
            rect = Rectangle(
                (x0, y0),
                x1 - x0,
                y1 - y0,
                facecolor=(0.95, 0.95, 0.95),
                edgecolor="none",
            )
            ax.add_patch(rect)
            ax.text(
                (x0 + x1) / 2.0,
                (y0 + y1) / 2.0,
                label,
                ha="center",
                va="center",
                fontsize=8.1,
                color="#333333",
            )

        border = Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            facecolor="none",
            edgecolor=edge,
            linewidth=2.0 if strength > 0.55 else 1.35,
        )
        ax.add_patch(border)

        ax.text(
            (x0 + x1) / 2.0,
            0.055,
            label,
            ha="center",
            va="bottom",
            fontsize=7.6,
            color="#333333",
        )

def draw_word_column(ax, labels):
    """Draw a compact PMR-style word column."""
    n = len(labels)
    ax.set_xlim(0, 1)
    ax.set_ylim(n - 0.5, -0.5)
    ax.axis("off")
    for row, label in enumerate(labels):
        token = _display_token(label)
        sentiment = token in DISPLAY_SENTIMENT_WORDS
        ax.text(
            0.96,
            row,
            str(label),
            ha="right",
            va="center",
            fontsize=10.1 if sentiment else 9.7,
            color="#D62728" if sentiment else "#222222",
            fontweight="semibold" if sentiment else "normal",
        )


def draw_method_meta(ax, panel_label, method_name, prediction, error):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.96,
        0.63,
        "{} {}".format(panel_label, method_name),
        ha="right",
        va="center",
        fontsize=11.9,
        fontweight="bold",
    )
    ax.text(
        0.96,
        0.42,
        "Pred  {:.2f}\nAE    {:.2f}".format(prediction, error),
        ha="right",
        va="center",
        fontsize=9.9,
        linespacing=1.20,
    )


def plot_figure(
    cli,
    selected,
    sample,
    labels,
    windows,
    window_labels,
    centers,
    active_visual,
    video_path,
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

    # Shared global normalization preserves comparisons across words, visual
    # windows, and methods. A monotonic gamma transform is then applied to both
    # models together to reveal weak-but-nonzero interactions without forcing
    # each row to contain a saturated maximum.
    combined = np.concatenate([base.ravel(), ours.ravel()])
    positive = combined[combined > 0]
    if positive.size:
        shared_vmax = float(np.quantile(positive, 0.97))
        shared_vmax = max(
            shared_vmax,
            float(positive.max()) * 0.20,
            1e-12,
        )
    else:
        shared_vmax = 1.0

    gamma = float(cli.display_gamma)
    if not (0.0 < gamma <= 1.0):
        raise ValueError("--display-gamma must lie in (0, 1].")

    base_linear = np.clip(base / shared_vmax, 0.0, 1.0)
    ours_linear = np.clip(ours / shared_vmax, 0.0, 1.0)

    base_disp = np.power(base_linear, gamma)
    ours_disp = np.power(ours_linear, gamma)

    # The displayed score is already transformed into [0,1], so the colorbar
    # itself remains linear and its 0.2 intervals have equal physical length.
    display_norm = Normalize(
        vmin=0.0,
        vmax=1.0,
        clip=True,
    )

    visual_importance = 0.5 * (
        baseline_pack["visual_effect"] + cfcompat_pack["visual_effect"]
    )
    frames, video_backend = load_video_frames(
        video_path,
        centers,
        active_visual,
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

    fig = plt.figure(figsize=(10.9, 6.45))
    gs = gridspec.GridSpec(
        3,
        3,
        width_ratios=[1.40, 1.00, 6.45],
        height_ratios=[1.02, 2.30, 2.30],
        hspace=0.15,
        wspace=0.035,
    )

    ax_header_meta = fig.add_subplot(gs[0, 0])
    ax_header_words = fig.add_subplot(gs[0, 1])
    ax_header = fig.add_subplot(gs[0, 2])

    ax_meta1 = fig.add_subplot(gs[1, 0])
    ax_words1 = fig.add_subplot(gs[1, 1])
    ax_map1 = fig.add_subplot(gs[1, 2])

    ax_meta2 = fig.add_subplot(gs[2, 0])
    ax_words2 = fig.add_subplot(gs[2, 1])
    ax_map2 = fig.add_subplot(gs[2, 2])

    ax_header_meta.axis("off")
    ax_header_words.axis("off")

    ax_header_words.text(
        0.96,
        0.57,
        "Visual\nframes",
        ha="right",
        va="center",
        fontsize=10,
        fontweight="semibold",
    )

    draw_visual_header(
        ax_header,
        window_labels,
        visual_importance,
        frames,
    )

    draw_word_column(ax_words1, display_labels)
    draw_word_column(ax_words2, display_labels)

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
        panel.set_yticklabels([""] * len(display_labels))
        panel.tick_params(axis="y", length=0)
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

    baseline_name = "Uniform KD" if cli.baseline == "fixedkd" else "DLF-ModDrop"
    draw_method_meta(
        ax_meta1,
        "(a)",
        baseline_name,
        bp,
        be,
    )
    draw_method_meta(
        ax_meta2,
        "(b)",
        "CFCompat",
        op,
        oe,
    )

    raw = str(selected["raw_text"]).strip()
    wrapped = textwrap.fill(raw, width=92)

    fig.suptitle(
        "Cross-modal Interaction on CMU-MOSI ({})".format(cli.condition),
        fontsize=13.3,
        y=0.992,
        fontweight="semibold",
    )
    fig.text(
        0.52,
        0.943,
        "Truth = {:.2f}   |   {}".format(truth, wrapped),
        ha="center",
        va="top",
        fontsize=10.6,
        linespacing=1.15,
    )

    cax = fig.add_axes([0.935, 0.190, 0.013, 0.565])
    color_ticks = np.linspace(0.0, 1.0, 6)
    cb = fig.colorbar(
        im2,
        cax=cax,
        ticks=color_ticks,
    )
    cb.ax.set_yticklabels(
        ["{:.1f}".format(v) for v in color_ticks]
    )
    cb.set_label("Relative interaction score", fontsize=10)
    cb.ax.tick_params(labelsize=8, length=3)
    cb.outline.set_linewidth(0.8)

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
        baseline_interaction_linear=base_linear,
        cfcompat_interaction_linear=ours_linear,
        baseline_interaction_display=base_disp,
        cfcompat_interaction_display=ours_disp,
        shared_display_vmax_raw=np.asarray(
            [shared_vmax],
            dtype=np.float64,
        ),
        display_gamma=np.asarray([gamma], dtype=np.float64),
        display_scale=np.asarray(
            ["shared_global_gamma_then_linear_colorbar"],
            dtype=object,
        ),
    )

    return png, pdf, npz, {
        "truth": truth,
        "baseline_prediction": bp,
        "cfcompat_prediction": op,
        "baseline_abs_error": be,
        "cfcompat_abs_error": oe,
        "displayed_words": display_labels,
        "visual_window_labels": window_labels,
        "display_scale": "shared_global_gamma_then_linear_colorbar",
        "shared_display_vmax_raw": shared_vmax,
        "display_gamma": gamma,
        "video_decode_backend": video_backend,
        "video_frames_rendered": 0 if frames is None else int(
            sum(frame is not None for frame in frames)
        ),
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

    video_path = None
    resolved_video_id = None
    resolved_segment_id = None
    resolved_segment_offset = None
    normalized_sample_id = _sample_id_to_python(sample["id"])

    if not cli.no_video_frames:
        if cli.video_file:
            explicit = Path(cli.video_file)
            if not explicit.is_file():
                raise FileNotFoundError(
                    "--video-file does not exist: {}".format(explicit)
                )
            video_path = explicit
        else:
            (
                video_path,
                resolved_video_id,
                resolved_segment_id,
                resolved_segment_offset,
                normalized_sample_id,
            ) = resolve_mosi_clip_path(
                sample["id"],
                cli.mosi_raw_root,
            )
            if video_path is None:
                warnings.warn(
                    "Could not resolve raw MOSI clip for sample id {!r} "
                    "(normalized={!r}) under {}. The figure will use "
                    "visual-window placeholders. Use --video-file to provide "
                    "the exact utterance clip.".format(
                        sample["id"],
                        normalized_sample_id,
                        cli.mosi_raw_root,
                    )
                )
            elif resolved_segment_offset == 1:
                warnings.warn(
                    "Mapped processed MOSI segment index to one-based raw clip "
                    "filename: sample id {!r} -> {}.".format(
                        normalized_sample_id,
                        video_path,
                    )
                )

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
        active_visual,
        video_path,
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
        "display_scale": "shared_global_gamma_then_linear_colorbar",
        "heatmap_semantics": (
            "all displayed cells share one robust raw normalization across "
            "Uniform KD and CFCompat, followed by the same monotonic gamma "
            "contrast transform; cross-row and cross-model ordering is preserved"
        ),
        "colorbar_semantics": (
            "linear 0--1 scale over the contrast-enhanced interaction score"
        ),
        "colorbar_ticks": [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "requested_display_gamma_legacy": float(cli.display_gamma),
        "raw_video_root": str(cli.mosi_raw_root),
        "resolved_video_path": None if video_path is None else str(video_path),
        "normalized_sample_id": str(normalized_sample_id),
        "resolved_video_id": resolved_video_id,
        "resolved_segment_id": resolved_segment_id,
        "resolved_segment_offset": resolved_segment_offset,
        "note": (
            "Padding-only visual steps are excluded before binning. No model "
            "parameter is changed. All displayed cells use one shared raw "
            "normalization across both models, followed by the same monotonic "
            "gamma contrast transform. This reveals weaker interactions without "
            "forcing a maximum in every row. The final colorbar is linear in "
            "the displayed interaction score, so every 0.2 interval has equal "
            "physical length. Stopword removal affects visualization only; raw "
            "interactions for all words are saved in NPZ."
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
    print("sample id (raw)      : {!r}".format(sample["id"]))
    print("sample id (norm)     : {!r}".format(normalized_sample_id))
    print("raw root exists      : {}".format(Path(cli.mosi_raw_root).is_dir()))
    print("video id / segment   : {} / {}".format(
        resolved_video_id,
        resolved_segment_id,
    ))
    print("segment offset       : {}".format(resolved_segment_offset))
    print("raw video            : {}".format(
        "<not resolved>" if video_path is None else video_path
    ))
    print("video decoder        : {}".format(
        plot_meta.get("video_decode_backend", "unknown")
    ))
    print("frames rendered      : {}".format(
        plot_meta.get("video_frames_rendered", 0)
    ))
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
