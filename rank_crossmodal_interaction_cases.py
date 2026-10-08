"""Validation-only search for illustrative CFCompat interaction case studies.

Two-stage protocol
------------------
1) Scan all validation predictions under the same target condition.
   Keep positive CFCompat error-gain examples with a nontrivial sentiment
   label, sufficient text and visual support, and at least one sentiment cue.
   Evaluate up to --contrast-pool-size examples with the highest prediction
   improvement. The screened pool is explicitly disclosed in CSV.
2) Compute *raw* word--visual-context pairwise occlusion interactions for
   both Uniform KD and CFCompat. Rank examples by a fixed equal-weight
   z-score sum of (prediction improvement, sentiment-cue concentration gain,
   and total-variation contrast of the two interaction maps).

Scoring uses raw interaction matrices. Gamma enhancement, color normalization,
and the top-k words shown in figures do not affect scores. Selection never
reads Test, changes parameters, or retrains a model. This is a qualitative
illustration search, not an unbiased performance evaluation.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from visualize_crossmodal_interaction import (
    DEFAULT_OUTPUT,
    DISPLAY_SENTIMENT_WORDS,
    DISPLAY_STOPWORDS,
    _display_token,
    _sample_id_to_python,
    bert_word_groups,
    interaction_from_predictions,
    plot_figure,
    resolve_mosi_clip_path,
    run_occlusion_variants,
    visual_windows,
)


def _safe_fraction(numerator, denominator):
    if denominator <= 1e-12:
        return 0.0
    return float(numerator) / float(denominator)


def lexical_indices(labels):
    """Select content-word rows, then sentiment-cue rows within that set.

    This explicit and limited lexicon is a visualization heuristic, not
    ground-truth word attribution. All words are still perturbed in the
    occlusion computation.
    """
    content = [
        i for i, token in enumerate(labels)
        if _display_token(token) not in DISPLAY_STOPWORDS
        and len(_display_token(token)) > 1
        and any(ch.isalpha() for ch in _display_token(token))
    ]
    sentiment = [
        i for i in content
        if _display_token(labels[i]) in DISPLAY_SENTIMENT_WORDS
    ]
    return content, sentiment


def interaction_metrics(baseline_pack, cfcompat_pack, labels):
    """Raw, pre-display interaction metrics computed on identical word/window grids."""
    a = np.asarray(baseline_pack["interaction"], dtype=np.float64)
    b = np.asarray(cfcompat_pack["interaction"], dtype=np.float64)

    if a.shape != b.shape or a.shape[0] != len(labels):
        raise ValueError("Interaction shapes/word labels must match across models.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Non-finite raw interaction values.")

    content, sentiment = lexical_indices(labels)
    if not content or not sentiment:
        raise ValueError("Sentiment-aware ranking requires content and sentiment cues.")

    a_content = a[content, :]
    b_content = b[content, :]

    content_to_local = {source_idx: i for i, source_idx in enumerate(content)}
    sentiment_local = [content_to_local[i] for i in sentiment]

    a_total = float(a_content.sum())
    b_total = float(b_content.sum())
    a_senti = float(a_content[sentiment_local, :].sum())
    b_senti = float(b_content[sentiment_local, :].sum())

    base_ratio = _safe_fraction(a_senti, a_total)
    ours_ratio = _safe_fraction(b_senti, b_total)

    # TV distance of L1-normalized interaction maps, on the SAME content words
    # and visual windows: 0 = identical allocation, 1 = maximally different.
    if a_total <= 1e-12 or b_total <= 1e-12:
        contrast = 0.0
    else:
        contrast = 0.5 * float(
            np.abs(a_content / a_total - b_content / b_total).sum()
        )

    return {
        "sentiment_words": ", ".join(
            str(labels[i]) for i in sentiment
        ),
        "content_word_count": int(len(content)),
        "sentiment_word_count": int(len(sentiment)),
        "uniform_sentiment_ratio": base_ratio,
        "cfcompat_sentiment_ratio": ours_ratio,
        "sentiment_ratio_gain": ours_ratio - base_ratio,
        "map_contrast_tv": float(np.clip(contrast, 0.0, 1.0)),
        "uniform_interaction_mass": a_total,
        "cfcompat_interaction_mass": b_total,
    }


def _standardize(values):
    values = np.asarray(values, dtype=np.float64)
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return np.zeros_like(values)
    return (values - mean) / std


def _first_stage(cli, dataset, candidates, baseline):
    """Screen validation with prediction gain and a predeclared cue lexicon."""
    screening = candidates.copy()
    screening["screen_eligible"] = (
        (screening["error_gain"] > 0)
        & (screening["abs_label"] >= float(cli.min_abs_label))
        & (screening["raw_word_count"] >= int(cli.min_raw_words))
        & (screening["raw_word_count"] <= int(cli.max_raw_words))
        & (screening["visual_active_steps"] >= int(cli.min_visual_steps))
    )
    screening["sentiment_words"] = ""
    screening["sentiment_word_count"] = 0
    screening["interaction_evaluated"] = False

    eligible = screening.loc[screening.screen_eligible].sort_values(
        ["error_gain", "sample_index"],
        ascending=[False, True],
        kind="mergesort",
    )

    # Inspect *all* eligible text labels to avoid inadvertently excluding an
    # example just because a top-k prediction-gain subset lacked a cue word.
    selected_indices = []
    for _, row in eligible.iterrows():
        idx = int(row.sample_index)
        _, words, _ = bert_word_groups(baseline, dataset[idx]["text"])
        _, sentiment = lexical_indices(words)
        if not sentiment:
            continue

        names = ", ".join(str(words[i]) for i in sentiment)
        screening.loc[
            screening.sample_index == idx, "sentiment_words"
        ] = names
        screening.loc[
            screening.sample_index == idx, "sentiment_word_count"
        ] = int(len(sentiment))

        if len(selected_indices) < int(cli.contrast_pool_size):
            selected_indices.append(idx)

    if selected_indices:
        screening.loc[
            screening.sample_index.isin(selected_indices),
            "interaction_evaluated",
        ] = True

    return screening, selected_indices


def _visual_case(cli, sample):
    """Resolve this sample's exact utterance clip without cross-case reuse."""
    if cli.no_video_frames:
        return None
    clip, _, _, _, _ = resolve_mosi_clip_path(
        sample["id"], cli.mosi_raw_root
    )
    return clip


def run_interaction_contrast(
    cli,
    cfg,
    dataset,
    baseline,
    cfcompat,
    candidates,
    baseline_ckpt,
    cfcompat_ckpt,
):
    if cli.split != "valid":
        raise ValueError("interaction_contrast is validation-only.")
    if cli.sample_index is not None:
        raise ValueError(
            "--sample-index is for manual single-case plots, not contrast ranking."
        )
    if cli.video_file:
        raise ValueError(
            "--video-file would incorrectly reuse a clip for multiple candidates. "
            "Use --mosi-raw-root for automatic per-example video resolution."
        )
    for key in ("contrast_pool_size", "contrast_topk", "contrast_render_top"):
        if int(getattr(cli, key)) < (0 if key == "contrast_render_top" else 1):
            raise ValueError("--{} has an invalid value.".format(
                key.replace("_", "-")
            ))

    output_dir = (
        Path(cli.output_dir) if cli.output_dir else
        DEFAULT_OUTPUT / cli.dataset / "seed{}".format(cli.seed)
        / "interaction_contrast"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    screening, indices = _first_stage(cli, dataset, candidates, baseline)
    screening_csv = output_dir / "validation_screening.csv"
    screening.to_csv(screening_csv, index=False)

    if not indices:
        raise RuntimeError(
            "No validation examples satisfy positive gain, label/text/visual "
            "filters and the explicit sentiment lexicon. Saved screening CSV "
            "at {}. Lower the screening thresholds or expand the documented "
            "lexicon; do not use Test for case selection.".format(screening_csv)
        )

    print("[Contrast] Validation candidate pool: {} / {}".format(
        len(indices), len(candidates)
    ))
    print("[Contrast] Screening CSV: {}".format(screening_csv))

    records = []
    cache = {}
    for rank, idx in enumerate(indices, start=1):
        sample = dataset[idx]
        word_groups, word_labels, mask_token_id = bert_word_groups(
            baseline, sample["text"]
        )
        windows, window_labels, centers, active_visual = visual_windows(
            sample["vision"], cli.visual_bins
        )

        print("[Contrast] Interaction {}/{} | sample={} | words={} | windows={}".format(
            rank, len(indices), idx, len(word_groups), len(windows)
        ), flush=True)

        b_preds = run_occlusion_variants(
            baseline, sample, cli.condition,
            word_groups, mask_token_id, windows,
            cli.occlusion_batch_size, cfg.device,
        )
        o_preds = run_occlusion_variants(
            cfcompat, sample, cli.condition,
            word_groups, mask_token_id, windows,
            cli.occlusion_batch_size, cfg.device,
        )
        b_pack = interaction_from_predictions(
            b_preds, len(word_groups), len(windows)
        )
        o_pack = interaction_from_predictions(
            o_preds, len(word_groups), len(windows)
        )

        metrics = interaction_metrics(b_pack, o_pack, word_labels)
        row = candidates.loc[
            candidates.sample_index.astype(int) == int(idx)
        ].iloc[0].to_dict()
        records.append({
            **row,
            **metrics,
            "pre_screen_gain_rank": int(rank),
        })

        # Retain only the small screened pool; candidates share no tensors in
        # the cache and no checkpoints/model parameters are modified.
        cache[int(idx)] = {
            "sample": sample,
            "word_labels": word_labels,
            "windows": windows,
            "window_labels": window_labels,
            "centers": centers,
            "active_visual": active_visual,
            "baseline_pack": b_pack,
            "cfcompat_pack": o_pack,
        }

    ranked = pd.DataFrame.from_records(records)

    # Fixed, equally weighted standardization across the WHOLE evaluated pool.
    # The score is for selecting a qualitative illustration, not test tuning.
    for metric, prefix in (
        ("error_gain", "z_prediction_gain"),
        ("sentiment_ratio_gain", "z_sentiment_gain"),
        ("map_contrast_tv", "z_map_contrast"),
    ):
        ranked[prefix] = _standardize(ranked[metric].to_numpy())

    ranked["contrast_score"] = (
        ranked["z_prediction_gain"]
        + ranked["z_sentiment_gain"]
        + ranked["z_map_contrast"]
    )
    ranked["positive_sentiment_shift"] = (
        ranked["sentiment_ratio_gain"] > 0.0
    )

    ranked = ranked.sort_values(
        ["contrast_score", "sentiment_ratio_gain", "sample_index"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["overall_rank"] = np.arange(1, len(ranked) + 1)

    ranked_csv = output_dir / "interaction_contrast_all_evaluated.csv"
    ranked.to_csv(ranked_csv, index=False)

    # Prefer positive sentiment concentration, which is the advertised
    # qualitative phenomenon. Never silently treat a negative gain as positive.
    eligible_figures = ranked.loc[ranked.positive_sentiment_shift].copy()
    eligible_figures = eligible_figures.sort_values(
        ["contrast_score", "sentiment_ratio_gain", "sample_index"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    eligible_figures["selection_rank"] = np.arange(
        1, len(eligible_figures) + 1
    )

    shortlist = eligible_figures.head(int(cli.contrast_topk)).copy()
    shortlist_csv = output_dir / "interaction_contrast_top_candidates.csv"
    shortlist.to_csv(shortlist_csv, index=False)

    protocol = {
        "dataset": cli.dataset,
        "split": cli.split,
        "seed": int(cli.seed),
        "condition": cli.condition,
        "baseline": cli.baseline,
        "baseline_checkpoint": str(baseline_ckpt),
        "cfcompat_checkpoint": str(cfcompat_ckpt),
        "prediction_gain_formula": (
            "abs(label-uniform_prediction)-abs(label-cfcompat_prediction)"
        ),
        "sentiment_concentration": (
            "raw interaction mass on the fixed DISPLAY_SENTIMENT_WORDS lexicon "
            "divided by raw interaction mass on all non-stopword content words"
        ),
        "map_contrast": (
            "0.5*L1 distance between the two interaction maps normalized to "
            "sum 1 over identical content-word rows and visual-window columns; "
            "range [0,1]"
        ),
        "score": (
            "z(prediction_gain)+z(sentiment_ratio_gain)+z(map_contrast_tv), "
            "with population z-scores computed over every stage-two sample"
        ),
        "screening": {
            "positive_prediction_gain": True,
            "minimum_absolute_label": float(cli.min_abs_label),
            "minimum_raw_words": int(cli.min_raw_words),
            "maximum_raw_words": int(cli.max_raw_words),
            "minimum_active_visual_steps": int(cli.min_visual_steps),
            "minimum_sentiment_cues": 1,
            "order": "decreasing prediction gain, then sample index",
            "pool_size_limit": int(cli.contrast_pool_size),
        },
        "figure_eligibility": "sentiment_ratio_gain > 0",
        "topk": int(cli.contrast_topk),
        "render_top": int(cli.contrast_render_top),
        "evaluated_candidates": int(len(ranked)),
        "eligible_figure_candidates": int(len(eligible_figures)),
        "limitation": (
            "Heuristic, post-hoc, validation-only qualitative case selection. "
            "Lexicon coverage is limited. A contrast case is not population-"
            "level evidence or proof of temporal word--frame alignment."
        ),
    }
    protocol_path = output_dir / "interaction_contrast_protocol.json"
    protocol_path.write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    if shortlist.empty:
        warnings.warn(
            "No evaluated case exhibits positive sentiment-cue concentration "
            "gain. All evaluated metrics are saved, but no favorable visual "
            "case will be generated."
        )
    else:
        print("\n[Contrast] Top candidates:")
        cols = [
            "selection_rank", "sample_index", "error_gain",
            "sentiment_ratio_gain", "map_contrast_tv", "contrast_score",
            "sentiment_words",
        ]
        print(shortlist[cols].to_string(index=False, float_format=lambda x: "{:.4f}".format(x)))

    for _, row in shortlist.head(int(cli.contrast_render_top)).iterrows():
        idx = int(row.sample_index)
        item = cache[idx]
        clip = _visual_case(cli, item["sample"])
        print("[Contrast] Rendering rank {} (sample {})".format(
            int(row.selection_rank), idx
        ), flush=True)

        png, pdf, npz, plot_meta = plot_figure(
            cli, row, item["sample"], item["word_labels"],
            item["windows"], item["window_labels"], item["centers"],
            item["active_visual"], clip, item["baseline_pack"],
            item["cfcompat_pack"], output_dir,
        )

        metadata = {
            "selection": "interaction_contrast",
            "sample_index": idx,
            "sample_id": str(_sample_id_to_python(item["sample"]["id"])),
            "selection_rank": int(row.selection_rank),
            "ranked_candidate_metrics": {
                key: (
                    bool(row[key]) if isinstance(row[key], (bool, np.bool_))
                    else float(row[key]) if isinstance(row[key], (float, np.floating))
                    else int(row[key]) if isinstance(row[key], (int, np.integer))
                    else str(row[key])
                )
                for key in [
                    "error_gain", "uniform_sentiment_ratio",
                    "cfcompat_sentiment_ratio", "sentiment_ratio_gain",
                    "map_contrast_tv", "contrast_score",
                    "z_prediction_gain", "z_sentiment_gain", "z_map_contrast",
                    "sentiment_words",
                ]
            },
            "raw_video": None if clip is None else str(clip),
            "interaction_formula": "abs(f(x_-i,-j)-f(x_-i)-f(x_-j)+f(x))",
            "selection_protocol_file": str(protocol_path),
            **plot_meta,
        }
        metadata_path = output_dir / (
            "contrast_rank{:02d}_sample{}_metadata.json".format(
                int(row.selection_rank), idx
            )
        )
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print("[Contrast] {} | {}".format(png, pdf))
        print("[Contrast] raw matrices: {}".format(npz))

    print("\n" + "=" * 84)
    print("Interaction-contrast validation search complete")
    print("All validation screening : {}".format(screening_csv))
    print("Full stage-two ranking   : {}".format(ranked_csv))
    print("Top candidates           : {}".format(shortlist_csv))
    print("Selection protocol       : {}".format(protocol_path))
    print("Figures directory        : {}".format(output_dir))
    print("=" * 84)
