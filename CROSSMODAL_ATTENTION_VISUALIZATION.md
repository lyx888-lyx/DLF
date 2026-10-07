# Figure-4-style cross-modal attention visualization

This analysis compares the same validation sample under the same LV target
condition using:

1. FixedKD (Uniform KD), and
2. CFCompat.

The visualization reads the **actual head-averaged DLF L<-V cross-modal
attention weights** from the selected Transformer layer. It does not use
gradient saliency or a fabricated relevance matrix.

## Why LV?

The figure visualizes text--vision attention. LV keeps both text and vision
observable while moving the student away from the reference LAV condition.
LA/L are therefore intentionally not used for this Figure-4-style view.

## Default case-selection protocol

Automatic selection is validation-only.

The script first keeps samples with:

- positive CFCompat absolute-error improvement over Uniform KD,
- |sentiment label| >= 1,
- 4--22 raw words.

The default `representative` rule then selects the sample whose positive
error gain is closest to the median eligible gain. This avoids choosing the
single most favorable example. Use `--selection largest_gain` only for
diagnostic exploration, not as the default paper protocol.

A test sample is allowed only when `--sample-index` is explicitly supplied;
automatic test-set case selection is rejected.

## Pull

```bash
cd /code/DLF
git switch analysis/compatibility-gain-v1
git pull origin analysis/compatibility-gain-v1
```

## Recommended run

```bash
CUDA_VISIBLE_DEVICES=0 python visualize_crossmodal_attention.py \
  --dataset mosi \
  --seed 1114 \
  --split valid \
  --condition LV \
  --baseline fixedkd \
  --selection representative \
  --gpu-ids 0
```

Default checkpoints:

```text
pt/missing_baseline/fixed_kd/DLF_mosi_seed1114_best.pth

pt/missing_baseline/cf_compat_kd_v1/benchmark_multiseed/seed1114/
DLF_mosi_seed1114_best_valid.pth
```

If your checkpoint paths differ:

```bash
CUDA_VISIBLE_DEVICES=0 python visualize_crossmodal_attention.py \
  --dataset mosi \
  --seed 1114 \
  --split valid \
  --condition LV \
  --baseline fixedkd \
  --baseline-checkpoint /path/to/fixedkd.pth \
  --cfcompat-checkpoint /path/to/cfcompat.pth \
  --gpu-ids 0
```

## Add real video frames

The aligned feature pickle does not contain raw RGB frames. If you have the
**exact utterance-level video clip** for the selected sample, rerun with:

```bash
CUDA_VISIBLE_DEVICES=0 python visualize_crossmodal_attention.py \
  --dataset mosi \
  --seed 1114 \
  --split valid \
  --condition LV \
  --baseline fixedkd \
  --sample-index <SELECTED_INDEX> \
  --video-file /path/to/exact_utterance_clip.mp4 \
  --gpu-ids 0
```

This requires `opencv-python`. Frames are sampled at the normalized centers
of the visual attention bins. Use only an utterance-level clip whose temporal
extent matches the processed sample.

Without `--video-file`, the figure uses clearly labelled visual-window
placeholders rather than pretending that processed visual features are images.

## Outputs

By default:

```text
result/analysis/crossmodal_attention_v1/mosi/seed1114/
```

The directory contains:

- publication PNG and PDF,
- `*_attention.npz` with raw/display attention matrices,
- `*_candidate_summary.csv` showing the transparent selection procedure,
- `*_metadata.json` with checkpoints, predictions, sample ID, and plotted
  token/window indices.

## Interpretation

Rows are text query positions after DLF's temporal projection. Columns are
contiguous visual key windows. The displayed matrix is derived from the
head-averaged attention returned by the actual `MultiheadAttention` module in
the selected `trans_l_with_v` layer.

The two panels share one color scale. Therefore visual differences should be
interpreted as changes in cross-modal attention allocation, not as separately
renormalized color maps.

The three strongest CFCompat query rows are highlighted in dark red only to
guide reading; this highlighting does not alter the attention values.

## Paper wording

A safe description is:

> We further visualize the text--vision cross-modal attention of Uniform KD
> and CFCompat on the same shifted validation example. The maps are extracted
> from the final L<-V attention layer using the same color scale. CFCompat
> exhibits a more concentrated allocation around sentiment-relevant textual
> queries and corresponding visual windows.

Only use the last sentence if the generated figure actually supports it.


## PMR-style raw MOSI frames

The current interaction visualizer automatically looks for utterance-level
MOSI clips under:

```text
/sharefile/lyx_model/MMSA_new/MOSI/Raw/Raw
```

using the sample ID stored in the processed feature file. The expected layout
is:

```text
<raw-root>/<video_id>/<segment_id>.mp4
```

For example:

```text
/sharefile/lyx_model/MMSA_new/MOSI/Raw/Raw/1DmNV9C1hbY/7.mp4
```

Run:

```bash
CUDA_VISIBLE_DEVICES=0 python visualize_crossmodal_interaction.py \
  --dataset mosi \
  --seed 1114 \
  --split valid \
  --condition LV \
  --baseline fixedkd \
  --selection representative \
  --visual-bins 8 \
  --max-words 14 \
  --display-gamma 0.45 \
  --gpu-ids 0
```

No additional video argument is required when the raw root above is correct.
Use `--mosi-raw-root /another/path` to override it, or `--video-file` to
force one exact utterance clip. The script never guesses a neighboring segment
number if the ID-to-file mapping fails.

The figure now uses separate columns for method metadata and word labels, so
`(a)/(b)`, prediction/AE values, and text tokens cannot overlap. Real video
frames are shown once above the heatmaps, one frame per displayed visual
window, following the presentation style of PMR Figure 4. Frame positions are
mapped over the non-padding visual support rather than the padded length of 50.
