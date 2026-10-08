# Validation-only interaction-contrast case selection

This script extends the existing MOSI qualitative visualizer, without any
retraining, model modification, or Test-set searching.

## Pull and run

\`\`\`bash
cd /code/DLF
git switch analysis/compatibility-gain-v1
git pull origin analysis/compatibility-gain-v1

CUDA_VISIBLE_DEVICES=0 python visualize_crossmodal_interaction.py \
  --dataset mosi \
  --seed 1114 \
  --split valid \
  --condition LV \
  --baseline fixedkd \
  --selection interaction_contrast \
  --contrast-pool-size 24 \
  --contrast-topk 10 \
  --contrast-render-top 3 \
  --visual-bins 8 \
  --max-words 8 \
  --gpu-ids 0
\`\`\`

The default raw-video root is:

\`\`\`text
/sharefile/lyx_model/MMSA_new/MOSI/Raw/Raw
\`\`\`

The tool automatically resolves each selected sample's exact utterance clip.
Do not provide a single \`--video-file\` when automatically rendering multiple
samples: that would associate the wrong video with some examples.

## Two-stage, predeclared screening

**Stage 1: cheap prediction scan across validation.** Keep examples satisfying
all of:

- CFCompat absolute error is lower than Uniform KD absolute error
  (\`error_gain > 0\`);
- absolute sentiment label >= \`--min-abs-label\` (default 1);
- \`--min-raw-words\` <= text length <= \`--max-raw-words\`
  (default 10 to 24);
- at least \`--min-visual-steps\` non-padding visual feature steps
  (default 10);
- at least one recognizable sentiment cue in the fixed
  \`DISPLAY_SENTIMENT_WORDS\` lexicon.

Sort by prediction gain descending, break ties by sample index, and retain up
to \`--contrast-pool-size\` (default 24). The full prediction-level screening
is saved in \`validation_screening.csv\`.

**Stage 2: raw pairwise occlusion interactions.** For each screened example,
compute both models' word x visual-context interaction maps using:

\`\`\`text
I[i,j] = abs(f(x_without_word_i_and_visual_window_j)
             - f(x_without_word_i)
             - f(x_without_visual_window_j)
             + f(x))
\`\`\`

No display gamma, color mapping, manual word highlighting, or top-8 display
filter is involved in scoring.

Three metrics are calculated:

1. **Prediction gain**:
   \`abs(y - prediction_uniform) - abs(y - prediction_CFCompat)\`.
2. **Sentiment concentration gain**:
   \`sentiment_ratio_CFCompat - sentiment_ratio_uniform\`. Each ratio is the
   fraction of **raw** interaction magnitude on the fixed sentiment-cue
   lexicon among all content words (stopwords excluded from the denominator).
3. **Map contrast**: the total-variation distance (0 to 1) between the
   two interaction matrices L1-normalized on identical content-word rows and
   visual-context-window columns.

The score is:

\`\`\`text
z(prediction_gain) + z(sentiment_concentration_gain) + z(map_contrast)
\`\`\`

Each population z-score is calibrated over the entire Stage-2 pool.
Equal-weighted terms and thresholds are fixed before examining the ranked
cases. We only put samples with a **positive sentiment-concentration gain**
onto the recommended figure shortlist. If none qualify, the script emits all
diagnostics and no figure rather than inventing a favorable example.

## Outputs

Default directory:

\`\`\`text
result/analysis/crossmodal_interaction_v1/mosi/seed1114/interaction_contrast/
\`\`\`

- \`validation_screening.csv\`: all validation samples and Stage-1 decisions.
- \`interaction_contrast_all_evaluated.csv\`: every Stage-2 example, with raw
  metrics, standardized terms and full rank.
- \`interaction_contrast_top_candidates.csv\`: up to 10 eligible candidates
  with positive sentiment-concentration gain.
- \`interaction_contrast_protocol.json\`: exact ranking formulas, thresholds,
  model checkpoints and limitations.
- PNG/PDF/NPZ for the top \`--contrast-render-top\` (default 3) examples, plus
  per-figure JSON metadata with the actual scores.

To compute a shortlist without rendering figures:

\`\`\`bash
CUDA_VISIBLE_DEVICES=0 python visualize_crossmodal_interaction.py \
  --selection interaction_contrast \
  --contrast-render-top 0 \
  --gpu-ids 0
\`\`\`

If more qualitative candidates are needed, change
\`--contrast-pool-size 24\` to \`--contrast-pool-size 36\`. This raises the
number of expensive BERT occlusion forwards; it is not a model hyperparameter.

## Important interpretation

These are **validation-selected qualitative examples**, not unbiased
estimates of how often CFCompat redistributes interactions beneficially.
They should not be used to tune the model, choose a checkpoint on Test, or
claim universal sentiment-word alignment. The interaction heatmap reports a
non-additive predictive response to joint word-and-visual-window occlusion; it
does **not** show word duration, literal attention weights, or temporal
word--frame synchronization.

For publication, disclose the validation-only case-selection criterion and
show at least one alternative candidate if space permits. Aggregated transfer
utility and task metrics remain the appropriate main quantitative evidence.
