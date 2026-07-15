# Stage 2.6: Train-only Label Density Smoothing for DLF-ModDrop

Method name: `DLF-LDS-ModDrop-v1`.

This branch starts from the Stage 1 `DLF-ModDrop` implementation.  The model,
missing audio and vision tokens, zero-initialized mask adapter, LAV view, and
independent LA/LV/L sample-wise missing view are unchanged.  It adds exactly
one variable: a fixed train-only LDS-v1 weight applied to the original
five-head L1 task loss.

## Fixed LDS-v1 construction

Only the MOSI training labels may construct the density map.  Labels must be in
`[-3, 3]`; values outside that range fail rather than being clipped.  The fixed
bin edges are `np.linspace(-3.0, 3.0, 61)`, producing 60 bins of width 0.1.
Bins are left closed and right open, except that the final bin includes `3.0`.

The train count vector is smoothed by the normalized Gaussian kernel with
positions `[-2,-1,0,1,2]`, radius 2, size 5, and sigma 1.0, using
`np.convolve(..., mode="same")`.  For a sample's smoothed train-bin density
`d`, its raw weight is `(d + 1e-6)**-0.5`.  Raw sample weights are normalized to
mean one, clipped to `[0.2, 5.0]`, then normalized again to mean one.  The
implementation rejects a configuration mismatch, non-finite or non-positive
weights, a final mean farther than `1e-6` from one, fewer than three distinct
weights, or more than 20% of samples clipping at the upper bound.

Weights are precomputed in the immutable `MMDataset` train-index order.  A
training batch gathers them with its returned `index` field, never from batch
position, and verifies every train index occurs exactly once per epoch.  The
Stage 1 shuffled DataLoader and the separately seeded
`torch.Generator().manual_seed(seed + 104729)` missing-mode sampler are
unchanged.

## Objective and validation

For every sample, the task vector is:

`|output_logit-y| + |logits_c-y| + 3|logits_l_hetero-y| + |logits_v_hetero-y| + |logits_a_hetero-y|`.

The reduction is `sum(weight * task_vector) / sum(weight)`.  The same one LDS
weight is used for all five heads.  In the full LAV view, only this task loss is
replaced: reconstruction, specific reconstruction, orthogonality, similarity,
and their Stage 1 coefficients are unchanged.  In the missing view, there are
no auxiliary losses.  The fixed objective is:

`L_total = L_full_LDS + 1.0 * L_missing_LDS`.

Validation remains unweighted and uses the unchanged four mode metric protocol
and checkpoint objective:

`J_val = 0.5 * MAE_LAV + 0.5 * (MAE_LA + MAE_LV + MAE_L) / 3`.

Five fixed emotion-bin diagnostics and train-weight-quartile density-group
diagnostics are report-only.  The latter maps validation labels through the
already fixed training density map and never estimates a validation density.

## Strict split boundary

The density audit builds one `MMDataset(..., mode="train")` and no model,
optimizer, batches, validation, or held-out loader.  Training builds exactly
train and validation loaders.  `eval_lds.py` is validation-only.  This stage
does not read held-out labels, predictions, audit CSVs, or results; it does not
contain a held-out evaluation option.

## Commands and isolated output

Train-label audit:

```bash
python train_lds.py --dataset mosi --seeds 1111 --audit-density-only
```

The audit writes `result/label_density/lds_v1/mosi/` with the fixed config,
60-bin table, train sample-weight summary, and fixed emotion-bin summary.

After static checks and the two-epoch smoke run, formal seed 1111 training is:

```bash
PYTHONUNBUFFERED=1 python train_lds.py --dataset mosi --seeds 1111 --eta 1.0
```

Formal checkpoints are isolated under
`pt/missing_baseline/lds_moddrop_v1/`; smoke checkpoints are under its `smoke/`
subdirectory.  Formal results are under
`result/missing_baseline/lds_moddrop_v1/train/`, with smoke output in `smoke/`.
No Stage 1, FixedKD, Gate 3, or prior audit output is overwritten.
