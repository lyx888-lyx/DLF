# Stage 2 Fixed-Weight Prediction Knowledge Distillation Protocol

## Scope

This branch implements Stage 2 only: fixed-weight, prediction-level knowledge
distillation for MOSI missing-modality regression.  It deliberately excludes
feature KD, attention KD, auxiliary heads, KL/temperature losses, dynamic or
reliability weighting, sensitivity branches, and all Stage 3 work.

No command in `train_fixed_kd.py` reads a test split.  The fixed-KD test
guard in `eval_missing.py` still requires an explicit one-time confirmation,
but Stage 2 training and smoke validation use train/valid only.

## Initialization and model ownership

For each seed, both models are initialized strictly from the same immutable
Gate 3 clean checkpoint:

```
pt/DLF_mosi_seed{seed}_best.pth
```

The teacher is a standalone plain `DLF` model.  It loads the clean checkpoint
strictly, is set to `eval()`, has every parameter frozen, is excluded from
the student optimizer, and is never stored in a student checkpoint.

The student is the Stage 1 `MissingModalityWrapper` around a separately
constructed `DLF` backbone loaded strictly from that same clean checkpoint.
Thus it retains the Stage 1 DirectMask/ModDrop behavior, mask tokens, adapter,
and residual integration.  Before the first update, teacher LAV and student
LAV predictions are checked with `torch.testing.assert_close`.

Teacher construction preserves and restores Python, NumPy, Torch, and CUDA RNG
states.  This makes teacher setup invisible to the Stage 1 random sequence.

## Fixed objective

The only new term is prediction KD on the final regression output:

```
L_predKD = SmoothL1(student_missing.output_logit,
                    teacher_full_LAV.output_logit.detach())
L_total = L_full + 1.0 * L_missing + 1.0 * L_predKD
```

`eta` and `lambda_kd` are both fixed at 1.0.  The CLI rejects any other
value; tuning is intentionally not supported.  Teacher LAV forwards run only
under `torch.inference_mode()`.  A narrow cache cleanup after each teacher
forward avoids leaking an inference-mode tensor into DLF's existing shared
sinusoidal-position cache; it does not modify the DLF model implementation.

The training loop preserves the Stage 1 order: student full LAV forward,
student sampled missing-mode forward, then teacher LAV prediction.  Missing
masks use `torch.Generator().manual_seed(seed + 104729)`, exactly as Stage 1.

## Runtime checks and validation

Each run verifies that teacher parameters stay frozen, remain absent from the
optimizer, and have no gradients after backward.  On the first batch of every
epoch it also verifies a nonzero gradient path from the KD loss into student
parameters.  Student regular and mask-token gradient paths are checked.

Selection, scheduler stepping, and early stopping use only the Stage 1
student-only validation modes LAV/LA/LV/L and the same J_val computation.
Teacher comparisons are diagnostic only: `Gap_LA`, `Gap_LV`, and `Gap_L`
on validation plus the train missing-mode gap.  They never affect checkpoint
selection, learning rate, or stopping.

## Outputs

Formal checkpoints and train results are isolated from Stage 1:

```
pt/missing_baseline/fixed_kd/DLF_mosi_seed{seed}_best.pth
result/missing_baseline/fixed_kd/train/
log/missing_baseline/DLF-mosi-fixedkd-train-*.log
```

Smoke outputs use the corresponding `fixed_kd/smoke/` checkpoint and result
subdirectories and `DLF-mosi-fixedkd-smoke-*.log` logs.  A fixed-KD
checkpoint contains only the student wrapper state dict.

`eval_missing.py --method fixedkd` loads only this student checkpoint, wraps
it with `MissingModalityWrapper`, and evaluates its four LAV/LA/LV/L modes.
It does not construct or load a teacher.

## Reproducible commands

Run the required two-epoch smoke validation before the formal run:

```bash
/usr/miniconda3/envs/DLF/bin/python train_fixed_kd.py \
  --dataset mosi --seeds 1111 --eta 1.0 --lambda-kd 1.0 \
  --smoke-test --max-epochs 2 --num-workers 0 --gpu-ids 0
```

After the smoke checks, commit and push the implementation, then start the
formal seed-1111 train with the same fixed weights and no smoke flag.  Do not
run a test-split evaluation as part of Stage 2.
