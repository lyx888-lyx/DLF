# Stage 3B-M: CFCompatKD-v1 Five-Seed Paired Replication

This experiment freezes `DLF-CFCompatKD-v1` at base commit
`d6adc7b170c8b7cc13df72dea93b86458c9ae36e`. It uses seeds 1111–1115 and
compares `CFCompatKD(seed=s)` only with `ModDrop(seed=s)`.

Seed 1111 is a locked historical result. The Stage 3B-M runner rejects seed1111
for every training/cache action and records it as `locked_existing` with source
`reused_locked_existing_result`.

## Frozen methods

- Gate 3 teacher/student initialization: each seed's own validation-best plain
  DLF checkpoint.
- ModDrop: `L_full + L_missing`, task-head weights 1,1,3,1,1, no KD/LDS/new loss.
- Missing-mask RNG: `torch.Generator().manual_seed(seed + 104729)`.
- Counterfactual evaluator: the same seed's ModDrop validation-best checkpoint.
- Cache: train-only, 1,284 unique samples; stable average rank independently for
  LA/LV/L; `q=(rank-0.5)/N`; `compat=1-q`.
- CFCompatKD: `gate_mode=compat`, SmoothL1 prediction KD,
  `sum(compat_i*kd_i)/(sum(compat_i)+1e-8)`, `eta=lambda_kd=1`.
- Main checkpoint selection: validation J only. Test-best checkpoints are
  separate and diagnostic-only.

No temperature, threshold, reliability, recoverability, residual
reconstruction, diffusion, feature KD, rank change, or hyperparameter tuning is
permitted.

## Required order

1. Audit Gate 3 checkpoints for seeds 1111–1115.
2. Smoke seed1112: two-epoch ModDrop, train-only cache, two-epoch CFCompatKD.
3. Commit and push the frozen implementation.
4. Formal Phase B: ModDrop controls 1112, 1113, 1114, 1115.
5. Formal Phase C: caches 1112, 1113, 1114, 1115.
6. Formal Phase D: CFCompatKD 1112, 1113, 1114, 1115.
7. Strict paired aggregation and A/B/C/D classification.

Formal training uses one seed, one process, and one GPU process at a time. Once
formal seed1112 starts, metrics cannot trigger code, formula, hyperparameter,
seed-order, early-stop, or rerun changes.

## Commands

```bash
python run_cf_compat_multiseed.py --action audit-gate3
python run_cf_compat_multiseed.py --action moddrop --seed 1112 --smoke-test --gpu-ids 0
python run_cf_compat_multiseed.py --action cache --seed 1112 --smoke-test --gpu-ids 0
python run_cf_compat_multiseed.py --action cfcompat --seed 1112 --smoke-test --gpu-ids 0

python run_cf_compat_multiseed.py --action moddrop --seed SEED --gpu-ids 0
python run_cf_compat_multiseed.py --action cache --seed SEED --gpu-ids 0
python run_cf_compat_multiseed.py --action cfcompat --seed SEED --gpu-ids 0
python aggregate_cf_compat_multiseed.py
```

`SEED` is run sequentially as 1112, 1113, 1114, 1115 in each formal phase.
Stage 3C/recoverability is not part of this protocol and has not started.
