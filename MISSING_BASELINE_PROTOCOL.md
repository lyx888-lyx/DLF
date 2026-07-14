# Stage 1 Missing-Modality Baseline Protocol

This branch implements only the Stage 1 baselines. It does not add a teacher,
knowledge distillation, diffusion, modality reconstruction, long-tail weighting,
Balanced MSE, SIMS, MOSEI work, or any later-stage module.

## Fixed modality protocol

The only order is [text, audio, vision].

| Mode | Mask | Inputs |
| --- | --- | --- |
| LAV | [1, 1, 1] | text, audio, vision |
| LA | [1, 1, 0] | text, audio |
| LV | [1, 0, 1] | text, vision |
| L | [1, 0, 0] | text only |

Text is always present. Stage 1 does not implement text loss, temporal loss,
partial-time loss, arbitrary modality combinations, or missing-rate sweeps.

## B1: DLF-DirectMask

eval_missing.py with --method directmask loads a Gate 3 validation-best checkpoint
and makes no structural change to DLF. LA zeroes the complete vision input, LV
zeroes the complete audio input, and L zeroes both. It never uses missing tokens
or retrains the model.

Validation example:

    python eval_missing.py --dataset mosi --seeds 1111 --method directmask --split valid

The CSV files are written under result/missing_baseline/directmask/valid/.

## B2: DLF-ModDrop

train_missing.py first loads the corresponding Gate 3 validation-best checkpoint,
then adds only:

- missing_audio_token with shape [1, 1, audio_feature_dim];
- missing_vision_token with shape [1, 1, vision_feature_dim];
- a bias-free Linear(3, final_fusion_dim) adapter initialized to zero.

The adapter receives 1 - modality_mask and is injected residually into the
final DLF fusion vector immediately before the final prediction head. Therefore
the complete LAV path receives an exactly zero adapter contribution. DirectMask
does not use this wrapper.

Each training sample receives one independently sampled LA, LV, or L missing
view, each with probability one third. The sampling generator is explicitly
seeded from the actual seed and is distinct from DataLoader shuffling.

The complete view uses the unchanged original DLF combined loss. The missing
view uses only the original five-head task loss, with the original weights
1, 1, 3, 1, 1. The fixed objective is:

    L_total = L_full + 1.0 * L_missing

No reconstruction, specific reconstruction, orthogonality, or similarity loss
is applied to the missing view.

## Validation and test-once guard

Training creates only train and valid loaders. At every epoch it evaluates LAV,
LA, LV, and L on validation and selects the checkpoint by:

    J_val = 0.5 * MAE_LAV + 0.5 * (MAE_LA + MAE_LV + MAE_L) / 3

The scheduler and early stopping also use only J_val. Metrics are written in raw
scale, for example 0.83 rather than 83.

Official test evaluation is refused unless the command contains both --split test
and --confirm-test-once. Do not use that option during this development task.

## Smoke test and formal run

Smoke test, limited to two epochs and isolated output directories:

    python train_missing.py --dataset mosi --seeds 1111 --eta 1.0 --mode moddrop --smoke-test --max-epochs 2

After the smoke test and all tests pass, formal MOSI seed 1111 training is:

    mkdir -p log/missing_baseline
    PYTHONUNBUFFERED=1 python train_missing.py --dataset mosi --seeds 1111 --eta 1.0 --mode moddrop \
      2>&1 | tee "log/missing_baseline/DLF-mosi-moddrop-seed1111-$(date +%Y%m%d-%H%M%S).log"

Formal validation-best checkpoints go to:

    pt/missing_baseline/moddrop/DLF_mosi_seed1111_best.pth

Smoke checkpoints are isolated under pt/missing_baseline/moddrop/smoke/ and must
not be reported as formal results.
