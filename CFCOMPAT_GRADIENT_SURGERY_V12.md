# CFCompatKD v12 — Asymmetric Gradient Surgery

## Motivation

The frozen v11 Train-OOF objective audit localized the v8-v10 Pareto tradeoff to residual-head gradient conflict rather than residual capacity, cross-fit variance, or late-epoch magnitude growth.

At the frozen v10 conservative checkpoints:

- supervised missing-label gradients harmed OOF Teacher-beneficial error in 5/5 folds;
- supervised Teacher-nonbeneficial gradients harmed OOF Teacher-beneficial error in 5/5 folds;
- selective DISTILL/PRESERVE gradients improved OOF Teacher-beneficial error in 5/5 folds;
- the active DISTILL/PRESERVE targets were locally safe on 100% of audited events;
- ABSTAIN still received supervised missing-label training.

v12 tests one specific mechanism hypothesis: can we preserve the supervised repair signal while removing only the part that directly conflicts with the selective safety signal?

## Single mechanism change relative to v10

Everything below is frozen to v10 unless explicitly stated:

- entire historical S0 function is frozen;
- exact v8 residual architecture: independent LA/LV/L MLP residual heads;
- five deterministic video-grouped Train folds;
- identical residual initialization across folds;
- v4 DISTILL/PRESERVE/ABSTAIN decisions and safe targets;
- `DISTILL_MARGIN=0.02`;
- `PRESERVE_MARGIN=0.02`;
- `LAMBDA_PRESERVE=0.25`;
- mild CFCompat weighting `0.75 + 0.25*compatibility`;
- Adam optimizer and learning rate;
- original `update_epochs=10` accumulation window;
- Train-video-holdout evaluation and early stopping;
- v10 conservative selector: earliest epoch within 1% of the absolute best Train-holdout J;
- 4-of-5 same-sign median residual consensus;
- official Valid first used only after all five fold banks are frozen;
- official Test is never constructed or accessed.

The only intervention is the residual optimizer gradient at each original update window.

## Gradient decomposition

For the residual bank, the complete-LAV objective has zero gradient because the residual heads are inactive for LAV. The missing-view objective is decomposed into:

```text
g_sup = gradient of supervised missing task loss

g_sel = gradient of KD_loss + 0.25 * preserve_loss
```

Both are accumulated across exactly the same mini-batches that v10 would have accumulated before one Adam step.

## Asymmetric surgery

If the accumulated gradients agree or are orthogonal, nothing changes:

```text
if dot(g_sup, g_sel) >= 0:
    g_sup_safe = g_sup
```

If they conflict:

```text
if dot(g_sup, g_sel) < 0:
    g_sup_safe = g_sup - dot(g_sup, g_sel) / ||g_sel||^2 * g_sel
```

Then:

```text
g_update = g_sup_safe + g_sel
```

Only `g_sup` is modified. `g_sel` is never projected or rescaled by the surgery.

This is deliberately asymmetric. v11 showed that supervised missing loss is the source that repairs Teacher-nonbeneficial cases but conflicts with Teacher-beneficial protection, while selective KD/PRESERVE has the opposite alignment. The experiment therefore asks whether removing only the explicitly conflicting supervised component can reduce the Pareto conflict without discarding supervised repair.

## What v12 does not do

v12 does not:

- remove supervised missing loss;
- change the residual architecture;
- increase residual capacity;
- use a learned usefulness gate;
- use Teacher-beneficial labels at inference;
- use OOF labels to route inference;
- tune gradient thresholds;
- tune the projection coefficient;
- change the 1% conservative selector;
- change the consensus rule;
- use official Valid for fold checkpoint selection;
- construct or access official Test.

The conflict threshold is exactly zero and the projection coefficient is analytically determined by the two gradients.

## Frozen promotion criteria

The same mechanism gates used for v9/v10 remain frozen:

- Valid J degradation relative to v8 <= 0.002;
- Teacher-beneficial NTR degradation relative to v8 <= 0.03;
- Teacher-nonbeneficial NTR reduction relative to v8 >= 0.05;
- overall NTR degradation relative to v4 <= 0.01.

No threshold is relaxed after seeing v12.

## Diagnostics

`gradient_surgery_v12_update_windows.csv` records every optimizer update window:

- pre-surgery supervised/selective dot product;
- pre-surgery cosine;
- conflict flag;
- supervised and selective gradient L2 norms;
- projection coefficient;
- projected supervised L2 norm;
- final update L2 norm;
- post-projection dot product;
- fraction of supervised L2 removed by surgery.

The fold manifest additionally records aggregate conflict rates and surgery strength.

## Interpretation

A positive mechanism signal would mean that directly removing measured residual-objective gradient conflict can retain the v8 beneficial safety while preserving a meaningful amount of the nonbeneficial repair seen in v9/v10.

A negative signal would be informative too. It would imply that the v11 local gradient conflict is not sufficient to explain the trajectory-level tradeoff, or that selective-gradient preservation alone is not an adequate function-level safety anchor.
