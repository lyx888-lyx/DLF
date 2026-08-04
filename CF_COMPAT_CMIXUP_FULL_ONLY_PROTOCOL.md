# CFCompatKD + Full-View C-Mixup v2

## Why v1 stopped

The seed1111 v1 result showed a specific valid-set asymmetry:

- LAV MAE improved by 0.002338;
- LA, LV, and L all degraded;
- MissingMacro degraded by 0.001891;
- the combined Valid J gain was only 0.000223;
- no epoch beat the locked baseline Valid J by 0.003.

The v1 mixed-loss applied the same synthetic interpolation target to both the
complete LAV fusion representation and a randomly sampled missing-modality
representation. This assumes that a straight line between two complete-view
representations and a straight line between two missing-view representations
have the same label semantics. The observed LAV/missing split is direct evidence
against that assumption.

## One permitted repair

V2 removes only the unsupported part:

- C-Mixup is applied to the complete LAV fusion representation only;
- the partner is selected from the whole training batch using the unchanged
  label-KDE rule;
- no random LA/LV/L group restricts the LAV partner pool;
- the original missing-view task loss and CFCompatKD loss remain unchanged;
- missing-view fusion features receive no mixed-label loss.

The frozen C-Mixup values remain:

- alpha = 2.0;
- bandwidth = 0.5 sentiment-label units;
- mix weight = 1.0;
- self-pairs excluded for batch size greater than one.

No Teacher target, compatibility value, auxiliary target, inference parameter,
router, expert, or second-stage prediction is created for mixed samples.

## Historical asset compatibility

The implementation accepts the locked seed1111 cache format that predates the
redundant `created_from_train_only` field only when all other bindings pass:
version, null seed, `source=train_only`, evaluator SHA, exact schema, unique
sample indices, and compatibility range. Relative checkpoint paths recorded by
old result CSVs are resolved against the parent of `result_root`.

## Hard stop

This is the only repair authorized for C-Mixup. Seed1111 uses the same promotion
gate as v1:

- Valid J gain at least 0.005;
- Valid LAV MAE gain at least 0.005;
- Valid MissingMacro degradation no greater than 0.002;
- at least two epochs beat baseline Valid J by 0.003.

If v2 fails, the C-Mixup direction stops. No alpha, bandwidth, mix-weight,
warmup, layer, or partner-policy search is authorized.