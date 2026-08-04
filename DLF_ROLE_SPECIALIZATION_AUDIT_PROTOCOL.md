# DLF role-specialization and long-tail audit v1

## Purpose

Evaluate the prerequisite for a hierarchical long-tail and semantic-risk
supervision method before training a new model. The proposed representation
roles are:

- shared representation: sentiment polarity;
- modality-specific representation: absolute sentiment intensity;
- final fused representation: continuous sentiment regression and decision risk.

This branch performs distribution analysis and frozen-representation probes only.
It does not modify or train DLF/CFCompatKD parameters.

## Frozen evidence

- Base branch: `feature/cfcompat-sam-valid-screen-v1`.
- MOSI checkpoints: the independently replayed `adam_replay` CFCompatKD
  checkpoints for seeds `1111` and `1114` from the SAM Valid-screen artifact.
- Probe fitting split: official train only.
- Probe evaluation split: official valid only.
- Official test is forbidden.
- MOSEI is used only for train/valid label-distribution auditing when its feature
  file exists; no MOSEI model checkpoint is required.

## Labels

A fixed neutral margin `tau=0.5` is used.

Polarity:

- negative: `y < -0.5`;
- neutral: `-0.5 <= y <= 0.5`;
- positive: `y > 0.5`.

Seven sentiment bins use fixed boundaries
`[-inf,-2.5,-1.5,-0.5,0.5,1.5,2.5,+inf]` and labels `-3..3`.

Absolute intensity uses four ordered levels:

- neutral: `|y| <= 0.5`;
- weak: `0.5 < |y| <= 1.5`;
- medium: `1.5 < |y| <= 2.5`;
- strong: `|y| > 2.5`.

Effective-number diagnostic weights are computed from train counts only with
`beta=(N-1)/N`, normalized to mean one, and additionally reported with a fixed
cap of three. These weights are descriptive and are not used for training in
this branch.

## Frozen representations

Representations are captured with forward hooks without changing DLF:

- `shared`: input to `backbone.proj1_c`, the shared fused representation;
- `specific_l/a/v`: inputs to the three high-level specific projection heads;
- `specific_present`: concatenation of only the modality-specific
  representations present in LAV/LA/LV/L;
- `final_fusion`: input to `backbone.proj1`.

The frozen checkpoint is evaluated in `eval()` mode. Model tensor-state hashes
before and after extraction must be identical.

## Linear probes

All probe choices are fixed before observing results.

Polarity probe:

- `StandardScaler`;
- multinomial logistic regression;
- `C=1`, `class_weight=balanced`, `max_iter=2000`, `random_state=0`.

Absolute-intensity probe:

- three cumulative binary logistic regressions for
  `|y|>0.5`, `|y|>1.5`, and `|y|>2.5`;
- the same scaler and fixed logistic-regression settings;
- cumulative probabilities are monotonized before decoding the four-level
  prediction.

Each seed, modality view, and representation family is fitted independently on
train and evaluated once on valid. No probe hyperparameter is selected by valid.

## Role-alignment gate

The primary comparison is `shared` versus `specific_present`.

For each seed and view:

- polarity advantage = shared polarity macro-F1 minus specific macro-F1;
- intensity advantage = shared ordinal MAE minus specific ordinal MAE, so a
  positive value means the specific representation is better for intensity.

`ROLE_ALIGNMENT_SUPPORTED` requires:

- both LAV seeds have positive polarity and intensity advantages;
- at least 75% of all seed-view comparisons have positive advantage for each
  task;
- mean polarity advantage is at least `0.01` macro-F1;
- mean intensity advantage is at least `0.02` ordinal levels.

A weaker positive average produces `PARTIAL_ROLE_ALIGNMENT_NEEDS_REVIEW`.
Otherwise the role hypothesis is not supported.

## Long-tail diagnostics

The audit reports train and valid counts, video coverage, imbalance ratios,
effective-number weights, per-bin baseline MAE, macro-bin MAE, worst-bin MAE,
polarity-flip rate, and neutral-escape rate for LAV/LA/LV/L.

Long-tail presence is descriptive and does not by itself authorize reweighting.

## Hard stops

- No official test construction or traversal.
- No DLF/CFCompatKD optimizer, scheduler, backward pass, or parameter update.
- No probe hyperparameter search.
- No manual class weights.
- No training of the proposed hierarchical heads in this branch.
- A Stage-B training method is authorized only by
  `PROMOTE_ROLE_SPECIALIZATION_STAGE_B`.
