# Stage 23A-MOSEI — Expert Complementarity and Personalized Teacher Audit

Final status: `STAGE23A_JUDGE_NOT_ACTIONABLE`

## Frozen committee

- Seven trained components per fold: `clean_seed1111`, `clean_seed1114`, `uniform_kd_seed1111`, `moddrop_seed1111`, `moddrop_seed1114`, `cfcompat_seed1111`, `cfcompat_seed1114`.
- Committee Experts: `uniform_kd_seed1111`, `moddrop_seed1111`, `moddrop_seed1114`, `cfcompat_seed1111`, `cfcompat_seed1114`.
- `clean_seed1111/1114` are base components only and never enter the committee.
- `uniform_kd_seed1114` remains excluded exactly as preregistered; no Expert was added or retrained.

## OOF integrity

- Train samples: 16326/16326; sources: 2249/2249.
- OOF rows: 326520/326520 = 16,326 × 4 modes × 5 Experts.
- duplicate keys: 0; missing cells: 0; fold sample overlap: 0; source overlap: 0.
- Fold/run-manifest SHA values are recorded; all referenced prediction/sample/protocol/split/checkpoint SHA checks passed.
- Every best epoch equals the training-side source-disjoint inner-valid argmin; outer OOF was not used for checkpoint selection.

## Expert complementarity

- Per-mode fixed simplex stacking J: 0.554975.
- Oracle single-existing-Expert selection J: 0.401363.
- Oracle ΔJ: -0.153612; frozen threshold ≤ -0.005: **passed**.
- Missing-mode Oracle ΔMAE: `{"L": -0.15276786332712938, "LA": -0.15336815299742407, "LV": -0.15320528685621765}`; all three modes pass the dynamic-space threshold.
- Larger complementarity source: `seed_difference`.

## Judge and Personalized Teacher

- Joint-risk vs per-mode fixed mean ΔJ: -0.00003776 (required ≤ -0.003).
- Worst split ΔJ: -0.00003507 (required ≤ +0.001).
- Joint-risk vs shuffled mean ΔJ: -0.00003569 (required ≤ -0.002).
- Improved missing modes: 3/3; split directions consistent: True.
- Dynamic weights noncollapsed: False; systematic classification degradation: True.
- J2 selected Expert is in the true top-2 on 32.61%/34.76% of the two Judge folds; mean selected regrets are 0.157863/0.155841.
- Result: the Oracle space is real, but the frozen Judge does not reliably realize it. Student training is not recommended.

## Additional diagnostics

- Best/second-best error gaps, top-1%/5%-trimmed Oracle, top-2 regret, dynamic-weight variance, fallback distance/rate, and Expert dominance are recorded in `stage23a_mosei_audit.json` and the TSV files.
- Removing high fixed-error tails is diagnostic only and does not change any gate.

## Data locks

- Official Valid access count: **0** (Judge gate failed, so no one-shot confirmation was permitted).
- Locked Test access count: **0**.
- Test loader constructed: **No**.
- Student trained: **No**.
