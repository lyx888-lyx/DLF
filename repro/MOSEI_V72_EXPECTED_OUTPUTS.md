# Expected MOSEI V7.2 outputs

Output directory: `result/anchored_committee_v72/mosei/seed_1111/`

Key files:

- `anchored_committee_v72_summary.json`
- `anchored_committee_v72_predictions.csv`
- `v72_committee_cv.csv`
- `v72_shrinkage_calibration.csv`
- `v72_test_comparison.csv`
- `v72_test_region_diagnostics.csv`
- `anchored_committee_teacher_cache.pth`

Primary checks:

1. Best single teacher versus uniform committee.
2. Uniform versus global simplex.
3. Global simplex versus region simplex under internal CV.
4. Valid-selected anchored shrinkage versus the best single teacher and full committee.
5. Bootstrap 95% confidence intervals for the final MAE gain.
