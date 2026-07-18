# MCAO Actual CFCompatKD Training Path Specification

- Audited commit: `427b3ae4cb6fb8b33a82426aad039ccdef4a40e9`
- Formal checkpoint producer: `run_cfcompat_stability_multiseed.py::train_one_seed`
- Underlying Stage3 entrypoint: `train_cf_compat_kd.py::train_one_seed`
- Model: `MissingModalityWrapper(DLF)`
- Missing-mode source: `sample_missing_masks` with a dedicated generator seeded by `seed + 104729`
- Full-view loss: `compute_full_dlf_loss(student(..., LAV))`
- Missing-view loss: `compute_task_loss(student(..., sampled_mask))`
- KD: `gated_kd_loss` using the locked train-only compatibility cache
- Backward: `total_loss = full_loss + missing_loss + kd_loss; total_loss.backward()`
- Update: Adam with original gradient accumulation; Stage8 also updates EMA after each optimizer step
- Original DLF trainer inherited: no; the objective and loop are explicitly rebuilt
- Existing auxiliary presence mask: no
- Existing mode-specific handling: input missing tokens/mask residual and KD compatibility only
- Hidden scaling: task weights 1:1:3:1:1; reconstruction/consistency 0.1; orthogonality/similarity 0.01
- Full and sampled-missing views are jointly computed in every training batch
- Historical Stage8 loop constructs a test loader and evaluates test each epoch: yes
- Stage17 may not reuse that evaluation loop; Stage17A constructs train/valid surfaces only

## Exact call chain

`main -> train_one_seed -> initialize_teacher_student -> compute_full_dlf_loss + compute_task_loss + gated_kd_loss -> total_loss.backward -> optimizer_step_and_update_ema`

## Source SHA-256

- `run_cfcompat_stability_multiseed.py`: `8a51bce59003c881ede471c7fb6e279d3a44ab9a5b3f8bce75e1d56dabdf837c`
- `train_cf_compat_kd.py`: `54e02c06cf42a2d9002144fac672ec87e55871ff999a0af1dafeb66840f002eb`
- `trains/singleTask/missing_utils.py`: `f2200d6cafbcd90da36a8c3949b28c56c03f0bfba03d92c62014f4d4e86bc5ba`
- `trains/singleTask/model/DLF.py`: `ac11e1528d91fbe62dda1b1958e141e3a739b2589e0012f16cb1dc90d7265ea4`
- `trains/singleTask/cf_compat_kd_utils.py`: `8666f568c473bd002360f130044eab5aad739b689bcce44a2897674520a630b4`
