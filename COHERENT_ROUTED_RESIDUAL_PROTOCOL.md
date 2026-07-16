# Stage 4A.1 Coherent Compatibility-Routed Residual Distillation

- Base: `d26f935e5b4dbcf245a28d21d30739d676889881` on `feature/cf-residual-recovery-v1`.
- Branch: `feature/coherent-routed-residual-v1`.
- Locked seed: 1111; locked residual-cache SHA-256: `8702d2ff15b80594094ea74469ad393a681992ecce1891c7095208f8c1004790`.
- Shared auxiliary budget: `mean(C * direct_each + (1-C) * residual_each)`.
- `base_shared`: direct KD uses the base prediction.
- `corrected_joint_shared`: direct KD uses the corrected prediction and updates residual heads.
- `corrected_stopres_shared`: direct KD uses `base + residual.detach()` and cannot update residual heads.
- The label task always uses the corrected prediction; LAV residual is exactly zero.
- Teacher, Student initialization, evaluator metadata, compatibility, signed residual targets, scales, optimizer, RNG, and benchmark protocol remain frozen.
- Alignment and gradient audits are diagnostic-only, train-only, parameter-preserving, and RNG-preserving.
- Main checkpoints are validation-selected; test-best checkpoints remain diagnostic-only.
- Execution order is alignment audit, three two-epoch smokes, commit/push, then formal A, B, C, final audit, stop.
