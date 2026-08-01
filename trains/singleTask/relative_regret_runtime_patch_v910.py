"""Runtime alignment and pool-compatibility patch for V9.10.

V9.10 consumes two semantically identical pool layouts:

* the strict Train OOF pool stores dense ``expert_predictions`` and
  ``expert_signatures`` tensors;
* the frozen Validation/Test pool stores values under ``experts[name]``.

The aligned entry point installs one adapter so the coach sees the same tensors
from either representation.  It also keeps deployment scale semantics aligned
with OOF policy calibration.
"""

from __future__ import annotations


def install_relative_regret_runtime_patch() -> None:
    from . import relative_regret_crossfit_v910 as crossfit

    crossfit.POLICY_PROFILES.clear()
    crossfit.POLICY_PROFILES.update(
        {
            "conservative": {
                "risk_aversion": 0.50,
                "confidence_z": 0.50,
                "min_predicted_gain": 0.050,
                "min_lower_gain": 0.005,
                "min_beat_probability": 0.75,
                "min_regret_margin": 0.025,
                "max_selected_scale": 0.25,
                "min_expert_confidence": 0.55,
                "beta": 0.50,
                "max_activation_rate": 0.20,
            },
            "balanced": {
                "risk_aversion": 0.25,
                "confidence_z": 0.25,
                "min_predicted_gain": 0.030,
                "min_lower_gain": 0.000,
                "min_beat_probability": 0.65,
                "min_regret_margin": 0.012,
                "max_selected_scale": 0.40,
                "min_expert_confidence": 0.40,
                "beta": 0.75,
                "max_activation_rate": 0.35,
            },
            "broad": {
                "risk_aversion": 0.10,
                "confidence_z": 0.10,
                "min_predicted_gain": 0.015,
                "min_lower_gain": -0.005,
                "min_beat_probability": 0.55,
                "min_regret_margin": 0.005,
                "max_selected_scale": 0.60,
                "min_expert_confidence": 0.00,
                "beta": 1.00,
                "max_activation_rate": 0.50,
            },
        }
    )

    if not getattr(crossfit, "_v910_pool_patch_installed", False):
        original_pool_tensors = crossfit.pool_tensors

        def pool_tensors_compatible(pool):
            # Frozen Validation/Test representation.
            if isinstance(pool.get("experts"), dict):
                return original_pool_tensors(pool)

            # Strict Train OOF representation produced by V9.9.
            required = {
                "anchor",
                "function_space",
                "expert_predictions",
                "expert_signatures",
            }
            missing = sorted(required - set(pool))
            if missing:
                raise KeyError(
                    "V9.10 pool has neither the frozen experts mapping nor the "
                    f"strict semantic tensors; missing={missing}"
                )

            predictions = pool["expert_predictions"].float()
            signatures = pool["expert_signatures"].float()
            if predictions.dim() != 3 or predictions.shape[1:] != (4, 1):
                raise ValueError(
                    "strict expert_predictions must have shape [N,4,1], got "
                    f"{tuple(predictions.shape)}"
                )
            if (
                signatures.dim() != 3
                or signatures.shape[1:] != (4, crossfit.SIGNATURE_DIM)
            ):
                raise ValueError(
                    "strict expert_signatures must have shape "
                    f"[N,4,{crossfit.SIGNATURE_DIM}], got "
                    f"{tuple(signatures.shape)}"
                )

            anchor = pool["anchor"].float()
            actions = crossfit.stack_action_predictions(anchor, predictions)
            full_signatures = crossfit.stack_action_signatures(anchor, signatures)
            context = crossfit.global_context_features(
                pool["function_space"].float(), actions
            )
            return context, actions, full_signatures[:, 1:]

        crossfit.pool_tensors = pool_tensors_compatible
        crossfit._v910_pool_patch_installed = True

    cls = crossfit.RelativeRegretCoachCrossFitterV910
    if not getattr(cls, "_v910_scale_patch_installed", False):
        original_collect_ensemble = cls.collect_ensemble

        def collect_ensemble_aligned(self, context, signatures):
            output = original_collect_ensemble(self, context, signatures)
            # OOF policies are calibrated against each model's relative-regret
            # scale. Keep that same meaning at deployment; model disagreement is
            # retained as a separate diagnostic instead of silently inflating the
            # policy scale.
            if "aleatoric_scale" in output:
                output["total_scale"] = output["predicted_scale"]
                output["predicted_scale"] = output["aleatoric_scale"]
            return output

        cls.collect_ensemble = collect_ensemble_aligned
        cls._v910_scale_patch_installed = True
