"""Runtime alignment patch for V9.10 policy scales and pre-registered profiles."""

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

    cls = crossfit.RelativeRegretCoachCrossFitterV910
    if getattr(cls, "_v910_scale_patch_installed", False):
        return
    original = cls.collect_ensemble

    def collect_ensemble_aligned(self, context, signatures):
        output = original(self, context, signatures)
        # OOF policies are calibrated against each model's relative-regret scale.
        # Keep that same meaning at deployment; model disagreement remains a
        # separate diagnostic instead of silently inflating the policy scale.
        if "aleatoric_scale" in output:
            output["total_scale"] = output["predicted_scale"]
            output["predicted_scale"] = output["aleatoric_scale"]
        return output

    cls.collect_ensemble = collect_ensemble_aligned
    cls._v910_scale_patch_installed = True
