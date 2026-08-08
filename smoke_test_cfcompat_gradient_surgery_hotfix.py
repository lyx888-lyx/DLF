"""Synthetic checks for the numerically stable v12 surgery hotfix."""
import math

import torch

from trains.singleTask.cfcompat_gradient_surgery_numerical_hotfix import (
    asymmetric_project_supervised_stable,
)


def main():
    # Exact small conflict: result should be the same as the original v12 rule.
    g_sup = (torch.tensor([-2.0, 1.0], dtype=torch.float32),)
    g_sel = (torch.tensor([1.0, 0.0], dtype=torch.float32),)
    projected, update, diagnostics = asymmetric_project_supervised_stable(g_sup, g_sel)
    assert diagnostics["conflict"] is True
    assert torch.allclose(projected[0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(update[0], torch.tensor([1.0, 1.0]), atol=1e-6)
    assert diagnostics["post_projection_cosine"] >= -diagnostics["post_projection_cosine_tolerance"]

    # Agreement: no surgery.
    g_sup2 = (torch.tensor([2.0, 1.0], dtype=torch.float32),)
    projected2, update2, diagnostics2 = asymmetric_project_supervised_stable(g_sup2, g_sel)
    assert diagnostics2["conflict"] is False
    assert torch.equal(projected2[0], g_sup2[0])
    assert torch.equal(update2[0], g_sup2[0] + g_sel[0])

    # Zero selective gradient: supervised passes through unchanged.
    zero = (torch.zeros(2, dtype=torch.float32),)
    projected3, update3, diagnostics3 = asymmetric_project_supervised_stable(g_sup, zero)
    assert diagnostics3["conflict"] is False
    assert torch.equal(projected3[0], g_sup[0])
    assert torch.equal(update3[0], g_sup[0])

    # Large float32 reduction stress case.  This is the regime that can leave a
    # tiny absolute negative dot after an analytically exact projection.
    generator = torch.Generator().manual_seed(20260808)
    selective = torch.randn(30147, generator=generator, dtype=torch.float32) * 250.0
    orthogonal_noise = torch.randn(30147, generator=generator, dtype=torch.float32)
    supervised = -3.0 * selective + 0.01 * orthogonal_noise
    projected4, update4, diagnostics4 = asymmetric_project_supervised_stable(
        (supervised,), (selective,)
    )
    assert diagnostics4["conflict"] is True
    assert diagnostics4["post_projection_cosine"] >= -diagnostics4["post_projection_cosine_tolerance"]
    assert torch.isfinite(projected4[0]).all()
    assert torch.isfinite(update4[0]).all()

    assert all(
        math.isfinite(float(diagnostics4[key]))
        for key in (
            "gradient_dot_before",
            "gradient_cosine_before",
            "supervised_gradient_l2",
            "selective_gradient_l2",
            "projected_supervised_gradient_l2",
            "update_gradient_l2",
            "post_projection_dot",
            "post_projection_cosine",
        )
    )
    print("v12 numerical-hotfix gradient surgery smoke passed")


if __name__ == "__main__":
    main()
