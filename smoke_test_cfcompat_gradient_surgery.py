"""Synthetic checks for the asymmetric gradient surgery used by v12."""
import math

import torch

from trains.singleTask.cfcompat_gradient_surgery_utils import (
    asymmetric_project_supervised,
    assign_gradient_tuple,
    gradient_dot,
)


def main():
    parameter = torch.nn.Parameter(torch.zeros(2))

    # Conflict: supervised points left, selective points right.
    g_sup = (torch.tensor([-2.0, 1.0]),)
    g_sel = (torch.tensor([1.0, 0.0]),)
    projected, update, diagnostics = asymmetric_project_supervised(g_sup, g_sel)
    assert diagnostics["conflict"] is True
    assert diagnostics["gradient_dot_before"] < 0.0
    assert abs(float(gradient_dot(projected, g_sel))) < 1e-6
    assert torch.allclose(projected[0], torch.tensor([0.0, 1.0]), atol=1e-6)
    assert torch.allclose(update[0], torch.tensor([1.0, 1.0]), atol=1e-6)
    assert diagnostics["supervised_l2_removed_fraction"] > 0.0

    # Agreement: no surgery at all.
    g_sup2 = (torch.tensor([2.0, 1.0]),)
    g_sel2 = (torch.tensor([1.0, 0.0]),)
    projected2, update2, diagnostics2 = asymmetric_project_supervised(g_sup2, g_sel2)
    assert diagnostics2["conflict"] is False
    assert torch.equal(projected2[0], g_sup2[0])
    assert torch.equal(update2[0], g_sup2[0] + g_sel2[0])

    # Zero selective gradient: supervised must pass through unchanged.
    zero = (torch.zeros(2),)
    projected3, update3, diagnostics3 = asymmetric_project_supervised(g_sup, zero)
    assert diagnostics3["conflict"] is False
    assert torch.equal(projected3[0], g_sup[0])
    assert torch.equal(update3[0], g_sup[0])

    assign_gradient_tuple([parameter], update)
    assert torch.equal(parameter.grad, update[0])
    assert all(
        math.isfinite(float(diagnostics[key]))
        for key in (
            "gradient_dot_before",
            "gradient_cosine_before",
            "supervised_gradient_l2",
            "selective_gradient_l2",
            "projected_supervised_gradient_l2",
            "update_gradient_l2",
            "post_projection_dot",
        )
    )
    print("v12 gradient surgery smoke passed")


if __name__ == "__main__":
    main()
