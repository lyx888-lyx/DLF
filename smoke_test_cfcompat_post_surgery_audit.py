"""Synthetic checks for the frozen v12.1 post-surgery gradient audit."""
import torch

from trains.singleTask.cfcompat_post_surgery_audit_utils import (
    replay_asymmetric_projection,
)


def main():
    # Conflict: remove only the supervised component anti-aligned with selective.
    g_sup = torch.tensor([-2.0, 1.0], dtype=torch.float64)
    g_sel = torch.tensor([1.0, 0.0], dtype=torch.float64)
    replay = replay_asymmetric_projection(g_sup, g_sel)
    assert replay["diagnostics"]["conflict"] is True
    assert torch.allclose(
        replay["SUPERVISED_PROJECTED"],
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        replay["SUPERVISED_REMOVED_CONFLICT"],
        torch.tensor([-2.0, 0.0], dtype=torch.float64),
        atol=1e-12,
    )
    assert torch.allclose(
        replay["SURGERY_UPDATE"],
        torch.tensor([1.0, 1.0], dtype=torch.float64),
        atol=1e-12,
    )
    assert abs(replay["diagnostics"]["post_projection_dot"]) < 1e-12
    assert torch.allclose(
        replay["SUPERVISED_PROJECTED"] + replay["SUPERVISED_REMOVED_CONFLICT"],
        g_sup,
        atol=1e-12,
    )

    # Agreement: no surgery.
    g_sup2 = torch.tensor([2.0, 1.0], dtype=torch.float64)
    replay2 = replay_asymmetric_projection(g_sup2, g_sel)
    assert replay2["diagnostics"]["conflict"] is False
    assert torch.equal(replay2["SUPERVISED_PROJECTED"], g_sup2)
    assert torch.count_nonzero(replay2["SUPERVISED_REMOVED_CONFLICT"]) == 0
    assert torch.equal(replay2["SURGERY_UPDATE"], g_sup2 + g_sel)

    # Zero selective gradient: supervised passes through unchanged.
    zero = torch.zeros_like(g_sel)
    replay3 = replay_asymmetric_projection(g_sup, zero)
    assert replay3["diagnostics"]["conflict"] is False
    assert torch.equal(replay3["SUPERVISED_PROJECTED"], g_sup)
    assert torch.equal(replay3["SURGERY_UPDATE"], g_sup)

    # Scale invariance of the geometric relation.
    replay4 = replay_asymmetric_projection(1000.0 * g_sup, 0.01 * g_sel)
    assert replay4["diagnostics"]["conflict"] is True
    assert abs(replay4["diagnostics"]["post_projection_cosine"]) < 1e-12
    assert torch.allclose(
        replay4["SUPERVISED_PROJECTED"],
        torch.tensor([0.0, 1000.0], dtype=torch.float64),
        atol=1e-9,
    )

    print("v12.1 post-surgery audit smoke passed")


if __name__ == "__main__":
    main()
