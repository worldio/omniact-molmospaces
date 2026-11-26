"""CPU stand-in for ``curobo.curobolib.tensor_step_cu``.

Ports the two entry points a joint-space plan actually calls -- ``step_idx_position2`` and
``step_position_backward2`` -- measured over a production-budget plan (60,008 and 60,000 calls
respectively). ``step_position2``, ``step_acceleration`` and ``step_acceleration_idx`` are never
reached on this path and are left unimplemented rather than written blind.

The kernel runs one thread per (batch, horizon, dof) and assembles a five-point stencil by hand,
with seven special cases for the ends of the trajectory. Those cases are not seven different
formulas: they are the same stencil against a position sequence that is *extrapolated* backwards
from the start state and *clamped* forwards at the last action. Writing it that way turns the whole
kernel into one gather plus a dot product with fixed coefficients, which is both far shorter and
far faster than reproducing the branches.
"""

from __future__ import annotations

import torch

# tensor_step_kernel.cu:22-23. Note the values are not 0/1 and BWD_DIFF is the negative one; the
# python wrapper's `mode=-1` default is therefore backward difference, while trajopt passes 0.
BWD_DIFF = -1
CENTRAL_DIFF = 0

# Central-difference coefficients (kernel:458-467), on the five-point stencil.
_VELOCITY = (0.083333333, -0.666666667, 0.0, 0.666666667, -0.083333333)
_ACCELERATION = (-0.083333333, 1.333333333, -2.5, 1.333333333, -0.083333333)
_JERK = (-0.5, 1.0, 0.0, -1.0, 0.5)


def _padded_positions(
    u_position: torch.Tensor,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    dt_inv: float,
    horizon: int,
) -> torch.Tensor:
    """The action sequence with four synthetic samples prepended.

    The kernel's ``h_idx in {0,1,2,3}`` branches each write a constant-jerk extrapolation of the
    start state one, two, three or four steps *before* the trajectory begins (kernel:374-414). They
    are the same three expressions reused at different offsets, so they are built once here as
    U[-4..-1] and the branches disappear. ``st_jerk`` is hard-coded to zero in the kernel
    (kernel:340), so its terms drop out.
    """
    position = start_position
    velocity = start_velocity * dt_inv
    acceleration = start_acceleration * dt_inv * dt_inv
    return torch.stack(
        (
            position - 3.0 * velocity - 1.5 * acceleration,  # u[-4]
            position - 2.0 * velocity - 2.0 * acceleration,  # u[-3]
            position - 1.0 * velocity - 1.5 * acceleration,  # u[-2]
            position,                                        # u[-1]
        ),
        dim=1,
    )


def step_idx_position2(
    out_position: torch.Tensor,
    out_velocity: torch.Tensor,
    out_acceleration: torch.Tensor,
    out_jerk: torch.Tensor,
    u_position: torch.Tensor,
    start_position: torch.Tensor,
    start_velocity: torch.Tensor,
    start_acceleration: torch.Tensor,
    start_idx: torch.Tensor,
    traj_dt: torch.Tensor,
    batch_size: int,
    horizon: int,
    dof: int,
    mode: int = CENTRAL_DIFF,
):
    """Mirrors ``tensor_step_cu.step_idx_position2``. Writes in place, returns the four outputs."""
    if mode != CENTRAL_DIFF:
        raise NotImplementedError(
            f"only CENTRAL_DIFF is ported; the planner was observed using it exclusively (mode={mode})"
        )
    dtype = out_position.dtype
    # traj_dt holds 1/dt, per the kernel's own note at :338.
    dt = float(traj_dt.reshape(-1)[0])
    dt_inv = 1.0 / dt

    # start_idx selects which start state each batch element uses.
    index = start_idx.reshape(-1).long()
    start_position = start_position.reshape(-1, dof)[index].to(dtype)
    start_velocity = start_velocity.reshape(-1, dof)[index].to(dtype)
    start_acceleration = start_acceleration.reshape(-1, dof)[index].to(dtype)

    actions = u_position.reshape(batch_size, horizon - 4, dof).to(dtype)
    prefix = _padded_positions(
        actions, start_position, start_velocity, start_acceleration, dt_inv, horizon
    )
    padded = torch.cat((prefix, actions), dim=1)  # (batch, horizon, dof)

    # in_pos[k] = padded[h + k], clamped at the last real action -- which is what the kernel's
    # horizon-4 .. horizon-1 branches do by copying the last value forward (kernel:417-454).
    steps = torch.arange(horizon, device=padded.device)
    offsets = torch.arange(5, device=padded.device)
    gather = (steps[:, None] + offsets[None, :]).clamp(max=horizon - 1)
    stencil = padded[:, gather, :]  # (batch, horizon, 5, dof)

    def combine(coefficients, scale):
        weights = torch.tensor(coefficients, dtype=dtype, device=padded.device)
        return (stencil * weights[None, None, :, None]).sum(dim=2) * scale

    out_position.copy_(stencil[:, :, 2, :])
    out_velocity.copy_(combine(_VELOCITY, dt))
    out_acceleration.copy_(combine(_ACCELERATION, dt * dt))
    out_jerk.copy_(combine(_JERK, dt * dt * dt))
    return [out_position, out_velocity, out_acceleration, out_jerk]


def step_position_backward2(
    out_grad_position: torch.Tensor,
    grad_position: torch.Tensor,
    grad_velocity: torch.Tensor,
    grad_acceleration: torch.Tensor,
    grad_jerk: torch.Tensor,
    traj_dt: torch.Tensor,
    batch_size: int,
    horizon: int,
    dof: int,
    mode: int = CENTRAL_DIFF,
):
    """Mirrors ``tensor_step_cu.step_position_backward2``.

    **Not the adjoint of the forward**, and reproducing it as one is wrong. cuRobo's kernel
    deliberately truncates at the trajectory ends: it emits nothing for ``h < 2`` or
    ``h >= horizon - 2`` (kernel:800), and at ``h == horizon - 3`` it returns only the sum of three
    position gradients, with the velocity/acceleration/jerk terms commented out and the reason
    given in the source -- "can cause oscilatory gradient steps" (kernel:862-887).

    A mathematically clean transpose therefore disagrees with it by a wide margin, and since these
    gradients drive L-BFGS, that disagreement does not stay small -- it steers the optimiser to a
    different trajectory. Measured before this was corrected: worst error 1.4e+04.

    In the interior the coefficients are the forward stencil *reversed*, which is the adjoint of a
    correlation, so the two do agree where cuRobo lets them.
    """
    if mode != CENTRAL_DIFF:
        raise NotImplementedError(f"only CENTRAL_DIFF is ported (mode={mode})")
    dtype, device = out_grad_position.dtype, out_grad_position.device
    dt = float(traj_dt.reshape(-1)[0])

    grads = {
        "position": grad_position.reshape(batch_size, horizon, dof).to(dtype),
        "velocity": grad_velocity.reshape(batch_size, horizon, dof).to(dtype),
        "acceleration": grad_acceleration.reshape(batch_size, horizon, dof).to(dtype),
        "jerk": grad_jerk.reshape(batch_size, horizon, dof).to(dtype),
    }
    result = torch.zeros((batch_size, horizon - 4, dof), dtype=dtype, device=device)

    # Interior: h in [2, horizon-4] -> action index h-2, reading a 5-wide window centred on h.
    interior = torch.arange(2, horizon - 3, device=device)
    if interior.numel():
        window = interior[:, None] + torch.arange(-2, 3, device=device)[None, :]
        accumulated = grads["position"][:, interior, :]
        for key, coefficients, scale in (
            ("velocity", (-0.083333333, 0.666666667, 0.0, -0.666666667, 0.083333333), dt),
            ("acceleration", _ACCELERATION, dt * dt),
            ("jerk", (0.5, -1.0, 0.0, 1.0, -0.5), dt * dt * dt),
        ):
            # Sum first, scale after, exactly as the kernel groups it (kernel:845-854). Folding
            # the scale into the coefficients is algebraically identical and numerically is not:
            # dt here is 1/dt, so the jerk term carries a factor of 125000 and the two groupings
            # disagree by ~2e-3 in float32.
            weights = torch.tensor(coefficients, dtype=dtype, device=device)
            accumulated = accumulated + (
                grads[key][:, window, :] * weights[None, None, :, None]
            ).sum(dim=2) * scale
        result[:, interior - 2, :] = accumulated

    # The truncated end case (kernel:860-876): three position gradients, nothing else.
    tail = horizon - 3
    if tail >= 2:
        result[:, tail - 2, :] = (
            grads["position"][:, tail, :]
            + grads["position"][:, tail + 1, :]
            + grads["position"][:, tail + 2, :]
        )

    out_grad_position.copy_(result.reshape(out_grad_position.shape))
    return [out_grad_position]
