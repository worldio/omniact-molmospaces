"""CPU stand-in for ``curobo.curobolib.lbfgs_step_cu``.

Ports ``forward`` (``lbfgs_update_buffer_and_step_v1``): one L-BFGS iteration -- push the newest
curvature pair into the history, then run the two-loop recursion to turn the gradient into a step.

The kernel fuses the buffer update and the recursion into a single launch and keeps the history in
shared memory. On CPU the history is small (m <= 31) and the batch is large, so the loops run over
history and vectorise over batch and variables -- the opposite of the GPU's parallelisation, and the
right one here.

**This kernel mutates its inputs**: ``s_buffer``, ``y_buffer``, ``rho_buffer``, ``x_0`` and
``grad_0`` all carry state into the next iteration. Getting the returned step right while getting
those wrong would look correct for exactly one iteration.
"""

from __future__ import annotations

import torch


def forward(
    step_vec: torch.Tensor,
    rho_buffer: torch.Tensor,
    y_buffer: torch.Tensor,
    s_buffer: torch.Tensor,
    q: torch.Tensor,
    grad_q: torch.Tensor,
    x_0: torch.Tensor,
    grad_0: torch.Tensor,
    epsilon: float,
    batchsize: int,
    history: int,
    v_dim: int,
    stable_mode: bool = False,
    use_shared_buffers: bool = True,
):
    """Mirrors ``lbfgs_step_cu.forward``. Writes in place, returns ``[step_vec]``."""
    dtype = torch.float32
    gradient = grad_q.reshape(batchsize, v_dim).to(dtype)
    position = q.reshape(batchsize, v_dim).to(dtype)
    previous_gradient = grad_0.reshape(batchsize, v_dim)
    previous_position = x_0.reshape(batchsize, v_dim)

    y = gradient - previous_gradient.to(dtype)
    s = position - previous_position.to(dtype)
    numerator = (y * s).sum(dim=1)  # (batch,)

    history_s = s_buffer.reshape(history, batchsize, v_dim)
    history_y = y_buffer.reshape(history, batchsize, v_dim)
    rho = rho_buffer.reshape(history, batchsize)

    # Roll the history down one slot and append the new pair. Every kernel instantiation is
    # compiled with rolled_ys=false, so this shift always happens -- there is no ring-buffer path.
    if history > 1:
        history_s[:-1] = history_s[1:].clone()
        history_y[:-1] = history_y[1:].clone()
        rho[:-1] = rho[1:].clone()
    history_s[history - 1] = s.to(history_s.dtype)
    history_y[history - 1] = y.to(history_y.dtype)

    newest_rho = 1.0 / numerator
    if stable_mode:
        newest_rho = torch.where(numerator == 0.0, torch.zeros_like(newest_rho), newest_rho)
    rho[history - 1] = newest_rho.to(rho.dtype)

    # The kernel overwrites the anchors *after* forming y and s from them.
    previous_gradient.copy_(gradient.to(previous_gradient.dtype))
    previous_position.copy_(position.to(previous_position.dtype))

    working = gradient.clone()
    alpha = torch.empty((history, batchsize), dtype=dtype)
    for i in range(history - 1, -1, -1):
        alpha[i] = (working * history_s[i].to(dtype)).sum(dim=1) * rho[i].to(dtype)
        working = working - alpha[i].unsqueeze(1) * history_y[i].to(dtype)

    denominator = (y * y).sum(dim=1)
    scaling = numerator / denominator
    if stable_mode:
        scaling = torch.where(denominator == 0.0, torch.full_like(scaling, epsilon), scaling)
    # relu, not abs: a negative curvature ratio means the pair is not positive-definite and the
    # kernel discards the whole scaled direction rather than flipping its sign (kernel:339).
    working = working * scaling.clamp_min(0.0).unsqueeze(1)

    for i in range(history):
        beta = (working * history_y[i].to(dtype)).sum(dim=1) * rho[i].to(dtype)
        working = working + (alpha[i] - beta).unsqueeze(1) * history_s[i].to(dtype)

    step_vec.reshape(batchsize, v_dim).copy_((-working).to(step_vec.dtype))
    return [step_vec]
