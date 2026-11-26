"""CPU stand-in for ``curobo.curobolib.line_search_cu``.

Ports the two entry points a joint-space plan calls: ``update_best`` and ``line_search`` (the Wolfe
condition search that drives L-BFGS step selection). Both run 60,000 times in a production plan.
"""

from __future__ import annotations

import torch


def _first_argmax(values: torch.Tensor) -> torch.Tensor:
    """Index of the maximum, first occurrence on ties.

    The kernel scans with a strict ``max < candidate`` (kernel:190), so an earlier equal value
    survives. Ties are not exotic here: every rejected candidate scores exactly 0, so a batch
    element where no step satisfies Wolfe is all-ties, and which index wins decides the step taken.
    """
    return values.argmax(dim=-1)


def line_search(
    best_x: torch.Tensor,
    best_c: torch.Tensor,
    best_grad: torch.Tensor,
    g_x: torch.Tensor,
    x_set: torch.Tensor,
    step_vec: torch.Tensor,
    c: torch.Tensor,
    alpha_list: torch.Tensor,
    c_idx: torch.Tensor,
    c_1: float,
    c_2: float,
    strong_wolfe: bool,
    approx_wolfe: bool,
    l1: int,
    l2: int,
    batchsize: int,
):
    """Mirrors ``line_search_cu.line_search`` (``line_search_kernel``).

    Picks a step length per problem from ``l1`` candidates by the Wolfe conditions:

    * sufficient decrease -- ``c[i] <= c[0] + c_1 * alpha[i] * g0.step``
    * curvature -- ``|g[i].step| <= c_2 |g0.step|`` (strong) or ``g[i].step >= c_2 g0.step`` (weak)

    Candidates are ranked by ``alpha + 0.1`` so the *largest* acceptable step wins, and the ``+0.1``
    keeps a zero alpha distinguishable from a rejected candidate. When nothing satisfies both
    conditions the search falls back to the best candidate satisfying sufficient decrease alone,
    then to index 1 -- i.e. it always takes *some* step rather than stalling (kernel:203-216).
    """
    dtype = torch.float32
    gradients = g_x.reshape(batchsize, l1, l2).to(dtype)
    steps = step_vec.reshape(batchsize, l2).to(dtype)
    costs = c.reshape(batchsize, l1).to(dtype)
    # alpha_list is shared across problems -- the kernel reads it without a batch offset
    # (kernel:153), unlike every other per-candidate array.
    alphas = alpha_list.reshape(-1)[:l1].to(dtype)

    directional = torch.einsum("bij,bj->bi", gradients, steps)
    reference = directional[:, 0:1]

    sufficient_decrease = costs <= costs[:, 0:1] + c_1 * alphas[None, :] * reference
    if strong_wolfe:
        curvature = directional.abs() <= c_2 * reference.abs()
    else:
        curvature = directional >= c_2 * reference
    both = sufficient_decrease & curvature

    ranked = alphas[None, :] + 0.1
    chosen = _first_argmax(torch.where(both, ranked, torch.zeros_like(ranked)))
    if not approx_wolfe:
        decrease_only = _first_argmax(
            torch.where(sufficient_decrease, ranked, torch.zeros_like(ranked))
        )
        chosen = torch.where(chosen == 0, decrease_only, chosen)
        chosen = torch.where(chosen == 0, torch.ones_like(chosen), chosen)

    # c_idx offsets into the flat (batchsize * l1) candidate set: the kernel indexes x_set, g_x and
    # c with a single global index rather than per batch (kernel:235-241).
    flat = chosen + c_idx.reshape(-1)[:batchsize].to(chosen.dtype)
    best_x.reshape(batchsize, l2).copy_(
        x_set.reshape(-1, l2)[flat].to(best_x.dtype)
    )
    best_grad.reshape(batchsize, l2).copy_(
        g_x.reshape(-1, l2)[flat].to(best_grad.dtype)
    )
    best_c.reshape(-1)[:batchsize] = c.reshape(-1)[flat].to(best_c.dtype)
    return [best_x, best_c, best_grad]


def update_best(
    best_cost: torch.Tensor,
    best_q: torch.Tensor,
    best_iteration: torch.Tensor,
    current_iteration: torch.Tensor,
    cost: torch.Tensor,
    q: torch.Tensor,
    d_opt: int,
    cost_s1: int,
    cost_s2: int,
    iteration: int,
    delta_threshold: float = 1e-5,
    relative_threshold: float = 0.999,
):
    """Mirrors ``line_search_cu.update_best`` (``update_best_kernel``).

    Keeps the best-so-far iterate per optimisation problem. A candidate replaces the incumbent only
    if it clears *both* an absolute and a relative margin -- ``best - new > delta_threshold`` and
    ``new < best * relative_threshold`` -- so noise-level improvements do not reset the stall
    counter. ``best_iteration`` counts *down* while no improvement lands, which is how the caller
    detects a stalled solve.
    """
    improved_by = best_cost.reshape(-1)[:cost_s1] - cost.reshape(-1)[:cost_s1]
    changed = (improved_by > delta_threshold) & (
        cost.reshape(-1)[:cost_s1] < best_cost.reshape(-1)[:cost_s1] * relative_threshold
    )

    # best_q is one row of d_opt variables per problem; the kernel updates the whole row together.
    rows = best_q.reshape(cost_s1, d_opt)
    rows[changed] = q.reshape(cost_s1, d_opt)[changed].to(rows.dtype)

    best_cost.reshape(-1)[:cost_s1] = torch.where(
        changed, cost.reshape(-1)[:cost_s1], best_cost.reshape(-1)[:cost_s1]
    )
    iterations = best_iteration.reshape(-1)[:cost_s1]
    best_iteration.reshape(-1)[:cost_s1] = torch.where(
        changed, torch.zeros_like(iterations), iterations - 1
    )
    return [best_cost, best_q, best_iteration]
