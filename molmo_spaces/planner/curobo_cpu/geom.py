"""CPU stand-in for ``curobo.curobolib.geom_cu``.

Ports the collision entry points a joint-space plan calls: ``self_collision_distance``,
``sphere_obb_clpt`` (``closest_point``) and ``swept_sphere_obb_clpt`` (``swept_closest_point``).
``pose_distance``/``pose_distance_backward`` are deliberately absent -- profiling a production plan
shows they are never reached when the goal is a joint configuration, so 883 lines of
``pose_distance_kernel.cu`` are not on the critical path.

These are the hot kernels: ``self_collision_distance`` and ``swept_closest_point`` together account
for most of the collision cost, and the swept variant is called 60,000 times in a production plan
against ``closest_point``'s 14.
"""

from __future__ import annotations

import torch

# Peak elements per pairwise chunk, i.e. b_size * pairs_per_chunk. Swept over 1e5..3.2e7 on the
# measured 800x11640 workload: 66 ms unchunked, 50/32/17/13.3/14.7/29/54 ms at 32M/4M/400k... with
# the minimum at 1M. Working set, not arithmetic, is what moves this -- 1M float32 lanes plus the
# 3-wide difference vector is roughly where the temporaries stop spilling L2.
_PAIR_CHUNK_ELEMENTS = 1_000_000


def _transform_into_box_frames(centres: torch.Tensor, position: torch.Tensor, quaternion: torch.Tensor):
    """Sphere centres expressed in each box's local frame.

    ``transform_sphere_quat`` (kernel:243) writes the rotation out as an expanded quaternion
    sandwich rather than building a matrix. Expanding it back into a rotation matrix is the same
    arithmetic and lets one batched matmul do every sphere against every box at once. The kernel's
    identity-quaternion shortcut needs no counterpart: the expansion already reduces to a
    translation when the vector part is zero.

    ``centres`` is (n, 3), ``position``/``quaternion`` are (n_boxes, 3) and (n_boxes, 4) as wxyz.
    Returns (n_boxes, n, 3).
    """
    w, x, y, z = quaternion.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(-1, 3, 3)
    return centres[None] @ rotation.transpose(1, 2) + position[:, None, :]


def _closest_point_delta(bounds: torch.Tensor, local: torch.Tensor, radius: torch.Tensor):
    """``compute_closest_point`` (kernel:421), vectorised.

    Returns ``(delta, signed_penetration)`` where ``delta`` is the unit direction the kernel writes
    into its gradient slot and ``signed_penetration`` is ``sphere.w - |offset to surface|`` outside
    the box and ``sphere.w + depth`` inside it.

    Outside, the closest point is the sphere centre clamped to the box. Inside, the kernel snaps to
    the *nearest face* -- picking the axis with the smallest remaining margin, with ties broken
    y, x, z in that order (kernel:466-505). The tie order matters: an exactly-centred sphere is a
    real case for a box that fits tightly, and choosing a different face flips the gradient.
    """
    absolute = local.abs()
    margin = bounds - absolute  # >0 per axis means inside along that axis
    inside = margin.amin(dim=-1) > 0.0

    clamped = torch.where(margin < 0, torch.copysign(bounds, local), local)

    # Nearest face, with the kernel's y <= x <= z preference among equal margins.
    y_wins = (margin[..., 1] <= margin[..., 0]) & (margin[..., 1] <= margin[..., 2])
    x_wins = ~y_wins & (margin[..., 0] <= margin[..., 1]) & (margin[..., 0] <= margin[..., 2])
    axis = torch.where(y_wins, 1, torch.where(x_wins, 0, 2))
    snapped = local.clone()
    face = torch.gather(torch.copysign(bounds, local), -1, axis.unsqueeze(-1))
    snapped.scatter_(-1, axis.unsqueeze(-1), face)

    point = torch.where(inside.unsqueeze(-1), snapped, clamped)
    delta = point - local
    distance = torch.linalg.vector_norm(delta, dim=-1)

    # Degenerate case: centre exactly on the closest point. The kernel falls back to -point.
    delta = torch.where((distance == 0.0).unsqueeze(-1), -point, delta)
    # Outside, the distance is negative; inside, the direction is reversed to point outwards.
    signed = torch.where(inside, distance, -distance)
    delta = torch.where(inside.unsqueeze(-1), -delta, delta)
    delta = delta / torch.linalg.vector_norm(delta, dim=-1, keepdim=True).clamp_min(1e-20)
    return delta, signed + radius


def _scale_eta_metric(delta: torch.Tensor, penetration: torch.Tensor, eta: float):
    """``scale_eta_metric`` (kernel:639): a quadratic ramp inside the activation band.

    Beyond ``eta`` the cost is linear (``d - eta/2``); within it the cost is ``d^2 / 2eta`` and the
    gradient is scaled by ``d / eta``, which is what makes the cost C1 at the boundary instead of
    kinking there. Non-penetrating spheres contribute nothing.
    """
    active = penetration > 0.0
    far = penetration > eta
    # eta can be zero -- cuRobo builds collision *constraints* with activation_distance 0 alongside
    # the costs. The kernel branches, so its quadratic arm is simply unreachable there (anything
    # with sph_dist > 0 is also > 0 == eta). This form evaluates both arms, so the division needs a
    # floor; ``far`` then discards the value it produced. Without it: ZeroDivisionError mid-plan.
    safe_eta = eta if eta > 0.0 else 1.0
    cost = torch.where(
        active,
        torch.where(far, penetration - 0.5 * eta, (0.5 / safe_eta) * penetration * penetration),
        torch.zeros_like(penetration),
    )
    scale = torch.where(active & ~far, penetration / safe_eta, torch.ones_like(penetration))
    gradient = delta * torch.where(active, scale, torch.zeros_like(scale)).unsqueeze(-1)
    return gradient, cost


def closest_point(
    sphere_position: torch.Tensor,
    distance: torch.Tensor,
    closest_point_out: torch.Tensor,
    sparsity_idx: torch.Tensor,
    weight: torch.Tensor,
    activation_distance: torch.Tensor,
    max_distance: torch.Tensor,
    obb_accel: torch.Tensor,
    obb_bounds: torch.Tensor,
    obb_pose: torch.Tensor,
    obb_enable: torch.Tensor,
    n_env_obb: torch.Tensor,
    env_query_idx: torch.Tensor,
    max_nobs: int,
    batch_size: int,
    horizon: int,
    n_spheres: int,
    transform_back: bool,
    compute_distance: bool,
    use_batch_env: bool,
    sum_collisions: bool = True,
    compute_esdf: bool = False,
):
    """Mirrors ``geom_cu.closest_point`` (``sphere_obb_distance_fn``, kernel:1029).

    Only the single-environment, ``sum_collisions`` configuration is implemented -- the one the
    planner uses. Batched-environment queries and the ESDF variant raise rather than silently
    returning something plausible.
    """
    if compute_esdf:
        raise NotImplementedError("the ESDF sphere-OBB variant is not ported")
    if use_batch_env and int(env_query_idx.reshape(-1).min()) != int(
        env_query_idx.reshape(-1).max()
    ):
        # The planner sets use_batch_env even with a single world, so the flag alone does not mean
        # the environments differ -- only a non-constant env_query_idx does, and that is genuinely
        # unported rather than merely untested.
        raise NotImplementedError("per-batch-element environments are not ported")

    dtype = torch.float32
    eta = float(activation_distance.reshape(-1)[0])
    scale = weight.reshape(-1)[0].to(dtype)
    total = batch_size * horizon * n_spheres

    spheres = sphere_position.reshape(total, 4).to(dtype)
    centres, radii = spheres[:, :3], spheres[:, 3]
    valid = radii >= 0.0
    # The kernel inflates the query radius by the activation distance so a sphere is "in collision"
    # as soon as it enters the band, not when it touches (kernel:1061).
    radii = radii + eta

    boxes = int(n_env_obb.reshape(-1)[0])
    enabled = obb_enable.reshape(-1)[:boxes] != 0
    pose = obb_pose.reshape(-1, 8)[:boxes]
    # load_obb_bounds (kernel:196) halves the stored dimensions: obb_bounds carries full extents,
    # the geometry works in half-extents. Missing this makes every box twice its true size, which
    # reads as a plausible-but-wrong collision cost rather than an obvious failure.
    bounds = obb_bounds.reshape(-1, 4)[:boxes, :3].to(dtype) / 2.0
    position, quaternion = pose[:, :3].to(dtype), pose[:, 3:7].to(dtype)
    keep = torch.nonzero(enabled, as_tuple=True)[0]
    position, quaternion, bounds = position[keep], quaternion[keep], bounds[keep]

    cost_total = torch.zeros(total, dtype=dtype)
    gradient_total = torch.zeros((total, 3), dtype=dtype) if transform_back else None
    # Chunked over boxes: the full (n_boxes, n_spheres, 3) intermediate is 1.4 GB at the measured
    # 419 boxes x 279,200 spheres. The kernel never materialises it -- each thread loops boxes and
    # accumulates -- so chunking is what makes the CPU version resemble it in memory as well as
    # arithmetic.
    chunk = max(1, _PAIR_CHUNK_ELEMENTS // max(total, 1))
    for start in range(0, keep.numel(), chunk):
        stop = min(start + chunk, keep.numel())
        box_position = position[start:stop]
        box_quaternion = quaternion[start:stop]
        box_bounds = bounds[start:stop]

        local = _transform_into_box_frames(centres, box_position, box_quaternion)
        # check_sphere_aabb rejects a box outright when the sphere is beyond it on any axis
        # (kernel:551), and the kernel simply `continue`s. Almost every (sphere, box) pair is
        # rejected here -- the arm touches a handful of the 419 obstacles at most -- so the pairs
        # that survive are gathered and only they pay for the closest-point geometry. Running that
        # geometry densely costs ~15x more for identical output.
        touching = (local.abs() - box_bounds[:, None, :]).amax(dim=-1) < radii[None, :]
        hits = torch.nonzero(touching, as_tuple=True)
        if hits[0].numel() == 0:
            continue
        box_index, sphere_index = hits
        delta, penetration = _closest_point_delta(
            box_bounds[box_index], local[box_index, sphere_index], radii[sphere_index]
        )
        gradient, cost = _scale_eta_metric(delta, penetration, eta)
        cost_total.index_add_(0, sphere_index, cost)
        if transform_back:
            # Gradients come back through the inverse rotation, then sum over boxes
            # (inv_transform_vec_quat_add, kernel:307).
            w, x, y, z = box_quaternion[box_index].unbind(dim=-1)
            rotation = torch.stack(
                (
                    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
                ),
                dim=-1,
            ).reshape(-1, 3, 3)
            rotated = (gradient.unsqueeze(1) @ rotation).squeeze(1)
            gradient_total.index_add_(0, sphere_index, rotated)

    cost_total = torch.where(valid, cost_total, torch.zeros_like(cost_total)) * scale
    distance.view(-1).copy_(cost_total.to(distance.dtype))
    if transform_back:
        out = closest_point_out.reshape(total, 4)
        out[:, :3] = (gradient_total * scale * valid.unsqueeze(-1)).to(out.dtype)
        out[cost_total == 0, :3] = 0
    sparsity_idx.view(-1).copy_((cost_total != 0).to(sparsity_idx.dtype))
    return [distance, closest_point_out, sparsity_idx]


def _surface_distance(bounds: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """``compute_distance_fn`` (kernel:567): signed distance to the box surface.

    The kernel gets this by re-running the full closest-point routine with the query radius replaced
    by a huge constant, so the AABB rejection never fires. That is wasteful here: the closest *point*
    is discarded, only its distance is used, and for a box the distance has a closed form --
    ``-|max(|p| - b, 0)|`` outside, ``+min(b - |p|)`` inside. Same value, none of the
    nearest-face snapping, normalisation or scatter, and it runs up to nine times per pair inside the
    sweep loop.
    """
    outside_extent = (local.abs() - bounds).clamp_min(0.0)
    inside = (bounds - local.abs()).amin(dim=-1)
    return torch.where(
        inside > 0.0, inside, -torch.linalg.vector_norm(outside_extent, dim=-1)
    )


def swept_closest_point(
    sphere_position: torch.Tensor,
    distance: torch.Tensor,
    closest_point_out: torch.Tensor,
    sparsity_idx: torch.Tensor,
    weight: torch.Tensor,
    activation_distance: torch.Tensor,
    speed_dt: torch.Tensor,
    obb_accel: torch.Tensor,
    obb_bounds: torch.Tensor,
    obb_pose: torch.Tensor,
    obb_enable: torch.Tensor,
    n_env_obb: torch.Tensor,
    env_query_idx: torch.Tensor,
    max_nobs: int,
    batch_size: int,
    horizon: int,
    n_spheres: int,
    sweep_steps: int,
    enable_speed_metric: bool,
    transform_back: bool,
    compute_distance: bool,
    use_batch_env: bool,
    sum_collisions: bool = True,
):
    """Mirrors ``geom_cu.swept_closest_point`` (``swept_sphere_obb_distance_fn``, kernel:1964).

    Continuous collision along the trajectory: each sphere is checked not only where it is, but
    swept toward its neighbours at ``h-1`` and ``h+1``, so a fast-moving sphere cannot tunnel
    through a thin obstacle between timesteps. That is why this is the hot kernel -- 60,000 calls
    at 257 us each with the real 419-obstacle world, ~11.5% of a plan's wall time.

    The sweep is adaptive rather than uniform: it advances by the current distance-to-obstacle
    (``curr_jump_distance``), so open space is crossed in one step and only near-contact regions
    are sampled finely, stopping after ``sweep_steps`` jumps or once it has covered half the gap to
    the neighbour. That data dependence is why the loop below carries a mask instead of being
    flattened.
    """
    if use_batch_env and int(env_query_idx.reshape(-1).min()) != int(
        env_query_idx.reshape(-1).max()
    ):
        raise NotImplementedError("per-batch-element environments are not ported")

    dtype = torch.float32
    eta = float(activation_distance.reshape(-1)[0])
    scale = weight.reshape(-1)[0].to(dtype)
    total = batch_size * horizon * n_spheres

    spheres = sphere_position.reshape(batch_size, horizon, n_spheres, 4).to(dtype)
    centres = spheres[..., :3].reshape(total, 3)
    radii = spheres[..., 3].reshape(total)
    valid = radii >= 0.0
    radii = radii + eta

    # Neighbouring timesteps, clamped at the ends: the kernel simply does not sweep past them.
    previous = torch.cat((spheres[:, :1], spheres[:, :-1]), dim=1)[..., :3].reshape(total, 3)
    following = torch.cat((spheres[:, 1:], spheres[:, -1:]), dim=1)[..., :3].reshape(total, 3)
    has_previous = torch.zeros(batch_size, horizon, n_spheres, dtype=torch.bool)
    has_previous[:, 1:] = True
    has_following = torch.zeros(batch_size, horizon, n_spheres, dtype=torch.bool)
    has_following[:, :-1] = True
    has_previous = has_previous.reshape(total)
    has_following = has_following.reshape(total)

    gap_previous = torch.linalg.vector_norm(previous - centres, dim=-1)
    gap_following = torch.linalg.vector_norm(following - centres, dim=-1)
    sweep_back = has_previous & (gap_previous > 0.0)
    sweep_forward = has_following & (gap_following > 0.0)
    length_previous = gap_previous + 2.0 * radii
    length_following = gap_following + 2.0 * radii

    boxes = int(n_env_obb.reshape(-1)[0])
    enabled = obb_enable.reshape(-1)[:boxes] != 0
    pose = obb_pose.reshape(-1, 8)[:boxes]
    bounds = obb_bounds.reshape(-1, 4)[:boxes, :3].to(dtype) / 2.0
    position, quaternion = pose[:, :3].to(dtype), pose[:, 3:7].to(dtype)
    keep = torch.nonzero(enabled, as_tuple=True)[0]
    position, quaternion, bounds = position[keep], quaternion[keep], bounds[keep]

    cost_total = torch.zeros(total, dtype=dtype)
    gradient_total = torch.zeros((total, 3), dtype=dtype)

    chunk = max(1, _PAIR_CHUNK_ELEMENTS // max(total, 1))
    for start in range(0, keep.numel(), chunk):
        stop = min(start + chunk, keep.numel())
        box_bounds = bounds[start:stop]
        box_position, box_quaternion = position[start:stop], quaternion[start:stop]

        local_1 = _transform_into_box_frames(centres, box_position, box_quaternion)
        local_0 = _transform_into_box_frames(previous, box_position, box_quaternion)
        local_2 = _transform_into_box_frames(following, box_position, box_quaternion)

        # Broad phase. Every sample the sweep will ever take lies inside the AABB of the three
        # sphere positions, so a pair whose AABB misses the box (inflated by the radius) cannot
        # contribute at any sweep step. Almost all 419 boxes miss almost every sphere, and skipping
        # them here avoids running the adaptive sweep -- which is up to nine closest-point
        # evaluations -- on pairs that were never going to touch.
        stacked = torch.stack((local_0, local_1, local_2))
        lower, upper = stacked.amin(dim=0), stacked.amax(dim=0)
        inflated = box_bounds[:, None, :] + radii[None, :, None]
        near = ((lower - inflated).maximum(-inflated - upper)).amax(dim=-1) < 0
        pairs_box, pairs_sphere = torch.nonzero(near, as_tuple=True)
        if pairs_box.numel() == 0:
            continue

        local_0 = local_0[pairs_box, pairs_sphere]
        local_1 = local_1[pairs_box, pairs_sphere]
        local_2 = local_2[pairs_box, pairs_sphere]
        bounds_b = box_bounds[pairs_box]
        radius_b = radii[pairs_sphere]

        # Cost at the exact timestep.
        touching = (local_1.abs() - bounds_b).amax(dim=-1) < radius_b
        delta, penetration = _closest_point_delta(bounds_b, local_1, radius_b)
        gradient, cost = _scale_eta_metric(delta, penetration, eta)
        summed = torch.where(touching, cost, torch.zeros_like(cost))
        summed_gradient = torch.where(touching.unsqueeze(-1), gradient, torch.zeros_like(gradient))

        # Where it does not touch, the free distance sets the first jump.
        free = _surface_distance(bounds_b, local_1)
        jump = torch.where(touching, torch.zeros_like(free), free)
        midpoint_jump = (jump.abs() - radius_b).maximum(radius_b)

        for neighbour_local, gap, length, active in (
            (local_0, gap_previous, length_previous, sweep_back),
            (local_2, gap_following, length_following, sweep_forward),
        ):
            pair_gap = gap[pairs_sphere]
            pair_length = length[pairs_sphere]
            running = midpoint_jump.clone()
            alive = active[pairs_sphere] & (midpoint_jump < pair_gap / 2)
            half = pair_length / 2
            for _ in range(sweep_steps):
                alive = alive & (running < half)
                if not bool(alive.any()):
                    break
                ratio = 1.0 - running / pair_length
                interpolated = (
                    ratio.unsqueeze(-1) * local_1 + (1 - ratio).unsqueeze(-1) * neighbour_local
                )
                hit = (interpolated.abs() - bounds_b).amax(dim=-1) < radius_b
                step_delta, step_penetration = _closest_point_delta(
                    bounds_b, interpolated, radius_b
                )
                step_gradient, step_cost = _scale_eta_metric(step_delta, step_penetration, eta)
                counted = alive & hit
                summed = summed + torch.where(counted, step_cost, torch.zeros_like(step_cost))
                summed_gradient = summed_gradient + torch.where(
                    counted.unsqueeze(-1), step_gradient, torch.zeros_like(step_gradient)
                )
                step_free = _surface_distance(bounds_b, interpolated)
                advance = torch.where(hit, step_penetration - radius_b, step_free)
                running = running + torch.where(
                    alive, advance.abs().maximum(radius_b), torch.zeros_like(running)
                )

        contributing = summed > 0
        cost_total.index_add_(
            0, pairs_sphere, torch.where(contributing, summed, torch.zeros_like(summed))
        )
        if transform_back:
            w, x, y, z = box_quaternion[pairs_box].unbind(dim=-1)
            rotation = torch.stack(
                (
                    1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
                    2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
                    2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
                ),
                dim=-1,
            ).reshape(-1, 3, 3)
            masked = torch.where(
                contributing.unsqueeze(-1), summed_gradient, torch.zeros_like(summed_gradient)
            )
            gradient_total.index_add_(0, pairs_sphere, (masked.unsqueeze(1) @ rotation).squeeze(1))

    cost_total = torch.where(valid, cost_total, torch.zeros_like(cost_total)) * scale
    distance.view(-1).copy_(cost_total.to(distance.dtype))
    if transform_back:
        out = closest_point_out.reshape(total, 4)
        out[:, :3] = (gradient_total * scale * valid.unsqueeze(-1)).to(out.dtype)
        out[cost_total == 0, :3] = 0
    sparsity_idx.view(-1).copy_((cost_total != 0).to(sparsity_idx.dtype))
    return [distance, closest_point_out, sparsity_idx]


def self_collision_distance(
    out_distance: torch.Tensor,
    out_vec: torch.Tensor,
    sparse_index: torch.Tensor,
    robot_spheres: torch.Tensor,
    collision_offset: torch.Tensor,
    weight: torch.Tensor,
    coll_matrix: torch.Tensor,
    thread_locations: torch.Tensor,
    thread_size: int,
    b_size: int,
    nspheres: int,
    compute_grad: bool = False,
    checks_per_thread: int = 32,
    experimental_kernel: bool = True,
):
    """Mirrors ``geom_cu.self_collision_distance`` (``self_collision_distance_kernel7``).

    Deepest penetration between any *allowed* pair of collision spheres, per batch element. The
    pair list arrives pre-computed in ``thread_locations`` -- cuRobo packs it per CUDA thread with
    ``(-1, -1)`` padding, but flattened it is simply the list of sphere index pairs that are
    permitted to collide, so the thread geometry carries no meaning on CPU.

    Only penetrating pairs count: the kernel seeds its running maximum at 0.0 and takes strictly
    greater, so separated pairs (negative ``f_diff``) never win and a batch element with no
    penetration reports exactly zero.
    """
    device = out_distance.device
    dtype = torch.float32

    pairs = thread_locations.reshape(-1, 2)[: thread_size // 2].long()
    valid = (pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)
    pairs = pairs[valid]

    spheres = robot_spheres.reshape(b_size, nspheres, 4).to(dtype)
    centres = spheres[..., :3]
    # The kernel folds each sphere's offset into its radius once, on load (kernel:351).
    radii = spheres[..., 3] + collision_offset.reshape(-1)[:nspheres].to(dtype)

    first, second = pairs[:, 0], pairs[:, 1]

    # Chunked over pairs, carrying a running maximum, rather than scoring every pair at once. The
    # full form allocates (b_size, n_pairs, 3) for the difference vectors -- 112 MB per call at the
    # measured 800x11640 -- and the allocation, not the arithmetic, is the cost: chunking is ~7x
    # faster here for identical output. Chunk width is in pairs so peak memory is bounded
    # regardless of how many collision pairs the robot has.
    chunk = max(1, _PAIR_CHUNK_ELEMENTS // max(b_size, 1))
    deepest = torch.zeros(b_size, dtype=dtype, device=device)
    argmax = torch.zeros(b_size, dtype=torch.long, device=device)
    for start in range(0, pairs.shape[0], chunk):
        stop = min(start + chunk, pairs.shape[0])
        i, j = first[start:stop], second[start:stop]
        separation = torch.linalg.vector_norm(centres[:, i, :] - centres[:, j, :], dim=-1)
        penetration = radii[:, i] + radii[:, j] - separation
        best, where = penetration.max(dim=1)
        improved = best > deepest
        deepest = torch.where(improved, best, deepest)
        argmax = torch.where(improved, where + start, argmax)
    colliding = deepest > 0.0
    scale = weight.reshape(-1)[0].to(dtype)
    # The caller's tensors are shaped (n_seeds, horizon, ...) while the kernel indexes them as one
    # flat batch of b_size; write through a flat view rather than assuming either shape.
    out_distance.view(-1).copy_(
        (torch.where(colliding, deepest, torch.zeros_like(deepest)) * scale).to(out_distance.dtype)
    )

    if not compute_grad:
        return [out_distance, out_vec, sparse_index]

    # The kernel keeps out_vec sparse: it clears only the entries its previous call marked in
    # sparse_index, then writes the two spheres of the new deepest pair. Zeroing both wholesale is
    # equivalent -- the invariant is that out_vec is non-zero exactly where sparse_index is 1 --
    # and it cannot leak a stale gradient if that invariant is ever broken upstream.
    out_vec.zero_()
    sparse_index.zero_()

    batch_index = torch.nonzero(colliding, as_tuple=True)[0]
    if batch_index.numel():
        i = first[argmax[batch_index]]
        j = second[argmax[batch_index]]
        direction = centres[batch_index, i, :] - centres[batch_index, j, :]
        direction = direction / torch.linalg.vector_norm(direction, dim=-1, keepdim=True)
        vectors = out_vec.reshape(b_size, nspheres, 4)
        vectors[batch_index, i, :3] = (-scale * direction).to(out_vec.dtype)
        vectors[batch_index, j, :3] = (scale * direction).to(out_vec.dtype)
        flags = sparse_index.reshape(b_size, nspheres)
        flags[batch_index, i] = 1
        flags[batch_index, j] = 1

    return [out_distance, out_vec, sparse_index]
