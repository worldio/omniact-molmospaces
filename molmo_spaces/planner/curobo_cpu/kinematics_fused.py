"""CPU stand-in for ``curobo.curobolib.kinematics_fused_cu``.

Ports ``matrix_to_quaternion`` and ``forward`` (``kin_fused_forward``) from
``curobolib/cpp/kinematics_fused_kernel.cu``.

The CUDA kernel parallelises one thread per *column* of each 4x4 transform and walks the kinematic
chain in shared memory. That layout exists to fill a warp; on CPU it buys nothing, so the chain walk
is a plain loop over links with batched 4x4 matmuls. The arithmetic is the same composition either
way:

    cumul[l] = cumul[parent(l)] @ fixed_transform[l] @ joint_motion(l, angle)

which is what the kernel's ``fixed_joint_fn`` / ``prism_fn`` / ``{x,y,z}rot_fn`` compute column by
column. Verified numerically against the CUDA kernel rather than by reading -- see
``scripts/curobo/check_cpu_kernel_parity.py``.

**Where the CPU time goes.** Only the chain walk is inherently sequential: link ``l`` needs its
parent's transform. Everything else -- the per-joint motion matrices, their composition with the
fixed transforms, the quaternion conversion, the sphere transform -- is built for all links at once.
The robot's structure (parents, joint types, joint indices, axis offsets) is fixed for the lifetime
of a planner, so it is derived once and cached rather than re-read from tensors on every call; doing
that per link per call is what made the first version slow, and on CUDA tensors each of those reads
is also a device sync.
"""

from __future__ import annotations

import torch

# Joint type codes, from kinematics_fused_kernel.cu:24-30.
FIXED = -1
X_PRISM, Y_PRISM, Z_PRISM = 0, 1, 2
X_ROT, Y_ROT, Z_ROT = 3, 4, 5

# Keyed on the identity of the structure tensors, which a planner allocates once and reuses.
_STRUCTURE_CACHE: dict = {}
_SPHERE_GROUP_CACHE: dict = {}


def _structure(link_map, joint_map, joint_map_type, joint_offset_map, n_links):
    """Per-robot constants, derived once: parents, joint index per link, links grouped by type.

    Keyed on the *contents* of the structure tensors, not their addresses. Keying on ``data_ptr()``
    is faster and wrong: cuRobo builds several robot models during setup (one before joint locking,
    one after) and a freed tensor's address gets reused, so a 7-joint model's structure came back
    for a 6-joint one and index_select walked off the end of the joint vector.
    """
    parents = [int(v) for v in link_map.reshape(-1)[:n_links]]
    types = [int(v) for v in joint_map_type.reshape(-1)[:n_links]]
    joints = [int(v) for v in joint_map.reshape(-1)[:n_links]]
    key = (tuple(parents), tuple(types), tuple(joints))
    cached = _STRUCTURE_CACHE.get(key)
    if cached is not None:
        return cached

    # Fixed links carry joint index -1. Their angle is never used (the identity motion ignores it),
    # but index_select still bounds-checks, so clamp rather than branch.
    joint_index = torch.as_tensor([max(v, 0) for v in joints], dtype=torch.long)
    grouped: dict[int, torch.Tensor] = {}
    for link in range(1, n_links):
        grouped.setdefault(types[link], []).append(link)
    grouped = {
        joint_type: torch.as_tensor(links, dtype=torch.long)
        for joint_type, links in grouped.items()
        if joint_type != FIXED
    }
    offsets = joint_offset_map.reshape(-1, 2)[:n_links].float()

    cached = (parents, joint_index, grouped, offsets)
    _STRUCTURE_CACHE[key] = cached
    return cached


def _sphere_groups(link_sphere_map: torch.Tensor):
    """Sphere indices grouped by the link they ride on.

    Indexing the link transforms per sphere instead -- ``cumul[link_sphere_map]`` -- is the obvious
    formulation and the expensive one: for 349 spheres over a batch of 4096 it materialises a
    (349, 4096, 4, 4) tensor, 91 MB of gathered copies of only 16 distinct matrices. Grouping turns
    that into 16 matmuls that each read one transform.
    """
    mapping = link_sphere_map.reshape(-1).tolist()
    key = tuple(mapping)
    cached = _SPHERE_GROUP_CACHE.get(key)
    if cached is not None:
        return cached

    groups: dict[int, list[int]] = {}
    for sphere, link in enumerate(mapping):
        groups.setdefault(int(link), []).append(sphere)
    cached = [
        (link, torch.as_tensor(spheres, dtype=torch.long))
        for link, spheres in sorted(groups.items())
    ]
    _SPHERE_GROUP_CACHE[key] = cached
    return cached


def _rot_to_quat(rotations: torch.Tensor) -> torch.Tensor:
    """Batched rotation matrix -> quaternion, following ``rot_to_quat`` (kernel:146).

    Output order is **xyzw**, not the wxyz the kernel's comment at :148 claims -- confirmed against
    the compiled kernel, whose output matches ``scipy.spatial.transform.Rotation.as_quat()``
    element for element. Getting this backwards would silently corrupt every link orientation.

    The four-branch structure is not decoration: each branch solves for whichever component is
    largest, which is what keeps the square root away from zero. All four are evaluated and
    selected with ``where`` rather than gathered under boolean masks: the masked form allocates
    four index tensors and, on CUDA, forces a host sync per branch to decide whether it is empty.
    ``n`` is selected *before* the reciprocal square root, so the branches that would divide by a
    negative never do.
    """
    flat = rotations.reshape(-1, 9)
    t0, t1, t2, t3, t4, t5, t6, t7, t8 = flat.unbind(dim=1)

    upper = t8 < 0.0
    left, right = t0 > t4, t0 < -t4

    def pick(a, b, c, d):
        return torch.where(upper, torch.where(left, a, b), torch.where(right, c, d))

    n = pick(1 + t0 - t4 - t8, 1 - t0 + t4 - t8, 1 - t0 - t4 + t8, 1 + t0 + t4 + t8)
    x = pick(n, t1 + t3, t6 + t2, t5 - t7)
    y = pick(t1 + t3, n, t5 + t7, t6 - t2)
    z = pick(t6 + t2, t5 + t7, n, t1 - t3)
    w = -pick(t5 - t7, t6 - t2, t1 - t3, n)

    quaternion = torch.stack((x, y, z, w), dim=-1) * (0.5 * torch.rsqrt(n)).unsqueeze(-1)

    # normalize_quaternion (kernel:71): unit length, then sign-flipped so w >= 0. The sign flip is
    # what makes the output a canonical representative -- q and -q are the same rotation, and
    # without it the four branches would disagree on which one they return.
    inverse_length = torch.rsqrt((quaternion * quaternion).sum(dim=-1, keepdim=True))
    return quaternion * torch.where(
        quaternion[:, 3:4] < 0.0, -inverse_length, inverse_length
    )


def matrix_to_quaternion(out_quat: torch.Tensor, in_rot: torch.Tensor):
    """Mirrors ``kinematics_fused_cu.matrix_to_quaternion``: writes into ``out_quat``, returns [it]."""
    out_quat.copy_(_rot_to_quat(in_rot).to(out_quat.dtype))
    return [out_quat]


def _joint_motions(grouped, angles, n_links, batch, dtype, device):
    """Every link's post-fixed-transform motion, built for all links at once.

    ``angles`` is (n_links, batch), already scaled by the axis-direction offsets. Fixed links keep
    the identity, which is what ``fixed_joint_fn`` amounts to.
    """
    motions = torch.eye(4, dtype=dtype, device=device).repeat(n_links, batch, 1, 1)
    motions = motions.reshape(n_links, batch, 4, 4)
    for joint_type, links in grouped.items():
        selected = angles[links]  # (k, batch)
        if joint_type <= Z_PRISM:
            motions[links, :, joint_type, 3] = selected
            continue
        cos, sin = torch.cos(selected), torch.sin(selected)
        if joint_type == X_ROT:
            rows = ((1, 1, cos), (1, 2, -sin), (2, 1, sin), (2, 2, cos))
        elif joint_type == Y_ROT:
            rows = ((0, 0, cos), (0, 2, sin), (2, 0, -sin), (2, 2, cos))
        else:  # Z_ROT
            rows = ((0, 0, cos), (0, 1, -sin), (1, 0, sin), (1, 1, cos))
        for row, col, value in rows:
            motions[links, :, row, col] = value
    return motions


def backward(
    grad_out: torch.Tensor,
    grad_link_pos: torch.Tensor,
    grad_link_quat: torch.Tensor,
    grad_spheres: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor,
    robot_spheres: torch.Tensor,
    link_map: torch.Tensor,
    joint_map: torch.Tensor,
    joint_map_type: torch.Tensor,
    store_link_map: torch.Tensor,
    link_sphere_map: torch.Tensor,
    link_chain_map: torch.Tensor,
    joint_offset_map: torch.Tensor,
    batch_size: int,
    n_joints: int,
    n_spheres: int,
    sparsity_opt: bool = True,
    use_global_cumul: bool = False,
):
    """Mirrors ``kinematics_fused_cu.backward`` (``kin_fused_backward_kernel3``).

    The manipulator Jacobian, contracted with the incoming gradients rather than formed. For a
    revolute joint ``j`` and a point ``p`` downstream of it, ``dp/dq_j = axis_j x (p - origin_j)``,
    so the joint's share of a positional gradient is ``(axis x (p - origin)) . g``; for a prismatic
    joint it is just ``axis . g``. Orientation gradients add ``axis . g_rot`` -- and only the
    *vector* part of the quaternion gradient participates (kernel:1038 reads ``.y .z .w``, skipping
    the scalar).

    ``link_chain_map[l, j]`` gates which joints reach which link, which is what makes this a
    kinematic-chain sum rather than an all-pairs one. ``joint_offset_map[j, 0]`` is the axis sign,
    carrying the negated-axis joints the forward folds into the angle.
    """
    dtype = torch.float32
    transforms = fixed_transform.reshape(-1, 4, 4).to(dtype)
    n_links = transforms.shape[0]

    parents, joint_index, grouped, offsets = _structure(
        link_map, joint_map, joint_map_type, joint_offset_map, n_links
    )
    types = [int(v) for v in joint_map_type.reshape(-1)[:n_links]]
    axis_sign = offsets[:, 0]

    if use_global_cumul:
        cumul = global_cumul_mat.reshape(batch_size, n_links, 4, 4).permute(1, 0, 2, 3).to(dtype)
    else:
        angles = joint_vec.to(dtype).index_select(1, joint_index.to(joint_vec.device)).t()
        angles = offsets[:, 0:1] * angles + offsets[:, 1:2]
        motions = _joint_motions(grouped, angles, n_links, batch_size, dtype, joint_vec.device)
        local = transforms.unsqueeze(1) @ motions
        cumul = torch.empty((n_links, batch_size, 4, 4), dtype=dtype, device=joint_vec.device)
        cumul[0] = transforms[0].expand(batch_size, 4, 4)
        for link in range(1, n_links):
            torch.matmul(cumul[parents[link]], local[link], out=cumul[link])

    origins = cumul[..., :3, 3]  # (n_links, batch, 3)
    # Joint axis in world: column 0/1/2 of the link's rotation for X/Y/Z respectively.
    axes = cumul[..., :3, :3]  # (n_links, batch, 3, 3); axes[..., :, k] is column k

    chain = link_chain_map.reshape(n_links, n_links)
    gradient = torch.zeros((batch_size, n_joints), dtype=dtype)

    def accumulate(link_of_point, points, point_gradients, rotation_gradients=None):
        """Add every joint's contribution for a set of points attached to ``link_of_point``.

        ``points`` and ``point_gradients`` are (batch, k, 3): every point riding on the same link
        shares the same chain of joints, so they are contracted together. Doing one point at a time
        instead costs 349 spheres x 16 links of Python per call, which measured 960x slower than
        CUDA -- the worst ratio of any kernel here -- for identical output.
        """
        for j in range(n_links):
            joint_type = types[j]
            if joint_type == FIXED or int(chain[link_of_point, j]) == 0:
                continue
            column = joint_type % 3  # X/Y/Z for both prismatic (0..2) and revolute (3..5)
            axis = axes[j][..., column].unsqueeze(1)  # (batch, 1, 3), broadcasts over k
            sign = axis_sign[j]
            if joint_type <= Z_PRISM:
                contribution = (axis * point_gradients).sum(dim=(-1, -2))
            else:
                lever = points - origins[j].unsqueeze(1)
                contribution = (
                    torch.cross(axis.expand_as(lever), lever, dim=-1) * point_gradients
                ).sum(dim=(-1, -2))
                if rotation_gradients is not None:
                    contribution = contribution + (axis * rotation_gradients).sum(dim=(-1, -2))
            gradient[:, int(joint_index[j])] += sign * contribution

    # Spheres contribute only through position -- a sphere has no orientation.
    spheres = robot_spheres.reshape(-1, 4).to(dtype)
    sphere_grads = grad_spheres.reshape(batch_size, n_spheres, 4).to(dtype)[..., :3]
    for link, sphere_index in _sphere_groups(link_sphere_map):
        gradients = sphere_grads[:, sphere_index, :]  # (batch, k, 3)
        if sparsity_opt and not bool(gradients.any()):
            continue
        frame = cumul[link]
        local_points = spheres[sphere_index, :3]
        world = local_points @ frame[:, :3, :3].transpose(1, 2) + frame[:, None, :3, 3]
        accumulate(link, world, gradients)

    stored = store_link_map.reshape(-1).tolist()
    position_grads = grad_link_pos.reshape(batch_size, len(stored), 3).to(dtype)
    quaternion_grads = grad_link_quat.reshape(batch_size, len(stored), 4).to(dtype)
    for index, link in enumerate(stored):
        position_gradient = position_grads[:, index, :]
        rotation_gradient = quaternion_grads[:, index, 1:]  # vector part only
        if sparsity_opt and not (
            bool(position_gradient.any()) or bool(rotation_gradient.any())
        ):
            continue
        accumulate(
            int(link),
            origins[int(link)].unsqueeze(1),
            position_gradient.unsqueeze(1),
            rotation_gradient.unsqueeze(1),
        )

    grad_out.reshape(batch_size, -1)[:, :n_joints] = gradient.to(grad_out.dtype)
    return [grad_out]


def forward(
    link_pos: torch.Tensor,
    link_quat: torch.Tensor,
    b_robot_spheres: torch.Tensor,
    global_cumul_mat: torch.Tensor,
    joint_vec: torch.Tensor,
    fixed_transform: torch.Tensor,
    robot_spheres: torch.Tensor,
    link_map: torch.Tensor,
    joint_map: torch.Tensor,
    joint_map_type: torch.Tensor,
    store_link_map: torch.Tensor,
    link_sphere_map: torch.Tensor,
    joint_offset_map: torch.Tensor,
    batch_size: int,
    n_joints: int,
    n_spheres: int,
    use_global_cumul: bool = False,
):
    """Mirrors ``kinematics_fused_cu.forward``. Writes in place, returns the four out-tensors."""
    device, dtype = joint_vec.device, torch.float32
    transforms = fixed_transform.reshape(-1, 4, 4).to(dtype)
    n_links = transforms.shape[0]

    parents, joint_index, grouped, offsets = _structure(
        link_map, joint_map, joint_map_type, joint_offset_map, n_links
    )

    # update_axis_direction (kernel:325): a negated axis is folded into the angle as scale/offset
    # rather than carried as a separate joint type. Applied to every link at once; fixed links
    # produce a value that the identity motion then ignores.
    angles = joint_vec.to(dtype).index_select(1, joint_index.to(joint_vec.device)).t()
    angles = offsets[:, 0:1] * angles + offsets[:, 1:2]

    motions = _joint_motions(grouped, angles, n_links, batch_size, dtype, device)
    # Fold the fixed transform into the motion for every link in one batched matmul, so the
    # sequential walk below is a single matmul per link instead of two.
    local = transforms.unsqueeze(1) @ motions  # (n_links, batch, 4, 4)

    cumul = torch.empty((n_links, batch_size, 4, 4), dtype=dtype, device=device)
    cumul[0] = transforms[0].expand(batch_size, 4, 4)
    for link in range(1, n_links):
        # Genuinely sequential: this link's frame is defined relative to its parent's.
        torch.matmul(cumul[parents[link]], local[link], out=cumul[link])

    if use_global_cumul:
        global_cumul_mat.view(batch_size, n_links, 4, 4).copy_(
            cumul.permute(1, 0, 2, 3).to(global_cumul_mat.dtype)
        )

    stored = cumul[store_link_map.long()]  # (store_n_links, batch, 4, 4)
    link_pos.copy_(stored[..., :3, 3].permute(1, 0, 2).to(link_pos.dtype))
    store_n_links = stored.shape[0]
    # ``link_quat`` is **wxyz**, unlike ``matrix_to_quaternion``'s xyzw: the forward kernel calls
    # ``mat_to_quat`` (kernel:90, writes q[0]=w) where the standalone op calls ``rot_to_quat``
    # (kernel:146, writes q.w last). Two orders in one file. Established by testing every
    # permutation against the compiled kernel -- (3,0,1,2) with residual 2.4e-7 -- because nothing
    # in the source says so and getting it wrong is silent: positions stay perfect while every
    # orientation is scrambled.
    quaternions = _rot_to_quat(
        stored[..., :3, :3].permute(1, 0, 2, 3).reshape(-1, 3, 3)
    )
    link_quat.copy_(
        quaternions[:, [3, 0, 1, 2]]
        .reshape(batch_size, store_n_links, 4)
        .to(link_quat.dtype)
    )

    # Spheres ride on their link: centre through the link transform, radius passed through. Done
    # one link at a time so each link's 4x4 is read once rather than copied per sphere.
    #
    # n_spheres can be zero: cuRobo calls this during robot construction purely to read link poses
    # (``cuda_robot_generator._get_link_poses``) and passes an empty sphere buffer, so the loop must
    # not run rather than index into a zero-width tensor.
    if n_spheres == 0 or b_robot_spheres.shape[-2] == 0:
        return [link_pos, link_quat, b_robot_spheres, global_cumul_mat]

    spheres = robot_spheres.reshape(-1, 4).to(dtype)
    for link, sphere_index in _sphere_groups(link_sphere_map):
        frame = cumul[link]  # (batch, 4, 4)
        local = spheres[sphere_index, :3]  # (k, 3)
        centres = local @ frame[:, :3, :3].transpose(1, 2) + frame[:, None, :3, 3]
        b_robot_spheres[:, sphere_index, :3] = centres.to(b_robot_spheres.dtype)
    b_robot_spheres[..., 3] = spheres[:, 3].expand(batch_size, n_spheres).to(
        b_robot_spheres.dtype
    )

    return [link_pos, link_quat, b_robot_spheres, global_cumul_mat]
