"""Arm kinematics: sampled segments -> concrete link frames, FK, and the degeneracy screen.

A segment carries a full quaternion and a full 3-vector offset relative to its parent, not a
token and a length. That is deliberate: real arms need it. Franka's FR3 places nearly all of
its structure *laterally* (offsets of 0.32, 0.08, 0.39 m with zero axial rise, because after a
90 degrees twist the chain runs sideways in the local frame), and 4 of YAM's 6 body
orientations are not any multiple of 90 degrees. A token-and-length parameterization cannot
represent either, so it could never contain them.

Tokens survive as a *sampling* device only -- see :mod:`.params`, where a segment quaternion is
a token composed with a perturbation. The token concentrates density on DH-like arms; the
perturbation is what makes the support all of SO(3).

Joint axes are always the body's local +z, and every emitted axis is still recorded explicitly
so nothing downstream re-derives the convention.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

S2 = float(np.sqrt(0.5))

# Twist tokens: quaternions (wxyz) that quarter-turn the chain. Sampling seeds, not a basis.
TWISTS = {
    "I": (1.0, 0.0, 0.0, 0.0),
    "X+": (S2, S2, 0.0, 0.0),
    "X-": (S2, -S2, 0.0, 0.0),
    "Y+": (S2, 0.0, S2, 0.0),
    "Y-": (S2, 0.0, -S2, 0.0),
}


def quat_to_mat(quat) -> np.ndarray:
    """wxyz -> 3x3. MuJoCo is scalar-first, scipy is scalar-last."""
    w, x, y, z = (float(v) for v in quat)
    return Rotation.from_quat([x, y, z, w]).as_matrix()


def mat_to_quat(matrix) -> tuple:
    """3x3 -> wxyz."""
    x, y, z, w = Rotation.from_matrix(np.asarray(matrix, float)).as_quat()
    return (float(w), float(x), float(y), float(z))


def quat_mul(a, b) -> tuple:
    """Hamilton product of two wxyz quaternions."""
    aw, ax, ay, az = (float(v) for v in a)
    bw, bx, by, bz = (float(v) for v in b)
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


@dataclass(frozen=True)
class LinkFrame:
    """Joint ``index``'s body frame, plus where that joint's axis points in the base frame."""

    index: int
    pos: tuple            # body position relative to its parent
    quat: tuple           # body orientation (wxyz) relative to its parent
    axis_world: tuple     # joint axis in the base frame at zero qpos (recorded, not assumed)
    origin_world: tuple   # joint origin in the base frame at zero qpos


def axes_at_zero(quats) -> np.ndarray:
    """``(n, 3)`` joint axes in the base frame at zero qpos: each cumulative rotation's z column."""
    rotation, axes = np.eye(3), []
    for quat in quats:
        rotation = rotation @ quat_to_mat(quat)
        axes.append(rotation[:, 2].copy())
    return np.stack(axes)


def axes_ok(axes: np.ndarray) -> bool:
    """At least 3 distinct axis directions, and no consecutive collinear pair about a vertical
    axis. Rejects wrists that cannot orient, and coaxial roll-roll pairs that waste a DoF."""
    distinct = []
    for axis in axes:
        if all(abs(float(axis @ seen)) < 0.95 for seen in distinct):
            distinct.append(axis)
    if len(distinct) < 3:
        return False
    for first, second in zip(axes[:-1], axes[1:]):
        if abs(float(first @ second)) > 0.95 and abs(float(first[2])) > 0.95:
            return False
    return True


def chain_at_zero(arm):
    """FK at zero qpos. Returns ``(origins, axes, flange)`` in the arm's base frame."""
    rotation, position = np.eye(3), np.zeros(3)
    origins, axes = [], []
    for segment in arm.segments:
        # MuJoCo semantics: a body's origin is parent_pos + parent_rot @ body_pos, and its
        # orientation is parent_rot @ body_quat. The offset is therefore rotated by the PARENT's
        # rotation -- applying the child's instead silently shortens every chain (it measured
        # YAM, a 0.7 m arm, at 0.14 m).
        position = position + rotation @ np.asarray(segment.offset, float)
        rotation = rotation @ quat_to_mat(segment.quat)
        origins.append(position.copy())
        axes.append(rotation[:, 2].copy())
    return np.stack(origins), np.stack(axes), position.copy()


def build_frames(arm) -> list:
    """Realize an :class:`~.params.ArmParams` as one :class:`LinkFrame` per joint."""
    origins, axes, _ = chain_at_zero(arm)
    return [LinkFrame(index=i + 1,
                      pos=tuple(float(v) for v in segment.offset),
                      quat=tuple(float(v) for v in segment.quat),
                      axis_world=tuple(float(v) for v in axes[i]),
                      origin_world=tuple(float(v) for v in origins[i]))
            for i, segment in enumerate(arm.segments)]


def flange_offset(arm) -> tuple:
    """The flange sits at the last segment's tip, i.e. at the last joint's origin. The graft
    point is that body's own frame, so the offset is zero -- kept explicit for the builder."""
    return (0.0, 0.0, 0.0)


def chain_length(arm) -> float:
    """Total kinematic length: the summed norms of the segment offsets.

    This is the arm's SCALE, and it is what the sampler budgets. It is an upper bound on
    workspace radius, not the radius itself -- FR3's chain is 1.212 m long but it reaches
    ~0.86 m, because its offsets are mutually perpendicular and its joint limits forbid full
    extension.

    Distance-to-flange at zero qpos is deliberately NOT used: it measures the home pose rather
    than the arm. YAM is folded at ``qpos=0``, so that metric calls a 0.7 m arm 0.20 m, and
    budgeting against it would size every generated arm by an arbitrary convention."""
    return float(sum(np.linalg.norm(np.asarray(s.offset, float)) for s in arm.segments))


def flange_at_zero(arm) -> np.ndarray:
    """Flange position in the base frame at zero qpos -- for posing and rendering, not sizing."""
    return chain_at_zero(arm)[2]
