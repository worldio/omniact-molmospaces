"""EMBODIMENT SPEC -> CUROBO ROBOT MODEL. The step that lets any embodiment be planned for.

cuRobo is configured by a ``robot_cfg`` dict naming a URDF, a base and end-effector link, a set of
collision spheres per link, and a cspace. MolmoSpaces ships exactly one such config, hand-authored
for RB-Y1 (``assets/robots/rby1m/curobo_config/``). Hand-authoring is why
``docs/embodiment_generation.md`` records that the ``curobo_*`` policies "need per-robot collision
configs, which is fatal at 1M robots".

Everything here is DERIVED from the assembled MuJoCo model, so a generated robot and a shipped one
go through the same path:

* the URDF is the MJCF body tree, one link per body, one joint per joint
* collision spheres come from the MJCF's own collision geoms
* ``self_collision_ignore`` mirrors MuJoCo's own contact filtering, so the planner's self-collision
  model agrees with the simulator's rather than with a hand-written list
* the cspace is the arm move group and its rest pose, straight from the spec

Two simplifications carry most of the weight, and both are deliberate:

**The URDF carries no geometry.** cuRobo takes collision from ``collision_spheres``; its URDF parser
only touches link meshes for ``mesh_link_names``. So no mesh is exported or converted, and the URDF
is a pure kinematic skeleton. (Verified against the cuRobo source: ``UrdfKinematicsParser.
get_link_parameters`` reads geometry only when asked for a mesh.)

**Everything that is not an arm DoF is locked.** Grippers are closed kinematic chains -- tendons and
equality constraints that URDF cannot express at all. Locking them at their open pose turns the
whole tool into one rigid attachment whose swept volume is still modelled by its spheres, which is
what the shipped RB-Y1 config does too (``lock_joints: {gripper_finger_r1: -0.05, ...}``). The
planner then solves for the arm only, which is the only thing the pick policies command.
"""

from __future__ import annotations

import hashlib
import logging
import math
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.robots.configured.names import resolve_model_name
from molmo_spaces.robots.configured.spec import (
    GripperMoveGroupSpec,
    JointMoveGroupSpec,
)

if TYPE_CHECKING:
    from molmo_spaces.robots.configured.config import ConfiguredRobotConfig

log = logging.getLogger(__name__)

# A URDF link needs an inertial or some parsers reject it. Kinematics and sphere collision never
# read these numbers -- cuRobo takes dynamics limits from the cspace -- so a unit placeholder is
# honest about being unused rather than inventing a plausible-looking mass.
_PLACEHOLDER_MASS = 1.0
_PLACEHOLDER_INERTIA = 1e-4

_SUPPORTED_JOINT_TYPES = {
    mujoco.mjtJoint.mjJNT_HINGE: "revolute",
    mujoco.mjtJoint.mjJNT_SLIDE: "prismatic",
}


@dataclass
class CuroboRobotModel:
    """A cuRobo robot description derived from one embodiment."""

    robot_cfg: dict
    urdf_path: Path
    dof_joint_names: list[str]
    """Planned joints, in the order cuRobo will report them."""
    locked_joint_names: list[str]
    base_link: str
    ee_link: str
    tcp_site_name: str
    """MuJoCo site the ``ee_link`` was placed on; FK parity is checked against it."""
    sphere_counts: dict[str, int] = field(default_factory=dict)
    static_body_names: list[str] = field(default_factory=list)
    """Robot links no planned joint can move -- the pedestal and whatever is bolted to it.

    Deliberately absent from the sphere model: see the note where they are excluded. The caller is
    expected to add them to the planning world instead, so the arm still avoids them."""


def _quat_to_rpy(quat_wxyz: np.ndarray) -> tuple[float, float, float]:
    """URDF orients with fixed-axis roll-pitch-yaw; MuJoCo stores a wxyz quaternion."""
    rotation = R.from_quat(np.asarray(quat_wxyz, dtype=float), scalar_first=True)
    # Gimbal lock makes the triple non-unique, never inexact: any triple scipy returns rebuilds
    # the same rotation, which is all URDF needs. The warning would otherwise fire on every
    # axis-aligned joint frame, and those are the common case.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return tuple(float(v) for v in rotation.as_euler("xyz"))


def _body_joints(model: mujoco.MjModel, body_id: int) -> list[int]:
    start = int(model.body_jntadr[body_id])
    count = int(model.body_jntnum[body_id])
    if start < 0 or count <= 0:
        return []
    return list(range(start, start + count))


def _subtree_bodies(model: mujoco.MjModel, root_body_id: int) -> list[int]:
    """Body ids of ``root`` and every descendant, parents before children."""
    ordered = [root_body_id]
    for body_id in range(root_body_id + 1, model.nbody):
        if int(model.body_parentid[body_id]) in ordered:
            ordered.append(body_id)
    return ordered


def _link_offset(model: mujoco.MjModel, body_id: int) -> np.ndarray:
    """Where this body's link frame sits relative to its MuJoCo body frame.

    URDF requires a joint's axis to pass through its child link's ORIGIN, while MuJoCo anchors a
    joint at ``jnt_pos`` in the body frame. Putting the link origin on the anchor makes the two
    agree exactly; every geom and site in the body is then expressed relative to that origin. The
    offset is zero for every joint of every robot checked so far, but a model that anchors away
    from the body origin would otherwise be silently mis-converted.
    """
    joints = _body_joints(model, body_id)
    if len(joints) != 1:
        return np.zeros(3)
    return np.asarray(model.jnt_pos[joints[0]], dtype=float).copy()


def _collision_geoms(model: mujoco.MjModel, body_id: int) -> list[int]:
    """Geoms MuJoCo would actually collide, i.e. those with a contact mask."""
    start = int(model.body_geomadr[body_id])
    count = int(model.body_geomnum[body_id])
    if start < 0 or count <= 0:
        return []
    return [
        geom_id
        for geom_id in range(start, start + count)
        if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
    ]


def _spheres_along_segment(
    start: np.ndarray, end: np.ndarray, radius: float, spacing_ratio: float
) -> list[tuple[np.ndarray, float]]:
    length = float(np.linalg.norm(end - start))
    steps = max(1, int(math.ceil(length / max(radius * spacing_ratio, 1e-6))))
    return [
        (start + (end - start) * (index / steps), radius)
        for index in range(steps + 1)
    ]


def _geom_spheres(
    model: mujoco.MjModel, geom_id: int, spacing_ratio: float, max_spheres: int
) -> list[tuple[np.ndarray, float]]:
    """Spheres covering one geom, in that geom's parent body frame.

    Every shape is covered CONSERVATIVELY (the union of spheres contains the geom). An
    under-covering fit would let the planner drive the robot through an obstacle; an over-covering
    one only costs plan success, and is visible as an inflated robot.
    """
    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=float)
    position = np.asarray(model.geom_pos[geom_id], dtype=float)
    rotation = R.from_quat(
        np.asarray(model.geom_quat[geom_id], dtype=float), scalar_first=True
    ).as_matrix()

    if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        return [(position, float(size[0]))]

    if geom_type in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
        # MuJoCo lays both out along local z with size = (radius, half-length). A cylinder is
        # covered by its capsule hull, which bulges past the flat caps -- conservative, and the
        # alternative is a much denser disc fill for geometry that is rarely the binding one.
        radius, half_length = float(size[0]), float(size[1])
        axis = rotation @ np.array([0.0, 0.0, 1.0])
        return _spheres_along_segment(
            position - axis * half_length, position + axis * half_length, radius, spacing_ratio
        )

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        # Lattice fill: one sphere per cell, radius = the cell's half-diagonal, so the union
        # provably contains the box.
        #
        # The cell size is grown until the count fits `max_spheres`. Without that cap a thin plate
        # -- a finger pad is ~1 mm x 20 mm x 30 mm -- lattices at the 1 mm thickness and emits
        # hundreds of spheres, and cuRobo's self-collision cost is quadratic in sphere count. The
        # cap trades fidelity for size in the only direction that is safe: cells only ever grow,
        # so the union still contains the box, it just contains more around it.
        cell_target = float(np.min(size))
        for _ in range(32):
            counts = [
                max(1, int(math.ceil(float(extent) / max(cell_target, 1e-6)))) for extent in size
            ]
            if counts[0] * counts[1] * counts[2] <= max_spheres:
                break
            cell_target *= 1.5
        cell = np.array([float(size[axis]) / counts[axis] for axis in range(3)])
        radius = float(np.linalg.norm(cell))
        spheres = []
        for i in range(counts[0]):
            for j in range(counts[1]):
                for k in range(counts[2]):
                    local = np.array(
                        [
                            -size[0] + cell[0] * (2 * i + 1),
                            -size[1] + cell[1] * (2 * j + 1),
                            -size[2] + cell[2] * (2 * k + 1),
                        ]
                    )
                    spheres.append((position + rotation @ local, radius))
        return spheres

    if geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        # Cover along the longest semi-axis with spheres of the largest remaining radius.
        order = np.argsort(size)[::-1]
        major, radius = float(size[order[0]]), float(size[order[1]])
        axis = rotation @ np.eye(3)[order[0]]
        span = max(major - radius, 0.0)
        return _spheres_along_segment(
            position - axis * span, position + axis * span, radius, spacing_ratio
        )

    if geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        # A mesh's own vertices bound it exactly; cover them with spheres sized to the vertex
        # spacing. cuRobo's fit_spheres_to_mesh is the better tool and can replace this, but it
        # needs a torch/CUDA context that model construction should not require.
        mesh_id = int(model.geom_dataid[geom_id])
        start = int(model.mesh_vertadr[mesh_id])
        count = int(model.mesh_vertnum[mesh_id])
        vertices = np.asarray(model.mesh_vert[start : start + count], dtype=float)
        if count == 0:
            return []
        centre = vertices.mean(axis=0)
        radius = float(np.max(np.linalg.norm(vertices - centre, axis=1)))
        return [(position + rotation @ centre, radius)]

    return []


def _link_spheres(
    model: mujoco.MjModel,
    body_id: int,
    offset: np.ndarray,
    spacing_ratio: float,
    buffer: float,
    max_spheres: int,
) -> list[dict]:
    spheres = []
    for geom_id in _collision_geoms(model, body_id):
        for centre, radius in _geom_spheres(model, geom_id, spacing_ratio, max_spheres):
            spheres.append(
                {
                    "center": [float(v) for v in (centre - offset)],
                    "radius": float(radius + buffer),
                }
            )
    return spheres


def _self_collision_ignore(
    model: mujoco.MjModel, bodies: list[int], names: dict[int, str], rigid_groups: list[set[int]]
) -> dict[str, list[str]]:
    """Mirror MuJoCo's own contact filtering instead of hand-listing pairs.

    MuJoCo skips a geom pair when the two bodies share a weld (are not separated by a joint) or are
    in a parent/child weld relationship, and when the pair is explicitly excluded. Anything locked
    into rigidity by ``lock_joints`` is added on top: those links cannot move relative to each
    other, so a collision between them is a fact about the model, not about the plan.
    """
    ignore: dict[str, set[str]] = {names[body_id]: set() for body_id in bodies}

    def add(first: int, second: int) -> None:
        if first in names and second in names and first != second:
            ignore[names[first]].add(names[second])
            ignore[names[second]].add(names[first])

    for body_id in bodies:
        parent_id = int(model.body_parentid[body_id])
        add(body_id, parent_id)
        if int(model.body_weldid[body_id]) != body_id:
            add(body_id, int(model.body_weldid[body_id]))

    for exclude_index in range(model.nexclude):
        signature = int(model.exclude_signature[exclude_index])
        add(signature >> 16, signature & 0xFFFF)

    for group in rigid_groups:
        for first in group:
            for second in group:
                add(first, second)

    return {link: sorted(peers) for link, peers in ignore.items() if peers}


def _calibrate_ignore(
    model: mujoco.MjModel,
    collision_spheres: dict[str, list[dict]],
    ignore: dict[str, list[str]],
    dof_joint_names: list[str],
    locked_values: dict[str, float],
    samples: int,
    seed: int,
    overlap_fraction: float,
) -> dict[str, list[str]]:
    """Add link pairs whose spheres overlap but whose real geometry never touches.

    Sphere covering is conservative, so every sphere bulges outside the geom it covers. On a large
    box the bulge is the cell half-diagonal, which for a robot pedestal is centimetres -- enough to
    swallow the arm mounted on top of it. The result is a model where EVERY configuration is
    self-colliding and no plan can ever succeed, which is what a hand-written
    ``self_collision_ignore`` (RB-Y1 ships one) exists to prevent.

    So measure it. Sample the arm's workspace, and for each link pair record how often the spheres
    overlap and whether MuJoCo's own collision detector ever reports the real geoms touching. A
    pair that overlaps systematically and never actually touches is an artefact of the covering,
    and is ignored. A pair that MuJoCo does put in contact is left alone however much it overlaps:
    that is a real self-collision, and silencing it would let the planner fold the arm into itself.
    """
    links = sorted(collision_spheres)
    body_ids = {link: model.body(link).id for link in links}
    centres = {
        link: np.array([sphere["center"] for sphere in collision_spheres[link]]) for link in links
    }
    radii = {
        link: np.array([sphere["radius"] for sphere in collision_spheres[link]]) for link in links
    }
    geom_bodies = {
        link: {int(model.geom_bodyid[g]) for g in _collision_geoms(model, body_ids[link])}
        for link in links
    }

    data = mujoco.MjData(model)
    limits = np.array([model.jnt_range[model.joint(name).id] for name in dof_joint_names])
    rng = np.random.default_rng(seed)

    overlaps = {pair: 0 for pair in _link_pairs(links, ignore)}
    touches = {pair: 0 for pair in overlaps}

    for _ in range(samples):
        values = dict(zip(dof_joint_names, rng.uniform(limits[:, 0], limits[:, 1]), strict=True))
        values.update(locked_values)
        for name, value in values.items():
            data.qpos[model.jnt_qposadr[model.joint(name).id]] = value
        mujoco.mj_forward(model, data)

        world = {}
        for link in links:
            body_id = body_ids[link]
            rotation = data.xmat[body_id].reshape(3, 3)
            world[link] = data.xpos[body_id] + centres[link] @ rotation.T

        contact_pairs = set()
        for index in range(data.ncon):
            contact = data.contact[index]
            first = int(model.geom_bodyid[contact.geom1])
            second = int(model.geom_bodyid[contact.geom2])
            contact_pairs.add((first, second))
            contact_pairs.add((second, first))

        for pair in overlaps:
            first, second = pair
            distances = np.linalg.norm(
                world[first][:, None, :] - world[second][None, :, :], axis=-1
            )
            if np.any(distances - radii[first][:, None] - radii[second][None, :] < 0.0):
                overlaps[pair] += 1
            if any(
                (a, b) in contact_pairs
                for a in geom_bodies[first]
                for b in geom_bodies[second]
            ):
                touches[pair] += 1

    calibrated = {link: set(peers) for link, peers in ignore.items()}
    added = 0
    for pair, count in overlaps.items():
        # The score is how often the sphere model claims a collision MuJoCo does not see. A pair
        # that is genuinely reachable into contact scores near zero (overlaps ~ touches) and is
        # kept; a pair the covering merely fattens into contact scores near one. Requiring
        # touches == 0 instead would keep every pair that can EVER collide -- including a pedestal
        # the arm can reach at full extension -- and those are exactly the fat ones.
        spurious = (count - touches[pair]) / max(samples, 1)
        if spurious >= overlap_fraction:
            first, second = pair
            calibrated.setdefault(first, set()).add(second)
            calibrated.setdefault(second, set()).add(first)
            added += 1
    if added:
        log.info(
            f"[curobo-bridge] calibrated away {added} sphere-artefact link pairs over "
            f"{samples} samples"
        )
    return {link: sorted(peers) for link, peers in calibrated.items() if peers}


def _link_pairs(links: list[str], ignore: dict[str, list[str]]) -> list[tuple[str, str]]:
    return [
        (first, second)
        for index, first in enumerate(links)
        for second in links[index + 1 :]
        if second not in ignore.get(first, [])
    ]


def _urdf_xml(
    model: mujoco.MjModel,
    bodies: list[int],
    names: dict[int, str],
    offsets: dict[int, np.ndarray],
    root_body_id: int,
    ee_link: str,
    ee_parent_id: int,
    ee_transform: tuple[np.ndarray, np.ndarray],
    extra_links: dict[str, str],
    robot_name: str,
) -> str:
    """A geometry-free URDF: links, joints, limits. Nothing cuRobo does not read."""
    robot = ET.Element("robot", {"name": robot_name})

    def add_link(name: str) -> None:
        link = ET.SubElement(robot, "link", {"name": name})
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(inertial, "mass", {"value": str(_PLACEHOLDER_MASS)})
        ET.SubElement(
            inertial,
            "inertia",
            {
                "ixx": str(_PLACEHOLDER_INERTIA), "ixy": "0", "ixz": "0",
                "iyy": str(_PLACEHOLDER_INERTIA), "iyz": "0",
                "izz": str(_PLACEHOLDER_INERTIA),
            },
        )

    for body_id in bodies:
        add_link(names[body_id])

    for body_id in bodies:
        if body_id == root_body_id:
            continue
        parent_id = int(model.body_parentid[body_id])
        body_position = np.asarray(model.body_pos[body_id], dtype=float)
        body_rotation = R.from_quat(
            np.asarray(model.body_quat[body_id], dtype=float), scalar_first=True
        )
        # Link frames sit on their joint anchors, so the parent's offset comes off the translation
        # and the child's goes back on through the child's own rotation.
        origin = body_position - offsets[parent_id] + body_rotation.as_matrix() @ offsets[body_id]
        roll, pitch, yaw = _quat_to_rpy(model.body_quat[body_id])

        joint_ids = _body_joints(model, body_id)
        if len(joint_ids) > 1:
            raise NotImplementedError(
                f"Body {names[body_id]!r} carries {len(joint_ids)} joints; URDF allows one per "
                "link. Split it into intermediate bodies in the MJCF."
            )

        if not joint_ids:
            joint_name, joint_type = f"{names[body_id]}_fixed", "fixed"
            axis = None
            limits = None
        else:
            joint_id = joint_ids[0]
            mj_type = mujoco.mjtJoint(int(model.jnt_type[joint_id]))
            if mj_type not in _SUPPORTED_JOINT_TYPES:
                raise NotImplementedError(
                    f"Joint {model.joint(joint_id).name!r} is {mj_type.name}, which has no URDF "
                    "equivalent. Free and ball joints must be modelled as a fixed base."
                )
            joint_name = model.joint(joint_id).name
            axis = np.asarray(model.jnt_axis[joint_id], dtype=float)
            # Locked joints stay REAL joints here and are pinned through `lock_joints`, the way
            # the shipped RB-Y1 config pins its fingers. Emitting them as fixed instead would
            # drop them from cuRobo's joint list, and cuRobo resolves `lock_joints` by name
            # against that list -- it raises rather than ignoring the entry.
            joint_type = _SUPPORTED_JOINT_TYPES[mj_type]
            if model.jnt_limited[joint_id]:
                limits = (
                    float(model.jnt_range[joint_id][0]),
                    float(model.jnt_range[joint_id][1]),
                )
            else:
                joint_type = "continuous" if joint_type == "revolute" else joint_type
                limits = None

        joint = ET.SubElement(robot, "joint", {"name": joint_name, "type": joint_type})
        ET.SubElement(
            joint,
            "origin",
            {
                "xyz": " ".join(f"{v:.10g}" for v in origin),
                "rpy": f"{roll:.10g} {pitch:.10g} {yaw:.10g}",
            },
        )
        ET.SubElement(joint, "parent", {"link": names[parent_id]})
        ET.SubElement(joint, "child", {"link": names[body_id]})
        if joint_type in ("revolute", "prismatic", "continuous") and axis is not None:
            ET.SubElement(joint, "axis", {"xyz": " ".join(f"{v:.10g}" for v in axis)})
        if limits is not None:
            ET.SubElement(
                joint,
                "limit",
                {
                    "lower": f"{limits[0]:.10g}", "upper": f"{limits[1]:.10g}",
                    "effort": "100", "velocity": "2.0",
                },
            )

    # The TCP. The whole point of the bridge: cuRobo's ee_link must be the very frame the grasp
    # machinery already targets (the gripper move group's leaf frame), or every pose handed to the
    # planner is silently offset from the one the policy computed.
    tcp_position, tcp_quat = ee_transform
    add_link(ee_link)
    tcp_joint = ET.SubElement(robot, "joint", {"name": f"{ee_link}_fixed", "type": "fixed"})
    roll, pitch, yaw = _quat_to_rpy(tcp_quat)
    ET.SubElement(
        tcp_joint,
        "origin",
        {
            "xyz": " ".join(f"{v:.10g}" for v in (tcp_position - offsets[ee_parent_id])),
            "rpy": f"{roll:.10g} {pitch:.10g} {yaw:.10g}",
        },
    )
    ET.SubElement(tcp_joint, "parent", {"link": names[ee_parent_id]})
    ET.SubElement(tcp_joint, "child", {"link": ee_link})

    for link_name, parent_name in extra_links.items():
        add_link(link_name)
        extra_joint = ET.SubElement(
            robot, "joint", {"name": f"{link_name}_fixed", "type": "fixed"}
        )
        ET.SubElement(extra_joint, "origin", {"xyz": "0 0 0", "rpy": "0 0 0"})
        ET.SubElement(extra_joint, "parent", {"link": parent_name})
        ET.SubElement(extra_joint, "child", {"link": link_name})

    ET.indent(robot, space="  ")
    return ET.tostring(robot, encoding="unicode")


def _gripper_open_qpos(group: GripperMoveGroupSpec, model: mujoco.MjModel, names: list[str]):
    """Joint values for the gripper's open pose.

    Locked at OPEN rather than at the spec's initial qpos because the pick policies open the
    gripper before every transit; planning a closed jaw would under-model the swept volume exactly
    when the tool is at its widest.
    """
    joint_limits = []
    for joint_name in names:
        joint_id = model.joint(joint_name).id
        joint_limits.append(
            (float(model.jnt_range[joint_id][0]), float(model.jnt_range[joint_id][1]))
        )
    command = float(np.max(np.abs(group.open_command)))
    # Each finger travels to whichever limit is further from zero in its own direction, capped by
    # the commanded opening, so mirrored fingers stay mirrored.
    values = []
    for lower, upper in joint_limits:
        values.append(min(upper, command) if abs(upper) >= abs(lower) else max(lower, -command))
    return values


def build_curobo_robot_model(
    robot_config: ConfiguredRobotConfig,
    output_dir: str | Path,
    *,
    arm_group_ids: list[str] | None = None,
    sphere_spacing_ratio: float = 1.0,
    max_spheres_per_geom: int = 12,
    collision_sphere_buffer: float = 0.005,
    self_collision_buffer: float = 0.0,
    max_acceleration: float = 15.0,
    max_jerk: float = 500.0,
    calibrate_self_collision: bool = True,
    calibration_samples: int = 200,
    calibration_seed: int = 0,
    calibration_overlap_fraction: float = 0.25,
) -> CuroboRobotModel:
    """Build a cuRobo ``robot_cfg`` for one configured embodiment.

    Args:
        robot_config: the embodiment, already loaded through ``ConfiguredRobotConfig``.
        output_dir: where the generated URDF is written. The file name is keyed by embodiment id
            and version so concurrent workers on one embodiment converge on the same path.
        arm_group_ids: move groups to plan for. Defaults to every ``role == "arm"`` group.
        sphere_spacing_ratio: sphere spacing along a capsule, in radii. Smaller is tighter and
            slower.
        max_spheres_per_geom: cap on the lattice fill of one box. Lower inflates thin geometry.
        collision_sphere_buffer: padding added to every sphere, in metres.
        calibrate_self_collision: measure which link pairs overlap only because spheres bulge,
            and ignore those. Without it a bulging pedestal makes every configuration collide.
    """
    from molmo_spaces.robots.configured.robot import ConfiguredRobot

    embodiment = robot_config.embodiment
    namespace = robot_config.robot_namespace
    view = embodiment.robot_view

    spec = mujoco.MjSpec()
    ConfiguredRobot.add_robot_to_scene(
        robot_config, spec, prefix=namespace, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0]
    )
    model = spec.compile()

    if arm_group_ids is None:
        arm_group_ids = [
            group_id
            for group_id, group in view.move_groups.items()
            if isinstance(group, JointMoveGroupSpec) and group.role == "arm"
        ]
    if not arm_group_ids:
        raise ValueError(f"Embodiment {embodiment.embodiment_id!r} declares no arm move group")

    dof_joint_names = [
        resolve_model_name(reference, namespace)
        for group_id in arm_group_ids
        for reference in view.move_groups[group_id].joint_names
    ]

    # The base link is the robot's base move group body, so cuRobo's base frame is exactly the one
    # `Robot.get_world_pose_tf_mat()` reports and the policy's world->base transform stays valid.
    base_group = view.move_groups[view.base_group]
    base_body_name = resolve_model_name(
        getattr(base_group, "body", None) or getattr(base_group, "root_body"), namespace
    )
    base_body_id = model.body(base_body_name).id

    bodies = _subtree_bodies(model, base_body_id)
    names = {body_id: model.body(body_id).name for body_id in bodies}
    offsets = {body_id: _link_offset(model, body_id) for body_id in bodies}

    # Lock every movable joint that is not a planned DoF, and record which links that welds
    # together so self-collision does not fire between them.
    locked_values: dict[str, float] = {}
    rigid_groups: list[set[int]] = []
    for group_id, group in view.move_groups.items():
        if group_id in arm_group_ids or not isinstance(
            group, (JointMoveGroupSpec, GripperMoveGroupSpec)
        ):
            continue
        joint_names = [resolve_model_name(ref, namespace) for ref in group.joint_names]
        if isinstance(group, GripperMoveGroupSpec):
            values = _gripper_open_qpos(group, model, joint_names)
        else:
            values = list(robot_config.init_qpos.get(group_id, [0.0] * len(joint_names)))
        for joint_name, value in zip(joint_names, values, strict=False):
            locked_values[joint_name] = float(value)
        group_bodies = {
            body_id
            for body_id in bodies
            if any(
                model.joint(joint_id).name in joint_names
                for joint_id in _body_joints(model, body_id)
            )
        }
        rigid_group = set(group_bodies)
        for body_id in bodies:
            ancestor = int(model.body_parentid[body_id])
            while ancestor in names:
                if ancestor in group_bodies:
                    rigid_group.add(body_id)
                    break
                ancestor = int(model.body_parentid[ancestor])
        if rigid_group:
            rigid_groups.append(rigid_group)

    for body_id in bodies:
        for joint_id in _body_joints(model, body_id):
            joint_name = model.joint(joint_id).name
            if joint_name not in dof_joint_names and joint_name not in locked_values:
                locked_values[joint_name] = 0.0

    end_effector = next(iter(view.end_effectors.values()))
    tcp_frame = end_effector.tcp_frame
    tcp_name = resolve_model_name(tcp_frame.name, namespace)
    if tcp_frame.kind == "site":
        site_id = model.site(tcp_name).id
        ee_parent_id = int(model.site_bodyid[site_id])
        ee_transform = (
            np.asarray(model.site_pos[site_id], dtype=float),
            np.asarray(model.site_quat[site_id], dtype=float),
        )
    else:
        ee_parent_id = model.body(tcp_name).id
        ee_transform = (np.zeros(3), np.array([1.0, 0.0, 0.0, 0.0]))

    ee_link = f"{namespace}curobo_tcp".replace("/", "_")
    attached_link = f"{namespace}attached_object".replace("/", "_")

    urdf = _urdf_xml(
        model, bodies, names, offsets, base_body_id, ee_link, ee_parent_id,
        ee_transform, {attached_link: ee_link}, embodiment.embodiment_id,
    )

    identity = hashlib.sha1(
        f"{embodiment.embodiment_id}|{embodiment.embodiment_version}|"
        f"{robot_config.control_profile}|{sorted(locked_values.items())}|"
        f"{calibrate_self_collision}|{calibration_samples}|{calibration_seed}".encode()
    ).hexdigest()[:12]
    # Absolute, always: cuRobo resolves a relative urdf_path against its own installed asset root
    # (`curobo/content/assets/`), so a relative path silently points into site-packages.
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    urdf_path = output_dir / f"{embodiment.embodiment_id.replace('/', '_')}.{identity}.urdf"
    # Write-then-rename: several datagen workers build the same embodiment concurrently and must
    # never observe a half-written URDF.
    tmp_path = urdf_path.with_suffix(".urdf.tmp")
    tmp_path.write_text(urdf)
    tmp_path.replace(urdf_path)

    # Links no planned joint can move are not part of the robot as far as planning is concerned:
    # their pose is a property of where the robot was parked. Left in the sphere model they are
    # checked against the world they can never move away from, and because the sphere fit of a
    # pedestal is necessarily coarse, that check fails. Measured on i2rt_yam: three of the five
    # static spheres have a 247 mm radius, six times the largest sphere that actually moves
    # (39 mm), and parking 471 mm from a wall put the start state 79 mm inside it -- so every plan
    # in that scene failed from its first waypoint, reported only as TRAJOPT_FAIL.
    #
    # They are still obstacles. The caller receives them in `static_body_names` and puts them in
    # the world, which keeps the arm from driving through its own pedestal while letting the
    # pedestal sit as close to a wall as the base sampler placed it.
    moved_by_plan: set[int] = set()
    for joint_name in dof_joint_names:
        child = int(model.jnt_bodyid[model.joint(joint_name).id])
        moved_by_plan |= set(_subtree_bodies(model, child))
    static_bodies = [body_id for body_id in bodies if body_id not in moved_by_plan]
    # A static link that a moving link is bolted to cannot be an obstacle either: adjacency
    # guarantees their volumes overlap, so as world geometry it reports a permanent collision the
    # planner cannot escape. Measured on the arm's mounting flange: five spheres 3 mm inside it,
    # which `collision_activation_distance` then widens to a 13 mm violation. This is the same
    # exclusion every self-collision ignore list makes for adjacent links; here the interface
    # geometry simply drops out of the model, enclosed between the pedestal and the first link.
    mount_bodies = {
        body_id for body_id in static_bodies
        if any(int(model.body_parentid[moved]) == body_id for moved in moved_by_plan)
    }
    obstacle_bodies = [body_id for body_id in static_bodies if body_id not in mount_bodies]

    collision_spheres = {}
    for body_id in bodies:
        if body_id in static_bodies:
            continue
        spheres = _link_spheres(
            model, body_id, offsets[body_id], sphere_spacing_ratio, collision_sphere_buffer,
            max_spheres_per_geom,
        )
        if spheres:
            collision_spheres[names[body_id]] = spheres
    # The attached-object link must be listed here, not just given a budget in
    # `extra_collision_spheres`. cuRobo builds its collision model by walking
    # `collision_link_names`, so a link absent from it gets no spheres allocated however large its
    # budget -- attach_objects_to_robot then reports "No spheres found, max_spheres: 0" and fits
    # nothing, and a carried object silently never enters the planner's world. Upstream RB-Y1 lists
    # its two attach links here for exactly this reason.
    collision_link_names = sorted(collision_spheres) + [attached_link]

    ignore = _self_collision_ignore(model, bodies, names, rigid_groups)
    ignore = {link: peers for link, peers in ignore.items() if link in collision_spheres}
    ignore = {
        link: [peer for peer in peers if peer in collision_spheres]
        for link, peers in ignore.items()
    }
    # An object held in the jaws necessarily overlaps them, so the attached link must ignore the
    # tool it is held by or every grasped pose is self-colliding by construction. The tool is
    # whatever the locking step welded into one rigid group with the gripper.
    tool_links = sorted(
        {names[body_id] for group in rigid_groups for body_id in group if body_id in names}
        | {names[ee_parent_id]}
    )
    ignore[attached_link] = [link for link in tool_links if link in collision_spheres]
    for link in ignore[attached_link]:
        ignore.setdefault(link, []).append(attached_link)

    if calibrate_self_collision:
        ignore = _calibrate_ignore(
            model, collision_spheres, ignore, dof_joint_names, locked_values,
            calibration_samples, calibration_seed, calibration_overlap_fraction,
        )

    retract = [
        float(value)
        for group_id in arm_group_ids
        for value in robot_config.init_qpos.get(
            group_id, [0.0] * len(view.move_groups[group_id].joint_names)
        )
    ]

    robot_cfg = {
        "kinematics": {
            "use_usd_kinematics": False,
            "urdf_path": str(urdf_path),
            "asset_root_path": "",
            "base_link": names[base_body_id],
            "ee_link": ee_link,
            "collision_link_names": collision_link_names,
            "collision_spheres": collision_spheres,
            "collision_sphere_buffer": 0.0,  # already folded into each radius
            # cuRobo divides this budget evenly across the bodies handed to it
            # (`n_spheres = int(max_spheres / len(object_names))`) and spends only the first 75% on
            # volume-filling spheres, padding the rest with surface points at a FIXED 1 mm radius.
            # At 40, a seven-geom bottle got five spheres per geom: three real ones and two specks,
            # a payload the planner could route a countertop through. Four times that gives ~22 per
            # geom, so the fit is dominated by volume spheres sized from the geometry.
            "extra_collision_spheres": {attached_link: 160},
            "self_collision_ignore": ignore,
            "self_collision_buffer": {
                link: self_collision_buffer for link in collision_link_names
            },
            "use_global_cumul": True,
            "mesh_link_names": None,
            "lock_joints": locked_values,
            "extra_links": {
                attached_link: {
                    "parent_link_name": ee_link,
                    "link_name": attached_link,
                    "fixed_transform": [0, 0, 0, 1, 0, 0, 0],
                    "joint_type": "FIXED",
                    "joint_name": f"{attached_link}_attach",
                }
            },
            "cspace": {
                "joint_names": dof_joint_names,
                "retract_config": retract,
                "null_space_weight": [1.0] * len(dof_joint_names),
                "cspace_distance_weight": [1.0] * len(dof_joint_names),
                "max_jerk": max_jerk,
                "max_acceleration": max_acceleration,
            },
        }
    }

    return CuroboRobotModel(
        robot_cfg=robot_cfg,
        urdf_path=urdf_path,
        dof_joint_names=dof_joint_names,
        locked_joint_names=sorted(locked_values),
        base_link=names[base_body_id],
        ee_link=ee_link,
        tcp_site_name=tcp_name,
        sphere_counts={link: len(s) for link, s in collision_spheres.items()},
        static_body_names=[names[body_id] for body_id in obstacle_bodies],
    )
