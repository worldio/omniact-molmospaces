"""The per-embodiment NODE SET: what a morphology-agnostic consumer needs to read a trajectory.

A flattened proprio vector is uninterpretable alone -- a consumer has to know how many nodes the robot
has, which are actuated, which slide rather than rotate, where each sits, and which qpos entry drives
each finger. This derives all of it from the built model.

A node is a typed anchor, ``{"kind": "body"|"geom", "name": ...}``:

* a BODY is a node of the kinematic tree -- MuJoCo solves for its pose and it may carry joints;
* a GEOM is a shape welded to a body at a fixed ``geom_pos``, with no freedom of its own.

Both are rigid and both resolve to a world pose, and admitting geoms is what makes the terminal rule
general. A revolute joint leaves its own axis unchanged, so an actuated node cannot report its angle
from its own centre and axis alone -- something rigidly attached but OFF the axis has to be visible.
robotiq's driver carries a rigid child body for that (the 4-bar coupler, 31.8 mm off-axis); rum's
fingers are LEAVES, and their only rigid off-axis feature is a geom, ``left_fingertip``, 97.1 mm off
the hinge and measured rigid to 0.00000 mm under 100 N.

NODE ORDER is breadth-first from the arm root, left branch before right.

CENTRES sit at the JOINT, which is the body origin only when ``jnt_pos`` is zero -- true for every arm
link, for yam_jaw and for robotiq, false for rum, whose hinges are 48.7 mm out.
"""

from __future__ import annotations

import functools
import re

import mujoco
import numpy as np

from molmo_spaces.embodiment_gen.constants import GRIPPER_PREFIX, TCP_SITE

BODY, GEOM, SITE = "body", "geom", "site"
JOINT_TERMINAL, JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_TCP = 0, 1, 2, 3
TCP_NODE = {"kind": SITE, "name": GRIPPER_PREFIX + TCP_SITE, "centre_offset": [0.0, 0.0, 0.0], "axis": [0.0, 0.0, -1.0]}


# An observable closer than this to the joint axis has no usable azimuth. Well below the real cases
# (robotiq 31.8 mm, yam_jaw 39.2 mm, rum 97.1 mm) and well above numerical noise.
OFF_AXIS_MINIMUM = 0.005


def derive(model, dof: int, gripper_joints: tuple[str, ...], lr_swap: bool = False, *, gripper: str) -> dict:
    """The node set for a compiled generated robot; ``gripper`` names its catalog entry (the calibration is measured
    on that entry's bare gripper, see ``_gripper_command_scale``)."""
    bodies = {i: _name(model, mujoco.mjtObj.mjOBJ_BODY, i) for i in range(model.nbody)}
    parent = {i: int(model.body_parentid[i]) for i in range(model.nbody)}

    nodes: list[dict] = []
    for index in range(1, dof + 1):
        name = f"link{index}"
        if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) < 0:
            raise ValueError(f"arm link {name} absent from the model")
        nodes.append({"kind": BODY, "name": name, "_role": "arm"})

    drivers, driver_qpos, prismatic = [], [], False
    for joint_name in gripper_joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if jid < 0:
            continue
        body = bodies[int(model.jnt_bodyid[jid])]
        if body in drivers:
            continue
        drivers.append(body)
        driver_qpos.append(int(model.jnt_qposadr[jid]))
        prismatic = prismatic or int(model.jnt_type[jid]) == mujoco.mjtJoint.mjJNT_SLIDE
    order = _left_then_right(drivers)
    if lr_swap:
        order = order[::-1]
    drivers = [drivers[i] for i in order]
    driver_qpos = [driver_qpos[i] for i in order]
    nodes += [{"kind": BODY, "name": name, "_role": "driver"} for name in drivers]

    found = [entry for driver in drivers
             if (entry := _observable(model, driver, parent, bodies)) is not None]
    term_order = _left_then_right([entry["name"] for entry in found])
    if lr_swap:
        term_order = term_order[::-1]
    nodes += [found[i] | {"_role": "terminal"} for i in term_order]

    _order_breadth_first(model, nodes, parent, lr_swap)

    for node in nodes:
        joints = ([] if node["_role"] == "terminal"
                  else _joints_of(model, _anchor_body(model, node)))
        node["centre_offset"] = ([round(float(v), 6) for v in model.jnt_pos[joints[0]]]
                                 if joints else [0.0, 0.0, 0.0])
        node["axis"] = ([round(float(v), 6) for v in _unit(model.jnt_axis[joints[0]])]
                        if joints else None)

    index_of = {(n["kind"], n["name"]): i for i, n in enumerate(nodes)}
    edges = []
    for child, node in enumerate(nodes):
        anchor = _anchor_body(model, node)
        ancestor = anchor if node["kind"] == GEOM else parent[anchor]
        while ancestor > 0 and (BODY, bodies[ancestor]) not in index_of:
            ancestor = parent[ancestor]
        key = (BODY, bodies.get(ancestor, ""))
        if ancestor > 0 and key in index_of and index_of[key] != child:
            edges.append([index_of[key], child])

    joint_type = [JOINT_TERMINAL if n["_role"] == "terminal"
                  else (JOINT_PRISMATIC if prismatic and n["_role"] == "driver"
                        else JOINT_REVOLUTE)
                  for n in nodes]
    for node in nodes:
        del node["_role"]

    # The calibration is a property of the gripper family: measured on the bare catalog gripper on a rigid mount,
    # not on this robot. The drivers are matched to it by joint name.
    alone = _standalone_gripper(gripper, int(model.opt.integrator), float(model.opt.timestep))
    driver_names = [_name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for address in driver_qpos
                    for j in range(model.njnt) if int(model.jnt_qposadr[j]) == address]
    alone_ids = [mujoco.mj_name2id(alone, mujoco.mjtObj.mjOBJ_JOINT, n) for n in driver_names]
    if len(alone_ids) != len(driver_qpos) or min(alone_ids) < 0:      # a -1 would silently index the last joint
        raise ValueError(f"driver joints {driver_names} not all found on the catalog gripper {gripper!r}")
    alone_qpos = [int(alone.jnt_qposadr[j]) for j in alone_ids]
    calibration = _gripper_command_scale(alone, alone_qpos, nodes, joint_type, dof)
    _orient_axes_toward_opening(nodes, joint_type, dof, calibration)
    # The TCP node (the gripper's canonical ``tcp_frame`` site, the catalog's second add_site -- GL-style: x toward
    # the right pad, y wrist-image-up, z back into the wrist; type JOINT_TCP, never actuated, its axis the
    # approach, -z) in its breadth-first place: a child of the last arm node like
    # the drivers, and with no side it sorts after them and before the terminals.
    tcp = dof + len(drivers)
    edges = [[a + (a >= tcp), b + (b >= tcp)] for a, b in edges]
    edges.insert(tcp - 1, [dof - 1, tcp])                    # edges are child-ordered
    nodes.insert(tcp, dict(TCP_NODE))
    joint_type.insert(tcp, JOINT_TCP)
    return {
        "nodes": nodes,
        "edges": edges,
        "joint_type": joint_type,
        "n_arm": dof,
        "driver_qpos": driver_qpos,
        "qpos_center": _qpos_center(model),
        **calibration,
    }


def _orient_axes_toward_opening(nodes, joint_type, n_arm: int, calibration: dict) -> None:
    """Flip each gripper driver's axis so it points the way that node moves as the jaw OPENS.

    ``jnt_axis`` alone does not say this. yam_jaw's two fingers have the SAME axis in the SAME body
    orientation -- measured world axes identical to 0.0 degrees -- and are mirrored purely by opposite
    qpos ranges, so both arrows pointed the same way while the fingers moved apart. robotiq and rum
    mirror through their body frames instead, so their axes already oppose and the sign comes out +1.

    Sign is ``qpos_open - qpos_closed`` per driver, from the calibration's measured endpoints, so it
    is a fact about the mechanism rather than about which end of ctrl the model happens to call zero
    -- and that matters, because yam_jaw's actuator commands CLOSING (its ctrl_min is open) while the
    other two command opening."""
    at_open, at_closed = calibration.get("gripper_qpos_at_open"), \
        calibration.get("gripper_qpos_at_closed")
    if not at_open or not at_closed:
        return
    drivers = [i for i in range(n_arm, len(nodes)) if joint_type[i] != JOINT_TERMINAL]
    for position, index in enumerate(drivers):
        if position >= len(at_open) or nodes[index]["axis"] is None:
            continue
        if at_open[position] - at_closed[position] < 0.0:
            nodes[index]["axis"] = [-v for v in nodes[index]["axis"]]


def node_frame(model, data, node: dict) -> tuple[np.ndarray, np.ndarray]:
    """A node's anchor pose in the world: ``(origin (3,), rotation (3, 3))`` of its body, geom or site."""
    if node["kind"] == GEOM:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, node["name"])
        return np.asarray(data.geom_xpos[gid], float), np.asarray(data.geom_xmat[gid], float).reshape(3, 3)
    if node["kind"] == SITE:
        sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, node["name"])
        assert sid >= 0, f"site node {node['name']!r} absent from the model (Kinematics injects tcp_frame)"
        return np.asarray(data.site_xpos[sid], float), np.asarray(data.site_xmat[sid], float).reshape(3, 3)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node["name"])
    return np.asarray(data.xpos[bid], float), np.asarray(data.xmat[bid], float).reshape(3, 3)


def node_world(model, data, node: dict) -> np.ndarray:
    """A node's world position: its anchor's pose, shifted by ``centre_offset`` in that frame."""
    if node["kind"] == GEOM:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, node["name"])
        return np.asarray(data.geom_xpos[gid], float)
    if node["kind"] == SITE:
        return node_frame(model, data, node)[0]
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node["name"])
    origin = np.asarray(data.xpos[bid], float)
    return origin + np.asarray(data.xmat[bid], float).reshape(3, 3) @ \
        np.asarray(node["centre_offset"], float)


def node_axis_world(model, data, node: dict) -> np.ndarray | None:
    """A node's joint axis in world coordinates (the approach axis for the TCP node), or None for a terminal."""
    if node["axis"] is None:
        return None
    if node["kind"] == SITE:
        return node_frame(model, data, node)[1] @ np.asarray(node["axis"], float)
    bid = _anchor_body(model, node)
    return np.asarray(data.xmat[bid], float).reshape(3, 3) @ np.asarray(node["axis"], float)


def _name(model, objtype, index) -> str:
    return mujoco.mj_id2name(model, objtype, index) or ""


def _joints_of(model, body: int) -> list[int]:
    start, count = int(model.body_jntadr[body]), int(model.body_jntnum[body])
    return [] if start < 0 else list(range(start, start + count))


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, float)
    norm = float(np.linalg.norm(vector))
    return vector / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0])


def _anchor_body(model, node: dict) -> int:
    """The body a node resolves against: itself for a BODY, its owner for a GEOM."""
    if node["kind"] == GEOM:
        return int(model.geom_bodyid[
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, node["name"])])
    if node["kind"] == SITE:
        return int(model.site_bodyid[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, node["name"])])
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, node["name"])


def _left_then_right(names: list[str]) -> list[int]:
    """Indices ordering a branch pair left before right, keeping input order when unlabelled."""
    def rank(name: str) -> tuple:
        lowered = name.lower()
        return (0 if "left" in lowered else (1 if "right" in lowered else 2), name)

    return sorted(range(len(names)), key=lambda i: rank(names[i]))


def _observable(model, driver: str, parent: dict, bodies: dict) -> dict | None:
    """A rigid anchor for ``driver``: a BODY in its rigid subtree if there is one, else a GEOM on it.

    What makes an anchor good depends on the JOINT TYPE, and conflating the two picks the wrong body:

    * REVOLUTE -- the centre sits on the axis and does not move, so only the AZIMUTH of something
      off-axis reports the angle. Criterion: furthest perpendicular from the axis.
    * PRISMATIC -- every rigid point translates by the same amount (measured: all of yam_jaw's
      candidates travel 37.4 mm regardless of position), so observability is automatic and off-axis
      distance is irrelevant. Criterion: the DEEPEST rigid body, which is the jaw itself rather than
      an intermediate mount.

    Applying the revolute rule to yam_jaw rejected ``rf_down`` for sitting 4.8 mm from the slide axis
    and fell back to the mount ``rf_rot``, while the left side kept ``lf_down`` at 83.2 mm -- an
    asymmetric pair of anchors on a symmetric gripper.

    Body first, so robotiq keeps the coupler. Geoms are considered only where the tree ends, and only
    NAMED ones -- an unnamed geom is an index that moves when an asset is re-exported. That is safe
    because the fallback is reached only by rum, whose geoms are named, while robotiq's are bare
    ``geom24``/``geom36`` and never reach it. A name containing "tip" wins over raw distance."""
    did = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, driver)
    joints = _joints_of(model, did)
    axis = _unit(model.jnt_axis[joints[0]]) if joints else np.array([0.0, 0.0, 1.0])
    origin = np.asarray(model.jnt_pos[joints[0]], float) if joints else np.zeros(3)
    sliding = bool(joints) and int(model.jnt_type[joints[0]]) == mujoco.mjtJoint.mjJNT_SLIDE

    def perpendicular(point) -> float:
        offset = np.asarray(point, float) - origin
        return float(np.linalg.norm(offset - (offset @ axis) * axis))

    # The whole RIGID subtree, not just direct children: a jointless descendant is just as rigid to
    # the driver as a jointless child, and is usually further out. yam_jaw's jaw is a grandchild --
    # link_left_finger -> lf_rot (2 geoms, a mount frame) -> lf_down (11 geoms, the actual jaw) -- so
    # a direct-children search stopped at the mount, 39.2 mm out, instead of the jaw at 44.0 mm.
    #
    # The walk stops at any joint, which is what keeps robotiq's follower fingertips out: they hang
    # off the 4-bar through their own joints, so they are NOT rigid to the driver.
    rigid, frontier = [], [(b, 1) for b in range(model.nbody) if parent[b] == did]
    while frontier:
        body, depth = frontier.pop()
        if _joints_of(model, body):
            continue                       # a joint here breaks rigidity: stop, do not descend
        rigid.append((depth, _offset_to(model, body, did, parent), bodies[body]))
        frontier += [(b, depth + 1) for b in range(model.nbody) if parent[b] == body]
    if rigid:
        if sliding:
            depth, _, name = max(rigid, key=lambda r: (r[0], r[2]))
            return {"kind": BODY, "name": name}
        distance, name = max((perpendicular(offset), name) for _, offset, name in rigid)
        if distance > OFF_AXIS_MINIMUM:
            return {"kind": BODY, "name": name}

    geoms = []
    for geom in range(model.ngeom):
        if int(model.geom_bodyid[geom]) != did:
            continue
        name = _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)
        if name:
            distance = perpendicular(model.geom_pos[geom])
            geoms.append((("tip" not in name.lower()), -distance, distance, name))
    if geoms:
        _, _, distance, name = min(geoms)
        if distance > OFF_AXIS_MINIMUM:
            return {"kind": GEOM, "name": name}
    return None


def _offset_to(model, body: int, ancestor: int, parent: dict) -> np.ndarray:
    """``body``'s origin expressed in ``ancestor``'s frame, composing the fixed links between them."""
    offset, rotation, walk = np.zeros(3), np.eye(3), body
    while walk != ancestor and walk > 0:
        local = np.asarray(model.body_pos[walk], float)
        frame = _quat_to_mat(model.body_quat[walk])
        offset = frame @ offset + local
        rotation = frame @ rotation
        walk = parent[walk]
    return offset


def _quat_to_mat(quat) -> np.ndarray:
    w, x, y, z = (float(v) for v in quat)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _order_breadth_first(model, nodes: list[dict], parent: dict, lr_swap: bool = False) -> None:
    """Sort in place: breadth-first from the arm root, left branch before right -- or right before
    left when lr_swap, so finger nodes follow the wrist view's VISUAL left->right for grippers whose
    wrist camera mirrors L/R (keeps index-position consistent across grippers)."""
    def depth(node: dict) -> int:
        walk, levels = _anchor_body(model, node), 1 if node["kind"] == GEOM else 0
        while walk > 0:
            walk = parent[walk]
            levels += 1
        return levels

    def side(name: str) -> int:
        low = name.lower(); left, right = "left" in low, "right" in low
        return (0 if right else 1 if left else 2) if lr_swap else (0 if left else 1 if right else 2)

    nodes.sort(key=lambda n: (depth(n), side(n["name"]), n["name"]))


def _qpos_center(model) -> list[float]:
    """Mid-range ``(lo + hi) / 2`` per qpos entry, so proprio zero means the same on every robot.

    Unlimited joints centre at 0, a mid-range being undefined without limits."""
    center = np.zeros(model.nq)
    for jid in range(model.njnt):
        adr = int(model.jnt_qposadr[jid])
        if 0 <= adr < model.nq and bool(model.jnt_limited[jid]):
            center[adr] = 0.5 * float(model.jnt_range[jid][0] + model.jnt_range[jid][1])
    return [round(float(v), 6) for v in center]


def _gripper_command_scale(model, driver_qpos, nodes=None, joint_type=None,
                           n_arm: int = 0) -> dict:
    """The gripper command in each driver's qpos units, MEASURED at both ctrl extremes.

    The relationship is not recoverable from the declared ranges. Measured: yam_jaw's RIGHT finger
    traverses its declared range backwards; rum reaches -0.786 of a declared [-1.57, 0]; robotiq 0.824
    of [0, 0.9]. An affine map between ctrl_range and jnt_range is therefore wrong in direction for a
    mirrored finger and wrong in scale for all three -- by 0.78 rad on rum.

    The actuator is found by NAME (the one that is not an arm joint's) because two of the three
    grippers drive a TENDON, so matching a transmission id against a joint id finds nothing.

    Measured WITHOUT GRAVITY on the BARE GRIPPER on a RIGID MOUNT (``_standalone_gripper``; ``derive`` passes that
    model, not the robot's), so the value is the servo's own equilibrium -- the COMMANDED finger position, a
    property of the gripper family rather than of the arm it is mounted on. Measured on the assembled robot,
    three things of the arm entered it: at the zero configuration some sampled arms fold back onto their own
    gripper and wedge the fingers against the wrist links, and the "calibration" then recorded a gripper that
    never moves (a 0.0001 rad span on a robotiq build whose healthy siblings span 0.82 rad); under gravity the
    finger's own weight, along whichever way the sagging arm leaves the gripper pointing, moved the rest
    position by up to 0.4 mm between otherwise identical grippers; and the constraint solver regularises the
    pad-on-pad contact with the bodies' compile-time inverse weights, which come from the whole robot's mass
    matrix, so the same closed contact landed 1.5e-4 rad apart between a light and a heavy arm. The gripper's
    own contacts stay on: its closed extreme is set by pad-on-pad contact."""
    model = _gripper_alone(model)
    actuator = next((a for a in range(model.nu)
                     if not re.fullmatch(r"joint\d+",
                                         _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a))), None)
    if actuator is None or not bool(model.actuator_ctrllimited[actuator]) or not driver_qpos:
        return {"gripper_ctrl_range": None, "gripper_qpos_at_ctrl": None}
    ctrl = [float(v) for v in model.actuator_ctrlrange[actuator]]
    if ctrl[0] == ctrl[1]:
        return {"gripper_ctrl_range": None, "gripper_qpos_at_ctrl": None}

    def settle(command: float) -> list[float]:
        data = mujoco.MjData(model)
        data.ctrl[actuator] = command
        for _ in range(4000):
            mujoco.mj_step(model, data)
        return [round(float(data.qpos[i]), 6) for i in driver_qpos]

    low, high = settle(ctrl[0]), settle(ctrl[1])
    result = {"gripper_ctrl_range": ctrl, "gripper_qpos_at_ctrl": [low, high]}
    if nodes is None or joint_type is None:
        return result

    # Which end is OPEN, measured by how far apart the TERMINALS are -- they are the parts that move.
    # Driver separation cannot answer it for a revolute gripper: those centres sit on their hinges, so
    # rum and robotiq measure an identical 60.3 mm and 61.9 mm at both extremes.
    terminals = [n for n, t in zip(nodes, joint_type) if t == JOINT_TERMINAL]
    probes = terminals if len(terminals) >= 2 else \
        [n for n, t in zip(nodes[n_arm:], joint_type[n_arm:]) if t != JOINT_TERMINAL]
    if len(probes) < 2:
        return result

    def separation(command: float) -> float:
        data = mujoco.MjData(model)
        data.ctrl[_gripper_actuator(model)] = command
        for _ in range(4000):
            mujoco.mj_step(model, data)
        points = [node_world(model, data, node) for node in probes[:2]]
        return float(np.linalg.norm(points[0] - points[1]))

    open_is_low = separation(ctrl[0]) > separation(ctrl[1])
    result["gripper_open_ctrl"] = ctrl[0] if open_is_low else ctrl[1]
    result["gripper_qpos_at_open"] = low if open_is_low else high
    result["gripper_qpos_at_closed"] = high if open_is_low else low
    return result


def _gripper_alone(model):
    """A copy of ``model`` in which only the gripper's geoms collide (contype/conaffinity 0 elsewhere) and gravity
    is zero -- what the settle needs when handed a model that is more than the bare gripper (the tests' stubs)."""
    import copy
    alone = copy.deepcopy(model)                       # mj_copyModel: an independent model either way
    alone.opt.gravity[:] = 0.0
    for geom in range(alone.ngeom):
        if not _name(alone, mujoco.mjtObj.mjOBJ_BODY, int(alone.geom_bodyid[geom])).startswith(GRIPPER_PREFIX):
            alone.geom_contype[geom] = 0
            alone.geom_conaffinity[geom] = 0
    return alone


@functools.lru_cache(maxsize=8)
def _standalone_gripper(name: str, integrator: int, timestep: float):
    """The bare catalog gripper ``name`` on a rigid mount, compiled once per process: a massless flange on a hinge
    that is stiff, damped and heavily armatured (rigid in the mass matrix, so the solver's inverse weights -- and
    with them the softness of the pad contact -- are the gripper's own), yet still a body of its own so MuJoCo's
    parent-child contact filtering keeps the fingers off the gripper base as it does on the assembled robot. The
    integrator and timestep are the robot's, gravity is zero."""
    from mujoco import MjSpec
    from molmo_spaces.embodiment_gen.grippers import graft

    spec = MjSpec()
    spec.option.integrator = integrator
    spec.option.timestep = timestep
    flange = spec.worldbody.add_body(name="flange", pos=[0.0, 0.0, 0.3])
    flange.add_joint(name="flange_stub", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0.0, 0.0, 1.0],
                     stiffness=1e5, damping=1e2, armature=1e6)
    graft(spec, flange, name)
    model = spec.compile()
    model.opt.gravity[:] = 0.0
    return model


def _gripper_actuator(model) -> int:
    return next(a for a in range(model.nu)
                if not re.fullmatch(r"joint\d+", _name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a)))
