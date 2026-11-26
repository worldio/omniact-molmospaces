"""PARAMETERS -> MODEL. The deterministic half of generation.

Takes an :class:`~.params.EmbodimentParams` and emits one MJCF containing the arm and its
grafted gripper. Contains no randomness and never sees a seed, so a hand-written parameter object
builds exactly like a sampled one -- which is what lets :mod:`.reference` exhibit a real robot's
parameters and have them build.

Actuators are emitted directly in the model, matching every shipped MolmoSpaces robot (their
specs only *override* actuators, never create them), using MuJoCo's affine convention
``(gain, bias_position, bias_velocity) = (kp, -kp, -kv)``.

The gripper is grafted with ``MjsFrame.attach_body``, which carries its actuator, coupling
constraints, tendon, sites and meshes across specs -- see :mod:`.grippers` for the four things
that bite.
"""

from __future__ import annotations

from dataclasses import replace

import mujoco
import numpy as np
from mujoco import MjSpec

from . import geom_pool
from .constants import (
    BASE_BODY,
    FLANGE_BODY,
    GRIPPER_PREFIX,
    PLANNER_SITE,
    WRIST_CAMERA,
    WRIST_CAMERA_MOUNT,
)
from .grippers import graft, load_catalog
from .params import SUPPORT, EmbodimentParams
from .skeleton import mat_to_quat, quat_to_mat

_GEOM_TYPE = {
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "mesh": mujoco.mjtGeom.mjGEOM_MESH,
}
_JOINT_TYPE = {"hinge": mujoco.mjtJoint.mjJNT_HINGE, "slide": mujoco.mjtJoint.mjJNT_SLIDE}


class BuildError(RuntimeError):
    """The parameters cannot be realized (compile failure). A rejection, not a crash."""


def link_name(index: int) -> str:
    return f"link{index}"


def joint_name(index: int) -> str:
    return f"joint{index}"


def _declare_pool_meshes(spec: MjSpec, params: EmbodimentParams) -> dict:
    """Re-declare, once per model, every mesh any pool link references.

    Mesh files resolve against their SOURCE model's directory, so a model mixing links from two
    source robots needs two mesh directories. MuJoCo has one ``meshdir``, hence absolute paths
    here: the cost is that every rank and consumer must see MLSPACES_ASSETS_DIR at the same path,
    which is already true of the gripper graft."""
    declared = {}
    # Keyed by STRETCH as well as by mesh: a resized link needs its own scaled asset, since MuJoCo
    # carries scale on the <mesh> rather than the geom. Two links at different sizes therefore
    # declare two assets from the same file.
    wanted = {(link.geom.source, geom.mesh, round(link.geom.stretch, 6))
              for link, entry in _pool_entries(params)
              for geom in entry.geoms if geom.mesh}
    for source, mesh, stretch in sorted(wanted):
        path = geom_pool.mesh_files(source).get(mesh)
        if path is None:
            raise BuildError(f"{source}: mesh {mesh!r} has no file")
        key = f"{source}__{mesh}__s{stretch:.6f}"
        if key in declared.values():
            continue
        spec.add_mesh(name=key, file=path, scale=[stretch] * 3)
        declared[(source, mesh, stretch)] = key
    return declared


def _declare_pool_appearance(spec: MjSpec, params: EmbodimentParams) -> dict:
    """Re-declare every material (and any texture it uses) that an inheriting pool link needs.

    Appearance is part of the embodiment: without this a pool link renders in MuJoCo's default
    grey, so the real robots would be in distribution geometrically and not visually. Names are
    source-prefixed because two source models may both call a material "white" and mean different
    greys (FR3's is 1.0, YAM's is 0.9)."""
    declared, texture_names = {}, {}
    wanted = {(link.geom.source, geom.material)
              for link, entry in _pool_entries(params)
              if link.geom.material_mode == "inherit"
              for geom in entry.geoms if geom.material}
    for source, name in sorted(wanted):
        material = geom_pool.materials(source).get(name)
        if material is None:
            raise BuildError(f"{source}: material {name!r} is not declared")
        used = []
        for texture in material.textures:
            if not texture:
                used.append("")
                continue
            key = texture_names.get((source, texture))
            if key is None:
                found = geom_pool.textures(source).get(texture)
                if found is None:
                    raise BuildError(f"{source}: texture {texture!r} is not declared")
                key = f"{source}__{texture}"
                kwargs = dict(name=key, type=found.type, builtin=found.builtin,
                              width=found.width, height=found.height,
                              rgb1=list(found.rgb1), rgb2=list(found.rgb2),
                              markrgb=list(found.markrgb))
                if found.file:
                    kwargs["file"] = found.file
                spec.add_texture(**kwargs)
                texture_names[(source, texture)] = key
            used.append(key)
        key = f"{source}__{name}"
        spec.add_material(name=key, rgba=list(material.rgba), specular=material.specular,
                          shininess=material.shininess, reflectance=material.reflectance,
                          emission=material.emission, texrepeat=list(material.texrepeat),
                          textures=used or None)
        declared[(source, name)] = key
    return declared


def _declare_palette(spec: MjSpec, params: EmbodimentParams) -> dict:
    """One material per palette role, plus the optional procedural texture they share."""
    dressing = params.dressing
    texture = None
    if dressing.texture_kind != "none":
        builtin = {"checker": 2, "gradient": 1, "flat": 3}[dressing.texture_kind]
        texture = "palette_texture"
        spec.add_texture(name=texture, type=int(mujoco.mjtTexture.mjTEXTURE_2D), builtin=builtin,
                         width=200, height=200, rgb1=list(dressing.texture_rgb1),
                         rgb2=list(dressing.texture_rgb2))
    out = {}
    for role, rgba in sorted(dressing.palette.items()):
        name = f"palette_{role}"
        kwargs = dict(name=name, rgba=list(rgba), specular=dressing.specular,
                      shininess=dressing.shininess, reflectance=dressing.reflectance,
                      emission=dressing.emission)
        if texture:
            kwargs["textures"] = [texture] + [""] * 9
            kwargs["texrepeat"] = [dressing.texture_repeat, dressing.texture_repeat]
        spec.add_material(**kwargs)
        out[role] = name
    return out


def _pool_entries(params):
    """``(LinkParams, LinkGeoms)`` for every link wearing real geometry, the BASE included --
    otherwise its meshes and materials are never declared and the build fails or renders bare."""
    links = list(params.arm.links)
    if params.arm.base is not None:
        links.append(params.arm.base)
    for link in links:
        if link.geom.mode == "pool":
            yield link, geom_pool.link_geoms(link.geom.source, link.geom.link_index)


def _hub(body, name: str, radius: float, material: str, pos=(0.0, 0.0, 0.0)) -> None:
    """A sphere at a body's own origin -- the child half of a strut pair.

    The child half CANNOT be a capsule spanning back to its parent. That span is fixed in the
    child's frame, so it rotates with the joint and splays away from the parent's bone as the joint
    moves: two capsules over one segment, opening into a V. That is exactly the "extra limb" artefact
    the invariant forbids, and a full-length backward strut is how it came back (seed 18).

    A sphere at the origin has no span to splay, cannot duplicate its parent's bone, and sits exactly
    where the parent's strut ends -- so the two overlap at every joint angle, which is the contact
    guarantee, at a fraction of the volume."""
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[radius, 0.0, 0.0],
                  pos=list(pos), material=material, group=0)


def _capsule(body, name: str, span, radius: float, material: str) -> None:
    """A capsule from the body's origin to ``span``, in the body's own frame."""
    length = float(np.linalg.norm(span))
    if length <= 1e-3:
        return
    direction = np.asarray(span, float) / length
    if direction[2] < -0.999999:                       # wxyz rotating local +z onto the span
        quat = [0.0, 1.0, 0.0, 0.0]
    else:
        q = np.array([1.0 + direction[2], -direction[1], direction[0], 0.0])
        quat = list(q / np.linalg.norm(q))
    body.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                  size=[radius, max(0.004, length / 2), 0.0],
                  pos=list(np.asarray(span, float) / 2), quat=quat, material=material, group=0)


def _add_primitive_geoms(body, index: int, params, bone, palette: dict, back=None) -> None:
    """A capsule along the bone plus a joint housing at the origin.

    Deliberately NO backward capsule. One was added early on so a primitive link could meet a
    borrowed neighbour, but it is redundant against a primitive parent -- which already draws a bone
    to this link's origin -- and it rotates with the CHILD. At qpos=0 the two coincide; turn the
    joint and they diverge into a V sprouting from every joint, which is exactly the "extra limb".
    Reaching a borrowed neighbour is now the measured repair pass's job (see repaired_spec), which
    adds a strut only where a gap is actually measured, so nothing redundant is emitted."""
    del back
    dressing = params.dressing
    dof = params.arm.dof
    fraction = index / max(dof - 1, 1)
    radius = dressing.radius * (1.0 - fraction * (1.0 - dressing.taper))
    role = "accent" if (index + 1) % max(dressing.accent_period, 1) == 0 else "base"
    material = palette.get(role, palette["base"])

    # Drawn whenever there IS a span: the old `length > 2.5 * radius` guard skipped every bone
    # under ~0.10 m, leaving a gap between consecutive joint hubs.
    _capsule(body, f"{link_name(index + 1)}_bone", bone, radius, material)
    body.add_geom(name=f"{link_name(index + 1)}_hub", type=mujoco.mjtGeom.mjGEOM_SPHERE,
                  size=[radius * 1.05, 0.0, 0.0], pos=[0.0, 0.0, 0.0],
                  material=palette["joint"], group=0)


def _add_pool_geoms(body, index: int, entry, meshes: dict, appearance: dict,
                    material_mode: str, palette: dict, stretch: float = 1.0) -> None:
    """Wear a real link's complete geom set: visual AND collision, poses and contact flags kept.

    The set is applied whole. FR3's link5 is three visual meshes plus one collision hull whose
    relative placement *is* the link, so mixing pieces across links would produce geometry no
    robot has."""
    for number, geom in enumerate(entry.geoms):
        # Resize the whole link coherently: mesh assets carry `scale`, primitives scale their size,
        # and every placement scales with them. Isotropic, so it commutes with the geom's rotation.
        kwargs = dict(name=f"{link_name(index + 1)}_pool{number}",
                      type=_GEOM_TYPE[geom.type],
                      pos=[v * stretch for v in geom.pos], quat=list(geom.quat),
                      group=geom.group, contype=geom.contype, conaffinity=geom.conaffinity)
        if material_mode == "inherit":
            if geom.material:
                kwargs["material"] = appearance[(entry.source, geom.material)]
            else:
                kwargs["rgba"] = list(geom.rgba)          # the source geom's own colour
        else:
            kwargs["material"] = palette["base"]          # visual DR over a real morphology
        if geom.mesh:
            kwargs["meshname"] = meshes[(entry.source, geom.mesh, round(stretch, 6))]
            kwargs["size"] = [0.0, 0.0, 0.0]
        else:
            kwargs["size"] = [v * stretch for v in geom.size]
        body.add_geom(**kwargs)


def _apply_inertial(body, inertial, stretch: float = 1.0) -> None:
    """Emit an explicit ``<inertial>``, as both shipped robots do. ``explicitinertial`` tells the
    compiler to trust it rather than recomputing from geoms -- the only route to a hand-specified
    inertia, which need not equal anything a density times a geom would produce."""
    if inertial is None:
        return
    if inertial.mode != "explicit":
        body.gravcomp = float(inertial.gravcomp)      # independent of where mass comes from
        return
    # Resizing a link resizes its mass properties too, at constant density: mass ~ s^3, and
    # inertia ~ m r^2 ~ s^5. At stretch 1 these are exactly the measured values.
    body.gravcomp = float(inertial.gravcomp)
    body.mass = inertial.mass * stretch ** 3
    body.inertia = [v * stretch ** 5 for v in inertial.inertia]
    body.ipos = [v * stretch for v in inertial.com]
    body.explicitinertial = True


def _dress(body, index: int, link, params, meshes, appearance, palette, bone, back=None) -> None:
    """Give a body its geometry, from the pool or from sampled primitives."""
    if link.geom.mode == "pool":
        entry = geom_pool.link_geoms(link.geom.source, link.geom.link_index)
        _add_pool_geoms(body, index, entry, meshes, appearance, link.geom.material_mode, palette,
                        stretch=link.geom.stretch)
    else:
        _add_primitive_geoms(body, index, params, bone or (0.0, 0.0, 0.0), palette, back=back)


def _grasp_frame(asset, model) -> tuple | None:
    """``(host body, position, rotation)`` of the gripper's grasp frame, in the host body's frame.

    Prefers the catalog's ``add_site`` declaration of the planner frame -- that is the authority on
    where the planner will aim, and it exists before any spec operation has been applied. Falls back
    to a compiled site of the same name, then to the shipped ``grasp_site``, so a catalog entry that
    declares its frame in the MJCF instead still resolves.

    ``xyaxes`` is honoured, not collapsed: a grasp frame's roll IS its closing axis, and reducing the
    two authored axes to a single look direction discards it."""
    for operation in asset.model_operations:
        if operation.get("kind") != "add_site":
            continue
        if operation["name"]["name"] != f"{GRIPPER_PREFIX}{PLANNER_SITE}":
            continue
        axes, quaternion = operation.get("xyaxes"), operation.get("quaternion_wxyz")
        if axes is not None:
            x_axis = np.asarray(axes[:3], float)
            y_axis = np.asarray(axes[3:], float)
            x_axis /= np.linalg.norm(x_axis)
            y_axis -= x_axis * float(x_axis @ y_axis)
            y_axis /= np.linalg.norm(y_axis)
            rotation = np.column_stack([x_axis, y_axis, np.cross(x_axis, y_axis)])
        else:
            rotation = quat_to_mat(tuple(quaternion or (1.0, 0.0, 0.0, 0.0)))
        return (operation["parent_body"]["name"],
                np.asarray(operation["position"], float), rotation)

    for name in (f"{GRIPPER_PREFIX}{PLANNER_SITE}", f"{GRIPPER_PREFIX}grasp_site"):
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site < 0:
            continue
        host = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site]))
        return (host, np.asarray(model.site_pos[site], float),
                quat_to_mat(tuple(np.asarray(model.site_quat[site], float))))
    return None


def _add_wrist_camera(spec: MjSpec, params: EmbodimentParams, model) -> None:
    """Declare ONE canonically named ``wrist_camera``, posed in the gripper's grasp frame.

    The grippers all ship a designed wrist view -- and disagree on what to call it: robotiq's is
    ``wrist_camera``, rum's is ``wrist_cam``, and yam_jaw has none. A camera system referencing an
    MJCF camera resolves it BY NAME, so without one uniform name no generic system can name a wrist
    camera at all, which is why the generic camera system was exocentric-only before this.

    Upstream cannot supply the name. yam_jaw's MJCF *is* i2rt_yam's arm, so adding a camera there
    would change the shipped robot's ncam and break witness parity; and RUM's ``wrist_cam`` name is
    referenced by scripts/datagen/robot_conversion_patches.py. So the uniform name is declared here,
    from the pose the catalog measured -- the same move already made for planner_grasp_site.

    Posed in the GRASP FRAME rather than on the flange, so it follows the jaws for any gripper and
    the derivation transfers to a gripper that ships no camera of its own. The grasp frame is read
    from the catalog's own ``add_site`` declaration rather than from a compiled site, because the
    site this aims at must be the one the PLANNER aims at. Reading the compiled model instead put
    yam_jaw's camera on ``gripper/grasp_site`` -- 49.7 mm from ``planner_grasp_site``, which at a
    185 mm standoff is 15.5 degrees of aim error -- since the planner site is created by a spec
    operation applied after this runs. That 49.7 mm is the same gap that once cost a grasp outright
    (see ``constants.PLANNER_SITE``); it is not a coincidence that it reappeared here.
    """
    asset = load_catalog()[params.gripper]
    entry = asset.wrist_camera
    if not entry:
        return
    grasp = _grasp_frame(asset, model)
    if grasp is None:
        return
    host_name, site_pos, rotation = grasp
    host = spec.body(host_name)
    if host is None:
        return
    offset = np.asarray(entry["pos"], float)
    position = site_pos + rotation @ offset
    if entry.get("quat") is not None:
        # The DESIGNED orientation, measured (or vendor-published) in the grasp frame. Using it
        # verbatim makes the wrist stream match the shipped robots' exactly -- the aim-at-grasp
        # derivation below survives only for a gripper with no designed view anywhere, and its
        # known cost is a view 15.5 deg into the jaws.
        frame = rotation @ quat_to_mat(tuple(entry["quat"]))
        mount = host.add_body(name=WRIST_CAMERA_MOUNT, pos=[float(v) for v in position],
                              quat=list(mat_to_quat(frame)))
        mount.add_camera(name=WRIST_CAMERA, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0],
                         fovy=float(entry["fovy"]), resolution=[int(v) for v in entry.get("resolution", (640, 480))])
        return
    # aim it back at the grasp point: MuJoCo cameras look along their own -z
    forward = -(rotation @ offset)
    distance = float(np.linalg.norm(forward))
    if distance < 1.0e-6:
        return
    forward /= distance
    up = np.array([0.0, 0.0, 1.0])
    if abs(float(forward @ up)) > 0.95:
        up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, -forward)
    right /= max(float(np.linalg.norm(right)), 1.0e-12)
    true_up = np.cross(-forward, right)
    frame = np.column_stack([right, true_up, -forward])       # camera looks along -z
    mount = host.add_body(name=WRIST_CAMERA_MOUNT, pos=[float(v) for v in position],
                          quat=list(mat_to_quat(frame)))
    mount.add_camera(name=WRIST_CAMERA, pos=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0],
                     fovy=float(entry["fovy"]), resolution=[int(v) for v in entry.get("resolution", (640, 480))])


def _add_cameras(spec: MjSpec, params: EmbodimentParams, base_body, flange_body) -> None:
    """Declare the robot's own cameras, each on a dedicated zero-geom mount body.

    That is exactly how FR3 carries ``wrist_cam`` on ``wrist_cam_body``, and it has to be in the
    model rather than in the experiment config: MolmoSpaces wrist cameras are ``MjcfCameraConfig``,
    which resolves a camera BY NAME out of the compiled model."""
    hosts = {"flange": flange_body, "base": base_body}
    for camera in params.arm.cameras:
        host = hosts.get(camera.mount)
        if host is None:
            raise BuildError(f"unknown camera mount {camera.mount!r}")
        mount = host.add_body(name=f"{camera.name}_mount", pos=list(camera.body_pos),
                              quat=list(camera.body_quat))
        mount.add_camera(name=camera.name, pos=list(camera.pos), quat=list(camera.quat),
                         fovy=camera.fovy, resolution=list(camera.resolution))


# Gaps up to this are normal CLEARANCE, not a disconnection. Measured on the shipped robots by true
# surface distance: franka_droid's consecutive links clear each other by at most 0.99 mm and
# i2rt_yam's by 0.00 mm, so 2 mm leaves both untouched while a genuine break (centimetres) gets
# repaired. A stricter 0.1 mm threshold added struts to the real robots themselves.
CLEARANCE_TOLERANCE = 0.002

# The share of an arm's joint space that may put it inside itself. The shipped robots measure ~0.05;
# this is set far looser so a generated arm is allowed to be much more self-obstructing than a
# designed one, and only pathological cases are repaired. Real arms keep this low with JOINT LIMITS,
# which is exactly the knob the repair turns -- narrowing a sampled range, never inventing one.
# The share of its joint space a robot may spend inside itself before it is REJECTED. 0.20 is
# chosen against the witnesses, not for convenience: FR3 measures 0.016-0.039 and YAM 0.078-0.114
# depending on how many configurations are sampled, so 0.10 would fail YAM roughly half the time
# and cost the in-distribution guarantee. 0.20 clears YAM's worst reading with room.
#
# Read this number ONLY together with a joint span. The rate is measured uniformly over the box of
# joint ranges, so it is steeply span-sensitive: scaling FR3's own limits by 1.25 takes it from
# 0.031 to 0.227, and by 0.5 to 0.000. A robot with tiny limits scores perfectly while being
# useless, which is why acceptance is reported alongside the span distribution it selects.
SELF_COLLISION_LIMIT = 0.20

# The share of reachable table-height targets at which a whole pick -- pregrasp, grasp and lift at
# one downward orientation -- must be solvable. Measured on the shipped robots: FR3 0.50, YAM 0.33.
# Every embodiment that failed a planner run scored 0.00-0.04, so the two populations do not overlap.
# This is ORTHOGONAL to self-collision and catches what that filter cannot: seed70 has the lowest
# collision rate in the set (0.00) and is still unusable, because it cannot point its gripper down
# where it can reach. Set below YAM rather than near FR3, so the weaker shipped robot passes.
GRASP_FEASIBILITY_LIMIT = 0.30

SELF_COLLISION_BUDGET = 0.35
RANGE_SHRINK = 0.85
# How far the skeleton may be stretched to clear self-collision, and in what steps.
SPACING_STEP = 1.18
SPACING_LIMIT = 3.0


def consecutive_breaks(model, dof, gripper_root="", fractions=(0.0, 0.3, 0.5, 0.7, 1.0),
                       distmax=0.6, tolerance=None):
    """``{pair index: worst gap}`` between consecutive geometry-bearing bodies, over several poses.

    Pair ``i`` is (base_link, link1) for i = 0, (link_i, link_{i+1}) after that, and ``dof`` is
    (link_dof, gripper root).

    Distance is TRUE SURFACE distance from ``mj_geomDistance``, not a bounding-sphere estimate.
    Using ``geom_rbound`` here was badly wrong: the bounding sphere of a 0.13 m half-length capsule
    has radius ~0.14 m, so two slim capsules whose surfaces were 20 cm apart registered as touching.
    Every gap this function used to report was thereby suppressed -- it claimed zero breaks while
    42% of arms actually fragmented into separate pieces.

    A body with NO geometry is TRANSPARENT rather than a break in the walk: yam_jaw claims the flange
    link, so that link is deliberately bare, and stopping there hid the arm-to-gripper junction. A
    gap across a transparent body is charged to every pair it spans so the repair reconnects through
    it.

    Swept over configurations because rotating a joint moves a child's geometry relative to its
    parent: a chain can be flush at qpos=0 and gapped elsewhere."""
    chain = [BASE_BODY] + [link_name(i) for i in range(1, dof + 1)]
    if gripper_root:
        chain.append(f"{GRIPPER_PREFIX}{gripper_root}")
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in chain]
    geoms = {b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for b in ids}
    if gripper_root and ids[-1] >= 0 and not geoms[ids[-1]]:
        # The graft root itself may be a bare adapter (linear_4310's mount-convention flip).
        # Transparency must extend through it: measure the junction against the whole gripper
        # subtree's geometry, or the arm-to-gripper pair silently vanishes from the walk and
        # no strut ever bridges it.
        def under(body):
            walk = body
            while walk > 0:
                if walk == ids[-1]:
                    return True
                walk = int(model.body_parentid[walk])
            return False
        geoms[ids[-1]] = [g for g in range(model.ngeom) if under(int(model.geom_bodyid[g]))]
    filled = [k for k, b in enumerate(ids) if b >= 0 and geoms[b]]

    worst = {}
    for fraction in fractions:
        data = mujoco.MjData(model)
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                low, high = model.jnt_range[j]
                data.qpos[model.jnt_qposadr[j]] = low + fraction * (high - low)
        mujoco.mj_forward(model, data)
        for left, right in zip(filled, filled[1:]):
            a, b = ids[left], ids[right]
            # The caller only asks `gap > tolerance`, so once the running minimum has dropped to
            # tolerance the answer is settled and the remaining geom pairs cannot change it: the
            # true minimum is <= what we already have. Bailing there is worth roughly a third of
            # screening time (py-spy: this comprehension was 29% of a profiled rank), because most
            # consecutive links are comfortably joined and their first few geom pairs say so.
            # Without a tolerance the exact minimum is still computed, so other callers are unaffected.
            gap = distmax
            for x in geoms[a]:
                for y in geoms[b]:
                    distance = mujoco.mj_geomDistance(model, data, x, y, distmax, None)
                    if distance < gap:
                        gap = distance
                        if tolerance is not None and gap <= tolerance:
                            break
                else:
                    continue
                break
            for index in range(left, right):     # charge every pair the gap spans
                worst[index] = max(worst.get(index, 0.0), max(0.0, gap))
    return worst


def permanent_overlaps(model, fractions=(0.0, 0.25, 0.5, 0.75, 1.0), distmax=0.6) -> tuple:
    """Body pairs that INTERPENETRATE at every configuration, i.e. overlap by construction.

    MuJoCo filters contacts between a body and its parent -- but the check compares *weld* bodies,
    and ``base_link`` carries no joint, so it is welded to the world and the base/link1 pair is not
    filtered. FR3 has the same structure (``fr3_link0`` welds to world); it gets away with it
    because its meshes are shaped with clearance. Generated links are not: a primitive link puts its
    hub AT the joint while the parent's bone REACHES that joint, so the two necessarily overlap, and
    the repair pass deliberately drives consecutive geometry into contact. Two interpenetrating
    bodies that rotate relative to each other generate enormous contact forces -- seed0's base and
    link1 overlapped 66.6 mm and joint 1 spent the whole rollout fighting it.

    Real MJCFs solve exactly this with ``<exclude>``: all six of FR3's exclusions are gripper pairs
    that overlap by design (base~left_driver, left_coupler~left_follower, ...). This emits the same
    declaration from a MEASUREMENT rather than a hard-coded list, which is what keeps the witnesses
    intact -- FR3's own base/link1 pair has clearance, measures non-overlapping, and gets nothing.

    The criterion is deliberately "always", not "ever". A pair that overlaps at some poses and not
    others is a genuine configuration-dependent self-collision, which a real arm also has and a
    planner is supposed to see; excluding those would let the arm pass through itself. A pair that
    overlaps at EVERY pose can never carry collision information, so the contact is pure noise.
    """
    bodies = [b for b in range(1, model.nbody)
              if any(model.geom_bodyid[g] == b for g in range(model.ngeom))]
    geoms = {b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for b in bodies}
    collide = {b: [g for g in geoms[b] if model.geom_contype[g] and model.geom_conaffinity[g]]
               for b in bodies}

    filterparent = not bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_FILTERPARENT)
    weld = model.body_weldid
    parent_weld = lambda b: int(weld[model.body_parentid[weld[b]]])
    already = set()
    for e in range(model.nexclude):
        signature = int(model.exclude_signature[e])
        already.add((signature >> 16, signature & 0xFFFF))
        already.add((signature & 0xFFFF, signature >> 16))

    def filtered(left, right):
        """MuJoCo's own contact filter, so this only reports pairs that really do collide.

        Reproduced rather than assumed: the ``weld != 0`` guard is why ``base_link~link1`` collides
        at all. MuJoCo skips the parent-child filter when either weld body is the WORLD, and a base
        with no joint welds to the world -- so the base pair is unfiltered while ``link1~link2``,
        whose geometry also interpenetrates at the joint, is filtered. Omitting this predicate made
        the FR3 witness emit 23 exclusions against the shipped 6, nearly all of them for pairs
        MuJoCo was already discarding."""
        w1, w2 = int(weld[left]), int(weld[right])
        if w1 == w2 or (left, right) in already:
            return True
        return bool(filterparent and w1 != 0 and w2 != 0
                    and (w1 == parent_weld(right) or w2 == parent_weld(left)))

    worst = {}
    for fraction in fractions:
        data = mujoco.MjData(model)
        for j in range(model.njnt):
            if model.jnt_limited[j]:
                low, high = model.jnt_range[j]
                data.qpos[model.jnt_qposadr[j]] = low + fraction * (high - low)
        mujoco.mj_forward(model, data)
        for i, left in enumerate(bodies):
            for right in bodies[i + 1:]:
                if not collide[left] or not collide[right] or filtered(left, right):
                    continue
                gap = min(mujoco.mj_geomDistance(model, data, x, y, distmax, None)
                          for x in collide[left] for y in collide[right])
                key = (left, right)
                # keep the LEAST penetrating pose: a pair only qualifies if even its best is inside
                worst[key] = max(worst.get(key, -np.inf), gap)
    name = lambda b: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
    return tuple(sorted((name(a), name(b)) for (a, b), gap in worst.items() if gap < 0.0))


# Irrational multipliers for the deterministic configuration lattice below: fractional parts of
# sqrt(prime) are linearly independent over the rationals, so the sequence frac(k * alpha) fills each
# joint's range evenly and the joints do not march in lockstep.
_LATTICE_ALPHAS = tuple(float(np.sqrt(prime) % 1.0) for prime in
                        (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53))


def self_collision_rate(model, samples=128) -> float:
    """Fraction of random in-limit configurations in which the ARM intersects itself.

    Contacts involving the gripper do not count: pads meeting when the jaw closes is the gripper
    working, and a long tool touching the arm at full wrist fold is planner-avoided reality on
    every real long-tool robot, not an arm defect -- tool clearance is grasp_feasibility's job.
    Counting it made the measure a tool-length detector that rejected the real yam arm carrying
    its own vendor gripper.

    Measured on the shipped robots this sits near 0.05; generated arms reached 0.92 (seed3) and 0.98
    (seed7), which is why those two could not be commanded anywhere. A robot that intersects itself
    in 98% of its joint space has almost no usable configurations, and every pose a planner picks
    starts inside a contact -- including qpos=0, so even the first ramp jammed."""
    data = mujoco.MjData(model)
    hinges = [j for j in range(model.njnt) if model.jnt_type[j] in
              (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE) and model.jnt_limited[j]]
    if not hinges:
        return 0.0
    gripper = {g for g in range(model.ngeom)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                   or "").startswith(GRIPPER_PREFIX)}
    # A DETERMINISTIC lattice, not random draws: builder.py must contain no randomness, so that a
    # parameter object always yields the same model and the same measurements. An additive-recurrence
    # (Kronecker) sequence covers each range more evenly than independent uniforms anyway.
    hits = 0
    for step in range(samples):
        for index, j in enumerate(hinges):
            low, high = model.jnt_range[j]
            alpha = _LATTICE_ALPHAS[index % len(_LATTICE_ALPHAS)]
            data.qpos[model.jnt_qposadr[j]] = low + ((step + 1) * alpha % 1.0) * (high - low)
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            if not ({int(data.contact[c].geom1), int(data.contact[c].geom2)} & gripper):
                hits += 1
                break
    return hits / samples


def _dls_ik(model, data, site, target, approach, dofs, qadr, joints, seed,
            iterations=60, tolerance=2.0e-3):
    """Damped least squares on position AND approach direction, clamped to joint limits.

    Mirrors what the production planner asks of an embodiment: not "can the gripper be at this
    point" but "can it be there pointing the right way", which is the question the arm keeps failing.
    Position-only IK would report these arms as fine.
    """
    q = np.asarray(seed, float).copy()
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    want = np.array([0.0, 0.0, -1.0])
    for _ in range(iterations):
        data.qpos[qadr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_comPos(model, data)
        position = np.asarray(data.site_xpos[site], float)
        rotation = np.asarray(data.site_xmat[site], float).reshape(3, 3)
        axis = rotation @ approach
        error = np.concatenate([target - position, np.cross(axis, want)])
        if np.linalg.norm(error[:3]) < tolerance and np.linalg.norm(error[3:]) < 0.10:
            return True
        mujoco.mj_jacSite(model, data, jacp, jacr, site)
        jacobian = np.vstack([jacp[:, dofs], jacr[:, dofs]])
        step = jacobian.T @ np.linalg.solve(
            jacobian @ jacobian.T + 1.0e-4 * np.eye(6), error)
        q = q + np.clip(step, -0.4, 0.4)
        for index, joint in enumerate(joints):
            if model.jnt_limited[joint]:
                q[index] = float(np.clip(q[index], *model.jnt_range[joint]))
    return False


def grasp_feasibility(model, dof, gripper_root="", targets=24, pregrasp=0.10, lift=0.15) -> float:
    """Fraction of reachable table-height targets where a whole PICK is solvable.

    A pick needs one downward orientation held across three poses -- pregrasp above the object, the
    grasp itself, and the lift. The planner calls that cumulative feasibility, and it is what these
    embodiments fail: seed133 had 209 collision-free candidates, 2 individually solvable at the grasp
    and ZERO solvable across all three. Neither reach nor self-collision predicts it. Measured on the
    shipped robots this sits high; on sampled arms it collapses, which is exactly the gap between
    "does not intersect itself" and "can do the task".

    Deterministic throughout -- targets come from a lattice and IK is seeded from the mid-range pose
    -- because builder.py may contain no randomness.
    """
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{GRIPPER_PREFIX}{PLANNER_SITE}")
    if site < 0:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{GRIPPER_PREFIX}grasp_site")
    if site < 0:
        return 0.0
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name(i))
              for i in range(1, dof + 1)]
    if any(j < 0 for j in joints):
        return 0.0
    qadr = np.array([model.jnt_qposadr[j] for j in joints])
    dofs = np.array([model.jnt_dofadr[j] for j in joints])
    approach = np.array([0.0, 0.0, 1.0])          # gripper approach axis in the grasp frame
    data = mujoco.MjData(model)

    # the mid-range pose doubles as the IK seed and as the probe for how far the arm reaches
    home = np.array([0.5 * (model.jnt_range[j][0] + model.jnt_range[j][1])
                     if model.jnt_limited[j] else 0.0 for j in joints])
    data.qpos[qadr] = home
    mujoco.mj_kinematics(model, data)
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    origin = np.asarray(data.xpos[base], float)
    span = 0.0
    for step in range(64):
        for index, joint in enumerate(joints):
            low, high = model.jnt_range[joint]
            alpha = (step + 1) * _LATTICE_ALPHAS[index % len(_LATTICE_ALPHAS)] % 1.0
            data.qpos[qadr[index]] = low + alpha * (high - low)
        mujoco.mj_kinematics(model, data)
        span = max(span, float(np.linalg.norm(np.asarray(data.site_xpos[site], float) - origin)))
    if span < 1.0e-3:
        return 0.0

    solved = 0
    for step in range(targets):
        # lattice over (radius, azimuth, height) inside the measured reach
        radius = span * (0.35 + 0.40 * ((step + 1) * _LATTICE_ALPHAS[0] % 1.0))
        azimuth = 2.0 * np.pi * ((step + 1) * _LATTICE_ALPHAS[1] % 1.0)
        height = span * (-0.30 + 0.45 * ((step + 1) * _LATTICE_ALPHAS[2] % 1.0))
        target = origin + np.array([radius * np.cos(azimuth), radius * np.sin(azimuth), height])
        if all(_dls_ik(model, data, site, target + np.array([0.0, 0.0, offset]), approach,
                       dofs, qadr, joints, home)
               for offset in (pregrasp, 0.0, lift)):
            solved += 1
    return solved / targets


def rest_pose(model, dof, mount_height=0.0, samples=768, clearance=0.02,
              mobility=0.85, probes=32, reach_ahead=0.15,
              approach=(0.0, 0.0, 1.0), skyward=0.35):
    """A tucked start configuration with the tool held UP and IN FRONT, free to move.

    Choosing each joint's value independently -- the legal number nearest zero -- is a numerical rule
    with no notion of posture, and for a sampled skeleton "all joints near zero" points wherever the
    twists happen to point. Measured on the emitted specs that meant four of six arms holding the
    tool BELOW their own base, one with geometry 0.10 m under the floor, and extensions of 76-82% of
    reach.

    The objective is taken from what the shipped poses do, measured in the base frame
    (x forward, y sideways, z up):

        franka_droid  tool front +0.31 side +0.00 up +0.44   opening points DOWN    (up -1.00)
        i2rt_yam      tool front +0.37 side +0.02 up +0.42   opening points FORWARD (front +0.99)

    Those two bracket the orientation requirement between them: an opening pointing down or forward
    is ready to descend on something, one pointing at the ceiling is not. ``skyward`` caps the
    upward component -- YAM's own +0.10 has to pass -- and the score then rewards whichever of
    down/forward is stronger.

    Both hold the tool forward AND up, square to the base, at moderate extension, and every small
    joint perturbation around them stays collision-free. An earlier version rewarded only height and
    compactness; it satisfied those numerically while putting FR3's gripper 0.17 m BEHIND its base
    (elbow folded backwards) and YAM's pointing straight at the ceiling. Both are useless as a place
    to start reaching from, which is what a rest pose is for.

    ``mobility`` is a hard requirement rather than a scoring term because the shipped poses both sit
    at 100%: a pose wedged against its own geometry cannot be driven anywhere, so it fails outright.

    Deterministic (lattice candidates, no RNG) because builder.py may contain no randomness.
    """
    site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{GRIPPER_PREFIX}{PLANNER_SITE}")
    if site < 0:
        site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{GRIPPER_PREFIX}grasp_site")
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name(i))
              for i in range(1, dof + 1)]
    if site < 0 or any(j < 0 for j in joints):
        return None
    qadr = [model.jnt_qposadr[j] for j in joints]
    base = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    gripper = {g for g in range(model.ngeom)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g]))
                   or "").startswith(GRIPPER_PREFIX)}
    lows = np.array([model.jnt_range[j][0] if model.jnt_limited[j] else -np.pi for j in joints])
    highs = np.array([model.jnt_range[j][1] if model.jnt_limited[j] else np.pi for j in joints])
    data = mujoco.MjData(model)

    def place(q):
        for address, value in zip(qadr, q):
            data.qpos[address] = value
        mujoco.mj_forward(model, data)

    def unobstructed():
        return all({int(data.contact[c].geom1), int(data.contact[c].geom2)} <= gripper
                   for c in range(data.ncon))

    def describe(q):
        place(q)
        if not unobstructed():
            return None
        floor = min(float(data.geom_xpos[g][2]) - float(model.geom_rbound[g])
                    for g in range(model.ngeom))
        if floor + mount_height < clearance:
            return None
        origin = np.asarray(data.xpos[base], float)
        axes = np.asarray(data.xmat[base], float).reshape(3, 3)
        local = axes.T @ (np.asarray(data.site_xpos[site], float) - origin)
        opening = axes.T @ (np.asarray(data.site_xmat[site], float).reshape(3, 3)
                            @ np.asarray(approach, float))
        if opening[2] > skyward:                  # the jaws face the ceiling: useless as a start
            return None
        return local, float(np.linalg.norm(local)), opening

    candidates = [0.5 * (lows + highs), np.clip(np.zeros(dof), lows, highs)]
    for step in range(samples):
        fractions = np.array([(step + 1) * _LATTICE_ALPHAS[i % len(_LATTICE_ALPHAS)] % 1.0
                              for i in range(dof)])
        candidates.append(lows + fractions * (highs - lows))

    described = [(q, *found) for q in candidates for found in [describe(q)] if found]
    if not described:
        return None
    reach = max(item[2] for item in described) or 1.0

    # forward and up, square to the base, not overextended, jaws aimed down or ahead
    def score(local, distance, opening):
        front, side, rise = float(local[0]), abs(float(local[1])), float(local[2])
        aim = max(-float(opening[2]), float(opening[0]), 0.0)
        return (min(front, 0.35 * reach) + min(max(rise, 0.0), 0.35 * reach)
                + 0.30 * reach * aim
                - 0.6 * side - 0.4 * max(0.0, distance / reach - 0.65))

    ranked = sorted(described, key=lambda item: -score(item[1], item[2], item[3]))
    ahead = [item for item in ranked if item[1][0] > reach_ahead * reach]
    for q, local, distance, opening in (ahead or ranked):
        # mobility last: it is the expensive test, so only the best-posed candidates pay for it
        free = 0
        for step in range(probes):
            offset = np.array([((step + 1) * _LATTICE_ALPHAS[i % len(_LATTICE_ALPHAS)] % 1.0) - 0.5
                               for i in range(dof)]) * 0.5
            place(np.clip(np.asarray(q, float) + offset, lows, highs))
            free += int(unobstructed())
        if free / probes >= mobility:
            return np.asarray(q, float).copy()
    return np.asarray(ranked[0][0], float).copy()


def geometry_components(model, fraction=0.0, tolerance=None, distmax=0.6):
    """Connected components of the model's geometry-bearing bodies, by TRUE surface distance.

    The structural form of "is this one robot or several pieces?", and free of any assumption about
    the kinematic chain -- which is what the pair-walk in :func:`consecutive_breaks` relies on. Two
    bodies are joined when their closest geoms are within ``tolerance``, the clearance real robots
    exhibit (see :data:`CLEARANCE_TOLERANCE`).

    Use ``mj_geomDistance``, never ``geom_rbound``. A bounding-sphere test called every arm connected
    while 42% of them actually fell into separate pieces: the sphere around a 0.13 m half-length
    capsule has radius ~0.14 m, so surfaces 20 cm apart appeared to touch."""
    tolerance = CLEARANCE_TOLERANCE if tolerance is None else tolerance
    data = mujoco.MjData(model)
    for j in range(model.njnt):
        if model.jnt_limited[j]:
            low, high = model.jnt_range[j]
            data.qpos[model.jnt_qposadr[j]] = low + fraction * (high - low)
    mujoco.mj_forward(model, data)

    bodies = [b for b in range(1, model.nbody)
              if any(model.geom_bodyid[g] == b for g in range(model.ngeom))]
    geoms = {b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for b in bodies}
    parent = {b: b for b in bodies}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for index, a in enumerate(bodies):
        for b in bodies[index + 1:]:
            closest = min(mujoco.mj_geomDistance(model, data, x, y, distmax, None)
                          for x in geoms[a] for y in geoms[b])
            if closest <= tolerance:
                parent[find(a)] = find(b)

    groups = {}
    for b in bodies:
        groups.setdefault(find(b), []).append(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"<body{b}>")
    return [sorted(names) for names in groups.values()]


def build_spec(params: EmbodimentParams, connectors=(), exclusions=()) -> MjSpec:
    """The arm plus its grafted gripper, as an uncompiled :class:`MjSpec`.

    ``connectors`` and ``exclusions`` are both MEASURED by :func:`repaired_spec` from a compiled
    trial build and fed back in; this function itself invents neither."""
    arm = params.arm
    spec = MjSpec()
    spec.compiler.degree = False
    # Both shipped robots set integrator="implicitfast", and a stiff position servo REQUIRES it.
    # MuJoCo defaults to explicit Euler, which cannot integrate kp up to 4500 at a 2 ms timestep: the
    # witness -- identical gains, armature, damping, masses and geometry to the real robot -- ran away
    # to 6109 rad on a 0.3 rad step command and threw "Nan, Inf or huge value in QACC", while the
    # shipped model tracked the same command to 0.000 rad. Nothing about the robot was wrong; the
    # integrator was.
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    meshes = _declare_pool_meshes(spec, params)
    appearance = _declare_pool_appearance(spec, params)
    palette = _declare_palette(spec, params)
    # A gripper whose graft root IS an arm link (yam_jaw's is link_6) supplies the flange link's
    # geometry and mass itself. Dressing that link as well emits it twice: yam measured 35 gripper
    # geoms against the real 30, with link_6's 0.367 kg counted double.
    claims_flange = bool(load_catalog()[params.gripper].claims_flange_link)

    # Struts that guarantee each body touches its neighbour. Sized like a structural member rather
    # than a wire: at 0.35x the bone radius they were ~14 mm across in a dark joint colour, about
    # five pixels in a preview tile, so a fully connected arm still read as separate pieces. A connector is added to BOTH bodies of
    # a pair and covers the same world segment (the child's origin to the parent's), so the two
    # capsules necessarily overlap and the pair is in contact whatever the surrounding geometry
    # looks like. Which pairs need one is MEASURED (see build), never sampled, so a robot whose
    # geometry already meets -- every real robot -- gets none and is unchanged.
    # {pair index: measured gap}. The gap is carried through because the arm-to-gripper strut has
    # to be SIZED to close it: the gripper's own geoms start some way from its root origin, so a
    # strut merely reaching that origin can still fall short.
    connectors = dict(connectors) if isinstance(connectors, dict) else {i: 0.0 for i in connectors}

    parent = spec.worldbody.add_body(name=BASE_BODY)
    if arm.base is not None:
        # The mounting structure: FR3's fr3_link0 (8 geoms, 2.82 kg), YAM's motor housing and
        # plate. It carries no joint, so nothing else in this loop would ever emit it.
        _apply_inertial(parent, arm.base.inertial, arm.base.geom.stretch)
        _dress(parent, geom_pool.BASE_INDEX, arm.base, params, meshes, appearance, palette,
               bone=arm.segments[0].offset)

    for index, (segment, joint, link) in enumerate(zip(arm.segments, arm.joints, arm.links)):
        body = parent.add_body(name=link_name(index + 1), pos=list(segment.offset),
                               quat=list(segment.quat))
        mjs_joint = body.add_joint(name=joint_name(index + 1), type=_JOINT_TYPE[joint.kind],
                                   axis=[0.0, 0.0, 1.0], range=list(joint.range))
        mjs_joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        mjs_joint.armature = joint.armature
        mjs_joint.damping = joint.damping
        mjs_joint.frictionloss = joint.frictionloss

        if index in connectors:
            _capsule(parent, link_name(index) + "_strut_out",
                     np.asarray(segment.offset, float),
                     max(0.018, 0.75 * params.dressing.radius), palette["joint"])
        flange_link = index == arm.dof - 1 and claims_flange
        if index == arm.dof - 1 and arm.dof in connectors:
            # Bridge the last arm link to the grafted gripper. Applies whatever the gripper is: a
            # non-claiming gripper (rum, robotiq) had no repair path here at all, which left a
            # 4.2 mm gap at the wrist that nothing closed.
            gap = float(connectors[arm.dof])
            body.add_geom(name=f"{link_name(index + 1)}_strut_tool",
                          type=mujoco.mjtGeom.mjGEOM_SPHERE,
                          size=[max(0.018, gap + 3.0 * CLEARANCE_TOLERANCE), 0.0, 0.0],
                          pos=list(arm.mount_pos), material=palette["joint"], group=0)
        if flange_link and (arm.dof - 1 in connectors or arm.dof in connectors):
            # The gripper supplies this link's geometry, so it is normally bare -- but a bare body
            # is transparent, and if the chain is broken across it the strut has to live here: it is
            # the only body a single joint away from BOTH neighbours.
            _capsule(body, f"{link_name(index + 1)}_strut_reach",
                     np.asarray(arm.mount_pos, float),
                     max(0.018, 0.75 * params.dressing.radius), palette["joint"])
        if not flange_link:
            _apply_inertial(body, link.inertial, link.geom.stretch)
            # The last link's child is the MOUNT POINT, not another joint. Passing no bone left it
            # with only a hub sphere while the gripper sat up to mount_offset away, which is a gap
            # -- the same "span to your child" rule applies, with the gripper as the child.
            nxt = (arm.segments[index + 1].offset if index + 1 < arm.dof else arm.mount_pos)
            # where the PARENT's origin sits in this body's frame, so the link can reach back
            back = -quat_to_mat(segment.quat).T @ np.asarray(segment.offset, float)
            _dress(body, index, link, params, meshes, appearance, palette, bone=nxt, back=back)

        actuator = spec.add_actuator(name=joint_name(index + 1),
                                    target=joint_name(index + 1),
                                    trntype=mujoco.mjtTrn.mjTRN_JOINT)
        # MuJoCo affine position servo: (gain, bias_position, bias_velocity) = (kp, -kp, -kv).
        # biastype MUST be affine. With the default mjBIAS_NONE, MuJoCo ignores biasprm entirely and
        # the actuator becomes a pure torque source (force = kp * ctrl), so commanding a joint
        # ANGLE applies a constant torque instead: arms slammed to their limits with 2.6-5.2 rad of
        # tracking error. Every shipped robot sets biastype affine, via a default class.
        actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        actuator.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        actuator.gainprm[0] = joint.kp
        actuator.biasprm[1] = -joint.kp
        actuator.biasprm[2] = -joint.kv
        actuator.ctrlrange = list(joint.range)
        actuator.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        if joint.force > 0.0:                  # 0 means unlimited, the way FR3 leaves its arm
            actuator.forcerange = [-joint.force, joint.force]
            actuator.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        if index in connectors or (flange_link and arm.dof in connectors):
            _hub(body, link_name(index + 1) + "_strut_in",
                 max(0.018, 0.75 * params.dressing.radius), palette["joint"])
        parent = body

    flange = parent.add_body(name=FLANGE_BODY, pos=[0.0, 0.0, 0.0])
    # The tool plate: where the gripper bolts on. FR3 declares the same thing as attachment_site,
    # and it lands at exactly the mount transform.
    hosts = {"flange": flange, "base": spec.body(BASE_BODY)}
    for site in arm.sites:
        host = hosts.get(site.mount)
        if host is None:
            raise BuildError(f"unknown site mount {site.mount!r}")
        host.add_site(name=site.name, pos=list(site.pos), quat=list(site.quat), group=site.group)
    _add_cameras(spec, params, base_body=hosts["base"], flange_body=flange)
    graft(spec, flange, params.gripper, pos=arm.mount_pos, quat=arm.mount_quat)
    # Contact exclusions for pairs that overlap by construction; see permanent_overlaps. Emitted
    # here so they land in the XML the same way FR3 ships its six.
    for first, second in exclusions:
        spec.add_exclude(bodyname1=first, bodyname2=second)
    return spec


def _spaced(params: EmbodimentParams, factor: float) -> EmbodimentParams:
    """The same arm with every segment offset scaled, geometry untouched.

    This is the one knob that separates how BIG the links are from how far APART they sit, which is
    the whole content of the defect: a sampled arm wears geometry harvested at one robot's scale on
    spacing drawn at another's, so its links pile into a ball. Scaling offsets leaves masses,
    inertias, meshes and joint limits exactly as sampled -- only the skeleton stretches."""
    segments = tuple(replace(segment, offset=tuple(factor * np.asarray(segment.offset, float)))
                     for segment in params.arm.segments)
    return replace(params, arm=replace(params.arm, segments=segments))


def spread(params: EmbodimentParams, budget: float = SELF_COLLISION_BUDGET, rounds: int = 8):
    """``params`` with the skeleton stretched until the arm stops intersecting itself.

    NOT in the build path: it works, and the cure is worse than the disease. Scaling only ever goes
    UP -- the factor starts at 1.0 and multiplies -- so every arm that self-collides gets bigger and
    none gets smaller, which skews the whole built population. Measured over 60 seeds: 100% of arms
    were scaled, mean chain length went 1.05 m -> 2.92 m with a 5.10 m maximum, and 20 of 60 breached
    the declared ``chain_length`` support of 3.20 m. For scale, the real robots are 0.76 m and
    1.21 m. Worse, the mean factor was 2.82 against a cap of 3.0, meaning most arms ran to the limit
    and still missed the budget -- so it bought a self-collision rate of 0.48 instead of 0.80 by
    turning the population into five-metre arms that no longer resemble anything real.

    Keeping generation honest matters more than the rate: a self-collision defect is visible and
    documented, whereas a silently 3x-inflated population violating its own declared support is
    neither. The rate is a GEOMETRY problem and has to be fixed where the geometry is chosen.

    Both shipped robots sweep their joint space without a single self-contact; sampled arms did so in
    83-100% of poses, interpenetrating by 50-72 mm, and the offenders were always distal parts inside
    proximal ones (``gripper/base~link1``, ``base_link~link4``) -- an arm folded into a ball rather
    than extended into a chain.

    Two other levers were tried and MEASURED WRONG first, both recorded here so they are not retried:

    * Narrowing joint ranges (see :func:`unfolded`) made it worse, because shrinking a range about
      its centre concentrates the arm at its mid-range pose, which is itself the folded one.
    * Requiring bounding spheres to clear (``|offset| >= bound_i + bound_j``) looks principled and is
      not: FR3 violates it on 6 of 6 non-adjacent pairs and YAM on 4 of 5, while both collide 0% of
      the time. Bounding spheres are far too conservative for shaped links -- the same error that
      once made ``geom_rbound`` report every disconnected arm as connected.

    So the criterion is the one the real robots actually satisfy, measured the way they satisfy it:
    TRUE surface distance over the joint space. A robot already inside the budget is returned
    untouched, which is what keeps both witnesses byte-exact.
    """
    factor = 1.0
    for _ in range(rounds):
        current = params if factor == 1.0 else _spaced(params, factor)
        try:
            model = build_spec(current).compile()
        except (ValueError, RuntimeError) as error:
            raise BuildError(f"{params.embodiment_id}: {error}") from error
        if self_collision_rate(model) <= budget:
            return current
        # Cap on total stretch: chain_length has a declared support band, and an arm that still
        # folds into itself at this much spacing is better rejected than silently made enormous.
        if factor >= SPACING_LIMIT:
            return current
        factor = min(SPACING_LIMIT, factor * SPACING_STEP)
    return _spaced(params, factor)


def _narrow(params: EmbodimentParams, factor: float) -> EmbodimentParams:
    """The same arm with every joint range shrunk about its own centre.

    Narrowing keeps each range inside ``SUPPORT["range_span"]``, so the result is a point the
    sampler could itself have drawn -- the repair moves within the declared distribution rather than
    stepping outside it. Centres are untouched, so an asymmetric range like FR3's joint4
    [-3.042, -0.152] stays asymmetric."""
    floor = SUPPORT["range_span"][0]
    joints = []
    for joint in params.arm.joints:
        low, high = joint.range
        centre, span = 0.5 * (low + high), high - low
        span = max(floor, span * factor)
        joints.append(replace(joint, range=(centre - 0.5 * span, centre + 0.5 * span)))
    return replace(params, arm=replace(params.arm, joints=tuple(joints)))


def unfolded(params: EmbodimentParams, budget: float = SELF_COLLISION_BUDGET, rounds: int = 5):
    """``params`` with joint ranges narrowed until the arm stops living inside itself.

    NOT in the build path: measured, and it does not work. Narrowing shrinks each range about its
    own CENTRE, which concentrates the arm near its mid-range pose -- and for these arms that pose is
    itself folded, so the rate went the wrong way (seed3 0.71 -> 0.98, seed7 0.94 -> 0.98). seed0
    measures 1.00 with no scenery present at all, meaning it has no collision-free configuration for
    any limits, so joint range is provably the wrong lever. Kept for the record and for anyone
    tempted to try it again.

    A designed arm keeps its self-collision rate low with joint limits; a sampled one draws limits
    with no knowledge of the geometry they have to clear, so seed3 intersected itself in 92% of its
    joint space and seed7 in 98% -- against ~5% for both shipped robots. Neither could be commanded
    anywhere: qpos=0 was already in contact, so the very first motion jammed against itself.

    Narrowing is the mechanism real arms use, and it is measurement-driven and monotone, so this
    terminates. Robots already inside the budget -- both witnesses included -- are returned
    untouched, which is what keeps witness parity exact."""
    current, factor = params, 1.0
    for _ in range(rounds):
        try:
            model = build_spec(current).compile()
        except (ValueError, RuntimeError) as error:
            raise BuildError(f"{params.embodiment_id}: {error}") from error
        if self_collision_rate(model) <= budget:
            return current
        factor *= RANGE_SHRINK
        current = _narrow(params, factor)
    return current


def _freeze_inertials(spec: MjSpec, model) -> None:
    """Write every body's compiled inertial back onto the spec as an explicit one.

    A body whose inertia is DERIVED from its geoms loses all of it when those geoms go away, and
    consumers do remove them: MolmoSpaces validates a scene with ``strip_meshes=True``, at which
    point a mesh-only link weighs nothing and MuJoCo refuses the model with "mass and inertia of
    moving bodies must be larger than mjMINVAL". 29 of 30 sampled arms had at least one such link.

    Both shipped robots avoid this by shipping explicit inertials on every link -- FR3 has 7, YAM
    has 8 -- so this is what the real models already do, not a new invention. Freezing preserves the
    values MuJoCo itself computed, so the geom-derived diversity survives while the numbers stop
    depending on the geometry still being there."""
    for body in spec.bodies:
        index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body.name)
        if index < 0 or float(model.body_mass[index]) <= 0.0:
            continue
        body.mass = float(model.body_mass[index])
        body.inertia = [float(v) for v in model.body_inertia[index]]
        body.ipos = [float(v) for v in model.body_ipos[index]]
        body.iquat = [float(v) for v in model.body_iquat[index]]
        body.explicitinertial = True


def repaired_spec(params: EmbodimentParams, tolerance: float = CLEARANCE_TOLERANCE,
                  wrist_camera: bool = False) -> MjSpec:
    """The model's spec WITH the connector struts measurement says it needs.

    Anything that renders or exports a generated robot must go through here rather than calling
    :func:`build_spec` directly. ``build_spec`` emits no struts, so a caller that skips this step
    gets the unrepaired robot -- which is exactly how every preview image came to show disconnected
    limbs while the tests, which do go through here, reported none."""
    root = load_catalog()[params.gripper].graft_root

    def compiled(connectors, exclusions=()):
        spec = build_spec(params, connectors, exclusions)
        try:
            return spec, spec.compile()
        except (ValueError, RuntimeError) as error:
            raise BuildError(f"{params.embodiment_id}: {error}") from error

    # Iterate to a FIXED POINT. One round is not enough: adding struts to a bare flange link makes
    # it opaque, which splits a pair that was previously spanned through it and can expose a gap
    # that was never measured. Converges in a couple of rounds; the cap is a backstop.
    spec, model = compiled({})
    connectors = {}
    for _ in range(5):
        broken = {index: gap for index, gap
                  in consecutive_breaks(model, params.arm.dof, root,
                                        tolerance=tolerance).items() if gap > tolerance}
        if not broken or all(index in connectors and gap <= connectors[index]
                             for index, gap in broken.items()):
            break
        for index, gap in broken.items():
            connectors[index] = max(connectors.get(index, 0.0), gap)
        spec, model = compiled(connectors)
    # Struts are what make consecutive geometry overlap, so exclusions are measured LAST, on the
    # repaired model. Doing it earlier would miss the overlaps the repair itself introduces.
    # (self-collision narrowing happens before all of this, in build/repaired_spec's caller)
    exclusions = permanent_overlaps(model)
    if exclusions:
        spec, model = compiled(connectors, exclusions)
    # Last, once the geometry is final: bake the inertials so the model survives a consumer that
    # strips its meshes.
    _freeze_inertials(spec, model)
    # The canonical wrist camera needs the grasp site's compiled pose, so it goes on after the final
    # compile. The site itself is spec-scoped, so resolve against a throwaway copy that has it.
    if not wrist_camera:
        # OPT-IN, because it changes the model's camera set: a witness must reproduce its source
        # exactly (FR3 ncam 2, YAM ncam 0), and adding a camera to every build would put both
        # shipped robots one camera out. Emission turns it on -- see spec_emit.emit -- because that
        # is where a robot is prepared for a consumer, and the consumer is what needs the name.
        return spec
    probe = spec.copy()
    for operation in load_catalog()[params.gripper].model_operations:
        if operation.get("kind") != "add_site" or operation.get("scope") != "robot_model":
            continue
        parent = probe.body(operation["parent_body"]["name"])
        if parent is None:
            continue
        arguments = dict(name=operation["name"]["name"], pos=operation["position"], group=4)
        if operation.get("quaternion_wxyz"):
            arguments["quat"] = operation["quaternion_wxyz"]
        parent.add_site(**arguments)
    try:
        _add_wrist_camera(spec, params, probe.compile())
    except (ValueError, RuntimeError) as error:
        raise BuildError(f"{params.embodiment_id}: wrist camera: {error}") from error
    return spec


def acceptable(model, dof: int = 0, gripper_root: str = "",
               collision_limit: float = SELF_COLLISION_LIMIT,
               feasibility_limit: float = GRASP_FEASIBILITY_LIMIT) -> bool:
    """Whether a built model is usable: it stays out of itself AND can actually perform a pick."""
    if self_collision_rate(model) > collision_limit:
        return False
    if dof and grasp_feasibility(model, dof, gripper_root) < feasibility_limit:
        return False
    return True


def build(params: EmbodimentParams, tolerance: float = CLEARANCE_TOLERANCE,
          max_self_collision: float | None = None,
          min_grasp_feasibility: float | None = None,
          wrist_camera: bool = False) -> tuple:
    """``(mjcf text, compiled model)``. A compile failure is a BuildError, i.e. a rejection.

    Builds, MEASURES whether every consecutive pair of bodies is in contact, and rebuilds with
    connecting struts wherever it is not. Contact is therefore an invariant of the output rather
    than a property some samples happen to have -- and because struts are added only where
    measurement says they are needed, a robot whose geometry already meets gets none."""
    spec = repaired_spec(params, tolerance, wrist_camera=wrist_camera)
    try:
        xml, model = spec.to_xml(), spec.compile()
    except (ValueError, RuntimeError) as error:
        raise BuildError(f"{params.embodiment_id}: {error}") from error
    # Rejection is OPT-IN, so callers that study the whole population -- the invariant tests, any
    # measurement of the defect itself -- still see every robot. Consumers that need a usable robot
    # pass a limit; see spec_emit.acceptable_seeds.
    if max_self_collision is not None:
        rate = self_collision_rate(model)
        if rate > max_self_collision:
            raise BuildError(f"{params.embodiment_id}: self-collision {rate:.2f} "
                             f"exceeds {max_self_collision:.2f}")
    if min_grasp_feasibility is not None:
        root = load_catalog()[params.gripper].graft_root
        share = grasp_feasibility(model, params.arm.dof, root)
        if share < min_grasp_feasibility:
            raise BuildError(f"{params.embodiment_id}: pick feasibility {share:.2f} "
                             f"below {min_grasp_feasibility:.2f}")
    return xml, model



def chain_in_model(model, dof: int, prefix: str = "") -> tuple:
    """``(origins, axes)`` of the arm joints in the base-link frame, read back from a compiled
    model at qpos=0. Used to check a build against what the parameters asked for."""
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + BASE_BODY)
    root_pos = np.array(data.xpos[root], float)
    root_mat = np.array(data.xmat[root], float).reshape(3, 3)
    origins, axes = [], []
    for index in range(1, dof + 1):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + joint_name(index))
        body = model.jnt_bodyid[jid]
        origins.append(root_mat.T @ (np.array(data.xpos[body], float) - root_pos))
        axes.append(root_mat.T @ (np.array(data.xmat[body], float).reshape(3, 3)
                                 @ np.array(model.jnt_axis[jid], float)))
    return np.stack(origins), np.stack(axes)


def gripper_actuator_count(model) -> int:
    return sum(1 for a in range(model.nu)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or "")
               .startswith(GRIPPER_PREFIX))
