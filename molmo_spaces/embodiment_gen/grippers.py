"""The gripper catalog: MolmoSpaces' shipped grippers, made graftable.

A generated arm never synthesizes a jaw. It grafts a curated gripper, which means the
gripper's actuator, its coupling constraints and its tendon come along INSIDE the asset --
so a generated robot's gripper behaves exactly like the shipped robot it came from, and
inherits that gripper's grasp annotations for free.

The graft is ``MjsFrame.attach_body``, the same mechanism ``molmo_spaces.robots.franka``
uses to put a robot into a scene. It carries actuators, equality constraints, tendons,
sites and mesh assets across specs, with every contributed name under ``GRIPPER_PREFIX``.
Verified: a gripper grafted onto an arm is bit-identical to the same gripper compiled
standalone.

Four things bite, all handled by :func:`source_spec`:

* ``attach_body`` CONSUMES the child spec, so callers get a fresh ``copy()`` each time.
* Source keyframes go stale the moment the root joint is dropped ("Keyframe 'home' has
  invalid qpos size").
* The graft root's own joint must be dropped: the floating grippers are rooted on a free
  joint ("free joint can only be used on top level"), and yam's gripper is rooted on the
  arm's last driven link.
* Unreferenced meshes ride along -- rby1's hand drags in 1112 of them -- and pruning must
  key on ``name or Path(file).stem``, because meshes declared by file alone have an EMPTY
  name until compile time. Keying on ``name`` deletes every mesh in the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import mujoco
import yaml
from mujoco import MjSpec

from molmo_spaces.molmo_spaces_constants import get_robot_path

import numpy as np

from .constants import CATALOG_DIR, GRIPPER_PREFIX
from .skeleton import mat_to_quat, quat_to_mat


def _compose(a, b):
    """(pos, R) composition: a then b."""
    return a[0] + a[1] @ b[0], a[1] @ b[1]


def _invert(position, rotation):
    return -rotation.T @ position, rotation.T


@dataclass(frozen=True)
class GripperAsset:
    """One catalog entry: how to obtain and mount a gripper, plus what it MEANS.

    The graft fields are derived from the source robot's MJCF and spec by
    ``scripts/embodiments/make_gripper_catalog.py`` -- nothing here is hand-measured.
    """

    name: str
    # --- where the model comes from
    resource_key: str
    resource_version: str
    mjcf: str
    graft_root: str            # the move group's root_body
    drop_joints: tuple         # that body's own joints (free joint, or a driven arm joint)
    keep_meshes: tuple         # meshes actually referenced by the grafted subtree
    # True when graft_root is one of the SOURCE robot's own arm links, as yam's link_6 is. Such a
    # gripper supplies the flange link's geometry and mass, so the arm must not dress that link
    # too. Derived by make_gripper_catalog, never hand-set.
    claims_flange_link: bool = False
    # --- semantics, lifted verbatim from the source EmbodimentSpec
    move_group: dict = field(default_factory=dict)
    controller: dict = field(default_factory=dict)
    grasp_profile_id: str = ""
    grasp_profile: dict = field(default_factory=dict)
    initial_qpos: list = field(default_factory=list)
    sensors: list = field(default_factory=list)
    model_operations: list = field(default_factory=list)
    # The wrist camera, expressed in the gripper's own GRASP FRAME so it follows the jaws wherever
    # they are. Grippers ship one under disagreeing names -- robotiq calls it wrist_camera, rum calls
    # it wrist_cam, yam_jaw has none -- so the builder re-declares one canonical name from this,
    # exactly as it does for planner_grasp_site. Upstream cannot be renamed: yam_jaw's MJCF *is*
    # i2rt_yam's arm (adding a camera there would change the shipped robot's ncam), and RUM's
    # wrist_cam name is load-bearing in scripts/datagen/robot_conversion_patches.py.
    wrist_camera: dict = field(default_factory=dict)
    # {actuator name: {MjsActuator field: values}}, applied at graft time. The in-repo way to
    # depart from a shipped asset whose actuation is wrong: the asset is a shared versioned
    # install this repo cannot edit, so the correction lives here, next to the version pin it
    # departs from. Empty means the copy is faithful.
    actuator_override: dict = field(default_factory=dict)
    # Rotation composed at graft time for assets whose own frame does not put the fingers along
    # the mount axis (floating_rum points them +x; the catalog convention is flange +z). Empty
    # means identity.
    mount_quat: list = field(default_factory=list)

    @property
    def model_path(self) -> Path:
        # An empty resource_key means the model is bundled under embodiment_gen/vendor/
        # rather than installed as a shipped robot asset.
        if not self.resource_key:
            return Path(__file__).parent / self.mjcf
        return get_robot_path(self.resource_key) / self.mjcf


def _catalog_path(name: str) -> Path:
    return CATALOG_DIR / f"{name}.yaml"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    """``{name: GripperAsset}`` for every entry in ``gripper_catalog/``."""
    out = {}
    for path in sorted(CATALOG_DIR.glob("*.yaml")):
        data = yaml.safe_load(path.read_text())
        data["drop_joints"] = tuple(data.get("drop_joints") or ())
        data["keep_meshes"] = tuple(data.get("keep_meshes") or ())
        out[data["name"]] = GripperAsset(**data)
    if not out:
        raise FileNotFoundError(
            f"no gripper catalog entries under {CATALOG_DIR} -- build them with "
            f"python scripts/embodiments/make_gripper_catalog.py")
    return out


def mesh_key(mesh) -> str:
    """A mesh's compile-time name. Meshes declared by ``file`` alone have an empty ``name``
    until MuJoCo derives it from the filename, so keying on ``name`` alone matches nothing
    and silently prunes the whole asset block."""
    return mesh.name or Path(mesh.file).stem


@lru_cache(maxsize=8)
def _template(name: str) -> MjSpec:
    """The gripper's source spec, re-rooted and pruned once per process. Never grafted
    directly -- see :func:`source_spec`, since attach consumes its argument."""
    asset = load_catalog()[name]
    spec = MjSpec.from_file(str(asset.model_path))

    for key in list(spec.keys):                     # stale once the root joint goes
        spec.delete(key)

    for joint_name in asset.drop_joints:
        joint = spec.joint(joint_name)
        if joint is None:
            raise ValueError(f"{name}: catalog names joint {joint_name!r}, absent from "
                             f"{asset.model_path}")
        for actuator in list(spec.actuators):       # an actuator on a dropped joint dangles
            if actuator.target == joint_name:
                spec.delete(actuator)
        spec.delete(joint)

    if asset.keep_meshes:
        keep = set(asset.keep_meshes)
        for mesh in list(spec.meshes):
            if mesh_key(mesh) not in keep:
                spec.delete(mesh)

    # Resolve mesh AND texture files to ABSOLUTE paths before the graft. A gripper MJCF names them
    # relative to its own compiler meshdir/texturedir, and attach_body carries the assets across but
    # not those directories -- so the grafted arm ends up referencing bare filenames like
    # "spring_link.stl" or "metal.png". That compiles only while the working directory happens to be
    # the gripper's and fails the moment the generated model is written anywhere else, which is
    # exactly what loading an emitted embodiment does. Textures were missed on the first pass and
    # cost both rum-gripper seeds their entire run ("Error opening file 'metal.png'"), even though
    # geom_pool.textures already resolves the same file for the ARM side of the graft.
    source_dir = Path(asset.model_path).parent
    for assets, subdir in ((spec.meshes, spec.meshdir), (spec.textures, spec.texturedir)):
        root = Path(subdir) if subdir else Path()
        if not root.is_absolute():
            root = source_dir / root
        for item in assets:
            if item.file and not Path(item.file).is_absolute():
                item.file = str((root / item.file).resolve())
    return spec


def source_spec(name: str) -> MjSpec:
    """A fresh, graftable copy of the gripper's model."""
    return _template(name).copy()


def graft(arm: MjSpec, flange, name: str, pos=None, quat=None):
    """Attach gripper ``name`` to ``flange`` (an ``MjsBody`` of ``arm``) under ``GRIPPER_PREFIX``.

    ``pos``/``quat`` place the gripper ROOT relative to the flange, and mean the same thing for
    every catalog entry. Getting there requires cancelling the root body's own pose, which
    ``attach_body`` carries along: attaching at a bare frame composes ``frame o source_root``, so a
    gripper whose root is not at its model's origin lands somewhere else entirely. yam_jaw's root is
    the arm's own link_6, whose transform the arm has already applied, so an uncancelled graft put
    its wrist plate and fingers visibly adrift from the arm. Composing the inverse makes the result
    exactly ``pos``/``quat`` regardless of the asset."""
    asset = load_catalog()[name]
    child = source_spec(name)
    body = child.body(asset.graft_root)
    if body is not None:
        cancel_pos, cancel_mat = _invert(np.asarray(body.pos, float), quat_to_mat(body.quat))
        if asset.mount_quat:
            cancel_pos, cancel_mat = _compose(
                (np.zeros(3), quat_to_mat(asset.mount_quat)), (cancel_pos, cancel_mat))
        mount_pos, mount_mat = _compose(
            (np.asarray(pos if pos is not None else (0.0, 0.0, 0.0), float),
             quat_to_mat(quat if quat is not None else (1.0, 0.0, 0.0, 0.0))),
            (cancel_pos, cancel_mat))
        frame = flange.add_frame(pos=[float(v) for v in mount_pos],
                                quat=list(mat_to_quat(mount_mat)))
    else:
        frame = flange.add_frame()
    if body is None:
        raise ValueError(f"{name}: graft root {asset.graft_root!r} absent from "
                         f"{asset.model_path}")
    attached = frame.attach_body(body, GRIPPER_PREFIX, "")
    for actuator_name, override in asset.actuator_override.items():
        actuator = arm.actuator(GRIPPER_PREFIX + actuator_name)
        for field_name, value in override.items():
            getattr(actuator, field_name)[: len(value)] = value
    return attached


def used_meshes(model: mujoco.MjModel, prefix: str = GRIPPER_PREFIX) -> set:
    """Names (prefix-stripped) of the meshes a compiled model's geoms actually reference.
    Used to compute a catalog entry's ``keep_meshes``."""
    names = set()
    for i in range(model.ngeom):
        if model.geom_type[i] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, int(model.geom_dataid[i]))
        if name:
            names.add(name.removeprefix(prefix))
    return names
