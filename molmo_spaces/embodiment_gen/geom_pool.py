"""Real link geometry, harvested from the shipped robots, as a pool a generated link can wear.

Why a pool rather than a widened primitive sampler: a generated arm dressed only in capsules can
never *be* an FR3, whose 28 arm geoms are all meshes (group 2 visual + group 3 collision), nor a
YAM, which mixes 12 visual meshes with 10 collision capsules. The natural mechanism is therefore
a mixture -- a link is dressed either with sampled primitives or with a real link's complete geom
set, poses, groups and contact flags intact -- so the real robots sit inside the distribution
instead of beside it.

Harvesting the *whole* set per link, not individual geoms, is what keeps a pool draw physically
coherent: FR3's link5 is three visual meshes plus one collision hull whose relative placement is
the link, and mixing pieces across links would produce geometry no robot has.

The pool is keyed ``(source, link_index)``. :mod:`.params` samples that key without touching
disk -- it reads only :data:`SOURCES` -- while :mod:`.builder` resolves the actual geoms. Same
split as the gripper catalog, for the same reason: sampling must not depend on asset state.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import mujoco
import numpy as np

from molmo_spaces.molmo_spaces_constants import get_robot_path

# Declared pool shape. `n_links` is asserted against the live harvest by
# test_geom_pool_matches_declaration, so this cannot silently go stale -- and because params
# samples from it, a change here is a deliberate GENERATOR_VERSION bump.
# The base is a link like any other -- it just has no joint. Harvesting keys off joints, so
# without an explicit index the mounting structure is silently dropped: FR3's fr3_link0 is 8 geoms
# and 2.82 kg, YAM's `arm` root is the motor housing and its mounting plate.
BASE_INDEX = -1

SOURCES = {
    "franka_droid": dict(
        mjcf="model.xml",
        exclude=("finger", "driver", "spring", "follower", "hand", "gripper"),
        n_links=7,
    ),
    "i2rt_yam": dict(mjcf="yam.xml", exclude=("finger", "lf_", "rf_"), n_links=6),
}


@dataclass(frozen=True)
class GeomSpec:
    """One geom of a harvested link, in that link's own frame."""

    type: str                 # "mesh" | "capsule" | "box" | "cylinder" | "sphere" | "ellipsoid"
    mesh: str | None          # mesh name in the SOURCE model, or None for a primitive
    size: tuple
    pos: tuple
    quat: tuple
    group: int
    contype: int
    conaffinity: int
    material: str | None      # material name in the SOURCE model
    rgba: tuple               # per-geom colour, used when no material is assigned

    @property
    def collides(self) -> bool:
        return bool(self.contype or self.conaffinity)


@dataclass(frozen=True)
class MaterialSpec:
    """A material of a source model. Appearance is part of the embodiment, not decoration: it is
    what a vision policy sees, so it has to be representable or the real robots are only
    *geometrically* in distribution."""

    name: str
    rgba: tuple
    specular: float
    shininess: float
    reflectance: float
    emission: float
    texrepeat: tuple
    textures: tuple           # texture names by role, "" where unset


@dataclass(frozen=True)
class TextureSpec:
    """A texture of a source model: a file, or a MuJoCo builtin (checker/gradient/flat)."""

    name: str
    type: int
    file: str | None          # absolute path, resolved against the source model's texturedir
    builtin: int
    width: int
    height: int
    rgb1: tuple
    rgb2: tuple
    markrgb: tuple


@dataclass(frozen=True)
class MountedCameraSpec:
    """A camera the SOURCE robot declares in its MJCF, with the body that carries it.

    Wrist cameras are always ``MjcfCameraConfig`` in MolmoSpaces -- the config names them and the
    MODEL fixes their pose -- so a generated robot that does not declare one simply has no wrist
    camera, and any config referencing it fails to resolve. Camera placement is therefore part of
    the embodiment, not of the experiment."""

    name: str
    parent: str               # the arm body the mount hangs off
    body_pos: tuple           # mount body pose relative to that parent
    body_quat: tuple
    pos: tuple                # camera pose within the mount body
    quat: tuple
    fovy: float
    resolution: tuple


@dataclass(frozen=True)
class MountedSiteSpec:
    """A site the source robot declares on an arm body (FR3's ``attachment_site``)."""

    name: str
    parent: str
    pos: tuple
    quat: tuple
    group: int


@dataclass(frozen=True)
class LinkGeoms:
    """Every geom of one real arm link, plus the inertial MuJoCo ended up with."""

    source: str
    index: int
    body: str
    geoms: tuple
    mass: float
    inertia: tuple
    com: tuple
    # The link's OWN pose relative to its parent in the source robot. Real link geometry is rigid:
    # it spans the source's spacing, so a link wearing it needs that spacing too or the geometry
    # does not reach the next joint and the arm renders as disconnected pieces.
    offset: tuple = (0.0, 0.0, 0.0)
    quat: tuple = (1.0, 0.0, 0.0, 0.0)


def _geom_type_name(value) -> str:
    return mujoco.mjtGeom(int(value)).name.replace("mjGEOM_", "").lower()


@lru_cache(maxsize=8)
def harvest(source: str) -> tuple:
    """``(LinkGeoms, ...)`` for one source robot's serial arm links, proximal -> distal.

    Geom placement is read from the SPEC (authored values), while mass properties are read from
    the compiled model. That split is not incidental: MuJoCo recenters mesh assets at compile
    time and folds the shift into ``geom_pos``, so harvesting the compiled pos and then
    re-declaring the same mesh applies the shift twice and moves the geometry. Authored values
    re-compile to exactly the source's compiled values.
    """
    if source not in SOURCES:
        raise KeyError(f"{source!r} is not a geom pool source; known: {sorted(SOURCES)}")
    config = SOURCES[source]
    path = get_robot_path(source) / config["mjcf"]
    spec = mujoco.MjSpec.from_file(str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    name = lambda objtype, i: mujoco.mj_id2name(model, objtype, i) or ""

    joints = [j for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
              and not any(s in name(mujoco.mjtObj.mjOBJ_JOINT, j) for s in config["exclude"])]

    links = []
    for index, joint in enumerate(joints):
        body_id = int(model.jnt_bodyid[joint])
        body_name = name(mujoco.mjtObj.mjOBJ_BODY, body_id)
        spec_body = spec.body(body_name)
        if spec_body is None:
            raise KeyError(f"{source}: body {body_name!r} present in the model, absent from spec")
        geoms = _geoms_of(spec_body)
        links.append(LinkGeoms(
            source=source, index=index, body=body_name, geoms=geoms,
            mass=float(model.body_mass[body_id]),
            inertia=tuple(float(v) for v in model.body_inertia[body_id]),
            com=tuple(float(v) for v in model.body_ipos[body_id]),
            offset=tuple(float(v) for v in model.body_pos[body_id]),
            quat=tuple(float(v) for v in model.body_quat[body_id]),
        ))
    return tuple(links)


def _geoms_of(spec_body) -> tuple:
    if spec_body is None:
        return ()
    return tuple(GeomSpec(
        type=_geom_type_name(geom.type),
        mesh=(geom.meshname or None),
        size=tuple(float(v) for v in geom.size),
        pos=tuple(float(v) for v in geom.pos),
        quat=tuple(float(v) for v in geom.quat),
        group=int(geom.group),
        contype=int(geom.contype),
        conaffinity=int(geom.conaffinity),
        material=(geom.material or None),
        rgba=tuple(float(v) for v in geom.rgba),
    ) for geom in spec_body.geoms)


@lru_cache(maxsize=8)
def harvest_base(source: str) -> LinkGeoms:
    """The source robot's BASE link: the body joint 1 hangs off.

    It carries no joint, so the joint-keyed harvest skips it -- which loses the entire mounting
    structure (FR3's fr3_link0: 8 geoms, 2.82 kg; YAM's `arm` root: the motor housing and its
    mounting plate). Indexed as :data:`BASE_INDEX`."""
    config = SOURCES[source]
    path = get_robot_path(source) / config["mjcf"]
    spec = mujoco.MjSpec.from_file(str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    name = lambda objtype, i: mujoco.mj_id2name(model, objtype, i) or ""
    joints = [j for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
              and not any(s in name(mujoco.mjtObj.mjOBJ_JOINT, j) for s in config["exclude"])]
    body_id = int(model.body_parentid[model.jnt_bodyid[joints[0]]])
    body_name = name(mujoco.mjtObj.mjOBJ_BODY, body_id)
    return LinkGeoms(source=source, index=BASE_INDEX, body=body_name,
                     geoms=_geoms_of(spec.body(body_name)),
                     mass=float(model.body_mass[body_id]),
                     inertia=tuple(float(v) for v in model.body_inertia[body_id]),
                     com=tuple(float(v) for v in model.body_ipos[body_id]))


def link_geoms(source: str, index: int) -> LinkGeoms:
    """One pool entry. ``BASE_INDEX`` selects the base; other indices wrap, so a sampled 8-DoF arm
    can dress from a 6-link source without the sampler knowing each source's length."""
    if index == BASE_INDEX:
        return harvest_base(source)
    pool = harvest(source)
    return pool[index % len(pool)]


@lru_cache(maxsize=8)
def mesh_files(source: str) -> dict:
    """``{mesh name: absolute path}`` for the source model, so the builder can re-declare the
    meshes a pool draw references.

    Resolution must honour the source model's ``meshdir``: both shipped robots keep their meshes
    in an ``assets/`` subdirectory, so resolving against the MJCF's directory alone finds nothing.
    Absolute paths are returned because one generated model may mix links from two source robots,
    and MuJoCo has only one ``meshdir`` to give."""
    mjcf = get_robot_path(source) / SOURCES[source]["mjcf"]
    spec = mujoco.MjSpec.from_file(str(mjcf))
    root = mjcf.parent / (spec.meshdir or "")
    out = {}
    for mesh in spec.meshes:
        if not mesh.file:
            continue
        key = mesh.name or mesh.file.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        out[key] = str((root / mesh.file).resolve())
    return out


@lru_cache(maxsize=8)
def materials(source: str) -> dict:
    """``{name: MaterialSpec}`` for one source model."""
    mjcf = get_robot_path(source) / SOURCES[source]["mjcf"]
    spec = mujoco.MjSpec.from_file(str(mjcf))
    return {material.name: MaterialSpec(
        name=material.name,
        rgba=tuple(float(v) for v in material.rgba),
        specular=float(material.specular),
        shininess=float(material.shininess),
        reflectance=float(material.reflectance),
        emission=float(material.emission),
        texrepeat=tuple(float(v) for v in material.texrepeat),
        textures=tuple(str(t or "") for t in material.textures),
    ) for material in spec.materials if material.name}


@lru_cache(maxsize=8)
def textures(source: str) -> dict:
    """``{name: TextureSpec}`` for one source model, file paths made absolute.

    Neither reference robot uses any -- both declare zero textures -- but other shipped assets do
    (floating_rum ships metal.png), so the path exists rather than being assumed away."""
    mjcf = get_robot_path(source) / SOURCES[source]["mjcf"]
    spec = mujoco.MjSpec.from_file(str(mjcf))
    root = mjcf.parent / (spec.texturedir or "")
    out = {}
    for texture in spec.textures:
        if not texture.name:
            continue
        out[texture.name] = TextureSpec(
            name=texture.name, type=int(texture.type),
            file=str((root / texture.file).resolve()) if texture.file else None,
            builtin=int(texture.builtin), width=int(texture.width), height=int(texture.height),
            rgb1=tuple(float(v) for v in texture.rgb1),
            rgb2=tuple(float(v) for v in texture.rgb2),
            markrgb=tuple(float(v) for v in texture.markrgb),
        )
    return out


@lru_cache(maxsize=8)
def harvest_cameras(source: str) -> tuple:
    """Cameras the source declares on its ARM (gripper-side ones ride along with the graft)."""
    config = SOURCES[source]
    path = get_robot_path(source) / config["mjcf"]
    spec = mujoco.MjSpec.from_file(str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    name = lambda objtype, i: mujoco.mj_id2name(model, objtype, i) or ""
    arm_bodies = {link.body for link in harvest(source)} | {harvest_base(source).body}

    out = []
    for c in range(model.ncam):
        cam_name = name(mujoco.mjtObj.mjOBJ_CAMERA, c)
        body_id = int(model.cam_bodyid[c])
        body_name = name(mujoco.mjtObj.mjOBJ_BODY, body_id)
        parent = name(mujoco.mjtObj.mjOBJ_BODY, int(model.body_parentid[body_id]))
        # Keep cameras on an arm body, or on a DEDICATED mount hanging off one. A mount body
        # carries no geoms; a body that does belongs to another assembly -- franka_droid's
        # gripper/wrist_camera sits on gripper/base, and that one rides along with the graft, so
        # re-emitting it here would duplicate it.
        if body_name in arm_bodies:
            pass
        elif parent in arm_bodies and len(spec.body(body_name).geoms) == 0:
            pass
        else:
            continue
        # Camera pose is read from the COMPILED model, not the spec. MJCF allows orientation to
        # be given as xyaxes / zaxis / euler / axisangle, and the spec's ``quat`` then still reads
        # identity -- FR3's wrist_cam compiles to (0, .707, .707, 0) while authoring identity, so
        # trusting the authored value flipped the camera 180 degrees. (Note this is the OPPOSITE of
        # geom placement, where the compiled value carries MuJoCo's mesh recentering and the
        # authored value is the one to keep.)
        carrier = body_name if body_name in arm_bodies else parent
        mount = spec.body(body_name)
        out.append(MountedCameraSpec(
            name=cam_name, parent=carrier,
            body_pos=(0.0, 0.0, 0.0) if body_name in arm_bodies
            else tuple(float(v) for v in model.body_pos[body_id]),
            body_quat=(1.0, 0.0, 0.0, 0.0) if body_name in arm_bodies
            else tuple(float(v) for v in model.body_quat[body_id]),
            pos=tuple(float(v) for v in model.cam_pos[c]),
            quat=tuple(float(v) for v in model.cam_quat[c]),
            fovy=float(model.cam_fovy[c]),
            resolution=tuple(int(v) for v in spec.camera(cam_name).resolution),
        ))
        del mount
    return tuple(out)


@lru_cache(maxsize=8)
def harvest_sites(source: str) -> tuple:
    """Sites the source declares on its arm bodies -- FR3's ``attachment_site``."""
    config = SOURCES[source]
    path = get_robot_path(source) / config["mjcf"]
    spec = mujoco.MjSpec.from_file(str(path))
    model = mujoco.MjModel.from_xml_path(str(path))
    name = lambda objtype, i: mujoco.mj_id2name(model, objtype, i) or ""
    arm_bodies = {link.body for link in harvest(source)} | {harvest_base(source).body}
    out = []
    for i in range(model.nsite):
        site_name = name(mujoco.mjtObj.mjOBJ_SITE, i)
        body_name = name(mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[i]))
        if body_name not in arm_bodies:
            continue
        # compiled, for the same reason as cameras: a site may declare xyaxes rather than quat
        out.append(MountedSiteSpec(name=site_name, parent=body_name,
                                   pos=tuple(float(v) for v in model.site_pos[i]),
                                   quat=tuple(float(v) for v in model.site_quat[i]),
                                   group=int(spec.site(site_name).group)))
    return tuple(out)


def measured_camera_bounds() -> dict:
    """Min/max of every camera mount the repo declares -- MJCF-side here; the config-side
    RobotMountedCameraConfig values are asserted separately by the support test."""
    offsets, fovs = [], []
    for source in SOURCES:
        for camera in harvest_cameras(source):
            offsets.append(float(np.linalg.norm(np.array(camera.body_pos) + np.array(camera.pos))))
            fovs.append(camera.fovy)
    if not offsets:
        return dict(offset=(0.0, 0.0), fovy=(0.0, 0.0))
    return dict(offset=(min(offsets), max(offsets)), fovy=(min(fovs), max(fovs)))


def measured_material_bounds() -> dict:
    """Min/max of the real materials' appearance properties, so the SUPPORT bands can be generic
    ranges that CONTAIN them rather than their values inserted as literals."""
    fields = ("specular", "shininess", "reflectance", "emission")
    seen = {field: [] for field in fields}
    for source in SOURCES:
        for material in materials(source).values():
            for field in fields:
                seen[field].append(getattr(material, field))
    return {field: (min(values), max(values)) for field, values in seen.items()}


def measured_inertial_bounds() -> dict:
    """Min/max of the real links' mass, inertia components and centre-of-mass offsets.

    The SUPPORT bands for inertials are set from this, not the other way round: the point is
    that the real robots fall inside a *generic* band, not that their values were hardcoded."""
    mass, inertia, com = [], [], []
    for source in SOURCES:
        for link in harvest(source):
            mass.append(link.mass)
            inertia.extend(v for v in link.inertia if v > 0)
            com.append(float(np.linalg.norm(link.com)))
    return dict(mass=(min(mass), max(mass)), inertia=(min(inertia), max(inertia)),
                com=(0.0, max(com)))
