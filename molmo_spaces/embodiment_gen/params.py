"""The PARAMETER CONTRACT: the complete description of one generated robot.

This module is the interface between the two halves of generation, and it deliberately contains
no randomness and no MuJoCo:

    seed  --[ sampling.py ]-->  EmbodimentParams  --[ builder.py ]-->  model.xml

Everything a robot IS lives in these dataclasses. :mod:`.sampling` decides *which* values a seed
draws; :mod:`.builder` decides *how* those values become a model. Neither knows the other, so a
hand-written EmbodimentParams builds exactly like a sampled one -- which is what makes it
meaningful to ask whether a real robot is in the distribution, and to answer by exhibiting its
parameters (see :mod:`.reference`).

:data:`SUPPORT` declares the closure of the admissible space. Every shipped MolmoSpaces arm must
lie inside it, else no seed could ever produce that robot however many you draw. The bands are
generic ranges chosen to CONTAIN the measured robots, never their values inserted as literals;
``test_embodiment_gen_support.py`` asserts the containment against the shipped MJCFs directly.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

GEOM_MODES = ("primitive", "pool")
MATERIAL_MODES = ("inherit", "palette")
TEXTURE_KINDS = ("none", "checker", "gradient", "flat")
INERTIAL_MODES = ("computed", "explicit")
JOINT_KINDS = ("hinge", "slide")
CAMERA_MOUNTS = ("flange", "base")

# The declared closure of the sampled space. Widen it only with a GENERATOR_VERSION bump.
SUPPORT = dict(
    dof=(6, 8),
    offset_abs=0.45,          # per-axis |body offset|; FR3's largest is 0.393
    # summed offset norms; YAM 0.762, FR3 1.212. The upper bound is above both because an arm
    # whose links wear real geometry keeps its NATURAL length rather than being rescaled to a
    # budget -- rescaling would undo the spacing that rigid link meshes need -- and a chain mixing
    # links from two source robots can sum longer than either.
    # The lower bound is below both real arms for the same reason: a pool-majority arm keeps its
    # natural length, and one drawing repeatedly from YAM's short distal links sums well under
    # either robot's total.
    # Wide because length is now a CONSEQUENCE rather than a budget: a pooled body contributes its
    # source's proportions and every body may be resized by `stretch`, so the two multiply.
    chain_length=(0.15, 3.20),
    range_span=(0.40, 6.30),  # hinge, rad; YAM joint4 3.14, YAM joint2 3.67, joint1 5.67
    range_abs=4.70,           # |lo| and |hi|; FR3 joint6 hi = 4.517
    slide_span=(0.005, 0.60),   # prismatic, m
    kp=(1.0, 6000.0),         # FR3 4500, YAM 40
    kv=(0.05, 600.0),         # FR3 450, YAM 2.5
    # 0 means unlimited, which is what FR3 does. The cap is deliberately far above any real torque:
    # an unlimited actuator is the MORE extreme case, so every finite value is interior to what the
    # shipped robots exhibit, and a tight cap is what broke seed7 (see saturation_headroom).
    force=(0.0, 20000.0),     # YAM 28; FR3 leaves the arm unlimited (0)
    armature=(0.0002, 0.25),  # YAM 0.0018 .. FR3 0.1; floor positive so kp/armature is bounded
    # Stiffness-to-inertia ratio, the quantity that decides whether a joint is SIMULABLE at the
    # 2 ms timestep both shipped robots use. Measured: FR3 20000-45000, YAM 1250-11111. Sampling kp
    # and armature independently put 46% of joints above the real maximum and 6.6% above 1e6, up to
    # 4.06e7 -- which MuJoCo reports as "Nan, Inf or huge value in QACC".
    stiffness_ratio=(800.0, 60000.0),
    # force / kp, in RAD: the tracking error at which the actuator saturates and stops being a
    # position servo at all. Measured: YAM 0.50 (kp 20, force 10) .. 1.00 (kp 10, force 10); FR3 is
    # unlimited, i.e. infinite headroom. Drawing force independently of kp gave seed7 kp=4461 with
    # force=250 -- saturation at 0.056 rad -- and joints 2-4 pinned at the limit 98% of the time,
    # permanently 1.07 rad from their target. seed3 (0.017 rad) never moved at all.
    saturation_headroom=(0.30, 4.00),
    damping=(0.0, 2.50),      # YAM 0, FR3 1.0
    frictionloss=(0.0, 0.50), # FR3 0, YAM 0.1
    # Inertial bands are generic ranges that CONTAIN the measured links, taken from
    # geom_pool.measured_inertial_bounds(): mass 0.124..2.936 kg, inertia components ~3e-7..0.030,
    # |com| up to 0.134 m, implied densities ~334..1729 kg/m3.
    mass=(0.02, 8.00),
    inertia=(0.0, 0.15),
    com_offset=(0.0, 0.25),
    density=(200.0, 3000.0),
    mount_offset=(0.0, 0.20),
    pedestal_height=(0.0, 1.10),
    # Camera mounts. Bands contain every mount the repo declares: FR3's MJCF wrist_cam
    # (|offset| 0.118, fovy 90) and gripper/wrist_camera (fovy 56.74), the DROID shoulder camera's
    # config-side mount on fr3_link0 (offset [0.1, 0.57, 0.66], |.| 0.876, fov 71), and the
    # and the widest randomized exocentric fov_range in the repo, (137, 140) on
    # FrankaRandomizedDroidCameraSystem's exo_camera_3 -- a band of (20, 130) excluded it.
    # 1.56 m is BimanualYamCameraSystem's exo_camera, mounted high above its reference body --
    # the widest mount in the repo, and a YAM one, so a band of 1.10 excluded YAM's own setup.
    camera_offset=(0.0, 2.00),
    camera_fovy=(15.0, 150.0),
    camera_count=(0, 3),
    # How far a borrowed link may be resized. Contains 1.0 in the interior, so a robot reproducing
    # a source asset at native size is an ordinary sample rather than a boundary case.
    stretch=(0.50, 2.00),
    # Appearance bands, generic ranges containing geom_pool.measured_material_bounds()
    # (both robots: specular 0.5, shininess 0.5, reflectance 0, emission 0).
    specular=(0.0, 1.0),
    shininess=(0.0, 1.0),
    reflectance=(0.0, 0.6),
    emission=(0.0, 0.5),
)

@dataclass(frozen=True)
class SegmentParams:
    """Joint ``i``'s body pose relative to its parent: a full quaternion and a full offset."""

    quat: tuple                   # wxyz, relative to the parent body
    offset: tuple                 # 3-vector, relative to the parent body


@dataclass(frozen=True)
class JointParams:
    """Per-joint kind, limits and dynamics. Per joint, not per arm: both real robots schedule
    these down the chain, spanning two decades of stiffness from shoulder to wrist."""

    kind: str                     # "hinge" (range in rad) | "slide" (range in m)
    range: tuple                  # (lo, hi), arbitrary -- may be offset from zero
    kp: float
    kv: float
    force: float                  # symmetric force range; 0 means unlimited, as FR3 leaves it
    armature: float
    damping: float
    frictionloss: float


@dataclass(frozen=True)
class GeomParams:
    """How a link is shaped: sampled primitives, or a real link's complete geom set.

    "pool" is not a visual skin -- it carries the source link's COLLISION geoms and contact flags
    too. That is required rather than cosmetic: FR3's collision geometry is mesh-based (all 28 of
    its arm geoms are meshes), so a capsule-only collision model could never be an FR3 however
    its visuals were dressed. YAM, by contrast, pairs visual meshes with capsule collision, so
    both idioms have to be representable."""

    mode: str                     # "primitive" | "pool"
    source: str | None            # a geom_pool source key when mode == "pool"
    link_index: int | None
    # Isotropic size of this link. Borrowed geometry is RIGID, so a link wearing a real mesh can
    # only fit a bone the mesh spans -- unless the mesh is resized with it. Scaling the asset makes
    # a real link a size class rather than a fixed part, which is what removes the need for any
    # special case pinning bone lengths to source values. Isotropic because a mesh's scale is in
    # ITS axes while links are arbitrarily rotated: anisotropic scaling would shear the geometry.
    stretch: float = 1.0
    # Appearance is a SEPARATE axis from shape, because the useful combinations are all four:
    # real shape with real material (a robot), real shape with sampled colour (visual DR), and
    # the primitive equivalents. "inherit" keeps each harvested geom's own material -- the only
    # route to FR3's white/black/button set -- and degrades to the palette for primitive links,
    # which have no material to inherit.
    material_mode: str = "palette"


@dataclass(frozen=True)
class InertialParams:
    """Where a link's mass properties come from.

    "computed" lets MuJoCo derive them from the geoms and a sampled density. "explicit" emits an
    <inertial> element, which is the only way to reach what the real robots ship: both declare
    theirs outright (FR3 7 elements, YAM 8), and a hand-specified inertia need not equal anything
    a density times a geom would produce."""

    mode: str                     # "computed" | "explicit"
    density: float                # used when mode == "computed"
    mass: float                   # used when mode == "explicit"
    inertia: tuple                # principal diagonal
    com: tuple                    # centre of mass in the link frame
    # MuJoCo body gravcomp: 1.0 cancels gravity on this link. A control choice, not a physical one,
    # and the shipped robots disagree about it -- YAM enables it on links 2-6, FR3 on none -- so it
    # has to be a parameter. Without it a generated arm with low gains cannot hold itself up: YAM's
    # witness sagged 0.081 rad against the real robot's 0.0002.
    gravcomp: float = 0.0


@dataclass(frozen=True)
class CameraParams:
    """A camera the robot declares in its own MJCF.

    Camera placement belongs to the EMBODIMENT, not the experiment: MolmoSpaces wrist cameras are
    all ``MjcfCameraConfig``, which names a camera and takes its pose from the model, so a robot
    that declares none simply has no wrist camera and any config referencing one fails to resolve.
    ``docs/embodiment_spec.md`` is explicit that DROID's transforms may not be reused -- "A YAM
    wrist or shoulder camera requires its own declared body, offset, orientation, and validation"
    -- so every generated robot has to carry its own.

    Emitted as a dedicated zero-geom mount body plus a camera inside it, which is exactly how FR3
    carries ``wrist_cam`` on ``wrist_cam_body``."""

    name: str
    mount: str                    # "flange" (the tool plate) | "base" (shoulder-camera style)
    body_pos: tuple               # mount body pose relative to the mount point
    body_quat: tuple
    pos: tuple                    # camera pose within the mount body
    quat: tuple
    fovy: float
    resolution: tuple


@dataclass(frozen=True)
class SiteParams:
    """A named frame the robot declares. Sites are what specs and camera configs reference by
    name, so which ones exist is part of the embodiment: FR3 declares ``attachment_site`` on its
    tool plate and YAM declares none on its arm, so hardcoding one put a site on YAM that the real
    robot does not have."""

    name: str
    mount: str                    # "flange" | "base"
    pos: tuple
    quat: tuple
    group: int = 4


@dataclass(frozen=True)
class LinkParams:
    geom: GeomParams
    inertial: InertialParams


@dataclass(frozen=True)
class ArmParams:
    grammar: str
    segments: tuple               # SegmentParams, proximal -> distal
    joints: tuple                 # JointParams, one per segment
    links: tuple                  # LinkParams, one per segment
    mount_pos: tuple              # flange -> gripper graft offset
    mount_quat: tuple             # flange -> gripper graft rotation (wxyz)
    pedestal_height: float        # the scene-side platform the arm is attached on top of
    pedestal_radius: float
    # The mounting structure joint 1 hangs off. A link like any other -- it just has no joint, so
    # it needs its own slot, or it is silently invisible and massless (FR3 lost 8 geoms and
    # 2.82 kg that way; YAM lost its motor housing and mounting plate).
    base: "LinkParams | None" = None
    cameras: tuple = ()           # CameraParams the robot declares in its MJCF
    sites: tuple = ()             # SiteParams the robot declares on its own bodies

    @property
    def dof(self) -> int:
        return len(self.segments)


@dataclass(frozen=True)
class DressingParams:
    """Colours and primitive proportions. Consulted only by links whose geom mode is
    "primitive" -- a pool link brings its own materials from the source model."""

    palette: dict                 # role -> [r, g, b, a]
    radius: float                 # proximal bone radius; tapers distally
    taper: float                  # distal/proximal radius ratio
    accent_period: int
    # Appearance properties of the palette materials. Sampled, not fixed, so that the real
    # robots' values (specular 0.5, shininess 0.5) are ordinary draws rather than a special case.
    specular: float = 0.5
    shininess: float = 0.5
    reflectance: float = 0.0
    emission: float = 0.0
    # A procedural texture for palette materials. MuJoCo builtins, so no asset is required and
    # the axis is exercised even though neither reference robot declares a texture.
    texture_kind: str = "none"    # "none" | "checker" | "gradient" | "flat"
    texture_rgb1: tuple = (0.3, 0.3, 0.35)
    texture_rgb2: tuple = (0.6, 0.6, 0.65)
    texture_repeat: float = 3.0


@dataclass(frozen=True)
class EmbodimentParams:
    seed: int
    version: int
    arm: ArmParams
    gripper: str
    dressing: DressingParams

    @property
    def embodiment_id(self) -> str:
        return f"gen_{self.seed}"

    def to_dict(self) -> dict:
        return asdict(self)


