"""SEED -> PARAMETERS. The only source of randomness in generation.

A pure function of ``(seed, GENERATOR_VERSION)`` via numpy PCG64: no I/O, no MuJoCo, no
filesystem, so the whole space is testable in isolation and a seed means the same robot on every
machine. The output is an :class:`~.params.EmbodimentParams` and nothing else; :mod:`.builder`
turns that into a model without ever seeing a seed.

DISTRIBUTIONS lists every draw in one place -- what is sampled, from what, over what range. Read
it as the specification of the generated population; :data:`~.params.SUPPORT` is the wider box
that population must stay inside.

SUPPORT vs DENSITY
------------------
Support is necessary but not sufficient: a continuous parameter hits any particular value with
probability zero. So each distribution keeps *density* on plausible arms while keeping *support*
wide enough to contain the real ones. Every widening below was forced by a measurement of a
shipped robot, and every one widens a MECHANISM rather than special-casing a value -- no real
robot's numbers appear as literals anywhere in this file.

============================  ===============================================================
was excluded                  measurement that forced the widening
============================  ===============================================================
5-token discrete twists       4 of YAM's 6 body quats are not multiples of 90 degrees
lateral offset capped .05     FR3's structure is lateral: 0.316, 0.0825, 0.3928, 0.088 m
strictly positive axial rise  FR3 has z == 0 on six of its seven segments
symmetric-or-zero ranges      FR3 joint4 [-3.042, -0.152], joint6 [0.544, 4.517]
span in [1.9, 3.05]           YAM joint4 +-1.571 (narrow), joint2 [0, 3.665] (wide)
geometric gain schedule       kp_0 == kp_1 forces decay 1, forbidding FR3's kp_2 = 3500
force capped at 250 N-m       FR3 is unlimited, so any finite cap excludes it; 250 next to
                              kp=4461 saturates the servo at 0.056 rad (seed7)
armature tied to kp's decay   YAM's armature steps 17.8x over three joints
capsule-only link geometry    FR3's 28 arm geoms are ALL meshes, collision included
geom-derived inertia only     both robots ship explicit inertials (FR3 7, YAM 8)
hinge-only joints             a generic arm should not be forbidden a prismatic joint
zero gripper mount pose       real flanges mount their tool at an offset and a rotation
============================  ===============================================================
"""

from __future__ import annotations

import numpy as np

from .constants import ATTACHMENT_SITE, GENERATOR_VERSION, GRIPPERS
from .geom_pool import BASE_INDEX, harvest
from .geom_pool import SOURCES as GEOM_SOURCES
from .params import (
    ArmParams,
    SiteParams,
    DressingParams,
    EmbodimentParams,
    GeomParams,
    InertialParams,
    JointParams,
    LinkParams,
    SUPPORT,
    SegmentParams,
)
from .skeleton import TWISTS, axes_at_zero, axes_ok, quat_mul

SAMPLED_GRAMMARS = ("alternating", "ur", "mixed")
GRAMMAR_WEIGHTS = (0.40, 0.30, 0.30)
# "fallback" is never drawn directly -- it is the label a robot gets when rejection sampling gives
# up and _fallback_quats is used (~1.3% of seeds). It belongs in the declared outcomes even though
# it is not in the choice, which is why the two tuples are separate: folding it into the choice
# silently desynchronised the probability vector.
GRAMMARS = SAMPLED_GRAMMARS + ("fallback",)

# Every draw, in one table. Ranges are the DENSITY of the population; params.SUPPORT is the
# wider admissible box. Keep this in sync with the code below -- it is documentation, not config.
DISTRIBUTIONS = """
dof                    categorical   6:0.50  7:0.35  8:0.15
grammar                categorical   alternating:0.40  ur:0.30  mixed:0.30
segment.quat           POOL body: its own source's quat
                       otherwise: token (x) rotation; angle U(0,0.20) w.p. 0.75 else U(0,pi)
segment.offset         ONE rule: offset = shape * stretch, both from the body being placed. Shape
                       is that body's own source offset if pooled, else a budgeted sampled bone
                       (nominal/dof * U(0.55,1.45), x0.45 distal, near-axial at joint 1)
nominal length         U(0.60, 1.30) m, split across sampled bones by normalised weights
link.geom.stretch      arm-wide U(0.75, 1.35) * per-link logN(0, 0.06), clipped to SUPPORT
joint.kind             always hinge for ARM joints (gripper joints come from the graft)
joint.range span       hinge: N(3.4, 1.3) clipped to SUPPORT; slide: U(0.005, 0.25)
joint.range center     0 w.p. 0.50; +-span/2 w.p. 0.20; U over the admissible window w.p. 0.30
joint.kp               max(logU(30, 5000) * decay**i * logN(0, 0.25), gravity-holding minimum)
                       where the minimum is 0 if gravcomp is on, else 4*m_downstream*g*lever/0.05
joint.kv               kp * U(0.02, 0.20) * logN(0, 0.30)
joint.force            0 (unlimited, as FR3) w.p. 0.35, else kp * headroom,
                       headroom = logU(0.40, 2.00) * logN(0, 0.30) clipped to SUPPORT
                       (derived, not independent: force/kp in rad is where the servo saturates)
joint.armature         kp / ratio, ratio = logU(1200, 40000) * logN(0, 0.50) clipped to SUPPORT
                       (derived, not independent: the ratio governs numerical stability)
joint.damping          U(0, 1.5) * logN(0, 0.20)
joint.frictionloss     U(0, 0.2) * logN(0, 0.20)
link.geom.mode/source  sticky walk over {primitive} + geom_pool.SOURCES: p(switch)=0.22 per link,
                       initial and switched choice primitive:0.45 else uniform over sources
link.inertial.mode     explicit w.p. 0.50, else computed
link.inertial.mass     logU(0.05, 6.0) kg
link.inertial.inertia  logU(1e-7, 5e-2) per principal component
link.inertial.com      uniform direction * U(0, 0.15) m
link.inertial.density  U(200, 3000) kg/m3
link.inertial.gravcomp 1.0 w.p. 0.5, arm-wide (base always 0)
link.geom.material     inherit w.p. 0.55 for pool links (else palette); palette for primitive
base                   drawn by the same mechanism as any link (its pool index is BASE_INDEX)
cameras                count 0:0.15  1:0.50  2:0.35; wrist on the flange, second on the base
camera.body_pos        uniform direction * U(0, 0.25) flange / U(0.15, 0.95) base
camera.quat            rotation, angle U(0, pi) -- a camera may point anywhere
camera.fovy            U(40, 100) degrees
camera.resolution      one of (1280, 720), (640, 480), (624, 352)
sites                  an attachment_site on the flange w.p. 0.60 (FR3 has one, YAM has none)
dressing.specular      U(0, 1)          dressing.shininess     U(0, 1)
dressing.reflectance   U(0, 0.35)       dressing.emission      U(0, 0.05)
dressing.texture_kind  none:0.55  checker:0.20  gradient:0.15  flat:0.10
dressing.texture_rgb*  uniform per channel; repeat U(1, 8)
mount_pos              uniform direction * U(0, 0.06) m
mount_quat             rotation; angle U(0, 0.20) w.p. 0.60 else U(0, pi)
pedestal_height        U(0.55, 0.85) m
pedestal_radius        U(0.10, 0.18) m
gripper                uniform over constants.GRIPPERS
dressing.palette       uniform over PALETTES, each channel jittered by U(-0.04, 0.04)
dressing.radius        U(0.030, 0.050) m
dressing.taper         U(0.55, 0.90)
dressing.accent_period integers(2, 4)
"""

PALETTES = {
    "pearl": ((0.93, 0.93, 0.95), (0.95, 0.47, 0.10), (0.16, 0.16, 0.17)),
    "graphite": ((0.27, 0.27, 0.30), (0.10, 0.65, 0.75), (0.62, 0.62, 0.64)),
    "sand": ((0.82, 0.77, 0.68), (0.12, 0.35, 0.70), (0.30, 0.28, 0.26)),
    "arctic": ((0.97, 0.97, 0.98), (0.35, 0.62, 0.90), (0.45, 0.48, 0.52)),
    "safety": ((0.95, 0.75, 0.10), (0.10, 0.10, 0.10), (0.25, 0.25, 0.25)),
    "forest": ((0.16, 0.35, 0.24), (0.80, 0.66, 0.25), (0.24, 0.26, 0.24)),
}


def _twist_tokens(rng, grammar: str, n: int) -> list:
    """A quarter-turn token sequence: the *density seed* for the segment quaternions."""
    if grammar == "alternating":
        tokens = ["I"] + [("X-", "X+")[i % 2] for i in range(n - 1)]
        if rng.random() < 0.4:
            tokens[-2] = str(rng.choice(["Y+", "Y-"]))
        return tokens
    if grammar == "ur":
        tokens = ["I", str(rng.choice(["X+", "X-"])), "I", "I"]
        tokens += [str(rng.choice(["X+", "X-"])), str(rng.choice(["Y+", "Y-", "X+", "X-"]))]
        while len(tokens) < n:
            tokens.insert(3, "I")
        return tokens[:n]
    body = [str(rng.choice(["X+", "X-", "Y+", "Y-", "I"])) for _ in range(n - 3)]
    wrist = [str(rng.choice(["X+", "X-", "Y+", "Y-"])) for _ in range(2)]
    return ["I"] + body + wrist


def _fallback_quats(dof: int) -> list:
    """A twist sequence guaranteed to pass :func:`axes_ok`, used when rejection sampling gives up.

    Plain "alternating" does NOT qualify, contrary to what this module claimed for a while: tokens
    I, X-, X+, X-, ... yield axes z, y, z, y, ... -- only TWO distinct directions, so it fails the
    three-direction requirement. Only the optional Y-flip in the wrist made it pass, which meant
    the fallback was degenerate 60% of the time and the seed -> robot map was not actually total.
    Verified by test_fallback_quats_are_non_degenerate."""
    tokens = ["I"] + [("X-", "X+")[i % 2] for i in range(dof - 1)]
    tokens[-2] = "Y+"                                  # the third, non-coplanar direction
    return [TWISTS[token] for token in tokens]


def _random_rotation(rng, small_prob: float = 0.75) -> tuple:
    """A rotation, usually small. The full-angle branch is what makes the support all of SO(3)
    rather than a discrete set, and so what lets YAM's off-token wrist exist at all."""
    angle = float(rng.uniform(0.0, 0.20) if rng.random() < small_prob
                  else rng.uniform(0.0, np.pi))
    axis = rng.normal(size=3)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    half = angle / 2.0
    return (float(np.cos(half)), *(float(v) for v in np.sin(half) * axis))


def _sample_range(rng, kind: str = "hinge") -> tuple:
    """An arbitrary joint limit. Symmetric is typical, but offset ranges must be reachable:
    FR3's joint6 is [0.544, 4.517] and its joint4 is [-3.042, -0.152]."""
    if kind == "slide":
        lo_span, hi_span = SUPPORT["slide_span"]
        span = float(np.clip(rng.uniform(lo_span, 0.25), lo_span, hi_span))
        limit = hi_span
        return _window(rng, span, limit)
    lo_span, hi_span = SUPPORT["range_span"]
    limit = SUPPORT["range_abs"]
    span = float(np.clip(rng.normal(3.4, 1.3), lo_span, hi_span))
    return _window(rng, span, limit)


def _window(rng, span: float, limit: float) -> tuple:
    """Place a window of width ``span`` inside +-``limit``."""
    draw = rng.random()
    if draw < 0.50:                                    # symmetric
        center = 0.0
    elif draw < 0.70:                                  # anchored at zero
        center = span / 2.0 if rng.random() < 0.5 else -span / 2.0
    else:                                              # offset, FR3-style
        center = float(rng.uniform(-limit + span / 2, limit - span / 2))
    lo, hi = center - span / 2, center + span / 2
    # SHIFT the window to fit inside +-limit rather than clipping its ends: clipping silently
    # shrinks the sampled span (and can push it under the declared minimum), whereas the span is
    # the quantity we actually drew. span <= 2 * limit, so a shift always fits.
    if hi > limit:
        lo, hi = lo - (hi - limit), limit
    if lo < -limit:
        lo, hi = -limit, hi + (-limit - lo)
    return (float(lo), float(hi))


def _sample_joints(rng, dof: int, links=None, nominal: float = 0.9) -> tuple:
    """Per-joint dynamics.

    Each joint's values are drawn INDEPENDENTLY around a softening schedule, rather than being
    a deterministic function of it. That distinction is the whole ballgame for support: with
    ``kp_i = kp_root * decay**i`` no seed can produce FR3's [4500, 4500, 3500, 3500, 2000, 2000,
    2000] -- kp_0 == kp_1 forces decay == 1, which then forbids kp_2 -- nor YAM's
    [40, 40, 40, 20, 10, 10]. The lognormal jitter has support on (0, inf), so composed with the
    SUPPORT clamps every admissible per-joint vector is reachable, while density stays on
    schedules that soften distally the way real arms do.

    kv, force and armature are jittered per joint for the same reason: YAM's kv/kp ratio is not
    constant down the chain ([.0625, .0625, .0625, .025, .1, .1]) and its armature is a two-tier
    step, neither of which a single global ratio can express.
    """
    kp_root = float(np.exp(rng.uniform(np.log(30.0), np.log(5000.0))))
    decay = float(rng.uniform(0.72, 1.0))
    kv_ratio = float(rng.uniform(0.02, 0.20))
    # Force is DERIVED from kp through a sampled saturation headroom (force/kp, in rad), for the same
    # reason armature is derived below: the ratio is the physically meaningful quantity. Independent
    # draws are what put kp=4461 next to force=250, saturating the actuator at 0.056 rad of error --
    # past which it is a constant-torque source, not a position servo. YAM's headroom is 0.50-1.00
    # rad, so the earlier "0.7 needs a 9-sigma jitter" note was measuring against the wrong scale:
    # in units of rad, 0.7 is mid-band. FR3 leaves its arm unlimited (force 0), so zero stays a real
    # outcome rather than a floor.
    unlimited = rng.random() < 0.35
    headroom_root = float(np.exp(rng.uniform(np.log(0.40), np.log(2.00))))
    # Armature is DERIVED from kp through a sampled stiffness-to-inertia ratio, not drawn
    # independently: the ratio is what governs numerical stability, and independent draws produced
    # joints 1000x stiffer per unit inertia than any real one, which blows up the integrator.
    # Jittered per joint because YAM's ratios are not constant down its chain (1250 .. 11111).
    ratio_lo, ratio_hi = SUPPORT["stiffness_ratio"]
    ratio_root = float(np.exp(rng.uniform(np.log(1200.0), np.log(40000.0))))
    damping_root = float(rng.uniform(0.0, 1.5))
    friction_root = float(rng.uniform(0.0, 0.2))
    jitter = lambda sigma: float(np.exp(rng.normal(0.0, sigma)))

    # A joint must be stiff enough to HOLD THE ARM BELOW IT, unless gravity is compensated. kp and
    # link mass were drawn independently, so a 20 kg arm could get kp=50 and sag 0.9 rad. The real
    # robots satisfy this two different ways: FR3 carries 15.5 kg on kp=4500 with no compensation,
    # YAM carries 3.4 kg on kp=40 WITH it.
    masses = [l.inertial.mass for l in links] if links else [1.0] * dof
    compensated = bool(links) and float(links[0].inertial.gravcomp) > 0.5
    lever = 0.5 * nominal

    joints = []
    for index in range(dof):
        scale = decay ** index
        downstream = float(sum(masses[index:]))
        # torque to hold it, converted to the stiffness that yields <= 0.05 rad of sag
        needed = 0.0 if compensated else 4.0 * downstream * 9.81 * lever / 0.05
        # ARM joints are revolute only. A prismatic arm joint translates its child, so the child's
        # origin is not fixed in the parent's frame and NO static geometry can span the changing
        # distance -- the link genuinely separates as it extends. That also breaks the guarantee the
        # connector struts rest on (see builder.build). Grippers are unaffected: they arrive whole
        # from the graft, and yam_jaw's fingers are prismatic.
        kind = "hinge"
        kp = float(np.clip(max(kp_root * scale * jitter(0.25), needed), *SUPPORT["kp"]))
        joints.append(JointParams(
            kind=kind,
            range=_sample_range(rng, kind),
            kp=kp,
            kv=float(np.clip(kp * kv_ratio * jitter(0.30), *SUPPORT["kv"])),
            force=0.0 if unlimited else float(np.clip(
                kp * float(np.clip(headroom_root * jitter(0.30), *SUPPORT["saturation_headroom"])),
                *SUPPORT["force"])),
            armature=float(np.clip(
                kp / float(np.clip(ratio_root * jitter(0.50), ratio_lo, ratio_hi)),
                *SUPPORT["armature"])),
            damping=float(np.clip(damping_root * jitter(0.20), *SUPPORT["damping"])),
            frictionloss=float(np.clip(friction_root * jitter(0.20), *SUPPORT["frictionloss"])),
        ))
    return tuple(joints)


def _box_inertia(rng, mass: float) -> tuple:
    """Principal moments of a uniform box: ``m/12 * (b^2+c^2, a^2+c^2, a^2+b^2)``.

    Guarantees the triangle inequality every valid inertia tensor must satisfy, and loses no
    generality: any admissible (I1, I2, I3) inverts to some half-extent triple."""
    extents = np.exp(rng.uniform(np.log(0.01), np.log(0.30), size=3))
    a, b, c = (float(v) for v in extents)
    raw = (mass / 12.0 * (b * b + c * c),
           mass / 12.0 * (a * a + c * c),
           mass / 12.0 * (a * a + b * b))
    return tuple(float(np.clip(v, *SUPPORT["inertia"])) for v in raw)


def _sample_links(rng, dof: int) -> tuple:
    """Per-link geometry, inertia and size.

    Geometry source is chosen per link by a STICKY walk down the chain: usually the same choice as
    the previous link, occasionally switching to another source robot or to primitives. Runs of real
    geometry therefore still emerge -- a real arm is mostly one robot -- but an arm may now mix
    franka_droid and i2rt_yam links, which a single run per arm made impossible.

    Mixing is safe because contact does not depend on the two sides agreeing. Every joint is
    revolute and anchored at its child's origin, so that origin is invariant in the parent's frame;
    the connector struts ``builder.build`` adds have an endpoint there on BOTH bodies, so they
    always intersect. The guarantee is geometric, not statistical, and holds at any source boundary
    in any pose.
    """
    options = [None] + sorted(GEOM_SOURCES)          # None means sampled primitives
    weights = [0.45] + [0.55 / len(GEOM_SOURCES)] * len(GEOM_SOURCES)
    current = options[int(rng.choice(len(options), p=weights))]

    arm_stretch = float(rng.uniform(0.75, 1.35))
    # One size per SOURCE: adjacent links of the same robot must scale together or the source's own
    # fit breaks mid-run. A different source is a different part, so it may have its own size.
    sizes = {name: float(np.clip(arm_stretch * float(np.exp(rng.normal(0.0, 0.05))),
                                 *SUPPORT["stretch"])) for name in options if name}
    material_mode = "inherit" if rng.random() < 0.55 else "palette"
    # Gravity compensation is an arm-wide control choice: on for the whole arm or off. YAM ships it
    # on (links 2-6), FR3 off, so both must be reachable.
    gravcomp = 1.0 if rng.random() < 0.5 else 0.0

    links = []
    for index in range(-1, dof):        # index -1 is the base: same mechanism, its own pool entry
        if index > -1 and rng.random() < 0.22:
            current = options[int(rng.choice(len(options), p=weights))]
        if current is None:
            stretch = float(np.clip(arm_stretch * float(np.exp(rng.normal(0.0, 0.06))),
                                    *SUPPORT["stretch"]))
            geom = GeomParams(mode="primitive", source=None, link_index=None,
                              material_mode="palette", stretch=stretch)
        else:
            n_links = GEOM_SOURCES[current]["n_links"]
            geom = GeomParams(
                mode="pool", source=current,
                link_index=BASE_INDEX if index == BASE_INDEX else _along(index, dof, n_links),
                material_mode=material_mode, stretch=sizes[current])

        direction = rng.normal(size=3)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        com_limit = SUPPORT["com_offset"][1]
        mass = float(np.clip(np.exp(rng.uniform(np.log(0.05), np.log(6.0))), *SUPPORT["mass"]))
        links.append(LinkParams(geom=geom, inertial=InertialParams(
            mode="explicit" if rng.random() < 0.50 else "computed",
            density=float(rng.uniform(*SUPPORT["density"])),
            mass=mass,
            inertia=_box_inertia(rng, mass),
            com=tuple(float(np.clip(v, -com_limit, com_limit))
                      for v in direction * rng.uniform(0.0, 0.15)),
            gravcomp=(0.0 if index == BASE_INDEX else gravcomp),
        )))
    return tuple(links[1:]), links[0]      # (arm links, base link)


def _sample_cameras(rng) -> tuple:
    """No cameras. A generated arm declares none of its own.

    This used to draw a mount pose -- count 0/1/2, a uniform direction up to 0.25 m out, a full
    random rotation, fovy U(40,100) -- treating where a camera is bolted on as a random variable.
    The shipped robots do not work that way. Franka's wrist camera has a DESIGNED pose baked into
    its gripper MJCF and its shoulder camera a designed offset in the camera config, each then
    jittered by a few millimetres and degrees to model mounting tolerance; only the exocentric
    cameras are placed at random. So camera placement belongs to the camera system, which is where
    upstream keeps it, and a generated arm gets its uniform ``wrist_camera`` at emission time from
    the pose the gripper catalog measured (see builder._add_wrist_camera).

    Kept as a function returning empty rather than deleted, so ArmParams.cameras stays in the
    contract: reference.py still harvests the real robots' cameras through it, which is what keeps
    the witnesses byte-exact.
    """
    del rng
    return ()


def _sample_sites(rng, mount_pos, mount_quat) -> tuple:
    """Named frames the robot declares. The tool plate site is optional because it is optional in
    reality: FR3 declares ``attachment_site``, YAM declares nothing on its arm."""
    if rng.random() >= 0.60:
        return ()
    return (SiteParams(name=ATTACHMENT_SITE, mount="flange", pos=tuple(mount_pos),
                       quat=tuple(mount_quat), group=4),)


def _along(index: int, dof: int, n_links: int) -> int:
    """Which source link belongs at arm position ``index``: the one at the same RELATIVE depth.

    Wrapping with ``index % n_links`` was the previous rule and it is what made arms fold into
    themselves. Real robots taper -- a shoulder casting is bulky and a wrist is slim (FR3's link
    bounding radii run 0.25 down to 0.17, YAM's 0.32 down to 0.10) -- and modular arithmetic put
    proximal castings at distal positions, where the sampled segments are shortest. A shoulder-sized
    chunk on a 38 mm wrist segment overlaps everything near it, which is exactly the
    ``gripper/base~link1`` and ``base_link~link4`` interpenetration measured at 50-72 mm.

    A monotone map preserves the taper for any (dof, n_links) pair, and reduces to the identity when
    they are equal -- which is the case for both witnesses, so their parameters are unchanged."""
    if n_links <= 1 or dof <= 1:
        return 0
    return int(round(index * (n_links - 1) / (dof - 1)))


def _forward_reach(link) -> float:
    """How far the source expected THIS body's own child to sit, in the source's own metres.

    The counterpart to :func:`_own_shape`. A body's geometry is designed to span the gap to its
    child, so a borrowed link needs at least the room its source gave it. Nothing used to ask for
    that -- each body claimed only its own BACKWARD offset, and _own_shape's docstring conceded that
    "the parent's forward reach may now fall short of its child", to be patched afterwards with
    struts. Patching it was the wrong trade: the struts are volume, and volume is what made arms
    intersect themselves (a gap-sized bridging SPHERE at the wrist is a 100 mm ball for a 50 mm gap).
    Asking for the room up front means there is no gap to fill."""
    if link is None or link.geom.mode != "pool" or link.geom.link_index == BASE_INDEX:
        return 0.0
    entry = harvest(link.geom.source)
    nxt = link.geom.link_index + 1
    if nxt >= len(entry):
        return 0.0
    return float(np.linalg.norm(entry[nxt].offset)) * link.geom.stretch


def _own_shape(link):
    """Where the source robot puts THIS body relative to its parent, or None if the body wears no
    real geometry.

    Each body follows its OWN source, not its parent's. That is the direction that matters to it:
    real link geometry straddles its own joint and reaches BACK toward its parent by exactly this
    offset, so taking the offset from the body's own source makes that backward reach fit exactly.

    Driving it from the PARENT's source instead handed a primitive child whatever spacing the
    borrowed parent expected -- FR3's link4 expects its child 0.384 m out laterally -- and the
    primitive child then spanned it with a single slim capsule: a 0.49 m rod jutting sideways out of
    chunky borrowed links, which read as an extra limb.

    The parent's forward reach may now fall short of its child. The connector struts close that, and
    their guarantee does not depend on the two sides agreeing."""
    if link is None or link.geom.mode != "pool" or link.geom.link_index == BASE_INDEX:
        return None
    entry = harvest(link.geom.source)[link.geom.link_index]
    return np.asarray(entry.offset, float), entry.quat


def _sample_segments(rng, dof: int, links, base, nominal: float) -> tuple:
    """Segment poses, from ONE rule applied to every body in the chain.

    ``offset = shape * stretch`` with both taken from THE BODY BEING PLACED: its own source's offset
    if it wears real geometry, a budgeted sampled bone otherwise. Because the shape and the geometry
    are resized by the same factor, a borrowed link's backward reach fits its own bone exactly -- no
    post-hoc rescale, no exemption from it, and no lookahead.

    ``nominal`` is normalised over EVERY segment, pooled ones included, not just the sampled ones.
    Dividing it among the sampled bones alone makes each one longer the fewer there are: an arm
    whose run covers all but one link gave that single bone the whole budget -- 0.64 m, four times
    its neighbours -- and the pooled link before it could not reach that far, which is precisely the
    detached-limb failure. Normalising over all segments keeps a sampled bone the size of a typical
    link regardless of how much of the chain is borrowed, while a fully-primitive arm still sums to
    ``nominal`` and so stays DoF-independent.
    """
    # Each segment places its OWN body, so shape and stretch both come from that body.
    del base
    shapes = [_own_shape(links[index]) for index in range(dof)]

    weights = []
    for index in range(dof):
        distal = index >= dof - 2
        weight = float(rng.uniform(0.55, 1.45)) * (0.45 if distal else 1.0)
        # a pooled segment still occupies a share of the chain, so it counts in the denominator
        weights.append(weight)
    total = sum(weights)

    segments = []
    for index in range(dof):
        stretch = links[index].geom.stretch
        shape = shapes[index]
        # The room the PARENT's source expected to its own child. Taking the max means both sides
        # fit: the child's backward reach and the parent's forward reach. For a single-source arm the
        # two are the same number, so a witness is untouched and stays byte-exact.
        parent_reach = _forward_reach(links[index - 1]) if index > 0 else 0.0
        if shape is not None:
            offset = np.asarray(shape[0], float) * stretch
            length = float(np.linalg.norm(offset))
            if parent_reach > length > 1e-9:
                offset = offset * (parent_reach / length)
            elif parent_reach > 1e-9 and length <= 1e-9:
                # a coincident body (FR3 has two) still has to clear a bulky parent
                offset = np.array([0.0, 0.0, parent_reach])
            segments.append((offset, shape[1]))
            continue
        length = nominal * weights[index] / total if total > 1e-9 else nominal / dof
        length = max(length, parent_reach / max(stretch, 1e-9))
        direction = rng.normal(size=3)
        direction[2] = abs(direction[2]) * float(rng.uniform(0.6, 1.8))
        if index == 0:
            # joint 1 sits on the mounting flange: a lateral offset here leaves the arm hanging
            # clear of its pedestal, and both reference arms measure exactly 0.0
            direction[:2] *= 0.12
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        segments.append((direction * length * stretch, None))
    return tuple(segments), SUPPORT["offset_abs"]


def sample_params(seed: int) -> EmbodimentParams:
    """The robot for ``seed``. Same seed + same GENERATOR_VERSION == same robot, anywhere."""
    rng = np.random.Generator(np.random.PCG64(seed))

    dof = int(rng.choice([6, 7, 8], p=[0.50, 0.35, 0.15]))
    grammar = str(rng.choice(SAMPLED_GRAMMARS, p=list(GRAMMAR_WEIGHTS)))

    # Rejection-sample the perturbed quaternions against the degeneracy screen. "alternating"
    # with no perturbation is degeneracy-free by construction, so it terminates the loop and
    # the seed -> robot map stays total.
    # Links are drawn FIRST: a link wearing real geometry constrains the segment pose it needs,
    # so the geometry decision has to precede the kinematics decision.
    arm_links, base_link = _sample_links(rng, dof)

    # Budgeted nominal length for sampled bones; pooled bodies use their source's proportions.
    nominal = float(rng.uniform(0.60, 1.30))
    shapes = None
    for _ in range(32):
        candidate, _limit = _sample_segments(rng, dof, arm_links, base_link, nominal)
        tokens = _twist_tokens(rng, grammar, dof)
        quats = [q if q is not None else quat_mul(TWISTS[token], _random_rotation(rng))
                 for (_o, q), token in zip(candidate, tokens)]
        if axes_ok(axes_at_zero(quats)):
            shapes = candidate
            break
    if shapes is None:
        grammar = "fallback"
        quats = _fallback_quats(dof)
        shapes, _limit = _sample_segments(rng, dof, arm_links, base_link, nominal)
    limit = SUPPORT["offset_abs"]
    segments = tuple(SegmentParams(tuple(float(v) for v in q),
                                   tuple(float(np.clip(v, -limit, limit)) for v in o))
                     for (o, _q), q in zip(shapes, quats))

    mount_direction = rng.normal(size=3)
    mount_direction /= max(float(np.linalg.norm(mount_direction)), 1e-12)
    arm = ArmParams(
        grammar=grammar,
        segments=segments,
        joints=_sample_joints(rng, dof, arm_links, nominal),
        links=arm_links,
        base=base_link,
        cameras=_sample_cameras(rng),
        sites=_sample_sites(rng, mount_direction * 0.0, (1.0, 0.0, 0.0, 0.0)),
        mount_pos=tuple(float(v) for v in mount_direction * rng.uniform(0.0, 0.06)),
        mount_quat=_random_rotation(rng, small_prob=0.6),
        pedestal_height=float(rng.uniform(0.55, 0.85)),
        pedestal_radius=float(rng.uniform(0.10, 0.18)),
    )

    base, accent, joint = PALETTES[str(rng.choice(sorted(PALETTES)))]
    jitter = lambda rgb: [float(np.clip(c + rng.uniform(-0.04, 0.04), 0.0, 1.0)) for c in rgb] + [1.0]
    texture_kind = str(rng.choice(["none", "checker", "gradient", "flat"],
                                  p=[0.55, 0.20, 0.15, 0.10]))
    dressing = DressingParams(
        palette={"base": jitter(base), "accent": jitter(accent), "joint": jitter(joint)},
        radius=float(rng.uniform(0.030, 0.050)),
        taper=float(rng.uniform(0.55, 0.90)),
        accent_period=int(rng.integers(2, 4)),
        specular=float(rng.uniform(*SUPPORT["specular"])),
        shininess=float(rng.uniform(*SUPPORT["shininess"])),
        reflectance=float(rng.uniform(0.0, 0.35)),
        emission=float(rng.uniform(0.0, 0.05)),
        texture_kind=texture_kind,
        texture_rgb1=tuple(float(v) for v in rng.uniform(0.05, 0.95, size=3)),
        texture_rgb2=tuple(float(v) for v in rng.uniform(0.05, 0.95, size=3)),
        texture_repeat=float(rng.uniform(1.0, 8.0)),
    )

    return EmbodimentParams(
        seed=int(seed),
        version=GENERATOR_VERSION,
        arm=arm,
        # Sampled from an explicit tuple, NOT from the catalog directory: adding a gripper must
        # be a deliberate GENERATOR_VERSION bump, never a silent reshuffle of every seed.
        gripper=str(rng.choice(GRIPPERS)),
        dressing=dressing,
    )
