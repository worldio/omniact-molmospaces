"""Are the real robots in the support of the generated distribution?

Not "does some seed happen to produce a Franka" -- a continuous parameter hits any particular
value with probability zero. The testable and meaningful property is that no real robot is
*excluded*: every quantity MolmoSpaces' shipped arms actually use lies inside the closure of
the sampled space, so with unboundedly many seeds one can come arbitrarily close.

Each measurement is read straight out of the shipped MJCF, so this test fails if an upstream
asset changes into something the generator could no longer contain.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from molmo_spaces.embodiment_gen.geom_pool import (
    SOURCES as GEOM_SOURCES,
    harvest,
    measured_inertial_bounds,
)
from molmo_spaces.embodiment_gen.params import SUPPORT
from molmo_spaces.embodiment_gen.sampling import sample_params
from molmo_spaces.embodiment_gen.skeleton import chain_length, quat_to_mat
from molmo_spaces.molmo_spaces_constants import get_robot_path

# (label, resource_key, mjcf, substrings marking NON-arm joints)
REAL_ARMS = [
    ("franka_droid", "franka_droid", "model.xml", ("finger", "driver", "spring", "follower")),
    ("i2rt_yam", "i2rt_yam", "yam.xml", ("finger",)),
]


def measure(resource_key: str, mjcf: str, exclude) -> dict:
    """Pull the arm's kinematics and dynamics out of a shipped MJCF."""
    model = mujoco.MjModel.from_xml_path(str(get_robot_path(resource_key) / mjcf))
    name = lambda objtype, i: mujoco.mj_id2name(model, objtype, i) or ""

    joints = [j for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
              and not any(s in name(mujoco.mjtObj.mjOBJ_JOINT, j) for s in exclude)]
    actuator_of = {int(model.actuator_trnid[a][0]): a for a in range(model.nu)
                   if model.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT}

    out = dict(dof=len(joints), offsets=[], quats=[], ranges=[], kp=[], kv=[], force=[],
               armature=[], damping=[], frictionloss=[], chain_length=0.0)
    rotation, position = np.eye(3), np.zeros(3)
    for j in joints:
        body = model.jnt_bodyid[j]
        out["offsets"].append(np.array(model.body_pos[body], float))
        out["quats"].append(np.array(model.body_quat[body], float))
        out["ranges"].append(tuple(float(v) for v in model.jnt_range[j]))
        dof = model.jnt_dofadr[j]
        out["armature"].append(float(model.dof_armature[dof]))
        out["damping"].append(float(model.dof_damping[dof]))
        out["frictionloss"].append(float(model.dof_frictionloss[dof]))
        a = actuator_of.get(j)
        out["kp"].append(float(model.actuator_gainprm[a][0]) if a is not None else 0.0)
        out["kv"].append(float(-model.actuator_biasprm[a][2]) if a is not None else 0.0)
        out["force"].append(float(abs(model.actuator_forcerange[a][1])) if a is not None else 0.0)
        position = position + rotation @ np.array(model.body_pos[body], float)
        rotation = rotation @ quat_to_mat(model.body_quat[body])
    out["chain_length"] = float(sum(np.linalg.norm(o) for o in out["offsets"]))
    out["flange_at_zero"] = float(np.linalg.norm(position))
    return out


REAL = {label: measure(key, mjcf, exclude) for label, key, mjcf, exclude in REAL_ARMS}


@pytest.mark.parametrize("label", sorted(REAL))
def test_dof_is_in_support(label):
    low, high = SUPPORT["dof"]
    assert low <= REAL[label]["dof"] <= high


@pytest.mark.parametrize("label", sorted(REAL))
def test_body_offsets_are_in_support(label):
    """FR3's structure is lateral (0.316, 0.0825, 0.3928, 0.088 m), which the original
    +-0.05 lateral cap excluded outright."""
    limit = SUPPORT["offset_abs"]
    for offset in REAL[label]["offsets"]:
        assert np.abs(offset).max() <= limit, f"{label}: offset {offset} exceeds +-{limit}"


@pytest.mark.parametrize("label", sorted(REAL))
def test_orientations_are_representable(label):
    """Segment orientations are full quaternions, so any rotation is representable. This is the
    axis that excluded YAM: 4 of its 6 body quats are not multiples of 90 degrees, so a
    5-token twist set could never express them."""
    for quat in REAL[label]["quats"]:
        assert abs(float(np.linalg.norm(quat)) - 1.0) < 1e-6
        matrix = quat_to_mat(quat)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)


@pytest.mark.parametrize("label", sorted(REAL))
def test_joint_ranges_are_in_support(label):
    """Offset ranges must be reachable: FR3 joint4 [-3.042, -0.152], joint6 [0.544, 4.517];
    and spans from YAM's narrow +-1.571 to its wide [0, 3.665]."""
    span_low, span_high = SUPPORT["range_span"]
    limit = SUPPORT["range_abs"]
    for low, high in REAL[label]["ranges"]:
        assert low < high, f"{label}: degenerate range ({low}, {high})"
        assert span_low <= high - low <= span_high, f"{label}: span {high - low:.3f}"
        assert max(abs(low), abs(high)) <= limit, f"{label}: |limit| {max(abs(low), abs(high)):.3f}"


@pytest.mark.parametrize("label", sorted(REAL))
@pytest.mark.parametrize("field", ["kp", "kv", "force", "armature", "damping", "frictionloss"])
def test_per_joint_dynamics_are_in_support(label, field):
    """Per joint, not per arm: FR3 runs kp 4500 -> 2000 while YAM runs 40 -> 10, two decades
    apart, and YAM's armature is two-tiered (0.032 x3 then 0.0018 x3)."""
    low, high = SUPPORT[field]
    for value in REAL[label][field]:
        assert low <= value <= high, f"{label}: {field}={value} outside [{low}, {high}]"


@pytest.mark.parametrize("label", sorted(REAL))
def test_chain_length_is_in_support(label):
    """Scale is measured as total chain length, not flange-distance-at-zero-qpos: YAM is folded
    at qpos=0, so the latter calls a 0.7 m arm 0.20 m and would size every arm by an arbitrary
    home-pose convention."""
    low, high = SUPPORT["chain_length"]
    got = REAL[label]["chain_length"]
    assert low <= got <= high, f"{label}: chain length {got:.3f} m outside [{low}, {high}]"


@pytest.mark.parametrize("label", sorted(REAL))
def test_gripper_is_exactly_available(label):
    """The one axis that is exact rather than merely in-support: both robots' grippers are
    catalog entries, so a generated robot can graft the identical asset."""
    from molmo_spaces.embodiment_gen.grippers import load_catalog
    expected = {"franka_droid": "robotiq_2f85", "i2rt_yam": "yam_jaw"}[label]
    assert expected in load_catalog()


def test_sampler_stays_inside_its_own_declared_support():
    """The support declaration is only meaningful if the sampler respects it too."""
    for seed in range(400):
        params = sample_params(seed)
        arm = params.arm
        assert SUPPORT["dof"][0] <= arm.dof <= SUPPORT["dof"][1]
        assert (SUPPORT["chain_length"][0] <= chain_length(arm)
                <= SUPPORT["chain_length"][1]), f"seed {seed}: {chain_length(arm):.3f} m"
        for segment in arm.segments:
            assert max(abs(v) for v in segment.offset) <= SUPPORT["offset_abs"], f"seed {seed}"
        for joint in arm.joints:
            low, high = joint.range
            assert low < high
            span = high - low
            # A prismatic joint's range is in METRES, so it must be screened against the slide
            # band, not the radian one -- otherwise adding the kind silently breaks range support.
            band = SUPPORT["slide_span"] if joint.kind == "slide" else SUPPORT["range_span"]
            assert band[0] - 1e-9 <= span <= band[1] + 1e-9, (
                f"seed {seed}: {joint.kind} span {span:.6f}")
            limit = SUPPORT["slide_span"][1] if joint.kind == "slide" else SUPPORT["range_abs"]
            assert max(abs(low), abs(high)) <= limit + 1e-9
            for field in ("kp", "kv", "force", "armature", "damping", "frictionloss"):
                lo, hi = SUPPORT[field]
                assert lo <= getattr(joint, field) <= hi, f"seed {seed}: {field}"


# ---------------------------------------------------------------------------------------------
# Constructive support: exhibit the parameters, do not merely bound-check them.
#
# Bound-checking each quantity marginally is not enough and gave a false pass once already: with
# kp_i = kp_root * decay**i every individual kp sat inside SUPPORT while FR3's actual schedule
# was unreachable for any seed (kp_0 == kp_1 forces decay == 1, which forbids kp_2). The only
# honest test is to build the parameter object and check it is (a) admissible and (b) realizes
# the real robot's kinematics.
# ---------------------------------------------------------------------------------------------

from molmo_spaces.embodiment_gen.constants import GENERATOR_VERSION  # noqa: E402
from molmo_spaces.embodiment_gen.params import (  # noqa: E402
    ArmParams,
    DressingParams,
    EmbodimentParams,
    GeomParams,
    InertialParams,
    JointParams,
    LinkParams,
    SegmentParams,
)
from molmo_spaces.embodiment_gen.skeleton import chain_at_zero  # noqa: E402

GRIPPER_OF = {"franka_droid": "robotiq_2f85", "i2rt_yam": "yam_jaw"}


def params_of_real(label: str) -> EmbodimentParams:
    """The EmbodimentParams that IS the real robot -- the witness that it is in the space."""
    r = REAL[label]
    arm = ArmParams(
        grammar="measured",
        segments=tuple(SegmentParams(quat=tuple(float(v) for v in q),
                                     offset=tuple(float(v) for v in o))
                       for q, o in zip(r["quats"], r["offsets"])),
        joints=tuple(JointParams(kind="hinge", range=rng_, kp=kp, kv=kv, force=f,
                                 armature=a, damping=d, frictionloss=fl)
                     for rng_, kp, kv, f, a, d, fl in zip(
                         r["ranges"], r["kp"], r["kv"], r["force"],
                         r["armature"], r["damping"], r["frictionloss"])),
        # The witness wears its OWN geometry: a coherent pool draw, link i from its own source.
        # That is the mechanism a sampled robot uses too -- nothing here is a special case.
        links=tuple(LinkParams(
            geom=GeomParams(mode="pool", source=label, link_index=index),
            inertial=InertialParams(mode="explicit", density=1000.0, mass=link.mass,
                                    inertia=link.inertia, com=link.com))
            for index, link in enumerate(harvest(label))),
        mount_pos=(0.0, 0.0, 0.0),
        mount_quat=(1.0, 0.0, 0.0, 0.0),
        pedestal_height=0.7,
        pedestal_radius=0.15,
    )
    return EmbodimentParams(
        seed=-1, version=GENERATOR_VERSION, arm=arm, gripper=GRIPPER_OF[label],
        dressing=DressingParams(palette={"base": [0.5, 0.5, 0.5, 1.0]},
                                radius=0.04, taper=0.7, accent_period=2),
    )


@pytest.mark.parametrize("label", sorted(REAL))
def test_real_robot_is_expressible_as_params(label):
    """Every measured value is assignable -- no coupling in the dataclasses forbids it."""
    params = params_of_real(label)
    r = REAL[label]
    assert params.arm.dof == r["dof"]
    assert [j.kp for j in params.arm.joints] == list(r["kp"])
    assert [j.kv for j in params.arm.joints] == list(r["kv"])
    assert [j.armature for j in params.arm.joints] == list(r["armature"])
    assert params.gripper in GRIPPER_OF.values()


@pytest.mark.parametrize("label", sorted(REAL))
def test_params_reproduce_the_real_kinematics_exactly(label):
    """The strong claim: the parameterization is expressive enough that the witness reproduces
    the real robot's joint origins and axes to numerical precision. If this passes, the
    generated distribution contains that robot's kinematics -- not something near it."""
    params = params_of_real(label)
    origins, axes, flange = chain_at_zero(params.arm)

    model = mujoco.MjModel.from_xml_path(
        str(get_robot_path(label if label != "franka_droid" else "franka_droid")
            / ("yam.xml" if label == "i2rt_yam" else "model.xml")))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)

    exclude = dict((lbl, exc) for lbl, _, _, exc in REAL_ARMS)[label]
    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
    joints = [j for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
              and not any(s in name(j) for s in exclude)]

    # MuJoCo reports these in world coordinates; the arm root is the first joint's parent body.
    root = model.body_parentid[model.jnt_bodyid[joints[0]]]
    root_pos = np.array(data.xpos[root], float)
    root_mat = np.array(data.xmat[root], float).reshape(3, 3)

    for index, j in enumerate(joints):
        body = model.jnt_bodyid[j]
        want_pos = root_mat.T @ (np.array(data.xpos[body], float) - root_pos)
        want_axis = root_mat.T @ (np.array(data.xmat[body], float).reshape(3, 3)
                                 @ np.array(model.jnt_axis[j], float))
        np.testing.assert_allclose(origins[index], want_pos, atol=1e-9,
                                   err_msg=f"{label}: joint {index + 1} origin")
        np.testing.assert_allclose(axes[index], want_axis, atol=1e-9,
                                   err_msg=f"{label}: joint {index + 1} axis")
    assert np.isfinite(flange).all()


@pytest.mark.parametrize("label", sorted(REAL))
def test_witness_params_lie_inside_declared_support(label):
    """Expressible is not enough; the witness must also be admissible, i.e. inside SUPPORT --
    otherwise the sampler's own clamps would reject it."""
    arm = params_of_real(label).arm
    assert SUPPORT["dof"][0] <= arm.dof <= SUPPORT["dof"][1]
    assert (SUPPORT["chain_length"][0] <= chain_length(arm) <= SUPPORT["chain_length"][1])
    for segment in arm.segments:
        assert max(abs(v) for v in segment.offset) <= SUPPORT["offset_abs"]
    for joint in arm.joints:
        low, high = joint.range
        assert low < high
        assert SUPPORT["range_span"][0] - 1e-9 <= high - low <= SUPPORT["range_span"][1] + 1e-9
        assert max(abs(low), abs(high)) <= SUPPORT["range_abs"]
        for field in ("kp", "kv", "force", "armature", "damping", "frictionloss"):
            lo, hi = SUPPORT[field]
            assert lo <= getattr(joint, field) <= hi, f"{label}: {field}={getattr(joint, field)}"
    for link in arm.links:
        inertial = link.inertial
        assert SUPPORT["mass"][0] <= inertial.mass <= SUPPORT["mass"][1]
        assert all(SUPPORT["inertia"][0] <= v <= SUPPORT["inertia"][1] for v in inertial.inertia)
        assert np.linalg.norm(inertial.com) <= SUPPORT["com_offset"][1]


# ---------------------------------------------------------------------------------------------
# Geometry and inertia: widened MECHANISMS, verified against the real robots.
# ---------------------------------------------------------------------------------------------


def test_geom_pool_matches_its_declaration():
    """params samples (source, link_index) without touching disk, so the declared link counts
    must match the live harvest or the sampler would index past the pool."""
    for source, spec in sorted(GEOM_SOURCES.items()):
        assert len(harvest(source)) == spec["n_links"], source


@pytest.mark.parametrize("label", sorted(REAL))
def test_pool_carries_collision_geometry_not_just_visuals(label):
    """A pool draw must bring contact geoms along. FR3's collision is mesh-based, so a
    capsule-only collision model could never be an FR3 whatever its visuals looked like."""
    pool = harvest(label)
    assert any(g.collides for link in pool for g in link.geoms), f"{label}: no contact geoms"
    assert any(g.mesh for link in pool for g in link.geoms), f"{label}: no mesh geoms"


def test_both_geometry_idioms_are_representable():
    """FR3 is all-mesh including collision; YAM pairs visual meshes with capsule collision.
    Both must survive harvesting, or one real idiom is outside the space."""
    fr3 = [g for link in harvest("franka_droid") for g in link.geoms]
    yam = [g for link in harvest("i2rt_yam") for g in link.geoms]
    assert all(g.mesh for g in fr3), "FR3 should be entirely mesh geoms"
    assert any(g.collides and g.mesh for g in fr3), "FR3 collision should be mesh-based"
    assert any(g.collides and g.type == "capsule" for g in yam), "YAM collision should be capsules"
    assert any(g.mesh and not g.collides for g in yam), "YAM should have visual-only meshes"


def test_inertial_support_contains_every_measured_link():
    """The inertial bands are generic ranges that CONTAIN the measured links; this asserts the
    containment directly rather than trusting the constants."""
    bounds = measured_inertial_bounds()
    assert SUPPORT["mass"][0] <= bounds["mass"][0] and bounds["mass"][1] <= SUPPORT["mass"][1]
    assert bounds["inertia"][1] <= SUPPORT["inertia"][1]
    assert bounds["com"][1] <= SUPPORT["com_offset"][1]


def test_sampler_reaches_both_geometry_and_inertial_modes():
    """Mixtures, not constants: if either mode never appears the corresponding real idiom is
    unreachable in practice however wide the declared support is."""
    geom_modes, inertial_modes, kinds, sources = set(), set(), set(), set()
    for seed in range(600):
        arm = sample_params(seed).arm
        for link in arm.links:
            geom_modes.add(link.geom.mode)
            inertial_modes.add(link.inertial.mode)
            if link.geom.source:
                sources.add(link.geom.source)
        kinds.update(joint.kind for joint in arm.joints)
    assert geom_modes == {"primitive", "pool"}, geom_modes
    assert inertial_modes == {"computed", "explicit"}, inertial_modes
    assert sources == set(GEOM_SOURCES), sources
    # ARM joints are revolute only: a prismatic arm joint translates its child, so no static
    # geometry can span the changing distance. Gripper joints may still be prismatic and yam_jaw's
    # are -- see test_arm_joints_are_revolute_but_grippers_may_be_prismatic.
    assert kinds == {"hinge"}, kinds


def test_coherent_pool_draws_actually_happen():
    """A wholly real-looking arm should be uncommon, not astronomically rare: some seeds must
    dress every link from one source at its own index."""
    coherent = 0
    for seed in range(2000):
        arm = sample_params(seed).arm
        geoms = [link.geom for link in arm.links]
        pooled = [g for g in geoms if g.mode == "pool"]
        if len(pooled) == len(geoms) and len({g.source for g in pooled}) == 1 \
                and all(g.link_index == i for i, g in enumerate(geoms)):
            coherent += 1
    assert coherent > 0, "no seed in 2000 produced a fully coherent real-geometry arm"


def test_sampler_stays_inside_geometry_and_inertial_support():
    for seed in range(400):
        arm = sample_params(seed).arm
        assert len(arm.links) == arm.dof
        assert np.linalg.norm(arm.mount_pos) <= SUPPORT["mount_offset"][1]
        assert abs(float(np.linalg.norm(arm.mount_quat)) - 1.0) < 1e-9
        for link in arm.links:
            geom, inertial = link.geom, link.inertial
            if geom.mode == "pool":
                assert geom.source in GEOM_SOURCES
                assert 0 <= geom.link_index < GEOM_SOURCES[geom.source]["n_links"]
            else:
                assert geom.source is None and geom.link_index is None
            assert SUPPORT["mass"][0] <= inertial.mass <= SUPPORT["mass"][1]
            assert SUPPORT["density"][0] <= inertial.density <= SUPPORT["density"][1]
            assert all(SUPPORT["inertia"][0] <= v <= SUPPORT["inertia"][1]
                       for v in inertial.inertia)
            assert np.linalg.norm(inertial.com) <= SUPPORT["com_offset"][1] * np.sqrt(3)


