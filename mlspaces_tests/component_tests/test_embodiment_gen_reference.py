"""Do the witness parameters actually BUILD the shipped robots?

The support tests ask whether the real robots are admissible. This asks the stronger question:
push their parameters through the ordinary ``builder`` and check the model that comes out has the
real robot's kinematics, dynamics, geometry and mass -- proving the two-stage split is faithful.

    reference.params_of(source)  ->  builder.build()  ->  compare against the shipped MJCF

Nothing here uses a private path: the witness is a coherent pool draw plus explicit inertials,
both of which ``sampling.py`` produces on its own.
"""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from molmo_spaces.embodiment_gen import reference
from molmo_spaces.embodiment_gen.builder import build, chain_in_model
from molmo_spaces.embodiment_gen.geom_pool import SOURCES, harvest
from molmo_spaces.embodiment_gen.params import SUPPORT
from molmo_spaces.embodiment_gen.sampling import sample_params
from molmo_spaces.embodiment_gen.skeleton import quat_to_mat
from molmo_spaces.molmo_spaces_constants import get_robot_path

SOURCE_NAMES = sorted(SOURCES)


def arm_link_body(source: str, index: int, dof: int) -> str:
    """Where link ``index`` (1-based) lands in the built model.

    A gripper whose graft root IS an arm link -- yam_jaw's is link_6 -- supplies the flange link
    itself, so that link's geometry and mass live under the GRAFTED body, not under ``link<dof>``.
    Checking ``link<dof>`` there would be checking a deliberately bare body."""
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    asset = load_catalog()[reference.GRIPPER_OF[source]]
    if index == dof and asset.claims_flange_link:
        return f"gripper/{asset.graft_root}"
    return f"link{index}"


def real_chain(source: str):
    """``(origins, axes)`` of the shipped robot's arm joints, in its arm-root frame at qpos=0."""
    spec = SOURCES[source]
    model = mujoco.MjModel.from_xml_path(str(get_robot_path(source) / spec["mjcf"]))
    data = mujoco.MjData(model)
    data.qpos[:] = 0.0
    mujoco.mj_forward(model, data)
    name = lambda i: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
    joints = [j for j in range(model.njnt)
              if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
              and not any(s in name(j) for s in spec["exclude"])]
    root = model.body_parentid[model.jnt_bodyid[joints[0]]]
    root_pos = np.array(data.xpos[root], float)
    root_mat = np.array(data.xmat[root], float).reshape(3, 3)
    origins, axes = [], []
    for joint in joints:
        body = model.jnt_bodyid[joint]
        origins.append(root_mat.T @ (np.array(data.xpos[body], float) - root_pos))
        axes.append(root_mat.T @ (np.array(data.xmat[body], float).reshape(3, 3)
                                @ np.array(model.jnt_axis[joint], float)))
    return np.stack(origins), np.stack(axes)


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_witness_builds(source):
    _, model = build(reference.params_of(source))
    dof = reference.params_of(source).arm.dof
    assert model.nu == dof + 1, f"{dof} arm actuators + 1 gripper, got {model.nu}"
    assert model.neq >= 1, "the grafted gripper's coupling did not survive"


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_built_kinematics_match_the_real_robot(source):
    """Joint origins and axes, in the arm-root frame, to numerical precision. This is the claim
    that the generated distribution contains the robot's kinematics -- not something near it."""
    params = reference.params_of(source)
    _, model = build(params)
    got_origins, got_axes = chain_in_model(model, params.arm.dof)
    want_origins, want_axes = real_chain(source)
    np.testing.assert_allclose(got_origins, want_origins, atol=1e-9)
    np.testing.assert_allclose(np.abs(got_axes), np.abs(want_axes), atol=1e-9)


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_built_dynamics_match_the_real_robot(source):
    """Per-joint limits, gains, armature, damping and friction, read back off the built model."""
    params = reference.params_of(source)
    _, model = build(params)
    for index, joint in enumerate(params.arm.joints, start=1):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{index}")
        aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"joint{index}")
        dof = model.jnt_dofadr[jid]
        np.testing.assert_allclose(model.jnt_range[jid], joint.range, atol=1e-9)
        assert float(model.actuator_gainprm[aid][0]) == pytest.approx(joint.kp)
        assert float(-model.actuator_biasprm[aid][2]) == pytest.approx(joint.kv)
        assert float(model.dof_armature[dof]) == pytest.approx(joint.armature)
        assert float(model.dof_damping[dof]) == pytest.approx(joint.damping)
        assert float(model.dof_frictionloss[dof]) == pytest.approx(joint.frictionloss)


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_built_masses_match_the_real_robot(source):
    """Explicit inertials are the only route to what the real robots ship, since a hand-specified
    inertia need not equal anything a density times a geom would produce."""
    params = reference.params_of(source)
    _, model = build(params)
    pool = harvest(source)
    for index, link in enumerate(pool, start=1):
        body = arm_link_body(source, index, params.arm.dof)
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        assert float(model.body_mass[bid]) == pytest.approx(link.mass, rel=1e-9), body
        np.testing.assert_allclose(model.body_inertia[bid], link.inertia, rtol=1e-6, atol=1e-12)
        np.testing.assert_allclose(model.body_ipos[bid], link.com, atol=1e-9)
    total = sum(link.mass for link in pool)
    arm_total = sum(float(model.body_mass[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, arm_link_body(source, i, params.arm.dof))])
        for i in range(1, len(pool) + 1))
    assert arm_total == pytest.approx(total, rel=1e-9)


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_built_geometry_matches_the_real_robot(source):
    """Per-link geom count, type, contact flags and COMPILED placement -- so a pool draw
    reproduces the real link's collision model, not merely its silhouette.

    Compiled placement is compared against the real model's compiled placement, never against the
    authored values: MuJoCo recenters mesh assets at compile time and folds the shift into
    ``geom_pos``, so the two representations legitimately differ (and harvesting the compiled one
    would double-apply the shift -- that bug moved link1 by 85 mm)."""
    params = reference.params_of(source)
    _, model = build(params)
    real = mujoco.MjModel.from_xml_path(
        str(get_robot_path(source) / SOURCES[source]["mjcf"]))

    compared = 0
    for index, link in enumerate(harvest(source), start=1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                arm_link_body(source, index, params.arm.dof))
        rid = mujoco.mj_name2id(real, mujoco.mjtObj.mjOBJ_BODY, link.body)
        built = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
        actual = [g for g in range(real.ngeom) if real.geom_bodyid[g] == rid]
        assert len(built) == len(actual) == len(link.geoms), f"link{index}: geom count"
        for got, want, spec in zip(built, actual, link.geoms):
            assert int(model.geom_type[got]) == int(real.geom_type[want]), f"link{index}: type"
            assert int(model.geom_contype[got]) == spec.contype, f"link{index}: contype"
            assert int(model.geom_conaffinity[got]) == spec.conaffinity, f"link{index}: conaff"
            assert int(model.geom_group[got]) == spec.group, f"link{index}: group"
            np.testing.assert_allclose(model.geom_pos[got], real.geom_pos[want], atol=1e-9)
            np.testing.assert_allclose(model.geom_size[got], real.geom_size[want], atol=1e-9)
            compared += 1
    assert compared == sum(len(link.geoms) for link in harvest(source))


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_witness_is_admissible(source):
    """Buildable is not enough -- the witness must lie inside SUPPORT, or the sampler's own
    clamps would reject the very robot we claim is in the distribution."""
    arm = reference.params_of(source).arm
    assert SUPPORT["dof"][0] <= arm.dof <= SUPPORT["dof"][1]
    for segment in arm.segments:
        assert max(abs(v) for v in segment.offset) <= SUPPORT["offset_abs"]
    for joint in arm.joints:
        low, high = joint.range
        assert SUPPORT["range_span"][0] - 1e-9 <= high - low <= SUPPORT["range_span"][1] + 1e-9
        assert max(abs(low), abs(high)) <= SUPPORT["range_abs"]
        for field in ("kp", "kv", "force", "armature", "damping", "frictionloss"):
            lo, hi = SUPPORT[field]
            assert lo <= getattr(joint, field) <= hi, field
    for link in arm.links:
        assert SUPPORT["mass"][0] <= link.inertial.mass <= SUPPORT["mass"][1]
        assert all(SUPPORT["inertia"][0] <= v <= SUPPORT["inertia"][1]
                   for v in link.inertial.inertia)
        assert np.linalg.norm(link.inertial.com) <= SUPPORT["com_offset"][1]


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_witness_uses_only_mechanisms_the_sampler_has(source):
    """The witness must not depend on a private path: every mode it uses must be one the sampler
    reaches on its own, else the demonstration proves nothing about the population."""
    arm = reference.params_of(source).arm
    assert {link.geom.mode for link in arm.links} <= {"primitive", "pool"}
    assert {link.inertial.mode for link in arm.links} <= {"computed", "explicit"}
    assert {joint.kind for joint in arm.joints} <= {"hinge", "slide"}

    reached_geom, reached_inertial = set(), set()
    for seed in range(400):
        for link in sample_params(seed).arm.links:
            reached_geom.add(link.geom.mode)
            reached_inertial.add(link.inertial.mode)
    assert {link.geom.mode for link in arm.links} <= reached_geom
    assert {link.inertial.mode for link in arm.links} <= reached_inertial


def _imported_modules(module) -> set:
    """Sibling modules a module actually imports -- parsed, not grepped, so prose in a docstring
    cannot masquerade as a dependency."""
    import ast
    import inspect

    names = set()
    for node in ast.walk(ast.parse(inspect.getsource(module))):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.lstrip("."))
            if node.level and not node.module:
                names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level:
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
    return {n.rsplit(".", 1)[-1] for n in names}


def test_the_two_stages_do_not_know_each_other():
    """The split is the point. sampling.py owns randomness and never builds; builder.py owns
    realization and never draws. Neither may import the reference witnesses, or a real robot's
    values could leak into the generated population."""
    import inspect

    from molmo_spaces.embodiment_gen import builder, sampling

    sampling_imports = _imported_modules(sampling)
    builder_imports = _imported_modules(builder)

    assert "builder" not in sampling_imports, sampling_imports
    assert "sampling" not in builder_imports, builder_imports
    for imports, name in ((sampling_imports, "sampling"), (builder_imports, "builder")):
        assert "reference" not in imports, f"{name} imports the reference witnesses"

    # the contract is the only thing they may share, and both must actually go through it
    assert "params" in sampling_imports, sampling_imports
    assert "params" in builder_imports, builder_imports

    builder_src = inspect.getsource(builder)
    assert "np.random" not in builder_src, "builder must contain no randomness"
    assert "Generator(" not in builder_src, "builder must contain no randomness"
    assert "sample_params" not in builder_src, "builder must never see a seed"


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_quaternions_are_reproduced_not_approximated(source):
    """The segment quaternions are read off the model and rebuilt, so each must round-trip to an
    orthonormal rotation -- the axis that a 5-token twist set could never express."""
    for segment in reference.params_of(source).arm.segments:
        matrix = quat_to_mat(segment.quat)
        np.testing.assert_allclose(matrix @ matrix.T, np.eye(3), atol=1e-9)
        assert abs(float(np.linalg.norm(segment.quat)) - 1.0) < 1e-9


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_built_materials_match_the_real_robot(source):
    """Appearance is part of the embodiment, not decoration: it is what a vision policy sees.
    Without material inheritance the witnesses rendered in MuJoCo's default grey while the real
    robots are white-and-black, i.e. they were in distribution geometrically but not visually."""
    from molmo_spaces.embodiment_gen.geom_pool import materials

    params = reference.params_of(source)
    _, model = build(params)
    real = mujoco.MjModel.from_xml_path(str(get_robot_path(source) / SOURCES[source]["mjcf"]))
    pool_materials = materials(source)

    compared = 0
    for index, link in enumerate(harvest(source), start=1):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                arm_link_body(source, index, params.arm.dof))
        built = [g for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
        for got, want in zip(built, link.geoms):
            mid = int(model.geom_matid[got])
            if want.material:
                assert mid >= 0, f"link{index}: {want.material} did not survive the build"
                expected = pool_materials[want.material]
                np.testing.assert_allclose(model.mat_rgba[mid], expected.rgba, atol=1e-6)
                assert float(model.mat_specular[mid]) == pytest.approx(expected.specular)
                assert float(model.mat_shininess[mid]) == pytest.approx(expected.shininess)
                assert float(model.mat_reflectance[mid]) == pytest.approx(expected.reflectance)
            else:
                # no material on the source geom: its own rgba carries the colour
                np.testing.assert_allclose(model.geom_rgba[got], want.rgba, atol=1e-6)
            compared += 1
    assert compared == sum(len(link.geoms) for link in harvest(source))
    del real


def test_sampler_reaches_both_material_modes_and_textures():
    """Appearance must be a mixture, or one real idiom is unreachable in practice: 'inherit' is
    the only route to a real robot's materials, 'palette' is visual DR over a real morphology."""
    from molmo_spaces.embodiment_gen.params import MATERIAL_MODES, TEXTURE_KINDS

    modes, kinds = set(), set()
    for seed in range(600):
        params = sample_params(seed)
        kinds.add(params.dressing.texture_kind)
        for link in params.arm.links:
            if link.geom.mode == "pool":
                modes.add(link.geom.material_mode)
    assert modes == set(MATERIAL_MODES), modes
    assert kinds == set(TEXTURE_KINDS), kinds


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_total_geometry_and_mass_match_the_real_robot(source):
    """The whole-robot totals, which per-link checks missed twice over.

    The base link carries no joint, so a joint-keyed harvest skipped it and the built robot was
    short FR3's fr3_link0 (8 geoms, 2.82 kg) and YAM's motor housing and mounting plate. And
    yam_jaw's graft root IS an arm link, so link_6 was emitted twice -- 35 gripper geoms against
    the real 30, with 0.367 kg double-counted. Per-link assertions passed throughout both."""
    _, built = build(reference.params_of(source))
    real = mujoco.MjModel.from_xml_path(str(get_robot_path(source) / SOURCES[source]["mjcf"]))
    assert built.ngeom == real.ngeom, f"{built.ngeom} geoms vs {real.ngeom}"
    assert float(built.body_mass.sum()) == pytest.approx(float(real.body_mass.sum()), rel=1e-9)


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_base_link_carries_the_mounting_structure(source):
    """The base is a link like any other; it just has no joint."""
    from molmo_spaces.embodiment_gen.geom_pool import harvest_base

    params = reference.params_of(source)
    assert params.arm.base is not None
    _, built = build(params)
    bid = mujoco.mj_name2id(built, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    geoms = [g for g in range(built.ngeom) if built.geom_bodyid[g] == bid]
    expected = harvest_base(source)
    assert len(geoms) == len(expected.geoms) > 0
    assert float(built.body_mass[bid]) == pytest.approx(expected.mass, rel=1e-9)


def test_a_gripper_claiming_the_flange_is_not_double_emitted():
    """yam_jaw's graft root is the arm's own link_6, so the arm must leave that link bare."""
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    catalog = load_catalog()
    assert catalog["yam_jaw"].claims_flange_link is True
    assert catalog["robotiq_2f85"].claims_flange_link is False

    params = reference.params_of("i2rt_yam")
    _, built = build(params)
    last = mujoco.mj_name2id(built, mujoco.mjtObj.mjOBJ_BODY, f"link{params.arm.dof}")
    assert not [g for g in range(built.ngeom) if built.geom_bodyid[g] == last], (
        "the flange link was dressed as well as grafted")


def test_sampler_gives_every_robot_a_base():
    for seed in range(200):
        assert sample_params(seed).arm.base is not None, seed


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_gripper_is_mounted_where_the_real_robot_mounts_it(source):
    """Every gripper geom, in the last arm link's frame, to machine precision.

    A zero mount transform is wrong for both robots and neither value is readable off one model:
    franka_droid was itself assembled by attaching its gripper at a FRAME, so the authored
    gripper/base pos reads (0,0,0) while the compiled placement carries the frame's pose -- the
    gripper sat 0.107 m and 90 degrees out. yam_jaw's graft root is the arm's own link_6, whose
    pose the arm already applies as segment 6, so the mount must cancel it or it lands twice
    (0.221 m out). Both are derived by reference.measure_mount."""
    params = reference.params_of(source)
    _, built = build(params)
    real = mujoco.MjModel.from_xml_path(str(get_robot_path(source) / SOURCES[source]["mjcf"]))

    def frame(model, data, body):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
        return np.array(data.xpos[i], float), np.array(data.xmat[i], float).reshape(3, 3)

    def rolled(model):
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        return data

    body_of = lambda m, g: mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY,
                                            int(m.geom_bodyid[g])) or ""
    built_data, real_data = rolled(built), rolled(real)
    last_real = harvest(source)[-1].body
    bp, bR = frame(built, built_data, f"link{params.arm.dof}")
    rp, rR = frame(real, real_data, last_real)

    grafted = [g for g in range(built.ngeom) if body_of(built, g).startswith("gripper/")]
    subtree = set()
    stack = [mujoco.mj_name2id(real, mujoco.mjtObj.mjOBJ_BODY,
                               load_catalog_root(source))]
    while stack:
        b = stack.pop()
        subtree.add(b)
        stack.extend(i for i in range(real.nbody) if real.body_parentid[i] == b and i != b)
    shipped = [g for g in range(real.ngeom) if int(real.geom_bodyid[g]) in subtree]
    assert len(grafted) == len(shipped) > 0, f"{len(grafted)} vs {len(shipped)}"

    for got, want in zip(grafted, shipped):
        a = bR.T @ (np.array(built_data.geom_xpos[got], float) - bp)
        b = rR.T @ (np.array(real_data.geom_xpos[want], float) - rp)
        np.testing.assert_allclose(a, b, atol=1e-9)


def load_catalog_root(source: str) -> str:
    """The body in the SHIPPED model that the graft reproduces."""
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    asset = load_catalog()[reference.GRIPPER_OF[source]]
    return asset.graft_root if asset.claims_flange_link else f"gripper/{asset.graft_root}"


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_measured_mount_means_flange_to_gripper(source):
    """``graft`` cancels the graft root's own pose, so the mount is the residual flange -> gripper
    transform and means the same thing for every catalog entry.

    The expected value differs by case and both cases are load-bearing. FR3 mounts its gripper
    0.107 m out and rotated 90 degrees, so identity there would be the old bug returning. YAM's
    graft root IS its last arm link, so the residual is exactly identity -- a non-zero value there
    would mean the root pose was being applied twice."""
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    pos, quat = reference.measure_mount(source, reference.GRIPPER_OF[source])
    if load_catalog()[reference.GRIPPER_OF[source]].claims_flange_link:
        np.testing.assert_allclose(pos, [0, 0, 0], atol=1e-9)
        assert abs(abs(quat[0]) - 1.0) < 1e-9, quat
    else:
        assert np.linalg.norm(pos) > 1e-3 or abs(abs(quat[0]) - 1.0) > 1e-3, (pos, quat)


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_cameras_and_sites_match_the_real_robot(source):
    """Camera and site placement belong to the EMBODIMENT.

    MolmoSpaces wrist cameras are all ``MjcfCameraConfig``: the config names a camera and the MODEL
    fixes its pose, so a robot that declares none has no wrist camera and any config naming one
    fails to resolve. Hardcoding instead of parameterizing broke this both ways -- FR3's wrist_cam
    and attachment_site were missing entirely, and YAM gained an attachment_site it does not have.
    """
    params = reference.params_of(source)
    _, built = build(params)
    real = mujoco.MjModel.from_xml_path(str(get_robot_path(source) / SOURCES[source]["mjcf"]))
    name = lambda m, t, i: mujoco.mj_id2name(m, t, i) or ""

    assert built.ncam == real.ncam, f"{built.ncam} cameras vs {real.ncam}"
    assert ({name(built, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(built.ncam)}
            == {name(real, mujoco.mjtObj.mjOBJ_CAMERA, i) for i in range(real.ncam)})
    assert built.nsite == real.nsite, f"{built.nsite} sites vs {real.nsite}"

    # every camera at the same pose in the arm-root frame, and the same fovy
    def rolled(model):
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)
        return data

    built_data, real_data = rolled(built), rolled(real)
    for i in range(real.ncam):
        cam = name(real, mujoco.mjtObj.mjOBJ_CAMERA, i)
        j = mujoco.mj_name2id(built, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        assert j >= 0, cam
        assert float(built.cam_fovy[j]) == pytest.approx(float(real.cam_fovy[i])), cam
        np.testing.assert_allclose(built_data.cam_xpos[j], real_data.cam_xpos[i], atol=1e-9,
                                   err_msg=cam)
        np.testing.assert_allclose(built_data.cam_xmat[j], real_data.cam_xmat[i], atol=1e-9,
                                   err_msg=cam)


def test_droid_and_yam_camera_mounts_are_representable():
    """Both shipped camera setups must be REPRESENTABLE, MJCF-side and config-side.

    DROID's shoulder camera is a RobotMountedCameraConfig on fr3_link0, i.e. a base-mounted camera
    with a calibrated offset -- 0.876 m from the base. The randomized exocentric cameras sweep
    fov 50-90. And YAM declares NO camera, so zero has to be expressible or its setup is outside
    what the params can say however wide the bands are.

    Representable, not sampled. The sampler draws no cameras at all (see
    ``test_camera_placement_is_not_sampled``); these bands bound what ``CameraParams`` can express,
    which is what the witness rebuild in ``test_cameras_and_sites_match_the_real_robot`` needs."""
    from molmo_spaces.embodiment_gen.geom_pool import measured_camera_bounds
    from molmo_spaces.embodiment_gen.params import SUPPORT

    bounds = measured_camera_bounds()
    assert bounds["offset"][1] <= SUPPORT["camera_offset"][1]
    assert SUPPORT["camera_fovy"][0] <= bounds["fovy"][0]
    assert bounds["fovy"][1] <= SUPPORT["camera_fovy"][1]

    # EVERY camera system in the repo, enumerated rather than hand-picked: checking only
    # FrankaDroidCameraSystem and FrankaRandomizedD405D455CameraSystem missed
    # FrankaRandomizedDroidCameraSystem's exo_camera_3, whose fov_range is (137, 140) and which a
    # band of (20, 130) excluded outright.
    import inspect

    from molmo_spaces.configs import camera_configs

    checked = 0
    for name, obj in sorted(vars(camera_configs).items()):
        if not (inspect.isclass(obj) and name.endswith("CameraSystem")):
            continue
        try:
            system = obj()
        except Exception:                                  # abstract or requires arguments
            continue
        for camera in system.cameras:
            fov = getattr(camera, "fov", None)
            if fov:
                assert SUPPORT["camera_fovy"][0] <= fov <= SUPPORT["camera_fovy"][1], (
                    f"{name}/{camera.name}: fov {fov}")
                checked += 1
            offset = getattr(camera, "camera_offset", None)
            if offset is not None:
                assert np.linalg.norm(offset) <= SUPPORT["camera_offset"][1], (
                    f"{name}/{camera.name}: |offset| {np.linalg.norm(offset):.3f}")
                checked += 1
            fov_range = getattr(camera, "fov_range", None)
            if fov_range:
                assert SUPPORT["camera_fovy"][0] <= min(fov_range), f"{name}/{camera.name}"
                assert max(fov_range) <= SUPPORT["camera_fovy"][1], (
                    f"{name}/{camera.name}: fov_range {tuple(fov_range)}")
                checked += 1
    assert checked >= 20, f"only {checked} config-side camera mounts checked"

    # zero cameras must be expressible, since YAM declares none, and so must FR3's two
    counts = {len(reference.params_of(source).arm.cameras) for source in SOURCE_NAMES}
    assert 0 in counts and counts & {1, 2}, counts


def test_camera_placement_is_not_sampled():
    """The sampler must draw NO cameras. Placement is the camera system's job.

    It used to draw them: 0-2 cameras per arm, each at a random offset up to 0.25 m under a
    uniformly random rotation. That is not a wrist camera -- it is a camera pointing somewhere, and
    an ``MjcfCameraConfig`` naming it produces a stream that shows a different thing on every seed,
    which is unusable as a training view however well-formed the MJCF is. The wrist camera is now
    DERIVED, per gripper, from the camera that gripper actually ships, expressed in its grasp frame,
    so the same gripper always yields the same view of its own jaws."""
    for seed in range(500):
        assert sample_params(seed).arm.cameras == (), f"seed {seed} sampled a camera"


def test_sampler_reaches_optional_sites():
    site_counts = {len(sample_params(seed).arm.sites) for seed in range(500)}
    assert {0, 1} <= site_counts, site_counts


def _worst_break(model, dof: int, gripper_root: str = "") -> float:
    """Largest gap between consecutive geometry-bearing bodies, via the builder's own measurement."""
    from molmo_spaces.embodiment_gen.builder import consecutive_breaks

    return max(consecutive_breaks(model, dof, gripper_root).values(), default=0.0)


def _fragments(model, fractions=(0.0, 0.3, 0.5, 0.7, 1.0)) -> int:
    """Most geometry components the robot falls into over a sweep of configurations."""
    from molmo_spaces.embodiment_gen.builder import geometry_components

    return max(len(geometry_components(model, fraction)) for fraction in fractions)


@pytest.mark.parametrize("source", SOURCE_NAMES)
def test_witness_geometry_is_connected(source):
    """The invariant that calibrates everything: a real robot is one piece and needs no help.

    Zero struts matters as much as one component. Struts are added only where measurement finds a
    break, so any appearing on a shipped robot would mean the threshold is mistaking that robot's
    ordinary joint clearance -- at most 0.99 mm on FR3 -- for a disconnection."""
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    params = reference.params_of(source)
    _, built = build(params)
    root = load_catalog()[params.gripper].graft_root

    struts = [g for g in range(built.ngeom)
              if "strut" in (mujoco.mj_id2name(built, mujoco.mjtObj.mjOBJ_GEOM, g) or "")]
    assert not struts, f"{source} needed {len(struts)} struts; the tolerance is too strict"
    # Not zero: a real robot's consecutive links clear each other, FR3's by up to 0.99 mm. The
    # threshold has to admit that clearance or the repair pass would "fix" the real robots.
    from molmo_spaces.embodiment_gen.builder import CLEARANCE_TOLERANCE

    assert _worst_break(built, params.arm.dof, root) <= CLEARANCE_TOLERANCE
    assert _fragments(built) == 1


def test_sampled_geometry_is_connected():
    """Every sampled robot is ONE piece, in every pose.

    Asserted structurally via connected components over true surface distances, because the
    pair-walk alone cannot see this and a bounding-sphere test cannot see it at all. Both mistakes
    were made here: ``geom_rbound`` reported zero breaks while 42% of arms fell into separate pieces,
    since the bounding sphere of a 0.13 m half-length capsule has radius ~0.14 m and surfaces 20 cm
    apart appeared to touch.

    Connectedness is guaranteed by construction rather than luck: every arm joint is revolute and
    anchored at its child's origin, so that origin is fixed in the parent's frame; where a break is
    measured, struts are added to BOTH bodies with an endpoint there; two capsules sharing a point
    always intersect."""
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    checked = 0
    for seed in range(120):
        params = sample_params(seed)
        try:
            _, built = build(params)
        except Exception:
            continue
        root = load_catalog()[params.gripper].graft_root
        assert _fragments(built) == 1, f"seed {seed} fragments"
        from molmo_spaces.embodiment_gen.builder import CLEARANCE_TOLERANCE

        assert _worst_break(built, params.arm.dof, root) <= CLEARANCE_TOLERANCE, f"seed {seed}"
        checked += 1
    assert checked > 100



def test_arm_joints_are_revolute_but_grippers_may_be_prismatic():
    """The asymmetry is deliberate. A prismatic ARM joint separates its own link as it extends, and
    no static geometry can bridge a distance that changes with qpos. A prismatic GRIPPER joint is
    fine -- and required, since yam_jaw's fingers are exactly that."""
    from mujoco import MjSpec

    from molmo_spaces.embodiment_gen.grippers import graft, load_catalog

    for seed in range(500):
        for joint in sample_params(seed).arm.joints:
            assert joint.kind == "hinge", f"seed {seed}: {joint.kind} arm joint"

    prismatic = {}
    for name in sorted(load_catalog()):
        arm = MjSpec()
        flange = arm.worldbody.add_body(name="flange", pos=[0, 0, 0.3])
        flange.add_geom(name="fg", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.01, 0])
        graft(arm, flange, name)
        model = arm.compile()
        prismatic[name] = any(model.jnt_type[j] == mujoco.mjtJoint.mjJNT_SLIDE
                              for j in range(model.njnt))
    assert prismatic["yam_jaw"], "yam_jaw's prismatic fingers must survive the graft"
    assert any(prismatic.values())


def test_sources_may_be_mixed_within_one_arm():
    """A single source per arm forbade franka+yam hybrids entirely. Mixing is admissible because
    contact does not depend on the two sides of a boundary agreeing -- see
    test_sampled_geometry_is_connected."""
    from collections import Counter

    kinds = Counter()
    for seed in range(2000):
        arm = sample_params(seed).arm
        used = {link.geom.source for link in arm.links if link.geom.mode == "pool"}
        if arm.base.geom.mode == "pool":
            used.add(arm.base.geom.source)
        kinds["none" if not used else ("mixed" if len(used) > 1 else "single")] += 1
    assert kinds["mixed"] > 0.10 * sum(kinds.values()), kinds
    assert kinds["single"] > 0.10 * sum(kinds.values()), kinds
    assert kinds["none"] > 0.05 * sum(kinds.values()), kinds


def _capsule_span(model, data, geom):
    """A capsule's two endpoints in world coordinates, or None for any other geom type."""
    if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
        return None
    half = float(model.geom_size[geom][1])
    axis = np.array(data.geom_xmat[geom], float).reshape(3, 3)[:, 2]   # xmat is on DATA, not MODEL
    centre = np.array(data.geom_xpos[geom], float)                     # xpos is on DATA too
    return centre - axis * half, centre + axis * half


def test_no_body_duplicates_its_parents_span():
    """No child may carry geometry covering the same segment its parent already covers.

    This is the "extra limb" defect, and it is a DUPLICATION rather than a gap -- which is why every
    other check here passed while it was present. Each primitive link drew a collar back to its
    parent's origin while the parent drew a bone forward to that same point. Identical at qpos=0, but
    the collar belongs to the child and so rotates with the joint: turn it and the two diverge into a
    V sprouting from every joint.

    Detected structurally, by comparing capsule endpoints, so any future variant is caught whatever
    it is named. Reaching a borrowed neighbour is the repair pass's job, and it emits a strut only
    where a gap is measured -- so nothing legitimately duplicates a span."""
    from molmo_spaces.embodiment_gen.builder import link_name

    for seed in range(60):
        params = sample_params(seed)
        try:
            _, model = build(params)
        except Exception:
            continue
        data = mujoco.MjData(model)
        data.qpos[:] = 0.0
        mujoco.mj_forward(model, data)

        chain = ["base_link"] + [link_name(i) for i in range(1, params.arm.dof + 1)]
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in chain]
        geoms = {b: [g for g in range(model.ngeom) if model.geom_bodyid[g] == b] for b in ids}

        for parent, child in zip(ids, ids[1:]):
            for a in geoms[parent]:
                span_a = _capsule_span(model, data, a)
                if span_a is None:
                    continue
                for b in geoms[child]:
                    span_b = _capsule_span(model, data, b)
                    if span_b is None:
                        continue
                    same = (max(np.linalg.norm(span_a[0] - span_b[0]),
                                np.linalg.norm(span_a[1] - span_b[1])) < 2e-3
                            or max(np.linalg.norm(span_a[0] - span_b[1]),
                                   np.linalg.norm(span_a[1] - span_b[0])) < 2e-3)
                    if same and not all("strut" in (mujoco.mj_id2name(
                            model, mujoco.mjtObj.mjOBJ_GEOM, g) or "") for g in (a, b)):
                        raise AssertionError(
                            f"seed {seed}: "
                            f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, a)} and "
                            f"{mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, b)} "
                            f"cover the same segment")


def test_primitive_links_carry_only_a_bone_a_hub_and_struts():
    """Pins the primitive construction, so reintroducing a redundant capsule fails loudly.

    A primitive link is a bone toward its child plus a housing at its own origin. Anything else is
    either a strut the repair pass measured a need for, or a regression."""
    from molmo_spaces.embodiment_gen.builder import link_name

    allowed_suffixes = ("_bone", "_hub", "_strut_in", "_strut_out", "_strut_reach", "_strut_tool")
    for seed in range(60):
        params = sample_params(seed)
        try:
            _, model = build(params)
        except Exception:
            continue
        for index, link in enumerate(params.arm.links, start=1):
            if link.geom.mode != "primitive":
                continue
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, link_name(index))
            names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
                     for g in range(model.ngeom) if model.geom_bodyid[g] == bid]
            for name in names:
                assert name.endswith(allowed_suffixes), f"seed {seed}: unexpected geom {name}"
            bones = [n for n in names if n.endswith("_bone")]
            assert len(bones) <= 1, f"seed {seed}: {len(bones)} bones on {link_name(index)}"


def _camera_system(name: str):
    from molmo_spaces.configs import camera_configs

    return getattr(camera_configs, name)()


def test_generic_camera_system_resolves_on_every_embodiment():
    """Every camera GenericOmniPurposeCameraSystem names must exist in the model it will run on.

    An MjcfCameraConfig resolves its camera by NAME out of the compiled scene, so a name the
    embodiment does not declare is not a degraded stream -- it is a hard failure at scene setup, and
    the previous generic system avoided it only by declaring no robot-mounted camera at all. That is
    what left generated data exocentric-only while the shipped Franka data is multi-view.

    Checked on generated arms AND on the two witnesses, because the same config has to serve both.
    """
    from molmo_spaces.embodiment_gen.constants import WRIST_CAMERA

    system = _camera_system("GenericOmniPurposeCameraSystem")
    mjcf_cameras = [c for c in system.cameras if hasattr(c, "mjcf_name")]
    mounted = [c for c in system.cameras if hasattr(c, "reference_body_names")]
    assert mjcf_cameras and mounted, "the system lost its robot-mounted cameras"

    def declared(model, objtype, count):
        return {mujoco.mj_id2name(model, objtype, i) or "" for i in range(count)}

    builds = [(f"seed{seed}", build(sample_params(seed), wrist_camera=True)[1])
              for seed in (2, 3, 5, 11, 29)]
    builds += [(source, build(reference.params_of(source), wrist_camera=True)[1])
               for source in SOURCE_NAMES]

    for label, model in builds:
        cameras = declared(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam)
        bodies = declared(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody)
        # Both fields resolve as "the first candidate that exists", so ONE has to match -- not all.
        # The aliases exist precisely so that a name meaningful to another robot may be absent here.
        for camera in mjcf_cameras:
            # the config carries the scene namespace ("robot_0/"); the model is the bare robot
            candidates = {name.split("/", 1)[-1] if name.startswith("robot_0/") else name
                          for name in camera.candidate_mjcf_names()}
            assert candidates & cameras, (
                f"{label}: config names {sorted(candidates)}, model has {sorted(cameras)}")
            assert WRIST_CAMERA in candidates, (
                f"the wrist camera name is a contract between builder and config: "
                f"{WRIST_CAMERA} missing from {sorted(candidates)}")
            # ...and on a GENERATED robot the canonical name must be the one that actually resolves,
            # or the builder's derived pose is being silently bypassed for some alias
            if label.startswith("seed"):
                assert WRIST_CAMERA in cameras, f"{label}: {WRIST_CAMERA} was not emitted"
        for camera in mounted:
            bare = {body.split("/", 1)[-1] for body in camera.reference_body_names}
            assert bare & bodies, (
                f"{label}: config mounts {camera.name} on one of {sorted(bare)}, "
                f"none of which the model has")


def test_generic_camera_system_follows_franka():
    """The generic system must reproduce Franka's view composition, not invent its own.

    The point of the generic system is that a generated robot yields the same KIND of data the
    shipped robot does -- same camera count, same roles, same exocentric distribution -- so a model
    trained across embodiments is not also learning which robot it is from the camera rig. Anything
    that differs here is a difference a downstream consumer has to know about, so each one is
    asserted deliberately rather than left to drift."""
    generic = _camera_system("GenericOmniPurposeCameraSystem")
    franka = _camera_system("FrankaOmniPurposeCameraSystem")

    assert generic.img_resolution == franka.img_resolution
    kinds = lambda system: [type(c).__name__ for c in system.cameras]
    assert kinds(generic) == kinds(franka), f"{kinds(generic)} vs {kinds(franka)}"

    by_kind = lambda system, kind: [c for c in system.cameras if type(c).__name__ == kind]

    # exocentric cameras: identical, so generated and shipped data share one exo distribution
    for ours, theirs in zip(by_kind(generic, "RandomizedExocentricCameraConfig"),
                            by_kind(franka, "RandomizedExocentricCameraConfig")):
        assert ours == theirs, f"{ours.name} diverges from {theirs.name}"

    # wrist camera: the name of the MJCF camera must differ (ours is the uniform one, theirs is
    # Franka's gripper) but every randomization knob must match
    ours, theirs = by_kind(generic, "MjcfCameraConfig")[0], by_kind(franka, "MjcfCameraConfig")[0]
    assert ours.mjcf_name != theirs.mjcf_name
    for field in ("pos_noise_range", "orientation_noise_degrees", "record_depth",
                  "robot_namespace"):
        assert getattr(ours, field) == getattr(theirs, field), field
    # ...but not its field, deliberately: ours renders each robot at its own camera's fovy (a
    # generated robot's gripper catalog entry, test_wrist_camera_renders_at_its_grippers_fovy),
    # jittered twice as wide; Franka's at a fixed fov
    assert ours.fov is None and theirs.fov is not None
    assert ours.fov_noise_degrees == (-8.0, 8.0) and theirs.fov_noise_degrees == (-4.0, 4.0)

    # shoulder camera: same offset, fov and jitter as the calibrated DROID rig. Only the aim differs
    # -- Franka's quaternion points at the DROID room, not at a generated arm's workspace.
    ours, theirs = by_kind(generic, "RobotMountedCameraConfig")[0], \
        by_kind(franka, "RobotMountedCameraConfig")[0]
    np.testing.assert_allclose(ours.camera_offset, theirs.camera_offset, atol=1e-9)
    assert ours.fov == theirs.fov
    assert ours.pos_noise_range == theirs.pos_noise_range
    assert ours.orientation_noise_degrees == theirs.orientation_noise_degrees

    # the aim must be a QUATERNION, not lookat_offset: camera_manager passes fov and applies
    # orientation noise only on the quaternion branch, and freeze_task_config calls
    # list(cam.camera_quaternion) unconditionally, so a lookat-only camera silently loses both
    # randomizations and then kills the episode at freeze time
    assert ours.camera_quaternion is not None, "the shoulder camera must declare a quaternion"

    # and that quaternion must actually point at the workspace, to the precision of the derivation
    from molmo_spaces.configs.camera_configs import SHOULDER_LOOKAT, SHOULDER_OFFSET

    matrix = quat_to_mat(tuple(ours.camera_quaternion))
    view = -matrix[:, 2]                                   # MuJoCo cameras look along local -z
    want = np.asarray(SHOULDER_LOOKAT, float) - np.asarray(SHOULDER_OFFSET, float)
    want /= np.linalg.norm(want)
    np.testing.assert_allclose(view, want, atol=1e-7)
    assert float(matrix[:, 1] @ np.array([0.0, 0.0, 1.0])) > 0.0, "the shoulder camera is upside down"

    # ...and it must point where FRANKA's calibrated quaternion points, to within the lateral residual
    # deliberately dropped from the aim point. This is the assertion that the generic shoulder camera
    # is Franka's camera re-derived rather than a new camera that happens to compile.
    theirs_view = -quat_to_mat(tuple(theirs.camera_quaternion))[:, 2]
    separation = np.degrees(np.arccos(np.clip(float(view @ theirs_view), -1.0, 1.0)))
    assert separation < 6.0, f"generic shoulder aim is {separation:.1f} deg off Franka's"


@pytest.mark.parametrize("seed", [2, 3, 5, 11, 29])
def test_wrist_camera_carries_the_designed_view(seed):
    """The wrist stream shows each gripper's DESIGNED view, not a derived aim-at-grasp one.

    Every catalog gripper has a designed wrist pose -- robotiq and rum ship a camera, yam_jaw has
    the vendor's D405 mount -- and those tilt deliberately AHEAD of the jaws (15.5 deg for
    robotiq, 25.7 deg for yam). The earlier derived aim pointed exactly at the grasp point, which
    filled the frame with the gripper's own hardware and put every stream 15-26 deg off the view
    the shipped robots record. So the invariant is: the designed pose, verbatim -- position at the
    catalog's measured offset from the planner grasp frame, optical axis within the designed tilt
    of the approach, and, where the source model ships the camera itself, exact agreement with it."""
    from molmo_spaces.embodiment_gen.builder import _grasp_frame
    from molmo_spaces.embodiment_gen.constants import WRIST_CAMERA
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    params = sample_params(seed)
    _, model = build(params, wrist_camera=True)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    entry = load_catalog()[params.gripper].wrist_camera
    frame = _grasp_frame(load_catalog()[params.gripper], model)
    assert frame is not None, f"{params.gripper}: no grasp frame to anchor to"
    host, local, rotation = frame
    body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, host)
    assert body >= 0, host
    body_rotation = np.asarray(data.xmat[body], float).reshape(3, 3)
    grasp = body_rotation @ local + np.asarray(data.xpos[body])

    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
    assert camera >= 0, f"{params.gripper}: no {WRIST_CAMERA} was declared"
    position = np.asarray(data.cam_xpos[camera], float)
    view = -np.asarray(data.cam_xmat[camera], float).reshape(3, 3)[:, 2]   # cameras look along -z

    expected = grasp + body_rotation @ (np.asarray(rotation, float) @ np.asarray(entry["pos"], float))
    assert np.linalg.norm(position - expected) < 1.0e-6, \
        f"{params.gripper}: camera is not at the catalog's designed offset"

    approach = body_rotation @ (np.asarray(rotation, float) @ [0.0, 0.0, 1.0])
    # each catalog entry records its designed aim (cos of the tilt): robotiq 0.94, yam's vendor
    # D405 mount 0.64 -- a 50 deg over-the-shoulder view is a design fact, not an error, so the
    # assertion is agreement with the RECORDED design rather than a universal tilt bound
    assert float(view @ approach) == pytest.approx(entry["aim"], abs=0.05), \
        f"{params.gripper}: view is " \
        f"{np.degrees(np.arccos(np.clip(float(view @ approach), -1, 1))):.1f} deg off the " \
        f"approach axis, but the design says {np.degrees(np.arccos(entry['aim'])):.1f}"
    assert float(model.cam_fovy[camera]) == pytest.approx(entry["fovy"], abs=1e-3)

    # where the SOURCE model ships the camera itself, the declared one must match it exactly
    for shipped_name in ("gripper/wrist_camera", "gripper/wrist_cam"):
        shipped = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, shipped_name)
        if shipped >= 0:
            assert np.linalg.norm(position - data.cam_xpos[shipped]) < 1.0e-6
            shipped_view = -np.asarray(data.cam_xmat[shipped], float).reshape(3, 3)[:, 2]
            assert float(view @ shipped_view) > 1.0 - 1.0e-9, \
                f"{params.gripper}: declared camera diverges from the shipped {shipped_name}"


def _params_with(gripper: str):
    """A robot wearing ``gripper``: the first sampled one; yam_jaw, which the sampler never draws,
    on the reference robot that ships it."""
    if gripper == "yam_jaw":
        return reference.params_of("i2rt_yam")
    return next(params for params in map(sample_params, range(500)) if params.gripper == gripper)


@pytest.mark.parametrize("gripper",
                         ["linear_4310", "lite6_gripper_wide", "robotiq_2f85", "rum", "yam_jaw"])
def test_wrist_camera_renders_at_its_grippers_fovy(gripper):
    """The generic wrist camera renders each robot at its own camera's field, jittered around it.

    The config names no fov, so the camera manager takes the camera's fovy -- for a generated
    robot, its gripper catalog entry's ``wrist_camera.fovy``, which the builder gives the camera
    (lite6_gripper_wide's is the deployed robot's view) -- and fov_noise_degrees jitters it."""
    from molmo_spaces.embodiment_gen.constants import WRIST_CAMERA
    from molmo_spaces.embodiment_gen.grippers import load_catalog
    from molmo_spaces.env.camera_manager import CameraManager

    generic = next(c for c in _camera_system("GenericOmniPurposeCameraSystem").cameras
                   if type(c).__name__ == "MjcfCameraConfig")
    assert generic.fov is None, "a fixed fov would override every gripper's own"
    fovy = load_catalog()[gripper].wrist_camera["fovy"]
    _, model = build(_params_with(gripper), wrist_camera=True)
    camera = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, WRIST_CAMERA)
    assert float(model.cam_fovy[camera]) == pytest.approx(fovy)
    low, high = generic.fov_noise_degrees
    fovs = [CameraManager.apply_mjcf_camera_noise(model.cam_pos[camera].copy(),
                                                  model.cam_quat[camera].copy(), fovy, generic,
                                                  np.random.RandomState(seed))[2]
            for seed in range(200)]
    assert fovy + low <= min(fovs) and max(fovs) <= fovy + high
    assert max(fovs) - min(fovs) > 0.9 * (high - low), "the fov jitter no longer spans its range"


def test_wrist_camera_reads_the_declared_frame_not_the_shipped_site():
    """yam_jaw is the case that distinguishes them, so assert on it by name rather than by luck."""
    from molmo_spaces.embodiment_gen.builder import _grasp_frame
    from molmo_spaces.embodiment_gen.grippers import load_catalog

    params = reference.params_of("i2rt_yam")
    _, model = build(params, wrist_camera=True)
    host, local, _ = _grasp_frame(load_catalog()["yam_jaw"], model)

    shipped = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper/grasp_site")
    assert shipped >= 0, "yam_jaw stopped shipping grasp_site; this test no longer distinguishes"
    gap = float(np.linalg.norm(local - np.asarray(model.site_pos[shipped], float)))
    assert gap == pytest.approx(0.0497, abs=1.0e-4), (
        f"the planner frame is {gap * 1000:.1f} mm from the shipped site, expected 49.7")
    assert host == mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY,
                                     int(model.site_bodyid[shipped]))


def test_camera_seed_is_reproducible_per_embodiment():
    """A camera seed has to give the same views for the same robot, different views per episode, and
    must leave every shipped camera system on the global stream it uses today."""
    from molmo_spaces.configs.camera_configs import (
        FrankaOmniPurposeCameraSystem,
        GenericOmniPurposeCameraSystem,
    )
    from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler

    class Sampler:                                     # only what camera_rng reads
        camera_rng = BaseMujocoTaskSampler.camera_rng
        embodiment_camera_seed = BaseMujocoTaskSampler.embodiment_camera_seed

        def __init__(self, embodiment_id, episode_seed):
            robot = type("Robot", (), {})()
            robot.embodiment = (type("Spec", (), {"embodiment_id": embodiment_id})()
                                if embodiment_id else None)
            robot.name = "" if embodiment_id else "franka"
            self.config = type("Config", (), {"robot_config": robot})()
            self.current_seed = episode_seed

    generic, franka = GenericOmniPurposeCameraSystem(), FrankaOmniPurposeCameraSystem()
    assert generic.camera_seed == "embodiment", "the generic system must seed from the embodiment"
    assert franka.camera_seed is None, "a shipped system must stay on the global stream"

    draw = lambda eid, ep: Sampler(eid, ep).camera_rng(generic).uniform(-1.0, 1.0, 8)
    np.testing.assert_array_equal(draw("gen_2198", 7), draw("gen_2198", 7))
    assert not np.array_equal(draw("gen_2198", 7), draw("gen_2198", 8)), "cameras froze across episodes"
    assert not np.array_equal(draw("gen_2198", 7), draw("gen_6459", 7)), "two robots share a draw"

    # and the seed must not depend on the process, or "reproducible" means only "within one run"
    assert Sampler("gen_2198", 0).embodiment_camera_seed() == 1694133868

    np.random.seed(4321)
    expected = np.random.uniform(-1.0, 1.0, 8)
    np.random.seed(4321)
    np.testing.assert_array_equal(Sampler(None, 7).camera_rng(franka).uniform(-1.0, 1.0, 8), expected)
