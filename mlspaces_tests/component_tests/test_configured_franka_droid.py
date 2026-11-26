import random
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.robots.configured import ConfiguredRobot, ConfiguredRobotConfig
from molmo_spaces.robots.configured.robot_view import ConfiguredRobotView
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.robot_views.franka_droid_view import FrankaDroidRobotView

SPEC_PATH = Path(__file__).parents[2] / "docs" / "embodiment_specs" / "franka_droid.yaml"


def _configured_model(
    config: ConfiguredRobotConfig,
    *,
    strip_meshes: bool = False,
    randomize_textures: bool = False,
):
    scene = mujoco.MjSpec()
    ConfiguredRobot.add_robot_to_scene(
        config,
        scene,
        prefix=config.robot_namespace,
        pos=[0.2, -0.3],
        quat=[1.0, 0.0, 0.0, 0.0],
        strip_meshes=strip_meshes,
        randomize_textures=randomize_textures,
    )
    return scene.compile()


def _legacy_model(*, strip_meshes: bool = False, randomize_textures: bool = False):
    config = FrankaRobotConfig()
    scene = mujoco.MjSpec()
    FrankaRobot.add_robot_to_scene(
        config,
        scene,
        prefix=config.robot_namespace,
        pos=[0.2, -0.3],
        quat=[1.0, 0.0, 0.0, 0.0],
        strip_meshes=strip_meshes,
        randomize_textures=randomize_textures,
    )
    FrankaRobot.apply_control_overrides(scene, config)
    return scene.compile()


@pytest.fixture(scope="module")
def configured_config() -> ConfiguredRobotConfig:
    return ConfiguredRobotConfig.from_file(SPEC_PATH)


@pytest.fixture(scope="module")
def configured_model(configured_config: ConfiguredRobotConfig):
    return _configured_model(configured_config)


@pytest.fixture(scope="module")
def configured_runtime(configured_config: ConfiguredRobotConfig, configured_model):
    data = mujoco.MjData(configured_model)
    experiment = SimpleNamespace(robot_config=configured_config)
    robot = ConfiguredRobot(data, experiment)
    robot.reset()
    mujoco.mj_forward(configured_model, data)
    return robot


def test_droid_schema_compiles_with_legacy_structure(
    configured_config: ConfiguredRobotConfig,
    configured_model,
):
    legacy = _legacy_model()
    configured = configured_model

    assert configured_config.embodiment.embodiment_id == "franka_droid.single.fixed"
    for dimension in (
        "nbody",
        "njnt",
        "nq",
        "nv",
        "nu",
        "nmocap",
        "ngeom",
        "nsite",
        "nmat",
        "ntex",
        "nmesh",
        "neq",
        "ntendon",
    ):
        assert getattr(configured, dimension) == getattr(legacy, dimension)

    for accessor, dimension in (
        ("body", "nbody"),
        ("joint", "njnt"),
        ("actuator", "nu"),
        ("site", "nsite"),
        ("material", "nmat"),
        ("texture", "ntex"),
    ):
        assert [
            getattr(configured, accessor)(index).name
            for index in range(getattr(configured, dimension))
        ] == [getattr(legacy, accessor)(index).name for index in range(getattr(legacy, dimension))]

    # The configured platform is explicitly named; legacy leaves only that geom unnamed.
    configured_geom_names = [configured.geom(index).name for index in range(configured.ngeom)]
    legacy_geom_names = [legacy.geom(index).name for index in range(legacy.ngeom)]
    differing_geom_names = [
        (actual, expected)
        for actual, expected in zip(configured_geom_names, legacy_geom_names)
        if actual != expected
    ]
    assert differing_geom_names == [("robot_0/robot_base_geom", "")]

    for array_name in (
        "body_pos",
        "body_quat",
        "jnt_range",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "mat_rgba",
        "mat_texid",
        "tex_data",
    ):
        np.testing.assert_allclose(getattr(configured, array_name), getattr(legacy, array_name))

    np.testing.assert_allclose(configured.body("robot_0/base").pos, [0.2, -0.3, 0.0])
    np.testing.assert_allclose(configured.body_gravcomp, legacy.body_gravcomp)


def test_droid_configured_view_matches_legacy(
    configured_config: ConfiguredRobotConfig,
    configured_model,
):
    data = mujoco.MjData(configured_model)
    configured = ConfiguredRobotView(
        data,
        configured_config.robot_namespace,
        view_spec=configured_config.embodiment.robot_view,
        robot_name="franka_droid",
        grasp_profiles=configured_config.embodiment.grasp_profiles,
    )
    legacy = FrankaDroidRobotView(data, configured_config.robot_namespace)
    configured.set_qpos_dict(
        {group_id: np.asarray(values) for group_id, values in configured_config.init_qpos.items()}
    )
    mujoco.mj_forward(configured_model, data)

    assert configured.move_group_ids() == legacy.move_group_ids() == ["base", "arm", "gripper"]
    for group_id in configured.move_group_ids():
        actual = configured.get_move_group(group_id)
        expected = legacy.get_move_group(group_id)
        assert (actual.n_joints, actual.pos_dim, actual.vel_dim, actual.n_actuators) == (
            expected.n_joints,
            expected.pos_dim,
            expected.vel_dim,
            expected.n_actuators,
        )
        np.testing.assert_allclose(actual.joint_pos_limits, expected.joint_pos_limits)
        np.testing.assert_allclose(actual.root_frame_to_world, expected.root_frame_to_world)
        np.testing.assert_allclose(actual.leaf_frame_to_world, expected.leaf_frame_to_world)

    actual_gripper = configured.get_gripper("gripper")
    expected_gripper = legacy.get_gripper("gripper")
    assert actual_gripper.inter_finger_dist == pytest.approx(
        expected_gripper.inter_finger_dist, abs=1e-12
    )
    assert actual_gripper.inter_finger_dist_range == (0.0, 0.087)
    actual_gripper.set_gripper_ctrl_open(True)
    np.testing.assert_allclose(actual_gripper.ctrl, [0.0])
    actual_gripper.set_gripper_ctrl_open(False)
    np.testing.assert_allclose(actual_gripper.ctrl, [255.0])
    assert configured.primary_end_effector.spec.grasp_profile_id == "droid_robotiq_2f85_v1"


def test_droid_runtime_builds_real_controllers_sensors_and_kinematics(configured_runtime):
    robot = configured_runtime
    assert set(robot.controllers) == {"arm", "gripper"}
    assert robot.get_arm_move_group_ids() == ["arm"]
    assert [sensor.uuid for sensor in robot.create_robot_sensors()] == ["tcp_pose"]

    qpos = robot.robot_view.get_qpos_dict()
    base_pose = robot.robot_view.base.pose
    online_fk = robot.kinematics.fk(qpos, base_pose, rel_to_base=True)
    parallel_fk = robot.parallel_kinematics.fk(qpos, base_pose, rel_to_base=True)
    np.testing.assert_allclose(online_fk["arm"], parallel_fk["arm"], atol=2e-6)
    np.testing.assert_allclose(online_fk["gripper"], parallel_fk["gripper"], atol=2e-6)

    target = online_fk["arm"]
    assert robot.kinematics.ik("arm", target, None, qpos, base_pose, rel_to_base=True) is not None
    assert (
        robot.parallel_kinematics.ik("arm", target, None, qpos, base_pose, rel_to_base=True)
        is not None
    )
    legacy_unlocked_groups = robot.robot_view.move_group_ids()
    assert (
        robot.kinematics.ik(
            "arm",
            target,
            legacy_unlocked_groups,
            qpos,
            base_pose,
            rel_to_base=True,
        )
        is not None
    )
    assert (
        robot.parallel_kinematics.ik(
            "arm",
            target,
            legacy_unlocked_groups,
            qpos,
            base_pose,
            rel_to_base=True,
        )
        is not None
    )

    commanded_qpos = {
        group_id: controller.target_pos for group_id, controller in robot.controllers.items()
    }
    command_fk = robot.kinematics.fk(commanded_qpos, base_pose, rel_to_base=True)
    assert command_fk["gripper"].shape == (4, 4)


def test_droid_tcp_sensor_observation_matches_declared_space(configured_runtime):
    robot = configured_runtime
    sensor = robot.create_robot_sensors()[0]
    env = SimpleNamespace(mj_datas=[robot.mj_data], robots=[robot])
    observation = sensor.get_observation(env, task=None)
    assert observation.shape == (7,)
    assert sensor.observation_space.contains(observation)


def test_droid_mesh_stripping_matches_legacy(configured_config: ConfiguredRobotConfig):
    configured = _configured_model(configured_config, strip_meshes=True)
    legacy = _legacy_model(strip_meshes=True)
    assert configured.ngeom == legacy.ngeom == 5
    assert all(geom_type != mujoco.mjtGeom.mjGEOM_MESH for geom_type in configured.geom_type)


@pytest.mark.parametrize(
    ("seed", "expected_speckled"),
    [(0, False), (1, True)],
)
def test_droid_randomized_visuals_match_legacy_seed_for_seed(
    configured_config: ConfiguredRobotConfig,
    seed: int,
    expected_speckled: bool,
):
    random.seed(seed)
    np.random.seed(seed)
    legacy = _legacy_model(randomize_textures=True)
    legacy_rng_tail = (random.random(), np.random.random())

    random.seed(seed)
    np.random.seed(seed)
    configured = _configured_model(configured_config, randomize_textures=True)
    configured_rng_tail = (random.random(), np.random.random())

    assert configured_rng_tail == legacy_rng_tail
    for accessor, dimension in (("material", "nmat"), ("texture", "ntex")):
        assert [
            getattr(configured, accessor)(index).name
            for index in range(getattr(configured, dimension))
        ] == [getattr(legacy, accessor)(index).name for index in range(getattr(legacy, dimension))]
    assert (configured.ntex > 1) is expected_speckled
    np.testing.assert_array_equal(configured.tex_data, legacy.tex_data)
    np.testing.assert_array_equal(configured.mat_texid, legacy.mat_texid)
    np.testing.assert_allclose(configured.mat_rgba, legacy.mat_rgba)
    np.testing.assert_array_equal(configured.geom_matid, legacy.geom_matid)

    deterministic = _configured_model(configured_config)
    for array_name in (
        "body_mass",
        "body_inertia",
        "jnt_type",
        "jnt_range",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "geom_type",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
    ):
        np.testing.assert_allclose(
            getattr(configured, array_name), getattr(deterministic, array_name)
        )
