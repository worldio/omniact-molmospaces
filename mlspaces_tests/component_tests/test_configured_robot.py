from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import pytest
from pydantic import ValidationError

from molmo_spaces.configs.robot_configs import I2rtYamRobotConfig
from molmo_spaces.robots.configured import (
    ConfiguredRobot,
    ConfiguredRobotConfig,
    EmbodimentSpec,
    load_embodiment_spec,
)
from molmo_spaces.robots.configured.assembler import MjcfAssembler
from molmo_spaces.robots.configured.robot_view import ConfiguredRobotView
from molmo_spaces.robots.i2rt_yam import I2rtYamRobot
from molmo_spaces.robots.robot_views.i2rt_yam_view import I2rtYamRobotView

SPEC_PATH = Path(__file__).parents[2] / "docs" / "embodiment_specs" / "i2rt_yam.yaml"


def _configured_spec_and_model(
    config: ConfiguredRobotConfig,
    *,
    prefix: str = "robot_0/",
    pos: list[float] | None = None,
    strip_meshes: bool = False,
) -> tuple[mujoco.MjSpec, mujoco.MjModel]:
    scene = mujoco.MjSpec()
    position = [0.2, -0.3] if pos is None else pos
    if prefix == config.robot_namespace:
        ConfiguredRobot.add_robot_to_scene(
            config,
            scene,
            prefix=prefix,
            pos=position,
            quat=[1.0, 0.0, 0.0, 0.0],
            strip_meshes=strip_meshes,
        )
    else:
        assembler = MjcfAssembler(config.embodiment)
        assembler.add_robot_to_scene(
            scene,
            prefix=prefix,
            pos=position,
            quat=[1.0, 0.0, 0.0, 0.0],
            strip_meshes=strip_meshes,
        )
        assembler.apply_control_overrides(scene, prefix)
    return scene, scene.compile()


def _legacy_model() -> mujoco.MjModel:
    config = I2rtYamRobotConfig()
    scene = mujoco.MjSpec()
    I2rtYamRobot.add_robot_to_scene(
        config,
        scene,
        prefix=config.robot_namespace,
        pos=[0.2, -0.3],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    I2rtYamRobot.apply_control_overrides(scene, config)
    return scene.compile()


@pytest.fixture(scope="module")
def embodiment() -> EmbodimentSpec:
    return load_embodiment_spec(SPEC_PATH)


@pytest.fixture(scope="module")
def configured_config(embodiment: EmbodimentSpec) -> ConfiguredRobotConfig:
    return ConfiguredRobotConfig(embodiment=embodiment)


def test_embodiment_schema_is_strict_and_cross_referenced(embodiment: EmbodimentSpec):
    assert embodiment.schema_version == 1
    assert embodiment.robot_view.base_group == "base"
    assert (
        embodiment.robot_view.end_effectors["primary"].grasp_profile_id in embodiment.grasp_profiles
    )

    invalid = embodiment.model_dump()
    invalid["unexpected_factory"] = "i2rt_yam"
    with pytest.raises(ValidationError, match="unexpected_factory"):
        EmbodimentSpec.model_validate(invalid)


def test_config_adapter_rehydrates_existing_runtime_hooks(configured_config: ConfiguredRobotConfig):
    assert configured_config.robot_cls is ConfiguredRobot
    assert configured_config.robot_factory is ConfiguredRobot
    assert configured_config.robot_namespace == "robot_0/"
    assert configured_config.name == "i2rt_yam"
    assert configured_config.get_robot_xml_path().name == "yam.xml"
    assert configured_config.base_placement_forward_offset == 0.0
    assert I2rtYamRobotConfig().base_placement_forward_offset == 0.0

    payload = configured_config.model_dump()
    payload["robot_cls"] = None
    payload["robot_factory"] = None
    payload["robot_view_factory"] = None
    restored = ConfiguredRobotConfig.model_validate(payload)
    assert restored.robot_cls is ConfiguredRobot
    assert restored.robot_factory is ConfiguredRobot
    assert restored.robot_view_factory is not None


def test_configured_yam_compiled_model_matches_legacy(configured_config: ConfiguredRobotConfig):
    legacy = _legacy_model()
    _, configured = _configured_spec_and_model(configured_config)

    for dimension in ("nbody", "njnt", "nq", "nv", "nu", "nmocap", "ngeom", "nsite"):
        assert getattr(configured, dimension) == getattr(legacy, dimension)

    accessors = (
        ("body", "nbody"),
        ("joint", "njnt"),
        ("actuator", "nu"),
        ("site", "nsite"),
    )
    for accessor, dimension in accessors:
        legacy_names = [
            getattr(legacy, accessor)(index).name for index in range(getattr(legacy, dimension))
        ]
        configured_names = [
            getattr(configured, accessor)(index).name
            for index in range(getattr(configured, dimension))
        ]
        assert configured_names == legacy_names

    np.testing.assert_allclose(configured.actuator_gainprm, legacy.actuator_gainprm)
    np.testing.assert_allclose(configured.actuator_biasprm, legacy.actuator_biasprm)
    np.testing.assert_allclose(configured.actuator_forcerange, legacy.actuator_forcerange)
    np.testing.assert_array_equal(configured.actuator_forcelimited, legacy.actuator_forcelimited)
    np.testing.assert_allclose(configured.body("robot_0/base").pos, [0.2, -0.3, 0.0])

    np.testing.assert_allclose(configured.body_gravcomp, legacy.body_gravcomp)


def test_configured_view_matches_yam_group_behavior(
    configured_config: ConfiguredRobotConfig,
    embodiment: EmbodimentSpec,
):
    _, model = _configured_spec_and_model(configured_config)
    data = mujoco.MjData(model)
    configured = ConfiguredRobotView(
        data,
        configured_config.robot_namespace,
        view_spec=embodiment.robot_view,
        robot_name="i2rt_yam",
    )
    legacy = I2rtYamRobotView(data, configured_config.robot_namespace)
    mujoco.mj_forward(model, data)

    assert configured.move_group_ids() == legacy.move_group_ids()
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

    gripper = configured.get_gripper("gripper")
    gripper.joint_pos = np.array([-0.002, 0.002])
    assert gripper.inter_finger_dist == 0.0
    assert gripper.inter_finger_dist_range == (0.0, 0.075)
    gripper.set_gripper_ctrl_open(True)
    np.testing.assert_allclose(gripper.ctrl, [0.041])
    gripper.set_gripper_ctrl_open(False)
    np.testing.assert_allclose(gripper.ctrl, [0.0])

    assert configured.end_effector_ids() == ["primary"]
    assert configured.primary_end_effector.spec.tool_group == "gripper"


def test_assembler_call_time_prefix_and_mesh_stripping(
    configured_config: ConfiguredRobotConfig,
):
    _, model = _configured_spec_and_model(
        configured_config,
        prefix="alternate/",
        pos=[1.0, 2.0, 0.5],
        strip_meshes=True,
    )
    np.testing.assert_allclose(model.body("alternate/base").pos, [1.0, 2.0, 0.5])
    gripper_id = model.actuator("alternate/gripper").id
    assert model.actuator_gainprm[gripper_id, 0] == 500.0
    assert all(geom_type != mujoco.mjtGeom.mjGEOM_MESH for geom_type in model.geom_type)


def test_configured_robot_uses_profile_controllers_and_sensors(
    configured_config: ConfiguredRobotConfig,
):
    _, model = _configured_spec_and_model(configured_config)
    data = mujoco.MjData(model)
    online_kinematics = object()
    parallel_kinematics = object()
    experiment = SimpleNamespace(robot_config=configured_config)

    with patch(
        "molmo_spaces.robots.configured.robot.build_kinematics",
        return_value=(online_kinematics, parallel_kinematics),
    ):
        robot = ConfiguredRobot(data, experiment)

    robot.reset()
    assert set(robot.controllers) == {"arm", "gripper"}
    assert robot.get_arm_move_group_ids() == ["arm"]
    assert [sensor.uuid for sensor in robot.create_robot_sensors()] == ["tcp_pose"]
    assert robot.kinematics is online_kinematics
    assert robot.parallel_kinematics is parallel_kinematics
    np.testing.assert_allclose(
        robot.robot_view.get_move_group("arm").joint_pos,
        configured_config.init_qpos["arm"],
    )
