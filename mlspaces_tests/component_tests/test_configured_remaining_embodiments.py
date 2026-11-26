from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from molmo_spaces.configs.robot_configs import (
    BimanualYamRobotConfig,
    FloatingRobotiq2f85RobotConfig,
    FloatingRUMRobotConfig,
    FrankaCAPRobotConfig,
    MobileFrankaRobotConfig,
    RBY1Config,
    RBY1MConfig,
    RBY1MOpenCloseConfig,
)
from molmo_spaces.env.sensors import _cmd_joint_pos
from molmo_spaces.robots.bimanual_yam import BimanualYamRobot
from molmo_spaces.robots.configured import ConfiguredRobot, ConfiguredRobotConfig
from molmo_spaces.robots.floating_robotiq import FloatingRobotiqRobot
from molmo_spaces.robots.floating_rum import FloatingRUMRobot
from molmo_spaces.robots.franka import FrankaRobot
from molmo_spaces.robots.mobile_franka import MobileFrankaRobot
from molmo_spaces.robots.rby1 import RBY1

SPEC_DIR = Path(__file__).parents[2] / "docs" / "embodiment_specs"


@dataclass(frozen=True)
class EmbodimentCase:
    id: str
    spec_file: str
    legacy_config: type
    legacy_robot: type
    position: tuple[float, ...]
    quaternion: tuple[float, float, float, float]
    dimensions: tuple[int, int, int, int, int, int, int, int]
    controller_ids: frozenset[str]
    sensor_ids: tuple[str, ...]


CASES = (
    EmbodimentCase(
        "franka_cap",
        "franka_cap.yaml",
        FrankaCAPRobotConfig,
        FrankaRobot,
        (0.2, -0.3),
        (1.0, 0.0, 0.0, 0.0),
        (16, 9, 9, 8, 1, 60, 3, 1),
        frozenset({"arm", "gripper"}),
        ("tcp_pose",),
    ),
    EmbodimentCase(
        "mobile_franka",
        "mobile_franka_droid.yaml",
        MobileFrankaRobotConfig,
        MobileFrankaRobot,
        (0.2, -0.3),
        (0.9887710779360422, 0.0, 0.0, 0.14943813247359922),
        (22, 16, 16, 11, 0, 65, 4, 2),
        frozenset({"base", "arm", "gripper"}),
        ("tcp_pose",),
    ),
    EmbodimentCase(
        "floating_rum",
        "floating_rum.yaml",
        FloatingRUMRobotConfig,
        FloatingRUMRobot,
        (0.1, 0.2, 0.3),
        (1.0, 0.0, 0.0, 0.0),
        (8, 3, 9, 1, 1, 24, 2, 1),
        frozenset({"base", "gripper"}),
        ("tcp_pose",),
    ),
    EmbodimentCase(
        "floating_robotiq",
        "floating_robotiq.yaml",
        FloatingRobotiq2f85RobotConfig,
        FloatingRobotiqRobot,
        (0.1, 0.2, 0.3),
        (1.0, 0.0, 0.0, 0.0),
        (13, 7, 13, 1, 1, 28, 1, 1),
        frozenset({"base", "gripper"}),
        ("tcp_pose",),
    ),
    EmbodimentCase(
        "bimanual_yam",
        "bimanual_yam.yaml",
        BimanualYamRobotConfig,
        BimanualYamRobot,
        (0.2, -0.3),
        (1.0, 0.0, 0.0, 0.0),
        (29, 16, 16, 14, 1, 109, 4, 0),
        frozenset({"left_arm", "right_arm", "left_gripper", "right_gripper"}),
        ("tcp_pose_left", "tcp_pose_right"),
    ),
    EmbodimentCase(
        "rby1",
        "rby1.yaml",
        RBY1Config,
        RBY1,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (34, 29, 29, 27, 0, 1124, 4, 6),
        frozenset({"base", "torso", "left_arm", "right_arm", "left_gripper", "right_gripper"}),
        ("rby1_left_grasp_state", "rby1_right_grasp_state"),
    ),
    EmbodimentCase(
        "rby1m",
        "rby1m.yaml",
        RBY1MConfig,
        RBY1,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (36, 29, 29, 27, 0, 1021, 4, 6),
        frozenset({"base", "torso", "left_arm", "right_arm", "left_gripper", "right_gripper"}),
        ("rby1_left_grasp_state", "rby1_right_grasp_state"),
    ),
    EmbodimentCase(
        "rby1m_open_close",
        "rby1m_open_close.yaml",
        RBY1MOpenCloseConfig,
        RBY1,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0, 0.0),
        (36, 29, 29, 27, 0, 1021, 4, 6),
        frozenset({"base", "torso", "left_arm", "right_arm", "left_gripper", "right_gripper"}),
        ("rby1_left_grasp_state", "rby1_right_grasp_state"),
    ),
)


def _assemble_legacy(case: EmbodimentCase):
    config = case.legacy_config()
    scene = mujoco.MjSpec()
    case.legacy_robot.add_robot_to_scene(
        config,
        scene,
        prefix=config.robot_namespace,
        pos=list(case.position),
        quat=list(case.quaternion),
    )
    case.legacy_robot.apply_control_overrides(scene, config)
    return scene.compile()


def _assemble_configured(case: EmbodimentCase):
    config = ConfiguredRobotConfig.from_file(SPEC_DIR / case.spec_file)
    scene = mujoco.MjSpec()
    ConfiguredRobot.add_robot_to_scene(
        config,
        scene,
        prefix=config.robot_namespace,
        pos=list(case.position),
        quat=list(case.quaternion),
    )
    return config, scene.compile()


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_remaining_embodiment_compiled_models_match_legacy(case: EmbodimentCase):
    legacy = _assemble_legacy(case)
    _, configured = _assemble_configured(case)

    actual_dimensions = (
        configured.nbody,
        configured.njnt,
        configured.nq,
        configured.nu,
        configured.nmocap,
        configured.ngeom,
        configured.nsite,
        configured.ncam,
    )
    assert actual_dimensions == case.dimensions
    assert actual_dimensions == (
        legacy.nbody,
        legacy.njnt,
        legacy.nq,
        legacy.nu,
        legacy.nmocap,
        legacy.ngeom,
        legacy.nsite,
        legacy.ncam,
    )

    for accessor, dimension in (
        ("body", "nbody"),
        ("joint", "njnt"),
        ("actuator", "nu"),
        ("site", "nsite"),
        ("camera", "ncam"),
        ("material", "nmat"),
        ("texture", "ntex"),
    ):
        assert [
            getattr(configured, accessor)(index).name
            for index in range(getattr(configured, dimension))
        ] == [getattr(legacy, accessor)(index).name for index in range(getattr(legacy, dimension))]

    for array_name in (
        "body_pos",
        "body_quat",
        "body_mass",
        "body_inertia",
        "body_gravcomp",
        "jnt_type",
        "jnt_axis",
        "jnt_pos",
        "jnt_range",
        "qpos0",
        "actuator_trntype",
        "actuator_trnid",
        "actuator_gear",
        "actuator_gainprm",
        "actuator_biasprm",
        "actuator_ctrlrange",
        "actuator_forcerange",
        "geom_type",
        "geom_pos",
        "geom_quat",
        "geom_size",
        "geom_friction",
        "geom_solref",
        "geom_solimp",
        "geom_contype",
        "geom_conaffinity",
        "geom_matid",
    ):
        np.testing.assert_allclose(
            getattr(configured, array_name),
            getattr(legacy, array_name),
            equal_nan=True,
        )


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.id)
def test_remaining_embodiment_runtime_constructs_and_resets(case: EmbodimentCase):
    config, model = _assemble_configured(case)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = ConfiguredRobot(data, SimpleNamespace(robot_config=config))
    robot.reset()
    mujoco.mj_forward(model, data)

    assert frozenset(robot.controllers) == case.controller_ids
    assert tuple(sensor.uuid for sensor in robot.create_robot_sensors()) == case.sensor_ids
    assert robot.robot_view.move_group_ids()[0] == config.embodiment.robot_view.base_group
    for group_id in config.init_qpos:
        assert np.all(np.isfinite(robot.robot_view.get_move_group(group_id).joint_pos))
    robot.compute_control()


def test_floating_composite_base_controls_mocap_target():
    case = next(case for case in CASES if case.id == "floating_rum")
    config, model = _assemble_configured(case)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = ConfiguredRobot(data, SimpleNamespace(robot_config=config))
    robot.reset()

    assert _cmd_joint_pos(robot) == {}

    target = np.array([0.4, -0.2, 0.8, 1.0, 0.0, 0.0, 0.0])
    robot.update_control({"base": target, "gripper": np.array([-255.0])})
    robot.compute_control()
    np.testing.assert_allclose(robot.robot_view.base.ctrl, target)
    np.testing.assert_allclose(robot.robot_view.get_gripper("gripper").ctrl, [-255.0])

    qpos = robot.robot_view.get_qpos_dict()
    base_pose = robot.robot_view.base.pose
    desired_base = np.eye(4)
    desired_base[:3, 3] = [0.3, 0.1, 0.7]
    base_solution = robot.kinematics.ik("base", desired_base, ["base"], qpos, base_pose)
    np.testing.assert_allclose(base_solution["base"][:3], desired_base[:3, 3])

    desired_tcp = desired_base @ robot.robot_view.get_gripper("gripper").leaf_frame_to_robot
    gripper_solution = robot.kinematics.ik("gripper", desired_tcp, ["base"], qpos, base_pose)
    np.testing.assert_allclose(gripper_solution["base"], base_solution["base"], atol=1e-7)


def test_rby_open_close_profile_maps_coupled_state_and_torso_height():
    case = next(case for case in CASES if case.id == "rby1m_open_close")
    config, model = _assemble_configured(case)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    robot = ConfiguredRobot(data, SimpleNamespace(robot_config=config))
    robot.reset()

    np.testing.assert_allclose(
        robot.robot_view.get_gripper("left_gripper").joint_pos, [-0.05, 0.05]
    )
    np.testing.assert_allclose(robot.robot_view.get_move_group("head").ctrl, [0.0, 0.6])
    torso = robot.controllers["torso"]
    assert torso.max_height == 0.738
    torso.set_target(np.array([0.2]))
    np.testing.assert_allclose(torso.compute_ctrl_inputs(), [0.0, 0.2, -0.4, 0.2, 0.0, 0.0])
