from unittest.mock import PropertyMock, patch

import mujoco
import numpy as np

from molmo_spaces.configs.robot_configs import I2rtYamRobotConfig
from molmo_spaces.robots.i2rt_yam import I2rtYamRobot
from molmo_spaces.robots.robot_views.i2rt_yam_view import I2rtYamGripperGroup


def test_yam_control_defaults():
    config = I2rtYamRobotConfig()
    assert config.K_stiffness == [80.0, 80.0, 80.0, 10.0, 10.0, 10.0]
    assert config.K_damping == [5.0, 5.0, 5.0, 1.5, 1.5, 1.5]
    assert config.gripper_effort_limit == 50.0
    assert config.gripper_position_gain == 500.0
    assert config.gripper_velocity_gain == 200.0


def test_control_overrides_are_compiled():
    config = I2rtYamRobotConfig(
        gripper_effort_limit=4.0,
        gripper_position_gain=400.0,
        gripper_velocity_gain=40.0,
    )
    spec = mujoco.MjSpec()
    I2rtYamRobot.add_robot_to_scene(
        config,
        spec,
        prefix=config.robot_namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    I2rtYamRobot.apply_control_overrides(spec, config)

    model = spec.compile()
    for index, (stiffness, damping, force_limit) in enumerate(
        zip(config.K_stiffness, config.K_damping, [28.0, 28.0, 28.0, 10.0, 10.0, 10.0]),
        start=1,
    ):
        arm_actuator_id = model.actuator(f"{config.robot_namespace}joint{index}").id
        assert model.actuator_gainprm[arm_actuator_id, 0] == stiffness
        assert model.actuator_biasprm[arm_actuator_id, 1] == -stiffness
        assert model.actuator_biasprm[arm_actuator_id, 2] == -damping
        np.testing.assert_array_equal(
            model.actuator_forcerange[arm_actuator_id], [-force_limit, force_limit]
        )

    actuator_id = model.actuator(f"{config.robot_namespace}gripper").id
    assert model.actuator_gainprm[actuator_id, 0] == 400.0
    assert model.actuator_biasprm[actuator_id, 1] == -400.0
    assert model.actuator_biasprm[actuator_id, 2] == -40.0
    assert model.actuator_forcelimited[actuator_id]
    np.testing.assert_array_equal(model.actuator_forcerange[actuator_id], [-4.0, 4.0])


def test_old_config_preserves_xml_gripper_control():
    config = I2rtYamRobotConfig()
    for field in (
        "gripper_effort_limit",
        "gripper_position_gain",
        "gripper_velocity_gain",
    ):
        del config.__dict__[field]

    spec = mujoco.MjSpec()
    I2rtYamRobot.add_robot_to_scene(
        config,
        spec,
        prefix=config.robot_namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
    )
    I2rtYamRobot.apply_control_overrides(spec, config)

    model = spec.compile()
    actuator_id = model.actuator(f"{config.robot_namespace}gripper").id
    assert model.actuator_gainprm[actuator_id, 0] == 100.0
    assert model.actuator_biasprm[actuator_id, 1] == -100.0
    assert model.actuator_biasprm[actuator_id, 2] == -10.0
    assert not model.actuator_forcelimited[actuator_id]


def test_gripper_opening_does_not_count_crossed_fingers():
    gripper = object.__new__(I2rtYamGripperGroup)
    with patch.object(
        I2rtYamGripperGroup,
        "joint_pos",
        new_callable=PropertyMock,
    ) as joint_pos:
        joint_pos.return_value = np.array([0.02, -0.02])
        assert gripper.inter_finger_dist == 0.04
        joint_pos.return_value = np.array([-0.002, 0.002])
        assert gripper.inter_finger_dist == 0.0
