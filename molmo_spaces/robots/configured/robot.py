"""Generic Robot runtime backed by an EmbodimentSpec."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import numpy as np
from mujoco import MjData, MjSpec

from molmo_spaces.controllers.abstract import Controller
from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.kinematics.mujoco_kinematics import MlSpacesKinematics
from molmo_spaces.kinematics.parallel.parallel_kinematics import ParallelKinematics
from molmo_spaces.robots.abstract import Robot
from molmo_spaces.robots.configured.assembler import MjcfAssembler
from molmo_spaces.robots.configured.controllers import build_controllers
from molmo_spaces.robots.configured.kinematics import build_kinematics
from molmo_spaces.robots.configured.move_groups import ConfiguredGripperMoveGroup
from molmo_spaces.robots.configured.robot_view import ConfiguredRobotView
from molmo_spaces.robots.configured.sensors import build_sensors
from molmo_spaces.robots.configured.spec import JointMoveGroupSpec

if TYPE_CHECKING:
    from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
    from molmo_spaces.configs.robot_configs import BaseRobotConfig
    from molmo_spaces.robots.configured.config import ConfiguredRobotConfig


class ConfiguredRobot(Robot):
    """Interpret one configured embodiment without embodiment-specific code."""

    def __init__(self, mj_data: MjData, config: MlSpacesExpConfig) -> None:
        super().__init__(mj_data, config)
        robot_config = self._configured_config(config.robot_config)
        if robot_config.robot_view_factory is None:
            raise ValueError("Configured robot has no robot_view_factory")
        self._robot_view = cast(
            ConfiguredRobotView,
            robot_config.robot_view_factory(mj_data, robot_config.robot_namespace),
        )
        self._validate_initial_state(robot_config)

        embodiment = robot_config.embodiment
        profile = embodiment.robot_config.control_profiles[robot_config.control_profile]
        self._controllers = build_controllers(
            embodiment.robot.controllers,
            profile,
            self._robot_view,
        )
        self._kinematics, self._parallel_kinematics = build_kinematics(
            robot_config,
            embodiment.robot.kinematics,
        )
        self._sensors = build_sensors(
            embodiment.robot.sensors,
            embodiment,
            mj_data,
            self._robot_view,
            robot_config.robot_namespace,
        )

    @staticmethod
    def _configured_config(robot_config: BaseRobotConfig) -> ConfiguredRobotConfig:
        from molmo_spaces.robots.configured.config import ConfiguredRobotConfig

        if not isinstance(robot_config, ConfiguredRobotConfig):
            raise TypeError("ConfiguredRobot requires ConfiguredRobotConfig")
        return robot_config

    def _validate_initial_state(self, robot_config: ConfiguredRobotConfig) -> None:
        for group_id, values in robot_config.init_qpos.items():
            group = self._robot_view.get_move_group(group_id)
            initial = np.asarray(values, dtype=float)
            if isinstance(group, ConfiguredGripperMoveGroup):
                initial = group.expand_joint_pos(initial)
            if initial.shape != (group.pos_dim,):
                raise ValueError(
                    f"Initial qpos for move group {group_id!r} has shape {initial.shape}; "
                    f"compiled model requires ({group.pos_dim},)"
                )
            if not np.all(np.isfinite(initial)):
                raise ValueError(f"Initial qpos for move group {group_id!r} must be finite")
            limits = group.joint_pos_limits
            policy = robot_config.embodiment.robot_config.initial_qpos_limit_policy.get(
                group_id, "strict"
            )
            if policy == "strict" and (
                np.any(initial < limits[:, 0]) or np.any(initial > limits[:, 1])
            ):
                raise ValueError(
                    f"Initial qpos for move group {group_id!r} violates compiled joint limits"
                )

    @property
    def namespace(self) -> str:
        return self.exp_config.robot_config.robot_namespace

    @property
    def robot_view(self) -> ConfiguredRobotView:
        return self._robot_view

    @property
    def kinematics(self) -> MlSpacesKinematics:
        return self._kinematics

    @property
    def parallel_kinematics(self) -> ParallelKinematics:
        return self._parallel_kinematics

    @property
    def controllers(self) -> dict[str, Controller]:
        return self._controllers

    def create_robot_sensors(self) -> list[Sensor]:
        return [*super().create_robot_sensors(), *self._sensors]

    def get_arm_move_group_ids(self) -> list[str]:
        robot_config = self._configured_config(self.exp_config.robot_config)
        action_noise = robot_config.embodiment.robot_config.action_noise
        if action_noise is not None:
            return list(action_noise.target_arm_groups)
        return [
            group_id
            for group_id, group in robot_config.embodiment.robot_view.move_groups.items()
            if isinstance(group, JointMoveGroupSpec) and group.role == "arm"
        ]

    def apply_action_noise(self, action: dict[str, object]) -> dict[str, object]:
        noisy_action = super().apply_action_noise(action)
        robot_config = self._configured_config(self.exp_config.robot_config)
        noise = robot_config.embodiment.robot_config.action_noise
        if (
            noise is None
            or not noise.enabled
            or noise.target_base_group is None
            or action.get(noise.target_base_group) is None
        ):
            return noisy_action

        group = self.robot_view.get_move_group(noise.target_base_group)
        command = np.asarray(action[noise.target_base_group], dtype=float)
        delta = command - group.joint_pos
        position_std = noise.base_action_scale_factor * np.linalg.norm(delta[:2])
        rotation_std = noise.base_action_scale_factor * abs(delta[2])
        position_noise = np.clip(
            np.random.randn(2) * position_std,
            -noise.max_base_position_noise,
            noise.max_base_position_noise,
        )
        rotation_noise = float(
            np.clip(
                np.random.randn() * rotation_std,
                -noise.max_base_rotation_noise,
                noise.max_base_rotation_noise,
            )
        )
        noisy_command = command.copy()
        noisy_command[:2] += position_noise
        noisy_command[2] += rotation_noise
        noisy_action[noise.target_base_group] = noisy_command
        return noisy_action

    def reset(self) -> None:
        robot_config = self._configured_config(self.exp_config.robot_config)
        for group_id, default_position in robot_config.init_qpos.items():
            self._robot_view.get_move_group(group_id).joint_pos = np.asarray(default_position)
        for controller in self._controllers.values():
            controller.reset()
        for group_id in self._robot_view.move_group_ids():
            if group_id in self._controllers:
                continue
            group = self._robot_view.get_move_group(group_id)
            if group.n_actuators:
                noop = np.asarray(group.noop_ctrl)
                if noop.shape == (group.n_actuators,):
                    group.ctrl = noop

    def get_world_pose_tf_mat(self) -> np.ndarray:
        return self.robot_view.base.pose

    def get_joint_position(self, move_group_ids: list[str]) -> np.ndarray:
        return np.concatenate(
            [self.robot_view.get_move_group(group_id).joint_pos for group_id in move_group_ids]
        )

    def get_joint_ranges(self, move_group_ids: list[str]) -> dict[str, tuple[int, int]]:
        ranges = {}
        offset = 0
        for group_id in move_group_ids:
            group = self.robot_view.get_move_group(group_id)
            ranges[group_id] = (offset, offset + group.n_joints)
            offset += group.n_joints
        return ranges

    @staticmethod
    def robot_model_root_name() -> str:
        """Return a diagnostic label; configured roots are instance data."""
        return "<configured>"

    @classmethod
    def _load_robot_spec(
        cls,
        robot_config: BaseRobotConfig,
        strip_meshes: bool = False,
    ) -> MjSpec:
        configured = cls._configured_config(robot_config)
        return MjcfAssembler(configured.embodiment).load_robot_spec(strip_meshes=strip_meshes)

    @classmethod
    def add_robot_to_scene(
        cls,
        robot_config: BaseRobotConfig,
        spec: MjSpec,
        prefix: str,
        pos: list[float],
        quat: list[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        configured = cls._configured_config(robot_config)
        if prefix != configured.robot_namespace:
            raise ValueError(
                f"ConfiguredRobot insertion prefix {prefix!r} must match configured namespace "
                f"{configured.robot_namespace!r}"
            )
        assembler = MjcfAssembler(configured.embodiment)
        assembler.add_robot_to_scene(
            spec,
            prefix,
            pos,
            quat,
            randomize_textures=randomize_textures,
            strip_meshes=strip_meshes,
        )
        assembler.apply_control_overrides(spec, prefix, configured.control_profile)

    @classmethod
    def apply_control_overrides(
        cls,
        spec: MjSpec,
        robot_config: BaseRobotConfig,
    ) -> None:
        configured = cls._configured_config(robot_config)
        MjcfAssembler(configured.embodiment).apply_control_overrides(
            spec,
            configured.robot_namespace,
            configured.control_profile,
        )
