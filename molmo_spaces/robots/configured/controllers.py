"""Controller implementations and builders for configured robots."""

from __future__ import annotations

from typing import cast

import numpy as np

from molmo_spaces.controllers.abstract import AbstractPositionController, Controller
from molmo_spaces.controllers.joint_pos import JointPosController
from molmo_spaces.controllers.joint_rel_pos import JointRelPosController
from molmo_spaces.robots.configured.spec import (
    ControllerSpec,
    ControlProfileSpec,
    DirectActuatorControllerSpec,
    JointPositionControllerSpec,
    JointRelativePositionControllerSpec,
    LinearSynergyControllerSpec,
    MocapPoseControllerSpec,
    PlanarPositionControllerSpec,
    RelativePlanarPositionControllerSpec,
)
from molmo_spaces.robots.robot_views.abstract import MoveGroup, RobotBaseGroup, RobotView
from molmo_spaces.utils.pose import pos_quat_to_pose_mat


class DirectActuatorController(Controller):
    """Pass a configured target directly to a move group's actuators."""

    def __init__(self, robot_move_group: MoveGroup) -> None:
        super().__init__(robot_move_group)
        self._target = robot_move_group.ctrl.copy()
        self._stationary = True

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape != (self.robot_move_group.n_actuators,):
            raise ValueError(
                f"Direct actuator target has shape {target.shape}; expected "
                f"({self.robot_move_group.n_actuators},)"
            )
        self._target = np.clip(
            target,
            self.robot_move_group.ctrl_limits[:, 0],
            self.robot_move_group.ctrl_limits[:, 1],
        )
        self._stationary = False

    def set_to_stationary(self) -> None:
        self._target = self.robot_move_group.ctrl.copy()
        self._stationary = True

    def compute_ctrl_inputs(self) -> np.ndarray:
        return self._target.copy()

    def reset(self) -> None:
        self.set_to_stationary()


class MocapPoseController(Controller):
    """Set a base pose while preserving the standard two-stage control API."""

    def __init__(self, robot_move_group: RobotBaseGroup) -> None:
        super().__init__(robot_move_group)
        self._target = robot_move_group.pose.copy()
        self._stationary = True

    @property
    def base(self) -> RobotBaseGroup:
        return cast(RobotBaseGroup, self.robot_move_group)

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape == (7,):
            target = pos_quat_to_pose_mat(target[:3], target[3:])
        if target.shape != (4, 4):
            raise ValueError("Mocap pose targets must have shape (4, 4) or (7,)")
        self._target = target.copy()
        self._stationary = False

    def set_to_stationary(self) -> None:
        self._target = self.base.pose.copy()
        self._stationary = True

    def compute_ctrl_inputs(self) -> np.ndarray:
        self.base.pose = self._target
        return np.empty(0)

    def reset(self) -> None:
        self.set_to_stationary()


class FreeJointPlanarController(MocapPoseController):
    """Interpret an absolute or relative x/y/yaw command for a free-joint base."""

    def __init__(self, robot_move_group: RobotBaseGroup, relative: bool) -> None:
        self._relative = relative
        super().__init__(robot_move_group)

    def set_target(self, target) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape != (3,):
            raise ValueError("Planar pose targets must have shape (3,)")
        pose = self.base.pose.copy()
        if self._relative:
            x = float(pose[0, 3] + target[0])
            y = float(pose[1, 3] + target[1])
            current_yaw = float(np.arctan2(pose[1, 0], pose[0, 0]))
            yaw = current_yaw + float(target[2])
        else:
            x, y, yaw = map(float, target)
        pose[0, 3] = x
        pose[1, 3] = y
        pose[:3, :3] = np.array(
            [
                [np.cos(yaw), -np.sin(yaw), 0.0],
                [np.sin(yaw), np.cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        self._target = pose
        self._stationary = False


class LinearSynergyController(AbstractPositionController):
    """Map a lower-dimensional command linearly into actuator positions."""

    def __init__(
        self,
        robot_move_group: MoveGroup,
        spec: LinearSynergyControllerSpec,
    ) -> None:
        super().__init__(robot_move_group)
        self._matrix = np.asarray(spec.matrix, dtype=float)
        self._bias = np.asarray(spec.bias, dtype=float)
        self._limits = (
            np.asarray(spec.output_limits, dtype=float)
            if spec.output_limits is not None
            else robot_move_group.ctrl_limits
        )
        self._command_limits = (
            np.asarray(spec.command_limits, dtype=float)
            if spec.command_limits is not None
            else None
        )
        self._state_to_command_matrix = (
            np.asarray(spec.state_to_command_matrix, dtype=float)
            if spec.state_to_command_matrix is not None
            else None
        )
        self._state_to_command_bias = (
            np.asarray(spec.state_to_command_bias, dtype=float)
            if spec.state_to_command_bias is not None
            else None
        )
        self._target = np.zeros(spec.command_dimension)
        self._target_pos = robot_move_group.noop_ctrl.copy()
        self._stationary = True

    @property
    def target(self) -> np.ndarray:
        return self._target

    @property
    def target_pos(self) -> np.ndarray:
        return self._target_pos.copy()

    @property
    def stationary(self) -> bool:
        return self._stationary

    def set_target(self, target) -> None:
        target = np.asarray(target, dtype=float)
        if target.shape != self._target.shape:
            raise ValueError(
                f"Linear synergy target has shape {target.shape}; expected {self._target.shape}"
            )
        if self._command_limits is not None:
            target = np.clip(target, self._command_limits[:, 0], self._command_limits[:, 1])
        self._target = target.copy()
        output = self._matrix @ target + self._bias
        self._target_pos = np.clip(output, self._limits[:, 0], self._limits[:, 1])
        self._stationary = False

    def set_to_stationary(self) -> None:
        if self._state_to_command_matrix is None:
            self._target_pos = self.robot_move_group.noop_ctrl.copy()
        else:
            command = (
                self._state_to_command_matrix @ self.robot_move_group.joint_pos
                + self._state_to_command_bias
            )
            if self._command_limits is not None:
                command = np.clip(command, self._command_limits[:, 0], self._command_limits[:, 1])
            self._target = command
            self._target_pos = np.clip(
                self._matrix @ command + self._bias,
                self._limits[:, 0],
                self._limits[:, 1],
            )
        self._stationary = True

    @property
    def max_height(self) -> float:
        if self._command_limits is None or len(self._command_limits) != 1:
            raise AttributeError("This synergy does not declare one-dimensional command limits")
        return float(self._command_limits[0, 1])

    def hold_at_height(self, height: float) -> None:
        self.set_target(np.asarray([height]))
        self._stationary = True

    def compute_ctrl_inputs(self) -> np.ndarray:
        return self._target_pos.copy()

    def reset(self) -> None:
        self.set_to_stationary()


def _build_controller(controller_spec: ControllerSpec, robot_view: RobotView) -> Controller:
    group = robot_view.get_move_group(controller_spec.move_group)
    if isinstance(controller_spec, JointPositionControllerSpec):
        return JointPosController(group)
    if isinstance(controller_spec, JointRelativePositionControllerSpec):
        return JointRelPosController(group)
    if isinstance(controller_spec, DirectActuatorControllerSpec):
        return DirectActuatorController(group)
    if isinstance(controller_spec, MocapPoseControllerSpec):
        if not isinstance(group, RobotBaseGroup):
            raise ValueError("Mocap pose controllers require a robot base move group")
        return MocapPoseController(group)
    if isinstance(
        controller_spec, (PlanarPositionControllerSpec, RelativePlanarPositionControllerSpec)
    ):
        if not isinstance(group, RobotBaseGroup):
            raise ValueError("Planar controllers require a robot base move group")
        if group.n_actuators:
            controller_type = (
                JointRelPosController
                if isinstance(controller_spec, RelativePlanarPositionControllerSpec)
                else JointPosController
            )
            return controller_type(group)
        return FreeJointPlanarController(
            group,
            relative=isinstance(controller_spec, RelativePlanarPositionControllerSpec),
        )
    if isinstance(controller_spec, LinearSynergyControllerSpec):
        return LinearSynergyController(group, controller_spec)
    raise TypeError(f"Unsupported controller kind: {controller_spec.kind!r}")


def build_controllers(
    controller_specs: dict[str, ControllerSpec],
    profile: ControlProfileSpec,
    robot_view: RobotView,
) -> dict[str, Controller]:
    """Build the active profile's controllers keyed by move-group ID."""
    controllers = {
        move_group_id: _build_controller(controller_specs[controller_id], robot_view)
        for move_group_id, controller_id in profile.controller_bindings.items()
    }
    for move_group_id, controller in controllers.items():
        group = robot_view.get_move_group(move_group_id)
        output = np.asarray(controller.compute_ctrl_inputs())
        if output.shape != (group.n_actuators,):
            raise ValueError(
                f"Controller for move group {move_group_id!r} produces shape {output.shape}; "
                f"compiled group requires ({group.n_actuators},)"
            )
    return controllers
