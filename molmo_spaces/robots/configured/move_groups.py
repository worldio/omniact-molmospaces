"""Concrete MolmoSpaces move groups assembled from data-only specifications."""

from __future__ import annotations

from functools import cached_property
from typing import Literal

import mujoco
import numpy as np
from mujoco import MjData

from molmo_spaces.robots.configured.names import (
    resolve_frame_id,
    resolve_model_id,
)
from molmo_spaces.robots.configured.spec import (
    FreeJointBaseMoveGroupSpec,
    GeomDistanceApertureSpec,
    GripperMoveGroupSpec,
    JointExpressionApertureSpec,
    JointMoveGroupSpec,
    MocapBaseMoveGroupSpec,
    PlanarBaseMoveGroupSpec,
)
from molmo_spaces.robots.robot_views.abstract import (
    FreeJointRobotBaseGroup,
    GripperGroup,
    HoloJointsRobotBaseGroup,
    MJCFFrameMixin,
    MocapRobotBaseGroup,
    RobotBaseGroup,
    SimplyActuatedMoveGroup,
)
from molmo_spaces.utils.mj_model_and_data_utils import body_pose


class ConfiguredMocapBaseGroup(MocapRobotBaseGroup):
    """A mocap base resolved from a configured body reference."""

    def __init__(
        self,
        mj_data: MjData,
        spec: MocapBaseMoveGroupSpec,
        namespace: str,
    ) -> None:
        body_id = resolve_model_id(mj_data.model, "body", spec.body, namespace)
        if mj_data.model.body_mocapid[body_id] < 0:
            name = mj_data.model.body(body_id).name
            raise ValueError(f"Configured mocap base body {name!r} is not a mocap body")
        self.runtime_mobile = spec.runtime_mobile
        super().__init__(mj_data, body_id)

    @cached_property
    def is_mobile(self) -> bool:
        return self.runtime_mobile


class ConfiguredFreeJointBaseGroup(FreeJointRobotBaseGroup):
    """A configured floating or ground-constrained free-joint base."""

    def __init__(
        self,
        mj_data: MjData,
        spec: FreeJointBaseMoveGroupSpec,
        namespace: str,
    ) -> None:
        free_joint_id = resolve_model_id(mj_data.model, "joint", spec.free_joint, namespace)
        body_id = resolve_model_id(mj_data.model, "body", spec.body, namespace)
        if int(mj_data.model.jnt_bodyid[free_joint_id]) != body_id:
            raise ValueError("Configured free joint does not belong to its declared base body")
        joint_ids = [
            resolve_model_id(mj_data.model, "joint", reference, namespace)
            for reference in spec.additional_joint_names
        ]
        actuator_ids = [
            resolve_model_id(mj_data.model, "actuator", reference, namespace)
            for reference in spec.actuator_names
        ]
        self.runtime_mobile = spec.runtime_mobile
        super().__init__(
            mj_data,
            free_joint_id,
            joint_ids,
            actuator_ids,
            floating=spec.floating,
        )
        self._mocap_target_id: int | None = None
        if spec.mocap_target_body is not None:
            target_body_id = resolve_model_id(
                mj_data.model, "body", spec.mocap_target_body, namespace
            )
            mocap_id = int(mj_data.model.body_mocapid[target_body_id])
            if mocap_id < 0:
                raise ValueError("Configured free-joint base mocap target is not a mocap body")
            self._mocap_target_id = mocap_id

    @property
    def noop_ctrl(self) -> np.ndarray:
        if self._mocap_target_id is not None:
            return self.joint_pos.copy()
        return self.ctrl.copy()

    @cached_property
    def n_actuators(self) -> int:
        if self._mocap_target_id is not None:
            return 7
        return len(self._actuator_ids)

    @property
    def ctrl(self) -> np.ndarray:
        if self._mocap_target_id is None:
            return self.mj_data.ctrl[self._actuator_ids]
        return np.concatenate(
            [
                self.mj_data.mocap_pos[self._mocap_target_id],
                self.mj_data.mocap_quat[self._mocap_target_id],
            ]
        )

    @ctrl.setter
    def ctrl(self, ctrl: np.ndarray) -> None:
        if self._mocap_target_id is None:
            self.mj_data.ctrl[self._actuator_ids] = ctrl
            return
        self.mj_data.mocap_pos[self._mocap_target_id] = ctrl[:3]
        self.mj_data.mocap_quat[self._mocap_target_id] = ctrl[3:]

    @cached_property
    def ctrl_limits(self) -> np.ndarray:
        if self._mocap_target_id is not None:
            limits = np.empty((7, 2))
            limits[:, 0] = -np.inf
            limits[:, 1] = np.inf
            return limits
        return super().ctrl_limits

    @cached_property
    def is_mobile(self) -> bool:
        return self.runtime_mobile


class ConfiguredPlanarBaseGroup(HoloJointsRobotBaseGroup):
    """A configured x/y/yaw base backed by three joints and actuators."""

    def __init__(
        self,
        mj_data: MjData,
        spec: PlanarBaseMoveGroupSpec,
        namespace: str,
    ) -> None:
        if spec.world_frame.kind != "site" or spec.base_frame.kind != "site":
            raise ValueError("Planar base world_frame and base_frame must be sites")
        joint_ids = [
            resolve_model_id(mj_data.model, "joint", reference, namespace)
            for reference in spec.joint_names
        ]
        actuator_ids = [
            resolve_model_id(mj_data.model, "actuator", reference, namespace)
            for reference in spec.actuator_names
        ]
        self.runtime_mobile = spec.runtime_mobile
        super().__init__(
            mj_data,
            resolve_frame_id(mj_data.model, spec.world_frame, namespace),
            resolve_frame_id(mj_data.model, spec.base_frame, namespace),
            joint_ids,
            actuator_ids,
            resolve_model_id(mj_data.model, "body", spec.root_body, namespace),
        )

    @cached_property
    def is_mobile(self) -> bool:
        return self.runtime_mobile


class ConfiguredJointMoveGroup(MJCFFrameMixin, SimplyActuatedMoveGroup):
    """A named joint/actuator group with an optional body or site leaf frame."""

    def __init__(
        self,
        mj_data: MjData,
        spec: JointMoveGroupSpec,
        robot_base_group: RobotBaseGroup,
        namespace: str,
    ) -> None:
        self.spec = spec
        self._root_id = resolve_model_id(mj_data.model, "body", spec.root_body, namespace)
        if spec.leaf_frame is None:
            self._leaf_frame_id = self._root_id
            self._leaf_frame_type: Literal["site", "body"] = "body"
        else:
            self._leaf_frame_id = resolve_frame_id(mj_data.model, spec.leaf_frame, namespace)
            self._leaf_frame_type = spec.leaf_frame.kind
        joint_ids = [
            resolve_model_id(mj_data.model, "joint", reference, namespace)
            for reference in spec.joint_names
        ]
        actuator_ids = [
            resolve_model_id(mj_data.model, "actuator", reference, namespace)
            for reference in spec.actuator_names
        ]
        super().__init__(mj_data, joint_ids, actuator_ids, self._root_id, robot_base_group)

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_frame_id

    @property
    def leaf_frame_type(self) -> Literal["site", "body"]:
        return self._leaf_frame_type

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return body_pose(self.mj_data, self._root_id)

    @property
    def side(self) -> str:
        return self.spec.side


class ConfiguredGripperMoveGroup(MJCFFrameMixin, GripperGroup):
    """A configured gripper with typed open/close and aperture semantics."""

    def __init__(
        self,
        mj_data: MjData,
        spec: GripperMoveGroupSpec,
        robot_base_group: RobotBaseGroup,
        namespace: str,
    ) -> None:
        self.spec = spec
        self._namespace = namespace
        self._root_id = resolve_model_id(mj_data.model, "body", spec.root_body, namespace)
        self._leaf_frame_id = resolve_frame_id(mj_data.model, spec.leaf_frame, namespace)
        self._leaf_frame_type: Literal["site", "body"] = spec.leaf_frame.kind
        joint_ids = [
            resolve_model_id(mj_data.model, "joint", reference, namespace)
            for reference in spec.joint_names
        ]
        actuator_ids = [
            resolve_model_id(mj_data.model, "actuator", reference, namespace)
            for reference in spec.actuator_names
        ]
        super().__init__(mj_data, joint_ids, actuator_ids, self._root_id, robot_base_group)

        self._state_matrix = (
            np.asarray(spec.state_mapping.matrix, dtype=float)
            if spec.state_mapping is not None
            else None
        )
        self._state_bias = (
            np.asarray(spec.state_mapping.bias, dtype=float)
            if spec.state_mapping is not None
            else None
        )

        aperture = spec.aperture
        if isinstance(aperture, JointExpressionApertureSpec):
            self._aperture_joint_ids = [
                resolve_model_id(mj_data.model, "joint", term.joint, namespace)
                for term in aperture.terms
            ]
            for joint_id in self._aperture_joint_ids:
                if mj_data.model.jnt_type[joint_id] not in {
                    mujoco.mjtJoint.mjJNT_HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE,
                }:
                    raise ValueError("Joint-expression aperture terms must reference scalar joints")
        elif isinstance(aperture, GeomDistanceApertureSpec):
            self._aperture_geom_ids = (
                resolve_model_id(mj_data.model, "geom", aperture.geom_a, namespace),
                resolve_model_id(mj_data.model, "geom", aperture.geom_b, namespace),
            )

    @property
    def leaf_frame_id(self) -> int:
        return self._leaf_frame_id

    @property
    def leaf_frame_type(self) -> Literal["site", "body"]:
        return self._leaf_frame_type

    @property
    def root_frame_to_world(self) -> np.ndarray:
        return self.leaf_frame_to_world

    @property
    def side(self) -> str:
        return self.spec.side

    def expand_joint_pos(self, joint_pos: np.ndarray) -> np.ndarray:
        joint_pos = np.asarray(joint_pos, dtype=float)
        if self._state_matrix is not None and joint_pos.shape == (
            self.spec.state_mapping.input_dimension,
        ):
            return self._state_matrix @ joint_pos + self._state_bias
        if self._state_matrix is None and joint_pos.shape == (1,):
            return np.full(self.pos_dim, joint_pos.item())
        return joint_pos

    @property
    def joint_pos(self) -> np.ndarray:
        return self.mj_data.qpos[self._joint_posadr]

    @joint_pos.setter
    def joint_pos(self, joint_pos: np.ndarray) -> None:
        expanded = self.expand_joint_pos(joint_pos)
        if expanded.shape != (self.pos_dim,):
            raise ValueError(
                f"Gripper joint input has shape {np.asarray(joint_pos).shape}; "
                f"expected ({self.pos_dim},)"
            )
        self.mj_data.qpos[self._joint_posadr] = expanded

    def set_gripper_ctrl_open(self, open: bool) -> None:
        self.ctrl = np.asarray(self.spec.open_command if open else self.spec.closed_command)

    @cached_property
    def inter_finger_dist_range(self) -> tuple[float, float]:
        aperture = self.spec.aperture
        if aperture is None:
            raise ValueError(f"Gripper {self.name!r} has no aperture specification")
        if isinstance(aperture, GeomDistanceApertureSpec):
            return 0.0, aperture.maximum_distance
        if aperture.clamp is not None:
            return aperture.clamp

        lower = aperture.offset
        upper = aperture.offset
        for term, joint_id in zip(aperture.terms, self._aperture_joint_ids):
            joint_range = self.mj_model.jnt_range[joint_id]
            if term.transform == "absolute":
                lower, upper = map(float, joint_range)
                candidates = [abs(lower), abs(upper)]
                minimum = 0.0 if lower <= 0.0 <= upper else min(candidates)
                term_range = (minimum, max(candidates))
            else:
                term_range = (float(joint_range[0]), float(joint_range[1]))
            scaled = (term.coefficient * term_range[0], term.coefficient * term_range[1])
            lower += min(scaled)
            upper += max(scaled)
        return lower, upper

    @property
    def inter_finger_dist(self) -> float:
        aperture = self.spec.aperture
        if aperture is None:
            raise ValueError(f"Gripper {self.name!r} has no aperture specification")
        if isinstance(aperture, GeomDistanceApertureSpec):
            first, second = self._aperture_geom_ids
            distance = float(
                mujoco.mj_geomDistance(
                    self.mj_model,
                    self.mj_data,
                    first,
                    second,
                    aperture.query_limit or aperture.maximum_distance,
                    None,
                )
            )
            return max(distance, 0.0)

        value = aperture.offset
        for term, joint_id in zip(aperture.terms, self._aperture_joint_ids):
            address = self.mj_model.jnt_qposadr[joint_id]
            joint_value = float(self.mj_data.qpos[address])
            if term.transform == "absolute":
                joint_value = abs(joint_value)
            value += term.coefficient * joint_value
        if aperture.clamp is not None:
            value = float(np.clip(value, *aperture.clamp))
        return value
