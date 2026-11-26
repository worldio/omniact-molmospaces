"""Generic RobotView construction from an EmbodimentSpec."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from mujoco import MjData

from molmo_spaces.robots.configured.move_groups import (
    ConfiguredFreeJointBaseGroup,
    ConfiguredGripperMoveGroup,
    ConfiguredJointMoveGroup,
    ConfiguredMocapBaseGroup,
    ConfiguredPlanarBaseGroup,
)
from molmo_spaces.robots.configured.names import resolve_frame_id
from molmo_spaces.robots.configured.spec import (
    EndEffectorSpec,
    FreeJointBaseMoveGroupSpec,
    GraspProfileSpec,
    GripperMoveGroupSpec,
    JointMoveGroupSpec,
    MocapBaseMoveGroupSpec,
    PlanarBaseMoveGroupSpec,
    RobotViewSpec,
)
from molmo_spaces.robots.robot_views.abstract import RobotBaseGroup, RobotView
from molmo_spaces.utils.linalg_utils import normalize_ang_error


@dataclass(frozen=True)
class ConfiguredEndEffector:
    """Resolved planner-facing end-effector binding."""

    id: str
    spec: EndEffectorSpec
    tcp_frame_id: int
    grasp_frame_id: int


class ConfiguredRobotView(RobotView):
    """A RobotView composed from closed configured move-group variants."""

    def __init__(
        self,
        mj_data: MjData,
        namespace: str,
        *,
        view_spec: RobotViewSpec,
        robot_name: str,
        grasp_profiles: dict[str, GraspProfileSpec] | None = None,
    ) -> None:
        self._namespace = namespace
        self._view_spec = view_spec
        self._robot_name = robot_name

        base_spec = view_spec.move_groups[view_spec.base_group]
        base = self._build_base(mj_data, base_spec, namespace)
        move_groups = {view_spec.base_group: base}
        for group_id, group_spec in view_spec.move_groups.items():
            if group_id == view_spec.base_group:
                continue
            if isinstance(group_spec, JointMoveGroupSpec):
                group = ConfiguredJointMoveGroup(mj_data, group_spec, base, namespace)
            elif isinstance(group_spec, GripperMoveGroupSpec):
                group = ConfiguredGripperMoveGroup(mj_data, group_spec, base, namespace)
            else:
                raise ValueError(
                    f"Only robot_view.base_group may use base move-group kind {group_spec.kind!r}"
                )
            move_groups[group_id] = group

        super().__init__(mj_data, move_groups)
        self._validate_unique_model_ownership()
        self._end_effectors = {
            end_effector_id: ConfiguredEndEffector(
                id=end_effector_id,
                spec=end_effector,
                tcp_frame_id=resolve_frame_id(mj_data.model, end_effector.tcp_frame, namespace),
                grasp_frame_id=resolve_frame_id(mj_data.model, end_effector.grasp_frame, namespace),
            )
            for end_effector_id, end_effector in view_spec.end_effectors.items()
        }
        if grasp_profiles is not None:
            self._validate_grasp_profiles(grasp_profiles, namespace)

    def _validate_unique_model_ownership(self) -> None:
        joint_owners: dict[int, str] = {}
        actuator_owners: dict[int, str] = {}
        for group_id, group in self._move_groups.items():
            for joint_id in group._joint_ids:
                if joint_id in joint_owners:
                    name = self.mj_model.joint(joint_id).name
                    raise ValueError(
                        f"Compiled joint {name!r} belongs to configured move groups "
                        f"{joint_owners[joint_id]!r} and {group_id!r}"
                    )
                joint_owners[joint_id] = group_id
            for actuator_id in group._actuator_ids:
                if actuator_id in actuator_owners:
                    name = self.mj_model.actuator(actuator_id).name
                    raise ValueError(
                        f"Compiled actuator {name!r} belongs to configured move groups "
                        f"{actuator_owners[actuator_id]!r} and {group_id!r}"
                    )
                actuator_owners[actuator_id] = group_id

    def _validate_grasp_profiles(
        self,
        grasp_profiles: dict[str, GraspProfileSpec],
        namespace: str,
    ) -> None:
        from molmo_spaces.robots.configured.names import resolve_model_id

        for profile in grasp_profiles.values():
            resolve_frame_id(self.mj_model, profile.grasp_frame, namespace)
            for contact_group in profile.contact_groups.values():
                for reference in contact_group.geom_names:
                    resolve_model_id(self.mj_model, "geom", reference, namespace)
                for reference in (
                    *contact_group.body_names,
                    *contact_group.include_descendants_of,
                    *contact_group.exclude_descendants_of,
                ):
                    resolve_model_id(self.mj_model, "body", reference, namespace)

    @staticmethod
    def _build_base(
        mj_data: MjData,
        spec: MocapBaseMoveGroupSpec | FreeJointBaseMoveGroupSpec | PlanarBaseMoveGroupSpec,
        namespace: str,
    ) -> RobotBaseGroup:
        if isinstance(spec, MocapBaseMoveGroupSpec):
            return ConfiguredMocapBaseGroup(mj_data, spec, namespace)
        if isinstance(spec, FreeJointBaseMoveGroupSpec):
            return ConfiguredFreeJointBaseGroup(mj_data, spec, namespace)
        if isinstance(spec, PlanarBaseMoveGroupSpec):
            return ConfiguredPlanarBaseGroup(mj_data, spec, namespace)
        raise TypeError(f"Unsupported configured base move group: {type(spec).__name__}")

    @property
    def name(self) -> str:
        return f"{self._namespace}{self._robot_name}"

    @property
    def base(self) -> RobotBaseGroup:
        return self._move_groups[self._view_spec.base_group]

    def end_effector_ids(self) -> list[str]:
        return list(self._end_effectors)

    def get_end_effector(self, end_effector_id: str) -> ConfiguredEndEffector:
        return self._end_effectors[end_effector_id]

    def get_joint_position(self, move_group_ids: list[str]) -> np.ndarray:
        return np.concatenate(
            [self.get_move_group(group_id).joint_pos for group_id in move_group_ids]
        )

    def distance_to(self, move_group_ids: list[str], target_pose: list[float]) -> float:
        if len(target_pose) != 3:
            raise ValueError(f"Expected [x, y, theta] pose, got {target_pose}")
        current = self.get_joint_position(move_group_ids)
        delta = np.asarray(
            [
                current[0] - target_pose[0],
                current[1] - target_pose[1],
                normalize_ang_error(current[2] - target_pose[2]),
            ]
        )
        return float(np.linalg.norm(delta))

    def is_close_to(
        self,
        move_group_ids: list[str],
        target_pose: list[float],
        threshold: float = 0.05,
    ) -> bool:
        return self.distance_to(move_group_ids, target_pose) < threshold

    @property
    def primary_end_effector(self) -> ConfiguredEndEffector:
        if not self._end_effectors:
            raise ValueError("Configured robot has no end effectors")
        return next(iter(self._end_effectors.values()))
