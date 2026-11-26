"""Generic sensors backed by configured frames and move groups."""

from __future__ import annotations

import gymnasium.spaces as gyms
import numpy as np
from mujoco import MjData

from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.robots.configured.names import resolve_frame_id, resolve_model_id
from molmo_spaces.robots.configured.robot_view import ConfiguredRobotView
from molmo_spaces.robots.configured.spec import (
    ApertureSensorSpec,
    ContactStateSensorSpec,
    EmbodimentSpec,
    FramePoseSensorSpec,
    JointStateSensorSpec,
    SensorSpec,
    TaskObjectBilateralContactSensorSpec,
)
from molmo_spaces.utils.linalg_utils import global_to_relative_transform
from molmo_spaces.utils.mj_model_and_data_utils import body_pose, descendant_geoms, site_pose
from molmo_spaces.utils.pose import pose_mat_to_7d


class ConfiguredJointStateSensor(Sensor):
    def __init__(self, spec: JointStateSensorSpec, robot_view: ConfiguredRobotView) -> None:
        self.spec = spec
        spaces = {}
        for group_id in spec.move_groups:
            group = robot_view.get_move_group(group_id)
            if spec.include_position:
                spaces[f"{group_id}/position"] = gyms.Box(
                    low=-np.inf, high=np.inf, shape=(group.pos_dim,), dtype=np.float32
                )
            if spec.include_velocity:
                spaces[f"{group_id}/velocity"] = gyms.Box(
                    low=-np.inf, high=np.inf, shape=(group.vel_dim,), dtype=np.float32
                )
        self.is_dict = True
        super().__init__(uuid=spec.uuid, observation_space=gyms.Dict(spaces))

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs):
        view = env.robots[batch_index].robot_view
        observation = {}
        for group_id in self.spec.move_groups:
            group = view.get_move_group(group_id)
            if self.spec.include_position:
                observation[f"{group_id}/position"] = group.joint_pos.astype(np.float32)
            if self.spec.include_velocity:
                observation[f"{group_id}/velocity"] = group.joint_vel.astype(np.float32)
        return observation


class ConfiguredFramePoseSensor(Sensor):
    def __init__(
        self,
        spec: FramePoseSensorSpec,
        mj_data: MjData,
        namespace: str,
    ) -> None:
        self.spec = spec
        self._frame_id = resolve_frame_id(mj_data.model, spec.frame, namespace)
        shape = (4, 4) if spec.output_format == "matrix_4x4" else (7,)
        super().__init__(
            uuid=spec.uuid,
            observation_space=gyms.Box(low=-np.inf, high=np.inf, shape=shape, dtype=np.float32),
        )

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs):
        data = env.mj_datas[batch_index]
        if self.spec.frame.kind == "site":
            pose = site_pose(data, self._frame_id)
        else:
            pose = body_pose(data, self._frame_id)
        if self.spec.relative_to_group is not None:
            group = env.robots[batch_index].robot_view.get_move_group(self.spec.relative_to_group)
            pose = global_to_relative_transform(pose, group.root_frame_to_world)
        if self.spec.output_format == "matrix_4x4":
            return pose.astype(np.float32)
        return pose_mat_to_7d(pose).astype(np.float32)


class ConfiguredApertureSensor(Sensor):
    def __init__(self, spec: ApertureSensorSpec) -> None:
        self.spec = spec
        super().__init__(
            uuid=spec.uuid,
            observation_space=gyms.Box(low=0.0, high=np.inf, shape=(1,), dtype=np.float32),
        )

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs):
        group = env.robots[batch_index].robot_view.get_gripper(self.spec.move_group)
        return np.asarray([group.inter_finger_dist], dtype=np.float32)


class ConfiguredContactStateSensor(Sensor):
    def __init__(
        self,
        spec: ContactStateSensorSpec,
        embodiment: EmbodimentSpec,
        mj_data: MjData,
        namespace: str,
    ) -> None:
        self.spec = spec
        end_effector = embodiment.robot_view.end_effectors[spec.end_effector]
        self._tool_group = end_effector.tool_group
        profile = embodiment.grasp_profiles[end_effector.grasp_profile_id]
        self._rule = profile.grasp_rule
        self._contact_groups: dict[str, set[int]] = {}
        for group_id, group in profile.contact_groups.items():
            geom_ids = {
                resolve_model_id(mj_data.model, "geom", reference, namespace)
                for reference in group.geom_names
            }
            for body_reference in group.body_names:
                body_id = resolve_model_id(mj_data.model, "body", body_reference, namespace)
                start = int(mj_data.model.body_geomadr[body_id])
                count = int(mj_data.model.body_geomnum[body_id])
                geom_ids.update(range(start, start + count))
            for body_reference in group.include_descendants_of:
                body_id = resolve_model_id(mj_data.model, "body", body_reference, namespace)
                geom_ids.update(descendant_geoms(mj_data.model, body_id, visible_only=False))
            for body_reference in group.exclude_descendants_of:
                body_id = resolve_model_id(mj_data.model, "body", body_reference, namespace)
                geom_ids.difference_update(
                    descendant_geoms(mj_data.model, body_id, visible_only=False)
                )
            self._contact_groups[group_id] = geom_ids
        self._robot_contact_geoms = set().union(*self._contact_groups.values())
        super().__init__(uuid=spec.uuid, observation_space=gyms.MultiBinary(1))

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs):
        data = env.mj_datas[batch_index]
        contact_counts = dict.fromkeys(self._contact_groups, 0)
        for contact in data.contact[: data.ncon]:
            geom_a, geom_b = int(contact.geom1), int(contact.geom2)
            for group_id, geom_ids in self._contact_groups.items():
                if geom_a in geom_ids and geom_b not in self._robot_contact_geoms:
                    contact_counts[group_id] += 1
                elif geom_b in geom_ids and geom_a not in self._robot_contact_geoms:
                    contact_counts[group_id] += 1
        valid = all(
            contact_counts[group_id] >= self._rule.minimum_contact_count
            for group_id in self._rule.required_contact_groups
        ) and all(contact_counts[group_id] == 0 for group_id in self._rule.forbidden_contact_groups)
        if self._rule.aperture_threshold is not None:
            gripper = env.robots[batch_index].robot_view.get_gripper(self._tool_group)
            valid = valid and gripper.inter_finger_dist >= self._rule.aperture_threshold
        return np.asarray([valid], dtype=np.int8)


class ConfiguredTaskObjectBilateralContactSensor(Sensor):
    def __init__(
        self,
        spec: TaskObjectBilateralContactSensorSpec,
        mj_data: MjData,
        namespace: str,
    ) -> None:
        self.spec = spec
        self._finger_body_ids = {
            resolve_model_id(mj_data.model, "body", spec.finger_a_body, namespace),
            resolve_model_id(mj_data.model, "body", spec.finger_b_body, namespace),
        }
        self._object_name: str | None = None
        super().__init__(
            uuid=spec.uuid,
            observation_space=gyms.Box(low=0, high=1, shape=(1,), dtype=np.uint8),
        )

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs):
        if self._object_name is None and task is not None:
            task_objects = task.get_task_objects(batch_index)
            for key in self.spec.task_object_keys:
                if key in task_objects:
                    self._object_name = task_objects[key]
                    break
        if self._object_name is None:
            return np.zeros(1, dtype=np.uint8)

        model = env.mj_model
        data = env.mj_datas[batch_index]
        contacted: set[int] = set()
        for contact in data.contact[: data.ncon]:
            body_a = int(model.geom_bodyid[contact.geom1])
            body_b = int(model.geom_bodyid[contact.geom2])
            name_a = model.body(body_a).name
            name_b = model.body(body_b).name
            if body_a in self._finger_body_ids and name_b == self._object_name:
                contacted.add(body_a)
            if body_b in self._finger_body_ids and name_a == self._object_name:
                contacted.add(body_b)
        return np.asarray([len(contacted) == 2], dtype=np.uint8)


def build_sensors(
    sensor_specs: list[SensorSpec],
    embodiment: EmbodimentSpec,
    mj_data: MjData,
    robot_view: ConfiguredRobotView,
    namespace: str,
) -> list[Sensor]:
    """Build all typed robot sensors declared by an embodiment."""
    sensors: list[Sensor] = []
    for sensor_spec in sensor_specs:
        if isinstance(sensor_spec, JointStateSensorSpec):
            sensor = ConfiguredJointStateSensor(sensor_spec, robot_view)
        elif isinstance(sensor_spec, FramePoseSensorSpec):
            sensor = ConfiguredFramePoseSensor(sensor_spec, mj_data, namespace)
        elif isinstance(sensor_spec, ApertureSensorSpec):
            sensor = ConfiguredApertureSensor(sensor_spec)
        elif isinstance(sensor_spec, ContactStateSensorSpec):
            sensor = ConfiguredContactStateSensor(sensor_spec, embodiment, mj_data, namespace)
        elif isinstance(sensor_spec, TaskObjectBilateralContactSensorSpec):
            sensor = ConfiguredTaskObjectBilateralContactSensor(sensor_spec, mj_data, namespace)
        else:
            raise TypeError(f"Unsupported sensor kind: {sensor_spec.kind!r}")
        sensors.append(sensor)
    return sensors
