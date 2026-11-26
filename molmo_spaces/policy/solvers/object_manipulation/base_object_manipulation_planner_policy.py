import copy
import json
import logging
from pathlib import Path
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import gymnasium.spaces as gyms
import mujoco
import numpy as np
from mujoco import MjSpec
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.abstract_exp_config import MlSpacesExpConfig
from molmo_spaces.configs.policy_configs import ObjectManipulationPlannerPolicyConfig
from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.env.data_views import create_mlspaces_body
from molmo_spaces.policy.base_policy import PlannerPolicy
from molmo_spaces.robots.robot_views.abstract import RobotView
from molmo_spaces.tasks.task import BaseMujocoTask
from molmo_spaces.utils.grasp_sample import add_grasp_collision_bodies
from molmo_spaces.utils.linalg_utils import transform_to_twist, twist_to_transform
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, descendant_geoms
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.profiler_utils import Timer

log = logging.getLogger(__name__)

TASK_BARRIER_BODY = "task_barrier"
TASK_BARRIER_GEOM = "task_barrier_geom"


@dataclass(frozen=True)
class DiagnosticRenderFrame:
    """Scratch state rendered in place of the live simulation."""

    data: mujoco.MjData = field(repr=False, compare=False)
    highlight_geom_ids: tuple[int, ...]
    highlight_rgba: tuple[float, float, float, float]


@dataclass(frozen=True)
class IKDiagnostic(DiagnosticRenderFrame):
    """Best solver iterate for an IK target that did not converge."""

    phase: str
    minimum_error: float
    position_error: float
    rotation_error: float
    best_iteration: int
    target_pose: np.ndarray

    def caption(self) -> tuple[str, str]:
        return (
            f"IK FAILURE: {self.phase.upper()} BEST ITERATE",
            f"position {self.position_error * 1000:.1f} mm | rotation {np.degrees(self.rotation_error):.1f} deg",
        )

    def to_info(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "minimum_error": self.minimum_error,
            "position_error": self.position_error,
            "rotation_error": self.rotation_error,
            "best_iteration": self.best_iteration,
            "target_pose": self.target_pose.tolist(),
        }


@dataclass
class MoveSegment:
    """
    A move segment is a single movement in a MoveSequence,
    i.e. moving from a point to another point.
    """

    name: str

    @property
    def duration(self) -> float:
        raise NotImplementedError


@dataclass
class TCPMoveSegment(MoveSegment):
    """
    A move segment that moves the robot TCP to a target pose.
    The movement is a straight line in task space.
    """

    start_pose: np.ndarray
    end_pose: np.ndarray
    speed: float
    decoupled_interpolation: bool = False

    @property
    def duration(self) -> float:
        lin_vel, _ = transform_to_twist(np.linalg.inv(self.start_pose) @ self.end_pose)
        return np.linalg.norm(lin_vel) / self.speed

    def interpolate(self, fraction: float) -> np.ndarray:
        fraction = float(np.clip(fraction, 0.0, 1.0))
        if not self.decoupled_interpolation:
            linear, angular = transform_to_twist(np.linalg.inv(self.start_pose) @ self.end_pose)
            return self.start_pose @ twist_to_transform(
                linear * fraction,
                angular * fraction,
            )

        pose = np.eye(4)
        pose[:3, 3] = (1.0 - fraction) * self.start_pose[:3, 3] + fraction * self.end_pose[:3, 3]
        relative_rotation = self.start_pose[:3, :3].T @ self.end_pose[:3, :3]
        rotation_vector = R.from_matrix(relative_rotation).as_rotvec()
        pose[:3, :3] = (
            self.start_pose[:3, :3] @ R.from_rotvec(fraction * rotation_vector).as_matrix()
        )
        return pose


@dataclass
class JointMoveSegment(MoveSegment):
    """
    A move segment that moves the robot joints to a target configuration.
    The movement is a straight line in joint space.
    """

    start_qpos: (
        dict[str, np.ndarray | list[float]] | None
    )  # if None, use the qpos at the start of the segment
    end_qpos: dict[str, np.ndarray | list[float]]
    duration_s: float

    @property
    def duration(self) -> float:
        return self.duration_s


class ActionPrimitive(ABC):
    """
    Abstract base class for action primitives.
    An action primitive is a single action that can be executed by the robot.
    """

    def __init__(self, robot_view: RobotView, duration: float):
        self.robot_view = robot_view
        self.duration = duration
        self.start_time = None

    @abstractmethod
    def execute(self) -> bool:
        pass

    @abstractmethod
    def get_current_action(self) -> dict[str, Any]:
        raise NotImplementedError

    def reset(self):
        self.start_time = None

    def elapsed_time(self) -> float:
        return self.robot_view.mj_data.time - self.start_time

    def check_failure(self) -> bool:
        return False

    def get_current_phase(self) -> str:
        return "unknown"

    def get_render_override(self):
        return None

    def get_render_caption(self) -> tuple[str, ...] | None:
        return None


class MoveSequence(ActionPrimitive):
    """
    A sequence of (TCP or joint) move segments executed sequentially without pauses.
    The sequence is terminated when the last move segment is completed, after
    a settling period.
    """

    def __init__(
        self,
        robot_view: RobotView,
        settle_time: float,
        move_segments: list[MoveSegment],
        is_holding_object: bool = False,
        gripper_empty_threshold: float = 0.0,
    ) -> None:
        super().__init__(robot_view, sum(seg.duration for seg in move_segments))
        self._move_segments = move_segments
        self.move_seg_idx = None
        self.move_seg_start_time = None
        self.settle_time = settle_time
        self.is_holding_object = is_holding_object
        self.gripper_empty_threshold = gripper_empty_threshold

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            self.move_seg_idx = None
            self.move_seg_start_time = self.start_time

        duration_cumsum = np.cumsum([0] + [seg.duration for seg in self._move_segments])
        idx = np.searchsorted(duration_cumsum, self.elapsed_time(), side="right").item() - 1
        if idx != self.move_seg_idx:
            if idx < len(self._move_segments):
                log.info(f"Moving to {self._move_segments[idx].name}")
            self.move_seg_idx = idx
            self.move_seg_start_time = self.robot_view.mj_data.time

        return self.elapsed_time() >= self.duration + self.settle_time

    def get_current_action(self) -> dict[str, Any]:
        elapsed_time = self.robot_view.mj_data.time - self.move_seg_start_time
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            t = min(1.0, elapsed_time / move_seg.duration)

            curr_target_pose = move_seg.interpolate(t)
        else:
            move_seg = self.move_segments[-1]
            curr_target_pose = move_seg.end_pose

        # Solve IK for current target pose
        mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.tcp_to_jp_fn(mg_id, curr_target_pose)

    def get_current_phase(self) -> str:
        if self.move_seg_idx is None:
            move_seg = self.move_segments[0]
        elif self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
        else:
            move_seg = self.move_segments[-1]
        return move_seg.name

    def reset(self):
        super().reset()
        self.move_seg_idx = None
        self.move_seg_start_time = None

    def check_failure(self) -> bool:
        if super().check_failure():
            return True

        if self.is_holding_object:
            gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
            gripper = self.robot_view.get_gripper(gripper_mg_id)
            if (
                gripper.inter_finger_dist
                < gripper.inter_finger_dist_range[0] + self.gripper_empty_threshold
            ):
                log.info(
                    f"Object is not in grasp! {gripper.inter_finger_dist:.05f} < {gripper.inter_finger_dist_range[0] + self.gripper_empty_threshold:05f}"
                )
                return True

        return False


class TCPMoveSequence(MoveSequence):
    """
    Action primitive that tracks a trajectory in task space.
    Waypoints are interpolated linearly in task space.
    The sequence is terminated after the cumulative duration of the move segments,
    plus the settling period.
    """

    def __init__(
        self,
        robot_view: RobotView,
        tcp_to_jp_fn: Callable[[str, np.ndarray], dict[str, Any]],
        settle_time: float,
        move_segments: list[TCPMoveSegment],
        is_holding_object: bool = False,
        gripper_empty_threshold: float = 0.0,
        tcp_pos_err_threshold: float = np.inf,
        tcp_rot_err_threshold: float = np.inf,
    ) -> None:
        super().__init__(
            robot_view,
            settle_time,
            move_segments,
            is_holding_object,
            gripper_empty_threshold,
        )
        self.tcp_to_jp_fn = tcp_to_jp_fn
        self.tcp_pos_err_threshold = tcp_pos_err_threshold
        self.tcp_rot_err_threshold = tcp_rot_err_threshold

    @property
    def move_segments(self) -> list[TCPMoveSegment]:
        return self._move_segments

    def get_current_target_pose(self) -> np.ndarray:
        elapsed_time = self.robot_view.mj_data.time - self.move_seg_start_time
        assert self.move_seg_idx is not None
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            t = min(1.0, elapsed_time / move_seg.duration)

            curr_target_pose = move_seg.interpolate(t)
        else:
            move_seg = self.move_segments[-1]
            curr_target_pose = move_seg.end_pose
        return curr_target_pose

    def get_current_action(self) -> dict[str, Any]:
        curr_target_pose = self.get_current_target_pose()

        # Solve IK for current target pose
        mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        return self.tcp_to_jp_fn(mg_id, curr_target_pose)

    def check_failure(self) -> bool:
        if super().check_failure():
            return True
        elif self.move_seg_idx is None:
            # we haven't started moving yet
            return False

        curr_target_pose = self.get_current_target_pose()
        gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
        gripper = self.robot_view.get_gripper(gripper_mg_id)

        trf = np.linalg.inv(gripper.leaf_frame_to_world) @ curr_target_pose
        pos_err = np.linalg.norm(trf[:3, 3])
        rot_err = R.from_matrix(trf[:3, :3]).magnitude()
        return pos_err > self.tcp_pos_err_threshold or rot_err > self.tcp_rot_err_threshold


class JointMoveSequence(MoveSequence):
    """
    Action primitive that tracks a trajectory in joint space.
    Waypoints are interpolated linearly in joint space.
    The sequence is terminated after the cumulative duration of the move segments,
    plus the settling period.
    """

    def __init__(
        self,
        robot_view: RobotView,
        settle_time: float,
        move_segments: list[JointMoveSegment],
        is_holding_object: bool = False,
        gripper_empty_threshold: float = 0.0,
    ) -> None:
        super().__init__(
            robot_view,
            settle_time,
            move_segments,
            is_holding_object,
            gripper_empty_threshold,
        )

    @property
    def move_segments(self) -> list[JointMoveSegment]:
        return self._move_segments

    def execute(self) -> bool:
        ret = super().execute()

        assert self.move_seg_idx is not None
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            if move_seg.start_qpos is None:
                move_seg.start_qpos = self.robot_view.get_qpos_dict()

        return ret

    def get_current_action(self) -> dict[str, Any]:
        elapsed_time = self.robot_view.mj_data.time - self.move_seg_start_time
        assert self.move_seg_idx is not None
        if self.move_seg_idx < len(self.move_segments):
            move_seg = self.move_segments[self.move_seg_idx]
            t = min(1.0, elapsed_time / move_seg.duration)

            curr_target_qpos = {**move_seg.start_qpos}
            for k in move_seg.end_qpos:
                q0 = np.asarray(move_seg.start_qpos[k])
                q1 = np.asarray(move_seg.end_qpos[k])
                curr_target_qpos[k] = q0 + (q1 - q0) * t
        else:
            move_seg = self.move_segments[-1]
            curr_target_qpos = {**move_seg.end_qpos}

        for mg_id in self.robot_view.get_gripper_movegroup_ids():
            if mg_id in curr_target_qpos:
                del curr_target_qpos[mg_id]
        return curr_target_qpos


class GripperAction(ActionPrimitive):
    """
    Action primitive that opens or closes the gripper.
    """

    def __init__(self, robot_view: RobotView, target_open: bool, duration: float) -> None:
        super().__init__(robot_view, duration)
        self.target_open = target_open

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            if self.target_open:
                log.info("Opening gripper...")
            else:
                log.info("Closing gripper...")

            mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
            gripper = self.robot_view.get_gripper(mg_id)
            gripper.set_gripper_ctrl_open(self.target_open)

        return self.elapsed_time() >= self.duration

    def get_current_action(self) -> dict[str, Any]:
        return self.robot_view.get_ctrl_dict()

    def get_current_phase(self):
        if self.target_open:
            return "gripper-open"
        else:
            return "gripper-close"


class NoopAction(ActionPrimitive):
    """
    Action primitive that does nothing for a given duration.
    """

    def __init__(
        self,
        robot_view: RobotView,
        duration: float,
        phase: str = "unknown",
        caption: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(robot_view, duration)
        self._action = None
        self.phase = phase
        self.caption = caption

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
        return self.elapsed_time() >= self.duration

    def get_current_action(self) -> dict[str, Any]:
        if self._action is None:
            self._action = self.robot_view.get_noop_ctrl_dict()
        return self._action

    def get_current_phase(self) -> str:
        return self.phase

    def get_render_caption(self) -> tuple[str, ...] | None:
        return self.caption


class IKDiagnosticAction(NoopAction):
    """Hold and label the best iterate from a failed IK solve."""

    def __init__(
        self,
        robot_view: RobotView,
        diagnostic: IKDiagnostic,
        duration: float = 1.0,
    ) -> None:
        self.diagnostic = diagnostic
        super().__init__(
            robot_view,
            duration=duration,
            phase=diagnostic.phase,
            caption=diagnostic.caption(),
        )

    def get_render_override(self):
        return self.diagnostic


class GraspPoseSensor(Sensor):
    """Sensor for the planned grasp pose in 7D format."""

    def __init__(self, uuid: str = "grasp_pose") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)
        self._logged_warning = False

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get the planned grasp pose, or NaNs when no valid plan exists."""
        if task._registered_policy is not None:
            assert isinstance(task._registered_policy, BaseObjectManipulationPlannerPolicy)
            grasp_pose = task._registered_policy.target_poses.get("grasp")
            if grasp_pose is None or not np.all(np.isfinite(grasp_pose)):
                return np.full(7, np.nan, dtype=np.float32)
            return np.array(
                pose_mat_to_7d(grasp_pose),
                dtype=np.float32,
            )
        else:
            if not self._logged_warning:
                log.warning("Policy is not registered or does not support grasp pose sensing.")
                self._logged_warning = True
            return np.full(7, np.nan, dtype=np.float32)


class ReleasePoseSensor(Sensor):
    """Sensor for the planned release TCP pose in 7D format."""

    def __init__(self, uuid: str = "release_pose") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        """Get the planned place pose, or NaNs when no complete place plan exists."""
        policy = task._registered_policy
        if isinstance(policy, BaseObjectManipulationPlannerPolicy):
            release_pose = policy.target_poses.get("place")
            if release_pose is not None and np.all(np.isfinite(release_pose)):
                return np.asarray(pose_mat_to_7d(release_pose), dtype=np.float32)
        return np.full(7, np.nan, dtype=np.float32)


class PhaseTargetPoseSensor(Sensor):
    """Sensor for the endpoint pose of the currently active motion phase."""

    def __init__(self, uuid: str = "phase_target_pose") -> None:
        observation_space = gyms.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> np.ndarray:
        policy = task._registered_policy
        if isinstance(policy, BaseObjectManipulationPlannerPolicy):
            target_pose = policy.get_current_phase_target_pose()
            if target_pose is not None and np.all(np.isfinite(target_pose)):
                return np.asarray(pose_mat_to_7d(target_pose), dtype=np.float32)
        return np.full(7, np.nan, dtype=np.float32)


class GraspPlanDiagnosticsSensor(Sensor):
    """Sensor for grasp filtering and per-phase IK diagnostics."""

    def __init__(self, uuid: str = "grasp_plan_diagnostics", str_max_len: int = 4000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        policy = task._registered_policy
        if isinstance(policy, BaseObjectManipulationPlannerPolicy):
            return policy.grasp_plan_diagnostics
        return {
            "status": "policy_unavailable",
            "failure_cause": "No object-manipulation policy is registered",
        }


def _has_ancestor(model, body_id: int, ancestor: int) -> bool:
    while body_id > 0:
        if body_id == ancestor:
            return True
        body_id = int(model.body_parentid[body_id])
    return False


class PickupObjectPoseSensor(Sensor):
    """The picked object's world pose and vertical extent, per step.

    Contacts cannot be sampled reliably from a policy step: ``env.py`` advances physics with a
    single batched ``mj_step(model, data, n_steps)``, so ``data.contact`` afterwards describes only
    the last substep and a graze inside the 66 ms window leaves no trace. Pose does not have that
    problem -- a bottle resting on a counter sits at the counter's height whenever you look -- and
    ``env_states/actors`` is empty, so this is the only way to know where the object went.
    """

    def __init__(self, uuid: str = "pickup_obj_pose", str_max_len: int = 1000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        name = getattr(task.config.task_config, "pickup_obj_name", None)
        if not name:
            return {}
        try:
            obj = env.object_managers[batch_index].get_object_by_name(name)
            centre, dims = body_aabb(env.current_model, env.current_data, obj.body_id,
                                     visible_only=False)
            return {
                "position": [float(v) for v in obj.position],
                "aabb_centre": [float(v) for v in centre],
                "aabb_dims": [float(v) for v in dims],
                "bottom_z": float(centre[2] - dims[2] / 2.0),
            }
        except Exception:
            return {}


class ObjectContactSensor(Sensor):
    """Per-step record of what the picked object touches, other than the gripper holding it."""

    def __init__(self, uuid: str = "object_contacts", str_max_len: int = 4000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        policy = task._registered_policy
        contacts = getattr(policy, "object_contacts", None)
        return {"contacts": contacts or []}


class BaseObjectManipulationPlannerPolicy(PlannerPolicy):
    """
    Base class for object manipulation (open/close/pick/pick-place) planner policies.
    This class provides common functionality for object manipulation planner policies,
    including concatenating multiple action primitives into a single trajectory.
    """

    def __init__(self, config: MlSpacesExpConfig, task: BaseMujocoTask) -> None:
        assert isinstance(config.policy_config, ObjectManipulationPlannerPolicyConfig)
        super().__init__(config, task)
        self.policy_config = config.policy_config
        self.robot_view = task.env.current_robot.robot_view

        self.action_primitives = []
        self.action_idx = 0
        self._retry_count = 0
        # Grasps (in the manipulated object's frame) already attempted this episode, so retries
        # can prefer a different grasp site. Persists across retries; cleared on a fresh episode.
        self._tried_object_grasps: list[np.ndarray] = []
        self.target_poses = {}
        self.ik_warmed_up = False
        self.sequential_ik_failures = 0
        self.planning_failure_reason: str | None = None
        self.grasp_plan_diagnostics: dict[str, Any] = {"status": "not_evaluated"}
        self._ik_solution_cache: dict[bytes, dict[str, np.ndarray]] = {}

    @property
    def retry_count(self) -> int:
        return self._retry_count

    @property
    def planners(self):
        return {}

    def create_policy_sensors(self) -> list[Sensor]:
        return super().create_policy_sensors() + [
            GraspPoseSensor(uuid="grasp_pose"),
            ReleasePoseSensor(uuid="release_pose"),
            PhaseTargetPoseSensor(uuid="phase_target_pose"),
            GraspPlanDiagnosticsSensor(uuid="grasp_plan_diagnostics"),
            ObjectContactSensor(uuid="object_contacts"),
            PickupObjectPoseSensor(uuid="pickup_obj_pose"),
        ]

    def get_all_phases(self):
        # Collect all possible phases here, even those not used for a particular trajectory computation
        phases = {
            "unknown": 0,
            "gripper-open": 1,
            "pregrasp": 2,
            "grasp": 3,
            "gripper-close": 4,
            "lift": 5,
            "preplace": 6,
            "place": 7,
            "retreat": 8,
            "go_home": 9,
        }
        return phases

    def reset(self, reset_retries: bool = True):
        # Once per episode, not once per retry. Placement is derived from where the pickup object
        # is at the instant it runs, so re-running it mid-episode rebuilds the barrier around an
        # object that is airborne in the gripper. Measured on a transport retry: the barrier rose
        # 0.52 m and shrank from 61.9 cm to 53.3 cm, because `floor_z` became the underside of the
        # lifted object and the height came from the world AABB of a tilted one. It is a mocap
        # body, so it stays there, and every plan afterwards is made against a different fixture
        # than the one the run set out to test.
        if reset_retries:
            self._place_task_barrier()
        self.object_contacts = []
        self._object_contact_seen = set()
        self.planning_failure_reason = None
        self.grasp_plan_diagnostics = {"status": "not_evaluated"}
        self._ik_solution_cache = {}
        self._continue_lift_extra = 0.0
        self._in_extra_lift = False
        if reset_retries:
            # Fresh episode (not a retry): forget grasps attempted for the previous task.
            self._tried_object_grasps = []
        if not self.ik_warmed_up:
            with Timer() as warmup_time:
                self.task.env.current_robot.parallel_kinematics.warmup_ik(
                    self.policy_config.grasp_feasibility_batch_size
                )
            self.ik_warmed_up = True
            log.info(f"Warmed up parallel IK solver in {warmup_time.value:.3f}s")

        try:
            self.action_primitives = self._compute_trajectory()
        except ValueError as error:
            expected_failure = str(error).startswith(
                (
                    "IK failed",
                    "No feasible grasp found",
                    "Could not solve ",
                    "Failed to find supporting geom id",
                )
            )
            if not expected_failure:
                raise
            log.info("Planning failed; recording a failed trajectory: %s", error)
            self.planning_failure_reason = str(error)
            self.action_primitives = [
                NoopAction(
                    self.robot_view,
                    duration=1.0,
                    caption=("PLANNING FAILURE", str(error)),
                )
            ]
            self.target_poses = {}
            self.action_idx = 0
            if reset_retries:
                self._retry_count = 0
            return

        self.action_idx = 0
        if reset_retries:
            self._retry_count = 0

        self.target_poses = {}
        for action_primitive in self.action_primitives:
            if isinstance(action_primitive, TCPMoveSequence):
                for move_segment in action_primitive.move_segments:
                    self.target_poses[move_segment.name] = move_segment.end_pose
            elif isinstance(action_primitive, JointMoveSequence):
                gripper_mg_id = self.robot_view.get_gripper_movegroup_ids()[0]
                kinematics = self.task.env.current_robot.kinematics
                for move_segment in action_primitive.move_segments:
                    end_qpos = move_segment.end_qpos
                    base_pose = self.robot_view.base.pose
                    self.target_poses[move_segment.name] = kinematics.fk(end_qpos, base_pose)[
                        gripper_mg_id
                    ]

        # Record the grasp about to be executed (in the object frame) so a later retry can
        # steer toward a different grasp site. Covers both the freshly-planned and the
        # base-sampling precomputed-plan paths, since both populate target_poses["grasp"].
        self._record_attempted_grasp()

    def _place_task_barrier(self) -> None:
        """Size and position the obstacle standing between the object and its destination.

        A test fixture, not scene generation: it exists so a run can be given something the
        straight-line policy must go through and a planner can go around. Both arms build it from
        this one method, so the barrier is identical on either side of an A/B.

        Sized and posed at episode start because neither is known earlier -- ``add_auxiliary_objects``
        runs while the scene is still being compiled, before the task sampler has chosen an object
        or a destination.
        """
        policy_config = self.config.policy_config
        if not getattr(policy_config, "barrier_enabled", False):
            return
        task_config = self.config.task_config
        destination = getattr(task_config, "place_receptacle_name", None)
        if not destination:
            return

        env = self.task.env
        model, data = env.current_model, env.current_data
        try:
            geom_id = model.geom(TASK_BARRIER_GEOM).id
            body_id = model.body(TASK_BARRIER_BODY).id
            object_manager = env.object_managers[env.current_batch_index]
            pickup = object_manager.get_object_by_name(task_config.pickup_obj_name)
            target = object_manager.get_object_by_name(destination)
        except Exception as error:
            log.warning(f"[BARRIER] not placed: {error}")
            return

        pickup_centre, pickup_dims = body_aabb(model, data, pickup.body_id, visible_only=False)
        target_centre, _ = body_aabb(model, data, target.body_id, visible_only=False)

        span = target_centre[:2] - pickup_centre[:2]
        distance = float(np.linalg.norm(span))
        if distance < 1e-6:
            log.warning("[BARRIER] object and destination coincide; not placed")
            return

        # Half the object's height, standing on whatever the object stands on.
        height = policy_config.barrier_height_fraction * float(pickup_dims[2])
        floor_z = float(pickup_centre[2] - pickup_dims[2] / 2.0)
        half_size = np.array(
            [policy_config.barrier_thickness / 2.0, policy_config.barrier_width / 2.0, height / 2.0]
        )

        model.geom_size[geom_id] = half_size
        # geom_size alone is not enough at runtime: MuJoCo broad-phases on geom_rbound and
        # body_aabb reads geom_aabb, so a resized geom that leaves those stale is both under-
        # collided and mis-measured.
        model.geom_rbound[geom_id] = float(np.linalg.norm(half_size))
        model.geom_aabb[geom_id] = np.concatenate([np.zeros(3), half_size])

        yaw = float(np.arctan2(span[1], span[0]))  # local x across the gap => thin face to it
        centre = np.array(
            [
                pickup_centre[0] + span[0] / 2.0,
                pickup_centre[1] + span[1] / 2.0,
                floor_z + height / 2.0,
            ]
        )
        data.mocap_pos[model.body_mocapid[body_id]] = centre
        data.mocap_quat[model.body_mocapid[body_id]] = R.from_euler("z", yaw).as_quat(
            scalar_first=True
        )
        mujoco.mj_forward(model, data)
        self._barrier_geom_id = geom_id
        self._barrier_contacts = 0
        self._barrier_contact_logged = False
        try:
            directory = Path(self.config.output_dir)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "barrier.json").write_text(
                json.dumps(
                    {
                        "centre": centre.tolist(),
                        "half_size": half_size.tolist(),
                        "yaw": yaw,
                        "object_height": float(pickup_dims[2]),
                        "gap": distance,
                    }
                )
            )
        except Exception as error:
            log.debug(f"[BARRIER] could not record geometry: {error}")
        log.info(
            f"[BARRIER] {policy_config.barrier_thickness * 1000:.0f} mm x "
            f"{policy_config.barrier_width * 100:.0f} cm x {height * 100:.1f} cm "
            f"(half the object's {pickup_dims[2] * 100:.1f} cm) at "
            f"{np.round(centre, 3).tolist()}, {distance * 100:.0f} cm gap"
        )

    def _get_manipulated_object_pose(self) -> np.ndarray | None:
        """World pose of the object being grasped, used for grasp-site retry tracking.

        Returns None (tracking disabled) by default; pick/pick-and-place override it.
        """
        return None

    def _record_attempted_grasp(self) -> None:
        grasp_pose_world = self.target_poses.get("grasp")
        object_pose = self._get_manipulated_object_pose()
        if grasp_pose_world is None or object_pose is None:
            return
        self._tried_object_grasps.append(np.linalg.inv(object_pose) @ grasp_pose_world)

    def get_render_override(self):
        if self.action_idx < len(self.action_primitives):
            return self.action_primitives[self.action_idx].get_render_override()
        return None

    def get_render_caption(self) -> tuple[str, ...] | None:
        if self.action_idx < len(self.action_primitives):
            return self.action_primitives[self.action_idx].get_render_caption()
        return None

    def get_info(self) -> dict:
        # (barrier_contacts is added below by the base implementation's caller)
        info = super().get_info()
        info["grasp_plan_diagnostics"] = self.grasp_plan_diagnostics
        for action_primitive in self.action_primitives:
            ik_diagnostic = getattr(action_primitive, "diagnostic", None)
            if isinstance(ik_diagnostic, IKDiagnostic):
                info["ik_diagnostic"] = ik_diagnostic.to_info()
                break
        if self.planning_failure_reason is not None:
            info["planning_failure_reason"] = self.planning_failure_reason
        return info

    def _record_grasp_phase_failure(self, phase: str, cause: str) -> None:
        self.grasp_plan_diagnostics["status"] = "phase_failure"
        self.grasp_plan_diagnostics["failure_phase"] = phase
        self.grasp_plan_diagnostics["failure_cause"] = cause

    def _record_contacts(self) -> None:
        """What the picked object and the robot are touching, this step.

        Generalises a barrier-only counter that existed because there was no way to ask "did it hit
        anything" without special-casing one geom. Nothing else in a rollout answers it: there is no
        contact channel in the saved data, ``env_states/actors`` is empty so the object's pose is
        never stored, and an episode can be scored a success having dragged the object across a
        counter with no trace of it. Reading ``data.contact`` per policy step is the cheap fix.
        """
        env = self.task.env
        model, data = env.current_model, env.current_data
        self.object_contacts = []

        pickup_name = getattr(self.config.task_config, "pickup_obj_name", None)
        object_bodies: set[int] = set()
        if pickup_name:
            try:
                body_id = env.object_managers[env.current_batch_index].get_object_by_name(
                    pickup_name
                ).body_id
                object_bodies = {
                    b for b in range(model.nbody) if _has_ancestor(model, b, body_id)
                }
            except Exception:
                object_bodies = set()

        gripper_bodies: set[int] = set()
        for group_id in self.robot_view.get_gripper_movegroup_ids():
            try:
                root = self.robot_view.get_move_group(group_id).root_body_id
            except Exception:
                continue
            gripper_bodies |= {b for b in range(model.nbody) if _has_ancestor(model, b, root)}

        for index in range(data.ncon):
            contact = data.contact[index]
            first = int(model.geom_bodyid[contact.geom1])
            second = int(model.geom_bodyid[contact.geom2])
            if not (object_bodies & {first, second}):
                continue
            other = second if first in object_bodies else first
            if other in gripper_bodies:
                continue  # the gripper holding it is not news
            self.object_contacts.append(
                {
                    "body": model.body(other).name or f"body{other}",
                    "phase": self.get_phase(),
                    "depth": float(contact.dist),
                }
            )

        # Report each body the first time it is touched. Logging only the very first contact
        # reports the surface the object was resting on and nothing else -- useless for finding
        # what it hit later.
        seen = getattr(self, "_object_contact_seen", None)
        if seen is None:
            seen = self._object_contact_seen = set()
        for name in sorted({c["body"] for c in self.object_contacts}):
            if name not in seen:
                seen.add(name)
                log.warning(
                    f"[CONTACT] picked object touched {name!r} during phase {self.get_phase()!r}"
                )

        # the barrier record stays, for the runs that use one
        geom_id = getattr(self, "_barrier_geom_id", None)
        if geom_id is None:
            return
        hits = sum(
            1
            for index in range(data.ncon)
            if geom_id in (data.contact[index].geom1, data.contact[index].geom2)
        )
        if hits:
            self._barrier_contacts = getattr(self, "_barrier_contacts", 0) + hits
            if not getattr(self, "_barrier_contact_logged", False):
                self._barrier_contact_logged = True
                log.warning(f"[BARRIER] contact during phase {self.get_phase()!r}")

    def get_action(self, info: dict[str, Any]) -> dict[str, Any]:
        self._record_contacts()
        if self._check_for_failures():
            return self._handle_failure()

        # get the action from the current action primitive, absorbing 0-length segments as necessary
        while self.action_idx < len(self.action_primitives):  # guaranteed to terminate
            act_prim = self.action_primitives[self.action_idx]
            proceed = act_prim.execute()
            # cannot have a 0-length segment that doesn't proceed
            assert proceed or act_prim.duration > 0.0
            if proceed:
                self.action_idx += 1
            if not proceed or act_prim.duration > 0.0:
                break
        if self.action_idx < len(self.action_primitives):
            action = act_prim.get_current_action()
        else:
            if self._lifts_until_success():
                # The planned lift trajectory has run in full. If the pick isn't a clean success
                # yet, keep raising the closed gripper (extra lift). We do NOT signal 'done' on
                # success here: the task ends the episode the moment the CURRENT (post-step) state is
                # a clean pick (Task.is_terminal + in_extra_lift), so termination lands exactly on
                # the success step and no further step can drop a marginally-held object. The planned
                # lift is never truncated — this branch only runs once the planned primitives are
                # exhausted. Once the full extra allowance has been raised without a clean pick,
                # give up (rather than holding at the top until the task horizon).
                self._in_extra_lift = True
                if self._continue_lift_extra >= self.policy_config.continue_lift_max_extra:
                    action = self.action_primitives[-1].get_current_action()
                    action["done"] = True
                else:
                    action = self._continue_lift_action()
                    if action is None:
                        action = self.action_primitives[-1].get_current_action()
                        action["done"] = True
            else:
                action = self.action_primitives[-1].get_current_action()
                action["done"] = True
        return action

    def _lifts_until_success(self) -> bool:
        """Whether this policy keeps lifting past its planned lift until the pick is a clean success
        (or the horizon times out). Overridden to True by the pick policy; other policies end when
        their planned trajectory completes."""
        return False

    def in_extra_lift(self) -> bool:
        """True once a lift-until-success policy is past its planned lift and is doing extra lifts.
        The task reads this to end the episode on the current-step success without truncating the
        planned lift."""
        return getattr(self, "_in_extra_lift", False)

    def _continue_lift_action(self) -> dict[str, Any] | None:
        """Raise the (closed) gripper incrementally past the planned lift pose so the pick keeps
        lifting until it succeeds. Holds at the reachable maximum once the arm can no longer go
        higher. Returns None if there is no lift pose to build on (e.g. a planning failure)."""
        lift_pose = self.target_poses.get("lift")
        if lift_pose is None:
            return None
        self._continue_lift_extra = min(
            self.policy_config.continue_lift_max_extra,
            self._continue_lift_extra + self.policy_config.continue_lift_step,
        )
        target = lift_pose.copy()
        target[2, 3] += self._continue_lift_extra
        gripper_mgs = set(self.robot_view.get_gripper_movegroup_ids())
        arm_mgs = [mg for mg in self.robot_view.move_group_ids() if mg not in gripper_mgs]
        jp = self._get_cached_ik_solution(target)
        if jp is None:
            kinematics = self.task.env.current_robot.kinematics
            jp = kinematics.ik(
                self.robot_view.get_gripper_movegroup_ids()[0],
                target,
                arm_mgs,
                self.robot_view.get_qpos_dict(),
                self.robot_view.base.pose,
            )
        # get_ctrl_dict preserves the current (closed) gripper command; only the arm is retargeted.
        action = self.robot_view.get_ctrl_dict()
        if jp is not None:
            action.update({mg: jp[mg] for mg in arm_mgs})
        return action

    def get_phase(self) -> str:
        if self.action_idx < len(self.action_primitives):
            act_prim = self.action_primitives[self.action_idx]
        else:
            act_prim = self.action_primitives[-1]
        return act_prim.get_current_phase()

    def get_current_phase_target_pose(self) -> np.ndarray | None:
        phase = self.get_phase()
        target_pose = self.target_poses.get(phase)
        if target_pose is not None:
            return target_pose.copy()
        if self.action_idx < len(self.action_primitives):
            diagnostic = getattr(self.action_primitives[self.action_idx], "diagnostic", None)
            if isinstance(diagnostic, IKDiagnostic) and diagnostic.phase == phase:
                return diagnostic.target_pose.copy()
        return None

    @abstractmethod
    def _compute_trajectory(self) -> list[ActionPrimitive]:
        raise NotImplementedError

    def _check_for_failures(self) -> bool:
        # TODO(abhayd): check for collision with other objects
        if self.action_idx >= len(self.action_primitives):
            # For lift-until-success policies, exhausting the planned primitives is NOT a failure:
            # act() keeps raising the gripper until the pick succeeds or the episode times out.
            # Otherwise, exhaustion here means the sequence over-ran and triggers the usual handling.
            return not self._lifts_until_success()
        action_primitive = self.action_primitives[self.action_idx]
        return action_primitive.check_failure()

    def _handle_failure(self) -> dict[str, Any]:
        if self.retry_count >= self.policy_config.max_retries:
            for index in range(self.action_idx + 1, len(self.action_primitives)):
                diagnostic_action = self.action_primitives[index]
                if isinstance(diagnostic_action, IKDiagnosticAction):
                    log.info("Max retries exceeded; advancing to saved diagnostic frame")
                    self.action_idx = index
                    diagnostic_action.reset()
                    return self.robot_view.get_noop_ctrl_dict()
            log.info(f"❌ Max retries ({self.policy_config.max_retries}) exceeded. Task failed.")
            return {"done": True, "success": False}

        self._retry_count += 1
        log.info(
            f"🔄 Failure detected! Initiating retry {self.retry_count}/{self.policy_config.max_retries}"
        )
        self.reset(reset_retries=False)

        action = self.robot_view.get_noop_ctrl_dict()
        return action

    def _tcp_to_jp_fn(self, mg_id: str, target_pose: np.ndarray) -> dict[str, Any]:
        kinematics = self.task.env.current_robot.kinematics

        gripper_mgs = set(self.robot_view.get_gripper_movegroup_ids())
        mgs_except_gripper = [x for x in self.robot_view.move_group_ids() if x not in gripper_mgs]

        jp = self._get_cached_ik_solution(target_pose)
        if jp is None:
            jp = kinematics.ik(
                mg_id,
                target_pose,
                mgs_except_gripper,
                self.robot_view.get_qpos_dict(),
                self.robot_view.base.pose,
            )

        action = self.robot_view.get_ctrl_dict()
        if jp is not None:
            self.sequential_ik_failures = 0
            action.update({mg_id: jp[mg_id] for mg_id in mgs_except_gripper})
        else:
            self.sequential_ik_failures += 1
            log.info(f"⚠️ IK failed, holding current position, fails:{self.sequential_ik_failures}")
            if self.sequential_ik_failures >= self.policy_config.max_sequential_ik_failures:
                log.info("❌ Too many sequential IK failures, triggering retry.")
                return self._handle_failure()

        return action

    def _augment_grasp_candidates(self, grasps: np.ndarray) -> np.ndarray:
        y_angles = self.policy_config.grasp_closing_axis_rotation_degrees
        x_angles = self.policy_config.grasp_local_x_rotation_degrees
        if x_angles == (0.0,) and y_angles == (0.0,):
            return grasps
        variants = []
        for x_angle in x_angles:
            for y_angle in y_angles:
                transform = (
                    R.from_euler("x", x_angle, degrees=True).as_matrix()
                    @ R.from_euler("y", y_angle, degrees=True).as_matrix()
                )
                rolled = grasps.copy()
                rolled[:, :3, :3] = rolled[:, :3, :3] @ transform
                variants.append(rolled)
        return np.concatenate(variants)

    def _consume_all_phase_ik_plan(self) -> dict[str, np.ndarray] | None:
        plans = getattr(self.task.env, "_all_phase_ik_plans", None)
        if not plans:
            return None
        pose_key = np.round(self.robot_view.base.pose, decimals=8).tobytes()
        plan = plans.pop(pose_key, None)
        if plan is None or not isinstance(self, plan["policy_cls"]):
            return None
        self.grasp_plan_diagnostics = copy.deepcopy(plan["grasp_plan_diagnostics"])
        self._ik_solution_cache = {
            key: {
                group_id: values.copy() for group_id, values in solution.items()
            }
            for key, solution in plan["ik_solution_cache"].items()
        }
        return {
            phase: pose.copy() for phase, pose in plan["target_poses"].items()
        }

    @staticmethod
    def _ik_cache_key(pose: np.ndarray) -> bytes:
        return np.round(np.asarray(pose, dtype=np.float64), decimals=8).tobytes()

    def _cache_ik_solution(
        self,
        pose: np.ndarray,
        solution: dict[str, np.ndarray],
    ) -> None:
        self._ik_solution_cache[self._ik_cache_key(pose)] = {
            group_id: np.asarray(values).copy() for group_id, values in solution.items()
        }

    def _get_cached_ik_solution(self, pose: np.ndarray) -> dict[str, np.ndarray] | None:
        solution = self._ik_solution_cache.get(self._ik_cache_key(pose))
        if solution is None:
            return None
        return {group_id: values.copy() for group_id, values in solution.items()}

    def _multistart_ik_for_pose(
        self,
        pose: np.ndarray,
        attempts: int = 24,
    ) -> dict[str, np.ndarray] | None:
        robot = self.task.env.current_robot
        robot_view = robot.robot_view
        arm_group_ids = robot.get_arm_move_group_ids()
        if not arm_group_ids:
            return None
        arm_group_id = arm_group_ids[0]
        tool_group_id = robot_view.get_gripper_movegroup_ids()[0]
        limits = robot_view.get_move_group(arm_group_id).joint_pos_limits
        rng = np.random.default_rng(self.config.seed)
        qpos = robot_view.get_qpos_dict()
        best_diagnostic = None
        for _ in range(attempts):
            seed_qpos = {group_id: values.copy() for group_id, values in qpos.items()}
            seed_qpos[arm_group_id] = rng.uniform(limits[:, 0], limits[:, 1])
            result = robot.kinematics.ik(
                tool_group_id,
                pose,
                [arm_group_id],
                seed_qpos,
                robot_view.base.pose,
                rel_to_base=False,
            )
            diagnostic = robot.kinematics.last_ik_diagnostic
            if diagnostic is not None and (
                best_diagnostic is None
                or diagnostic["minimum_error"] < best_diagnostic["minimum_error"]
            ):
                best_diagnostic = diagnostic
            if result is not None:
                self._cache_ik_solution(pose, result)
                return result
        robot.kinematics.last_ik_diagnostic = best_diagnostic
        return None

    def _capture_ik_diagnostic(
        self,
        phase: str,
        pickup_obj=None,
        grasp_pose: np.ndarray | None = None,
        holding_object: bool = False,
    ) -> IKDiagnostic | None:
        robot = self.task.env.current_robot
        solver_diagnostic = robot.kinematics.last_ik_diagnostic
        if solver_diagnostic is None or solver_diagnostic.get("best_qpos") is None:
            return None

        live_data = self.task.env.current_data
        model = live_data.model
        data = mujoco.MjData(model)
        mujoco.mj_copyData(data, model, live_data)
        if self.config.robot_config.robot_view_factory is None:
            return None
        scratch_view = self.config.robot_config.robot_view_factory(
            data,
            self.config.robot_config.robot_namespace,
        )
        scratch_view.set_qpos_dict(solver_diagnostic["best_qpos"])
        mujoco.mj_fwdPosition(model, data)

        if holding_object and pickup_obj is not None and grasp_pose is not None:
            tool_group_id = scratch_view.get_gripper_movegroup_ids()[0]
            tool_to_pickup = np.linalg.inv(grasp_pose) @ pickup_obj.pose
            attached_body = create_mlspaces_body(data, pickup_obj.name)
            attached_body.pose = (
                scratch_view.get_move_group(tool_group_id).leaf_frame_to_world @ tool_to_pickup
            )
            mujoco.mj_fwdPosition(model, data)

        arm_group_ids = robot.get_arm_move_group_ids()
        highlight_geom_ids: set[int] = set()
        if arm_group_ids:
            arm_root_id = scratch_view.get_move_group(arm_group_ids[0]).root_body_id
            highlight_geom_ids.update(descendant_geoms(model, arm_root_id, visible_only=True))
        return IKDiagnostic(
            data=data,
            highlight_geom_ids=tuple(sorted(highlight_geom_ids)),
            highlight_rgba=(1.0, 0.5, 0.0, 1.0),
            phase=phase,
            minimum_error=float(solver_diagnostic["minimum_error"]),
            position_error=float(solver_diagnostic["position_error"]),
            rotation_error=float(solver_diagnostic["rotation_error"]),
            best_iteration=int(solver_diagnostic["best_iteration"]),
            target_pose=np.asarray(solver_diagnostic["target_pose"]).copy(),
        )

    def _batch_ik_for_pose(
        self,
        pose: np.ndarray,
        qpos: dict[str, np.ndarray],
        tool_group_id: str,
        base_pose: np.ndarray,
    ) -> dict[str, np.ndarray] | None:
        batch_size = self.policy_config.grasp_feasibility_batch_size
        pose_batch = np.broadcast_to(pose, (batch_size, 4, 4)).copy()
        results = self.task.env.current_robot.parallel_kinematics.ik(
            tool_group_id,
            pose_batch,
            None,
            {group_id: values.copy() for group_id, values in qpos.items()},
            base_pose,
            rel_to_base=False,
        )
        return results[0]

    def check_feasible_ik(self, pose: np.ndarray) -> bool:
        if pose.ndim > 2:
            assert pose.shape[1:] == (4, 4)
            batch_size = pose.shape[0]
            assert batch_size <= self.policy_config.grasp_feasibility_batch_size

            if batch_size == 0:
                return np.array([], dtype=bool)

            # pad to the batch size to avoid triggering recompilation of the IK solver
            if batch_size < self.policy_config.grasp_feasibility_batch_size:
                n_pad = self.policy_config.grasp_feasibility_batch_size - batch_size
                pose = np.concatenate([pose, np.broadcast_to(pose[-1:], (n_pad, 4, 4))])

            robot_view = self.task.env.current_robot.robot_view
            parallel_kinematics = self.task.env.current_robot.parallel_kinematics
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            jp_dicts = parallel_kinematics.ik(
                gripper_mg_id,
                pose,
                None,
                robot_view.get_qpos_dict(),
                robot_view.base.pose,
                rel_to_base=False,
            )
            return np.array([jp_dict is not None for jp_dict in jp_dicts[:batch_size]])
        else:
            assert pose.shape == (4, 4)
            robot_view = self.task.env.current_robot.robot_view
            kinematics = self.task.env.current_robot.kinematics
            gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
            jp_dict = self._get_cached_ik_solution(pose)
            if jp_dict is None:
                jp_dict = kinematics.ik(
                    gripper_mg_id,
                    pose,
                    robot_view.move_group_ids(),
                    robot_view.get_qpos_dict(),
                    base_pose=robot_view.base.pose,
                )
            if jp_dict is None:
                jp_dict = self._batch_ik_for_pose(
                    pose,
                    robot_view.get_qpos_dict(),
                    gripper_mg_id,
                    robot_view.base.pose,
                )
            return jp_dict is not None

    def _show_poses(self, poses, style, color=(1, 0, 0, 1)) -> None:
        if self.task.viewer is None:
            return

        assert poses.ndim == 3 and poses.shape[1:] == (4, 4)
        viewer = self.task.viewer
        ngeom = viewer.user_scn.ngeom
        # Define relative parts of the gripper

        half_length = self.policy_config.grasp_length / 2
        half_width = self.policy_config.grasp_width / 2
        grasp_base_pos = np.array(self.policy_config.grasp_base_pos)
        gripper_parts = [
            ("sphere", mujoco.mjtGeom.mjGEOM_SPHERE, [0.003, 0, 0], [0.0, 0.0, 0]),
            (
                "cylinder_left",
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [0.003, half_length, 0],
                np.array([0.0, half_width, half_length]) + grasp_base_pos,
            ),
            (
                "cylinder_right",
                mujoco.mjtGeom.mjGEOM_CYLINDER,
                [0.003, half_length, 0],
                np.array([0.0, -half_width, half_length]) + grasp_base_pos,
            ),
            (
                "connecting_bar",
                mujoco.mjtGeom.mjGEOM_BOX,
                [0.002, half_width, 0.002],
                grasp_base_pos,
            ),
        ]
        i = 0
        for T in poses:
            for _part_name, geom_type, size, offset in gripper_parts:
                offset_trf = np.eye(4)
                offset_trf[:3, 3] = offset
                A = T.copy() @ offset_trf
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[ngeom + i],
                    type=geom_type,
                    size=np.array(size),
                    pos=A[:3, 3],
                    mat=T[:3, :3].flatten(),
                    rgba=color,
                )
                i += 1
        viewer.user_scn.ngeom = ngeom + i

    @staticmethod
    def add_auxiliary_objects(config: MlSpacesExpConfig, spec: MjSpec) -> None:
        PlannerPolicy.add_auxiliary_objects(config, spec)
        policy_config = config.policy_config
        assert isinstance(policy_config, ObjectManipulationPlannerPolicyConfig)
        if policy_config.barrier_enabled:
            # Parked below the floor: its size and pose depend on where the task objects end up,
            # which is not known until the episode is sampled. _place_task_barrier moves it.
            body = spec.worldbody.add_body(
                name=TASK_BARRIER_BODY, pos=[0.0, 0.0, -10.0], mocap=True
            )
            body.add_geom(
                name=TASK_BARRIER_GEOM,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.01, 0.1, 0.05],
                rgba=list(policy_config.barrier_rgba),
                group=0,
                contype=1,
                conaffinity=1,
            )
        if policy_config.filter_colliding_grasps:
            add_grasp_collision_bodies(
                spec,
                policy_config.grasp_collision_batch_size,
                policy_config.grasp_width,
                policy_config.grasp_length,
                policy_config.grasp_height,
                np.array(policy_config.grasp_base_pos),
            )
