import logging

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as R

from molmo_spaces.configs.task_configs import PickAndPlaceTaskConfig
from molmo_spaces.env.data_views import MlSpacesObject, create_mlspaces_body
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    BaseObjectManipulationPlannerPolicy,
    GripperAction,
    IKDiagnosticAction,
    JointMoveSegment,
    JointMoveSequence,
    NoopAction,
    TCPMoveSegment,
    TCPMoveSequence,
)
from molmo_spaces.utils.grasp_sample import select_grasp_pose
from molmo_spaces.utils.grasps import get_pickup_grasps
from molmo_spaces.utils.mj_model_and_data_utils import body_aabb

log = logging.getLogger(__name__)


class TargetPoseIKFailure(ValueError):
    def __init__(
        self,
        phase: str,
        target_poses: dict[str, np.ndarray],
        diagnostic=None,
    ) -> None:
        super().__init__(f"IK failed for {phase} pose")
        self.phase = phase
        self.target_poses = target_poses
        self.diagnostic = diagnostic


class PickAndPlacePlannerPolicy(BaseObjectManipulationPlannerPolicy):
    def _uses_independent_placement_orientations(self) -> bool:
        return self.policy_config.placement_local_x_rotation_degrees != (
            0.0,
        ) or self.policy_config.placement_local_y_rotation_degrees != (0.0,)

    def _placement_orientation_offsets(self) -> list[tuple[float, float]]:
        offsets = [
            (x_degrees, y_degrees)
            for x_degrees in self.policy_config.placement_local_x_rotation_degrees
            for y_degrees in self.policy_config.placement_local_y_rotation_degrees
        ]
        return sorted(
            offsets,
            key=lambda offset: (
                abs(offset[0]) + abs(offset[1]),
                abs(offset[0]),
                abs(offset[1]),
                offset,
            ),
        )

    def _get_candidate_grasps(self, pickup_obj: MlSpacesObject) -> np.ndarray:
        grasps = get_pickup_grasps(
            self.task.env,
            pickup_obj,
            grasp_libraries=self.policy_config.grasp_libraries,
        )
        return self._augment_grasp_candidates(grasps)

    def _get_manipulated_object_pose(self) -> np.ndarray | None:
        task_config = self.config.task_config
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        return om.get_object_by_name(task_config.pickup_obj_name).pose

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        robot_view = self.task.env.current_robot.robot_view
        gripper_mg_id = robot_view.get_gripper_movegroup_ids()[0]
        start_ee_pose = robot_view.get_move_group(gripper_mg_id).leaf_frame_to_world
        failure = None
        target_poses = self._consume_all_phase_ik_plan()
        if target_poses is None:
            try:
                target_poses = self._compute_target_poses()
            except TargetPoseIKFailure as error:
                target_poses = error.target_poses
                failure = error
                self.planning_failure_reason = str(error)
                self._record_grasp_phase_failure(error.phase, str(error))

        if "pregrasp" not in target_poses:
            assert failure is not None
            if failure.diagnostic is not None:
                return [IKDiagnosticAction(robot_view, failure.diagnostic)]
            return [
                NoopAction(
                    robot_view,
                    1.0,
                    phase=failure.phase,
                    caption=(f"IK FAILURE: {failure.phase.upper()}", str(failure)),
                )
            ]

        action_primitives: list[ActionPrimitive] = [
            GripperAction(robot_view, True, 0.0),
        ]
        pickup_segments = [
            TCPMoveSegment(
                name="pregrasp",
                start_pose=start_ee_pose,
                end_pose=target_poses["pregrasp"],
                speed=self.policy_config.speed_fast,
            )
        ]
        if "grasp" in target_poses:
            pickup_segments.append(
                TCPMoveSegment(
                    name="grasp",
                    start_pose=target_poses["pregrasp"],
                    end_pose=target_poses["grasp"],
                    speed=self.policy_config.speed_slow,
                )
            )
        # When the grasp orientation was only reachable best-effort (the arm cannot achieve the
        # requested wrist rotation exactly), relax the rotation-tracking threshold so it proceeds
        # with the closest achievable orientation instead of retrying (mirrors best-effort-lift).
        best_effort_orient = "orientation_fallback" in self.grasp_plan_diagnostics
        grasp_rot_thr = np.inf if best_effort_orient else self.policy_config.tcp_rot_err_threshold
        action_primitives.append(
            TCPMoveSequence(
                robot_view,
                self._tcp_to_jp_fn,
                self.policy_config.move_settle_time,
                gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                tcp_rot_err_threshold=grasp_rot_thr,
                move_segments=pickup_segments,
            )
        )

        if "grasp" in target_poses:
            action_primitives.append(
                GripperAction(
                    robot_view,
                    False,
                    self.policy_config.gripper_close_duration,
                )
            )
            holding_segments = []
            previous_phase = "grasp"
            for phase, speed in (
                ("lift", self.policy_config.speed_slow),
                ("preplace", self.policy_config.speed_fast),
                ("place", self.policy_config.speed_slow),
            ):
                if phase not in target_poses:
                    break
                segment_speed = (
                    self.policy_config.speed_slow
                    if phase == "preplace" and self._uses_independent_placement_orientations()
                    else speed
                )
                holding_segments.append(
                    TCPMoveSegment(
                        name=phase,
                        start_pose=target_poses[previous_phase],
                        end_pose=target_poses[phase],
                        speed=segment_speed,
                        decoupled_interpolation=(
                            phase == "preplace" and self._uses_independent_placement_orientations()
                        ),
                    )
                )
                previous_phase = phase
            if holding_segments:
                # When the lift was only reachable best-effort, the arm may not fully track the
                # (reduced) lift waypoint. Don't treat that tracking error as a failure/retry —
                # press on to preplace/place instead. Object-drop detection (is_holding_object)
                # still applies, so a genuinely dropped object still triggers a retry.
                best_effort_lift = "lift_fallback" in self.grasp_plan_diagnostics
                holding_pos_thr = (
                    np.inf if best_effort_lift else self.policy_config.tcp_pos_err_threshold
                )
                holding_rot_thr = (
                    np.inf
                    if (best_effort_lift or best_effort_orient)
                    else self.policy_config.tcp_rot_err_threshold
                )
                action_primitives.append(
                    TCPMoveSequence(
                        robot_view,
                        self._tcp_to_jp_fn,
                        self.policy_config.move_settle_time,
                        is_holding_object=True,
                        gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                        tcp_pos_err_threshold=holding_pos_thr,
                        tcp_rot_err_threshold=holding_rot_thr,
                        move_segments=holding_segments,
                    )
                )

        if failure is not None:
            if failure.diagnostic is not None:
                action_primitives.append(IKDiagnosticAction(robot_view, failure.diagnostic))
            else:
                action_primitives.append(
                    NoopAction(
                        robot_view,
                        1.0,
                        phase=failure.phase,
                        caption=(f"IK FAILURE: {failure.phase.upper()}", str(failure)),
                    )
                )
            return action_primitives

        action_primitives.extend(
            [
                GripperAction(robot_view, True, self.policy_config.gripper_open_duration),
                TCPMoveSequence(
                    robot_view,
                    self._tcp_to_jp_fn,
                    self.policy_config.move_settle_time,
                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                    tcp_pos_err_threshold=self.policy_config.tcp_pos_err_threshold,
                    tcp_rot_err_threshold=self.policy_config.tcp_rot_err_threshold,
                    move_segments=[
                        TCPMoveSegment(
                            name="retreat",
                            start_pose=target_poses["place"],
                            end_pose=target_poses["postplace"],
                            speed=self.policy_config.speed_fast,
                        )
                    ],
                ),
                JointMoveSequence(
                    robot_view,
                    self.policy_config.move_settle_time,
                    gripper_empty_threshold=self.policy_config.gripper_empty_threshold,
                    move_segments=[
                        JointMoveSegment(
                            name="go_home",
                            start_qpos=None,
                            end_qpos=self.config.robot_config.init_qpos,
                            duration_s=4.0,
                        )
                    ],
                ),
                NoopAction(robot_view, 2.0),
            ]
        )
        return action_primitives

    def _get_grasp_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
        robot_view,
        task_config: PickAndPlaceTaskConfig,
        allow_best_effort_lift: bool = False,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        randomize_pregrasp = False
        if randomize_pregrasp:
            # Random height variations
            pregrasp_height_offset = np.random.uniform(
                -self.policy_config.pregrasp_height_noise,
                self.policy_config.pregrasp_height_noise,
            )
            postgrasp_height_offset = np.random.uniform(
                -self.policy_config.postgrasp_height_noise,
                self.policy_config.postgrasp_height_noise,
            )
        else:
            pregrasp_height_offset = 0.0
            postgrasp_height_offset = 0.0

        pregrasp_pose = grasp_pose_world.copy()
        # Pregrasp pose - above the pickup object with randomization
        pregrasp_pose[:3, 3] -= (
            self.policy_config.pregrasp_z_offset + pregrasp_height_offset
        ) * pregrasp_pose[:3, 2]

        log.debug(f"  - obj_start (p): {pickup_obj.position}")
        log.debug(f"  - obj_start (t): {task_config.pickup_obj_start_pose}")
        log.debug(f"  - obj_end (t): {task_config.pickup_obj_goal_pose}")
        log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")

        if not self.check_feasible_ik(pregrasp_pose):
            log.debug("  - ❌ IK FAILED for pregrasp pose!")
            log.debug(f"  - Pregrasp position: {pregrasp_pose[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {pregrasp_pose[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            raise TargetPoseIKFailure(
                "pregrasp",
                {},
                self._capture_ik_diagnostic("pregrasp"),
            )

        log.debug(f"  - Grasp pose position: {grasp_pose_world[:3, 3]}")
        log.debug(
            f"  - Grasp height above robot base: {grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
        )
        if not self.check_feasible_ik(grasp_pose_world):
            log.debug("  - ❌ IK FAILED for grasp pose!")
            log.debug(f"  - Grasp position: {grasp_pose_world[:3, 3]}")
            log.debug(f"  - Robot base: {robot_view.base.pose[:3, 3]}")
            log.debug(
                f"  - Height difference: {grasp_pose_world[2, 3] - robot_view.base.pose[2, 3]:.3f}m"
            )
            raise TargetPoseIKFailure(
                "grasp",
                {"pregrasp": pregrasp_pose},
                self._capture_ik_diagnostic("grasp"),
            )

        # Lift pose - above grasp position
        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)
        lift_pose = grasp_pose_world.copy()
        lift_pose[2, 3] = (
            receptacle_top_z
            + pickup_obj_clearance_offset
            + self.policy_config.place_z_offset
            + postgrasp_height_offset
        )

        if not self.check_feasible_ik(lift_pose):
            log.debug("  - ❌ IK FAILED for lift pose!")
            if allow_best_effort_lift:
                lift_pose = self._get_best_effort_lift_pose(grasp_pose_world, lift_pose)
                return pregrasp_pose, grasp_pose_world, lift_pose
            raise TargetPoseIKFailure(
                "lift",
                {
                    "pregrasp": pregrasp_pose,
                    "grasp": grasp_pose_world,
                },
                self._capture_ik_diagnostic(
                    "lift",
                    pickup_obj=pickup_obj,
                    grasp_pose=grasp_pose_world,
                    holding_object=True,
                ),
            )

        return pregrasp_pose, grasp_pose_world, lift_pose

    def _get_best_effort_lift_pose(
        self,
        grasp_pose: np.ndarray,
        requested_lift_pose: np.ndarray,
    ) -> np.ndarray:
        requested_height = max(
            float(requested_lift_pose[2, 3] - grasp_pose[2, 3]),
            0.0,
        )
        num_steps = max(1, int(np.ceil(requested_height / 0.01 - 1e-9)))
        selected_pose = grasp_pose.copy()
        selected_height = 0.0
        for height in np.linspace(requested_height, 0.0, num_steps + 1)[1:-1]:
            candidate = requested_lift_pose.copy()
            candidate[2, 3] = grasp_pose[2, 3] + height
            if self.check_feasible_ik(candidate):
                selected_pose = candidate
                selected_height = float(height)
                break

        self.grasp_plan_diagnostics["lift_fallback"] = {
            "status": (
                "selected_reduced_height" if selected_height > 0.0 else "selected_zero_height"
            ),
            "requested_height": requested_height,
            "selected_height": selected_height,
        }
        log.info(
            "Using best-effort lift height %.3f m instead of %.3f m",
            selected_height,
            requested_height,
        )
        return selected_pose

    def _get_independent_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        model = self.task.env.current_data.model
        live_data = self.task.env.current_data
        receptacle_center, receptacle_size = body_aabb(
            model,
            live_data,
            place_receptacle.object_id,
        )
        receptacle_top_z = receptacle_center[2] + receptacle_size[2] / 2
        tool_to_object = np.linalg.inv(grasp_pose_world) @ pickup_obj.pose

        scratch_data = mujoco.MjData(model)
        mujoco.mj_copyData(scratch_data, model, live_data)
        scratch_pickup = create_mlspaces_body(scratch_data, pickup_obj.name)
        attempts = []
        feasible_poses = []
        orientation_candidates = []
        best_failure = None
        best_failure_error = float("inf")
        best_place_failure = None
        best_place_failure_error = float("inf")

        for x_degrees, y_degrees in self._placement_orientation_offsets():
            rotation_x = R.from_euler("x", x_degrees, degrees=True).as_matrix()
            rotation_y = R.from_euler("y", y_degrees, degrees=True).as_matrix()
            place_pose = grasp_pose_world.copy()
            place_pose[:3, :3] = grasp_pose_world[:3, :3] @ (rotation_x @ rotation_y)

            object_offset = place_pose[:3, :3] @ tool_to_object[:3, 3]
            place_pose[:2, 3] = place_receptacle.position[:2] - object_offset[:2]

            # Shift the tool so the rotated object's actual AABB rests on the receptacle.
            scratch_pickup.pose = place_pose @ tool_to_object
            mujoco.mj_fwdPosition(model, scratch_data)
            object_center, object_size = body_aabb(
                model,
                scratch_data,
                scratch_pickup.body_id,
            )
            object_bottom_z = object_center[2] - object_size[2] / 2
            place_pose[2, 3] += receptacle_top_z - object_bottom_z

            preplace_pose = place_pose.copy()
            preplace_pose[2, 3] += self.policy_config.place_z_offset
            attempt = {
                "x_degrees": x_degrees,
                "y_degrees": y_degrees,
                "preplace_feasible": False,
                "place_feasible": False,
            }
            preplace_failure = None
            if self.check_feasible_ik(preplace_pose):
                attempt["preplace_feasible"] = True
            else:
                preplace_diagnostic = self._capture_ik_diagnostic(
                    "preplace",
                    pickup_obj=pickup_obj,
                    grasp_pose=grasp_pose_world,
                    holding_object=True,
                )
                preplace_failure = TargetPoseIKFailure("preplace", {}, preplace_diagnostic)

            place_failure = None
            place_best_effort = False
            if self.check_feasible_ik(place_pose):
                attempt["place_feasible"] = True
            else:
                solver_diagnostic = self.task.env.current_robot.kinematics.last_ik_diagnostic
                place_diagnostic = self._capture_ik_diagnostic(
                    "place",
                    pickup_obj=pickup_obj,
                    grasp_pose=grasp_pose_world,
                    holding_object=True,
                )
                place_failure = TargetPoseIKFailure("place", {}, place_diagnostic)
                if (
                    solver_diagnostic is not None
                    and solver_diagnostic.get("best_qpos") is not None
                    and solver_diagnostic["position_error"]
                    <= self.policy_config.placement_fallback_position_tolerance
                    and solver_diagnostic["rotation_error"]
                    <= self.policy_config.placement_fallback_rotation_tolerance
                ):
                    place_best_effort = True
                    attempt["place_best_effort"] = True
                    attempt["place_position_error"] = float(solver_diagnostic["position_error"])
                    attempt["place_rotation_error"] = float(solver_diagnostic["rotation_error"])
                    self._cache_ik_solution(place_pose, solver_diagnostic["best_qpos"])

            failed_phases = [
                phase
                for phase, failure in (
                    ("preplace", preplace_failure),
                    ("place", place_failure),
                )
                if failure is not None
            ]
            if failed_phases:
                attempt["failure_phase"] = "+".join(failed_phases)
            attempts.append(attempt)
            orientation_candidates.append(
                {
                    "x_degrees": x_degrees,
                    "y_degrees": y_degrees,
                    "preplace_pose": preplace_pose,
                    "place_pose": place_pose,
                    "preplace_feasible": preplace_failure is None,
                    "place_feasible": place_failure is None,
                    "place_usable": place_failure is None or place_best_effort,
                    "place_exact": place_failure is None,
                }
            )

            if preplace_failure is None and place_failure is None:
                postplace_pose = place_pose.copy()
                postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]
                feasible_poses.append(
                    (
                        x_degrees,
                        y_degrees,
                        preplace_pose,
                        place_pose,
                        postplace_pose,
                    )
                )

            for failure in (preplace_failure, place_failure):
                if failure is None:
                    continue
                diagnostic_error = (
                    failure.diagnostic.minimum_error
                    if failure.diagnostic is not None
                    else float("inf")
                )
                if diagnostic_error < best_failure_error:
                    best_failure = failure
                    best_failure_error = diagnostic_error
            if place_failure is not None:
                diagnostic_error = (
                    place_failure.diagnostic.minimum_error
                    if place_failure.diagnostic is not None
                    else float("inf")
                )
                if diagnostic_error < best_place_failure_error:
                    best_place_failure = place_failure
                    best_place_failure_error = diagnostic_error

        search_diagnostic = {
            "attempts": attempts,
            "num_orientations": len(attempts),
            "num_feasible": len(feasible_poses),
        }
        if feasible_poses:
            x_degrees, y_degrees, preplace_pose, place_pose, postplace_pose = feasible_poses[0]
            search_diagnostic.update(
                {
                    "status": "selected",
                    "selected_x_degrees": x_degrees,
                    "selected_y_degrees": y_degrees,
                }
            )
            self.grasp_plan_diagnostics["placement_orientation_search"] = search_diagnostic
            return preplace_pose, place_pose, postplace_pose

        requested_clearance = float(self.policy_config.place_z_offset)
        fallback_poses = []
        num_clearance_steps = max(1, int(np.ceil(requested_clearance / 0.01 - 1e-9)))
        clearance_candidates = np.linspace(requested_clearance, 0.0, num_clearance_steps + 1)
        for candidate, attempt in zip(orientation_candidates, attempts):
            if not candidate["place_usable"]:
                continue
            if candidate["preplace_feasible"]:
                selected_preplace = candidate["preplace_pose"]
                selected_clearance = requested_clearance
            else:
                selected_preplace = None
                selected_clearance = 0.0
                for clearance in clearance_candidates[1:]:
                    trial_preplace = candidate["place_pose"].copy()
                    trial_preplace[2, 3] += clearance
                    if clearance == 0.0 or self.check_feasible_ik(trial_preplace):
                        selected_preplace = trial_preplace
                        selected_clearance = float(clearance)
                        break
            if selected_preplace is None:
                continue
            attempt["fallback_preplace_clearance"] = selected_clearance
            postplace_pose = candidate["place_pose"].copy()
            postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]
            fallback_poses.append(
                (
                    selected_clearance,
                    candidate["x_degrees"],
                    candidate["y_degrees"],
                    selected_preplace,
                    candidate["place_pose"],
                    postplace_pose,
                )
            )

        if fallback_poses:
            fallback_poses.sort(key=lambda candidate: -candidate[0])
            (
                selected_clearance,
                x_degrees,
                y_degrees,
                preplace_pose,
                place_pose,
                postplace_pose,
            ) = fallback_poses[0]
            search_diagnostic.update(
                {
                    "status": "selected_place_priority_fallback",
                    "selected_x_degrees": x_degrees,
                    "selected_y_degrees": y_degrees,
                    "requested_preplace_clearance": requested_clearance,
                    "selected_preplace_clearance": selected_clearance,
                    "selected_place_exact": next(
                        candidate["place_exact"]
                        for candidate in orientation_candidates
                        if candidate["x_degrees"] == x_degrees
                        and candidate["y_degrees"] == y_degrees
                    ),
                }
            )
            self.grasp_plan_diagnostics["placement_orientation_search"] = search_diagnostic
            return preplace_pose, place_pose, postplace_pose

        search_diagnostic["status"] = "failed"
        self.grasp_plan_diagnostics["placement_orientation_search"] = search_diagnostic
        if best_place_failure is not None:
            raise best_place_failure
        if best_failure is not None:
            raise best_failure
        raise TargetPoseIKFailure("preplace", {}, None)

    def _get_placement_poses(
        self,
        grasp_pose_world: np.ndarray,
        pickup_obj: MlSpacesObject,
        place_receptacle: MlSpacesObject,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._uses_independent_placement_orientations():
            return self._get_independent_placement_poses(
                grasp_pose_world,
                pickup_obj,
                place_receptacle,
            )

        place_receptacle_aabb_center, place_receptacle_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, place_receptacle.object_id
        )
        receptacle_top_z = place_receptacle_aabb_center[2] + place_receptacle_aabb_size[2] / 2
        pickup_obj_aabb_center, pickup_obj_aabb_size = body_aabb(
            self.task.env.current_data.model, self.task.env.current_data, pickup_obj.object_id
        )
        pickup_obj_bottom_z = pickup_obj_aabb_center[2] - pickup_obj_aabb_size[2] / 2
        pickup_obj_clearance_offset = max(grasp_pose_world[2, 3] - pickup_obj_bottom_z, 0.0)

        preplace_pose = grasp_pose_world.copy()
        preplace_pose[:2, 3] = place_receptacle.position[:2]
        preplace_pose[2, 3] = (
            receptacle_top_z + pickup_obj_clearance_offset + self.policy_config.place_z_offset
        )
        # offset the EE to ensure the pickup object is in the middle of the receptacle
        preplace_pose[:3, 3] += grasp_pose_world[:3, 3] - pickup_obj.position
        if not self.check_feasible_ik(preplace_pose):
            log.debug("  - ❌ IK FAILED for preplace pose!")
            raise TargetPoseIKFailure(
                "preplace",
                {},
                self._capture_ik_diagnostic(
                    "preplace",
                    pickup_obj=pickup_obj,
                    grasp_pose=grasp_pose_world,
                    holding_object=True,
                ),
            )

        place_pose = preplace_pose.copy()
        place_pose[2, 3] = receptacle_top_z + pickup_obj_clearance_offset
        if not self.check_feasible_ik(place_pose):
            log.debug("  - ❌ IK FAILED for place pose!")
            raise TargetPoseIKFailure(
                "place",
                {"preplace": preplace_pose},
                self._capture_ik_diagnostic(
                    "place",
                    pickup_obj=pickup_obj,
                    grasp_pose=grasp_pose_world,
                    holding_object=True,
                ),
            )

        postplace_pose = place_pose.copy()
        postplace_pose[:3, 3] -= self.policy_config.end_z_offset * postplace_pose[:3, 2]

        return preplace_pose, place_pose, postplace_pose

    def _compute_target_poses(self, allow_best_effort: bool = True) -> dict[str, np.ndarray]:
        task_config = self.config.task_config
        assert isinstance(task_config, PickAndPlaceTaskConfig)
        target_poses = {}

        robot_view = self.task.env.current_robot.robot_view
        om = self.task.env.object_managers[self.task.env.current_batch_index]
        pickup_obj: MlSpacesObject = om.get_object_by_name(task_config.pickup_obj_name)
        place_receptacle: MlSpacesObject = om.get_object_by_name(task_config.place_receptacle_name)

        candidate_grasps = self._get_candidate_grasps(pickup_obj)
        pregrasp_candidates = candidate_grasps.copy()
        pregrasp_candidates[:, :3, 3] -= (
            self.policy_config.pregrasp_z_offset * candidate_grasps[:, :3, 2]
        )
        model = self.task.env.current_data.model
        data = self.task.env.current_data
        receptacle_center, receptacle_size = body_aabb(
            model,
            data,
            place_receptacle.object_id,
        )
        pickup_center, pickup_size = body_aabb(model, data, pickup_obj.object_id)
        receptacle_top = receptacle_center[2] + receptacle_size[2] / 2
        pickup_bottom = pickup_center[2] - pickup_size[2] / 2
        pickup_clearance = np.maximum(
            candidate_grasps[:, 2, 3] - pickup_bottom,
            0.0,
        )
        lift_candidates = candidate_grasps.copy()
        lift_candidates[:, 2, 3] = (
            receptacle_top + pickup_clearance + self.policy_config.place_z_offset
        )
        preplace_candidates = candidate_grasps.copy()
        preplace_candidates[:, :2, 3] = place_receptacle.position[:2]
        preplace_candidates[:, 2, 3] = (
            receptacle_top + pickup_clearance + self.policy_config.place_z_offset
        )
        preplace_candidates[:, :3, 3] += candidate_grasps[:, :3, 3] - pickup_obj.position
        place_candidates = preplace_candidates.copy()
        place_candidates[:, 2, 3] = receptacle_top + pickup_clearance
        selection_kwargs = dict(
            check_collision=self.policy_config.filter_colliding_grasps,
            n_collision_checks=self.policy_config.grasp_collision_max_grasps,
            collision_batch_size=self.policy_config.grasp_collision_batch_size,
            check_ik=self.policy_config.filter_feasible_grasps,
            n_ik_checks=self.policy_config.grasp_feasibility_max_grasps,
            ik_batch_size=self.policy_config.grasp_feasibility_batch_size,
            pos_cost_weight=self.policy_config.grasp_pos_cost_weight,
            rot_cost_weight=self.policy_config.grasp_rot_cost_weight,
            vertical_cost_weight=self.policy_config.grasp_vertical_cost_weight,
            com_dist_cost_weight=self.policy_config.grasp_com_dist_cost_weight,
            best_effort_rot_tol=(
                self.policy_config.best_effort_grasp_max_rot_error
                if (self.policy_config.allow_best_effort_grasp_orientation and allow_best_effort)
                else None
            ),
            best_effort_pos_tol=self.policy_config.best_effort_grasp_max_pos_error,
            best_effort_max_candidates=self.policy_config.best_effort_grasp_max_candidates,
        )
        additional_ik_poses = [pregrasp_candidates, lift_candidates]
        ik_pose_names = ["grasp", "pregrasp", "lift"]
        if not self._uses_independent_placement_orientations():
            additional_ik_poses.extend([preplace_candidates, place_candidates])
            ik_pose_names.extend(["preplace", "place"])
        allow_best_effort_lift = False
        try:
            grasp_pose_world = select_grasp_pose(
                self.task.env,
                candidate_grasps,
                pickup_obj.pose,
                additional_ik_poses_world=additional_ik_poses,
                ik_pose_names=ik_pose_names,
                exclude_object_grasps=self._tried_object_grasps,
                exclude_radius=self.policy_config.retry_grasp_exclusion_radius,
                diagnostic=self.grasp_plan_diagnostics,
                **selection_kwargs,
            )
        except ValueError as error:
            if str(error) != "No feasible grasp found":
                raise
            full_sequence_diagnostic = self.grasp_plan_diagnostics
            fallback_diagnostic = {}
            try:
                grasp_pose_world = select_grasp_pose(
                    self.task.env,
                    candidate_grasps,
                    pickup_obj.pose,
                    additional_ik_poses_world=[pregrasp_candidates],
                    ik_pose_names=["grasp", "pregrasp"],
                    exclude_object_grasps=self._tried_object_grasps,
                    exclude_radius=self.policy_config.retry_grasp_exclusion_radius,
                    diagnostic=fallback_diagnostic,
                    **selection_kwargs,
                )
            except ValueError:
                full_sequence_diagnostic["fallback"] = fallback_diagnostic
                raise
            full_sequence_diagnostic["fallback"] = fallback_diagnostic
            full_sequence_diagnostic["status"] = "selected_via_grasp_pregrasp_fallback"
            full_sequence_diagnostic["selection_note"] = (
                "Pickup-sequence batch IK found no joint candidate; grasp+pregrasp fallback selected"
                if self._uses_independent_placement_orientations()
                else "Full-sequence batch IK found no joint candidate; grasp+pregrasp fallback selected"
            )
            full_sequence_diagnostic["selected_index"] = fallback_diagnostic.get("selected_index")
            allow_best_effort_lift = full_sequence_diagnostic.get("failure_phase") == "lift"
            pregrasp_pose = grasp_pose_world.copy()
            pregrasp_pose[:3, 3] -= self.policy_config.pregrasp_z_offset * pregrasp_pose[:3, 2]
            if self._multistart_ik_for_pose(pregrasp_pose) is None:
                self._record_grasp_phase_failure(
                    "pregrasp",
                    "Deterministic multistart IK failed for the fallback pregrasp pose",
                )
                raise error
            log.info("Using deterministic multistart IK for pregrasp")

        pregrasp_pose, grasp_pose, lift_pose = self._get_grasp_poses(
            grasp_pose_world=grasp_pose_world,
            pickup_obj=pickup_obj,
            place_receptacle=place_receptacle,
            robot_view=robot_view,
            task_config=task_config,
            allow_best_effort_lift=allow_best_effort_lift,
        )
        target_poses["pregrasp"] = pregrasp_pose
        target_poses["grasp"] = grasp_pose
        target_poses["lift"] = lift_pose

        try:
            preplace_pose, place_pose, postplace_pose = self._get_placement_poses(
                grasp_pose_world=grasp_pose_world,
                pickup_obj=pickup_obj,
                place_receptacle=place_receptacle,
            )
        except TargetPoseIKFailure as error:
            error.target_poses = {**target_poses, **error.target_poses}
            raise
        target_poses["preplace"] = preplace_pose
        target_poses["place"] = place_pose
        target_poses["postplace"] = postplace_pose

        # debug
        visualize_poses = True
        if visualize_poses and self.task.viewer is not None:
            self._show_poses(np.stack(list(target_poses.values()), axis=0), style="tcp")
            if self.task.viewer:
                self.task.viewer.sync()

        return target_poses
