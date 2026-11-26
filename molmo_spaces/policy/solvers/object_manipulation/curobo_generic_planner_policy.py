"""Pick-and-place whose TRANSIT to pregrasp is planned by cuRobo, and nothing else changed.

The baseline moves the TCP in a straight line from wherever the arm is to the pregrasp pose. That
line is chosen without reference to the scene, so it is the one part of the task that can drive the
arm through a counter or a neighbouring object. This subclass replaces exactly that segment with a
collision-free joint-space plan and leaves the rest of the policy alone.

**Execution follows the shipped RB-Y1 cuRobo policy, not a clock.** MolmoSpaces already has a
working cuRobo integration (``policy/solvers/curobo_planner_policy.py``) and the transferable part
of it is how it *drives* a plan: command a waypoint, hold it until the joints actually arrive, then
advance; carry the gripper explicitly in every action; clamp joint velocity. What is NOT
transferable is its structure -- it is a parallel policy family that hard-codes ``left_arm`` /
``right_arm`` / ``left_gripper`` / ``base`` and hand-authored ``planner_joint_ranges``, so it only
works for one robot. Here the same semantics live in one action primitive, with every group name
and index range derived from the embodiment.

Time-based execution was tried first and is why this file is worth its comments. Driving the plan
through the stock time-parameterised ``JointMoveSequence`` produced a robot that stood still for 18
control steps and then moved the entire transit in one step at 10.4 rad/s: a segment shorter than
one control step is re-clocked on every tick, so its interpolation fraction never leaves zero, and
the sequence then falls through to its final waypoint. Arrival-gated tracking cannot express that
failure -- a waypoint is held until it is reached -- which is exactly why upstream does it that way.

What this deliberately does NOT override is grasp selection.
``PickTaskSampler._create_all_phase_ik_base_pose_filter`` instantiates ``policy_config.policy_cls``
and calls ``_compute_target_poses`` while it is *sampling the robot's base pose*
(``pick_task_sampler.py:1100``). Changing that method would change the sampled base pose with it, so
the cuRobo run would face a different scene, a different grasp and a different camera, and a
comparison against the baseline would measure nothing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    ActionPrimitive,
    JointMoveSegment,
    JointMoveSequence,
    TCPMoveSequence,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from gymnasium import spaces as gyms

from molmo_spaces.env.abstract_sensors import Sensor
from molmo_spaces.utils.mj_model_and_data_utils import descendant_geoms

log = logging.getLogger(__name__)


class _NamedBody:
    """Minimal stand-in for an object-manager object: ``_world_cuboids`` needs only these two.

    ``own_geoms_only`` switches off the descendant walk. That walk is right for a scene body, whose
    children are parts of the same object, and wrong for a robot link, whose children are the links
    that move: taking the subtree of the arm's mounting body produced 75 boxes covering the whole
    arm and 1934 self-penetrations, every one of them a link colliding with a box drawn around
    itself.
    """

    def __init__(self, name: str, body_id: int, own_geoms_only: bool = False) -> None:
        self.name = name
        self.body_id = body_id
        self.own_geoms_only = own_geoms_only


class CuroboPlanSensor(Sensor):
    """Publish the current phase's planned path, per step, so the saved video can draw it.

    A sensor rather than a side-channel because that is how the pipeline's ``_waypoints.mp4``
    annotator receives everything else it draws -- it is handed observations and nothing more
    (``save_utils._annotate_waypoint_frames``). Publishing here means every future run gets the
    overlay with no change at the call site and no post-hoc script.

    Points are TCP positions in the robot BASE frame, matching ``tcp_pose``'s convention, so the
    annotator composes both with ``robot_base_pose`` the same way.
    """

    def __init__(self, uuid: str = "curobo_plan", str_max_len: int = 12000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        policy = task._registered_policy
        points = getattr(policy, "plan_points", None)
        if not points:
            return {"phase": None, "points": [], "status": "no_plan"}
        phase = policy.get_phase() if hasattr(policy, "get_phase") else None
        entry = points.get(phase)
        if entry is None:
            return {"phase": phase, "points": [], "status": "phase_not_planned"}
        return {"phase": phase, "points": entry["points"], "status": entry["status"]}


class CuroboSphereSensor(Sensor):
    """Publish cuRobo's collision spheres for the current step, so the video can draw them.

    Companion to :class:`CuroboPlanSensor`: that one shows the route the planner chose, this one
    shows the body it was routing. ``spheres`` are in the robot base frame like ``curobo_plan``;
    ``payload`` is in the TCP frame, because the attached object is a fixed child of the TCP link
    and so follows ``tcp_pose`` exactly.
    """

    def __init__(self, uuid: str = "curobo_spheres", str_max_len: int = 20000) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        policy = task._registered_policy
        if not hasattr(policy, "collision_spheres_now"):
            return {"phase": None, "spheres": [], "payload": []}
        try:
            record = policy.collision_spheres_now()
        except Exception as error:  # a diagnostic must never take an episode down
            log.debug(f"[CUROBO] could not record collision spheres: {error}")
            return {"phase": None, "spheres": [], "payload": []}
        record["phase"] = policy.get_phase() if hasattr(policy, "get_phase") else None
        return record


class CuroboObstacleSensor(Sensor):
    """Publish the obstacle boxes the current phase was planned against.

    The third of the trio: :class:`CuroboPlanSensor` is the route, :class:`CuroboSphereSensor` is
    the body being routed, and this is the world both were computed in. Without it a plan that
    drives through furniture is ambiguous -- the planner could have known about the obstacle and
    ignored it, or never been told. Drawing the boxes settles which, and it settled it here: the
    counter's clutter was absent from every plan for weeks of runs.

    Boxes are oriented, in the robot base frame like everything else the annotator draws, and are
    published only during the phase they belong to.
    """

    def __init__(self, uuid: str = "curobo_obstacles", str_max_len: int = 65536) -> None:
        self.str_max_len = str_max_len
        self.is_dict = True
        observation_space = gyms.Box(low=0, high=255, shape=(str_max_len,), dtype=np.uint8)
        super().__init__(uuid=uuid, observation_space=observation_space)

    def get_observation(self, env, task, batch_index: int = 0, *args, **kwargs) -> dict:
        policy = task._registered_policy
        boxes = getattr(policy, "plan_obstacles", None)
        phase = policy.get_phase() if hasattr(policy, "get_phase") else None
        if not boxes or phase is None:
            return {"phase": phase, "boxes": []}
        return {"phase": phase, "boxes": boxes.get(phase, [])}


class ArrivalGatedJointPath(JointMoveSequence):
    """Track a joint-space path by ARRIVAL rather than by elapsed time.

    The execution rule of ``CuroboPlannerPolicy._execute_trajectory``, generalised: hold each
    waypoint until every planned joint is within ``tolerance`` of it, skip waypoints that are
    already satisfied, and give up on a waypoint after ``max_steps_per_waypoint`` control steps so
    the policy's own retry machinery can replan.

    Two properties come from this rather than from tuning:

    * **No motion can outrun the controller.** Each command is the current position stepped toward
      the waypoint by at most ``max_joint_velocity * control_dt``, so a distant waypoint is
      approached over several steps instead of commanded in one.
    * **Nothing else on the robot is disturbed.** The action starts from
      ``robot_view.get_ctrl_dict()`` and overwrites only the planned groups, so the gripper keeps
      whatever the preceding ``GripperAction`` set -- the same mechanism ``_tcp_to_jp_fn`` relies
      on. Building the action from the waypoint alone drops the gripper from the action entirely,
      and it closes: the arm then arrives at the object with its jaws shut.

    It subclasses ``JointMoveSequence`` for one concrete reason: the policy's ``reset()`` populates
    ``target_poses`` by walking the returned primitives and forward-kinematicising the segments of
    anything that *is* a ``JointMoveSequence`` (``base_object_manipulation_planner_policy.py:669``).
    Keeping the type keeps ``target_poses["pregrasp"]`` populated for the phase sensors, with no
    change to shared code. Only ``execute`` and ``get_current_action`` are replaced; the inherited
    time-based versions of both are unused.
    """

    def __init__(
        self,
        robot_view,
        move_segments: list[JointMoveSegment],
        *,
        tolerance: float,
        max_steps_per_waypoint: int,
        max_joint_velocity: float,
        control_dt: float,
        settle_steps: int,
        is_holding_object: bool = False,
        gripper_empty_threshold: float = 0.0,
    ) -> None:
        super().__init__(
            robot_view,
            settle_time=0.0,
            move_segments=move_segments,
            is_holding_object=is_holding_object,
            gripper_empty_threshold=gripper_empty_threshold,
        )
        self.tolerance = tolerance
        self.max_steps_per_waypoint = max_steps_per_waypoint
        self.max_joint_velocity = max_joint_velocity
        self.control_dt = control_dt
        self.settle_steps = settle_steps
        self._steps_in_waypoint = 0
        self._steps_settling = 0
        self._steps_allowed = max_steps_per_waypoint

    # -- state ------------------------------------------------------------------------

    def reset(self) -> None:
        super().reset()
        self._steps_in_waypoint = 0
        self._steps_settling = 0
        self._steps_allowed = self.max_steps_per_waypoint

    def _reached(self, index: int) -> bool:
        target = self.move_segments[index].end_qpos
        return all(
            bool(
                np.all(
                    np.abs(
                        np.asarray(self.robot_view.get_move_group(group_id).joint_pos, dtype=float)
                        - np.asarray(values, dtype=float)
                    )
                    < self.tolerance
                )
            )
            for group_id, values in target.items()
        )

    # -- driving ----------------------------------------------------------------------

    def execute(self) -> bool:
        if self.start_time is None:
            self.start_time = self.robot_view.mj_data.time
            self.move_seg_idx = 0
            self._steps_in_waypoint = 0
            self._steps_settling = 0

        # Dense plans contain waypoints closer together than the arrival tolerance; skipping the
        # satisfied ones is what lets the planner's own interpolation resolution be used directly,
        # with no resampling step to get wrong.
        while self.move_seg_idx < len(self.move_segments) and self._reached(self.move_seg_idx):
            self.move_seg_idx += 1
            self._steps_in_waypoint = 0

        if self.move_seg_idx >= len(self.move_segments):
            self._steps_settling += 1
            return self._steps_settling > self.settle_steps

        if self._steps_in_waypoint == 0:
            self._steps_allowed = self._budget_from_here(self.move_seg_idx)
        self._steps_in_waypoint += 1
        return False

    def _budget_from_here(self, index: int) -> int:
        """How many control steps this waypoint can possibly need, from where the arm is now.

        A flat allowance cannot work: ``get_current_action`` clamps every step to
        ``max_joint_velocity * control_dt``, so the steps a waypoint needs are set by how far away
        it is, and a planned route's spacing varies by an order of magnitude along its length. A
        flat ten steps buys about 0.7 rad at the shipped limits, which was ample for the dense tail
        of a plan and impossible for its long opening strides -- transport plans were abandoned at
        waypoint 15/62, 24/67 and 20/48 with nothing actually wrong with them.

        Distance is measured live rather than from the plan's own spacing because satisfied
        waypoints are skipped, so the arm may be several waypoints back from this one.
        """
        step = self.max_joint_velocity * self.control_dt
        if step <= 0.0:
            return self.max_steps_per_waypoint
        span = 0.0
        for group_id, values in self.move_segments[index].end_qpos.items():
            current = np.asarray(self.robot_view.get_move_group(group_id).joint_pos, dtype=float)
            span = max(span, float(np.max(np.abs(np.asarray(values, dtype=float) - current))))
        # The config value stops being the budget and becomes the slack on top of it: time for the
        # controller to settle inside the tolerance once the clamped ramp has covered the distance.
        return int(np.ceil(span / step)) + self.max_steps_per_waypoint

    def check_failure(self) -> bool:
        if super().check_failure():
            return True
        if self._steps_in_waypoint > self._steps_allowed:
            log.warning(
                f"[CUROBO] waypoint {self.move_seg_idx}/{len(self.move_segments)} not reached in "
                f"{self._steps_in_waypoint} steps (allowed {self._steps_allowed}); handing back to "
                "the policy's retry"
            )
            return True
        return False

    def get_current_action(self) -> dict[str, Any]:
        action = self.robot_view.get_ctrl_dict()
        index = min(self.move_seg_idx or 0, len(self.move_segments) - 1)
        step = self.max_joint_velocity * self.control_dt
        for group_id, values in self.move_segments[index].end_qpos.items():
            current = np.asarray(self.robot_view.get_move_group(group_id).joint_pos, dtype=float)
            delta = np.asarray(values, dtype=float) - current
            action[group_id] = current + np.clip(delta, -step, step)
        return action

    def get_current_phase(self) -> str:
        return self.move_segments[0].name


class DeferredPlan(TCPMoveSequence):
    """A transit that asks cuRobo for its route at the moment its own phase begins.

    Planning every phase during ``reset()`` means planning against a scene that has not happened
    yet. The obstacle boxes and the carried object are read out of the simulator as it stands at
    reset, while the start configuration is the cached IK of a phase that has not run. On the
    transport segment those two disagreed by an entire lift: the payload was attached 109 mm below
    where the bottle would really be, so cuRobo refused every route because a phantom bottle swept
    the countertop. Deferring makes the start configuration, the obstacles and the payload all
    readings of the same instant of the same simulator state.

    Being a ``TCPMoveSequence`` rather than a wrapper around one is what keeps this cheap. Until
    its phase starts it simply IS the straight-line segment -- which is also exactly the fallback
    when a plan fails -- so ``target_poses`` still finds its end pose by the ordinary walk over
    primitives, and a failed plan needs no special case.
    """

    def __init__(self, policy, sequence: TCPMoveSequence, segment) -> None:
        super().__init__(
            sequence.robot_view,
            policy._tcp_to_jp_fn,
            sequence.settle_time,
            is_holding_object=sequence.is_holding_object,
            gripper_empty_threshold=sequence.gripper_empty_threshold,
            tcp_pos_err_threshold=sequence.tcp_pos_err_threshold,
            tcp_rot_err_threshold=sequence.tcp_rot_err_threshold,
            move_segments=[segment],
        )
        self._policy = policy
        self._segment = segment
        self._planned: ArrivalGatedJointPath | None = None
        self._asked = False

    def _plan_now(self) -> None:
        """Plan once, on the first step of the phase. Costs planner time but no simulator time."""
        if self._asked:
            return
        self._asked = True
        group_id = self._policy._planned_group_ids()[0]
        # The arm's ACTUAL configuration, not a prediction of it.
        start = np.asarray(self.robot_view.get_move_group(group_id).joint_pos, dtype=float)
        self._planned = self._policy._plan_segment(
            self._segment, start_config=start, holding=self.is_holding_object
        )

    def execute(self) -> bool:
        self._plan_now()
        return self._planned.execute() if self._planned else super().execute()

    def get_current_action(self) -> dict[str, Any]:
        if self._planned is not None:
            return self._planned.get_current_action()
        return super().get_current_action()

    def check_failure(self) -> bool:
        if self._planned is not None:
            return self._planned.check_failure()
        return super().check_failure()

    def get_render_override(self):
        if self._planned is not None:
            return self._planned.get_render_override()
        return super().get_render_override()

    def get_render_caption(self) -> tuple[str, ...] | None:
        if self._planned is not None:
            return self._planned.get_render_caption()
        return super().get_render_caption()

    def reset(self):
        super().reset()
        if self._planned is not None:
            self._planned.reset()


class CuroboGenericPickAndPlacePlannerPolicy(PickAndPlacePlannerPolicy):
    """``PickAndPlacePlannerPolicy`` whose approach to pregrasp is a planned path."""

    def __init__(self, config, task) -> None:
        super().__init__(config, task)
        self._planner = None
        self._bridged = None
        self.curobo_diagnostics: dict[str, Any] = {}
        self.plan_points: dict[str, dict] = {}
        self.plan_obstacles: dict[str, list] = {}
        self.payload_spheres: list[list[float]] = []
        self._plan_index = 0

    # ---------- planner ----------

    def create_policy_sensors(self):
        return super().create_policy_sensors() + [
            CuroboPlanSensor(uuid="curobo_plan"),
            CuroboSphereSensor(uuid="curobo_spheres"),
            CuroboObstacleSensor(uuid="curobo_obstacles"),
        ]

    def collision_spheres_now(self) -> dict:
        """cuRobo's collision model at the CURRENT configuration, in the robot base frame.

        The planner does not see the robot, it sees these spheres, and the difference is where the
        surprises live: a plan that looks like it clears an obstacle by centimetres can be refused
        because the sphere fit of the payload is far larger than the payload. Drawing them next to
        the obstacle boxes turns that from an inference into something visible.

        Robot links are placed from MuJoCo rather than by asking cuRobo, so the overlay is a check
        on the bridge rather than a restatement of it: if the spheres do not sit on the arm, the
        link frames the bridge exported disagree with the ones the simulator is using.
        """
        bridged = self._bridged
        if bridged is None:
            return {"spheres": [], "payload": []}
        model = self.task.env.current_model
        data = self.task.env.current_data
        to_base = np.linalg.inv(self.task.env.current_robot.get_world_pose_tf_mat())

        spheres = []
        for link, entries in bridged.robot_cfg["kinematics"]["collision_spheres"].items():
            try:
                body_id = model.body(link).id
            except Exception:
                continue  # the attached link is not a simulator body; it rides on the TCP instead
            rotation = data.xmat[body_id].reshape(3, 3)
            for entry in entries:
                centre = data.xpos[body_id] + rotation @ np.asarray(entry["center"], dtype=float)
                centre = to_base[:3, :3] @ centre + to_base[:3, 3]
                spheres.append([round(float(v), 4) for v in centre]
                               + [round(float(entry["radius"]), 4)])
        # The payload rides on the TCP frame, so it is published there and composed with the
        # per-step ``tcp_pose`` by the annotator -- no second FK, and it lands wherever the gripper
        # is even on the steps between plans.
        return {"spheres": spheres, "payload": self.payload_spheres}

    @staticmethod
    def _rejected_waypoints(result) -> list:
        """The trajectory a failed plan converged to, as joint waypoints.

        ``plan_to_joint_config`` returns nothing on failure because nothing is safe to execute, but
        the optimiser's last iterate survives on the result and is what makes a failure legible:
        it shows whether cuRobo barely deformed the seed or fought its way most of the way there.
        """
        plan = getattr(result, "interpolated_plan", None)
        position = getattr(plan, "position", None)
        if position is None:
            return []
        try:
            values = position.detach().cpu().numpy()
        except AttributeError:
            values = np.asarray(position)
        return [row for row in (values[0] if values.ndim == 3 else values)]

    def _record_plan_points(self, phase: str, waypoints, group_id: str, status: str) -> None:
        """TCP position of each planned waypoint, for the per-step sensor."""
        try:
            kinematics = self.task.env.current_robot.kinematics
            gripper = self.robot_view.get_gripper_movegroup_ids()[0]
            base_pose = self.robot_view.base.pose
            points = [
                kinematics.fk({group_id: np.asarray(w, dtype=float)}, base_pose, rel_to_base=True)[
                    gripper
                ][:3, 3].tolist()
                for w in waypoints
            ]
        except Exception as error:  # never let a diagnostic break an episode
            log.debug(f"[CUROBO] could not record plan points: {error}")
            points = []
        self.plan_points[phase] = {"points": points, "status": status}

    def _get_planner(self):
        """Build the planner on first use.

        Lazily, because constructing ``MotionGen`` costs GPU memory and a warmup of seconds, and a
        worker that never reaches a transit (task sampling failed, IK failed) should not pay it.
        """
        if self._planner is not None:
            return self._planner

        from molmo_spaces.planner.curobo_bridge import build_curobo_robot_model
        from molmo_spaces.planner.curobo_planner import CuroboPlanner, CuroboPlannerConfig

        policy_config = self.policy_config
        self._bridged = build_curobo_robot_model(
            self.config.robot_config,
            policy_config.curobo_output_dir,
            collision_sphere_buffer=policy_config.collision_sphere_buffer,
            max_spheres_per_geom=policy_config.max_spheres_per_geom,
        )
        self._planner = CuroboPlanner(
            CuroboPlannerConfig(
                robot_cfg=self._bridged.robot_cfg,
                interpolation_dt=policy_config.interpolation_dt,
                collision_activation_distance=policy_config.collision_activation_distance,
                num_ik_seeds=policy_config.num_ik_seeds,
                num_trajopt_seeds=policy_config.num_trajopt_seeds,
                trajopt_tsteps=policy_config.trajopt_tsteps,
                max_attempts=policy_config.max_attempts,
                enable_graph_attempt=policy_config.enable_graph_attempt,
                grad_trajopt_iters=policy_config.grad_trajopt_iters,
                finetune_trajopt_iters=policy_config.finetune_trajopt_iters,
                check_start_validity=False,
                # This policy plans in joint space; warmup only warms the Cartesian path, at the
                # full configured iteration budget. See CuroboPlannerConfig.warmup.
                warmup=False,
            )
        )
        log.info(
            f"[CUROBO] planner ready for {self.config.robot_config.embodiment.embodiment_id}: "
            f"{len(self._bridged.dof_joint_names)} dofs, "
            f"{sum(self._bridged.sphere_counts.values())} spheres"
        )
        return self._planner

    def _planned_group_ids(self) -> list[str]:
        """The move groups cuRobo plans for.

        Derived from the robot, never named. Upstream carries the equivalent as a hand-written
        ``planner_joint_ranges = {"base": (0, 3), "left_arm": (3, 10)}`` per arm per robot, which is
        the part of its design that does not survive contact with a second embodiment.
        """
        return self.task.env.current_robot.get_arm_move_group_ids()

    # ---------- scene ----------

    def _obstacles(self) -> list:
        """Scene bodies worth avoiding on the way to the object.

        The surface the object rests on and whatever else is standing on it: that is what a
        straight line to a pregrasp pose actually hits. Everything is an axis-aligned box, which is
        what the planner's world model accepts.

        NOT the pickup object. The transit ends at the pregrasp pose, which puts the open jaws
        around where the object is about to be; with the object in the world that goal is in
        collision by construction and every plan fails. The straight-line approach that follows is
        the part that must reach the object, and it is deliberately left unplanned.
        """
        # Both constants from the same module, imported together and OUTSIDE the try below. The
        # previous version imported this one from `molmo_spaces.editor.constants`, which does not
        # exist; the ImportError was caught by the same handler that excuses a scene with no
        # resolvable support and logged at DEBUG, so every loose object on the surface was silently
        # dropped from the world while the surface itself was kept -- an obstacle set that looks
        # reasonable and is missing all the clutter.
        from molmo_spaces.utils.constants.object_constants import (
            ALL_PICKUP_TYPES_THOR,
            RECEPTACLE_TYPES_THOR,
        )

        object_manager = self.task.env.object_managers[self.task.env.current_batch_index]
        task_config = self.config.task_config
        names: set[str] = set()
        place_receptacle = getattr(task_config, "place_receptacle_name", None)
        if place_receptacle:
            names.add(place_receptacle)
        try:
            support = object_manager.get_support_below(
                task_config.pickup_obj_name, RECEPTACLE_TYPES_THOR
            )
            names.add(support)
            for neighbour in object_manager.get_objects_that_are_on_top_of_object(
                support, pickup_types=ALL_PICKUP_TYPES_THOR
            ):
                # Everything standing on the surface EXCEPT the thing being picked up. It is on
                # that surface too, so the query returns it, and as an obstacle it puts the
                # pregrasp goal in collision by construction -- see the note above.
                if neighbour.name != task_config.pickup_obj_name:
                    names.add(neighbour.name)
        except Exception as error:  # a scene without a resolvable support is still plannable
            log.warning(f"[CUROBO] no support surface resolved: {error}")

        # The task barrier, when one exists. It is a scene body rather than a task object, so it
        # is not reachable through the object manager and is added by name.
        from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
            TASK_BARRIER_BODY,
        )

        obstacles = []
        # The robot's own pedestal, as geometry rather than as part of the robot. The bridge leaves
        # these links out of the sphere model because planning cannot move them; they still have to
        # be avoided, so they enter here like any other scene body.
        model = self.task.env.current_model
        for link in getattr(self._bridged, "static_body_names", []) or []:
            try:
                obstacles.append(_NamedBody(link, model.body(link).id, own_geoms_only=True))
            except Exception as error:
                log.debug(f"[CUROBO] static link {link!r} not in the scene: {error}")
        if getattr(self.policy_config, "barrier_enabled", False):
            try:
                obstacles.append(_NamedBody(TASK_BARRIER_BODY, model.body(TASK_BARRIER_BODY).id))
            except Exception as error:
                log.warning(f"[CUROBO] barrier enabled but not found in the scene: {error}")

        for name in sorted(n for n in names if n):
            try:
                obstacles.append(object_manager.get_object_by_name(name))
            except Exception:
                continue
        return obstacles

    def _world_cuboids(self, obstacles: list):
        """One ORIENTED box per collision geom, in the robot base frame the planner works in.

        Not one world-axis-aligned box per body, which is what this used to be and which is wrong
        in three separate ways:

        * **Rotation was dropped.** An axis-aligned box around a rotated thin object is enormously
          larger than the object. Measured on a 1 cm wall yawed 92 degrees: the planner was handed a
          30 cm slab, a 30x inflation, which filled the entire 21 cm gap it was supposed to plan
          through and made the transport unsolvable. Every obstacle before that happened to be
          near-axis-aligned, which is why it stayed invisible.
        * **Multi-geom bodies became solid.** One box around a whole body fills the space between
          its parts -- a table becomes a block from floor to top. The shipped RB-Y1 planner carries
          a hard-coded ``if object.name == "scene/table"`` override for exactly this; per-geom boxes
          remove the need for that class of special case.
        * **Visual geometry counted.** Geoms with no contact mask are not obstacles.

        ``model.geom_aabb`` is expressed in the geom's own frame, so pairing it with the geom's pose
        is exact for a box and tight for a mesh.
        """
        from curobo.geom.types import Cuboid

        from molmo_spaces.utils.pose import pose_mat_to_7d

        model = self.task.env.current_model
        data = self.task.env.current_data
        to_base = np.linalg.inv(self.task.env.current_robot.get_world_pose_tf_mat())
        limit = self.policy_config.max_obstacle_boxes

        cuboids = []
        dropped = 0
        for obstacle in obstacles:
            if getattr(obstacle, "own_geoms_only", False):
                start = int(model.body_geomadr[obstacle.body_id])
                candidates = range(start, start + int(model.body_geomnum[obstacle.body_id]))
            else:
                candidates = descendant_geoms(model, obstacle.body_id, visible_only=False)
            geom_ids = [
                geom_id
                for geom_id in candidates
                if model.geom_contype[geom_id] or model.geom_conaffinity[geom_id]
            ]
            if not geom_ids:
                continue
            for order, geom_id in enumerate(geom_ids):
                if len(cuboids) >= limit:
                    dropped += 1
                    continue
                local = model.geom_aabb[geom_id]
                rotation = data.geom_xmat[geom_id].reshape(3, 3)
                pose_world = np.eye(4)
                pose_world[:3, :3] = rotation
                pose_world[:3, 3] = data.geom_xpos[geom_id] + rotation @ local[:3]
                cuboids.append(
                    Cuboid(
                        name=f"{obstacle.name}#{order}",
                        pose=pose_mat_to_7d(to_base @ pose_world).tolist(),
                        dims=[float(v) * 2.0 for v in local[3:]],
                    )
                )
        if dropped:
            # Never silently: a truncated world is a world the planner believes is emptier than it
            # is, and the resulting plan looks perfectly successful.
            log.warning(
                f"[CUROBO] obstacle budget {limit} reached; {dropped} collision geoms omitted "
                "from the planning world"
            )
        return cuboids

    def _carried_geometry(self) -> list:
        """The pickup object as planner boxes, in the base frame, as of right now.

        One source for both the attach and the failure dump, so a replayed failure is the failure
        that happened rather than an approximation of it.
        """
        name = getattr(self.config.task_config, "pickup_obj_name", None)
        if not name:
            return []
        try:
            manager = self.task.env.object_managers[self.task.env.current_batch_index]
            return self._world_cuboids([manager.get_object_by_name(name)])
        except Exception as error:
            log.debug(f"[CUROBO] no geometry for the carried object: {error}")
            return []

    # ---------- the one override ----------

    def _joint_config_for(self, pose: np.ndarray, group_id: str) -> np.ndarray | None:
        """The arm configuration the baseline would reach at ``pose``.

        The all-phase base-pose filter has already solved and cached IK for every phase of the
        episode (``pick_task_sampler.py:1078``), so this is normally a cache hit and the planned
        endpoints are exactly the configurations the unplanned policy would have passed through.
        """
        solution = self._get_cached_ik_solution(pose)
        if solution is None:
            robot_view = self.task.env.current_robot.robot_view
            solution = self.task.env.current_robot.kinematics.ik(
                robot_view.get_gripper_movegroup_ids()[0],
                pose,
                [group_id],
                robot_view.get_qpos_dict(),
                robot_view.base.pose,
            )
        if solution is None or group_id not in solution:
            return None
        return np.asarray(solution[group_id], dtype=float)

    def _attach_carried_object(self, planner, start_config: list[float]) -> str | None:
        """Put the grasped object into the planner's robot, returning the link it went on.

        A transport plan computed with an empty gripper routes the WRIST around an obstacle and
        drags the carried object straight through it -- measured on the barrier scene: the plan
        dived the TCP 228 mm, and the bottle it was holding was pressed into the countertop 125 mm
        short of that, where the arm stalled until the waypoint watchdog gave up.

        The object is passed as geometry rather than by name because it is deliberately NOT in the
        planner's world -- see ``_obstacles``. The same per-geom oriented boxes the world is built
        from are reused, so the planner's idea of the payload matches its idea of everything else.
        """
        name = self.config.task_config.pickup_obj_name
        link = next(iter(planner.attached_link_names), None)
        if link is None or not name:
            return None
        try:
            geometry = self._carried_geometry()
            planner.attach_external_obj(geometry, start_config, link)
            # Kept past the detach below: it is what the planner believed it was carrying, and the
            # video overlay needs it on every step of the phase, not just while the attach is live.
            self.payload_spheres = planner.link_spheres(link)
            log.info(
                f"[CUROBO] carrying {name!r}: {len(geometry)} bodies attached to {link!r} "
                f"as {len(self.payload_spheres)} spheres"
            )
            return link
        except Exception as error:
            # Not fatal: the plan is then correct for the gripper and optimistic about the object.
            log.warning(f"[CUROBO] could not attach {name!r} for transport planning: {error}")
            return None

    def _plan_segment(
        self, segment, *, start_config: np.ndarray, holding: bool
    ) -> ArrivalGatedJointPath | None:
        """Replace one straight-line TCP segment with a planned joint path.

        ``start_config`` is where the arm will BE when this segment begins -- the current pose for
        the first segment of the episode, otherwise the cached IK solution of the previous phase.
        Planning between the same two configurations the baseline would pass through keeps the two
        runs comparable: only the route between them differs.
        """
        from curobo.geom.types import WorldConfig

        robot_view = self.task.env.current_robot.robot_view
        group_id = self._planned_group_ids()[0]
        policy_config = self.policy_config

        goal = self._joint_config_for(segment.end_pose, group_id)
        if goal is None:
            log.info(f"[CUROBO] no IK for {segment.name!r} goal; leaving it a straight line")
            return None

        planner = self._get_planner()
        if policy_config.planner_seed is not None:
            # Per plan, not once: two segments in the same episode should be independent draws,
            # and the sequence must be reproducible from the seed alone.
            import torch

            torch.manual_seed(policy_config.planner_seed * 1000 + self._plan_index)
        self._plan_index += 1
        obstacles = self._obstacles()
        cuboids = self._world_cuboids(obstacles)
        planner.set_world(cuboids)
        # Kept per phase, like the plan itself: the video should show the world THIS phase was
        # planned against, not whatever the world became later.
        # Flat [x, y, z, qw, qx, qy, qz, dx, dy, dz], no names: hundreds of boxes per phase go
        # through a fixed-width string sensor, and the names are three times the numbers.
        self.plan_obstacles[segment.name] = [
            [round(float(v), 3) for v in (*cuboid.pose, *cuboid.dims)] for cuboid in cuboids
        ]

        attached_link = None
        if holding and policy_config.attach_carried_object:
            attached_link = self._attach_carried_object(planner, start_config.tolist())
        try:
            waypoints, motion_time, result = planner.plan_to_joint_config(
                start_config.tolist(), goal.tolist(), scene_objects=None, verbose=True
            )
        finally:
            if attached_link is not None:
                planner.detach_obj([attached_link])

        if not waypoints:
            # Record what it converged to before rejecting it, not nothing. cuRobo populates
            # ``interpolated_plan`` even when ``success`` is False, and that curve is the whole
            # answer to "what did it try" -- without it a failure is indistinguishable in the video
            # from never having run.
            self._record_plan_points(
                segment.name, self._rejected_waypoints(result), group_id, "plan_failed"
            )
            self._diagnose_failure(
                planner, segment, start_config, goal, obstacles, holding, result
            )
            log.warning(
                f"[CUROBO] {segment.name!r} planning failed; falling back to the straight line"
            )
            return None

        # Every interpolated waypoint becomes a segment. No resampling: arrival-gated tracking
        # skips whatever is already satisfied, so the planner's own resolution is used as-is and
        # there is no step count to get wrong.
        control_dt = self.config.policy_dt_ms / 1000.0
        joint_segments = [
            JointMoveSegment(
                name=segment.name,
                start_qpos=None,
                end_qpos={group_id: np.asarray(waypoint, dtype=float)},
                # Unused by arrival gating; a positive value keeps the primitive driver's
                # "a zero-length primitive must proceed" assertion satisfied.
                duration_s=control_dt,
            )
            for waypoint in waypoints
        ]

        self._record_plan_points(segment.name, waypoints, group_id, "planned")
        self._dump_plan(segment.name, waypoints, motion_time, obstacles, group_id, attached_link)
        self.curobo_diagnostics[segment.name] = {
            "status": "planned",
            "waypoints": len(waypoints),
            "motion_time": motion_time,
            "obstacles": len(obstacles),
            "attached": attached_link,
        }
        log.info(
            f"[CUROBO] {segment.name!r} planned: {len(waypoints)} waypoints, {motion_time:.2f}s of "
            f"planner time, {len(obstacles)} obstacles"
            f"{', object attached' if attached_link else ''}"
        )
        return ArrivalGatedJointPath(
            robot_view,
            joint_segments,
            tolerance=policy_config.waypoint_tolerance,
            max_steps_per_waypoint=policy_config.max_steps_per_waypoint,
            max_joint_velocity=policy_config.max_joint_velocity,
            control_dt=control_dt,
            settle_steps=policy_config.transit_settle_steps,
            is_holding_object=holding,
            gripper_empty_threshold=policy_config.gripper_empty_threshold,
        )

    def _deferred(self, sequence: TCPMoveSequence, segment) -> "DeferredPlan":
        return DeferredPlan(self, sequence, segment)

    def _clone_sequence(self, original: TCPMoveSequence, segments: list) -> TCPMoveSequence:
        """A TCPMoveSequence over ``segments`` with every tuning of ``original`` preserved.

        The thresholds are not defaults: the base policy relaxes them when the grasp or lift was
        only reachable best-effort, and rebuilding a sequence from config values would silently
        re-tighten them.
        """
        return TCPMoveSequence(
            original.robot_view,
            self._tcp_to_jp_fn,
            original.settle_time,
            is_holding_object=original.is_holding_object,
            gripper_empty_threshold=original.gripper_empty_threshold,
            tcp_pos_err_threshold=original.tcp_pos_err_threshold,
            tcp_rot_err_threshold=original.tcp_rot_err_threshold,
            move_segments=segments,
        )

    def _rebuild_sequence(self, sequence: TCPMoveSequence) -> list[ActionPrimitive]:
        """Split one TCP sequence around whichever of its segments cuRobo plans.

        ``[lift, preplace, place]`` becomes ``[TCP(lift), DeferredPlan(preplace), TCP(place)]``.
        Only the free-space transits are candidates; the short contact-adjacent moves keep their
        straight lines because their goals are deliberately in contact -- cuRobo would either
        refuse them or curve a descent that has to be vertical.

        Nothing is planned here, and no start configuration is predicted here. Each DeferredPlan
        asks for its route when its own phase begins; see that class for why guessing the start
        configuration ahead of time was wrong.
        """
        rebuilt: list[ActionPrimitive] = []
        straight: list = []
        for segment in sequence.move_segments:
            if segment.name not in self.policy_config.planned_segments:
                straight.append(segment)
                continue
            if straight:
                rebuilt.append(self._clone_sequence(sequence, straight))
                straight = []
            rebuilt.append(self._deferred(sequence, segment))
        if straight:
            rebuilt.append(self._clone_sequence(sequence, straight))
        return rebuilt

    def _diagnose_failure(
        self, planner, segment, start_config, goal, obstacles, holding, result
    ) -> None:
        """Re-plan the SAME query varying one thing at a time, and record the exact inputs.

        The previous version of this removed the obstacles and the attached object together, so a
        failure it blamed on "start or goal" could equally have been either. One variable at a
        time, and the inputs written out verbatim, is the difference between a diagnosis and a
        guess -- three causal claims about this failure have already not survived measurement.
        """
        import json

        from curobo.geom.types import WorldConfig

        cuboids = self._world_cuboids(obstacles)
        matrix = {}
        for with_world in (True, False):
            for with_object in (True, False):
                if with_object and not holding:
                    continue
                planner.set_world(cuboids if with_world else [])
                link = (
                    self._attach_carried_object(planner, start_config.tolist())
                    if with_object
                    else None
                )
                try:
                    points, _, outcome = planner.plan_to_joint_config(
                        start_config.tolist(), goal.tolist(), verbose=False
                    )
                finally:
                    if link is not None:
                        planner.detach_obj([link])
                key = f"world={int(with_world)},attached={int(with_object)}"
                matrix[key] = (
                    f"OK {len(points)} waypoints" if points else str(getattr(outcome, "status", "?"))
                )

        # Where the two endpoints actually are, since a plan can only be as sane as its inputs.
        robot = self.task.env.current_robot
        kinematics = robot.kinematics
        gripper = robot.robot_view.get_gripper_movegroup_ids()[0]
        group_id = self._planned_group_ids()[0]
        base_pose = robot.robot_view.base.pose
        poses = {}
        for label, values in (("start", start_config), ("goal", goal)):
            try:
                poses[label] = kinematics.fk({group_id: values}, base_pose)[gripper][:3, 3].tolist()
            except Exception:
                poses[label] = None

        payload = {
            "phase": segment.name,
            "status": str(getattr(result, "status", None)),
            "start_config": [float(v) for v in start_config],
            "goal_config": [float(v) for v in goal],
            "tcp_start_world": poses["start"],
            "tcp_goal_world": poses["goal"],
            "segment_start_pose": np.asarray(segment.start_pose).tolist(),
            "segment_end_pose": np.asarray(segment.end_pose).tolist(),
            "holding": holding,
            "matrix": matrix,
            "obstacles": [
                {"name": c.name, "pose": list(map(float, c.pose)), "dims": list(map(float, c.dims))}
                for c in cuboids
            ],
            # The payload too, or the dump only replays the empty-gripper query -- which is the
            # one cell of the matrix that already succeeds. Same base frame and same instant as
            # the obstacles, so a replay can attach it and reproduce the failure exactly.
            "payload": [
                {"name": c.name, "pose": list(map(float, c.pose)), "dims": list(map(float, c.dims))}
                for c in (self._carried_geometry() if holding else [])
            ],
        }
        self.curobo_diagnostics[segment.name] = {"status": "plan_failed", **matrix}
        log.warning(f"[CUROBO] {segment.name!r} failure matrix: {matrix}")
        try:
            directory = Path(self.config.output_dir) / "curobo_plans"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / f"FAILED_{segment.name}.json").write_text(json.dumps(payload, indent=1))
        except Exception as error:
            log.debug(f"[CUROBO] could not dump failure: {error}")

    def _dump_plan(
        self, phase, waypoints, motion_time, obstacles, group_id: str, attached_link
    ) -> None:
        """Write one plan to ``<output_dir>/curobo_plans/`` as JSON.

        The waypoints are the only record of what the planner asked for; the trajectory in the
        HDF5 is what the robot did, and the two turning out to differ is precisely the failure
        worth being able to see after the fact.
        """
        directory = Path(self.config.output_dir) / "curobo_plans"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            index = len(list(directory.glob("plan_*.json")))
            (directory / f"plan_{index:03d}_{phase}.json").write_text(
                json.dumps(
                    {
                        "phase": phase,
                        "attached_link": attached_link,
                        "move_group": group_id,
                        "joint_names": list(self._bridged.dof_joint_names),
                        "interpolated_waypoints": [list(map(float, w)) for w in waypoints],
                        "motion_time": float(motion_time),
                        "obstacles": [obstacle.name for obstacle in obstacles],
                        "base_pose": self.task.env.current_robot.get_world_pose_tf_mat().tolist(),
                    }
                )
            )
        except Exception as error:  # a diagnostic must never take the episode down with it
            log.debug(f"[CUROBO] could not dump plan: {error}")

    def _compute_trajectory(self) -> list[ActionPrimitive]:
        primitives = super()._compute_trajectory()
        self.curobo_diagnostics = {}
        self.plan_points = {}
        self.plan_obstacles = {}

        # Segment targets are read off the primitives rather than from self.target_poses: reset()
        # populates that dict by walking the primitives this method RETURNS, so it is still empty
        # while this method runs.
        rebuilt: list[ActionPrimitive] = []
        for primitive in primitives:
            if isinstance(primitive, TCPMoveSequence):
                rebuilt.extend(self._rebuild_sequence(primitive))
            else:
                rebuilt.append(primitive)
        return rebuilt

    def get_info(self) -> dict:
        info = super().get_info()
        info["curobo"] = dict(self.curobo_diagnostics)
        return info
