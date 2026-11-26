"""Reject successes the robot did not cause, during datagen.

The task predicates never require the robot: pick-and-place checks support + no contact at the success
frame, close checks one joint value, open takes the max over every joint of every instance of the target's
category. So a rollout can "succeed" from a lid that falls shut, an object already resting in its receptacle,
or an object knocked into place. This is the datagen side of embgen post-run Fix 8 (branch embgen-postrun-fix,
scripts/embgen_postrun_fix/postrun_uncaused_success_fix.py), with the same three rules judged at ``s`` = the first
frame of the last success run (frame 0 = the reset state, frame i = after policy step i):

  no_contact      all tasks with a target: no robot body touched the target object's body subtree
                  (an articulated object's doors/drawers included) in frames [0, s]
  no_grasp        pick family: the gripper closure never rose MIN_GRASP_CLOSURE above its running minimum
                  in frames [0, s] (closure 0 = fingers at their max distance, 1 = at their min distance)
  target_unmoved  pick family: the target object moved < MIN_TARGET_POS (m) and < MIN_TARGET_ROT (rad)
                  from its frame-0 pose in frames [0, s]
  wrong_instance  open: the named target's own opening at s (max of OpeningTask.joint_openings, the per-object
                  terms of its get_reward) is below task_success_threshold -- the success came from another instance
                  of its category (any_inst_of_category), not the object the task names

There is no switch: ParallelRolloutRunner.run_single_rollout builds the gate for every rollout whose task names a
target (pickup_obj_name), prints a ``SUCCESS_GATE {json}`` line for every success it judges, and turns a success
with any reason into a failure. Runners that reuse run_single_rollout (e.g. JsonEvalRunner) are gated too.

Offline the scene has to be rebuilt for no_contact; here the live MjData already holds the contacts, so all
three are a few array reads per policy step. Contacts and poses are sampled after each policy step (the
same frames the replay bundle records), not at every physics substep.

The gripper closure uses the gripper's own inter_finger_dist / inter_finger_dist_range, which every
GripperGroup provides; the offline filter normalises driver qpos by the embodiment's open/closed calibration
instead, so the two closures agree in direction and range but not exactly in scale.
"""

from __future__ import annotations

import numpy as np

MIN_GRASP_CLOSURE = 0.04
MIN_TARGET_POS = 0.01
MIN_TARGET_ROT = 0.1


def success_run_start(frame_success: list[bool]) -> int | None:
    """First frame of the last success run, or None if no frame succeeded."""
    hits = [i for i, ok in enumerate(frame_success) if ok]
    if not hits:
        return None
    start = hits[-1]
    while start > 0 and frame_success[start - 1]:
        start -= 1
    return start


def measure(
    s: int,
    contact: list[bool],
    closure: list[float] | None,
    positions: list[np.ndarray] | None,
    quats: list[np.ndarray] | None,
    opening: list[float] | None = None,
    open_threshold: float | None = None,
) -> dict:
    """The rule signals over frames [0, s]: first robot-target contact frame (None if none), largest gripper
    closure rise, largest target displacement (m, rad), the target's opening at s. A signal is None where its rule
    does not apply (not a pick-family / open task, or unavailable)."""
    hits = [i for i, touching in enumerate(contact[: s + 1]) if touching]
    out = {"success_start": s, "first_contact": hits[0] if hits else None,
           "grasp_closure": None, "target_pos": None, "target_rot": None,
           "target_opening": float(opening[s]) if opening is not None else None, "open_threshold": open_threshold}
    if closure is not None:
        c = np.asarray(closure[: s + 1], dtype=np.float64)
        out["grasp_closure"] = float((c - np.minimum.accumulate(c)).max())
    if positions is not None and quats is not None:
        pos = np.asarray(positions[: s + 1], dtype=np.float64)
        quat = np.asarray(quats[: s + 1], dtype=np.float64)
        out["target_pos"] = float(np.linalg.norm(pos - pos[0], axis=1).max())
        out["target_rot"] = float((2 * np.arccos(np.clip(np.abs((quat * quat[0]).sum(axis=1)), 0.0, 1.0))).max())
    return out


def reasons_from(signals: dict) -> list[str]:
    reasons = []
    if signals["first_contact"] is None:
        reasons.append("no_contact")
    if signals["grasp_closure"] is not None and signals["grasp_closure"] < MIN_GRASP_CLOSURE:
        reasons.append("no_grasp")
    if signals["target_pos"] is not None and (signals["target_pos"] < MIN_TARGET_POS
                                             and signals["target_rot"] < MIN_TARGET_ROT):
        reasons.append("target_unmoved")
    if signals.get("target_opening") is not None and signals["target_opening"] < signals["open_threshold"]:
        reasons.append("wrong_instance")
    return reasons


def judge(
    s: int,
    contact: list[bool],
    closure: list[float] | None,
    positions: list[np.ndarray] | None,
    quats: list[np.ndarray] | None,
    opening: list[float] | None = None,
    open_threshold: float | None = None,
) -> list[str]:
    """Rule verdicts over frames [0, s]. ``closure`` / ``positions`` + ``quats`` / ``opening`` are None where the
    rule does not apply (not a pick-family / open task, or the signal is unavailable)."""
    return reasons_from(measure(s, contact, closure, positions, quats, opening, open_threshold))


def robot_target_contact(model, data, robot_bodies: np.ndarray, target_bodies: np.ndarray) -> bool:
    """Whether any active contact pairs a robot body with a target body."""
    if not data.ncon:
        return False
    pairs = model.geom_bodyid[data.contact.geom[: data.ncon]]
    return bool(((robot_bodies[pairs[:, 0]] & target_bodies[pairs[:, 1]])
                 | (robot_bodies[pairs[:, 1]] & target_bodies[pairs[:, 0]])).any())


class RobotCausedSuccessGate:
    """Per-frame signal recorder for one rollout. Construct right after ``task.reset()`` (captures frame 0),
    call ``capture(success)`` after every ``task.step``, then ``reasons(final_success)``."""

    def __init__(self, task):
        from molmo_spaces.env.data_views import MlSpacesArticulationObject, MlSpacesObject
        from molmo_spaces.tasks.opening_tasks import OpeningTask
        from molmo_spaces.tasks.pick_and_place_task import PickAndPlaceTask
        from molmo_spaces.tasks.pick_task import PickTask

        self.task = task
        self.enabled = False
        self.disabled_reason = None
        task_config = getattr(getattr(self.task, "config", None), "task_config", None)
        name = getattr(task_config, "pickup_obj_name", None)
        if not name:
            self.disabled_reason = "task has no pickup_obj_name target"
            return
        env = self.task.env
        self._model, self._data = env.current_model, env.current_data
        robot_view = env.current_robot.robot_view

        self._target_obj = MlSpacesObject(name, self._data)
        self._target_bodies = np.zeros(self._model.nbody, dtype=bool)
        self._target_bodies[list(self._target_obj.body_ids)] = True
        self._robot_bodies = self._model.body_rootid == robot_view.base.root_body_id

        pick_family = isinstance(self.task, (PickTask, PickAndPlaceTask)) and not isinstance(self.task, OpeningTask)
        self._gripper = None
        if pick_family:
            try:
                self._gripper = robot_view.get_gripper(robot_view.get_gripper_movegroup_ids()[0])
                lo, hi = self._gripper.inter_finger_dist_range
                self._finger_range = (float(lo), float(hi))
                if not hi > lo:
                    raise ValueError(f"degenerate inter_finger_dist_range {self._gripper.inter_finger_dist_range}")
            except Exception as exc:  # noqa: BLE001 -- no_grasp is skipped, the other rules still apply
                print(f"success gate: no_grasp disabled ({type(exc).__name__}: {exc})")
                self._gripper = None
        self._pick_family = pick_family
        self._open_threshold = None
        if isinstance(self.task, OpeningTask) and getattr(self.task.config, "task_type", None) == "open":
            self._open_target = MlSpacesArticulationObject(name, self._data)
            self._joint_openings = OpeningTask.joint_openings
            self._open_threshold = float(task_config.task_success_threshold)

        self._contact: list[bool] = []
        self._closure: list[float] = []
        self._positions: list[np.ndarray] = []
        self._quats: list[np.ndarray] = []
        self._opening: list[float] = []
        self._success: list[bool] = []
        self.enabled = True
        self.capture(False)   # frame 0: the reset state

    def capture(self, success: bool) -> None:
        if not self.enabled:
            return
        self._success.append(bool(success))
        self._contact.append(robot_target_contact(self._model, self._data, self._robot_bodies, self._target_bodies))
        if self._pick_family:
            self._positions.append(np.asarray(self._target_obj.position, dtype=np.float64))
            self._quats.append(np.asarray(self._target_obj.quat, dtype=np.float64))
            if self._gripper is not None:
                lo, hi = self._finger_range
                self._closure.append((hi - float(self._gripper.inter_finger_dist)) / (hi - lo))
        if self._open_threshold is not None:
            self._opening.append(float(max(self._joint_openings(self._open_target), default=0.0)))

    def evaluate(self, final_success: bool) -> dict | None:
        """Signals + drop reasons for a rollout judged successful (None otherwise). The success frame is the start
        of the last per-step success run, or the final frame when only the post-rollout judge_success() held."""
        if not (self.enabled and final_success):
            return None
        s = success_run_start(self._success)
        if s is None:
            s = len(self._success) - 1
        signals = measure(
            s,
            self._contact,
            self._closure if self._gripper is not None else None,
            self._positions if self._pick_family else None,
            self._quats if self._pick_family else None,
            self._opening if self._open_threshold is not None else None,
            self._open_threshold,
        )
        return {**signals, "n_frames": len(self._success), "reasons": reasons_from(signals)}

    def reasons(self, final_success: bool) -> list[str]:
        """Drop reasons for a rollout judged successful; [] keeps it."""
        verdict = self.evaluate(final_success)
        return verdict["reasons"] if verdict else []
