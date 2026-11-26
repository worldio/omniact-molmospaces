"""Datagen success gate (molmo_spaces/data_generation/success_gates.py): the three rules on synthetic signals, and
the live reads (contacts, target pose, gripper closure) on a tiny MuJoCo scene driven like a rollout.

The scene: a mocap "robot" block (robot_0/base) with a fake gripper, a free cube on the floor, a chest with a hinged
lid (range 1.5 rad). Frame 0 is captured at
construction; every mj_step + capture() stands in for one policy step.
"""

from types import SimpleNamespace

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
sg = pytest.importorskip("molmo_spaces.data_generation.success_gates")
PickTask = pytest.importorskip("molmo_spaces.tasks.pick_task").PickTask
OpeningTask = pytest.importorskip("molmo_spaces.tasks.opening_tasks").OpeningTask

SCENE = """
<mujoco>
  <compiler angle="radian"/>
  <option timestep="0.002"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="robot_0/base" mocap="true" pos="0.3 0 0.05">
      <geom name="finger" type="box" size="0.03 0.03 0.03"/>
    </body>
    <body name="cube" pos="0 0 0.05">
      <freejoint/>
      <geom type="box" size="0.05 0.05 0.05" mass="0.2"/>
    </body>
    <body name="chest" pos="-0.6 0 0.05">
      <geom type="box" size="0.05 0.05 0.05"/>
      <body name="chest_lid" pos="0 0 0.06">
        <joint name="lid_hinge" type="hinge" axis="0 1 0" range="0 1.5"/>
        <geom type="box" size="0.05 0.05 0.005" mass="0.05"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# ---- pure rules --------------------------------------------------------------------------------------------
def test_success_run_start_is_the_last_run():
    assert sg.success_run_start([False, True, True, False, False, True, True]) == 5
    assert sg.success_run_start([False, False]) is None
    assert sg.success_run_start([False, True]) == 1


def _poses(n, moved_from=None, dz=0.05):
    pos = [np.zeros(3) if moved_from is None or i < moved_from else np.array([0, 0, dz]) for i in range(n)]
    return pos, [np.array([1.0, 0, 0, 0])] * n


def test_judge_untouched_unmoved_ungrasped_fails_all_three():
    pos, quat = _poses(6)
    assert sg.judge(4, [False] * 6, [0.0] * 6, pos, quat) == ["no_contact", "no_grasp", "target_unmoved"]


def test_judge_genuine_pick_passes():
    pos, quat = _poses(6, moved_from=4)
    assert sg.judge(5, [False, False, True, True, True, True], [0, 0, 0.3, 0.6, 0.6, 0.6], pos, quat) == []


def test_judge_ignores_everything_after_the_success_frame():
    pos, quat = _poses(8, moved_from=5)
    reasons = sg.judge(3, [False] * 4 + [True] * 4, [0, 0, 0, 0, 0.9, 0.9, 0.9, 0.9], pos, quat)
    assert reasons == ["no_contact", "no_grasp", "target_unmoved"]


def test_judge_initial_gripper_opening_is_not_a_grasp():
    pos, quat = _poses(5, moved_from=2)
    assert sg.judge(4, [True] * 5, [1.0, 0.5, 0.0, 0.0, 0.01], pos, quat) == ["no_grasp"]


def test_judge_open_success_from_another_instance():
    touched = [True] * 4
    assert sg.judge(3, touched, None, None, None, [0.0, 0.0, 0.1, 0.01], 0.15) == ["wrong_instance"]
    assert sg.judge(3, touched, None, None, None, [0.0, 0.1, 0.2, 0.3], 0.15) == []


def test_judge_articulated_task_checks_contact_only():
    assert sg.judge(3, [False, False, True, True], None, None, None) == []
    assert sg.judge(1, [False, False, True, True], None, None, None) == ["no_contact"]


# ---- live reads through a PickTask / OpeningTask instance ------------------------------------------------
class FakeGripper:
    inter_finger_dist_range = (0.0, 0.08)

    def __init__(self):
        self.inter_finger_dist = 0.08   # fully open


def _task(task_cls):
    model = mujoco.MjModel.from_xml_string(SCENE)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    gripper = FakeGripper()
    robot_view = SimpleNamespace(
        base=SimpleNamespace(root_body_id=model.body("robot_0/base").id),
        get_gripper_movegroup_ids=lambda: ["gripper"],
        get_gripper=lambda _mg: gripper,
    )
    task = task_cls.__new__(task_cls)   # bypass the env-bound constructor; the gate reads only these
    task._env = SimpleNamespace(current_model=model, current_data=data,
                                current_robot=SimpleNamespace(robot_view=robot_view))
    task.config = SimpleNamespace(task_config=SimpleNamespace(pickup_obj_name="cube"))
    return task, model, data, gripper


def _step(model, data, gate, success, n=25):
    for _ in range(n):
        mujoco.mj_step(model, data)
    gate.capture(success)


def test_gate_drops_an_untouched_pick_success():
    task, model, data, _gripper = _task(PickTask)
    gate = sg.RobotCausedSuccessGate(task)
    assert gate.enabled
    for i in range(6):
        _step(model, data, gate, success=i >= 3)
    assert gate.reasons(True) == ["no_contact", "no_grasp", "target_unmoved"]
    assert gate.reasons(False) == [] and gate.evaluate(False) is None   # only a success is ever judged
    verdict = gate.evaluate(True)
    assert verdict["success_start"] == 4 and verdict["n_frames"] == 7 and verdict["first_contact"] is None
    assert verdict["grasp_closure"] == 0.0 and verdict["target_pos"] < 0.01


def test_gate_keeps_a_push_with_grasp_and_motion():
    task, model, data, gripper = _task(PickTask)
    gate = sg.RobotCausedSuccessGate(task)
    for i in range(40):
        data.mocap_pos[0][0] = max(0.3 - 0.01 * i, 0.0)   # sweep the robot block through the cube
        gripper.inter_finger_dist = 0.08 if i < 5 else 0.02
        _step(model, data, gate, success=i >= 35)
    verdict = gate.evaluate(True)
    assert verdict["first_contact"] is not None and verdict["first_contact"] <= verdict["success_start"] == 36
    assert verdict["grasp_closure"] > 0.5 and verdict["target_pos"] > 0.01
    assert verdict["reasons"] == []


def test_gate_articulated_task_applies_no_contact_only():
    task, model, data, _gripper = _task(OpeningTask)
    gate = sg.RobotCausedSuccessGate(task)
    for i in range(5):
        _step(model, data, gate, success=i >= 2)
    assert gate._closure == [] and gate._positions == []
    assert gate.reasons(True) == ["no_contact"]


@pytest.mark.parametrize("task_type, lid, expect", [
    ("open", 0.6, ["no_contact"]),                     # the named chest opened to 0.4: its own success
    ("open", 0.0, ["no_contact", "wrong_instance"]),   # the named chest shut: the success came from another instance
    ("close", 0.0, ["no_contact"]),                    # close scores the named object alone
])
def test_gate_open_requires_the_named_target_open(task_type, lid, expect):
    task, model, data, _gripper = _task(OpeningTask)
    task.config.task_type = task_type
    task.config.task_config = SimpleNamespace(pickup_obj_name="chest", task_success_threshold=0.15)
    gate = sg.RobotCausedSuccessGate(task)
    for i in range(5):
        data.qpos[model.jnt_qposadr[model.joint("lid_hinge").id]] = lid if i >= 2 else 0.0
        mujoco.mj_forward(model, data)
        gate.capture(i >= 2)
    verdict = gate.evaluate(True)
    assert verdict["reasons"] == expect, verdict
    if task_type == "open":
        assert verdict["target_opening"] == pytest.approx(lid / 1.5) and verdict["open_threshold"] == 0.15


def test_gate_disables_without_a_target():
    task, _model, _data, _gripper = _task(PickTask)
    task.config.task_config.pickup_obj_name = None
    gate = sg.RobotCausedSuccessGate(task)
    assert not gate.enabled and gate.reasons(True) == []
