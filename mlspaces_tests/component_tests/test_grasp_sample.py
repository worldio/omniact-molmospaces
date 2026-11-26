from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    TCPMoveSegment,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
    PickAndPlacePlannerPolicy,
)
from molmo_spaces.utils.grasp_sample import get_feasible_grasp_idx, select_grasp_pose


class _FakeParallelKinematics:
    def __init__(self, phase_masks: list[list[bool]]) -> None:
        self.phase_masks = iter(phase_masks)

    def ik(self, *args, **kwargs):
        return [object() if feasible else None for feasible in next(self.phase_masks)]


def test_decoupled_tcp_interpolation_keeps_position_linear_during_rotation():
    start = np.eye(4)
    start[2, 3] = 1.0
    end = np.eye(4)
    end[:3, :3] = R.from_euler("y", 90, degrees=True).as_matrix()
    end[:3, 3] = [1.0, 0.0, 1.2]
    segment = TCPMoveSegment(
        name="preplace",
        start_pose=start,
        end_pose=end,
        speed=0.2,
        decoupled_interpolation=True,
    )

    midpoint = segment.interpolate(0.5)

    np.testing.assert_allclose(midpoint[:3, 3], [0.5, 0.0, 1.1])
    np.testing.assert_allclose(
        midpoint[:3, :3],
        R.from_euler("y", 45, degrees=True).as_matrix(),
        atol=1e-7,
    )


def test_grasp_ik_diagnostic_counts_each_phase_and_cumulative_survivors():
    robot = SimpleNamespace(
        parallel_kinematics=_FakeParallelKinematics(
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ]
        ),
        robot_view=SimpleNamespace(
            get_qpos_dict=lambda: {},
            base=SimpleNamespace(pose=np.eye(4)),
        ),
    )
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    diagnostic = {}

    result = get_feasible_grasp_idx(
        "gripper",
        robot,
        poses,
        n_ik_checks=3,
        ik_batch_size=3,
        additional_ik_poses_world=[poses, poses],
        pose_names=["grasp", "pregrasp", "lift"],
        diagnostic=diagnostic,
    )

    assert result is None
    assert diagnostic == {
        "ik_checked_count": 3,
        "phase_feasible_counts": {
            "grasp": {"feasible": 2, "cumulative": 2},
            "pregrasp": {"feasible": 2, "cumulative": 1},
            "lift": {"feasible": 2, "cumulative": 0},
        },
        "jointly_feasible_count": 0,
    }


def test_grasp_selector_identifies_first_phase_with_no_cumulative_survivor():
    robot_view = SimpleNamespace(
        get_gripper_movegroup_ids=lambda: ["gripper"],
        get_move_group=lambda _: SimpleNamespace(leaf_frame_to_world=np.eye(4)),
        get_qpos_dict=lambda: {},
        base=SimpleNamespace(pose=np.eye(4)),
    )
    robot = SimpleNamespace(
        parallel_kinematics=_FakeParallelKinematics(
            [
                [True, True, False],
                [True, False, True],
                [False, True, True],
            ]
        ),
        robot_view=robot_view,
    )
    env = SimpleNamespace(current_robot=robot)
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    diagnostic = {}

    with pytest.raises(ValueError, match="No feasible grasp found"):
        select_grasp_pose(
            env,
            poses,
            np.eye(4),
            check_collision=False,
            n_collision_checks=3,
            collision_batch_size=3,
            check_ik=True,
            n_ik_checks=3,
            ik_batch_size=3,
            additional_ik_poses_world=[poses, poses],
            ik_pose_names=["grasp", "pregrasp", "lift"],
            diagnostic=diagnostic,
        )

    assert diagnostic["failure_phase"] == "lift"
    assert diagnostic["failure_cause"] == (
        "Candidates passed lift IK individually, but none also passed every preceding phase"
    )


class _AllFeasibleKinematics:
    def ik(self, _mg_id, pose_batch, *args, **kwargs):
        return [object() for _ in range(len(pose_batch))]


def _grasp_selection_env():
    robot_view = SimpleNamespace(
        get_gripper_movegroup_ids=lambda: ["gripper"],
        get_move_group=lambda _: SimpleNamespace(leaf_frame_to_world=np.eye(4)),
        get_qpos_dict=lambda: {},
        base=SimpleNamespace(pose=np.eye(4)),
    )
    robot = SimpleNamespace(
        parallel_kinematics=_AllFeasibleKinematics(),
        robot_view=robot_view,
    )
    return SimpleNamespace(current_robot=robot)


def test_retry_prefers_a_different_grasp_site_then_falls_back():
    env = _grasp_selection_env()
    poses = np.broadcast_to(np.eye(4), (3, 4, 4)).copy()
    poses[0, :3, 3] = [0.00, 0.0, 0.0]
    poses[1, :3, 3] = [0.10, 0.0, 0.0]
    poses[2, :3, 3] = [0.20, 0.0, 0.0]
    kwargs = dict(
        check_collision=False,
        n_collision_checks=3,
        collision_batch_size=3,
        check_ik=True,
        n_ik_checks=3,
        ik_batch_size=3,
    )

    # No exclusion: the best-ranked (nearest) grasp is chosen.
    baseline = select_grasp_pose(env, poses, np.eye(4), **kwargs)
    np.testing.assert_allclose(baseline[:3, 3], [0.0, 0.0, 0.0])

    # Having attempted the x=0 grasp, a retry must pick a different site.
    retried = select_grasp_pose(
        env,
        poses,
        np.eye(4),
        exclude_object_grasps=[poses[0].copy()],
        exclude_radius=0.04,
        **kwargs,
    )
    np.testing.assert_allclose(retried[:3, 3], [0.10, 0.0, 0.0])

    # When every site has been attempted, fall back to any feasible grasp ("otherwise any grasp").
    fallback = select_grasp_pose(
        env,
        poses,
        np.eye(4),
        exclude_object_grasps=[p.copy() for p in poses],
        exclude_radius=0.04,
        **kwargs,
    )
    np.testing.assert_allclose(fallback[:3, 3], [0.0, 0.0, 0.0])


def test_independent_placement_orientations_try_zero_rotation_first():
    policy = object.__new__(PickAndPlacePlannerPolicy)
    policy.policy_config = SimpleNamespace(
        placement_local_x_rotation_degrees=(-90.0, 0.0, 90.0),
        placement_local_y_rotation_degrees=(-90.0, 0.0, 90.0),
    )

    offsets = policy._placement_orientation_offsets()

    assert offsets[0] == (0.0, 0.0)
    assert len(offsets) == 9
    assert len(set(offsets)) == 9


def test_pick_orientation_transform_uses_local_closing_axis():
    policy = object.__new__(PickAndPlacePlannerPolicy)
    policy.policy_config = SimpleNamespace(
        grasp_closing_axis_rotation_degrees=(90.0,),
        grasp_local_x_rotation_degrees=(0.0,),
    )
    grasp = np.eye(4)[None]

    augmented = policy._augment_grasp_candidates(grasp)

    assert len(augmented) == 1
    np.testing.assert_allclose(
        augmented[0, :3, :3],
        R.from_euler("y", 90.0, degrees=True).as_matrix(),
        atol=1e-7,
    )


def test_best_effort_preplace_reduces_clearance_for_exact_place():
    policy = object.__new__(PickAndPlacePlannerPolicy)
    policy.policy_config = SimpleNamespace(
        placement_local_x_rotation_degrees=(0.0,),
        placement_local_y_rotation_degrees=(0.0,),
        place_z_offset=0.07,
        end_z_offset=0.05,
        placement_fallback_position_tolerance=0.005,
        placement_fallback_rotation_tolerance=np.radians(1.0),
    )
    model = object()
    policy.task = SimpleNamespace(
        env=SimpleNamespace(
            current_data=SimpleNamespace(model=model),
            current_robot=SimpleNamespace(
                kinematics=SimpleNamespace(last_ik_diagnostic=None),
            ),
        )
    )
    policy.grasp_plan_diagnostics = {}
    policy._ik_solution_cache = {}
    policy._capture_ik_diagnostic = lambda *args, **kwargs: None

    checked_heights = []

    def check_feasible_ik(pose):
        checked_heights.append(float(pose[2, 3]))
        return pose[2, 3] <= 1.14 + 1e-9

    policy.check_feasible_ik = check_feasible_ik
    pickup_obj = SimpleNamespace(name="pickup", object_id=2, pose=np.eye(4))
    place_receptacle = SimpleNamespace(
        object_id=1,
        position=np.array([0.0, 0.0, 1.0]),
    )
    scratch_pickup = SimpleNamespace(body_id=2, pose=None)

    with (
        patch(
            "molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy.mujoco.MjData",
            return_value=SimpleNamespace(),
        ),
        patch(
            "molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy.mujoco.mj_copyData",
        ),
        patch(
            "molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy.mujoco.mj_fwdPosition",
        ),
        patch(
            "molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy.create_mlspaces_body",
            return_value=scratch_pickup,
        ),
        patch(
            "molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy.body_aabb",
            side_effect=[
                (np.array([0.0, 0.0, 1.0]), np.array([1.0, 1.0, 0.2])),
                (np.array([0.0, 0.0, 0.05]), np.array([0.1, 0.1, 0.1])),
            ],
        ),
    ):
        preplace, place, _ = policy._get_independent_placement_poses(
            np.eye(4),
            pickup_obj,
            place_receptacle,
        )

    assert place[2, 3] == pytest.approx(1.1)
    assert preplace[2, 3] == pytest.approx(1.14)
    assert checked_heights == pytest.approx([1.17, 1.1, 1.16, 1.15, 1.14])
    assert policy.grasp_plan_diagnostics["placement_orientation_search"] == {
        "attempts": [
            {
                "x_degrees": 0.0,
                "y_degrees": 0.0,
                "preplace_feasible": False,
                "place_feasible": True,
                "failure_phase": "preplace",
                "fallback_preplace_clearance": pytest.approx(0.04),
            }
        ],
        "num_orientations": 1,
        "num_feasible": 0,
        "status": "selected_place_priority_fallback",
        "selected_x_degrees": 0.0,
        "selected_y_degrees": 0.0,
        "requested_preplace_clearance": pytest.approx(0.07),
        "selected_preplace_clearance": pytest.approx(0.04),
        "selected_place_exact": True,
    }


def test_best_effort_lift_selects_highest_reachable_reduced_height():
    policy = object.__new__(PickAndPlacePlannerPolicy)
    policy.grasp_plan_diagnostics = {}
    checked_heights = []

    def check_feasible_ik(pose):
        height = float(pose[2, 3] - 1.0)
        checked_heights.append(height)
        return height <= 0.04 + 1e-9

    policy.check_feasible_ik = check_feasible_ik
    grasp_pose = np.eye(4)
    grasp_pose[2, 3] = 1.0
    requested_lift_pose = grasp_pose.copy()
    requested_lift_pose[2, 3] = 1.08

    selected = policy._get_best_effort_lift_pose(grasp_pose, requested_lift_pose)

    assert selected[2, 3] == pytest.approx(1.04)
    assert checked_heights == pytest.approx([0.07, 0.06, 0.05, 0.04])
    assert policy.grasp_plan_diagnostics["lift_fallback"] == {
        "status": "selected_reduced_height",
        "requested_height": pytest.approx(0.08),
        "selected_height": pytest.approx(0.04),
    }


def test_best_effort_lift_falls_back_to_grasp_pose_when_no_lift_is_reachable():
    policy = object.__new__(PickAndPlacePlannerPolicy)
    policy.grasp_plan_diagnostics = {}
    policy.check_feasible_ik = lambda pose: False
    grasp_pose = np.eye(4)
    requested_lift_pose = grasp_pose.copy()
    requested_lift_pose[2, 3] = 0.03

    selected = policy._get_best_effort_lift_pose(grasp_pose, requested_lift_pose)

    np.testing.assert_array_equal(selected, grasp_pose)
    assert policy.grasp_plan_diagnostics["lift_fallback"] == {
        "status": "selected_zero_height",
        "requested_height": pytest.approx(0.03),
        "selected_height": 0.0,
    }
