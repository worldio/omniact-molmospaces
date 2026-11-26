from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from molmo_spaces.env.env import CPUMujocoEnv
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    BaseObjectManipulationPlannerPolicy,
)
from molmo_spaces.tasks.pick_task_sampler import PickTaskSampler


class _FakePolicy:
    def __init__(self, targets, diagnostics=None, infeasible_pose=None):
        self.targets = targets
        self.grasp_plan_diagnostics = diagnostics or {"status": "selected"}
        self.infeasible_pose = infeasible_pose

    def _compute_target_poses(self):
        return self.targets

    def check_feasible_ik(self, pose):
        return pose is not self.infeasible_pose


class _FakePlannerPolicy(BaseObjectManipulationPlannerPolicy):
    def __init__(self, config, task):
        self.grasp_plan_diagnostics = {"status": "not_evaluated"}
        self._ik_solution_cache = {}
        self.target_poses = {
            phase: np.eye(4)
            for phase in ("pregrasp", "grasp", "lift", "preplace", "place", "postplace")
        }

    def _compute_trajectory(self):
        return []


def test_all_phase_ik_requires_every_exact_endpoint():
    poses = {phase: np.eye(4) for phase in ("pregrasp", "grasp", "lift")}
    policy = _FakePolicy(poses)

    assert PickTaskSampler._policy_has_usable_all_phase_ik(policy, list(poses))

    policy.infeasible_pose = poses["lift"]
    assert not PickTaskSampler._policy_has_usable_all_phase_ik(policy, list(poses))


def test_all_phase_ik_accepts_usable_policy_and_placement_fallbacks():
    phases = ["pregrasp", "grasp", "lift", "preplace", "place", "postplace"]
    poses = {phase: np.eye(4) for phase in phases}

    policy = _FakePolicy(poses, {"status": "selected_via_grasp_pregrasp_fallback"})
    assert PickTaskSampler._policy_has_usable_all_phase_ik(policy, phases)

    policy.grasp_plan_diagnostics = {
        "status": "selected",
        "placement_orientation_search": {"status": "selected_place_priority_fallback"},
    }
    assert PickTaskSampler._policy_has_usable_all_phase_ik(policy, phases)


def test_all_phase_filter_supports_non_pick_sampler_callers():
    robot_view = SimpleNamespace(base=SimpleNamespace(pose=np.eye(4)))
    env = SimpleNamespace(
        current_robot=SimpleNamespace(
            robot_view=robot_view,
            parallel_kinematics=SimpleNamespace(warmup_ik=lambda _: None),
        ),
        current_model=object(),
        current_data=object(),
    )
    sampler = SimpleNamespace(
        config=SimpleNamespace(
            task_type="pick_and_place",
            task_sampler_config=SimpleNamespace(
                ensure_jointly_ik_feasible_base_pose=True,
            ),
            policy_config=SimpleNamespace(
                policy_cls=_FakePlannerPolicy,
                grasp_feasibility_batch_size=8,
            ),
        )
    )

    with (
        patch(
            "molmo_spaces.tasks.pick_task_sampler.mujoco.mj_forward",
        ),
        patch.object(
            PickTaskSampler,
            "_policy_has_usable_all_phase_ik",
            return_value=True,
        ) as exact_ik,
    ):
        candidate_filter = PickTaskSampler._create_all_phase_ik_base_pose_filter(
            sampler, env
        )
        assert candidate_filter(np.eye(4))

    exact_ik.assert_called_once()
    checked_policy, required_phases = exact_ik.call_args.args
    assert isinstance(checked_policy, _FakePlannerPolicy)
    assert required_phases == [
        "pregrasp",
        "grasp",
        "lift",
        "preplace",
        "place",
        "postplace",
    ]
    assert len(env._all_phase_ik_plans) == 1


def test_place_robot_near_resamples_until_candidate_filter_accepts():
    env = object.__new__(CPUMujocoEnv)
    env._mj_model = object()
    env._current_batch_index = 0
    env._mj_datas = [object()]
    env._n_batch = 1
    env.close = lambda: None
    env.config = SimpleNamespace(
        collision_free_pose_limit=3,
        robot_config=SimpleNamespace(
            base_placement_forward_offset=0.0,
            base_placement_forward_offset_step=0.02,
        ),
        task_sampler_config=SimpleNamespace(
            robot_placement_exclusion_threshold=0.15,
            robot_placement_rotation_range_rad=0.0,
        ),
    )
    env.get_thormap = lambda **kwargs: SimpleNamespace(
        get_free_points=lambda: np.array([[0.1, 0.0, 0.0], [0.2, 0.0, 0.0]])
    )
    env.check_if_robot_collision_at_base_pose = lambda *args, **kwargs: False
    robot_view = SimpleNamespace(base=SimpleNamespace(pose=np.eye(4)))
    checked = []

    def candidate_filter(pose):
        checked.append((float(pose[0, 3]), float(pose[2, 3])))
        return pose[0, 3] == 0.2 and pose[2, 3] == 0.19

    with (
        patch(
            "molmo_spaces.env.env.sample_around_point",
            side_effect=[np.array([0.1, 0.0]), np.array([0.2, 0.0])],
        ),
        patch("molmo_spaces.env.env.np.random.uniform", side_effect=[0.11, 0.19]),
        patch("molmo_spaces.env.env.mujoco.mj_forward"),
    ):
        placed = env.place_robot_near(
            robot_view,
            target=np.array([0.0, 0.0, 0.0]),
            max_tries=2,
            sampling_radius_range=(0.0, 1.0),
            robot_safety_radius=0.15,
            preserve_z=0.3,
            sampling_z_range=(0.1, 0.2),
            face_target=False,
            candidate_pose_filter=candidate_filter,
        )

    assert placed
    assert checked == [(0.1, 0.11), (0.2, 0.19)]
    np.testing.assert_allclose(robot_view.base.pose[:3, 3], [0.2, 0.0, 0.19])
