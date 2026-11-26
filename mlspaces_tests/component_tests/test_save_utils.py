from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np

from molmo_spaces.env.abstract_sensors import SensorSuite
from molmo_spaces.env.sensors_cameras import CameraSensor
from molmo_spaces.policy.solvers.object_manipulation.base_object_manipulation_planner_policy import (
    GraspPlanDiagnosticsSensor,
    GraspPoseSensor,
    PhaseTargetPoseSensor,
    ReleasePoseSensor,
)
from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import PickPlannerPolicy
from molmo_spaces.utils.save_utils import (
    SAVED_VIDEO_SENSORS_KEY,
    _annotate_grasp_frames,
    _grasp_diagnostic_lines,
    _save_extra_data_from_batched,
    _save_sensor_data_from_batched,
    byte_array_to_string,
    prepare_episode_for_saving,
)


def test_grasp_pose_sensor_returns_nan_without_a_valid_plan():
    policy = object.__new__(PickPlannerPolicy)
    policy.target_poses = {}
    task = SimpleNamespace(_registered_policy=policy)

    observation = GraspPoseSensor().get_observation(None, task)

    assert np.isnan(observation).all()


def test_grasp_plan_diagnostics_sensor_returns_selector_details():
    policy = object.__new__(PickPlannerPolicy)
    policy.grasp_plan_diagnostics = {"status": "failed", "failure_phase": "lift"}
    task = SimpleNamespace(_registered_policy=policy)

    observation = GraspPlanDiagnosticsSensor().get_observation(None, task)

    assert observation == {"status": "failed", "failure_phase": "lift"}


def test_release_pose_sensor_requires_a_complete_place_plan():
    policy = object.__new__(PickPlannerPolicy)
    policy.target_poses = {}
    task = SimpleNamespace(_registered_policy=policy)
    sensor = ReleasePoseSensor()

    assert np.isnan(sensor.get_observation(None, task)).all()

    policy.target_poses = {"place": np.eye(4)}
    np.testing.assert_allclose(
        sensor.get_observation(None, task),
        [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
    )


def test_phase_target_pose_sensor_tracks_active_motion_endpoint():
    policy = object.__new__(PickPlannerPolicy)
    target_pose = np.eye(4)
    target_pose[:3, 3] = [0.1, 0.2, 0.3]
    policy.target_poses = {"lift": target_pose}
    policy.action_idx = 0
    policy.action_primitives = [SimpleNamespace(get_current_phase=lambda: "lift")]
    task = SimpleNamespace(_registered_policy=policy)

    observation = PhaseTargetPoseSensor().get_observation(None, task)

    np.testing.assert_allclose(observation, [0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0])


def test_grasp_diagnostic_lines_show_every_phase_and_cause():
    lines = _grasp_diagnostic_lines(
        {
            "status": "failed",
            "candidate_count": 7000,
            "ranked_count": 512,
            "collision_free_count": 18,
            "ik_checked_count": 18,
            "jointly_feasible_count": 0,
            "phase_feasible_counts": {
                "grasp": {"feasible": 14, "cumulative": 14},
                "pregrasp": {"feasible": 9, "cumulative": 7},
                "lift": {"feasible": 5, "cumulative": 2},
                "preplace": {"feasible": 3, "cumulative": 0},
                "place": {"feasible": 4, "cumulative": 0},
            },
            "fallback": {
                "status": "selected",
                "jointly_feasible_count": 4,
                "phase_feasible_counts": {
                    "grasp": {"feasible": 8, "cumulative": 8},
                    "pregrasp": {"feasible": 6, "cumulative": 4},
                },
            },
            "lift_fallback": {
                "status": "selected_reduced_height",
                "requested_height": 0.181,
                "selected_height": 0.134,
            },
            "failure_cause": "No candidate passed every required IK phase",
            "placement_orientation_search": {
                "status": "selected_place_priority_fallback",
                "num_orientations": 9,
                "num_feasible": 2,
                "selected_x_degrees": 0.0,
                "selected_y_degrees": 90.0,
                "requested_preplace_clearance": 0.07,
                "selected_preplace_clearance": 0.04,
                "selected_place_exact": False,
                "attempts": [
                    {
                        "x_degrees": 0.0,
                        "y_degrees": 90.0,
                        "place_position_error": 0.003,
                        "place_rotation_error": np.radians(0.5),
                    }
                ],
            },
        }
    )

    assert "candidates: raw=7000  ranked=512  collision-free=18  IK-checked=18" in lines
    for phase in ("grasp", "pregrasp", "lift", "preplace", "place"):
        assert any(line.startswith(f"{phase}:") for line in lines)
    assert "fallback (grasp+pregrasp): selected; jointly feasible=4" in lines
    assert "lift best effort: 13.4 cm (requested 18.1 cm)" in lines
    assert "placement orientations: feasible=2/9" in lines
    assert "placement selected: X=0 deg, Y=90 deg" in lines
    assert "preplace clearance: 4.0 cm (requested 7.0 cm)" in lines
    assert "place best effort: 3.0 mm, 0.5 deg" in lines
    assert lines[-1] == "cause: No candidate passed every required IK phase"


def test_grasp_annotation_suppresses_marker_for_invalid_pose():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    camera_name = "camera"
    camera_params = {
        "extrinsic_cv": np.eye(3, 4),
        "intrinsic_cv": np.array(
            [
                [50.0, 0.0, 50.0],
                [0.0, 50.0, 50.0],
                [0.0, 0.0, 1.0],
            ]
        ),
    }

    def annotate(grasp_pose, release_pose=None, phase_target_pose=None, phase=0):
        observations = [
            {
                "grasp_pose": np.asarray(grasp_pose),
                "release_pose": np.asarray(
                    np.full(7, np.nan) if release_pose is None else release_pose
                ),
                "phase_target_pose": np.asarray(
                    np.full(7, np.nan) if phase_target_pose is None else phase_target_pose
                ),
                "policy_phase": phase,
                f"sensor_param_{camera_name}": camera_params,
            }
        ]
        return _annotate_grasp_frames([frame], observations, camera_name)[0]

    valid = annotate([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0])
    invalid = annotate([0.0, 0.0, 1.0, np.nan, np.nan, np.nan, np.nan])
    magenta = np.array([255, 0, 255])

    assert np.any(np.all(valid == magenta, axis=-1))
    assert not np.any(np.all(invalid == magenta, axis=-1))

    release_pose = [0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    before_pickup = annotate(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        release_pose,
        phase=4,
    )
    after_pickup = annotate(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        release_pose,
        phase_target_pose=[-0.2, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        phase=5,
    )
    cyan = np.array([0, 220, 255])
    assert not np.any(np.all(before_pickup == cyan, axis=-1))
    assert np.any(np.all(after_pickup == cyan, axis=-1))
    green = np.array([80, 255, 80])
    assert np.any(np.all(after_pickup == green, axis=-1))

    release_observations = [
        {
            "grasp_pose": np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
            "release_pose": np.asarray(release_pose),
            "policy_phase": phase,
            f"sensor_param_{camera_name}": camera_params,
        }
        for phase in (7, 1, 1)
    ]
    release_frames = _annotate_grasp_frames(
        [frame, frame, frame],
        release_observations,
        camera_name,
    )
    assert all(np.any(np.all(release_frame == cyan, axis=-1)) for release_frame in release_frames)


def test_saved_video_references_survive_camera_frame_removal(tmp_path):
    camera_name = "randomized_zed2_analogue_1"
    sensor_suite = SensorSuite(
        [CameraSensor(camera_name=camera_name, img_resolution=(4, 4), uuid=camera_name)]
    )
    frame = np.zeros((4, 4, 3), dtype=np.uint8)
    history = {
        "observations": [[{camera_name: frame}], [{camera_name: frame}]],
    }

    with patch("molmo_spaces.utils.save_utils.save_videos_from_raw_observations"):
        episode = prepare_episode_for_saving(
            history,
            sensor_suite,
            fps=15.0,
            save_dir=tmp_path,
            episode_idx=3,
            save_file_suffix="_batch_1_of_1",
        )

    assert camera_name not in episode
    assert episode[SAVED_VIDEO_SENSORS_KEY] == {camera_name: "rgb"}

    with h5py.File(tmp_path / "refs.h5", "w") as output:
        obs_group = output.create_group("obs")
        _save_sensor_data_from_batched(
            obs_group,
            episode,
            episode_idx=3,
            save_file_suffix="_batch_1_of_1",
        )
        reference = obs_group["sensor_data"][camera_name][:]

    assert byte_array_to_string(reference) == (
        "episode_00000003_randomized_zed2_analogue_1_batch_1_of_1.mp4"
    )


def test_grasp_plan_diagnostics_are_saved_to_hdf5(tmp_path):
    encoded = np.zeros((2, 32), dtype=np.uint8)
    encoded[:, :2] = np.frombuffer(b"{}", dtype=np.uint8)
    release_poses = np.zeros((2, 7), dtype=np.float32)
    phase_target_poses = np.zeros((2, 7), dtype=np.float32)

    with h5py.File(tmp_path / "diagnostics.h5", "w") as output:
        obs_group = output.create_group("obs")
        _save_extra_data_from_batched(
            obs_group,
            {
                "grasp_plan_diagnostics": encoded,
                "release_pose": release_poses,
                "phase_target_pose": phase_target_poses,
            },
        )
        saved = obs_group["extra/grasp_plan_diagnostics"][:]
        saved_release_poses = obs_group["extra/release_pose"][:]
        saved_phase_target_poses = obs_group["extra/phase_target_pose"][:]

    np.testing.assert_array_equal(saved, encoded)
    np.testing.assert_array_equal(saved_release_poses, release_poses)
    np.testing.assert_array_equal(saved_phase_target_poses, phase_target_poses)
