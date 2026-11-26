from pathlib import Path
from types import SimpleNamespace

import pytest

from molmo_spaces.configs.camera_configs import (
    FrankaOmniPurposeCameraSystem,
    GenericOmniPurposeCameraSystem,
)
from molmo_spaces.configs.robot_configs import FrankaRobotConfig
from molmo_spaces.data_generation.config.nav_to_obj_configs import (
    ConfiguredRBYNavToObjDataGenConfig,
)
from molmo_spaces.data_generation.config.object_manipulation_datagen_configs import (
    FrankaPickOmniCamConfig,
    GenericPickAndPlaceOmniCamConfig,
    GenericPickOmniCamConfig,
)
from molmo_spaces.data_generation.main import (
    apply_cli_overrides,
    create_experiment_config,
)
from molmo_spaces.robots.configured import ConfiguredRobotConfig

SPEC_DIR = Path(__file__).parents[2] / "docs" / "embodiment_specs"


@pytest.mark.parametrize(
    ("filename", "embodiment_id"),
    [("franka_droid.yaml", "franka_droid.single.fixed")],
)
def test_generic_omni_config_injects_embodiment(filename: str, embodiment_id: str):
    """The generic config must give a generated robot the SAME view composition Franka gets.

    It used to be exocentric-only -- three RandomizedExocentricCameraConfigs and nothing mounted on
    the robot -- because no single MJCF camera name resolved across embodiments. That made generated
    data structurally different from the shipped Franka data, which is multi-view, so a model could
    not be trained on both. It is now wrist + shoulder + the same exocentric trio."""
    config, run_name = create_experiment_config(
        "GenericPickOmniCamConfig",
        SPEC_DIR / filename,
    )

    assert isinstance(config, GenericPickOmniCamConfig)
    assert isinstance(config.robot_config, ConfiguredRobotConfig)
    assert config.robot_config.embodiment.embodiment_id == embodiment_id
    assert embodiment_id in run_name
    assert isinstance(config.camera_config, GenericOmniPurposeCameraSystem)
    kinds = [type(camera).__name__ for camera in config.camera_config.cameras]
    assert (
        kinds
        == ["MjcfCameraConfig", "RobotMountedCameraConfig"]
        + ["RandomizedExocentricCameraConfig"] * 3
    ), kinds
    assert not config.filter_for_successful_trajectories


def test_generic_omni_config_rejects_an_embodiment_with_no_wrist_camera():
    """An MjcfCameraConfig resolves BY NAME, so an embodiment whose MJCF declares no camera at all
    cannot have a wrist stream -- and that has to fail loudly at config time, not silently drop a
    column halfway through a dataset.

    i2rt_yam is that embodiment: its MJCF has no camera under any name. A GENERATED robot carrying
    the yam_jaw gripper does have one, because embodiment_gen derives and emits it (see
    ``builder._add_wrist_camera``); the shipped spec cannot, since the spec schema has no add_camera
    operation and the MJCF is downloaded and versioned."""
    with pytest.raises(ValueError, match="wrist_camera_zed_mini"):
        create_experiment_config("GenericPickOmniCamConfig", SPEC_DIR / "i2rt_yam.yaml")


def test_generic_config_requires_embodiment():
    with pytest.raises(ValueError, match="requires --embodiment-spec"):
        create_experiment_config("GenericPickOmniCamConfig")


def test_generic_pick_and_place_config_injects_embodiment():
    config, _ = create_experiment_config(
        "GenericPickAndPlaceOmniCamConfig",
        SPEC_DIR / "franka_droid.yaml",
    )
    original_franka_config, _ = create_experiment_config("FrankaPickOmniCamConfig")

    assert isinstance(config, GenericPickAndPlaceOmniCamConfig)
    assert isinstance(config.robot_config, ConfiguredRobotConfig)
    assert config.end_on_success
    assert not config.filter_for_successful_trajectories
    assert config.task_sampler_config.ensure_jointly_ik_feasible_base_pose
    assert config.task_sampler_config.max_robot_placement_attempts == 50
    assert config.policy_config.grasp_closing_axis_rotation_degrees == (90.0,)
    assert config.policy_config.placement_local_x_rotation_degrees == (-90.0, 0.0, 90.0)
    assert config.policy_config.placement_local_y_rotation_degrees == (-90.0, 0.0, 90.0)
    assert config.robot_config.action_noise_config == (
        original_franka_config.robot_config.action_noise_config
    )


def test_curobo_planner_seed_requires_a_planner_that_has_one(tmp_path: Path):
    """--curobo-planner-seed exists to reseed cuRobo's solver without moving the scene, so it is
    meaningless on a policy that has no solver seed. Rejected rather than silently ignored."""
    config, _ = create_experiment_config("GenericPickOmniCamConfig", SPEC_DIR / "franka_droid.yaml")
    args = SimpleNamespace(
        output_dir=None, seed=None, num_workers=None, max_tasks=None, samples_per_house=None,
        episodes_per_batch=None, scene_dataset=None, data_split=None, house_ind=None,
        pickup_obj_name=None, curobo_planner_seed=7,
    )
    with pytest.raises(ValueError, match="requires a cuRobo planner policy config"):
        apply_cli_overrides(config, args)


def test_manipulation_rejects_embodiment_without_batch_ik():
    with pytest.raises(ValueError, match="requires configured batch IK"):
        create_experiment_config(
            "GenericPickOmniCamConfig",
            SPEC_DIR / "rby1.yaml",
        )


def test_configured_rby_navigation_uses_required_head_camera():
    config, _ = create_experiment_config(
        "ConfiguredRBYNavToObjDataGenConfig",
        SPEC_DIR / "rby1m.yaml",
    )

    assert isinstance(config, ConfiguredRBYNavToObjDataGenConfig)
    assert config.camera_config.get_camera_by_name("head_camera") is not None
    assert not config.filter_for_successful_trajectories


def test_navigation_rejects_relative_base_profile():
    with pytest.raises(ValueError, match="absolute planar-position"):
        create_experiment_config(
            "ConfiguredRBYNavToObjDataGenConfig",
            SPEC_DIR / "rby1m_open_close.yaml",
        )


def test_navigation_accepts_explicit_default_profile():
    config, run_name = create_experiment_config(
        "ConfiguredRBYNavToObjDataGenConfig",
        SPEC_DIR / "rby1m_open_close.yaml",
        control_profile="default",
    )

    assert config.robot_config.control_profile == "default"
    assert run_name.endswith("__profile-default")


def test_control_profile_requires_embodiment():
    with pytest.raises(ValueError, match="requires --embodiment-spec"):
        create_experiment_config("FrankaPickOmniCamConfig", control_profile="default")


def test_legacy_registered_config_keeps_declared_robot_and_cameras():
    config, run_name = create_experiment_config("FrankaPickOmniCamConfig")
    assert isinstance(config, FrankaPickOmniCamConfig)
    assert isinstance(config.robot_config, FrankaRobotConfig)
    assert isinstance(config.camera_config, FrankaOmniPurposeCameraSystem)
    assert run_name == "FrankaPickOmniCamConfig"


def test_droid_mounted_omni_cameras_are_rejected_for_yam():
    with pytest.raises(ValueError, match="Camera system is incompatible"):
        create_experiment_config(
            "FrankaPickOmniCamConfig",
            SPEC_DIR / "i2rt_yam.yaml",
        )


def test_cli_overrides_follow_existing_exp_config_structure(tmp_path: Path):
    config, _ = create_experiment_config(
        "GenericPickOmniCamConfig",
        SPEC_DIR / "franka_droid.yaml",
    )
    args = SimpleNamespace(
        output_dir=tmp_path,
        seed=42,
        num_workers=2,
        max_tasks=3,
        samples_per_house=5,
        episodes_per_batch=5,
        scene_dataset="ithor",
        data_split="val",
        house_ind=4,
        pickup_obj_name="pickup",
        barrier_height_fraction=None,
        # every flag apply_cli_overrides reads must appear here, since it reads them off args
        # directly -- argparse always supplies them, so a missing one is this fixture falling
        # behind, and it surfaces as an AttributeError rather than a useful failure
        curobo_planner_seed=None,
    )
    apply_cli_overrides(config, args)

    assert config.output_dir == tmp_path
    assert config.seed == 42
    assert config.num_workers == 2
    assert config.task_sampler_config.max_tasks == 3
    assert config.task_sampler_config.samples_per_house == 5
    assert config.task_sampler_config.episodes_per_batch == 5
    assert config.scene_dataset == "ithor"
    assert config.data_split == "val"
    assert config.task_sampler_config.house_inds == [4]
    assert config.task_config.pickup_obj_name == "pickup"
