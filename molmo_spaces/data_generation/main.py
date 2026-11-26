import argparse
import datetime
import glob
import importlib
import os
import re
from pathlib import Path

# Before ``import mujoco``, which is the deadline: mujoco/gl_context.py:24 reads MUJOCO_GL at module
# import and binds the GL backend there, so setting it later has no effect. Unset, it picks GLFW and
# a headless machine dies with "gladLoadGL error" -- the GPU path only avoided this by accident,
# because importing molmo_spaces.renderer.opengl_context forces MUJOCO_GL=egl as a side effect
# (opengl_context.py:111) and the no-GPU branch never imports it. EGL degrades to Mesa software
# rendering on its own when there is no GPU, so this is all a GPU-free run needs.
from molmo_spaces.utils.patch_renderer_flags import patch_renderer_flags

patch_renderer_flags()

import mujoco  # noqa: E402

from molmo_spaces.configs.camera_configs import MjcfCameraConfig, RobotMountedCameraConfig
from molmo_spaces.data_generation.config_registry import get_config_class
from molmo_spaces.data_generation.pipeline import ParallelRolloutRunner
from molmo_spaces.robots.configured import ConfiguredRobotConfig

"""
Main script entry for data generation.

To run:
- Set terminal at molmo-spaces root directory
- For MacOS, set the following environment variables:
  - export PYTHONPATH="${PYTHONPATH}:."
  - export MUJOCO_GL=egl
  - export PYOPENGL_PLATFORM=egl
- Example commands:
  - python -m molmo_spaces.data_generation.main DoorOpeningDebugConfig
  - python -m molmo_spaces.data_generation.main DoorOpeningDataGenConfig
  - python -m molmo_spaces.data_generation.main GenericPickOmniCamConfig --embodiment-spec docs/embodiment_specs/i2rt_yam.yaml
- You may also pass additional experiment config arguments for your experiment config class as command line arguments.

Config classes are auto-discovered from the config_registry. To add a new config:
1. Create your config class in any file under config/
2. Add @register_config("YourConfigName") decorator to register it
3. Use the registered name as the command line argument
"""


def get_args(argv=None):
    parser = argparse.ArgumentParser(
        description="MolmoSpaces data generation pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "exp_config_cls",
        type=str,
        help="Name of the experiment config class to use (e.g., FrankaPickDroidDataGenConfig), "
        "optionally with the module name prepended with a colon (e.g. molmo_spaces.data_generation.config.object_manipulation_datagen_configs:FrankaPickDroidDataGenConfig). "
        "If the module is specified, only that module will be imported to populate the registry. Otherwise, all config files will be imported.",
    )
    parser.add_argument(
        "--embodiment-spec",
        type=Path,
        help="EmbodimentSpec YAML to inject into the selected experiment config",
    )
    parser.add_argument(
        "--control-profile",
        help="Override the embodiment's default control profile",
    )
    parser.add_argument("--output-dir", type=Path, help="Override the base output directory")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Write a per-episode sim-replay bundle (a lean DR recipe + full per-step MuJoCo "
        "integration state + camera poses + meta) next to each saved trajectory, sufficient to "
        "deterministically re-sim and re-render the episode by fresh-compiling the base scene. "
        "Off by default so kept outputs are unchanged; the capture draws no RNG and mutates nothing.",
    )
    parser.add_argument("--seed", type=int, help="Override the experiment seed")
    parser.add_argument("--num-workers", type=int, help="Override worker count")
    parser.add_argument("--max-tasks", type=int, help="Limit sampled tasks")
    parser.add_argument(
        "--samples-per-house",
        type=int,
        help="Override the number of completed trajectories requested per house",
    )
    parser.add_argument(
        "--episodes-per-batch",
        type=int,
        help="Override the number of trajectories saved in each batch",
    )
    parser.add_argument(
        "--scene-dataset",
        help="Override the scene dataset houses are drawn from (ithor, procthor-10k, "
        "procthor-objaverse, holodeck-objaverse)",
    )
    parser.add_argument("--data-split", help="Override the scene dataset split")
    parser.add_argument("--house-ind", type=int, help="Run one house index")
    parser.add_argument(
        "--curobo-planner-seed",
        type=int,
        help="Seed cuRobo's solver independently of --seed, so a flaky solve can be retried "
        "without changing the scene or the task",
    )
    parser.add_argument(
        "--barrier-height-fraction",
        type=float,
        help="Height of the task barrier as a multiple of the pickup object's height, so one "
        "config can be swept over easy and impossible fixtures without editing it",
    )
    parser.add_argument("--pickup-obj-name", help="Select one pickup object by scene name")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Run cuRobo on the CPU instead of a GPU. Forces the CPU kernel backend even when a GPU "
        "is visible, which is how you A/B the two on one machine; with no GPU it is selected "
        "automatically and this flag is redundant. Rendering needs nothing extra -- MuJoCo's EGL "
        "path falls back to software rendering on its own when no GPU is visible.",
    )
    return parser.parse_args(argv)


def auto_import_configs() -> None:
    """Auto-import all config files so they register themselves"""
    # Get the config directory path
    current_dir = os.path.dirname(__file__)
    config_dir = os.path.join(current_dir, "config")

    if not os.path.exists(config_dir):
        print(f"Warning: Config directory not found: {config_dir}")
        return

    # Import all .py files in the config directory
    config_files = glob.glob(os.path.join(config_dir, "*.py"))

    for config_path in config_files:
        # Skip __init__.py
        if config_path.endswith("__init__.py"):
            continue

        # Load the module with the full module path for proper pickling
        module_filename = os.path.splitext(os.path.basename(config_path))[0]
        full_module_name = f"molmo_spaces.data_generation.config.{module_filename}"

        # Use standard import instead of spec_from_file_location
        # This ensures the module has the correct __name__ for pickling
        try:
            importlib.import_module(full_module_name)
        except Exception as e:
            print(f"Warning: Could not load config from {full_module_name}: {e}")
            continue


def _resolve_config_class(selector: str):
    config_name = selector
    if ":" in selector:
        module_name, config_name = selector.split(":", maxsplit=1)
        importlib.import_module(module_name)
    else:
        auto_import_configs()
    return config_name, get_config_class(config_name)


def validate_robot_camera_compatibility(exp_config) -> None:
    """Resolve mounted-camera references before starting worker processes."""
    camera_config = exp_config.camera_config
    robot_config = exp_config.robot_config
    scene = mujoco.MjSpec()
    robot_config.robot_cls.add_robot_to_scene(
        robot_config,
        scene,
        prefix=robot_config.robot_namespace,
        pos=[0.0, 0.0, 0.0],
        quat=[1.0, 0.0, 0.0, 0.0],
        strip_meshes=True,
    )
    model = scene.compile()

    if camera_config is None:
        return
    camera_names = [camera.name for camera in camera_config.cameras]
    if len(camera_names) != len(set(camera_names)):
        raise ValueError("camera_config contains duplicate camera names")

    mounted_cameras = [
        camera
        for camera in camera_config.cameras
        if isinstance(camera, MjcfCameraConfig | RobotMountedCameraConfig)
    ]

    unresolved = []
    for camera in mounted_cameras:
        if isinstance(camera, MjcfCameraConfig):
            candidates = camera.candidate_mjcf_names()
            declared = {model.camera(i).name for i in range(model.ncam)}
            if not declared & set(candidates):
                unresolved.append(f"camera {camera.name!r} -> MJCF camera {' / '.join(candidates)}")
        else:
            if not any(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name) >= 0
                for body_name in camera.reference_body_names
            ):
                unresolved.append(
                    f"camera {camera.name!r} -> bodies {camera.reference_body_names!r}"
                )
    if unresolved:
        details = "; ".join(unresolved)
        raise ValueError(f"Camera system is incompatible with the configured embodiment: {details}")


def validate_robot_task_compatibility(exp_config) -> None:
    """Reject configured embodiment and task combinations that policies cannot control."""
    robot_config = exp_config.robot_config
    if not isinstance(robot_config, ConfiguredRobotConfig):
        return

    embodiment = robot_config.embodiment
    profile = embodiment.robot_config.control_profiles[robot_config.control_profile]
    controller_kinds = {
        group_id: embodiment.robot.controllers[controller_id].kind
        for group_id, controller_id in profile.controller_bindings.items()
    }

    if exp_config.task_type in {"pick", "pick_and_place", "open", "close"}:
        if not embodiment.robot_view.end_effectors:
            raise ValueError("Manipulation tasks require at least one configured end effector")

        kinematics = embodiment.robot.kinematics
        has_batch_ik = kinematics.kind == "floating_frame" or (
            kinematics.kind == "damped_least_squares" and kinematics.batch_backend is not None
        )
        if not has_batch_ik:
            raise ValueError(
                f"Task {exp_config.task_type!r} requires configured batch IK, but embodiment "
                f"{embodiment.embodiment_id!r} does not provide it"
            )

        for end_effector in embodiment.robot_view.end_effectors.values():
            if end_effector.arm_group is not None:
                arm_controller = controller_kinds.get(end_effector.arm_group)
                if arm_controller != "joint_position":
                    raise ValueError(
                        f"Task {exp_config.task_type!r} requires absolute joint-position arm "
                        f"control; {end_effector.arm_group!r} uses {arm_controller!r}"
                    )
            tool_controller = controller_kinds.get(end_effector.tool_group)
            if tool_controller not in {"joint_position", "direct_actuator"}:
                raise ValueError(
                    f"Task {exp_config.task_type!r} cannot control tool "
                    f"{end_effector.tool_group!r} with {tool_controller!r}"
                )

    if exp_config.task_type == "nav_to_obj":
        placement = embodiment.robot_config.placement
        if placement.runtime_mobility != "se2":
            raise ValueError("Navigation tasks require an SE(2) configured embodiment")
        base_group = embodiment.robot_view.base_group
        if controller_kinds.get(base_group) != "planar_position":
            raise ValueError("Navigation tasks require absolute planar-position base control")
        camera_names = (
            {camera.name for camera in exp_config.camera_config.cameras}
            if exp_config.camera_config is not None
            else set()
        )
        if "head_camera" not in camera_names:
            raise ValueError("Navigation tasks require a camera named 'head_camera'")


def create_experiment_config(
    selector: str,
    embodiment_spec: Path | None = None,
    control_profile: str | None = None,
):
    """Instantiate a registered experiment config and optionally inject an embodiment."""
    config_name, config_class = _resolve_config_class(selector)
    if control_profile is not None and embodiment_spec is None:
        raise ValueError("--control-profile requires --embodiment-spec")

    if embodiment_spec is not None:
        configured_robot = ConfiguredRobotConfig.from_file(
            embodiment_spec,
            control_profile=control_profile,
        )
        exp_config = config_class(robot_config=configured_robot)
        embodiment_name = re.sub(
            r"[^A-Za-z0-9_.-]+", "_", configured_robot.embodiment.embodiment_id
        )
        config_name = f"{config_name}__{embodiment_name}"
        if control_profile is not None:
            profile_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", configured_robot.control_profile)
            config_name = f"{config_name}__profile-{profile_name}"
    else:
        exp_config = config_class()
    if exp_config.robot_config is None:
        raise ValueError(
            f"Experiment config {config_name!r} requires --embodiment-spec because it has no robot_config"
        )
    validate_robot_task_compatibility(exp_config)
    validate_robot_camera_compatibility(exp_config)
    return exp_config, config_name


def apply_cli_overrides(exp_config, args) -> None:
    if args.output_dir is not None:
        exp_config.output_dir = args.output_dir
    if args.seed is not None:
        exp_config.seed = args.seed
    if args.num_workers is not None:
        exp_config.num_workers = args.num_workers
    if args.max_tasks is not None:
        exp_config.task_sampler_config.max_tasks = args.max_tasks
    if args.samples_per_house is not None:
        exp_config.task_sampler_config.samples_per_house = args.samples_per_house
    if args.episodes_per_batch is not None:
        exp_config.task_sampler_config.episodes_per_batch = args.episodes_per_batch
    if args.scene_dataset is not None:
        exp_config.scene_dataset = args.scene_dataset
    if args.data_split is not None:
        exp_config.data_split = args.data_split
    if args.house_ind is not None:
        exp_config.task_sampler_config.house_inds = [args.house_ind]
    if args.curobo_planner_seed is not None:
        if not hasattr(exp_config.policy_config, "planner_seed"):
            raise ValueError("--curobo-planner-seed requires a cuRobo planner policy config")
        exp_config.policy_config.planner_seed = args.curobo_planner_seed
    if args.barrier_height_fraction is not None:
        if not hasattr(exp_config.policy_config, "barrier_height_fraction"):
            raise ValueError("--barrier-height-fraction requires a policy config with a barrier")
        exp_config.policy_config.barrier_height_fraction = args.barrier_height_fraction
        exp_config.policy_config.barrier_enabled = args.barrier_height_fraction > 0.0
    if args.pickup_obj_name is not None:
        exp_config.task_config.pickup_obj_name = args.pickup_obj_name
    if args.replay:
        exp_config.capture_replay = True
        # Inputs to fresh-compile a bundle's base scene at replay (scene/split/house are already on exp_config).
        exp_config.replay_recon_info = {
            "config_selector": args.exp_config_cls,
            "embodiment_spec": str(args.embodiment_spec) if args.embodiment_spec else None,
            "control_profile": args.control_profile,
            "asset_blacklist_path": os.environ.get("MLSPACES_ASSET_BLACKLIST_PATH"),
        }


def main(argv=None) -> None:
    args = get_args(argv)

    # Before anything constructs a planner: with no GPU (or with --cpu) this points cuRobo's compiled
    # CUDA extensions at the CPU backend. A no-op on a GPU machine unless asked for.
    #
    # Exported rather than passed down: worker processes are spawned, so they re-read the environment
    # but not this function's arguments, and the planner they build has to make the same choice.
    if args.cpu:
        os.environ["CUROBO_CPU"] = "1"

    from molmo_spaces.planner.curobo_cpu.install import maybe_install

    if maybe_install():
        print("[CUROBO] running on CPU: kernel shim installed")

    exp_config, exp_config_name = create_experiment_config(
        args.exp_config_cls,
        args.embodiment_spec,
        args.control_profile,
    )
    apply_cli_overrides(exp_config, args)

    # Optional: Modify the config parameters here if needed
    # Eg. for hyperparamter sweeps etc.

    # Generate unique run name
    # Determine output directory structure
    # For local debugging (non-shared paths), add timestamp to avoid collisions
    # For production (datagen output targets shared filesystem paths), use simple structure
    is_shared_fs_path = "/mnt/shared" in str(
        exp_config.output_dir
    )  # or whatever your mount point is

    if is_shared_fs_path:
        # Production: simple structure without timestamp
        exp_config.output_dir = exp_config.output_dir / exp_config_name
    else:
        # Local debug: add timestamp to avoid collisions
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_config.output_dir = exp_config.output_dir / exp_config_name / timestamp

    os.makedirs(exp_config.output_dir, exist_ok=True)

    # Initialize wandb if enabled - with auto run name
    if exp_config.use_wandb:
        import wandb

        if exp_config.wandb_name is None:
            # Generate timestamp for wandb run name
            wandb_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            exp_config.wandb_name = f"{exp_config_name}_{wandb_timestamp}"
        wandb.init(
            project=exp_config.wandb_project, name=exp_config.wandb_name, config=vars(exp_config)
        )

    exp_config.save_config()

    # Create rollout runner with the set config parameters
    runner = ParallelRolloutRunner(exp_config)

    success_count, total_count = runner.run()
    print(f"Success count: {success_count}, Total count: {total_count}")

    # Close wandb run
    if exp_config.use_wandb:
        wandb.finish()


def run_redirected(argv: list[str], log_path: str) -> None:
    """Forkserver entry for one datagen run (see generate_embodiment_dataset.run_datagen). Sends every
    write -- including C-level MuJoCo/GL output -- to log_path for the caller to parse, then runs it."""
    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)
    main(argv)


if __name__ == "__main__":
    main()
