import logging
import os
from copy import deepcopy

import numpy as np
import torch

# Before the first cuRobo import, not after: importing ``curobo.wrap.reacher.motion_gen`` evaluates
# ``Pose.from_list(..., TensorDeviceType())`` in the MotionGen class body (motion_gen.py:4204), which
# touches CUDA at *import* time and dies with "No CUDA GPUs are available" on a CPU-only machine.
# Installing here rather than from an entry point also covers spawned worker processes, which get a
# fresh interpreter and never run main().
from molmo_spaces.planner.curobo_cpu.install import maybe_install as _maybe_install_cpu_backend

_maybe_install_cpu_backend()

from curobo.geom.sdf.world import CollisionCheckerType  # noqa: E402

log = logging.getLogger(__name__)
from curobo.geom.sphere_fit import SphereFitType
from curobo.geom.types import Cuboid, WorldConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.types.base import TensorDeviceType

# cuRobo
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import (
    get_robot_configs_path,
    join_path,
    load_yaml,
)
from curobo.wrap.reacher.motion_gen import (
    MotionGen,
    MotionGenConfig,
    MotionGenPlanConfig,
)

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.env.data_views import MlSpacesObject
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.planner.abstract import Planner
from molmo_spaces.utils.pose import pose_mat_to_7d
from molmo_spaces.utils.spatial_utils import Transform


class CuroboPlannerConfig(Config):
    # --- Curobo setup parameters ---
    curobo_robot_config_path: str | None = None
    robot_cfg: dict | None = None
    """An already-built ``robot_cfg`` mapping, as produced by :mod:`~.curobo_bridge`.

    Mutually exclusive with ``curobo_robot_config_path``. Embodiments supplied at the command line
    have no YAML on disk to point at, so their config is derived at runtime instead of loaded.
    """
    world_config: WorldConfig = None
    kinematics_config: dict = None
    lock_joints: dict | None = None  # Override locked joint values: {joint_name: value}

    # --- Robot asset paths (optional, defaults to rby1 for backward compatibility) ---
    urdf_path: str | None = None
    asset_root_path: str | None = None
    usd_robot_root: str | None = None
    collision_spheres_path: str | None = None

    # --- Motion planner parameters ---
    trajopt_tsteps: int = 20
    interpolation_dt: float = 0.02  # Match control_dt for smooth execution in MuJoCo
    time_dilation_factor: float = 1.0
    collision_activation_distance: float = (
        0.2  # collision cost calculated within this distance (metres)
    )
    num_ik_seeds: int = 64
    num_trajopt_seeds: int = 4
    fixed_iters_trajopt: bool = True
    maximum_trajectory_dt: float = 0.5
    max_attempts: int = 5
    # Preallocated on the GPU, so this is a ceiling on how much world can ever be loaded. 80 was
    # enough for a handful of bodies; a cluttered kitchen counter is 426 collision geoms once the
    # objects standing on it are included, and a world silently clipped to a fraction of that
    # produces plans that look clean and drive through the furniture that got dropped.
    collision_cache: dict = {"mesh": 3, "obb": 512}
    check_start_validity: bool = True
    warmup: bool = True
    """Run a throwaway plan at construction to pay for CUDA graph capture up front.

    On by default because the pre-existing callers -- the RBY1-style policies and the planner
    server -- plan through ``plan_batch`` with Pose goals, which is the Cartesian path warmup
    targets, and their first-call latency is the point of warming.

    Callers that plan in JOINT space should pass ``warmup=False``: warmup is invoked with
    ``warmup_js_trajopt=False``, so it does not touch the ``plan_single_js`` path at all. Measured
    on the same datagen episode either way: time to "planner ready" 18m03s warmed against 4m51s
    unwarmed, and the gap from there to the first completed plan unchanged at 1m39s versus 1m42s
    -- all cost, no benefit, because warmup runs at whatever iteration budget is configured on a
    synthetic query that does not converge and so runs to the cap."""
    enable_finetune_trajopt: bool = True
    grad_trajopt_iters: int | None = None
    """Trajectory-optimisation iterations per attempt. ``None`` keeps cuRobo's default."""
    finetune_trajopt_iters: int | None = None
    warmup: bool | None = None
    """Run ``MotionGen.warmup`` after building. ``None`` means "only on CUDA".

    Warmup runs a full **pose-goal** plan at ``finetune_trajopt_iters``, which is expensive (~14 min
    of a 26 min GPU episode at 80000 iters) and mostly exists to pre-record CUDA graphs -- and
    ``use_cuda_graph`` is False here anyway. On CPU it is also the only thing that touches
    ``geom_cu.pose_distance``, which has no CPU implementation because a joint-space plan never
    reaches it. So skipping it on CPU removes the dominant cost and the dependency together.
    """

    device: str | None = None
    """Where cuRobo runs. ``None`` resolves to CUDA when a GPU is visible and CPU otherwise.

    Defaulting to the literal ``"cuda"`` was wrong: nothing in the policy configs sets this, so a
    CPU-only run still asked for a GPU and died inside the planner constructor. Auto-resolution
    means the CPU path needs no config change anywhere -- the same datagen config runs on either.

    On CPU the kernel shim is also required
    (:func:`molmo_spaces.planner.curobo_cpu.install.maybe_install`, installed at import of this
    module) since cuRobo's compiled extensions assert ``x.is_cuda()`` on entry."""

    enable_graph: bool = False
    enable_graph_attempt: int | None = None
    """Attempt number at which to fall back to cuRobo's GRAPH search.

    ``None`` disables the fallback, leaving pure local trajectory optimisation -- which cannot find
    a path that requires a topological detour, such as going over or around a wall standing between
    start and goal. cuRobo's own default is 3; it was turned off here for RB-Y1, whose scenes have
    no such obstacle."""


class CuroboPlanner(Planner):
    def __init__(
        self,
        config: CuroboPlannerConfig,
    ) -> None:
        self.config = config
        # An explicit index matters: cuRobo's own default is ``torch.device("cuda", 0)``, and warp
        # indexes its device list with ``torch_device.index`` (``warp/_src/torch.py:53``), so a
        # bare ``torch.device("cuda")`` with index None raises there. Resolving the index needs a
        # live CUDA context, so it must not be attempted when there is no GPU.
        requested = self.config.device
        if requested is None:
            # CUROBO_CPU (set by main.py's --cpu) forces CPU even where a GPU is visible, matching
            # what the kernel shim keys off so the two cannot disagree about which backend is live.
            forced = os.environ.get("CUROBO_CPU")
            if forced is not None and bool(int(forced)):
                requested = "cpu"
            else:
                requested = "cuda" if torch.cuda.is_available() else "cpu"
        device = torch.device(requested)
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CuroboPlannerConfig.device asked for CUDA but no GPU is visible. Leave it as "
                    "None to run on whichever is present."
                )
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
        self.tensor_args = TensorDeviceType(device=device)
        self.world_config = (
            self.config.world_config if self.config.world_config is not None else WorldConfig()
        )
        self._build_robot_config(
            self.config.curobo_robot_config_path, self.config.kinematics_config
        )

        # Curobo motion planner parameters
        self.trajopt_tsteps = self.config.trajopt_tsteps
        self.interpolation_dt = self.config.interpolation_dt
        self.time_dilation_factor = self.config.time_dilation_factor
        self.collision_activation_distance = self.config.collision_activation_distance
        self.max_attempts = self.config.max_attempts
        self.fixed_iters_trajopt = self.config.fixed_iters_trajopt
        self.maximum_trajectory_dt = self.config.maximum_trajectory_dt
        self.collision_cache = self.config.collision_cache
        self.num_ik_seeds = self.config.num_ik_seeds
        self.num_trajopt_seeds = self.config.num_trajopt_seeds

        self.plan_config = MotionGenPlanConfig(
            enable_graph=self.config.enable_graph,
            max_attempts=self.max_attempts,
            enable_graph_attempt=self.config.enable_graph_attempt,
            enable_finetune_trajopt=self.config.enable_finetune_trajopt,
            partial_ik_opt=False,
            parallel_finetune=True,
            time_dilation_factor=self.time_dilation_factor,
            num_ik_seeds=self.num_ik_seeds,
            num_trajopt_seeds=self.num_trajopt_seeds,
            check_start_validity=self.config.check_start_validity,
        )

        self.attached_object2link_dict = {}
        self.attached_link2object_dict = {}
        self._build_motion_gen()

    def reset(self) -> None:
        """
        Reset the motion generator.
        """
        self.motion_gen.reset()
        self.attached_object2link_dict = {}
        self.attached_link2object_dict = {}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _get_curobo_config_paths(self) -> dict:
        # Use config paths if provided, otherwise default to rby1 for backward compatibility
        paths = {
            "urdf_path": (
                self.config.urdf_path
                if self.config.urdf_path is not None
                else str(ASSETS_DIR / "robots/rby1m/curobo_config/urdf/model_holobase.urdf")
            ),
            "asset_root_path": (
                self.config.asset_root_path
                if self.config.asset_root_path is not None
                else str(ASSETS_DIR / "robots/rby1m/curobo_config/urdf/meshes")
            ),
            "usd_robot_root": (
                self.config.usd_robot_root
                if self.config.usd_robot_root is not None
                else str(ASSETS_DIR / "robots/rby1m/curobo_config")
            ),
            "collision_spheres": (
                self.config.collision_spheres_path
                if self.config.collision_spheres_path is not None
                else str(ASSETS_DIR / "robots/rby1m/curobo_config/rby1m_holobase_spheres.yml")
            ),
        }

        return paths

    def _build_robot_config(
        self, curobo_robot_config_path: str | None, kinematics_config: dict = None
    ) -> None:
        if self.config.robot_cfg is not None:
            # A derived config already names its own URDF and spheres. The asset-path defaults
            # below are RB-Y1's and would overwrite them, so this branch takes the mapping as-is.
            curobo_robot_config_dict = deepcopy(self.config.robot_cfg)
            if "robot_cfg" in curobo_robot_config_dict:
                curobo_robot_config_dict = curobo_robot_config_dict["robot_cfg"]
            for config_key in kinematics_config or {}:
                curobo_robot_config_dict["kinematics"][config_key] = kinematics_config[config_key]
        else:
            if curobo_robot_config_path is None:
                raise ValueError(
                    "CuroboPlannerConfig needs either curobo_robot_config_path or robot_cfg"
                )
            curobo_robot_config_dict = load_yaml(
                join_path(get_robot_configs_path(), curobo_robot_config_path)
            )["robot_cfg"]
            if kinematics_config is None:
                kinematics_config = {}
            kinematics_config.update(self._get_curobo_config_paths())
            if "robot_cfg" in curobo_robot_config_dict:
                curobo_robot_config_dict = curobo_robot_config_dict["robot_cfg"]

            for config_key in kinematics_config:
                curobo_robot_config_dict["kinematics"][config_key] = kinematics_config[config_key]

        # Merge lock_joints into kinematics if provided
        if self.config.lock_joints is not None:
            if "lock_joints" not in curobo_robot_config_dict["kinematics"]:
                curobo_robot_config_dict["kinematics"]["lock_joints"] = {}
            curobo_robot_config_dict["kinematics"]["lock_joints"].update(self.config.lock_joints)

        self.curobo_robot_config_path = curobo_robot_config_path
        self.curobo_robot_config_dict = curobo_robot_config_dict  # Store the dict for later use
        self.curobo_robot_cfg = RobotConfig.from_dict(curobo_robot_config_dict, self.tensor_args)
        self.joint_names = self.curobo_robot_cfg.kinematics.cspace.joint_names
        self.default_config = self.curobo_robot_cfg.kinematics.cspace.retract_config
        self.attached_link_names = list(
            curobo_robot_config_dict["kinematics"].get("extra_links", {}).keys()
        )
        self.joint_limits = (
            self.curobo_robot_cfg.kinematics.kinematics_config.joint_limits.position.cpu().numpy()
        )

    def _build_motion_gen(self) -> None:
        motion_gen_config = MotionGenConfig.load_from_robot_config(
            self.curobo_robot_cfg,
            self.world_config,
            self.tensor_args,
            collision_checker_type=CollisionCheckerType.MESH,
            use_cuda_graph=False,
            # trim_steps=[1, -1],  # (Optional) remove first and last step (redundant)
            trajopt_tsteps=self.trajopt_tsteps,
            interpolation_dt=self.interpolation_dt,
            collision_cache=self.collision_cache,
            collision_activation_distance=self.collision_activation_distance,
            fixed_iters_trajopt=self.fixed_iters_trajopt,
            **{k: v for k, v in (
                ("grad_trajopt_iters", self.config.grad_trajopt_iters),
                ("finetune_trajopt_iters", self.config.finetune_trajopt_iters),
            ) if v is not None},
            maximum_trajectory_dt=self.maximum_trajectory_dt,
            self_collision_check=True,
            self_collision_opt=True,
            # cuRobo times its solve stages with torch.cuda.synchronize, which raises on a CPU
            # device. The timings are diagnostics only.
            sync_cuda_time=self.tensor_args.device.type == "cuda",
        )
        self.motion_gen = MotionGen(motion_gen_config)
        if self.config.warmup:
            self.motion_gen.warmup(enable_graph=False, warmup_js_trajopt=False)
        self.link_names = self.motion_gen.kinematics.link_names
        self.ee_link_name = self.motion_gen.kinematics.ee_link
        self.kin_state = self.motion_gen.kinematics.get_state(
            self.motion_gen.get_retract_config().view(1, -1)
        )
        self.link_retract_pose = self.kin_state.link_pose

    def plan_batch(
        self,
        joint_positions: list[list],
        goal_pose_lists: list[list],
        kinematics_config: dict = None,
        scene_objects: list[MlSpacesObject] = None,
        verbose: bool = False,
    ):
        """Plan motion for multiple start states and goal poses in parallel.

        Args:
            joint_positions: List of joint position lists, one for each query in the batch.
                Each joint position list should follow the same format as motion_plan.
            goal_pose_lists: List of goal pose lists (7D: [x, y, z, qw, qx, qy, qz]), one for each query.
                For now, only single end-effector goals are supported (no additional link poses).
            kinematics_config: Optional kinematics configuration dict (rebuilds planner if provided).
            scene_objects: Optional list of scene objects for collision checking.
            verbose: If True, prints detailed debug information about the planning result.

        Returns:
            tuple: (trajectories, result) where:
                - trajectories: List of trajectory lists. Each trajectory is a list of joint position waypoints.
                  Empty list for failed queries.
                - result: The full MotionGenResult object from the batch planner.
        """
        # Update the kinematics config if provided
        if kinematics_config is not None:
            print(
                "[WARNING] Rebuilding motion gen with new kinematics config. This will take a while."
            )
            self._build_robot_config(self.curobo_robot_config_path, kinematics_config)
            self._build_motion_gen()

        # Update scene objects if provided
        if scene_objects is not None:
            if verbose:
                print("Updating scene objects...")
            excluded_scene_objects = []
            for scene_object in scene_objects:
                if scene_object.name not in self.attached_object2link_dict:
                    excluded_scene_objects.append(scene_object)
            self.update_world(excluded_scene_objects)

        batch_size = len(joint_positions)
        assert len(goal_pose_lists) == batch_size, (
            "Number of joint positions and goal poses must match"
        )

        # Convert joint positions to batched JointState
        positions_tensor = torch.tensor(joint_positions, dtype=torch.float32, device=self.tensor_args.device)
        joint_state = JointState.from_position(positions_tensor, self.joint_names)
        start_state = JointState(
            position=self.tensor_args.to_device(joint_state.position),
            velocity=self.tensor_args.to_device(joint_state.velocity) * 0.0,
            acceleration=self.tensor_args.to_device(joint_state.acceleration) * 0.0,
            jerk=self.tensor_args.to_device(joint_state.jerk) * 0.0,
            joint_names=self.joint_names,
        )
        start_state = start_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        # Convert goal poses to batched Pose
        # Stack all goal poses and create a batched Pose object
        goal_poses_tensor = torch.tensor(goal_pose_lists, dtype=torch.float32, device=self.tensor_args.device)
        goal_pose = Pose(
            position=goal_poses_tensor[:, :3],
            quaternion=goal_poses_tensor[:, 3:],
        )

        # Plan batch
        result = self.motion_gen.plan_batch(
            start_state,
            goal_pose,
            self.plan_config,
        )
        # Free GPU input tensors — they are no longer needed after planning completes.
        del start_state, goal_pose, positions_tensor, goal_poses_tensor, joint_state

        if verbose:
            successes = result.success.cpu().numpy()
            print("=== Curobo Batch Plan Result ===")
            print(f"Batch size: {batch_size}")
            print(f"Successes: {np.sum(successes)}/{batch_size}")
            print("Status:", result.status)
            print("Position errors:", result.position_error)
            print("Rotation errors:", result.rotation_error)
            print("Solve time:", result.solve_time)
            print("==========================")

        # Return result without extracting trajectories to save GPU memory
        # Caller can call result.get_interpolated_plan() when needed
        return result

    def motion_plan(
        self,
        joint_position: list,
        goal_pose_lists: list[list],
        kinematics_config: dict = None,
        scene_objects: list[MlSpacesObject] = None,
        verbose: bool = False,
        pose_cost_metric: PoseCostMetric | None = None,
    ) -> list:
        """
        Args:
            joint_position: A list of joint positions representing the current state of the robot.
                - Only include the subset of joints that are relevant to the IK/planning task.
                - For example, if solving full-body IK, provide all joint positions (e.g., a list of length 20);
                  if solving single-arm IK, provide only the arm's joint positions (e.g., a list of length 7).
            goal_pose_lists: A list of goal poses to plan for. The first pose is for the main end-effector,
                and additional poses (if any) are assigned to the remaining link names in `self.link_names[1:]`.
            verbose: If True, prints detailed debug information about the planning result.

        Returns:
            trajectory: A list of joint position waypoints representing the planned trajectory.
            result: The full result object from the motion planner, including status and debug info.
        """
        # Update the kinematics config with the lock joints
        if kinematics_config is not None:
            print(
                "[WARNING] Rebuilding motion gen with new kinematics config. This will take a while."
            )
            self._build_robot_config(self.curobo_robot_config_path, kinematics_config)
            self._build_motion_gen()

        if scene_objects is not None:
            if verbose:
                print("Updating scene objects...")

            # exclude the attached objects from the world config
            excluded_scene_objects = []
            for scene_object in scene_objects:
                if scene_object.name not in self.attached_object2link_dict:
                    excluded_scene_objects.append(scene_object)
            self.update_world(excluded_scene_objects)
        # else: # Cache must be cleared explicitly
        #     if verbose:
        #         print("No scene objects provided. Using default world config.")
        #         print("Clearing world cache...")
        #         self.motion_gen.clear_world_cache()

        position = torch.tensor(joint_position).unsqueeze(0).float().to(self.tensor_args.device)
        joint_state = JointState.from_position(position, self.joint_names)
        start_state = JointState(
            position=self.tensor_args.to_device(joint_state.position),
            velocity=self.tensor_args.to_device(joint_state.velocity) * 0.0,
            acceleration=self.tensor_args.to_device(joint_state.acceleration) * 0.0,
            jerk=self.tensor_args.to_device(joint_state.jerk) * 0.0,
            joint_names=self.joint_names,
        )
        start_state = start_state.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

        # In bimanual case, left arm is the main arm.
        main_goal_pose = Pose.from_list(goal_pose_lists[0])
        additional_goal_poses = None
        if len(goal_pose_lists) > 1:
            additional_goal_poses = {}
            for link_name, goal_pose in zip(self.link_names[1:], goal_pose_lists[1:]):
                additional_goal_poses[link_name] = Pose.from_list(goal_pose)

        # Use custom plan config if pose_cost_metric is provided
        plan_config = self.plan_config
        if pose_cost_metric is not None:
            from dataclasses import replace

            plan_config = replace(self.plan_config, pose_cost_metric=pose_cost_metric)

        result = self.motion_gen.plan_single(
            start_state, main_goal_pose, plan_config, link_poses=additional_goal_poses
        )
        # Free GPU input tensors — they are no longer needed after planning completes.
        del start_state, main_goal_pose, position, joint_state
        if additional_goal_poses is not None:
            del additional_goal_poses
        succ = result.success.item()  # ik_result.success.item()

        trajectory = []
        if succ:
            traj = result.get_interpolated_plan()
            # traj = self.motion_gen.get_full_js(traj)
            for i in range(len(traj)):
                trajectory.append(traj[i].position.cpu().tolist())
            del traj
        else:
            trajectory.append(joint_position)
            if verbose:
                print("=== Curobo Plan Result ===")
                print(
                    "Success:",
                    result.success.item() if result.success is not None else result.success,
                )
                print("Status:", result.status)
                print("Valid query:", result.valid_query)
                print("Position error:", result.position_error)
                print("Rotation error:", result.rotation_error)
                print("Cspace error:", result.cspace_error)
                print("Solve time:", result.solve_time)
                print("Debug info:", result.debug_info)
                if isinstance(result.debug_info, dict):
                    for k, v in result.debug_info.items():
                        print(f"  {k}: {v}")
                print("==========================")
                print(self.joint_limits)
                print(joint_position)
                print(self.joint_names)
                print("==========================")
        return trajectory, result

    def _joint_state(self, joint_position: list) -> JointState:
        position = torch.tensor(joint_position).unsqueeze(0).float().to(self.tensor_args.device)
        state = JointState.from_position(position, self.joint_names)
        ordered = JointState(
            position=self.tensor_args.to_device(state.position),
            velocity=self.tensor_args.to_device(state.velocity) * 0.0,
            acceleration=self.tensor_args.to_device(state.acceleration) * 0.0,
            jerk=self.tensor_args.to_device(state.jerk) * 0.0,
            joint_names=self.joint_names,
        )
        return ordered.get_ordered_joint_state(self.motion_gen.kinematics.joint_names)

    def plan_to_joint_config(
        self,
        start_joint_position: list,
        goal_joint_position: list,
        scene_objects: list[MlSpacesObject] = None,
        verbose: bool = False,
    ) -> tuple[list, float, object]:
        """Plan a collision-free path to a JOINT configuration.

        A joint goal rather than a pose goal, because the caller already solved IK for the target
        and executes the rest of the task from that exact configuration. Planning to the pose
        instead would let the optimiser return a different IK branch -- same TCP, different elbow
        -- and everything after the transit would then differ from the unplanned baseline for a
        reason that has nothing to do with obstacle avoidance.

        The returned trajectory is JOINT space: cuRobo's ``MotionGenResult.interpolated_plan`` is a
        ``JointState``, and link poses appear only as goal inputs.

        Returns:
            (waypoints, motion_time_seconds, result). ``waypoints`` is empty when planning failed.
        """
        if scene_objects is not None:
            self.update_world(
                [
                    scene_object
                    for scene_object in scene_objects
                    if scene_object.name not in self.attached_object2link_dict
                ]
            )

        result = self.motion_gen.plan_single_js(
            self._joint_state(start_joint_position),
            self._joint_state(goal_joint_position),
            self.plan_config,
        )

        waypoints: list = []
        motion_time = 0.0
        if result.success is not None and bool(result.success.item()):
            plan = result.get_interpolated_plan()
            waypoints = [plan[index].position.cpu().tolist() for index in range(len(plan))]
            motion_time = float(
                result.motion_time.item()
                if hasattr(result.motion_time, "item")
                else result.motion_time
            )
            del plan
        elif verbose:
            log.info(
                f"[CUROBO] joint-space plan failed: status={result.status} "
                f"valid_query={result.valid_query}"
            )
        return waypoints, motion_time, result

    def set_world(self, cuboids: list) -> None:
        """Replace the planning world.

        Clears the cache first: ``update_world`` with an empty ``WorldConfig`` does NOT remove
        previously loaded obstacles. Measured -- a plan that succeeds in a fresh empty world fails
        identically after 17 obstacles were loaded and then "cleared" this way, which silently
        invalidated every empty-world control run through it.
        """
        self.motion_gen.clear_world_cache()
        new_world = WorldConfig(cuboid=list(cuboids))
        self.motion_gen.update_world(new_world)
        self.world_config = new_world

    def update_world_obstacle_poses(
        self,
        obstacle_names_to_poses: dict[str, np.ndarray],
        pre_invert_poses: bool = True,
    ) -> None:
        # in practice, it is much faster to pre-invert poses before sending them to curobo
        for obstacle_name, pose in obstacle_names_to_poses.items():
            if pre_invert_poses:
                inv_pose = Transform.from_list(pose).inv().to_list()
                self.motion_gen.world_coll_checker.update_obstacle_pose(
                    name=obstacle_name, obj_w_pose=Pose.from_list(inv_pose)
                )
            else:
                self.motion_gen.world_coll_checker.update_obstacle_pose(
                    name=obstacle_name, w_obj_pose=Pose.from_list(pose)
                )

    def enable_obstacles(
        self,
        obstacle_names: list[str],
        enable: bool,
    ) -> None:
        # Enable or disable obstacles in the collision checker
        for obstacle_name in obstacle_names:
            self.motion_gen.world_coll_checker.enable_obstacle(obstacle_name, enable)

    def update_world(
        self,
        scene_objects: list[MlSpacesObject],
    ) -> None:
        curobo_cuboids = []
        for object in scene_objects:
            if object.name == "scene/table":
                curobo_cuboids.append(
                    Cuboid(
                        name=object.name,
                        pose=pose_mat_to_7d(object.pose).tolist(),
                        dims=(np.array([0.3, 0.4, 0.02]) * 2).tolist(),
                    )
                )
            else:
                curobo_cuboids.append(
                    Cuboid(
                        name=object.name,
                        pose=pose_mat_to_7d(object.pose).tolist(),
                        dims=(object.aabb_size * 2).tolist(),
                    )
                )
        new_world_cfg = WorldConfig(cuboid=curobo_cuboids)
        self.motion_gen.update_world(new_world_cfg)
        self.world_config = new_world_cfg

    def attach_obj(
        self,
        object_names: list[str],
        joint_position: list,
        attach_link_names: list[str],
    ) -> None:
        for link_name in attach_link_names:
            if link_name not in self.attached_link_names:
                raise ValueError(f"Link name {link_name} not found in attached link names")

        # joint_position = np.clip(joint_position, self.joint_limits[0, :], self.joint_limits[1, :])
        joint_position = torch.tensor(joint_position).unsqueeze(0).float().to(self.tensor_args.device)
        joint_state = JointState.from_position(joint_position, self.joint_names)

        cu_js = JointState(
            position=self.tensor_args.to_device(joint_state.position),
            velocity=self.tensor_args.to_device(joint_state.velocity) * 0.0,
            acceleration=self.tensor_args.to_device(joint_state.acceleration) * 0.0,
            jerk=self.tensor_args.to_device(joint_state.jerk) * 0.0,
            joint_names=self.joint_names,
        )

        for object_name, link_name in zip(object_names, attach_link_names, strict=True):
            # Check what objects are in the world before attaching
            world_coll_checker = self.motion_gen.world_coll_checker
            if (
                hasattr(world_coll_checker, "world_model")
                and world_coll_checker.world_model is not None
            ):
                world_cfg = world_coll_checker.world_model
                world_objs = []
                if world_cfg.cuboid:
                    world_objs.extend([c.name for c in world_cfg.cuboid])
                if world_cfg.mesh:
                    world_objs.extend([m.name for m in world_cfg.mesh])
                log.debug(f"Objects in CuRobo world before attach: {world_objs}")
                if object_name not in world_objs:
                    log.warning(f"Object '{object_name}' not found in CuRobo world model!")

            log.debug(f"Calling attach_objects_to_robot for '{object_name}' to link '{link_name}'")
            self.motion_gen.attach_objects_to_robot(
                cu_js,
                [object_name],
                link_name=link_name,
                sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
                # world_objects_pose_offset should not be used - it incorrectly inserts the offset
                # in the middle of the transformation chain (ee_T_w * offset * w_T_obj)
                # instead of applying it to the object pose directly
                # remove_obstacles_from_world_config=True,
            )
            self.attached_object2link_dict[object_name] = link_name
            self.attached_link2object_dict[link_name] = object_name
            log.debug(f"Successfully attached '{object_name}' to '{link_name}'")

    def attach_external_obj(self, geometry: list, joint_position: list, link_name: str) -> None:
        """Attach geometry that is deliberately absent from the world model to a link.

        ``attach_obj`` resolves objects by name out of the world, which cannot work for a carried
        object: it is kept out of the world on purpose, because with it there the grasp pose is in
        collision by construction and every plan fails. cuRobo's own answer is
        ``attach_external_objects_to_robot``, which takes the geometry itself.

        ``geometry`` poses must be in the frame the planner works in -- the robot base frame -- as
        they are pushed through the end-effector pose computed from ``joint_position``, which must
        be the configuration the arm is in while holding the object.
        """
        if link_name not in self.attached_link_names:
            raise ValueError(f"Link name {link_name} not found in attached link names")
        if not geometry:
            raise ValueError("No geometry to attach")

        # The link's sphere budget is split evenly across the bodies handed over, so a body count
        # above the budget leaves zero spheres each and cuRobo declines the whole attach.
        attached = self.motion_gen.attach_external_objects_to_robot(
            self._joint_state(joint_position),
            list(geometry),
            link_name=link_name,
            sphere_fit_type=SphereFitType.VOXEL_VOLUME_SAMPLE_SURFACE,
        )
        if not attached:
            # A False here is the quiet failure mode that matters: the link keeps its empty sphere
            # set and every later plan is computed for a robot that is not carrying anything.
            raise RuntimeError(
                f"cuRobo declined to attach {len(geometry)} bodies to {link_name!r}; its sphere "
                f"budget does not divide across that many"
            )
        name = geometry[0].name
        self.attached_object2link_dict[name] = link_name
        self.attached_link2object_dict[link_name] = name
        log.debug(f"Attached {len(geometry)} external bodies to '{link_name}'")

    def link_spheres(self, link_name: str) -> list[list[float]]:
        """The spheres cuRobo holds for one link, ``[x, y, z, radius]`` in that link's frame.

        For an attached link this is the sphere fit of the payload, which is what the planner
        actually avoids collisions with -- not the payload's own geometry. Unused slots are parked
        with a negative radius and dropped here.
        """
        spheres = self.motion_gen.kinematics.kinematics_config.get_link_spheres(link_name)
        return [
            [round(float(v), 4) for v in sphere]
            for sphere in spheres.detach().cpu().numpy().tolist()
            if sphere[3] > 0.0
        ]

    def detach_obj(self, attach_link_names: list[str]) -> None:
        for link_name in attach_link_names:
            if link_name not in self.attached_link_names:
                raise ValueError(f"Link name {link_name} not found in attached link names")

        for link_name in attach_link_names:
            self.motion_gen.detach_object_from_robot(link_name)
            del self.attached_object2link_dict[self.attached_link2object_dict[link_name]]
            del self.attached_link2object_dict[link_name]

    def fk_solve(
        self,
        joint_config: list,
    ) -> np.ndarray:
        """Compute forward kinematics for a given joint configuration.

        Args:
            joint_config: Joint configuration as a list.

        Returns:
            np.ndarray: TCP pose in 7D format [x, y, z, qw, qx, qy, qz].
        """
        q_tensor = torch.tensor(joint_config).unsqueeze(0).float().to(self.tensor_args.device)
        kin_state = self.motion_gen.kinematics.get_state(q_tensor)
        # ee_pose is the end-effector pose
        ee_pose = kin_state.ee_pose
        position = ee_pose.position[0].cpu().numpy()
        quaternion = ee_pose.quaternion[0].cpu().numpy()  # [qw, qx, qy, qz]
        return np.concatenate([position, quaternion])

    def ik_solve(
        self,
        goal_pose: list,
        seed_config: list | None = None,
        return_seeds: int = 1,
        disable_collision: bool = False,
    ) -> tuple:
        """Solve inverse kinematics for a given goal pose.

        Args:
            goal_pose: Goal pose in 7D format [x, y, z, qw, qx, qy, qz].
            seed_config: Optional seed joint configuration. If provided, used as initial guess.
            return_seeds: Number of IK solutions to return.
            disable_collision: If True, ignore collision checking during IK solve.

        Returns:
            tuple: (joint_config, success) where joint_config is a list of joint positions
                   if successful, None otherwise.
        """
        goal_pose_curobo = Pose.from_list(goal_pose)

        # Prepare seed config if provided
        # Shape must be (batch, num_seeds, dof) for curobo IK solver
        seed_config_tensor = None
        if seed_config is not None:
            # unsqueeze twice: once for batch dim, once for num_seeds dim
            seed_config_tensor = torch.tensor(seed_config).unsqueeze(0).unsqueeze(0).float().to(self.tensor_args.device)

        # Disable collision checking if requested
        if disable_collision:
            for rollout in self.motion_gen.ik_solver.get_all_rollout_instances():
                rollout.primitive_collision_constraint.disable_cost()
                rollout.robot_self_collision_constraint.disable_cost()

        try:
            # Solve IK using motion_gen's ik solver
            ik_result = self.motion_gen.solve_ik(
                goal_pose=goal_pose_curobo,
                seed_config=seed_config_tensor,
                return_seeds=return_seeds,
            )
        finally:
            # Re-enable collision checking
            if disable_collision:
                for rollout in self.motion_gen.ik_solver.get_all_rollout_instances():
                    rollout.primitive_collision_constraint.enable_cost()
                    rollout.robot_self_collision_constraint.enable_cost()

        success = ik_result.success.item()
        if success:
            # Extract the solution
            joint_config = ik_result.solution[0].squeeze().cpu().tolist()
            del ik_result
            return joint_config, None
        else:
            log.debug(
                f"IK solve failed: pos_err={ik_result.position_error.item():.4f}, "
                f"rot_err={ik_result.rotation_error.item():.4f}"
            )
            del ik_result
            return None, None
