"""Policy configuration classes for MolmoSpaces experiments."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from pathlib import Path

from molmo_spaces.configs.abstract_config import Config
from molmo_spaces.molmo_spaces_constants import ASSETS_DIR
from molmo_spaces.planner.astar_planner import AStarPlannerConfig
from molmo_spaces.policy.base_policy import BasePolicy, PolicyFactory
from molmo_spaces.utils.function_utils import make_lenient

# Import CuroboPlannerConfig if available (requires GPU), otherwise create a stub
try:
    from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
except (ImportError, RuntimeError):
    # Create a stub class when CuRobo isn't available (e.g., on non-GPU nodes)
    # This allows Pydantic to resolve forward references during config validation
    if TYPE_CHECKING:
        from molmo_spaces.planner.curobo_planner import CuroboPlannerConfig
    else:

        class CuroboPlannerConfig(Config):  # type: ignore
            """Stub for CuroboPlannerConfig when CuRobo is not available."""

            pass


class BasePolicyConfig(Config):
    """Base configuration for policies."""

    policy_cls: type[BasePolicy]
    policy_factory: PolicyFactory
    """
    Factory function to create the policy instance from a config and task, can be same as ``policy_cls``.
    """
    policy_type: str  # Type of the policy, e.g., "planner", "teleop", "learned", etc.
    force_enable_depth: bool = False
    """
    If true, require all cameras to record depth.
    In eval the cameras will be overridden, otherwise it will just require the camera system config to enable depth.
    """


class ObjectManipulationPlannerPolicyConfig(BasePolicyConfig):
    """Configuration for Franka pick planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_factory: PolicyFactory | None = None
    policy_type: str = "planner"

    # Pick-and-place pose offsets
    pregrasp_z_offset: float = 0.04  # Height above object for pregrasp
    postgrasp_z_offset: float = 0.05  # Height above object for postgrasp
    grasp_z_offset: float = 0.03  # Lower distance from pregrasp to grasp
    place_z_offset: float = 0.07  # Lower distance from preplace to place
    end_z_offset: float = 0.05  # Height above place target for final pose

    # Speed settings
    speed_slow: float = 0.08  # m/s for precise movements
    speed_fast: float = 0.20  # m/s for transport movements
    move_settle_time: float = 0.1  # seconds

    # Gripper timing
    gripper_close_duration: float = 0.5  # Time to close gripper
    gripper_open_duration: float = 0.25  # Time to open gripper

    # Randomization parameters
    randomize_grasp: bool = False  # Enable grasp pose randomization
    grasp_xy_noise: float = 0.02  # Max XY offset from object center (meters)
    grasp_yaw_noise: float = 0.5  # Max rotation around Z-axis (radians)
    pregrasp_height_noise: float = 0.03  # Additional height variation for pregrasp
    postgrasp_height_noise: float = 0.02  # Height variation for lift phase

    # Retry behavior parameters
    max_retries: int = 3  # Maximum number of retry attempts
    gripper_empty_threshold: float = 0.002  # Gripper separation to detect empty gripper (meters)
    phase_timeout: float = 10.0  # Maximum time to spend in any phase (seconds)
    max_sequential_ik_failures: int = 8  # Maximum number of IK failures
    tcp_pos_err_threshold: float = 0.1  # Retry if position error is greater than this
    tcp_rot_err_threshold: float = np.radians(30.0)  # Retry if rotation error is greater than this
    # On retry, prefer a grasp whose contact point is at least this far (meters, in the object
    # frame) from any grasp already attempted this episode; fall back to any feasible grasp if
    # no sufficiently-different site is reachable. 0 disables (always re-pick the best-ranked).
    retry_grasp_exclusion_radius: float = 0.04

    # grasp sampling configuration (collision checking)
    filter_colliding_grasps: bool = True
    grasp_collision_batch_size: int = 128
    grasp_collision_max_grasps: int = 512
    grasp_width: float = 0.08
    grasp_length: float = 0.05
    grasp_height: float = 0.01
    grasp_base_pos: list[float] = [0.0, 0.0, -0.04]  # position of grasp base in tcp frame
    # grasp sampling configuration (cost weighting)
    grasp_pos_cost_weight: float = 1.0
    grasp_rot_cost_weight: float = 0.01
    grasp_vertical_cost_weight: float = 2.0
    grasp_com_dist_cost_weight: float = 8.0
    # Grasp-orientation sweep: each library grasp is also tried rotated about its local y
    # (closing) axis and local x axis by these angles, so the grasp gripper angle is swept
    # (symmetric with the placement_local_* sweep) and the reachable one is chosen.
    grasp_closing_axis_rotation_degrees: tuple[float, ...] = (0.0,)
    grasp_local_x_rotation_degrees: tuple[float, ...] = (0.0,)
    placement_local_x_rotation_degrees: tuple[float, ...] = (0.0,)
    placement_local_y_rotation_degrees: tuple[float, ...] = (0.0,)
    placement_fallback_position_tolerance: float = 0.005
    placement_fallback_rotation_tolerance: float = np.radians(1.0)
    # grasp sampling configuration (feasibility checking)
    filter_feasible_grasps: bool = True
    grasp_feasibility_batch_size: int = 256
    grasp_feasibility_max_grasps: int = 256

    # Best-effort grasp orientation: when no candidate is strictly IK-feasible (e.g. the arm
    # cannot reach the library's exact approach direction), fall back to the lowest-cost grasp
    # whose POSITION is reachable with the closest achievable orientation (within the tolerance
    # below), and execute it with a relaxed rotation-tracking threshold. Mirrors best-effort-lift.
    allow_best_effort_grasp_orientation: bool = False
    best_effort_grasp_max_rot_error: float = np.radians(45.0)
    best_effort_grasp_max_pos_error: float = 0.01
    best_effort_grasp_max_candidates: int = 60

    # Extra-lift tuning for the pick "lift until success" behaviour (baked into PickPlannerPolicy):
    # once the planned lift has run in full, if the pick isn't a clean success yet the gripper keeps
    # rising by this step (up to this cap beyond the planned lift pose) until the pick succeeds or
    # the task horizon times out.
    continue_lift_step: float = 0.002        # metres to raise the TCP per control step
    continue_lift_max_extra: float = 0.15    # extra raise allowance above the planned lift; give up after

    # which grasp libraries to use, in descending priority (will be filtered by availability for each asset)
    # if None, all available libraries for the object will be used
    grasp_libraries: list[str] | None = None

    # Task barrier: an obstacle standing between the object and its destination. A test fixture
    # for planning, off by default; both the planned and unplanned policies build it from the same
    # code so an A/B sees identical geometry.
    barrier_enabled: bool = False
    barrier_thickness: float = 0.01     # metres, across the object -> destination line
    barrier_width: float = 0.30         # metres, along the wall
    barrier_height_fraction: float = 0.5  # of the picked object's own height
    barrier_rgba: tuple[float, float, float, float] = (0.85, 0.10, 0.10, 1.0)

    # Debugging
    debug_poses: bool = False  # Enable debug printing for poses
    verbose: bool = True  # Enable verbose output for debugging


class OpenClosePlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    # For opening tasks: horizontal orientation is strongly preferred over position
    # grasp_horizontal_cost_weight is multiplied by 10x for opening tasks to strongly penalize vertical orientations
    # The cost uses squared term: (abs(z-axis z-component))^2, so vertical orientations get heavily penalized
    grasp_pos_cost_weight: float = 1.0
    grasp_rot_cost_weight: float = 0.05
    grasp_vertical_cost_weight: float = 0.0
    grasp_horizontal_cost_weight: float = (
        10.0  # Base weight, multiplied by 10x for opening tasks (effective: 20.0)
    )
    grasp_com_dist_cost_weight: float = 0.0
    pregrasp_z_offset: float = 0.04  # Height above object for postgrasp

    # Speed settings
    speed_slow: float = 0.04  # m/s for precise movements
    speed_fast: float = 0.08  # m/s for transport movements
    move_settle_time: float = 0.2  # seconds

    grasp_libraries: list[str] | None = ["droid"]  # only thor provides articulated grasps

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.open_close_planner_policy import (
                OpenClosePlannerPolicy,
            )

            self.policy_cls = OpenClosePlannerPolicy
            self.policy_factory = OpenClosePlannerPolicy


class PickPlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    postgrasp_z_offset: float = 0.08  # Height above object for postgrasp

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_planner_policy import (
                PickPlannerPolicy,
            )

            self.policy_cls = PickPlannerPolicy
            self.policy_factory = PickPlannerPolicy


class PickAndPlacePlannerPolicyConfig(ObjectManipulationPlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    move_settle_time: float = 0.5

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_planner_policy import (
                PickAndPlacePlannerPolicy,
            )

            self.policy_cls = PickAndPlacePlannerPolicy
            self.policy_factory = PickAndPlacePlannerPolicy


class CuroboOpenClosePlannerPolicyConfig(OpenClosePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    left_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    right_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    enable_collision_avoidance: bool = True
    batch_size: int = 4
    max_grasping_timesteps: int = 5
    max_opening_timesteps: int = 5
    max_steps_per_waypoint: int = 10
    max_batch_plan_attempts: int = 4
    pregrasp_z_offset: float = 0.02
    max_planning_reattempts: int = 2
    gripper_closed_pos: float = 0.0
    gripper_closed_tolerance: float = 0.005
    velocity_constraints: dict[str, float] = {
        "base": 0.5,
        "head": 0.5,
        "right_arm": 0.5,
        "left_arm": 0.5,
    }
    grasp_vertical_cost_weight: float = 2.0
    attach_obj: bool = False
    max_settle_steps: int = 5
    max_height_adjustment_steps: int = 10
    server_timeout: float | None = (
        300.0  # gRPC deadline for motion planning calls (seconds), None = no deadline
    )
    server_urls: list[str] = [
        "jupiter-cs-aus-107.reviz.ai2.in:10002",
    ]


class CuroboPickAndPlacePlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports
    left_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    right_curobo_planner_config: CuroboPlannerConfig | None = None  # will be set in model_post_init
    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    enable_collision_avoidance: bool = True
    batch_size: int = 4
    max_grasping_timesteps: int = 5
    max_opening_timesteps: int = 5
    max_steps_per_waypoint: int = 10
    max_batch_plan_attempts: int = 4
    pregrasp_z_offset: float = 0.02  # [m]
    max_planning_reattempts: int = 5
    gripper_closed_pos: float = 0.0  # [m]
    gripper_closed_tolerance: float = 0.005  # [m]
    velocity_constraints: dict[str, float] = {
        "base": 0.5,  # [m / policy_dt_ms]
        "head": 0.5,  # [rad / policy_dt_ms]
        "right_arm": 0.5,  # [rad / policy_dt_ms]
        "left_arm": 0.5,  # [rad / policy_dt_ms]
    }
    grasp_vertical_cost_weight: float = 0.5
    attach_obj: bool = False
    max_settle_steps: int = 5
    server_timeout: float | None = (
        300.0  # gRPC deadline for motion planning calls (seconds), None = no deadline
    )
    server_urls: list[str] = [
        "jupiter-cs-aus-107.reviz.ai2.in:10002",
    ]


class CuroboGenericPickAndPlacePolicyConfig(PickAndPlacePlannerPolicyConfig):
    """The baseline pick-and-place policy with its TRANSIT planned by cuRobo.

    Every inherited field keeps its baseline value on purpose: this config differs from
    ``PickAndPlacePlannerPolicyConfig`` only by the policy class and the planner's own knobs, so an
    A/B against it isolates the planner.
    """

    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    curobo_output_dir: Path = ASSETS_DIR / "curobo_generated"
    """Where the derived URDF is cached. Keyed by embodiment id and version."""

    # Robot model
    collision_sphere_buffer: float = 0.005
    max_spheres_per_geom: int = 12

    # Motion generation
    interpolation_dt: float = 0.02
    collision_activation_distance: float = 0.01
    num_ik_seeds: int = 64
    num_trajopt_seeds: int = 8
    trajopt_tsteps: int = 32
    max_attempts: int = 8

    # Execution. Arrival-gated, following the shipped RB-Y1 cuRobo policy: a waypoint is held
    # until the joints reach it, so none of these is a duration and none of them can make the arm
    # outrun its controller.
    waypoint_tolerance: float = 0.0275
    """Joint error, in radians, at which a waypoint counts as reached. RB-Y1's value
    (``CuroboPlannerPolicy._is_waypoint_reached``); waypoints denser than this are skipped."""
    max_steps_per_waypoint: int = 10
    """Control steps to spend on one waypoint before handing back to the policy's retry."""
    max_joint_velocity: float = 1.0
    """Per-joint speed cap, rad/s. Upstream expresses the same limit per named move group
    (``velocity_constraints``, 0.5 rad/s); one scalar avoids naming groups that vary per robot.
    The default sits above the shipped YAM's own peak (0.61 rad/s) so it bounds rather than
    throttles."""
    transit_settle_steps: int = 2
    """Steps to hold the final waypoint so the arm has converged before the grasp approach."""

    planned_segments: tuple[str, ...] = ("pregrasp", "preplace")
    """Which phases cuRobo plans: the free-space transits — the approach to the object and the
    transport to the destination. ``grasp``, ``lift`` and ``place`` keep their straight lines
    because their goals are deliberately in contact, so a planner either refuses them or curves a
    descent that has to be vertical."""
    grad_trajopt_iters: int | None = None
    """Trajectory-optimisation iterations per attempt; ``None`` keeps cuRobo's default (100).

    Measured on the barrier transport: at the default the optimiser stops 8 samples short of a
    collision-free path and all 7 attempts fail, while 10x solves it on the FIRST attempt. Extra
    seeds and extra attempts do not help -- every seed hits the same ceiling -- so this is the knob
    that matters, and it costs ~3x rather than 10x because the default was spending its budget on
    doomed retries."""
    finetune_trajopt_iters: int | None = None

    enable_graph_attempt: int | None = 2
    """Attempt at which cuRobo falls back to graph search. Local trajectory optimisation cannot
    change a path's homotopy, so it cannot route over or around an obstacle standing between the
    start and the goal -- it just fails. ``None`` reproduces the RB-Y1 setting (no global search)."""

    planner_seed: int | None = None
    """Seed for cuRobo's own solver randomness, independent of the experiment seed.

    Trajopt draws its seeds from the torch RNG, which the task sampler seeds per work item -- so
    without this the planner's randomness is welded to the scene's, and a solver failure cannot be
    retried on a different draw without also changing the scene. ``None`` keeps the previous
    behaviour (inherit whatever torch state the episode left)."""

    max_obstacle_boxes: int = 512
    """Cap on collision boxes handed to the planner. Must stay under the planner's own
    ``collision_cache`` obb budget; anything past it is logged, never dropped quietly.

    Was 60, which was ample while a broken import meant the obstacle set held nothing but the
    support surface and the destination. With the surface's clutter restored, one kitchen counter
    measures 426 collision geoms -- convex decompositions run to a dozen or more meshes per object
    -- so 60 kept 14% of the world and silently chose which 14% by iteration order."""
    attach_carried_object: bool = False
    """Attach the grasped object to the planner's robot while planning transport. Off by default
    because with nothing to avoid it only costs planning time; it matters as soon as the transport
    has to route around something, since an unattached plan plots the wrist clear of an obstacle
    and drags the carried object straight through it."""

    def model_post_init(self, __context) -> None:
        # Claim policy_cls BEFORE the parent runs: its own model_post_init fills an unset
        # policy_cls with the baseline policy, and would win by going first.
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.object_manipulation.curobo_generic_planner_policy import (
                CuroboGenericPickAndPlacePlannerPolicy,
            )

            self.policy_cls = CuroboGenericPickAndPlacePlannerPolicy
            self.policy_factory = CuroboGenericPickAndPlacePlannerPolicy
        super().model_post_init(__context)


class PickAndPlaceNextToPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_next_to_planner_policy import (
            PickAndPlaceNextToPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceNextToPlannerPolicy
        self.policy_factory = PickAndPlaceNextToPlannerPolicy


class PickAndPlaceColorPlannerPolicyConfig(PickAndPlacePlannerPolicyConfig):
    policy_cls: type = None  # Will be set in model_post_init to avoid circular imports

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        from molmo_spaces.policy.solvers.object_manipulation.pick_and_place_color_planner_policy import (
            PickAndPlaceColorPlannerPolicy,
        )

        self.policy_cls = PickAndPlaceColorPlannerPolicy
        self.policy_factory = PickAndPlaceColorPlannerPolicy


class DoorOpeningPolicyConfig(BasePolicyConfig):
    """Configuration for RBY1 door opening planner policy."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_factory: PolicyFactory | None = None
    policy_type: str = "planner"

    # RBY1-specific policy parameters
    # Motion planning parameters
    left_curobo_planner_config: CuroboPlannerConfig | None = (
        None  # will be set in __init_policy_config
    )
    right_curobo_planner_config: CuroboPlannerConfig | None = (
        None  # will be set in __init_policy_config
    )

    left_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "left_arm": (3, 10),
    }
    right_planner_joint_ranges: dict[
        str, tuple
    ] = {  # Joint ranges for motion planning. Should match curobo config.
        # Move group : Joint indices in curobo config
        "base": (0, 3),
        "right_arm": (3, 10),
    }
    velocity_constraints: dict[str, float] = {
        "base": 0.5,
        "head": 0.5,
        "right_arm": 0.5,
        "left_arm": 0.5,
    }
    enable_collision_avoidance: bool = True  # Whether to enable collision avoidance
    relevant_collision_objects_radius: float = (
        3.0  # Radius in meters from the door handle around which collision objects are considered
    )
    plan_in_robot_frame: bool = (
        True  # Whether to plan in robot frame or world frame (True keeps base stable)
    )
    max_planning_failures: int = 15

    # Trajectory execution parameters
    max_steps_per_waypoint: int = 10
    joint_position_tolerance: float = 0.0275

    # Gripper control parameters
    gripper_closed_pos: float = 0.0
    left_gripper_close_command: dict = {"left_gripper": 100.0}
    left_gripper_open_command: dict = {"left_gripper": -100.0}
    right_gripper_close_command: dict = {"right_gripper": 100.0}
    right_gripper_open_command: dict = {"right_gripper": -100.0}
    gripper_closed_tolerance: float = 0.005  # [m]
    max_grasping_timesteps: int = 5

    # Door opening parameters
    pre_grasp_distance: float = -0.18  # distance from door handle before grasping it
    articulation_deltas: list[float] = [
        (np.pi / 180.0) * 13.0
    ]  # delta radians to articulate door joint(s)
    first_pushing_articulation_deltas: list[float] = [
        (np.pi / 180.0) * 30.0
    ]  # special first delta articulation when pushing door

    # Recovery motion parameters
    recovery_motion_backward_distance: float = 0.02
    num_recovery_steps: int = 8

    # Debugging
    verbose: bool = False  # Enable verbose output for debugging


class NavToObjPlannerPolicyConfig(BasePolicyConfig):
    """Base configuration for navigation to object planner policies."""

    policy_cls: type = None  # Will be set by importing module to avoid circular imports
    policy_factory: PolicyFactory | None = None
    policy_type: str = "planner"

    # Recovery motion parameters
    recovery_motion_backward_distance: float = 0.02
    num_recovery_steps: int = 8

    # Debugging
    verbose: bool = True  # Enable verbose output for debugging


class AStarNavToObjPolicyConfig(NavToObjPlannerPolicyConfig):
    """Configuration for A* navigation policy (discrete grid-based planner)."""

    policy_cls: type = None

    # A* planner configuration
    planner_config: AStarPlannerConfig = AStarPlannerConfig()

    # A* planner parameters (for backward compatibility)
    map_path: str | None = None  # Path to occupancy map
    downscale: int = 5  # Downscaling factor for grid

    # Policy-related parameters
    path_interpolation_density: int = (
        1  # Num points to add between planner waypoint pairs (regardless of distance)
    )
    path_max_inter_waypoint_dist: float = 0.25  # Max distance between consecutive waypoints
    path_max_inter_waypoint_angle: float = float(
        np.deg2rad(10)
    )  # Max arc length between consecutive waypoints
    path_min_dist_to_target_center: float = (
        0.8  # Skip approaching target center below this distance
    )
    plan_max_retries: int = 3  # Allowed number of planning retries in episode

    # TODO the replanning criterion is weak, as it does not rely on actual collision,
    #  but a loose estimate based on rate decrease of spatial-angular distance to next waypoint.
    #  It needs further work to be usable, so you may want to keep a large value to prevent it for now.
    plan_fail_after_waypoint_steps: int = (
        10  # Number of steps within current waypoint to check for need to replan
    )

    plan_fail_max_dist_delta: float = 0.01  # Max difference between dists to waypoint to consider need to replan after plan_fail_after_waypoint_steps
    plan_stick_to_original_target: bool = (
        False  # Allows replanning to other possible valid targets when False
    )

    def model_post_init(self, __context) -> None:
        """Set policy_cls after initialization to avoid circular imports."""
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.solvers.navigation.astar_planner_policy import (
                AStarSmoothPlannerPolicy,
            )

            self.policy_cls = AStarSmoothPlannerPolicy
            self.policy_factory = AStarSmoothPlannerPolicy


class DummyPolicyConfig(BasePolicyConfig):
    """Policy config that uses DummyPolicy for testing."""

    policy_type: str = "dummy"
    policy_cls: type = None  # Set in model_post_init
    policy_factory: PolicyFactory | None = None

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import DummyPolicy

            self.policy_cls = DummyPolicy
            self.policy_factory = make_lenient(DummyPolicy)


class BrownianMotionPolicyConfig(BasePolicyConfig):
    """Policy that applies Gaussian noise increments over noop control, resulting in Brownian motion."""

    policy_cls: type = None
    policy_factory: PolicyFactory | None = None
    policy_type: str = "dummy"
    std: float = 0.1

    def model_post_init(self, __context) -> None:
        super().model_post_init(__context)
        if self.policy_cls is None:
            from molmo_spaces.policy.dummy_policy import BrownianMotionPolicy

            self.policy_cls = BrownianMotionPolicy
            self.policy_factory = make_lenient(BrownianMotionPolicy)
