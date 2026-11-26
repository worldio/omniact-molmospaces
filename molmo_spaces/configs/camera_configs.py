"""Camera configuration classes for MolmoSpaces experiments."""

import logging
from abc import ABC
from typing import ClassVar, Literal, TypeAlias, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

from molmo_spaces.configs.abstract_config import Config

T = TypeVar("T")
Triple: TypeAlias = tuple[T, T, T]


class CameraConfig(Config, ABC):
    """Base specification for a single camera.

    Each camera spec defines how one camera should be created and configured.
    Subclasses implement different camera types (MJCF, robot-mounted, exocentric).
    """

    name: str  # Unique identifier for this camera in the registry
    fov: float | None = None  # Field of view in degrees
    is_warped: bool = False  # Whether camera has lens distortion (e.g., GoPro fisheye)
    record_depth: bool = False  # Whether to record depth images for this camera
    skip_erosion: bool = (
        False  # Skip erosion for object point sampling (useful for wide FOV cameras)
    )

    # Visibility constraints for robot placement validation (optional)
    # Maps body names to minimum visibility thresholds (0.0 to 1.0)
    # Can use special keys like "__gripper__" or "__task_objects__" (resolved at placement time)
    # If specified, these constraints will be checked during robot placement when enabled
    visibility_constraints: dict[str, float] | None = None


class MjcfCameraConfig(CameraConfig):
    """Camera defined in the MJCF file.

    This references a camera that already exists in the scene MJCF or robot MJCF.
    Useful for cameras with fixed mounting in robot models.
    """

    mjcf_name: str  # Full name of camera in MJCF (may include namespace)
    robot_namespace: str | None = None  # If specified, prepends to mjcf_name (e.g., "robot_0/")

    # Further MJCF names to try if mjcf_name is absent, in order -- the same "uses the first that
    # exists" idiom RobotMountedCameraConfig.reference_body_names uses, for the same reason. Robots
    # disagree on what to call the same physical camera (franka_droid ships gripper/wrist_camera, RUM
    # ships wrist_cam, embodiment_gen emits wrist_camera), and the name is baked into downloaded,
    # versioned MJCFs that cannot be renamed. Listing the aliases lets ONE camera system serve every
    # robot and still record the stream under one stable key.
    mjcf_name_alternatives: list[str] = []

    # Optional noise for MJCF cameras (applied to their fixed mounting)
    pos_noise_range: tuple[float, float] | tuple[Triple[float], Triple[float]] | None = (
        None  # Add noise to camera position (min, max)
    )
    orientation_noise_degrees: float | Triple[float] | None = (
        None  # Random rotation noise in degrees
    )
    fov_noise_degrees: tuple[float, float] | None = None  # Add noise to FOV (min, max)

    def candidate_mjcf_names(self) -> list[str]:
        """Every MJCF camera name this config will accept, namespaced, most preferred first."""
        prefix = self.robot_namespace or ""
        return [f"{prefix}{name}" for name in (self.mjcf_name, *self.mjcf_name_alternatives)]


class RobotMountedCameraConfig(CameraConfig):
    """Camera dynamically mounted to a robot body.

    Camera follows the specified reference body with configurable offset and orientation.
    Can use either lookat-based positioning or quaternion-based orientation.
    """

    reference_body_names: list[str]  # Body names to try (uses first that exists)
    camera_offset: list[float] = [0.10, 0.0, -0.15]  # Position relative to reference body

    # Orientation method 1: Look-at based (simpler, more intuitive)
    lookat_offset: list[float] = [0.0, 0.0, 0.08]  # Point to look at (relative to reference body)
    up_axis: str = "z"  # Which local axis is "up" ("x", "y", or "z")

    # Orientation method 2: Quaternion-based (more precise control)
    camera_quaternion: list[float] | None = None  # [w, x, y, z] relative to reference body

    # Optional randomization at camera setup time
    pos_noise_range: tuple[float, float] | None = None  # Add noise to camera_offset (min, max)
    lookat_noise_range: tuple[float, float] | None = None  # Add noise to lookat_offset (min, max)
    orientation_noise_degrees: float | None = (
        None  # Random rotation noise in degrees (for quaternion-based)
    )


class FixedExocentricCameraConfig(CameraConfig):
    """Fixed external camera at a specific world position.

    Useful for consistent third-person views, overhead cameras, or monitoring positions.
    Can optionally add small amounts of noise for data augmentation.
    # TODO: should this also have a quaternion option? was figuring this would be most useful for fixed eval episodes
    """

    pos: list[float]  # World position [x, y, z]
    forward: list[float]  # Forward direction vector
    up: list[float]  # Up direction vector

    # Optional noise for data augmentation
    pos_noise_range: tuple[float, float] | tuple[Triple[float], Triple[float]] | None = (
        None  # Add noise to position (min, max)
    )
    orientation_noise_degrees: float | Triple[float] | None = (
        None  # Random rotation noise in degrees
    )


class RandomizedExocentricCameraConfig(CameraConfig):
    """Randomized external camera positioned around a workspace center.

    Samples camera position within specified ranges around a workspace center.
    Can use visibility constraints to ensure good views of important objects.
    CORE ASSUMPTION: workspace center will be sourced from task sampler callback function get_workspace_center
    you will always be looking at the workspace center (with optional noise).
    """

    # Sampling ranges (spherical coordinates around workspace center)
    distance_range: tuple[float, float]  # (min, max) distance from workspace center
    height_range: tuple[float, float]  # (min, max) height above workspace
    azimuth_range: tuple[float, float]  # (min, max) azimuth angle in radians
    fov_range: tuple[float, float] | None = None  # (min, max) FOV range in degrees

    # # Lookat configuration
    # NOTE CHANGE: always look at workspace center (workspace center can just be a 3d point or a body position). simplifies logic.
    lookat_noise_range: tuple[float, float] | None = None  # Add noise to lookat point

    # Visibility constraints for camera placement (optional)
    # Maps body names to minimum visibility thresholds (0.0 to 1.0)
    # Can use special keys like "__gripper__" or "__task_objects__" (resolved at setup time)
    # Note: ALL constraints must be met (no "at least one of" logic currently supported)
    # TODO: Add support for "at least one of" groups in visibility constraints
    # task sampler should implement resolve_visibility_object to provide body names for special keys if you add any
    visibility_constraints: dict[str, float] | None = None
    max_placement_attempts: int = 100  # Max attempts to satisfy visibility constraints
    allow_relaxed_constraints: bool = False  # Use best attempt if constraints not met


class EvalRobotMountedCameraConfig(RobotMountedCameraConfig):
    """Robot-mounted camera with additional intrinsics perturbation for eval."""

    fov_noise_degrees: tuple[float, float] | None = None


class EvalExocentricCameraConfig(FixedExocentricCameraConfig):
    """Eval exocentric camera whose level-0 pose is derived from a shoulder mount.

    At runtime the reference body pose is resolved into world-frame
    pos/forward/up, decomposed into spherical coordinates relative to the
    workspace center, perturbed according to the spherical noise params, and
    then placed via the normal fixed-exocentric path.

    ``pos``, ``forward``, ``up`` default to ``None`` here (overriding the
    required parent fields) because they are computed at runtime.
    """

    # Override parent required fields with None defaults (resolved at runtime)
    pos: list[float] | None = None
    forward: list[float] | None = None
    up: list[float] | None = None

    # Spherical perturbation ranges (around the reference shoulder-mount pose)
    azimuth_range: tuple[float, float] | None = None  # radians, symmetric around ref azimuth
    distance_range: tuple[float, float] | None = None  # meters, offset from ref distance
    height_range: tuple[float, float] | None = None  # meters, offset from ref height
    workspace_center_weight: float | None = (
        None  # [0,1] blend from calibrated target → workspace center
    )
    lookat_noise_range: tuple[float, float] | None = None  # meters, per-axis noise on lookat target
    fov_range: tuple[float, float] | None = None  # degrees, FOV sampling range

    max_placement_attempts: int = 50

    # Reference shoulder-mounted pose (used to compute level-0 pos/forward/up)
    reference_body_names: list[str] = ["robot_0/fr3_link0"]
    camera_offset: list[float] = [0.1, 0.57, 0.66]
    camera_quaternion: list[float] = [-0.3633, -0.1241, 0.4263, 0.8191]


AllCameraTypes: TypeAlias = (
    MjcfCameraConfig
    | RobotMountedCameraConfig
    | FixedExocentricCameraConfig
    | RandomizedExocentricCameraConfig
    | EvalRobotMountedCameraConfig
    | EvalExocentricCameraConfig
)


class CameraSystemConfig(Config):
    """Complete camera system configuration.

    Defines all cameras that should be set up in the environment,
    along with shared settings like resolution.
    """

    # Shared settings for all cameras
    img_resolution: tuple[int, int] = (640, 480)  # (width, height)

    # Individual camera specifications
    cameras: list[AllCameraTypes] = []

    # Where the per-episode placement noise is drawn from.
    #   None            -- the process-global np.random stream, which seed_task_sampling seeds. This
    #                      is reproducible for an identical run, but camera placement then depends on
    #                      every other draw made before it, so changing how many retries the object
    #                      sampler happens to make moves the cameras. Default, so no shipped system's
    #                      data changes.
    #   int             -- that seed, mixed with the episode seed so episodes still differ.
    #   "embodiment"    -- seeded from the robot's own identity, so a given robot's camera sequence is
    #                      a property of that robot and reproduces wherever it is run.
    # Resolved in BaseTaskSampler.setup_cameras.
    camera_seed: int | Literal["embodiment"] | None = None

    # Gate for ObjectImagePointsSensor (a per-step per-camera segmentation render). Default True; a
    # system sets it False when object_image_points is not part of its dataset.
    record_object_image_points: bool = True

    def add_camera(self, camera_spec: CameraConfig) -> None:
        """Add a camera specification to the system."""
        self.cameras.append(camera_spec)

    def get_camera_by_name(self, name: str) -> CameraConfig | None:
        """Get a camera spec by name."""
        for camera in self.cameras:
            if camera.name == name:
                return camera
        return None


class RBY1MjcfCameraSystem(CameraSystemConfig):
    """Camera system using RBY1's built-in MJCF cameras."""

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        MjcfCameraConfig(
            name="head_camera",
            mjcf_name="head_camera",
            robot_namespace="robot_0/",
            fov=139.0,
            skip_erosion=True,
        ),
        MjcfCameraConfig(
            name="wrist_camera_l",
            mjcf_name="wrist_camera_l",
            robot_namespace="robot_0/",
            record_depth=True,
        ),
        MjcfCameraConfig(
            name="wrist_camera_r",
            mjcf_name="wrist_camera_r",
            robot_namespace="robot_0/",
            record_depth=True,
        ),
        MjcfCameraConfig(
            name="camera_follower",
            mjcf_name="camera_follower",
            robot_namespace="robot_0/",
        ),
        # MjcfCameraConfig(
        #     name="camera_thirdview_follower_1",
        #     mjcf_name="camera_thirdview_follower_1",
        #     robot_namespace="robot_0/",
        # ),
        # MjcfCameraConfig(
        #     name="camera_thirdview_follower_2",
        #     mjcf_name="camera_thirdview_follower_2",
        #     robot_namespace="robot_0/",
        # ),
    ]


class RBY1GoProD455CameraSystem(CameraSystemConfig):
    """Camera system for RBY1 with GoPro head camera and D455 wrist cameras.

    Renders at 1024x576 (16:9) to accommodate both:
    - Head camera: GoPro analogue (4:3, crop to 768x576 in post-processing)
    - Wrist cameras: D455 analogue (16:9, use full frame)

    All cameras include randomization for sim-to-real transfer.
    """

    img_resolution: tuple[int, int] = (1024, 576)
    cameras: list[AllCameraTypes] = [
        # Head camera - GoPro analogue (4:3, VFOV ~94deg for wide mode)
        # Crop to 768x576 in post-processing to get 4:3 aspect ratio
        MjcfCameraConfig(
            name="head_camera",
            mjcf_name="head_camera",
            robot_namespace="robot_0/",
            fov=139.0,  # GoPro wide mode VFOV
            fov_noise_degrees=(-3.0, 3.0),
            pos_noise_range=((-0.01, -0.01, -0.01), (0.01, 0.01, 0.01)),
            orientation_noise_degrees=(4.0, 4.0, 4.0),
            skip_erosion=True,
        ),
        # Left wrist camera - D455 analogue (16:9, VFOV ~58deg)
        MjcfCameraConfig(
            name="wrist_camera_l",
            mjcf_name="wrist_camera_l",
            robot_namespace="robot_0/",
            fov=58.0,  # D455 depth VFOV
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
        # Right wrist camera - D455 analogue (16:9, VFOV ~58deg)
        MjcfCameraConfig(
            name="wrist_camera_r",
            mjcf_name="wrist_camera_r",
            robot_namespace="robot_0/",
            fov=58.0,  # D455 depth VFOV
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
    ]


class FrankaRandomizedD405D455CameraSystem(CameraSystemConfig):
    """Camera system for Franka pick-and-place tasks with wrist cam and 2 randomized exo cams.

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (624, 352)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_cam",
            robot_namespace="robot_0/",
            fov=58.0,
            fov_noise_degrees=(-10.0, 10.0),  # ±10° FOV noise
            pos_noise_range=(-0.015, 0.015),  # ±1.5cm position noise
            orientation_noise_degrees=8.0,  # ±8° rotation noise
        ),
        # Two randomized exocentric cameras positioned around workspace center
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.2, 0.8),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(50, 90),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.2, 0.8),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(50, 90),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaDroidCameraSystem(CameraSystemConfig):
    """Camera system for Franka with DROID-style fixed cameras.

    Uses wrist camera plus DROID-style exocentric camera mounted to robot base.
    All cameras are deterministic (no noise) for consistent, reproducible viewpoints.
    This matches the behavior of the old `cameras_fixed_droid=True` setting.
    """

    img_resolution: tuple[int, int] = (
        624,
        352,
    )  # 16:9 aspect ratio and divisible by 16px for video encoding
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera (with depth for D405 simulation)
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=56.74,
            # record_depth=True,  # Enable depth recording for wrist camera
        ),
        # DROID-style exocentric camera (mounted to robot base)
        RobotMountedCameraConfig(
            name="exo_camera_1",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=71.0,
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
                # "__gripper__": 0.001,  # not necessarily visible from this cam - just focus on the object
            },
        ),
    ]


class FrankaEasyRandomizedDroidCameraSystem(CameraSystemConfig):
    """Camera system for Franka DROID system with wrist cam (ZED mini) and 2 randomized exo cams (ZED 2/ZED 2i).

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (624, 352)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.0,
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
        RobotMountedCameraConfig(  # left shoulder
            name="exo_camera_1",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=71.0,
            pos_noise_range=(-0.05, 0.05),
            orientation_noise_degrees=8.0,
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
            },
        ),
        # only use one camera at a time (having both at the same time tanks placement success)
        # RobotMountedCameraConfig(  # right shoulder
        #     name="exo_camera_2",
        #     reference_body_names=["robot_0/fr3_link0"],
        #     camera_offset=[0.1, -0.57, 0.66],
        #     camera_quaternion=[0.8190819, -0.42629058, 0.12409726, -0.36329197],
        #     fov=71.0,
        #     pos_noise_range=(-0.05, 0.05),
        #     orientation_noise_degrees=8.0,
        #     visibility_constraints={
        #         "__task_objects__": 0.001,  # Resolved by task sampler
        #     },
        # ),
    ]


# The generated shoulder camera, stated as the two things that actually carry intent: where the
# camera sits in the arm's base frame, and what it looks at. Both are Franka's, not invented here --
# SHOULDER_OFFSET is its calibrated offset from the arm root, and SHOULDER_LOOKAT is where its
# calibrated quaternion turns out to point, [0.66, -0.09, 0], with the small lateral term dropped so
# the aim is centred rather than carrying one rig's residual. The pedestal top is roughly table
# height, so z = 0 is the tabletop.
SHOULDER_OFFSET = [0.1, 0.57, 0.66]
SHOULDER_LOOKAT = [0.66, 0.0, 0.0]


def _lookat_quaternion(camera_offset, lookat_offset, up=(0.0, 0.0, 1.0)) -> list[float]:
    """``[w, x, y, z]``, relative to the reference body, aiming a camera at ``lookat_offset``.

    Reproduces ``CameraManager.create_lookat_pose``'s construction in the reference body's own frame,
    so a camera declared with this quaternion points where the same ``lookat_offset`` would have
    pointed it -- see the shoulder camera below for why the quaternion path is the one that works.
    MuJoCo cameras look along their local -z with +y up, hence the column order."""
    position, target, up_hint = (np.asarray(v, float) for v in (camera_offset, lookat_offset, up))
    forward = target - position
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up_hint)
    right /= np.linalg.norm(right)
    frame = np.column_stack([right, np.cross(right, forward), -forward])
    # rotation matrix -> quaternion, scalar first; Shepperd's method via the largest diagonal term
    trace = float(np.trace(frame))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = [
            0.25 * s,
            (frame[2, 1] - frame[1, 2]) / s,
            (frame[0, 2] - frame[2, 0]) / s,
            (frame[1, 0] - frame[0, 1]) / s,
        ]
    else:
        i = int(np.argmax(np.diag(frame)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = np.sqrt(1.0 + frame[i, i] - frame[j, j] - frame[k, k]) * 2.0
        quat = [(frame[k, j] - frame[j, k]) / s, 0.0, 0.0, 0.0]
        quat[1 + i], quat[1 + j], quat[1 + k] = (
            0.25 * s,
            (frame[j, i] + frame[i, j]) / s,
            (frame[k, i] + frame[i, k]) / s,
        )
    return [round(float(v), 8) for v in np.asarray(quat) / np.linalg.norm(quat)]


class GenericOmniPurposeCameraSystem(CameraSystemConfig):
    """Robot-agnostic wrist camera plus randomized exocentrics, for procedurally generated arms.

    The generic analogue of :class:`FrankaOmniPurposeCameraSystem`. That system cannot be reused for
    a generated embodiment because it names Franka's own MJCF camera
    (``mjcf_name="gripper/wrist_camera"``) and mounts its shoulder camera at a hand-measured offset
    from ``fr3_link0``. The camera system it replaces sidestepped the problem by dropping
    robot-mounted cameras entirely, which left generated data exocentric-only while the shipped
    Franka data is multi-view.

    ``molmo_spaces.embodiment_gen`` emits every arm with a wrist camera under one stable name,
    ``wrist_camera``, so a wrist stream CAN be named for a generated robot even though it cannot be
    for an arbitrary third-party one. The pose is not sampled: it is derived from whichever camera
    the chosen gripper actually ships, expressed in that gripper's grasp frame, so the stream shows
    the same view of the jaws on every robot carrying that gripper. Camera placement is therefore a
    fact about the embodiment, and the only thing this system contributes is the per-episode jitter
    -- which is copied from the Franka entry verbatim rather than widened.

    The shoulder camera is Franka's, re-derived rather than copied. Its calibrated quaternion is a
    rotation relative to ``fr3_link0``; read back, it aims at ``[0.66, -0.09, 0]`` -- 0.66 m ahead of
    the arm root at root height, essentially on the centreline. That intent DOES transfer, because
    ``embodiment_gen`` emits every base in the same convention (z up, the reach ahead along +x, which
    is what ``builder.rest_pose`` scores against), so :func:`_lookat_quaternion` reproduces it from
    the aim point instead of inheriting a number measured on one rig. The derived aim lands 4.1 deg
    off Franka's -- that residual is exactly the rig's lateral offset, dropped so the view is centred
    on the arm rather than carrying one installation's calibration. A test pins it.
    """

    img_resolution: tuple[int, int] = (624, 352)

    # Camera placement follows the ROBOT, not the run: a generated embodiment is identified by its
    # seed, and this makes its camera sequence a property of that seed too, so re-running one robot
    # reproduces its views and two robots in one batch do not share a camera draw. See
    # CameraSystemConfig.camera_seed for what the alternatives do.
    camera_seed: int | Literal["embodiment"] | None = "embodiment"

    # Lean datagen: RGB only; object_image_points is dropped by the wods converter, so skip its
    # per-step segmentation render.
    record_object_image_points: bool = False

    cameras: list[AllCameraTypes] = [
        # Wrist camera. Franka references its gripper's own MJCF camera; every embodiment_gen robot
        # is emitted with one under the uniform name "wrist_camera", posed from whichever camera its
        # gripper actually ships (robotiq's wrist_camera, rum's wrist_cam) or, for a gripper that
        # ships none, from the convention those two follow. Jitter copies Franka's exactly.
        MjcfCameraConfig(
            name="wrist_camera_zed_mini",
            mjcf_name="wrist_camera",
            # ...and the names the shipped robots use for the same camera, so this one system also
            # serves an embodiment that was not generated. The recorded stream is keyed on `name`, so
            # a dataset mixing generated and shipped robots still has one wrist column.
            mjcf_name_alternatives=["gripper/wrist_camera", "gripper/wrist_cam", "wrist_cam"],
            robot_namespace="robot_0/",
            fov=None,
            fov_noise_degrees=(-8.0, 8.0),
            pos_noise_range=((-0.015, -0.005, -0.02), (0.015, 0.005, 0.02)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=False,  # lean datagen: RGB only
        ),
        # Lean datagen: the shoulder camera and the GoPro exocentric are dropped; only the two ZED2
        # exocentrics remain (verbatim from FrankaOmniPurposeCameraSystem).
        RandomizedExocentricCameraConfig(
            name="randomized_zed2_analogue_1",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,
                "__gripper__": 0.0001,
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="randomized_zed2_analogue_2",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,
                "__gripper__": 0.0001,
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaOmniPurposeCameraSystem(CameraSystemConfig):
    """Camera system for Franka DROID system with wrist cam (ZED mini), droid-alike left shoulder cam,
    2 randomized Zed2 cams, and 1 randomized GoPro cam. Intended such that data with this camera system
    can be used for a wide variety of purposes and maximally consistent ablations.

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (624, 352)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera_zed_mini",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.0,
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.02), (0.015, 0.005, 0.02)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
            record_depth=True,
        ),
        RobotMountedCameraConfig(  # left shoulder
            name="droid_shoulder_light_randomization",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=71.0,
            pos_noise_range=(-0.05, 0.05),
            orientation_noise_degrees=8.0,
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
            },
        ),
        # Two randomized exocentric cameras positioned around workspace center
        RandomizedExocentricCameraConfig(
            name="randomized_zed2_analogue_1",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="randomized_zed2_analogue_2",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="randomized_gopro_analogue_1",
            distance_range=(0.2, 0.5),
            height_range=(0.1, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(137, 140),  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaRandomizedDroidCameraSystem(CameraSystemConfig):
    """Camera system for Franka DROID system with wrist cam (ZED mini) and 2 randomized exo cams (ZED 2/ZED 2i).

    Uses workspace center from task sampler for dynamic placement. The task sampler
    should implement get_workspace_center() and resolve_visibility_object() to provide
    runtime information without modifying the camera config.
    """

    img_resolution: tuple[int, int] = (624, 352)
    cameras: list[AllCameraTypes] = [
        # Wrist-mounted camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.0,
            fov_noise_degrees=(-4.0, 4.0),
            pos_noise_range=((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
            orientation_noise_degrees=(8.0, 4.0, 4.0),
        ),
        # Two randomized exocentric cameras positioned around workspace center
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.2, 0.8),
            height_range=(0.05, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(64, 72),
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_3",
            distance_range=(0.2, 0.5),
            height_range=(0.1, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov_range=(137, 140),  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaGoProD405D455CameraSystem(CameraSystemConfig):
    """Camera system for Franka with GoPro and D405 analogue cameras with noise.

    Uses:
    - D405 analogue wrist camera: VFOV=58°, resolution 640x480, with position and orientation noise
    - 455 analogue exo camera: VFOV=58°, resolution 640x480, with position and orientation noise but around droid shoulder
    - GoPro analogue exo camera: VFOV=139°, resolution 640x480, with position and orientation noise

    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # D405-style wrist camera with noise
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=58.0,  # D405 vertical FOV
            record_depth=True,  # D405 has depth capability
            pos_noise_range=(-0.01, 0.01),  # ±1cm position noise
            orientation_noise_degrees=2.0,  # ±2° rotation noise
        ),
        # 455 analogue in noisy droid position
        RobotMountedCameraConfig(
            name="exo_camera_1",
            reference_body_names=["robot_0/fr3_link0"],
            camera_offset=[0.1, 0.57, 0.66],
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],
            fov=58.0,  # 455 vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            pos_noise_range=(-0.10, 0.10),  # ±2cm position noise
            orientation_noise_degrees=3.0,  # ±3° rotation noise
            visibility_constraints={
                "__task_objects__": 0.001,  # Resolved by task sampler
                # "__gripper__": 0.001,  # not necessarily visible from this cam - just focus on the object
            },
        ),
        # fully randomized gopro-analogue exo camera
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.2, 0.5),
            height_range=(0.1, 0.6),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaGoProD405RandomizedCameraSystem(CameraSystemConfig):
    """Camera system for Franka with D405 wrist cam and 2 randomized GoPro exo cams.

    Uses:
    - D405 analogue wrist camera: VFOV=58°, resolution 640x480, with position and orientation noise
    - Two randomized GoPro exo cameras: VFOV=139°, resolution 640x480, with visibility constraints

    Workspace center sourced from task sampler, exo cameras positioned to maximize visibility
    of pickup object and gripper.
    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # D405-style wrist camera with noise
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_cam",
            robot_namespace="robot_0/",
            fov=58.0,  # D405 vertical FOV
            record_depth=True,  # D405 has depth capability
            pos_noise_range=(-0.01, 0.01),  # ±1cm position noise
            orientation_noise_degrees=2.0,  # ±2° rotation noise
        ),
        # Two randomized GoPro-style exocentric cameras
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.4, 1.0),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_2",
            distance_range=(0.4, 1.0),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__task_objects__": 0.0001,  # Resolved by task sampler
                "__gripper__": 0.0001,  # Resolved by task sampler
            },
            max_placement_attempts=20,
            allow_relaxed_constraints=False,
        ),
    ]


class FrankaRobotiq2f85CameraSystem(CameraSystemConfig):
    """Camera system for Franka with Robotiq 2f85 wrist cam and 2 randomized GoPro exo cams.

    Uses:
    - Robotiq 2f85 wrist camera: VFOV=56.74°, resolution 1280x720, with position and orientation noise
    - Two randomized GoPro exo cameras: VFOV=139°, resolution 640x480, with visibility constraints
    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # Robotiq 2f85-style wrist camera with noise
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/",
        ),
        RandomizedExocentricCameraConfig(
            name="exo_camera_1",
            distance_range=(0.4, 1.0),
            height_range=(0.4, 0.8),
            azimuth_range=(0, 2 * np.pi),
            fov=139.0,  # GoPro vertical FOV
            is_warped=False,  # NOTE: baked in warping not yet implemented
            lookat_noise_range=(-0.1, 0.1),
            visibility_constraints={
                "__pickup_object__": 0.001,  # Resolved by task sampler
            },
            max_placement_attempts=200,
            allow_relaxed_constraints=True,
        ),
    ]


class I2rtYamCameraSystem(CameraSystemConfig):
    """Camera system for i2rt YAM robot.

    Uses robot-mounted exo camera since YAM doesn't have built-in MJCF cameras.
    The exo camera is mounted relative to the robot base (mocap body at ground level).

    Note: Camera offset z must account for the base platform height (0.7m).
    To achieve similar viewing angle as Franka DROID (camera at ~1.24m total height),
    we use z offset = 0.7 (platform) + 0.5 (above platform) = 1.2m
    """

    img_resolution: tuple[int, int] = (640, 480)
    cameras: list[AllCameraTypes] = [
        # Exocentric camera mounted relative to robot base (mocap body at ground level)
        RobotMountedCameraConfig(
            name="exo_camera_1",
            reference_body_names=["robot_0/base", "robot_0/arm"],  # Try base first (mocap body)
            camera_offset=[0.1, 0.5, 1.2],  # z=1.2 to account for 0.7m platform + 0.5m above
            camera_quaternion=[-0.3633, -0.1241, 0.4263, 0.8191],  # Similar angle to Franka DROID
            fov=71.0,
            visibility_constraints={
                "__task_objects__": 0.001,
            },
        ),
    ]


class BimanualYamCameraSystem(CameraSystemConfig):
    """Camera system for bimanual YAM robot.

    Includes wrist cameras on both arms (defined in yam.xml MJCF) and a robot-mounted
    exo camera positioned to see both arms and the workspace between them.

    Note: Camera offset z must account for the base platform height (0.7m).
    The exo camera is positioned slightly back and higher to capture both arms.
    """

    img_resolution: tuple[int, int] = (640, 360)
    cameras: list[AllCameraTypes] = [
        # Left wrist camera (defined in yam.xml, attached to left_link_6)
        MjcfCameraConfig(
            name="left_wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/left_",
            fov=58.0,
        ),
        # Right wrist camera (defined in yam.xml, attached to right_link_6)
        MjcfCameraConfig(
            name="right_wrist_camera",
            mjcf_name="wrist_camera",
            robot_namespace="robot_0/right_",
            fov=58.0,
        ),
        # Exocentric camera mounted relative to robot base (mocap body at ground level)
        RobotMountedCameraConfig(
            name="exo_camera",  # Name matches policy camera_mapping expectation
            reference_body_names=[
                "robot_0/base",
                "robot_0/left_arm",
            ],  # Try base first (mocap body)
            camera_offset=[0.0, 0.0, 1.56],  # Centered between both arms, 86cm above platform
            camera_quaternion=[
                0.6870,
                0.1675,
                -0.1675,
                -0.6870,
            ],  # 90° CW roll + 46.5° pitch down, arms left/right
            fov=58.0,
            visibility_constraints={
                "__task_objects__": 0.001,
            },
        ),
    ]


class FrankaEvalCameraSystem(CameraSystemConfig):
    """Unified Franka eval camera system with progressive perturbation (0-100 level).

    Cameras:
    - 1 wrist camera (FrankaDroid-like)
    - 1 ZED2-like exocentric cameras (stored pose from episode + perturbation)

    The cameras list holds only calibrated (level 0) specs with no randomization ranges.
    All randomization ranges are in per-reference level, per-camera dicts.
    Level scaling is applied by apply_eval_camera_randomization_level() in eval_camera_randomization_utils.py
    using an N-piece linear curve (e.g. 0→low→high→100 for N=3).

    Exo camera pos/forward/up are placeholders; the runtime loads stored poses from episode specs.
    """

    img_resolution: tuple[int, int] = (624, 352)

    ref_level_ranges: ClassVar[
        list[tuple[float, dict[str, dict[str, float | tuple[float, ...]]]]]
    ] = [
        (
            0.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                    "orientation_noise_degrees": (0.0, 0.0, 0.0),
                    "fov_noise_degrees": (0.0, 0.0),
                },
                "exo_camera_1": {
                    "azimuth_range": (0.0, 0.0),
                    "distance_range": (0.0, 0.0),
                    "height_range": (0.0, 0.0),
                    "workspace_center_weight": 0.0,  # keep calibrated orientation
                    "lookat_noise_range": (0.0, 0.0),
                    "fov_range": (71.0, 71.0),
                },
            },
        ),
        (
            10.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": {
                    "azimuth_range": (-np.pi / 4, np.pi / 4),
                    "distance_range": (-0.05, 0.05),
                    "height_range": (-0.05, 0.05),
                    "workspace_center_weight": 1.0,  # look at workspace center
                    "lookat_noise_range": (-0.01, 0.01),
                    "fov_range": (71.0, 71.0),
                },
            },
        ),
        (
            40.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": {
                    "azimuth_range": (-np.pi / 2, np.pi / 2),
                    "distance_range": (-0.5, 0.5),
                    "height_range": (-0.1, 0.5),
                    "workspace_center_weight": 1.0,  # look at workspace center
                    "lookat_noise_range": (-0.05, 0.05),
                    "fov_range": (64.0, 72.0),
                },
            },
        ),
        (
            75.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": dict(
                    azimuth_range=(-np.pi, np.pi),
                    distance_range=(-0.5, 1.0),
                    height_range=(-0.2, 0.7),
                    workspace_center_weight=1.0,  # look at workspace center
                    lookat_noise_range=(-0.1, 0.1),
                    fov_range=(64.0, 72.0),
                ),
            },
        ),
        (
            100.0,
            {
                "wrist_camera": {
                    "pos_noise_range": ((-0.015, -0.005, -0.01), (0.015, 0.005, 0.01)),
                    "orientation_noise_degrees": (8.0, 4.0, 4.0),
                    "fov_noise_degrees": (-4.0, 4.0),
                },
                "exo_camera_1": dict(
                    azimuth_range=(-np.pi, np.pi),
                    distance_range=(-0.5, 1.5),
                    height_range=(-0.3, 0.8),
                    workspace_center_weight=1.0,  # look at workspace center
                    lookat_noise_range=(-0.15, 0.15),
                    fov_range=(64.0, 78.0),
                ),
            },
        ),
    ]

    # Calibrated (level 0) camera specs only; no randomization params.
    cameras: list[AllCameraTypes] = [
        # Wrist camera
        MjcfCameraConfig(
            name="wrist_camera",
            mjcf_name="gripper/wrist_camera",
            robot_namespace="robot_0/",
            fov=52.00,
        ),
        # Shoulder camera
        EvalExocentricCameraConfig(
            name="exo_camera_1",
            fov=71.0,
            visibility_constraints={
                "__task_objects__": 0.0001,
                "__gripper__": 0.0001,
            },
        ),
    ]


AllCameraSystems: TypeAlias = (
    RBY1MjcfCameraSystem
    | RBY1GoProD455CameraSystem
    | FrankaRandomizedD405D455CameraSystem
    | FrankaEasyRandomizedDroidCameraSystem
    | FrankaDroidCameraSystem
    | FrankaOmniPurposeCameraSystem
    | GenericOmniPurposeCameraSystem
    | FrankaRandomizedDroidCameraSystem
    | FrankaGoProD405D455CameraSystem
    | FrankaGoProD405RandomizedCameraSystem
    | FrankaRobotiq2f85CameraSystem
    | FrankaEvalCameraSystem
    | I2rtYamCameraSystem
    | BimanualYamCameraSystem
    | FrankaEvalCameraSystem
)
