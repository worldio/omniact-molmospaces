# EmbodimentSpec

!!! warning
    Schema version 1 and its generic runtime are implemented as an experimental
    API. Every current `BaseRobotConfig` embodiment has an executable,
    legacy-compared specification. The format is not a stable public API.

## Summary

`EmbodimentSpec` is a data-only serialization of MolmoSpaces' existing robot
abstractions. It mirrors the current class structure:

```text
BaseRobotConfig
Robot
RobotView
MoveGroup
Controller
Kinematics
Sensor
```

The specification contains no Python import paths, factories, callbacks, or
embodiment-specific classes such as `I2rtYamRobot`. A generic runtime interprets
closed, versioned schema operations to construct the robot.

MJCF remains the source of truth for bodies, joints, actuators, limits,
inertias, geometry, tendons, equality constraints, and other MuJoCo behavior.
`EmbodimentSpec` supplies the semantic and runtime information that MJCF does
not define.

## Goals

- Represent every currently supported MolmoSpaces embodiment.
- Preserve every MJCF body, joint, actuator, site, and geometry name used by
  the runtime.
- Make arms, tools, end effectors, bases, controllers, and command encodings
  explicit.
- Allow the generic planner to select an end effector without relying on
  naming conventions or the first gripper in a model.
- Derive dimensions and limits from compiled MJCF instead of duplicating them.
- Support fixed, planar, and floating bases; coupled joints; scalar synergies;
  single-arm, bimanual, and gripper-only embodiments.
- Keep grasp compatibility and grasp-frame conventions explicit and versioned.
- Permit complete validation before a rollout starts.

## Non-goals

- Replacing MJCF as a physical robot-description format.
- Embedding arbitrary executable Python in YAML or JSON.
- Encoding task, scene, camera-system, or policy-model configuration in the
  embodiment.
- Guaranteeing that every future mechanism can be represented without a schema
  revision.
- Standardizing hardware communication protocols.

## Design Rules

1. The format follows MolmoSpaces concepts rather than introducing an opaque
   robot factory.
2. Every `kind` is a closed schema variant with documented behavior.
3. Every MJCF name is explicit and namespace-relative unless marked absolute.
4. MJCF-derived values, including joint dimensions and limits, are validated
   but not serialized redundantly.
5. Planner-critical behavior cannot be placed in an untyped `extensions`
   dictionary.
6. New behavior requires a typed schema addition and interpreter support.
7. Stable IDs and versions identify embodiments and grasp profiles; Python
   class names do not.

## Top-level Structure

```python
class EmbodimentSpec(Config):
    schema_version: int
    embodiment_id: str
    embodiment_version: str

    robot_config: RobotConfigSpec
    robot: RobotSpec
    robot_view: RobotViewSpec

    grasp_profiles: dict[str, GraspProfileSpec]
```

The three main sections intentionally correspond to the current
`BaseRobotConfig`, `Robot`, and `RobotView` layers.

The implementation lives in `molmo_spaces.robots.configured`. Use
`load_embodiment_spec()` to validate YAML and `ConfiguredRobotConfig.from_file()`
to expose it through the existing task-sampler and kinematics hooks.

Executable examples:

- [`embodiment_specs/i2rt_yam.yaml`](embodiment_specs/i2rt_yam.yaml)
- [`embodiment_specs/bimanual_yam.yaml`](embodiment_specs/bimanual_yam.yaml)
- [`embodiment_specs/franka_droid.yaml`](embodiment_specs/franka_droid.yaml)
- [`embodiment_specs/franka_cap.yaml`](embodiment_specs/franka_cap.yaml)
- [`embodiment_specs/mobile_franka_droid.yaml`](embodiment_specs/mobile_franka_droid.yaml)
- [`embodiment_specs/floating_rum.yaml`](embodiment_specs/floating_rum.yaml)
- [`embodiment_specs/floating_robotiq.yaml`](embodiment_specs/floating_robotiq.yaml)
- [`embodiment_specs/rby1.yaml`](embodiment_specs/rby1.yaml)
- [`embodiment_specs/rby1m.yaml`](embodiment_specs/rby1m.yaml)
- [`embodiment_specs/rby1m_open_close.yaml`](embodiment_specs/rby1m_open_close.yaml)

Closely related variants may declare `extends` with a relative YAML path. The
loader recursively merges mappings, rejects inheritance cycles, and validates
the fully expanded result against the same strict schema.

## Experiment Composition

`EmbodimentSpec` intentionally does not contain task, camera, sampler, policy,
scene, or output configuration. This follows the existing `MlSpacesExpConfig`
structure, where `robot_config`, `camera_config`, `task_config`,
`task_sampler_config`, and `policy_config` are sibling components.

A generic experiment config should remain a normal `MlSpacesExpConfig` subclass
and use the same field names and concrete config types as existing registered
configs. It must not contain an embodiment path. Instead, it declares an empty
robot-config injection slot:

```python
@register_config("GenericPickOmniCamConfig")
class GenericPickOmniCamConfig(PickBaseConfig):
    robot_config: BaseRobotConfig | None = None
    camera_config: CameraSystemConfig = GenericOmniCameraSystem()
    task_config: PickTaskConfig = PickTaskConfig(task_cls=PickTask)
    task_sampler_config: PickTaskSamplerConfig = PickTaskSamplerConfig(
        task_sampler_class=PickTaskSampler,
    )
    policy_config: BasePolicyConfig = PickPlannerPolicyConfig()

    filter_for_successful_trajectories: bool = False
    output_dir: Path = ASSETS_DIR / "experiment_output" / "datagen" / "pick_omni"
```

Equivalent generic configs may be registered for pick-and-place, open/close,
navigation, or other task families. The embodiment does not select the task.

If experiment configs are later serialized, their schema should mirror
`MlSpacesExpConfig` directly rather than introducing differently named task,
sampler, policy, or output wrappers:

```python
class DataGenerationExpConfigSpec(Config):
    schema_version: int
    num_envs: int
    num_workers: int
    seed: int | None
    task_horizon: int | None

    camera_config: CameraSystemSpec
    task_config: TaskConfigSpec
    task_sampler_config: TaskSamplerConfigSpec
    policy_config: PolicyConfigSpec

    filter_for_successful_trajectories: bool
    output_dir: Path
```

The embodiment is a separate CLI input. `main.py` instantiates the experiment
config first, loads the embodiment independently, and injects
`ConfiguredRobotConfig` into the existing `robot_config` field before validation:

```bash
python molmo_spaces/data_generation/main.py \
  GenericPickOmniCamConfig \
  --embodiment-spec docs/embodiment_specs/i2rt_yam.yaml
```

Existing registered experiment-config names remain supported without
`--embodiment-spec`; they continue using their declared `robot_config`. A generic
config with `robot_config=None` must require `--embodiment-spec`. Passing an
embodiment YAML as the experiment config is invalid because an embodiment does
not determine a task, camera system, sampler, or policy.

### Camera system schema

`CameraSystemSpec` should mirror the existing `CameraSystemConfig` hierarchy as
a closed union:

```python
CameraSpec = Annotated[
    MjcfCameraSpec
    | RobotMountedCameraSpec
    | FixedExocentricCameraSpec
    | RandomizedExocentricCameraSpec,
    Field(discriminator="kind"),
]


class CameraSystemSpec(Config):
    img_resolution: tuple[int, int]
    cameras: list[CameraSpec]
```

MJCF cameras reference an explicit model camera name. Robot-mounted cameras
reference an explicit body plus their calibrated offset and orientation.
Exocentric cameras remain independent of robot topology. Every camera name must
be unique.

The current `FrankaOmniPurposeCameraSystem` is only partly generic. Its three
randomized exocentric cameras are reusable, while its wrist camera references
`gripper/wrist_camera` and its shoulder camera references `fr3_link0`. The
camera definitions should therefore be split into:

- A generic exocentric Omni camera system.
- A DROID camera system that adds its calibrated wrist and shoulder cameras.
- Optional embodiment-specific camera systems, such as YAM, that add only
  explicitly calibrated mounts.

DROID mounting transforms must not be copied automatically to another
embodiment. A YAM wrist or shoulder camera requires its own declared body,
offset, orientation, and validation.

### Composition validation

Before scene rollout, `main.py` must:

1. Instantiate the selected registered or serialized experiment config.
2. Load the separately supplied embodiment and inject `ConfiguredRobotConfig`
   into `exp_config.robot_config`.
3. Validate the resulting `MlSpacesExpConfig` using its existing sibling config
   structure.
4. Assemble and compile the configured robot.
5. Resolve every MJCF camera and robot-mounted body reference.
6. Validate camera names, visibility constraints, task sensor requirements,
   action dimensions, selected end effectors, and required kinematics backend.
7. Reject incompatible combinations with a clear error instead of dropping
   cameras or substituting robot-specific defaults.

The generic experiment configs set `filter_for_successful_trajectories=False`,
which uses the existing pipeline behavior to persist completed successful and
failed trajectories. Camera videos are saved for every persisted trajectory
using all cameras selected by `camera_config`.

Implementation status: `GenericPickOmniCamConfig`, `GenericOmniCameraSystem`,
and separate `main.py --embodiment-spec` injection are implemented. Registered
legacy configs retain their declared robots and cameras. Mounted-camera names
and body references are validated before worker launch. YAML serialization of
the camera and experiment config hierarchies remains future work.

## Shared References

All model names are explicit. Names are interpreted relative to
`robot_config.namespace` unless `absolute` is true.

```python
class ModelNameRef(Config):
    name: str
    absolute: bool = False


class FrameRef(Config):
    kind: Literal["body", "site"]
    name: ModelNameRef
```

Transforms use homogeneous 4-by-4 matrices and the notation `a_T_b`, meaning
the pose of frame `b` expressed in frame `a`.

```python
class TransformSpec(Config):
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
```

## RobotConfigSpec

`RobotConfigSpec` serializes the data currently held by `BaseRobotConfig` and
its subclasses.

```python
class RobotConfigSpec(Config):
    model: ModelSpec
    namespace: str

    initial_qpos: dict[str, list[float]]
    initial_qpos_noise: dict[str, list[float]] | None

    default_control_profile: str
    control_profiles: dict[str, ControlProfileSpec]

    gravity_compensation: GravityCompensationSpec
    action_noise: ActionNoiseSpec | None
    placement: PlacementSpec
```

### Model source

```python
class ModelSpec(Config):
    source: Literal["builtin", "path"]
    resource_key: str | None
    resource_version: str | None
    root_path: Path | None
    mjcf_path: Path

    model_root_body: ModelNameRef
    pre_namespaced: bool = False
    prefix_names: bool = True
```

`resource_key` identifies a versioned asset package. It does not select runtime
behavior.

### Initial state

`initial_qpos` is keyed by move-group ID and stores state-space joint values,
not controller command values. The loader validates each value against the
compiled group's actual `qpos` dimension and joint limits.

### Control profiles

```python
class ControlProfileSpec(Config):
    controller_bindings: dict[str, str]
    actuator_overrides: dict[str, ActuatorOverrideSpec]


class ActuatorOverrideSpec(Config):
    actuator: ModelNameRef
    gain: float | None = None
    bias_position: float | None = None
    bias_velocity: float | None = None
    force_range: tuple[float, float] | None = None
```

This represents current stiffness, damping, force-limit, and gripper-servo
overrides without a robot-specific method.

### Gravity compensation

```python
class GravityCompensationSpec(Config):
    enabled: bool
    root_body: ModelNameRef | None
    include_root: bool = True
    include_descendants: bool = True
```

### Action noise

```python
class ActionNoiseSpec(Config):
    enabled: bool
    target_arm_groups: list[str]
    target_base_group: str | None
    action_scale_factor: float
    rotation_noise_scale: float
    max_tcp_position_noise: float
    max_tcp_rotation_noise: float
    base_action_scale_factor: float
    max_base_position_noise: float
    max_base_rotation_noise: float
```

### Placement semantics

```python
class PlacementSpec(Config):
    reset_repositionable: bool
    runtime_mobility: Literal["fixed", "se2", "se3"]
    footprint_xy: tuple[float, float] | None
    platform_size_xyz: tuple[float, float, float] | None
    forward_offset: float = 0.0
    forward_offset_step: float = 0.02
```

The planner may derive reachability from kinematics, but reset mobility and
footprint semantics cannot be inferred safely from a free joint or mocap body.

## RobotSpec

`RobotSpec` serializes behavior currently implemented by concrete `Robot`
subclasses.

```python
class RobotSpec(Config):
    model_operations: list[ModelOperationSpec]
    controllers: dict[str, ControllerSpec]
    kinematics: KinematicsSpec
    sensors: list[SensorSpec]
```

### Model operations

Model operations are applied in order before compilation.
Every operation declares `scope: "robot_model" | "scene"` so name resolution
before and after model attachment is unambiguous.

```python
ModelOperationSpec = Annotated[
    AddSiteSpec
    | AddBodySpec
    | AddTextureSpec
    | AddMaterialSpec
    | AddGeomSpec
    | AddJointSpec
    | AddActuatorSpec
    | AddMocapBodySpec
    | AddWeldSpec
    | AddPlatformSpec
    | AttachModelSpec
    | SetBodyPropertiesSpec,
    Field(discriminator="kind"),
]
```

Representative variants:

```python
class AddSiteSpec(Config):
    kind: Literal["add_site"]
    parent_body: ModelNameRef
    name: ModelNameRef
    position: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float] | None = None
    xyaxes: tuple[float, float, float, float, float, float] | None = None


class AddMocapBodySpec(Config):
    kind: Literal["add_mocap_body"]
    name: ModelNameRef
    initial_pose: TransformSpec


class AddTextureSpec(Config):
    kind: Literal["add_texture"]
    name: ModelNameRef
    texture_type: Literal["2d", "cube"]
    file: Path
    randomized_files: list[Path] = []


class SpeckleRgbaMaterialsSpec(Config):
    kind: Literal["speckle_rgba_materials"]
    scope: Literal["scene"]
    model_root_body: ModelNameRef
    probability: float
    texture_size: int = 256
    noise_strength: float = 0.1
    num_blobs: int = 80
    blob_size_range: tuple[int, int] = (5, 25)
    blob_variation: float = 0.1


class AddWeldSpec(Config):
    kind: Literal["add_weld"]
    body_a: ModelNameRef
    body_b: ModelNameRef
    solref: tuple[float, float] | None = None
    solimp: tuple[float, float, float, float, float] | None = None
```

These operations cover behavior currently implemented procedurally, including
YAM planner-site injection, DROID visual randomization, mocap bases,
floating-gripper welds, platforms, and mobile-base augmentation.

### Controllers

Controller behavior is represented by a closed union rather than a class name.

```python
ControllerSpec = Annotated[
    JointPositionControllerSpec
    | JointRelativePositionControllerSpec
    | DirectActuatorControllerSpec
    | MocapPoseControllerSpec
    | PlanarPositionControllerSpec
    | RelativePlanarPositionControllerSpec
    | LinearSynergyControllerSpec,
    Field(discriminator="kind"),
]
```

Examples:

```python
class JointPositionControllerSpec(Config):
    kind: Literal["joint_position"]
    move_group: str


class LinearSynergyControllerSpec(Config):
    kind: Literal["linear_synergy"]
    move_group: str
    command_dimension: int
    matrix: list[list[float]]
    bias: list[float]
    output_limits: list[tuple[float, float]] | None = None
```

The linear-synergy form covers mappings such as RBY's scalar torso-height
command without embedding an RBY controller class.

### Kinematics

```python
KinematicsSpec = Annotated[
    DampedLeastSquaresKinematicsSpec
    | FloatingFrameKinematicsSpec
    | NoKinematicsSpec,
    Field(discriminator="kind"),
]


class DampedLeastSquaresKinematicsSpec(Config):
    kind: Literal["damped_least_squares"]
    online_backend: Literal["numpy"]
    batch_backend: Literal["warp"] | None
    optimizable_groups: list[str]
    max_iterations: int
    convergence_tolerance: float
    success_tolerance: float
    damping: float
    integration_step: float
    online_max_iterations: int | None = None
    online_success_tolerance: float | None = None


class FloatingFrameKinematicsSpec(Config):
    kind: Literal["floating_frame"]
    base_group: str
    tcp_frame: FrameRef
```

The algorithm and its parameters are specified. Online and batch backends are
declared independently because current robots may expose both
`MlSpacesKinematics` and `SimpleWarpKinematics`. A backend is an implementation
choice with declared semantics, not an embodiment factory.

### Sensors

Sensors needed to interpret the embodiment are also typed.

```python
SensorSpec = Annotated[
    JointStateSensorSpec
    | FramePoseSensorSpec
    | ContactStateSensorSpec
    | ApertureSensorSpec,
    Field(discriminator="kind"),
]
```

Scene cameras and task-specific object sensors remain experiment or task
configuration.

## RobotViewSpec

`RobotViewSpec` serializes the move-group and frame structure currently encoded
in concrete `RobotView` classes.

```python
class RobotViewSpec(Config):
    base_group: str
    move_groups: dict[str, MoveGroupSpec]
    end_effectors: dict[str, EndEffectorSpec]
```

### Move groups

```python
MoveGroupSpec = Annotated[
    JointMoveGroupSpec
    | GripperMoveGroupSpec
    | MocapBaseMoveGroupSpec
    | FreeJointBaseMoveGroupSpec
    | PlanarBaseMoveGroupSpec,
    Field(discriminator="kind"),
]
```

```python
class JointMoveGroupSpec(Config):
    kind: Literal["joint"]
    role: Literal["arm", "torso", "head", "other"]
    side: Literal["left", "right", "center"]
    parent_group: str

    root_body: ModelNameRef
    joint_names: list[ModelNameRef]
    actuator_names: list[ModelNameRef]
    leaf_frame: FrameRef | None


class GripperMoveGroupSpec(Config):
    kind: Literal["gripper"]
    role: Literal["tool"]
    side: Literal["left", "right", "center"]
    parent_group: str

    root_body: ModelNameRef
    joint_names: list[ModelNameRef]
    actuator_names: list[ModelNameRef]
    leaf_frame: FrameRef

    open_command: list[float]
    closed_command: list[float]
    aperture: ApertureSpec | None
```

### Aperture measurements

```python
ApertureSpec = Annotated[
    JointExpressionApertureSpec | GeomDistanceApertureSpec,
    Field(discriminator="kind"),
]


class JointTermSpec(Config):
    joint: ModelNameRef
    coefficient: float
    transform: Literal["identity", "absolute"] = "identity"


class JointExpressionApertureSpec(Config):
    kind: Literal["joint_expression"]
    terms: list[JointTermSpec]
    offset: float = 0.0
    clamp: tuple[float, float] | None = None


class GeomDistanceApertureSpec(Config):
    kind: Literal["geom_distance"]
    geom_a: ModelNameRef
    geom_b: ModelNameRef
    maximum_distance: float
```

These variants cover the existing YAM signed coupled-joint gap, RBY absolute
joint sum, and DROID/CAP/RUM geometry-distance measurements.

### End effectors

```python
class EndEffectorSpec(Config):
    arm_group: str | None
    tool_group: str
    tcp_frame: FrameRef
    grasp_frame: FrameRef
    grasp_profile_id: str
```

Every end effector explicitly associates one manipulator group, one tool group,
and the frames used by IK and grasp data. Bimanual embodiments declare two end
effectors instead of relying on move-group ordering.

`ConfiguredRobotView` exposes these bindings, but existing planner call sites
still select the first gripper. Planner migration is separate from embodiment
construction.

## GraspProfileSpec

`grasp_profile_id` is not a filename and must not trigger an implicit Python or
global registry lookup. It is a reference to a fully declared entry in the
top-level `grasp_profiles` mapping.

```python
class GraspProfileSpec(Config):
    profile_version: str
    tool_type: Literal[
        "parallel_jaw",
        "suction",
        "multi_finger",
        "magnetic",
    ]

    grasp_frame: FrameRef
    approach_axis_in_grasp_frame: tuple[float, float, float]
    closing_axis_in_grasp_frame: tuple[float, float, float] | None
    symmetries: list[TransformSpec]

    open_configuration: list[float]
    closed_configuration: list[float]

    contact_groups: dict[str, ContactGroupSpec]
    grasp_rule: GraspRuleSpec
    release_rule: GraspRuleSpec


class ContactGroupSpec(Config):
    geom_names: list[ModelNameRef]
    body_names: list[ModelNameRef] = []
    include_descendants_of: list[ModelNameRef] = []
    exclude_descendants_of: list[ModelNameRef] = []


class GraspRuleSpec(Config):
    required_contact_groups: list[str]
    forbidden_contact_groups: list[str] = []
    minimum_contact_count: int = 1
    aperture_threshold: float | None = None
```

Example:

```yaml
grasp_profiles:
  yam_parallel_jaw_v1:
    profile_version: "1"
    tool_type: parallel_jaw
    grasp_frame:
      kind: site
      name: {name: planner_grasp_site}
    approach_axis_in_grasp_frame: [0.0, 0.0, 1.0]
    closing_axis_in_grasp_frame: [0.0, 1.0, 0.0]
    symmetries:
      - position: [0.0, 0.0, 0.0]
        quaternion_wxyz: [0.0, 0.0, 0.0, 1.0]
    open_configuration: [0.041]
    closed_configuration: [0.0]
    contact_groups:
      left_finger:
        geom_names: []
        body_names: []
        include_descendants_of: [{name: link_left_finger}]
      right_finger:
        geom_names: []
        body_names: []
        include_descendants_of: [{name: link_right_finger}]
      palm:
        geom_names: []
        body_names: [{name: link_6}]
        include_descendants_of: []
    grasp_rule:
      required_contact_groups: [left_finger, right_finger]
      forbidden_contact_groups: [palm]
    release_rule:
      required_contact_groups: []
```

Grasp-library manifests refer to the same `grasp_profile_id`. A library is
compatible only when its declared profile matches the selected end effector or
declares a versioned compatibility relationship.

Profiles may later be moved into reusable files, but such a reference must be
explicit and reproducible:

```yaml
grasp_profile_source:
  path: profiles/yam_parallel_jaw_v1.yaml
  sha256: ...
```

An unversioned filename or implicit process-global registration is not part of
this proposal.

## YAM Schema Walkthrough

The following abbreviated walkthrough shows how existing YAM behavior maps into
the schema without an `i2rt_yam` runtime class reference. A complete instance is
available at [`embodiment_specs/i2rt_yam.yaml`](embodiment_specs/i2rt_yam.yaml).

```yaml
schema_version: 1
embodiment_id: i2rt_yam.single.fixed
embodiment_version: "20260223"

robot_config:
  namespace: robot_0/
  model:
    source: builtin
    resource_key: i2rt_yam
    resource_version: "20260223"
    mjcf_path: yam.xml
    model_root_body: {name: arm}
    pre_namespaced: false
    prefix_names: true
  initial_qpos:
    arm: [0.0, 1.047, 1.047, 0.1, -0.1, 0.0]
    gripper: [0.0, 0.0]
  initial_qpos_noise: null
  default_control_profile: default
  control_profiles:
    default:
      controller_bindings:
        arm: arm_position
        gripper: gripper_position
      actuator_overrides:
        joint1: {actuator: {name: joint1}, gain: 80.0, bias_position: -80.0, bias_velocity: -5.0}
        joint2: {actuator: {name: joint2}, gain: 80.0, bias_position: -80.0, bias_velocity: -5.0}
        joint3: {actuator: {name: joint3}, gain: 80.0, bias_position: -80.0, bias_velocity: -5.0}
        joint4: {actuator: {name: joint4}, gain: 10.0, bias_position: -10.0, bias_velocity: -1.5}
        joint5: {actuator: {name: joint5}, gain: 10.0, bias_position: -10.0, bias_velocity: -1.5}
        joint6: {actuator: {name: joint6}, gain: 10.0, bias_position: -10.0, bias_velocity: -1.5}
        gripper: {actuator: {name: gripper}, gain: 5000.0, bias_position: -5000.0, bias_velocity: -200.0, force_range: [-50.0, 50.0]}
  gravity_compensation:
    enabled: true
    root_body: {name: arm}
    include_descendants: true
  action_noise:
    enabled: true
    action_scale_factor: 0.1
    rotation_noise_scale: 0.1
    max_tcp_position_noise: 0.02
    max_tcp_rotation_noise: 0.1
    base_action_scale_factor: 0.1
    max_base_position_noise: 0.02
    max_base_rotation_noise: 0.05
  placement:
    reset_repositionable: true
    runtime_mobility: fixed
    footprint_xy: [0.3, 0.3]
    platform_size_xyz: [0.3, 0.3, 0.7]
    forward_offset: 0.2131228
    forward_offset_step: 0.02

robot:
  model_operations:
    - kind: add_site
      parent_body: {name: link_6}
      name: {name: planner_grasp_site}
      position: [0.0, -0.04945, 0.12983]
      xyaxes: [0.0, 0.0, -1.0, 1.0, 0.0, 0.0]
  controllers:
    arm_position: {kind: joint_position, move_group: arm}
    gripper_position: {kind: joint_position, move_group: gripper}
  kinematics:
    kind: damped_least_squares
    online_backend: numpy
    batch_backend: warp
    optimizable_groups: [arm]
    max_iterations: 50
    convergence_tolerance: 0.001
    success_tolerance: 0.0005
    damping: 1.0e-12
  sensors:
    - kind: joint_state
      move_groups: [arm, gripper]
    - kind: frame_pose
      frame: {kind: site, name: {name: planner_grasp_site}}

robot_view:
  base_group: base
  move_groups:
    base:
      kind: mocap_base
      role: base
      body: {name: base}
    arm:
      kind: joint
      role: arm
      side: center
      parent_group: base
      root_body: {name: arm}
      joint_names: [{name: joint1}, {name: joint2}, {name: joint3}, {name: joint4}, {name: joint5}, {name: joint6}]
      actuator_names: [{name: joint1}, {name: joint2}, {name: joint3}, {name: joint4}, {name: joint5}, {name: joint6}]
      leaf_frame: {kind: site, name: {name: planner_grasp_site}}
    gripper:
      kind: gripper
      role: tool
      side: center
      parent_group: base
      root_body: {name: link_6}
      joint_names: [{name: left_finger}, {name: right_finger}]
      actuator_names: [{name: gripper}]
      leaf_frame: {kind: site, name: {name: planner_grasp_site}}
      open_command: [0.041]
      closed_command: [0.0]
      aperture:
        kind: joint_expression
        terms:
          - {joint: {name: left_finger}, coefficient: 1.0}
          - {joint: {name: right_finger}, coefficient: -1.0}
        clamp: [0.0, 0.075]
  end_effectors:
    primary:
      arm_group: arm
      tool_group: gripper
      tcp_frame: {kind: site, name: {name: planner_grasp_site}}
      grasp_frame: {kind: site, name: {name: planner_grasp_site}}
      grasp_profile_id: yam_parallel_jaw_v1

grasp_profiles:
  yam_parallel_jaw_v1:
    profile_version: "1"
    tool_type: parallel_jaw
    grasp_frame: {kind: site, name: {name: planner_grasp_site}}
    approach_axis_in_grasp_frame: [0.0, 0.0, 1.0]
    closing_axis_in_grasp_frame: [0.0, 1.0, 0.0]
    symmetries: []
    open_configuration: [0.041]
    closed_configuration: [0.0]
    contact_groups: {}
    grasp_rule:
      required_contact_groups: [left_finger, right_finger]
    release_rule:
      required_contact_groups: []
```

The external complete example fills the current YAM body, joint, actuator, and
site names; base construction; model attachment; control overrides; and contact
groups. It is also the executable parity fixture for the configured runtime.

## Generic Construction

The runtime contains one interpreter per schema version, not one factory per
embodiment. Existing MolmoSpaces call sites consume it through the standard
`BaseRobotConfig` hooks:

```python
robot_config = ConfiguredRobotConfig.from_file("embodiment.yaml")
scene = mujoco.MjSpec()
robot_config.robot_cls.add_robot_to_scene(
    robot_config,
    scene,
    prefix=robot_config.robot_namespace,
    pos=[0.0, 0.0, 0.0],
    quat=[1.0, 0.0, 0.0, 0.0],
)
model = scene.compile()
robot = robot_config.robot_factory(mujoco.MjData(model), exp_config)
```

Construction performs these steps:

1. Resolve and version-check the MJCF resource.
2. Apply namespace rules and ordered model operations.
3. Compile MJCF.
4. Resolve every declared body, joint, actuator, site, and geometry name.
5. Build move groups and derive their state and actuator dimensions.
6. Validate initial state and control dimensions.
7. Build controllers, kinematics, and sensors from typed variants.
8. Bind end effectors and grasp profiles.
9. Publish the canonical named action and state schemas.

## Validation Requirements

These are the validation requirements for the format. The current runtime
validates schema references first and compiled names, initial-state dimensions,
resolved ownership, grasp contacts, and controller output dimensions when the
robot/view is constructed. TCP reachability, external grasp-library manifests,
and exhaustive controller-limit compatibility remain migration work.

- `embodiment_id` and version must be present and stable.
- Resource version and MJCF file must resolve.
- Namespace and insertion rules must be internally consistent.
- Every referenced model name must exist after model operations.
- A joint or actuator cannot belong to conflicting move groups.
- Initial state dimensions must match derived `qpos` dimensions.
- Controller command dimensions must match their move groups or declared
  synergy matrices.
- Controller output limits must be compatible with actuator limits.
- Every end effector must reference valid arm and tool groups.
- TCP and grasp frames must be reachable from the declared manipulator.
- Kinematics active groups must exist and be compatible with the backend.
- Grasp profile frames and contact groups must resolve.
- Grasp libraries must declare a matching profile or explicit compatibility.

## Issues and Tradeoffs

### The schema interpreter becomes a second robot framework

Current robot behavior is spread across Python subclasses. Replacing those
classes with declarative operations moves complexity into a generic
interpreter. The interpreter must be tested as carefully as the existing robot
implementations.

### Existing Python behavior is not automatically serializable

Arbitrary side effects in `_load_robot_spec`, `add_robot_to_scene`, controller
construction, and sensor construction cannot be recovered from object fields.
Each existing embodiment must be audited and translated explicitly. Compiled
model and rollout equivalence tests are required.

Franka DROID construction includes typed visual-only randomization: selection
from an explicit resource-relative base-texture list and probabilistic procedural
speckling of RGB materials. Both preserve the legacy Python and NumPy RNG order,
and seed-for-seed tests compare the resulting compiled texture data. New visual
randomization algorithms still require new typed schema operations.

### Arbitrary embodiments are bounded by schema expressiveness

Removing executable callbacks means the format can represent arbitrary
combinations of known primitives, not arbitrary future algorithms. A novel
controller, tool, or model mutation requires a typed schema extension and a
new interpreter version.

### `kind` still maps to implementation code

Values such as `kind: joint_position` require code in the interpreter. The
difference from `name: i2rt_yam` is that the operation's inputs, outputs, units,
and behavior are defined by the schema and cannot hide embodiment-specific
logic. These operations require conformance tests and semantic versioning.

### MJCF and the spec can drift

The spec repeats model names and some semantic ranges. Asset updates can rename
or remove them. Validation catches structural drift, while resource hashes and
versions make the result reproducible.

### Model-operation ordering is significant

Prefixing, attaching, adding sites, wrapping a base, and applying actuator
overrides may not commute. Ordered operations must be deterministic and included
in the embodiment version.

### Coupled and underactuated systems are difficult

Linear synergies cover current RBY torso and simple coupled fingers, but tendon,
nonlinear, force-controlled, or state-dependent transmissions may require richer
typed mappings.

### Contact-based grasp state is simulator-sensitive

Geometry names and MuJoCo contact behavior affect grasp rules. A contact rule
must distinguish intended finger contacts, palm collisions, support contacts,
and compliant closure. Hardware may need a separate state provider with the same
semantic output.

### Grasp profiles need strict frame and version semantics

Two tools may look compatible while differing in TCP calibration, approach
direction, opening width, or palm geometry. Compatibility must be explicit and
must not be inferred from both tools being parallel-jaw grippers.

### Bimanual planning needs more than a serializable model

The schema can declare two effectors, but coordinated planning still requires
multi-goal IK, synchronization, shared-payload constraints, and bimanual failure
handling.

### Learned policies have separate action contracts

Existing learned policies assume fixed dimensions and raw gripper conventions.
They need codecs or a model-to-embodiment handshake even after the planner uses
the new specification.

### Large declarative files may be difficult to maintain

A fully expanded embodiment repeats many names and control values. Tooling should
generate an initial spec from MJCF and existing classes, format it canonically,
and report only fields that cannot be derived.

### No external standard covers the complete contract

MJCF covers mechanics, while SRDF, `ros2_control`, and MoveIt messages cover
parts of semantics and planning. This schema should borrow their concepts, but
it remains a MolmoSpaces format.

## Task Compatibility Plan

Compiled-model and runtime component parity does not imply that every task works
with every robot. The current tests prove that all configured embodiments load,
assemble, construct their views/controllers/sensors/kinematics, reset, and
produce controls. They do not prove task-policy compatibility or task success.

Known task-level blockers:

- Existing bimanual planner paths still select the first gripper and do not
  provide coordinated multi-target IK.
- Grasp libraries are selected outside `EmbodimentSpec`; profile compatibility
  and library manifests are not yet enforced before rollout.
- Camera layouts, learned-policy action dimensions, and raw gripper conventions
  remain robot-specific contracts for several tasks.
- RBY exposes online IK only, so task paths requiring parallel feasibility IK
  are unsupported until they gain a compatible fallback.
- Mobile Franka preserves the legacy unlimited-yaw actuator range behavior;
  the current position controller interprets its stored `[0, 0]` range as a
  hard limit and therefore cannot command useful yaw motion.
- Floating-robot placement contains legacy checks based on concrete robot class
  names. Those checks must consume configured placement semantics instead.

A task-by-robot support matrix is required. Each intended pair must be marked
`supported`, `unsupported`, or `pending`, with an explicit reason for every
non-supported entry. The matrix should cover pick, pick-and-place, open/close,
navigation, learned-policy, and bimanual task families.

Before a pair is marked supported, an automated smoke rollout must verify:

1. Scene assembly, camera resolution, placement, and collision filtering.
2. Robot reset, action-space dimensions, controller dispatch, and no-op behavior.
3. Required online or parallel IK and explicit end-effector selection.
4. Grasp-library/profile compatibility and task-specific sensor observations.
5. At least one complete short rollout without exceptions or invalid controls.
6. Task success on a deterministic fixture, or a documented reason that only a
   non-success smoke test is appropriate.

Current rollout evidence:

- Franka DROID pick: legacy and configured runtimes produced identical outcomes
  on all eight frozen seed-42 fixtures. Both succeeded on 4/8 episodes (50%),
  with matching persisted success flags and trajectory lengths.
- Configured DROID versus configured YAM pick: DROID succeeded on 4/8 frozen
  seed-42 fixtures (50%) and YAM succeeded on 0/8 (0%). All completed success
  and failure trajectories retained raw camera videos, and every case has clean
  and annotated five-view paired comparison videos.
- Generic experiment-config path with reach-normalized placement: configured
  DROID succeeded on 4/8 fixtures (50%) and configured YAM on 3/8 (37.5%).
  Comparison-only cameras remained external to the production camera config;
  all 128 raw views and 16 paired videos were verified.

Embodiment-specific classes must not be deprecated until all intended task pairs
pass this matrix. Unsupported Cartesian-product combinations do not need to be
made valid, but they must fail early with a clear compatibility error.

## Migration Strategy

1. Define the schema and validator without changing legacy runtime behavior. (Complete.)
2. Manually encode every current embodiment. (Complete for all current robot configs.)
3. Compare compiled MJCF names, dimensions, limits, frames, and initial states
   against existing classes. (Component parity complete.)
4. Add action-equivalence and short-rollout golden tests. (Task matrix pending.)
5. Introduce opt-in `ConfiguredRobot` and `ConfiguredRobotView` adapters. (Complete.)
6. Move one simple embodiment, then DROID and YAM, to the generic runtime.
   (Construction complete; task rollout migration pending.)
7. Add RBY, floating, mobile, and bimanual variants after the required schema
   primitives are proven. (Construction and component parity complete.)
8. Implement generic registered experiment configs and separate
   `--embodiment-spec` injection in `data_generation/main.py` without a
   Franka-specific default. (Complete for generic pick; serialized camera and
   experiment YAML schemas remain pending.)
9. Make the generic planner consume named groups and explicit end effectors.
   (Pending.)
10. Complete the task-by-robot rollout matrix with persisted success/failure
    videos. (Pending.)
11. Deprecate embodiment-specific classes only after full behavioral parity.

## Open Questions

- Should model operations remain in the embodiment or be baked into final MJCF
  resources?
- Should control profiles be separate reusable files?
- Which controller and kinematics algorithms belong in schema version 1?
- How should hardware-specific command mappings share an embodiment ID with
  simulation mappings?
- Should contact geometry be listed explicitly or selected by body ancestry?
- How are grasp-profile compatibility relationships reviewed and versioned?
- Should grasp profiles be embedded, shared by URI and hash, or both?
- How should multi-robot scenes namespace multiple copies of one embodiment?
- Should serialized experiment configs embed camera systems or reference them
  as reusable, versioned sidecar files?
- What exact equivalence threshold is required before replacing an existing
  robot class?
