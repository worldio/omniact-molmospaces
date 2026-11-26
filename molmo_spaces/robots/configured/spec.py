"""Data-only schema for configured robot embodiments."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal, Self, TypeAlias

from pydantic import AfterValidator, ConfigDict, Field, field_validator, model_validator

from molmo_spaces.configs.abstract_config import Config


def _ordered_pair(value: tuple[float, float]) -> tuple[float, float]:
    if value[0] > value[1]:
        raise ValueError("lower bound must not exceed upper bound")
    return value


def _nonzero_vector(value: tuple[float, float, float]) -> tuple[float, float, float]:
    if not any(value):
        raise ValueError("vector must be nonzero")
    return value


def _nonzero_quaternion(
    value: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    if not any(value):
        raise ValueError("quaternion must be nonzero")
    return value


def _rgba(value: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if any(component < 0.0 or component > 1.0 for component in value):
        raise ValueError("RGBA components must be between 0 and 1")
    return value


NonEmptyStr = Annotated[str, Field(min_length=1)]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]
NonNegativeFloat = Annotated[float, Field(ge=0.0)]
PositiveFloat = Annotated[float, Field(gt=0.0)]
Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
NonzeroVector3 = Annotated[Vector3, AfterValidator(_nonzero_vector)]
QuaternionWxyz = Annotated[tuple[float, float, float, float], AfterValidator(_nonzero_quaternion)]
Rgba = Annotated[tuple[float, float, float, float], AfterValidator(_rgba)]
Range = Annotated[tuple[float, float], AfterValidator(_ordered_pair)]


class SpecConfig(Config):
    """Strict base model for serialized embodiment data."""

    model_config = ConfigDict(
        extra="forbid",
        arbitrary_types_allowed=False,
        allow_inf_nan=False,
    )


class ModelNameRef(SpecConfig):
    name: NonEmptyStr
    absolute: bool = False


class FrameRef(SpecConfig):
    kind: Literal["body", "site"]
    name: ModelNameRef


class TransformSpec(SpecConfig):
    position: Vector3
    quaternion_wxyz: QuaternionWxyz


class ModelSpec(SpecConfig):
    source: Literal["builtin", "path"]
    resource_key: NonEmptyStr | None = None
    resource_version: NonEmptyStr | None = None
    root_path: Path | None = None
    mjcf_path: Path
    model_root_body: ModelNameRef
    pre_namespaced: bool = False
    prefix_names: bool = True

    @model_validator(mode="after")
    def validate_source(self) -> Self:
        if self.source == "builtin":
            if self.resource_key is None or self.resource_version is None:
                raise ValueError("builtin models require resource_key and resource_version")
        else:
            if self.resource_key is not None or self.resource_version is not None:
                raise ValueError("path models cannot declare resource metadata")
            if not self.mjcf_path.is_absolute() and self.root_path is None:
                raise ValueError("relative path models require root_path")
        if self.pre_namespaced and self.prefix_names:
            raise ValueError("pre_namespaced and prefix_names cannot both be true")
        return self


class ActuatorOverrideSpec(SpecConfig):
    actuator: ModelNameRef
    gain: float | None = None
    bias_position: float | None = None
    bias_velocity: float | None = None
    force_range: Range | None = None


class ControlProfileSpec(SpecConfig):
    controller_bindings: dict[NonEmptyStr, NonEmptyStr]
    actuator_overrides: dict[NonEmptyStr, ActuatorOverrideSpec]


class GravityCompensationSpec(SpecConfig):
    enabled: bool
    root_body: ModelNameRef | None
    include_root: bool = True
    include_descendants: bool = True

    @model_validator(mode="after")
    def validate_root(self) -> Self:
        if self.enabled and self.root_body is None:
            raise ValueError("enabled gravity compensation requires root_body")
        return self


class ActionNoiseSpec(SpecConfig):
    enabled: bool
    target_arm_groups: list[NonEmptyStr]
    target_base_group: NonEmptyStr | None
    action_scale_factor: NonNegativeFloat
    rotation_noise_scale: NonNegativeFloat
    max_tcp_position_noise: NonNegativeFloat
    max_tcp_rotation_noise: NonNegativeFloat
    base_action_scale_factor: NonNegativeFloat
    max_base_position_noise: NonNegativeFloat
    max_base_rotation_noise: NonNegativeFloat

    @field_validator("target_arm_groups")
    @classmethod
    def validate_unique_arm_groups(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)):
            raise ValueError("target_arm_groups must not contain duplicates")
        return value


class PlacementSpec(SpecConfig):
    reset_repositionable: bool
    runtime_mobility: Literal["fixed", "se2", "se3"]
    footprint_xy: Vector2 | None
    platform_size_xyz: Vector3 | None
    forward_offset: float = 0.0
    forward_offset_step: PositiveFloat = 0.02

    @model_validator(mode="after")
    def validate_sizes(self) -> Self:
        if self.footprint_xy is not None and any(size <= 0.0 for size in self.footprint_xy):
            raise ValueError("footprint_xy dimensions must be positive")
        if self.platform_size_xyz is not None and any(
            size <= 0.0 for size in self.platform_size_xyz
        ):
            raise ValueError("platform_size_xyz dimensions must be positive")
        return self


class RobotConfigSpec(SpecConfig):
    model: ModelSpec
    namespace: str
    initial_qpos: dict[NonEmptyStr, list[float]]
    initial_qpos_noise: dict[NonEmptyStr, list[NonNegativeFloat]] | None
    initial_qpos_limit_policy: dict[NonEmptyStr, Literal["strict", "allow_out_of_range"]] = Field(
        default_factory=dict
    )
    default_control_profile: NonEmptyStr
    control_profiles: dict[NonEmptyStr, ControlProfileSpec]
    gravity_compensation: GravityCompensationSpec
    action_noise: ActionNoiseSpec | None
    placement: PlacementSpec

    @field_validator("namespace")
    @classmethod
    def validate_namespace(cls, value: str) -> str:
        if value and not value.endswith("/"):
            raise ValueError("namespace must be empty or end with '/'")
        return value


OperationScope = Literal["robot_model", "scene"]


class AddSiteSpec(SpecConfig):
    kind: Literal["add_site"]
    scope: OperationScope
    parent_body: ModelNameRef
    name: ModelNameRef
    position: Vector3
    quaternion_wxyz: QuaternionWxyz | None = None
    xyaxes: tuple[float, float, float, float, float, float] | None = None
    group: NonNegativeInt | None = None
    rgba: Rgba | None = None

    @model_validator(mode="after")
    def validate_orientation(self) -> Self:
        if self.quaternion_wxyz is not None and self.xyaxes is not None:
            raise ValueError("quaternion_wxyz and xyaxes are mutually exclusive")
        if self.xyaxes is not None:
            x_axis = self.xyaxes[:3]
            y_axis = self.xyaxes[3:]
            if not any(x_axis) or not any(y_axis):
                raise ValueError("xyaxes must contain two nonzero axes")
        return self


class AddBodySpec(SpecConfig):
    kind: Literal["add_body"]
    scope: OperationScope
    name: ModelNameRef
    parent_body: ModelNameRef | None = None
    position: Vector3 = (0.0, 0.0, 0.0)
    quaternion_wxyz: QuaternionWxyz = (1.0, 0.0, 0.0, 0.0)
    use_base_pose: bool = False
    base_pose_z_for_2d: float = 0.0


class AddTextureSpec(SpecConfig):
    kind: Literal["add_texture"]
    scope: OperationScope
    name: ModelNameRef
    texture_type: Literal["2d", "cube"]
    file: Path
    randomized_files: list[Path] = Field(default_factory=list)

    @field_validator("file")
    @classmethod
    def validate_file(cls, value: Path) -> Path:
        if value.is_absolute() or ".." in value.parts:
            raise ValueError("texture file must be relative to the model resource root")
        return value

    @field_validator("randomized_files")
    @classmethod
    def validate_randomized_files(cls, value: list[Path]) -> list[Path]:
        if len(value) != len(set(value)):
            raise ValueError("randomized texture files must be unique")
        for path in value:
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(
                    "randomized texture files must be relative to the model resource root"
                )
        return value


class AddMaterialSpec(SpecConfig):
    kind: Literal["add_material"]
    scope: OperationScope
    name: ModelNameRef
    rgba: Rgba | None = None
    texture: ModelNameRef | None = None

    @model_validator(mode="after")
    def validate_appearance(self) -> Self:
        if self.rgba is None and self.texture is None:
            raise ValueError("material requires rgba, texture, or both")
        return self


class SpeckleRgbaMaterialsSpec(SpecConfig):
    kind: Literal["speckle_rgba_materials"]
    scope: Literal["scene"]
    model_root_body: ModelNameRef
    probability: Annotated[float, Field(ge=0.0, le=1.0)]
    texture_size: PositiveInt = 256
    noise_strength: NonNegativeFloat = 0.1
    num_blobs: NonNegativeInt = 80
    blob_size_range: tuple[PositiveInt, PositiveInt] = (5, 25)
    blob_variation: NonNegativeFloat = 0.1
    texture_name_suffix: NonEmptyStr = "_perturbed_tex"
    material_name_suffix: NonEmptyStr = "_perturbed"

    @field_validator("blob_size_range")
    @classmethod
    def validate_blob_size_range(cls, value: tuple[int, int]) -> tuple[int, int]:
        if value[0] >= value[1]:
            raise ValueError("blob_size_range must be strictly increasing")
        return value


GeomType = Literal["box", "capsule", "cylinder", "ellipsoid", "plane", "sphere"]


class AddGeomSpec(SpecConfig):
    kind: Literal["add_geom"]
    scope: OperationScope
    parent_body: ModelNameRef
    name: ModelNameRef
    geom_type: GeomType
    size: Annotated[tuple[float, ...], Field(min_length=1, max_length=3)]
    position: Vector3 = (0.0, 0.0, 0.0)
    quaternion_wxyz: QuaternionWxyz = (1.0, 0.0, 0.0, 0.0)
    material: ModelNameRef | None = None
    group: NonNegativeInt = 0
    rgba: Rgba | None = None

    @model_validator(mode="after")
    def validate_size(self) -> Self:
        expected_dimensions = {
            "sphere": 1,
            "capsule": 2,
            "cylinder": 2,
            "box": 3,
            "ellipsoid": 3,
            "plane": 3,
        }
        if len(self.size) != expected_dimensions[self.geom_type]:
            raise ValueError(
                f"{self.geom_type} geom size must have {expected_dimensions[self.geom_type]} values"
            )
        if any(size <= 0.0 for size in self.size):
            raise ValueError("geom sizes must be positive")
        return self


class AddJointSpec(SpecConfig):
    kind: Literal["add_joint"]
    scope: OperationScope
    parent_body: ModelNameRef
    name: ModelNameRef
    joint_type: Literal["ball", "free", "hinge", "slide"]
    position: Vector3 = (0.0, 0.0, 0.0)
    axis: NonzeroVector3 | None = None
    axis_frame: Literal["local", "world"] = "local"
    range: Range | None = None
    limited: bool | None = None
    damping: NonNegativeFloat = 0.0
    armature: NonNegativeFloat = 0.0
    stiffness: NonNegativeFloat = 0.0
    reference: float = 0.0
    reference_source: Literal["literal", "base_x", "base_y", "base_yaw"] = "literal"

    @model_validator(mode="after")
    def validate_joint(self) -> Self:
        if self.joint_type in {"ball", "free"} and self.range is not None:
            raise ValueError(f"{self.joint_type} joints cannot declare a scalar range")
        if self.range is not None and self.limited is False:
            raise ValueError("a joint with range cannot set limited=false")
        if self.reference_source != "literal" and self.joint_type in {"ball", "free"}:
            raise ValueError("ball and free joints cannot derive a scalar base-pose reference")
        return self


class AddActuatorSpec(SpecConfig):
    kind: Literal["add_actuator"]
    scope: OperationScope
    name: ModelNameRef
    actuator_type: Literal["general", "motor", "position", "velocity"] = "general"
    joint: ModelNameRef | None = None
    site: ModelNameRef | None = None
    reference_site: ModelNameRef | None = None
    gear: Annotated[tuple[float, ...], Field(min_length=1, max_length=6)] | None = None
    gain: float | None = None
    damping_ratio: NonNegativeFloat | None = None
    bias_position: float | None = None
    bias_velocity: float | None = None
    control_range: Range | None = None
    force_range: Range | None = None

    @model_validator(mode="after")
    def validate_transmission(self) -> Self:
        if (self.joint is None) == (self.site is None):
            raise ValueError("exactly one of joint or site must be provided")
        if self.joint is not None and self.reference_site is not None:
            raise ValueError("reference_site is only valid for site actuators")
        if self.site is not None:
            if self.reference_site is None:
                raise ValueError("site actuators require reference_site")
            if self.gear is not None and len(self.gear) != 6:
                raise ValueError("site actuator gear must have six values")
        if self.damping_ratio is not None and self.actuator_type != "position":
            raise ValueError("damping_ratio is only valid for position actuators")
        if self.damping_ratio is not None and self.bias_velocity is not None:
            raise ValueError("damping_ratio and bias_velocity are mutually exclusive")
        return self


class AddMocapBodySpec(SpecConfig):
    kind: Literal["add_mocap_body"]
    scope: OperationScope
    name: ModelNameRef
    parent_body: ModelNameRef | None = None
    initial_pose: TransformSpec
    use_base_pose: bool = False


class AddWeldSpec(SpecConfig):
    kind: Literal["add_weld"]
    scope: OperationScope
    body_a: ModelNameRef
    body_b: ModelNameRef
    solref: tuple[float, float] | None = None
    solimp: tuple[float, float, float, float, float] | None = None

    @model_validator(mode="after")
    def validate_bodies(self) -> Self:
        if self.body_a == self.body_b:
            raise ValueError("weld bodies must be different")
        return self


class AddPlatformSpec(SpecConfig):
    kind: Literal["add_platform"]
    scope: OperationScope
    parent_body: ModelNameRef
    name: ModelNameRef
    size_xyz: Vector3
    position: Vector3
    quaternion_wxyz: QuaternionWxyz = (1.0, 0.0, 0.0, 0.0)
    material: ModelNameRef | None = None
    rgba: Rgba | None = None
    group: NonNegativeInt = 0

    @field_validator("size_xyz")
    @classmethod
    def validate_platform_size(cls, value: Vector3) -> Vector3:
        if any(size <= 0.0 for size in value):
            raise ValueError("platform dimensions must be positive")
        return value


class AttachModelSpec(SpecConfig):
    kind: Literal["attach_model"]
    scope: OperationScope
    parent_body: ModelNameRef
    child_root_body: ModelNameRef
    child_pose: TransformSpec
    apply_namespace: bool = True
    use_base_pose: bool = False


class SetBodyPropertiesSpec(SpecConfig):
    kind: Literal["set_body_properties"]
    scope: OperationScope
    body: ModelNameRef
    position: Vector3 | None = None
    quaternion_wxyz: QuaternionWxyz | None = None
    gravity_compensation: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    mass: PositiveFloat | None = None
    center_of_mass: Vector3 | None = None
    inertia_diagonal: Vector3 | None = None

    @model_validator(mode="after")
    def validate_properties(self) -> Self:
        properties = (
            self.position,
            self.quaternion_wxyz,
            self.gravity_compensation,
            self.mass,
            self.center_of_mass,
            self.inertia_diagonal,
        )
        if all(value is None for value in properties):
            raise ValueError("at least one body property must be provided")
        if self.inertia_diagonal is not None and any(
            value <= 0.0 for value in self.inertia_diagonal
        ):
            raise ValueError("inertia_diagonal values must be positive")
        return self


ModelOperationSpec: TypeAlias = Annotated[
    AddSiteSpec
    | AddBodySpec
    | AddTextureSpec
    | AddMaterialSpec
    | SpeckleRgbaMaterialsSpec
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


class JointPositionControllerSpec(SpecConfig):
    kind: Literal["joint_position"]
    move_group: NonEmptyStr


class JointRelativePositionControllerSpec(SpecConfig):
    kind: Literal["joint_relative_position"]
    move_group: NonEmptyStr


class DirectActuatorControllerSpec(SpecConfig):
    kind: Literal["direct_actuator"]
    move_group: NonEmptyStr


class MocapPoseControllerSpec(SpecConfig):
    kind: Literal["mocap_pose"]
    move_group: NonEmptyStr


class PlanarPositionControllerSpec(SpecConfig):
    kind: Literal["planar_position"]
    move_group: NonEmptyStr


class RelativePlanarPositionControllerSpec(SpecConfig):
    kind: Literal["relative_planar_position"]
    move_group: NonEmptyStr


class LinearSynergyControllerSpec(SpecConfig):
    kind: Literal["linear_synergy"]
    move_group: NonEmptyStr
    command_dimension: PositiveInt
    matrix: list[list[float]]
    bias: list[float]
    output_limits: list[Range] | None = None
    command_limits: list[Range] | None = None
    state_to_command_matrix: list[list[float]] | None = None
    state_to_command_bias: list[float] | None = None

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if not self.matrix:
            raise ValueError("matrix must contain at least one output row")
        for row in self.matrix:
            if len(row) != self.command_dimension:
                raise ValueError("every matrix row must have command_dimension entries")
        if len(self.bias) != len(self.matrix):
            raise ValueError("bias length must equal the number of matrix rows")
        if self.output_limits is not None and len(self.output_limits) != len(self.matrix):
            raise ValueError("output_limits length must equal the number of matrix rows")
        if self.command_limits is not None and len(self.command_limits) != self.command_dimension:
            raise ValueError("command_limits length must equal command_dimension")
        if (self.state_to_command_matrix is None) != (self.state_to_command_bias is None):
            raise ValueError(
                "state_to_command_matrix and state_to_command_bias must be declared together"
            )
        if self.state_to_command_matrix is not None:
            if len(self.state_to_command_matrix) != self.command_dimension:
                raise ValueError("state_to_command_matrix must have command_dimension rows")
            if any(len(row) != len(self.matrix) for row in self.state_to_command_matrix):
                raise ValueError(
                    "state_to_command_matrix rows must match the synergy output dimension"
                )
            if len(self.state_to_command_bias) != self.command_dimension:
                raise ValueError("state_to_command_bias must match command_dimension")
        return self


ControllerSpec: TypeAlias = Annotated[
    JointPositionControllerSpec
    | JointRelativePositionControllerSpec
    | DirectActuatorControllerSpec
    | MocapPoseControllerSpec
    | PlanarPositionControllerSpec
    | RelativePlanarPositionControllerSpec
    | LinearSynergyControllerSpec,
    Field(discriminator="kind"),
]


class DampedLeastSquaresKinematicsSpec(SpecConfig):
    kind: Literal["damped_least_squares"]
    online_backend: Literal["numpy"]
    batch_backend: Literal["warp", "sequential"] | None
    optimizable_groups: list[NonEmptyStr]
    max_iterations: PositiveInt
    convergence_tolerance: PositiveFloat
    success_tolerance: PositiveFloat
    damping: NonNegativeFloat
    integration_step: PositiveFloat
    online_max_iterations: PositiveInt | None = None
    online_success_tolerance: PositiveFloat | None = None

    @field_validator("optimizable_groups")
    @classmethod
    def validate_optimizable_groups(cls, value: list[str]) -> list[str]:
        if not value:
            raise ValueError("optimizable_groups must not be empty")
        if len(value) != len(set(value)):
            raise ValueError("optimizable_groups must not contain duplicates")
        return value


class FloatingFrameKinematicsSpec(SpecConfig):
    kind: Literal["floating_frame"]
    base_group: NonEmptyStr
    tcp_frame: FrameRef


class NoKinematicsSpec(SpecConfig):
    kind: Literal["none"]


KinematicsSpec: TypeAlias = Annotated[
    DampedLeastSquaresKinematicsSpec | FloatingFrameKinematicsSpec | NoKinematicsSpec,
    Field(discriminator="kind"),
]


class JointStateSensorSpec(SpecConfig):
    kind: Literal["joint_state"]
    uuid: NonEmptyStr = "joint_state"
    move_groups: list[NonEmptyStr]
    include_position: bool = True
    include_velocity: bool = True

    @model_validator(mode="after")
    def validate_outputs(self) -> Self:
        if not self.move_groups:
            raise ValueError("move_groups must not be empty")
        if len(self.move_groups) != len(set(self.move_groups)):
            raise ValueError("move_groups must not contain duplicates")
        if not self.include_position and not self.include_velocity:
            raise ValueError("joint state sensor must include position or velocity")
        return self


class FramePoseSensorSpec(SpecConfig):
    kind: Literal["frame_pose"]
    uuid: NonEmptyStr = "frame_pose"
    frame: FrameRef
    relative_to_group: NonEmptyStr | None = None
    output_format: Literal["matrix_4x4", "xyz_quaternion_wxyz"] = "xyz_quaternion_wxyz"


class ContactStateSensorSpec(SpecConfig):
    kind: Literal["contact_state"]
    uuid: NonEmptyStr = "contact_state"
    end_effector: NonEmptyStr


class ApertureSensorSpec(SpecConfig):
    kind: Literal["aperture"]
    uuid: NonEmptyStr = "aperture"
    move_group: NonEmptyStr


class TaskObjectBilateralContactSensorSpec(SpecConfig):
    kind: Literal["task_object_bilateral_contact"]
    uuid: NonEmptyStr
    finger_a_body: ModelNameRef
    finger_b_body: ModelNameRef
    task_object_keys: Annotated[list[NonEmptyStr], Field(min_length=1)]


SensorSpec: TypeAlias = Annotated[
    JointStateSensorSpec
    | FramePoseSensorSpec
    | ContactStateSensorSpec
    | ApertureSensorSpec
    | TaskObjectBilateralContactSensorSpec,
    Field(discriminator="kind"),
]


class RobotSpec(SpecConfig):
    model_operations: list[ModelOperationSpec]
    controllers: dict[NonEmptyStr, ControllerSpec]
    kinematics: KinematicsSpec
    sensors: list[SensorSpec]

    @field_validator("sensors")
    @classmethod
    def validate_sensor_uuids(cls, value: list[SensorSpec]) -> list[SensorSpec]:
        uuids = [sensor.uuid for sensor in value]
        if len(uuids) != len(set(uuids)):
            raise ValueError("sensor UUIDs must be unique")
        return value


class JointTermSpec(SpecConfig):
    joint: ModelNameRef
    coefficient: float
    transform: Literal["identity", "absolute"] = "identity"


class JointExpressionApertureSpec(SpecConfig):
    kind: Literal["joint_expression"]
    terms: list[JointTermSpec]
    offset: float = 0.0
    clamp: Range | None = None

    @field_validator("terms")
    @classmethod
    def validate_terms(cls, value: list[JointTermSpec]) -> list[JointTermSpec]:
        if not value:
            raise ValueError("joint expression must contain at least one term")
        joints = [(term.joint.name, term.joint.absolute) for term in value]
        if len(joints) != len(set(joints)):
            raise ValueError("joint expression cannot repeat a joint")
        return value


class GeomDistanceApertureSpec(SpecConfig):
    kind: Literal["geom_distance"]
    geom_a: ModelNameRef
    geom_b: ModelNameRef
    maximum_distance: PositiveFloat
    query_limit: PositiveFloat | None = None

    @model_validator(mode="after")
    def validate_geoms(self) -> Self:
        if self.geom_a == self.geom_b:
            raise ValueError("geom_a and geom_b must be different")
        if self.query_limit is not None and self.query_limit < self.maximum_distance:
            raise ValueError("query_limit must not be smaller than maximum_distance")
        return self


ApertureSpec: TypeAlias = Annotated[
    JointExpressionApertureSpec | GeomDistanceApertureSpec,
    Field(discriminator="kind"),
]


class LinearStateMappingSpec(SpecConfig):
    input_dimension: PositiveInt
    matrix: list[list[float]]
    bias: list[float]

    @model_validator(mode="after")
    def validate_matrix(self) -> Self:
        if not self.matrix:
            raise ValueError("state mapping matrix must not be empty")
        if any(len(row) != self.input_dimension for row in self.matrix):
            raise ValueError("state mapping rows must match input_dimension")
        if len(self.bias) != len(self.matrix):
            raise ValueError("state mapping bias must match output dimension")
        return self


class JointMoveGroupSpec(SpecConfig):
    kind: Literal["joint"]
    role: Literal["arm", "torso", "head", "other"]
    side: Literal["left", "right", "center"]
    parent_group: NonEmptyStr
    root_body: ModelNameRef
    joint_names: Annotated[list[ModelNameRef], Field(min_length=1)]
    actuator_names: list[ModelNameRef]
    leaf_frame: FrameRef | None


class GripperMoveGroupSpec(SpecConfig):
    kind: Literal["gripper"]
    role: Literal["tool"]
    side: Literal["left", "right", "center"]
    parent_group: NonEmptyStr
    root_body: ModelNameRef
    joint_names: Annotated[list[ModelNameRef], Field(min_length=1)]
    actuator_names: Annotated[list[ModelNameRef], Field(min_length=1)]
    leaf_frame: FrameRef
    open_command: Annotated[list[float], Field(min_length=1)]
    closed_command: Annotated[list[float], Field(min_length=1)]
    aperture: ApertureSpec | None
    state_mapping: LinearStateMappingSpec | None = None

    @model_validator(mode="after")
    def validate_commands(self) -> Self:
        if len(self.open_command) != len(self.closed_command):
            raise ValueError("open_command and closed_command dimensions must match")
        if len(self.open_command) != len(self.actuator_names):
            raise ValueError("gripper command dimension must match its actuator count")
        if self.state_mapping is not None and len(self.state_mapping.matrix) != len(
            self.joint_names
        ):
            raise ValueError("gripper state mapping output must match its joint count")
        return self


class MocapBaseMoveGroupSpec(SpecConfig):
    kind: Literal["mocap_base"]
    role: Literal["base"]
    side: Literal["left", "right", "center"]
    body: ModelNameRef
    runtime_mobile: bool


class FreeJointBaseMoveGroupSpec(SpecConfig):
    kind: Literal["free_joint_base"]
    role: Literal["base"]
    side: Literal["left", "right", "center"]
    body: ModelNameRef
    free_joint: ModelNameRef
    additional_joint_names: list[ModelNameRef] = Field(default_factory=list)
    actuator_names: list[ModelNameRef] = Field(default_factory=list)
    mocap_target_body: ModelNameRef | None = None
    floating: bool = False
    runtime_mobile: bool = True


class PlanarBaseMoveGroupSpec(SpecConfig):
    kind: Literal["planar_base"]
    role: Literal["base"]
    side: Literal["left", "right", "center"]
    root_body: ModelNameRef
    joint_names: Annotated[list[ModelNameRef], Field(min_length=3, max_length=3)]
    actuator_names: Annotated[list[ModelNameRef], Field(min_length=3, max_length=3)]
    world_frame: FrameRef
    base_frame: FrameRef
    runtime_mobile: bool = True


MoveGroupSpec: TypeAlias = Annotated[
    JointMoveGroupSpec
    | GripperMoveGroupSpec
    | MocapBaseMoveGroupSpec
    | FreeJointBaseMoveGroupSpec
    | PlanarBaseMoveGroupSpec,
    Field(discriminator="kind"),
]


class EndEffectorSpec(SpecConfig):
    arm_group: NonEmptyStr | None
    tool_group: NonEmptyStr
    tcp_frame: FrameRef
    grasp_frame: FrameRef
    grasp_profile_id: NonEmptyStr


class RobotViewSpec(SpecConfig):
    base_group: NonEmptyStr
    runtime_name: NonEmptyStr | None = None
    move_groups: dict[NonEmptyStr, MoveGroupSpec]
    end_effectors: dict[NonEmptyStr, EndEffectorSpec]

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        if self.base_group not in self.move_groups:
            raise ValueError(f"base_group {self.base_group!r} is not a move group")
        base = self.move_groups[self.base_group]
        if not isinstance(
            base,
            (MocapBaseMoveGroupSpec, FreeJointBaseMoveGroupSpec, PlanarBaseMoveGroupSpec),
        ):
            raise ValueError(f"base_group {self.base_group!r} is not a base move group")

        parents: dict[str, str] = {}
        joint_owners: dict[tuple[str, bool], str] = {}
        actuator_owners: dict[tuple[str, bool], str] = {}
        for group_id, group in self.move_groups.items():
            if isinstance(group, (JointMoveGroupSpec, GripperMoveGroupSpec)):
                if group.parent_group not in self.move_groups:
                    raise ValueError(
                        f"move group {group_id!r} references unknown parent {group.parent_group!r}"
                    )
                if group.parent_group == group_id:
                    raise ValueError(f"move group {group_id!r} cannot parent itself")
                parents[group_id] = group.parent_group
                joint_names = group.joint_names
                actuator_names = group.actuator_names
            elif isinstance(group, PlanarBaseMoveGroupSpec):
                joint_names = group.joint_names
                actuator_names = group.actuator_names
            elif isinstance(group, FreeJointBaseMoveGroupSpec):
                joint_names = [group.free_joint, *group.additional_joint_names]
                actuator_names = group.actuator_names
            else:
                joint_names = []
                actuator_names = []

            for reference in joint_names:
                key = (reference.name, reference.absolute)
                if key in joint_owners:
                    raise ValueError(
                        f"joint {reference.name!r} belongs to conflicting move groups "
                        f"{joint_owners[key]!r} and {group_id!r}"
                    )
                joint_owners[key] = group_id
            for reference in actuator_names:
                key = (reference.name, reference.absolute)
                if key in actuator_owners:
                    raise ValueError(
                        f"actuator {reference.name!r} belongs to conflicting move groups "
                        f"{actuator_owners[key]!r} and {group_id!r}"
                    )
                actuator_owners[key] = group_id

        for group_id in parents:
            seen = {group_id}
            current = group_id
            while current in parents:
                current = parents[current]
                if current in seen:
                    raise ValueError(f"move group parent cycle includes {current!r}")
                seen.add(current)

        for end_effector_id, end_effector in self.end_effectors.items():
            if end_effector.arm_group is not None:
                if end_effector.arm_group not in self.move_groups:
                    raise ValueError(
                        f"end effector {end_effector_id!r} references unknown arm group "
                        f"{end_effector.arm_group!r}"
                    )
                arm = self.move_groups[end_effector.arm_group]
                if not isinstance(arm, JointMoveGroupSpec) or arm.role != "arm":
                    raise ValueError(
                        f"end effector {end_effector_id!r} arm_group must have role 'arm'"
                    )
            if end_effector.tool_group not in self.move_groups:
                raise ValueError(
                    f"end effector {end_effector_id!r} references unknown tool group "
                    f"{end_effector.tool_group!r}"
                )
            if not isinstance(self.move_groups[end_effector.tool_group], GripperMoveGroupSpec):
                raise ValueError(f"end effector {end_effector_id!r} tool_group must be a gripper")
        return self


class ContactGroupSpec(SpecConfig):
    geom_names: list[ModelNameRef]
    body_names: list[ModelNameRef] = Field(default_factory=list)
    include_descendants_of: list[ModelNameRef] = Field(default_factory=list)
    exclude_descendants_of: list[ModelNameRef] = Field(default_factory=list)


class GraspRuleSpec(SpecConfig):
    required_contact_groups: list[NonEmptyStr]
    forbidden_contact_groups: list[NonEmptyStr] = Field(default_factory=list)
    minimum_contact_count: NonNegativeInt = 1
    aperture_threshold: NonNegativeFloat | None = None

    @model_validator(mode="after")
    def validate_groups(self) -> Self:
        required = set(self.required_contact_groups)
        forbidden = set(self.forbidden_contact_groups)
        if len(required) != len(self.required_contact_groups):
            raise ValueError("required_contact_groups must not contain duplicates")
        if len(forbidden) != len(self.forbidden_contact_groups):
            raise ValueError("forbidden_contact_groups must not contain duplicates")
        overlap = required & forbidden
        if overlap:
            raise ValueError(
                f"contact groups cannot be both required and forbidden: {sorted(overlap)}"
            )
        return self


class GraspProfileSpec(SpecConfig):
    profile_version: NonEmptyStr
    tool_type: Literal["parallel_jaw", "suction", "multi_finger", "magnetic"]
    grasp_frame: FrameRef
    approach_axis_in_grasp_frame: NonzeroVector3
    closing_axis_in_grasp_frame: NonzeroVector3 | None
    symmetries: list[TransformSpec]
    open_configuration: list[float]
    closed_configuration: list[float]
    contact_groups: dict[NonEmptyStr, ContactGroupSpec]
    grasp_rule: GraspRuleSpec
    release_rule: GraspRuleSpec

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if len(self.open_configuration) != len(self.closed_configuration):
            raise ValueError("open_configuration and closed_configuration dimensions must match")
        contact_group_ids = set(self.contact_groups)
        for rule_name, rule in (
            ("grasp_rule", self.grasp_rule),
            ("release_rule", self.release_rule),
        ):
            referenced = set(rule.required_contact_groups) | set(rule.forbidden_contact_groups)
            unknown = referenced - contact_group_ids
            if unknown:
                raise ValueError(
                    f"{rule_name} references unknown contact groups: {sorted(unknown)}"
                )
        return self


def _actuator_names(group: MoveGroupSpec) -> list[ModelNameRef]:
    if isinstance(
        group,
        (
            JointMoveGroupSpec,
            GripperMoveGroupSpec,
            FreeJointBaseMoveGroupSpec,
            PlanarBaseMoveGroupSpec,
        ),
    ):
        return group.actuator_names
    return []


class EmbodimentSpec(SpecConfig):
    schema_version: Literal[1]
    embodiment_id: NonEmptyStr
    embodiment_version: NonEmptyStr
    robot_config: RobotConfigSpec
    robot: RobotSpec
    robot_view: RobotViewSpec
    grasp_profiles: dict[NonEmptyStr, GraspProfileSpec]

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        groups = self.robot_view.move_groups
        controllers = self.robot.controllers

        if self.robot_config.default_control_profile not in self.robot_config.control_profiles:
            raise ValueError(
                "default_control_profile references unknown profile "
                f"{self.robot_config.default_control_profile!r}"
            )

        for group_id, qpos in self.robot_config.initial_qpos.items():
            if group_id not in groups:
                raise ValueError(f"initial_qpos references unknown move group {group_id!r}")
            if isinstance(groups[group_id], FreeJointBaseMoveGroupSpec):
                if len(qpos) < 7:
                    raise ValueError(
                        f"initial_qpos[{group_id!r}] must contain at least seven free-joint values"
                    )
                if not any(qpos[3:7]):
                    raise ValueError(
                        f"initial_qpos[{group_id!r}] contains a zero free-joint quaternion"
                    )

        if self.robot_config.initial_qpos_noise is not None:
            for group_id, noise in self.robot_config.initial_qpos_noise.items():
                if group_id not in groups:
                    raise ValueError(
                        f"initial_qpos_noise references unknown move group {group_id!r}"
                    )
                if group_id not in self.robot_config.initial_qpos:
                    raise ValueError(
                        f"initial_qpos_noise[{group_id!r}] has no matching initial_qpos"
                    )
                expected = len(self.robot_config.initial_qpos[group_id])
                if len(noise) != expected:
                    raise ValueError(
                        f"initial_qpos_noise[{group_id!r}] has dimension {len(noise)}; "
                        f"initial_qpos has dimension {expected}"
                    )

        for group_id in self.robot_config.initial_qpos_limit_policy:
            if group_id not in self.robot_config.initial_qpos:
                raise ValueError(
                    f"initial_qpos_limit_policy references group {group_id!r} without initial_qpos"
                )

        action_noise = self.robot_config.action_noise
        if action_noise is not None:
            for group_id in action_noise.target_arm_groups:
                if group_id not in groups:
                    raise ValueError(f"action_noise references unknown arm group {group_id!r}")
                group = groups[group_id]
                if not isinstance(group, JointMoveGroupSpec) or group.role != "arm":
                    raise ValueError(f"action_noise target {group_id!r} is not an arm move group")
            if action_noise.target_base_group is not None:
                if action_noise.target_base_group != self.robot_view.base_group:
                    raise ValueError(
                        "action_noise target_base_group must equal robot_view.base_group"
                    )

        for controller_id, controller in controllers.items():
            if controller.move_group not in groups:
                raise ValueError(
                    f"controller {controller_id!r} references unknown move group "
                    f"{controller.move_group!r}"
                )
            group = groups[controller.move_group]
            if isinstance(controller, MocapPoseControllerSpec) and not isinstance(
                group, (MocapBaseMoveGroupSpec, FreeJointBaseMoveGroupSpec)
            ):
                raise ValueError(
                    f"mocap controller {controller_id!r} requires a mocap or free-joint base"
                )
            if isinstance(
                controller,
                (PlanarPositionControllerSpec, RelativePlanarPositionControllerSpec),
            ) and not isinstance(group, (PlanarBaseMoveGroupSpec, FreeJointBaseMoveGroupSpec)):
                raise ValueError(
                    f"planar controller {controller_id!r} requires a planar or free-joint base"
                )
            if isinstance(controller, LinearSynergyControllerSpec):
                output_dimension = len(_actuator_names(group))
                if len(controller.matrix) != output_dimension:
                    raise ValueError(
                        f"linear synergy controller {controller_id!r} has output dimension "
                        f"{len(controller.matrix)}; expected {output_dimension}"
                    )
            elif (
                isinstance(
                    controller,
                    (
                        JointPositionControllerSpec,
                        JointRelativePositionControllerSpec,
                        DirectActuatorControllerSpec,
                    ),
                )
                and not _actuator_names(group)
                and not (
                    isinstance(controller, DirectActuatorControllerSpec)
                    and isinstance(group, FreeJointBaseMoveGroupSpec)
                    and group.mocap_target_body is not None
                )
            ):
                raise ValueError(
                    f"controller {controller_id!r} targets a move group without actuators"
                )

        declared_actuators = {
            (actuator.name, actuator.absolute)
            for group in groups.values()
            for actuator in _actuator_names(group)
        }
        for profile_id, profile in self.robot_config.control_profiles.items():
            for group_id, controller_id in profile.controller_bindings.items():
                if group_id not in groups:
                    raise ValueError(
                        f"control profile {profile_id!r} binds unknown move group {group_id!r}"
                    )
                if controller_id not in controllers:
                    raise ValueError(
                        f"control profile {profile_id!r} references unknown controller "
                        f"{controller_id!r}"
                    )
                if controllers[controller_id].move_group != group_id:
                    raise ValueError(
                        f"control profile {profile_id!r} binds controller {controller_id!r} "
                        f"to {group_id!r}, but it targets "
                        f"{controllers[controller_id].move_group!r}"
                    )
            for override_id, override in profile.actuator_overrides.items():
                actuator_key = (override.actuator.name, override.actuator.absolute)
                if actuator_key not in declared_actuators:
                    raise ValueError(
                        f"actuator override {override_id!r} in control profile {profile_id!r} "
                        f"references undeclared actuator {override.actuator.name!r}"
                    )

        kinematics = self.robot.kinematics
        if isinstance(kinematics, DampedLeastSquaresKinematicsSpec):
            for group_id in kinematics.optimizable_groups:
                if group_id not in groups:
                    raise ValueError(f"kinematics references unknown move group {group_id!r}")
                if isinstance(groups[group_id], MocapBaseMoveGroupSpec):
                    raise ValueError(f"kinematics cannot optimize mocap move group {group_id!r}")
        elif isinstance(kinematics, FloatingFrameKinematicsSpec):
            if kinematics.base_group not in groups:
                raise ValueError(
                    f"floating kinematics references unknown base group {kinematics.base_group!r}"
                )
            if kinematics.base_group != self.robot_view.base_group:
                raise ValueError("floating kinematics base_group must equal robot_view.base_group")
            if not isinstance(groups[kinematics.base_group], FreeJointBaseMoveGroupSpec):
                raise ValueError("floating kinematics requires a free-joint base group")

        for sensor in self.robot.sensors:
            if isinstance(sensor, JointStateSensorSpec):
                unknown = set(sensor.move_groups) - set(groups)
                if unknown:
                    raise ValueError(
                        f"joint state sensor {sensor.uuid!r} references unknown move groups: "
                        f"{sorted(unknown)}"
                    )
            elif isinstance(sensor, FramePoseSensorSpec):
                if sensor.relative_to_group is not None and sensor.relative_to_group not in groups:
                    raise ValueError(
                        f"frame pose sensor {sensor.uuid!r} references unknown relative group "
                        f"{sensor.relative_to_group!r}"
                    )
            elif isinstance(sensor, ContactStateSensorSpec):
                if sensor.end_effector not in self.robot_view.end_effectors:
                    raise ValueError(
                        f"contact state sensor {sensor.uuid!r} references unknown end effector "
                        f"{sensor.end_effector!r}"
                    )
            elif isinstance(sensor, ApertureSensorSpec):
                if sensor.move_group not in groups:
                    raise ValueError(
                        f"aperture sensor {sensor.uuid!r} references unknown move group "
                        f"{sensor.move_group!r}"
                    )
                group = groups[sensor.move_group]
                if not isinstance(group, GripperMoveGroupSpec) or group.aperture is None:
                    raise ValueError(
                        f"aperture sensor {sensor.uuid!r} requires a gripper with aperture data"
                    )

        for end_effector_id, end_effector in self.robot_view.end_effectors.items():
            if end_effector.grasp_profile_id not in self.grasp_profiles:
                raise ValueError(
                    f"end effector {end_effector_id!r} references unknown grasp profile "
                    f"{end_effector.grasp_profile_id!r}"
                )
            profile = self.grasp_profiles[end_effector.grasp_profile_id]
            if end_effector.grasp_frame != profile.grasp_frame:
                raise ValueError(
                    f"end effector {end_effector_id!r} grasp_frame does not match "
                    f"profile {end_effector.grasp_profile_id!r}"
                )
            tool = groups[end_effector.tool_group]
            if not isinstance(tool, GripperMoveGroupSpec):
                continue
            command_dimension = len(tool.open_command)
            if len(profile.open_configuration) != command_dimension:
                raise ValueError(
                    f"grasp profile {end_effector.grasp_profile_id!r} configuration dimension "
                    f"{len(profile.open_configuration)} does not match tool group "
                    f"{end_effector.tool_group!r} command dimension {command_dimension}"
                )
        return self
