"""MuJoCo assembly for configured robot embodiments."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Literal, TypeVar

import mujoco
import numpy as np
from mujoco import MjSpec
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation as R

from molmo_spaces.molmo_spaces_constants import (
    DATA_TYPE_TO_SOURCE_TO_VERSION,
    get_robot_path,
)
from molmo_spaces.robots.configured.spec import (
    ActuatorOverrideSpec,
    AddActuatorSpec,
    AddBodySpec,
    AddGeomSpec,
    AddJointSpec,
    AddMaterialSpec,
    AddMocapBodySpec,
    AddPlatformSpec,
    AddSiteSpec,
    AddTextureSpec,
    AddWeldSpec,
    AttachModelSpec,
    EmbodimentSpec,
    MocapBaseMoveGroupSpec,
    ModelNameRef,
    ModelOperationSpec,
    SetBodyPropertiesSpec,
    SpeckleRgbaMaterialsSpec,
)

_GEOM_TYPES = {
    "box": mujoco.mjtGeom.mjGEOM_BOX,
    "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    "cylinder": mujoco.mjtGeom.mjGEOM_CYLINDER,
    "ellipsoid": mujoco.mjtGeom.mjGEOM_ELLIPSOID,
    "plane": mujoco.mjtGeom.mjGEOM_PLANE,
    "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
}

_JOINT_TYPES = {
    "ball": mujoco.mjtJoint.mjJNT_BALL,
    "free": mujoco.mjtJoint.mjJNT_FREE,
    "hinge": mujoco.mjtJoint.mjJNT_HINGE,
    "slide": mujoco.mjtJoint.mjJNT_SLIDE,
}

_TEXTURE_TYPES = {
    "2d": mujoco.mjtTexture.mjTEXTURE_2D,
    "cube": mujoco.mjtTexture.mjTEXTURE_CUBE,
}

_T = TypeVar("_T")
_Scope = Literal["robot_model", "scene"]


@dataclass(frozen=True)
class _CallPose:
    position: tuple[float, float, float]
    quaternion: tuple[float, float, float, float]
    position_was_2d: bool


class MjcfAssembler:
    """Interpret an :class:`EmbodimentSpec` into MuJoCo specifications."""

    def __init__(self, embodiment_spec: EmbodimentSpec) -> None:
        self.embodiment_spec = embodiment_spec
        self._model_path = self._resolve_model_path()

    @property
    def model_path(self) -> Path:
        """Return the deterministic, validated MJCF path."""
        return self._model_path

    def load_robot_spec(self, *, strip_meshes: bool = False) -> MjSpec:
        """Load the robot MJCF and apply all pre-attachment operations."""
        try:
            robot_spec = MjSpec.from_file(str(self._model_path))
        except (OSError, ValueError) as error:
            raise ValueError(f"Could not load robot MJCF {self._model_path}: {error}") from error

        # Generated MJCFs store portable ${ROOT} tokens; materialize them to this machine's paths.
        from molmo_spaces.utils.asset_tokens import resolve_spec_asset_files

        resolve_spec_asset_files(robot_spec)

        if strip_meshes:
            self._strip_mesh_geometries(robot_spec)

        for operation in self.embodiment_spec.robot.model_operations:
            if operation.scope != "robot_model":
                continue
            if isinstance(operation, AttachModelSpec):
                raise ValueError("attach_model only supports scope='scene'")
            self._apply_operation(
                robot_spec,
                operation,
                expected_scope="robot_model",
                prefix="",
            )

        root_name = self.embodiment_spec.robot_config.model.model_root_body.name
        self._require(
            robot_spec.body(root_name),
            "body",
            root_name,
            "configured model_root_body after robot_model operations",
        )
        return robot_spec

    def add_robot_to_scene(
        self,
        spec: MjSpec,
        prefix: str,
        pos: Sequence[float],
        quat: Sequence[float],
        randomize_textures: bool = False,
        strip_meshes: bool = False,
    ) -> None:
        """Assemble this embodiment into a destination scene specification."""
        if not isinstance(prefix, str):
            raise ValueError("prefix must be a string")

        call_pose = self._validate_pose(pos, quat)
        operations = self.embodiment_spec.robot.model_operations
        invalid_attachments = [
            operation
            for operation in operations
            if isinstance(operation, AttachModelSpec) and operation.scope != "scene"
        ]
        if invalid_attachments:
            raise ValueError("attach_model only supports scope='scene'")

        attachments = [
            operation
            for operation in operations
            if isinstance(operation, AttachModelSpec) and operation.scope == "scene"
        ]
        if len(attachments) != 1:
            raise ValueError(
                "Scene assembly requires exactly one scope='scene' attach_model operation; "
                f"found {len(attachments)}"
            )
        self._validate_attachment_policy(attachments[0])

        robot_spec = self.load_robot_spec(strip_meshes=strip_meshes)
        base_mocap_name = self._base_mocap_name(prefix)

        for operation in operations:
            if operation.scope != "scene":
                continue
            if isinstance(operation, AttachModelSpec):
                self._attach_model(spec, robot_spec, operation, prefix, call_pose)
            elif isinstance(operation, SpeckleRgbaMaterialsSpec):
                if randomize_textures:
                    self._speckle_rgba_materials(
                        spec,
                        robot_spec,
                        operation,
                        prefix,
                    )
            else:
                self._apply_operation(
                    spec,
                    operation,
                    expected_scope="scene",
                    prefix=prefix,
                    base_mocap_name=base_mocap_name,
                    call_pose=call_pose,
                    randomize_textures=randomize_textures,
                )

        if base_mocap_name is not None:
            base = self._require(
                spec.body(base_mocap_name),
                "body",
                base_mocap_name,
                "configured mocap base",
            )
            if not base.mocap:
                raise ValueError(
                    f"Configured mocap base body {base_mocap_name!r} is not a MuJoCo mocap body"
                )
            base.pos = call_pose.position
            base.quat = call_pose.quaternion

    def apply_control_profile(self, spec: MjSpec, prefix: str, profile_name: str) -> None:
        """Apply one named control profile's actuator overrides idempotently."""
        profiles = self.embodiment_spec.robot_config.control_profiles
        if profile_name not in profiles:
            raise ValueError(
                f"Unknown control profile {profile_name!r}; available profiles: {sorted(profiles)}"
            )

        resolved: list[tuple[str, mujoco.MjsActuator, ActuatorOverrideSpec]] = []
        actuator_owners: dict[str, str] = {}
        for override_id, override in profiles[profile_name].actuator_overrides.items():
            actuator_name = self._resolve_name(override.actuator, "scene", prefix)
            if actuator_name in actuator_owners:
                raise ValueError(
                    f"Control profile {profile_name!r} overrides actuator {actuator_name!r} "
                    f"more than once ({actuator_owners[actuator_name]!r} and {override_id!r})"
                )
            actuator_owners[actuator_name] = override_id
            actuator = self._require(
                spec.actuator(actuator_name),
                "actuator",
                actuator_name,
                f"control profile {profile_name!r} override {override_id!r}",
            )
            self._validate_strict_range(
                override.force_range,
                f"control profile {profile_name!r} force_range for {actuator_name!r}",
            )
            resolved.append((override_id, actuator, override))

        for _, actuator, override in resolved:
            if override.gain is not None:
                actuator.gainprm[0] = override.gain
            if override.bias_position is not None or override.bias_velocity is not None:
                actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
            if override.bias_position is not None:
                actuator.biasprm[1] = override.bias_position
            if override.bias_velocity is not None:
                actuator.biasprm[2] = override.bias_velocity
            if override.force_range is not None:
                actuator.forcelimited = 1
                actuator.forcerange[:] = override.force_range

    def apply_gravity_compensation(self, spec: MjSpec, prefix: str) -> None:
        """Apply configured gravity compensation to the selected body subtree."""
        gravity = self.embodiment_spec.robot_config.gravity_compensation
        if not gravity.enabled:
            return
        if gravity.root_body is None:
            raise ValueError("Enabled gravity compensation has no root_body")

        root_name = self._resolve_name(gravity.root_body, "scene", prefix)
        root = self._require(
            spec.body(root_name),
            "body",
            root_name,
            "gravity compensation root",
        )
        bodies = [root] if gravity.include_root else []
        if gravity.include_descendants:
            bodies.extend(root.find_all("body"))
        for body in bodies:
            body.gravcomp = 1.0

    def apply_control_overrides(
        self,
        spec: MjSpec,
        prefix: str,
        profile_name: str | None = None,
    ) -> None:
        """Apply a control profile and gravity compensation idempotently."""
        selected_profile = (
            self.embodiment_spec.robot_config.default_control_profile
            if profile_name is None
            else profile_name
        )
        self.apply_control_profile(spec, prefix, selected_profile)
        self.apply_gravity_compensation(spec, prefix)

    def _resolve_model_path(self) -> Path:
        model = self.embodiment_spec.robot_config.model
        if model.source == "builtin":
            resource_key = model.resource_key
            resource_version = model.resource_version
            if resource_key is None or resource_version is None:
                raise ValueError("Builtin model is missing resource_key or resource_version")
            expected_version = DATA_TYPE_TO_SOURCE_TO_VERSION.get("robots", {}).get(resource_key)
            if expected_version is None:
                raise ValueError(f"Unknown builtin robot resource {resource_key!r}")
            if resource_version != expected_version:
                raise ValueError(
                    f"Builtin robot resource {resource_key!r} declares version "
                    f"{resource_version!r}, but this runtime requires {expected_version!r}"
                )
            if model.root_path is not None:
                raise ValueError("Builtin models cannot set root_path")
            if model.mjcf_path.is_absolute():
                raise ValueError(
                    "Builtin model mjcf_path must be relative to its resource directory"
                )
            root = Path(get_robot_path(resource_key)).expanduser().resolve()
            path = self._resolve_relative_path(
                root,
                model.mjcf_path,
                "builtin model mjcf_path",
            )
        else:
            mjcf_path = model.mjcf_path.expanduser()
            if mjcf_path.is_absolute():
                if model.root_path is not None:
                    raise ValueError("An absolute path model mjcf_path cannot also set root_path")
                path = mjcf_path.resolve()
            else:
                if model.root_path is None:
                    raise ValueError("A relative path model requires root_path")
                root = model.root_path.expanduser().resolve()
                path = self._resolve_relative_path(root, mjcf_path, "path model mjcf_path")

        if not path.is_file():
            raise ValueError(f"Robot MJCF does not exist or is not a file: {path}")
        return path

    @staticmethod
    def _resolve_relative_path(root: Path, path: Path, description: str) -> Path:
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"{description} {path} escapes resource root {root}")
        return (root / path).resolve()

    @staticmethod
    def _strip_mesh_geometries(spec: MjSpec) -> None:
        for geom in list(spec.geoms):
            if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
                spec.delete(geom)

    @staticmethod
    def _validate_pose(pos: Sequence[float], quat: Sequence[float]) -> _CallPose:
        try:
            position_values = tuple(float(value) for value in pos)
            quaternion_values = tuple(float(value) for value in quat)
        except (TypeError, ValueError) as error:
            raise ValueError("pos and quat must contain numeric values") from error

        if len(position_values) not in {2, 3}:
            raise ValueError(f"pos must contain two or three values; got {len(position_values)}")
        if len(quaternion_values) != 4:
            raise ValueError(f"quat must contain four wxyz values; got {len(quaternion_values)}")
        if not all(math.isfinite(value) for value in (*position_values, *quaternion_values)):
            raise ValueError("pos and quat must contain only finite values")
        if not any(quaternion_values):
            raise ValueError("quat must be nonzero")

        position_was_2d = len(position_values) == 2
        if position_was_2d:
            position_values = (*position_values, 0.0)
        return _CallPose(position_values, quaternion_values, position_was_2d)

    def _base_mocap_name(self, prefix: str) -> str | None:
        robot_view = self.embodiment_spec.robot_view
        base_group = robot_view.move_groups[robot_view.base_group]
        if not isinstance(base_group, MocapBaseMoveGroupSpec):
            return None
        return self._resolve_name(base_group.body, "scene", prefix)

    def _validate_attachment_policy(self, operation: AttachModelSpec) -> None:
        model = self.embodiment_spec.robot_config.model
        if model.pre_namespaced and operation.apply_namespace:
            raise ValueError("A pre_namespaced model cannot apply a namespace during attachment")
        if operation.apply_namespace != model.prefix_names:
            raise ValueError(
                "attach_model.apply_namespace must match model.prefix_names "
                f"({operation.apply_namespace!r} != {model.prefix_names!r})"
            )

    def _attach_model(
        self,
        destination: MjSpec,
        robot_spec: MjSpec,
        operation: AttachModelSpec,
        prefix: str,
        call_pose: _CallPose,
    ) -> None:
        parent_name = self._resolve_name(operation.parent_body, "scene", prefix)
        parent = self._require(
            destination.body(parent_name),
            "body",
            parent_name,
            "scene attach_model parent_body",
        )

        child_name = operation.child_root_body.name
        configured_root = self.embodiment_spec.robot_config.model.model_root_body.name
        if child_name != configured_root:
            raise ValueError(
                f"attach_model child_root_body {child_name!r} does not match configured "
                f"model_root_body {configured_root!r}"
            )
        child = self._require(
            robot_spec.body(child_name),
            "body",
            child_name,
            "local attach_model child_root_body",
        )

        position = operation.child_pose.position
        quaternion = operation.child_pose.quaternion_wxyz
        if operation.use_base_pose:
            position = call_pose.position
            quaternion = call_pose.quaternion
        frame = parent.add_frame(pos=position, quat=quaternion)
        attachment_prefix = prefix if operation.apply_namespace else ""
        try:
            frame.attach_body(child, attachment_prefix, "")
        except ValueError as error:
            raise ValueError(
                f"Could not attach local body {child_name!r} under scene body "
                f"{parent_name!r} with prefix {attachment_prefix!r}: {error}"
            ) from error

    def _apply_operation(
        self,
        spec: MjSpec,
        operation: ModelOperationSpec,
        *,
        expected_scope: _Scope,
        prefix: str,
        base_mocap_name: str | None = None,
        call_pose: _CallPose | None = None,
        randomize_textures: bool = False,
    ) -> None:
        if operation.scope != expected_scope:
            raise ValueError(
                f"Operation {operation.kind!r} has scope {operation.scope!r}; "
                f"expected {expected_scope!r}"
            )
        if isinstance(operation, AttachModelSpec):
            raise ValueError("attach_model must be handled by scene attachment")

        context = f"{expected_scope} {operation.kind}"

        if isinstance(operation, AddSiteSpec):
            parent_name = self._resolve_name(operation.parent_body, expected_scope, prefix)
            parent = self._require(spec.body(parent_name), "body", parent_name, context)
            name = self._resolve_name(operation.name, expected_scope, prefix)
            self._ensure_absent(spec.site(name), "site", name, context)
            parent.add_site(
                name=name,
                pos=operation.position,
                quat=operation.quaternion_wxyz,
                xyaxes=operation.xyaxes,
                group=operation.group,
                rgba=operation.rgba,
            )
            return

        if isinstance(operation, AddBodySpec):
            parent = spec.worldbody
            if operation.parent_body is not None:
                parent_name = self._resolve_name(operation.parent_body, expected_scope, prefix)
                parent = self._require(spec.body(parent_name), "body", parent_name, context)
            name = self._resolve_name(operation.name, expected_scope, prefix)
            self._ensure_absent(spec.body(name), "body", name, context)
            position = operation.position
            quaternion = operation.quaternion_wxyz
            if operation.use_base_pose:
                if call_pose is None:
                    raise ValueError(f"{context} requires a call-time base pose")
                position = call_pose.position
                if call_pose.position_was_2d:
                    position = (position[0], position[1], operation.base_pose_z_for_2d)
                quaternion = call_pose.quaternion
            parent.add_body(
                name=name,
                pos=position,
                quat=quaternion,
            )
            return

        if isinstance(operation, AddTextureSpec):
            name = self._resolve_name(operation.name, expected_scope, prefix)
            self._ensure_absent(spec.texture(name), "texture", name, context)
            texture_file = operation.file
            if randomize_textures and operation.randomized_files:
                randomized_paths = [
                    self._resolve_relative_path(
                        self._model_path.parent,
                        path,
                        f"{context} randomized texture file",
                    )
                    for path in operation.randomized_files
                ]
                missing_paths = [path for path in randomized_paths if not path.is_file()]
                if missing_paths:
                    raise ValueError(f"Randomized texture files do not exist: {missing_paths}")
                texture_path = random.choice(randomized_paths)
            else:
                texture_path = self._resolve_relative_path(
                    self._model_path.parent,
                    texture_file,
                    f"{context} texture file",
                )
            if not texture_path.is_file():
                raise ValueError(f"Texture file does not exist: {texture_path}")
            spec.add_texture(
                name=name,
                type=_TEXTURE_TYPES[operation.texture_type],
                file=str(texture_path),
            )
            return

        if isinstance(operation, AddMaterialSpec):
            name = self._resolve_name(operation.name, expected_scope, prefix)
            self._ensure_absent(spec.material(name), "material", name, context)
            material = spec.add_material(name=name, rgba=operation.rgba)
            if operation.texture is not None:
                texture_name = self._resolve_name(operation.texture, expected_scope, prefix)
                self._require(spec.texture(texture_name), "texture", texture_name, context)
                material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
            return

        if isinstance(operation, AddGeomSpec):
            parent_name = self._resolve_name(operation.parent_body, expected_scope, prefix)
            parent = self._require(spec.body(parent_name), "body", parent_name, context)
            name = self._resolve_name(operation.name, expected_scope, prefix)
            self._ensure_absent(spec.geom(name), "geom", name, context)
            material_name = None
            if operation.material is not None:
                material_name = self._resolve_name(operation.material, expected_scope, prefix)
                self._require(spec.material(material_name), "material", material_name, context)
            parent.add_geom(
                name=name,
                type=_GEOM_TYPES[operation.geom_type],
                size=operation.size,
                pos=operation.position,
                quat=operation.quaternion_wxyz,
                material=material_name,
                group=operation.group,
                rgba=operation.rgba,
            )
            return

        if isinstance(operation, AddJointSpec):
            self._add_joint(spec, operation, expected_scope, prefix, context, call_pose)
            return

        if isinstance(operation, AddActuatorSpec):
            self._add_actuator(spec, operation, expected_scope, prefix, context)
            return

        if isinstance(operation, AddMocapBodySpec):
            parent = spec.worldbody
            if operation.parent_body is not None:
                parent_name = self._resolve_name(operation.parent_body, expected_scope, prefix)
                parent = self._require(spec.body(parent_name), "body", parent_name, context)
                if parent.name != "world":
                    raise ValueError(
                        f"{context} cannot add mocap body below non-world body {parent_name!r}"
                    )
            name = self._resolve_name(operation.name, expected_scope, prefix)
            self._ensure_absent(spec.body(name), "body", name, context)
            position = operation.initial_pose.position
            quaternion = operation.initial_pose.quaternion_wxyz
            if (name == base_mocap_name or operation.use_base_pose) and call_pose is not None:
                position = call_pose.position
                quaternion = call_pose.quaternion
            parent.add_body(name=name, pos=position, quat=quaternion, mocap=True)
            return

        if isinstance(operation, AddWeldSpec):
            name_a = self._resolve_name(operation.body_a, expected_scope, prefix)
            name_b = self._resolve_name(operation.body_b, expected_scope, prefix)
            self._require(spec.body(name_a), "body", name_a, context)
            self._require(spec.body(name_b), "body", name_b, context)
            if name_a == name_b:
                raise ValueError(f"{context} resolves both weld bodies to {name_a!r}")
            spec.add_equality(
                type=mujoco.mjtEq.mjEQ_WELD,
                objtype=mujoco.mjtObj.mjOBJ_BODY,
                name1=name_a,
                name2=name_b,
                solref=operation.solref,
                solimp=operation.solimp,
            )
            return

        if isinstance(operation, AddPlatformSpec):
            parent_name = self._resolve_name(operation.parent_body, expected_scope, prefix)
            parent = self._require(spec.body(parent_name), "body", parent_name, context)
            name = self._resolve_name(operation.name, expected_scope, prefix)
            self._ensure_absent(spec.geom(name), "geom", name, context)
            material_name = None
            if operation.material is not None:
                material_name = self._resolve_name(operation.material, expected_scope, prefix)
                self._require(spec.material(material_name), "material", material_name, context)
            parent.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=tuple(size / 2.0 for size in operation.size_xyz),
                pos=operation.position,
                quat=operation.quaternion_wxyz,
                material=material_name,
                rgba=operation.rgba,
                group=operation.group,
            )
            return

        if isinstance(operation, SetBodyPropertiesSpec):
            self._set_body_properties(spec, operation, expected_scope, prefix, context)
            return

        if isinstance(operation, SpeckleRgbaMaterialsSpec):
            raise ValueError("speckle_rgba_materials must be handled before model attachment")

        raise ValueError(f"Unsupported model operation type: {type(operation).__name__}")

    def _speckle_rgba_materials(
        self,
        destination: MjSpec,
        robot_spec: MjSpec,
        operation: SpeckleRgbaMaterialsSpec,
        prefix: str,
    ) -> None:
        if random.random() > operation.probability:
            return

        perturbed_materials: dict[str, str] = {}
        for material in robot_spec.materials:
            is_rgba_material = all(
                material.textures[index] == "" for index in range(mujoco.mjtTextureRole.mjNTEXROLE)
            )
            if not is_rgba_material:
                continue

            image = self._speckle_texture(material.rgba[:3], operation)
            buffer = BytesIO()
            image.save(buffer, format="PNG")
            buffer.seek(0)

            texture_name = f"{material.name}{operation.texture_name_suffix}"
            material_name = f"{material.name}{operation.material_name_suffix}"
            filename = f"{prefix}{texture_name}.png".replace("/", "__")
            destination.assets[filename] = buffer.getvalue()
            robot_spec.add_texture(
                name=texture_name,
                type=mujoco.mjtTexture.mjTEXTURE_2D,
                file=filename,
            )
            perturbed_material = robot_spec.add_material(name=material_name)
            perturbed_material.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = texture_name
            perturbed_materials[material.name] = material_name

        root_name = operation.model_root_body.name
        root = self._require(
            robot_spec.body(root_name),
            "body",
            root_name,
            "speckle_rgba_materials model_root_body",
        )

        def replace_materials(body: mujoco.MjsBody) -> None:
            for geom in body.geoms:
                if geom.material in perturbed_materials:
                    geom.material = perturbed_materials[geom.material]
            for child in body.bodies:
                replace_materials(child)

        replace_materials(root)

    @staticmethod
    def _speckle_texture(
        base_color: Sequence[float],
        operation: SpeckleRgbaMaterialsSpec,
    ) -> Image.Image:
        image = np.ones((operation.texture_size, operation.texture_size, 3)) * np.array(base_color)
        noise = np.random.normal(
            0,
            operation.noise_strength,
            (operation.texture_size, operation.texture_size, 1),
        )
        image = np.clip(image + noise, 0, 1)
        image = (image * 255).astype(np.uint8)

        pil_image = Image.fromarray(image)
        draw = ImageDraw.Draw(pil_image)
        for _ in range(operation.num_blobs):
            x = np.random.randint(0, operation.texture_size)
            y = np.random.randint(0, operation.texture_size)
            width = np.random.randint(*operation.blob_size_range)
            height = np.random.randint(*operation.blob_size_range)
            variation = np.random.uniform(
                -operation.blob_variation,
                operation.blob_variation,
            )
            blob_color = tuple(
                int(np.clip((component + variation) * 255, 0, 255)) for component in base_color
            )
            shape = (x, y, x + width, y + height)
            if np.random.random() > 0.5:
                draw.ellipse(shape, fill=blob_color)
            else:
                draw.rectangle(shape, fill=blob_color)
        return pil_image

    def _add_joint(
        self,
        spec: MjSpec,
        operation: AddJointSpec,
        scope: _Scope,
        prefix: str,
        context: str,
        call_pose: _CallPose | None,
    ) -> None:
        parent_name = self._resolve_name(operation.parent_body, scope, prefix)
        parent = self._require(spec.body(parent_name), "body", parent_name, context)
        name = self._resolve_name(operation.name, scope, prefix)
        self._ensure_absent(spec.joint(name), "joint", name, context)

        if operation.joint_type in {"ball", "free"}:
            if operation.axis is not None:
                raise ValueError(f"{context} {operation.joint_type} joint cannot declare axis")
            if operation.limited:
                raise ValueError(f"{context} {operation.joint_type} joint cannot set limited=true")
            if operation.reference != 0.0:
                raise ValueError(
                    f"{context} {operation.joint_type} joint cannot declare a scalar reference"
                )
        if operation.joint_type == "free":
            if parent.parent.name != "world":
                raise ValueError(f"{context} free joint body {parent_name!r} must be a root body")
            if parent.joints:
                raise ValueError(f"{context} free joint must be the body's only joint")
        elif any(joint.type == mujoco.mjtJoint.mjJNT_FREE for joint in parent.joints):
            raise ValueError(f"{context} cannot add another joint to a body with a free joint")

        self._validate_strict_range(operation.range, f"{context} range for joint {name!r}")
        if operation.limited is True and operation.range is None:
            raise ValueError(f"{context} limited joint {name!r} requires a range")
        limited = operation.limited
        if operation.range is not None and limited is None:
            limited = True
        axis = operation.axis
        if operation.axis_frame == "world" and axis is not None:
            if call_pose is None:
                raise ValueError(f"{context} world-frame axis requires a call-time base pose")
            rotation = R.from_quat(call_pose.quaternion, scalar_first=True)
            axis = tuple(rotation.inv().apply(axis))

        reference = operation.reference
        if operation.reference_source != "literal":
            if call_pose is None:
                raise ValueError(f"{context} base-pose reference requires a call-time base pose")
            if operation.reference_source == "base_x":
                reference = call_pose.position[0]
            elif operation.reference_source == "base_y":
                reference = call_pose.position[1]
            else:
                euler = R.from_quat(call_pose.quaternion, scalar_first=True).as_euler("xyz")
                if not np.allclose(euler[:2], 0.0, atol=1e-6):
                    raise ValueError(
                        "Planar base insertion quaternion must have zero roll and pitch"
                    )
                reference = float(euler[2])
        parent.add_joint(
            name=name,
            type=_JOINT_TYPES[operation.joint_type],
            pos=operation.position,
            axis=axis,
            range=operation.range,
            limited=limited,
            damping=operation.damping,
            armature=operation.armature,
            stiffness=operation.stiffness,
            ref=reference,
        )

    def _add_actuator(
        self,
        spec: MjSpec,
        operation: AddActuatorSpec,
        scope: _Scope,
        prefix: str,
        context: str,
    ) -> None:
        name = self._resolve_name(operation.name, scope, prefix)
        self._ensure_absent(spec.actuator(name), "actuator", name, context)

        if operation.joint is not None:
            target_name = self._resolve_name(operation.joint, scope, prefix)
            self._require(spec.joint(target_name), "joint", target_name, context)
            transmission = mujoco.mjtTrn.mjTRN_JOINT
            reference_site = None
        elif operation.site is not None:
            target_name = self._resolve_name(operation.site, scope, prefix)
            self._require(spec.site(target_name), "site", target_name, context)
            transmission = mujoco.mjtTrn.mjTRN_SITE
            if operation.reference_site is None:
                raise ValueError(f"{context} site actuator requires reference_site")
            reference_site = self._resolve_name(operation.reference_site, scope, prefix)
            self._require(spec.site(reference_site), "site", reference_site, context)
            if reference_site == target_name:
                raise ValueError(f"{context} site and reference_site resolve to the same site")
        else:
            raise ValueError(f"{context} requires exactly one transmission target")

        self._validate_strict_range(
            operation.control_range,
            f"{context} control_range for actuator {name!r}",
        )
        self._validate_strict_range(
            operation.force_range,
            f"{context} force_range for actuator {name!r}",
        )
        gear = None
        if operation.gear is not None:
            gear = [*operation.gear, *([0.0] * (6 - len(operation.gear)))]

        actuator = spec.add_actuator(
            name=name,
            trntype=transmission,
            target=target_name,
            refsite=reference_site,
            gear=gear,
        )
        if operation.actuator_type == "motor":
            actuator.set_to_motor()
        elif operation.actuator_type == "position":
            actuator.set_to_position(
                operation.gain if operation.gain is not None else 1.0,
                operation.damping_ratio if operation.damping_ratio is not None else 0.0,
            )
        elif operation.actuator_type == "velocity":
            actuator.set_to_velocity(operation.gain if operation.gain is not None else 1.0)

        if operation.gain is not None:
            actuator.gainprm[0] = operation.gain
        if operation.bias_position is not None or operation.bias_velocity is not None:
            actuator.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        if operation.bias_position is not None:
            actuator.biasprm[1] = operation.bias_position
        if operation.bias_velocity is not None:
            actuator.biasprm[2] = operation.bias_velocity
        if operation.control_range is not None:
            actuator.ctrllimited = 1
            actuator.ctrlrange[:] = operation.control_range
        if operation.force_range is not None:
            actuator.forcelimited = 1
            actuator.forcerange[:] = operation.force_range

    def _set_body_properties(
        self,
        spec: MjSpec,
        operation: SetBodyPropertiesSpec,
        scope: _Scope,
        prefix: str,
        context: str,
    ) -> None:
        name = self._resolve_name(operation.body, scope, prefix)
        body = self._require(spec.body(name), "body", name, context)
        if operation.position is not None:
            body.pos = operation.position
        if operation.quaternion_wxyz is not None:
            body.quat = operation.quaternion_wxyz
        if operation.gravity_compensation is not None:
            body.gravcomp = operation.gravity_compensation

        inertial_values = (
            operation.mass,
            operation.center_of_mass,
            operation.inertia_diagonal,
        )
        if any(value is not None for value in inertial_values):
            if body.name == "world":
                raise ValueError(f"{context} cannot set inertial properties on the world body")
            if not body.explicitinertial:
                self._materialize_inertia(spec, body, name, context)
            if operation.mass is not None:
                body.mass = operation.mass
            if operation.center_of_mass is not None:
                body.ipos = operation.center_of_mass
            if operation.inertia_diagonal is not None:
                body.inertia = operation.inertia_diagonal

    @staticmethod
    def _materialize_inertia(
        spec: MjSpec,
        body: mujoco.MjsBody,
        name: str,
        context: str,
    ) -> None:
        try:
            model = spec.copy().compile()
        except ValueError as error:
            raise ValueError(
                f"{context} cannot preserve inferred inertia for body {name!r}: {error}"
            ) from error
        model_body = model.body(name)
        body_id = model_body.id
        body.mass = float(model.body_mass[body_id])
        body.ipos = model.body_ipos[body_id]
        body.iquat = model.body_iquat[body_id]
        body.inertia = model.body_inertia[body_id]
        body.explicitinertial = 1

    @staticmethod
    def _resolve_name(reference: ModelNameRef, scope: _Scope, prefix: str) -> str:
        if scope == "robot_model" or reference.absolute:
            return reference.name
        return f"{prefix}{reference.name}"

    @staticmethod
    def _require(value: _T | None, kind: str, name: str, context: str) -> _T:
        if value is None:
            raise ValueError(f"Unresolved {kind} name {name!r} while applying {context}")
        return value

    @staticmethod
    def _ensure_absent(value: object | None, kind: str, name: str, context: str) -> None:
        if value is not None:
            raise ValueError(f"Cannot add {kind} {name!r} in {context}: name already exists")

    @staticmethod
    def _validate_strict_range(
        value: tuple[float, float] | None,
        description: str,
    ) -> None:
        if value is not None and value[0] == value[1]:
            raise ValueError(f"{description} must have distinct lower and upper bounds")
