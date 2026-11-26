"""PARAMETERS + MODEL -> EMBODIMENT SPEC. The step that makes a generated robot loadable.

:mod:`.builder` produces an MJCF, which is a *model*. MolmoSpaces drives a robot through an
:class:`EmbodimentSpec`: which joints form the arm, which the gripper, what the controllers are,
where the TCP is, what counts as a grasp. Without that, a generated robot cannot be handed to
``RobotView`` or to the production planner at all, and the only way to test one is a bespoke script
that re-implements planning badly.

Everything here is DERIVED, never authored by hand:

* arm move group, actuators and gains come from :class:`~.params.EmbodimentParams`
* the whole tool side -- move group, controller, grasp profile, sensors, model operations -- comes
  from the gripper catalog, which was harvested from the shipped assets for this purpose
* the mount comes from the sampled pedestal

so the spec for a generated robot is a function of its parameters, exactly as its MJCF is.

Frames are uniformly ``planner_grasp_site``. Every catalog entry declares it and carries the
``add_site`` operation that creates it, so the TCP, the leaf frames, the sensor and the grasp frame
all name one site regardless of which gripper was drawn.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import yaml

from .builder import (GRASP_FEASIBILITY_LIMIT, SELF_COLLISION_LIMIT, BuildError,
                      build, joint_name, rest_pose)
from .constants import (
    BASE_BODY,
    GRIPPER_PREFIX,
    MJCF_NAME,
    NODE_SPEC_NAME,
    PARAMS_NAME,
    PLANNER_SITE,
    SPEC_NAME,
)
from .grippers import load_catalog
from .params import EmbodimentParams
from molmo_spaces.utils.asset_tokens import to_tokens

PLANNER_FRAME = {"kind": "site", "name": {"name": f"{GRIPPER_PREFIX}{PLANNER_SITE}",
                                          "absolute": False}}


def _ref(name: str) -> dict:
    return {"name": name, "absolute": False}


def _actuator_overrides(params: EmbodimentParams, gripper) -> dict:
    """One override per arm joint, in MuJoCo's affine convention, plus the tool's own.

    The gains are the sampled ones rather than a default: a generated arm's kp is drawn per joint
    and the spec has to agree with the model, or the control profile silently retunes the robot."""
    overrides = {}
    for index, joint in enumerate(params.arm.joints, start=1):
        name = joint_name(index)
        entry = {
            "actuator": _ref(name),
            "gain": float(joint.kp),
            "bias_position": float(-joint.kp),
            "bias_velocity": float(-joint.kv),
        }
        # force 0 means unlimited, the way FR3 leaves its arm; the spec says so by omission
        if joint.force > 0.0:
            entry["force_range"] = [float(-joint.force), float(joint.force)]
        overrides[name] = entry
    for actuator in gripper.move_group["actuator_names"]:
        overrides[actuator["name"]] = {"actuator": dict(actuator)}
    return overrides


# Distance held back from a joint limit when choosing the start pose. MJCF serialises numbers to
# six significant figures, so a limit written as -0.4457187650 reads back as -0.445719 -- BEYOND the
# original. A start pose sitting exactly on that limit is then rejected by the strict initial-qpos
# check ("violates compiled joint limits"), which cost seed68 every episode of a run. The inset is
# far larger than any rounding at this precision and far smaller than anything kinematically
# meaningful.
LIMIT_INSET = 1.0e-3


def _initial_qpos(params: EmbodimentParams, gripper, model=None) -> dict:
    """A start pose: the measured tucked rest configuration, held inside the joint limits.

    Falls back to the value nearest zero per joint only if no rest pose can be found. That fallback
    is what used to be the rule outright, and it is a numerical choice with no notion of posture --
    see :func:`~.builder.rest_pose` for what it produced."""
    arm = None
    if model is not None:
        # the gripper's own approach axis, not an assumed one: rest_pose needs it to tell a pose
        # whose jaws face forward from one facing the ceiling
        axis = tuple(gripper.grasp_profile["approach_axis_in_grasp_frame"])
        found = rest_pose(model, params.arm.dof, params.arm.pedestal_height, approach=axis)
        if found is not None:
            arm = [float(v) for v in found]
    if arm is None:
        arm = []
        for joint in params.arm.joints:
            low, high = joint.range
            inset = min(LIMIT_INSET, 0.25 * (high - low))
            arm.append(float(min(max(0.0, low + inset), high - inset)))
    # keep clear of the endpoints regardless: MJCF serialises to six significant figures, so a value
    # sitting exactly on a limit can read back outside it and fail the strict initial-qpos check
    clamped = []
    for value, joint in zip(arm, params.arm.joints):
        low, high = joint.range
        inset = min(LIMIT_INSET, 0.25 * (high - low))
        clamped.append(float(min(max(value, low + inset), high - inset)))
    return {"arm": clamped, "gripper": [float(v) for v in gripper.initial_qpos]}


def _model_operations(params: EmbodimentParams, gripper) -> list:
    """The gripper's own operations, then the mount that stands the arm up in a scene."""
    height = float(params.arm.pedestal_height)
    radius = float(params.arm.pedestal_radius)
    operations = [dict(op) for op in gripper.model_operations]
    operations += [
        {"kind": "add_mocap_body", "scope": "scene", "name": _ref("base"), "parent_body": None,
         "initial_pose": {"position": [0.0, 0.0, 0.0], "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]}},
        {"kind": "add_material", "scope": "scene", "name": _ref("robot_base_material"),
         "rgba": [0.30, 0.20, 0.10, 1.0]},
        {"kind": "add_geom", "scope": "scene", "parent_body": _ref("base"),
         "name": _ref("robot_base_geom"), "geom_type": "box",
         "size": [radius, radius, height / 2.0],
         "position": [0.0, 0.0, height / 2.0],
         "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
         "material": _ref("robot_base_material"), "group": 0},
        {"kind": "attach_model", "scope": "scene", "parent_body": _ref("base"),
         "child_root_body": _ref(BASE_BODY),
         "child_pose": {"position": [0.0, 0.0, height],
                        "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0]},
         "apply_namespace": True},
    ]
    return operations


def _robot_view(params: EmbodimentParams, gripper) -> dict:
    """base -> arm -> gripper, with every frame naming the uniform planner site."""
    joints = [_ref(joint_name(i)) for i in range(1, params.arm.dof + 1)]
    tool = dict(gripper.move_group)
    tool["parent_group"] = "base"
    tool["leaf_frame"] = PLANNER_FRAME          # uniform, whichever gripper was drawn
    return {
        "base_group": "base",
        "move_groups": {
            "base": {"kind": "mocap_base", "role": "base", "side": "center",
                     "body": _ref("base"), "runtime_mobile": False},
            "arm": {"kind": "joint", "role": "arm", "side": "center", "parent_group": "base",
                    "root_body": _ref(BASE_BODY),
                    "joint_names": joints, "actuator_names": list(joints),
                    "leaf_frame": PLANNER_FRAME},
            "gripper": tool,
        },
        "end_effectors": {
            "primary": {"arm_group": "arm", "tool_group": "gripper",
                        "tcp_frame": PLANNER_FRAME, "grasp_frame": PLANNER_FRAME,
                        "grasp_profile_id": gripper.grasp_profile_id},
        },
    }


def spec_dict(params: EmbodimentParams, mjcf_path: str = MJCF_NAME, model=None) -> dict:
    """The EmbodimentSpec for ``params``, as a plain mapping."""
    gripper = load_catalog()[params.gripper]
    sensors = [{"kind": "frame_pose", "uuid": "tcp_pose", "frame": PLANNER_FRAME,
                "relative_to_group": "base", "output_format": "xyz_quaternion_wxyz"}]
    return {
        "schema_version": 1,
        "embodiment_id": params.embodiment_id,
        "embodiment_version": f"generator{params.version}",
        "robot_config": {
            "namespace": "robot_0/",
            "model": {"source": "path", "resource_key": None, "resource_version": None,
                      "root_path": ".", "mjcf_path": mjcf_path,
                      "model_root_body": _ref(BASE_BODY), "pre_namespaced": False,
                      "prefix_names": True},
            "initial_qpos": _initial_qpos(params, gripper, model),
            "initial_qpos_noise": None,
            "default_control_profile": "default",
            "control_profiles": {
                "default": {
                    "controller_bindings": {"arm": "arm_position",
                                            "gripper": gripper.controller["kind"]
                                            and "gripper_position"},
                    "actuator_overrides": _actuator_overrides(params, gripper),
                },
            },
            # Always disabled here. Gravity compensation is already baked into the MJCF as body
            # gravcomp wherever the parameters asked for it (YAM's links 2-6 do, FR3's do not), and
            # switching it on again in the spec would apply it twice.
            "gravity_compensation": {"enabled": False, "root_body": _ref(BASE_BODY),
                                     "include_root": False, "include_descendants": True},
            "action_noise": {"enabled": True, "target_arm_groups": ["arm"],
                             "target_base_group": None, "action_scale_factor": 0.1,
                             "rotation_noise_scale": 0.1, "max_tcp_position_noise": 0.02,
                             "max_tcp_rotation_noise": 0.1, "base_action_scale_factor": 0.1,
                             "max_base_position_noise": 0.02, "max_base_rotation_noise": 0.05},
            "placement": {"reset_repositionable": True, "runtime_mobility": "fixed",
                          "footprint_xy": [0.3, 0.3],
                          "platform_size_xyz": [0.3, 0.3, float(params.arm.pedestal_height)],
                          "forward_offset_step": 0.02},
        },
        "robot": {
            "model_operations": _model_operations(params, gripper),
            "controllers": {
                "arm_position": {"kind": "joint_position", "move_group": "arm"},
                "arm_relative_position": {"kind": "joint_relative_position", "move_group": "arm"},
                "gripper_position": dict(gripper.controller),
            },
            # max_iterations is the batch SCREEN budget, deliberately below
            # online_max_iterations: raising it to 1000 cost 23x per screen call
            # (2.4 s -> 55 s) and found nothing more on a probe episode.
            "kinematics": {"kind": "damped_least_squares", "online_backend": "numpy",
                           "batch_backend": "warp", "optimizable_groups": ["arm"],
                           "max_iterations": 50, "convergence_tolerance": 0.001,
                           "success_tolerance": 0.0005, "damping": 1.0e-12,
                           "integration_step": 1.0, "online_max_iterations": 1000,
                           "online_success_tolerance": 0.0001},
            "sensors": sensors,
        },
        "robot_view": _robot_view(params, gripper),
        "grasp_profiles": {gripper.grasp_profile_id: dict(gripper.grasp_profile)},
    }


def acceptable_seeds(seeds, limit: float = SELF_COLLISION_LIMIT,
                     feasibility: float | None = GRASP_FEASIBILITY_LIMIT):
    """The subset of ``seeds`` whose robots are usable, in order.

    Generation is a filter, not a function: a seed that produces an arm folded through itself is
    rejected rather than emitted. Expect to discard most of them -- acceptance at the default limit
    is a few percent -- so iterate a range rather than a fixed list."""
    from .sampling import sample_params
    for seed in seeds:
        try:
            build(sample_params(seed), max_self_collision=limit,
                  min_grasp_feasibility=feasibility)
        except BuildError:
            continue
        except (ValueError, RuntimeError):
            continue
        yield seed


def emit(params: EmbodimentParams, directory: Path,
         max_self_collision: float | None = SELF_COLLISION_LIMIT,
         min_grasp_feasibility: float | None = None) -> Path:
    """Write ``model.xml``, ``spec.yaml`` and ``params.json`` into ``directory``.

    The three together are the whole embodiment: the model, the contract a consumer drives it
    through, and the parameters that produced both -- so a robot can be regenerated or diffed
    without re-running the sampler."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    xml, model = build(params, max_self_collision=max_self_collision,
                       min_grasp_feasibility=min_grasp_feasibility, wrist_camera=True)
    # Store portable ${ROOT} tokens rather than machine-specific absolute asset paths; the assembler
    # materializes them on load (molmo_spaces.utils.asset_tokens).
    (directory / MJCF_NAME).write_text(to_tokens(xml))
    # The node set, alongside the model. A morphology-agnostic consumer needs it to read a trajectory
    # at all -- which nodes exist, which are actuated, which slide, and which qpos entry drives each
    # finger -- and virtual_vla's add_transfer_queries_to_wods reads exactly this out of the library
    # entry for a generated robot. Kept as its own file because EmbodimentSpec's schema is strict and
    # would reject an extra key.
    from .node_spec import derive

    gripper = load_catalog()[params.gripper]
    joints = tuple(entry["name"] for entry in gripper.move_group["joint_names"])
    # Order the gripper's finger nodes by their VISUAL left->right in the wrist view rather than by
    # name, for grippers whose wrist camera mirrors L/R -- keeps node index-position consistent
    # across grippers. Per-gripper flag on the catalog's wrist_camera entry.
    lr_swap = bool((gripper.wrist_camera or {}).get("lr_swap"))
    (directory / NODE_SPEC_NAME).write_text(
        json.dumps(derive(model, params.arm.dof, joints, lr_swap=lr_swap, gripper=params.gripper), indent=2))
    (directory / PARAMS_NAME).write_text(json.dumps(asdict(params), indent=2, default=list))
    path = directory / SPEC_NAME
    path.write_text(yaml.safe_dump(spec_dict(params, model=model), sort_keys=False, width=100))
    return path
