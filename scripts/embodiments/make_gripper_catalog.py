"""Build ``molmo_spaces/embodiment_gen/gripper_catalog/*.yaml`` from shipped robot specs.

Every field of a catalog entry is DERIVED, never hand-authored:

* ``graft_root``   = the gripper move group's ``root_body``
* ``drop_joints``  = that body's own joints (a free joint for the floating grippers, the
                     arm's last driven joint for yam)
* ``keep_meshes``  = the meshes a trial graft's geoms actually reference
* semantics        = the gripper move group, its controller, its grasp profile, its
                     sensors and its ``add_site`` operations, lifted verbatim from the
                     source ``EmbodimentSpec`` with every name moved under ``gripper/``

So refreshing the catalog after an upstream asset or spec bump is one command, and the
generator never re-derives gripper meaning it could inherit.

    MLSPACES_ASSETS_DIR=... python scripts/embodiments/make_gripper_catalog.py
    ... --only yam_jaw --check          # verify the committed entries still reproduce
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import yaml
from mujoco import MjSpec

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from molmo_spaces.embodiment_gen.constants import (  # noqa: E402
    CATALOG_DIR,
    GRIPPER_PREFIX,
    PLANNER_SITE,
)
from molmo_spaces.embodiment_gen.grippers import mesh_key, used_meshes  # noqa: E402
from molmo_spaces.utils.pose import GL_FROM_CV  # noqa: E402
from molmo_spaces.embodiment_gen.geom_pool import SOURCES as GEOM_SOURCES  # noqa: E402
from molmo_spaces.embodiment_gen.geom_pool import harvest  # noqa: E402
from molmo_spaces.molmo_spaces_constants import (  # noqa: E402
    DATA_TYPE_TO_SOURCE_TO_VERSION,
    get_robot_path,
)

# catalog name -> the shipped spec to lift from. franka_droid/franka_cap are deliberately
# absent: they reuse floating_robotiq's 2F-85 and floating_rum's jaw respectively. None means
# the entry has no source EmbodimentSpec and is built by its own function: linear_4310 is a
# vendor gripper bundled with the repo, not a shipped robot.
SOURCES = {
    "robotiq_2f85": "floating_robotiq.yaml",
    "rum": "floating_rum.yaml",
    "yam_jaw": "i2rt_yam.yaml",
    "linear_4310": None,
}


def prefix_names(node, names: set):
    """Recursively rewrite ``{name: X, ...}`` refs to ``gripper/X`` for every X the graft
    moved under the prefix. Names the graft did not touch are left alone."""
    if isinstance(node, dict):
        out = {}
        for key, value in node.items():
            if key == "name" and isinstance(value, str) and value in names:
                out[key] = GRIPPER_PREFIX + value
            else:
                out[key] = prefix_names(value, names)
        return out
    if isinstance(node, list):
        return [prefix_names(v, names) for v in node]
    return node


def collect_refs(node, out: set):
    """Every ``name`` string reachable in a spec fragment."""
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "name" and isinstance(value, str):
                out.add(value)
            else:
                collect_refs(value, out)
    elif isinstance(node, list):
        for value in node:
            collect_refs(value, out)
    return out


def gripper_group(spec: dict) -> tuple:
    for name, group in spec["robot_view"]["move_groups"].items():
        if group.get("kind") == "gripper":
            return name, group
    raise ValueError("spec declares no move group of kind 'gripper'")


def trial_graft(model_path: Path, graft_root: str, drop: list, keep=None):
    """Graft the candidate subtree onto a bare flange and compile. Returns
    ``(compiled model, grafted body names)``."""
    child = MjSpec.from_file(str(model_path))
    for key in list(child.keys):
        child.delete(key)
    for joint_name in drop:
        joint = child.joint(joint_name)
        for actuator in list(child.actuators):
            if actuator.target == joint_name:
                child.delete(actuator)
        if joint is not None:
            child.delete(joint)
    if keep is not None:
        for mesh in list(child.meshes):
            if mesh_key(mesh) not in keep:
                child.delete(mesh)
    arm = MjSpec()
    flange = arm.worldbody.add_body(name="flange", pos=[0, 0, 0.3])
    flange.add_geom(name="flange_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                    size=[0.03, 0.01, 0])
    flange.add_frame().attach_body(child.body(graft_root), GRIPPER_PREFIX, "")
    model = arm.compile()
    bodies = {(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "")
              .removeprefix(GRIPPER_PREFIX) for b in range(model.nbody)}
    return model, bodies


def build(name: str, spec_file: Path) -> dict:
    spec = yaml.safe_load(spec_file.read_text())
    group_name, group = gripper_group(spec)
    model = spec["robot_config"]["model"]
    resource_key = model["resource_key"]
    model_path = get_robot_path(resource_key) / model["mjcf_path"]

    graft_root = group["root_body"]["name"]
    source = MjSpec.from_file(str(model_path))
    root = source.body(graft_root)
    if root is None:
        raise ValueError(f"{name}: root_body {graft_root!r} absent from {model_path}")
    drop = [j.name for j in root.joints]

    compiled, bodies = trial_graft(model_path, graft_root, drop)
    keep = sorted(used_meshes(compiled))
    # the pruned graft must still compile and keep its actuator + coupling
    pruned, _ = trial_graft(model_path, graft_root, drop, keep=set(keep))
    assert pruned.nu == compiled.nu and pruned.neq == compiled.neq, (
        f"{name}: pruning changed the model ({compiled.nu}->{pruned.nu} actuators, "
        f"{compiled.neq}->{pruned.neq} equality constraints)")

    # names the graft moved under the prefix: everything the compiled model contributed
    moved = set()
    for objtype, count in ((mujoco.mjtObj.mjOBJ_BODY, pruned.nbody),
                           (mujoco.mjtObj.mjOBJ_JOINT, pruned.njnt),
                           (mujoco.mjtObj.mjOBJ_ACTUATOR, pruned.nu),
                           (mujoco.mjtObj.mjOBJ_SITE, pruned.nsite),
                           (mujoco.mjtObj.mjOBJ_GEOM, pruned.ngeom),
                           (mujoco.mjtObj.mjOBJ_TENDON, pruned.ntendon)):
        for i in range(count):
            nm = mujoco.mj_id2name(pruned, objtype, i) or ""
            if nm.startswith(GRIPPER_PREFIX):
                moved.add(nm.removeprefix(GRIPPER_PREFIX))

    profile_id = next((ee["grasp_profile_id"] for ee in spec["robot_view"]["end_effectors"].values()
                       if ee.get("tool_group") == group_name), "")
    profile = spec.get("grasp_profiles", {}).get(profile_id, {})
    compiled_source = mujoco.MjModel.from_xml_path(str(model_path))
    source_model = source

    default_profile = spec["robot_config"]["default_control_profile"]
    binding = (spec["robot_config"]["control_profiles"][default_profile]
               ["controller_bindings"].get(group_name))
    controller = dict(spec["robot"].get("controllers", {}).get(binding, {}))

    # EVERY entry must declare its grasp frame as `planner_grasp_site`. yam_jaw's source spec
    # already adds one; floating_robotiq and floating_rum point their profiles at `grasp_site`, a
    # site that happens to be in their MJCF. Leaving that inconsistent means a planner has to ask
    # each gripper what its frame is called -- and aiming the wrong one is not a small error: yam's
    # grasp_site sits 49.7 mm from planner_grasp_site and 44.1 mm from the jaws, so a 30 mm cube is
    # missed entirely. Synthesised at the pose of whatever site the source profile named, so the
    # frame is identical in space and only the name is made uniform.
    frame_name = (profile.get("grasp_frame", {}).get("name", {}) or {}).get("name")
    synthesised = None
    if profile and frame_name and frame_name != PLANNER_SITE:
        source_site = source_model.site(frame_name) if hasattr(source_model, "site") else None
        sid = mujoco.mj_name2id(compiled_source, mujoco.mjtObj.mjOBJ_SITE, frame_name)
        if sid < 0:
            raise ValueError(f"{name}: profile names site {frame_name!r}, absent from the model")
        body = mujoco.mj_id2name(compiled_source, mujoco.mjtObj.mjOBJ_BODY,
                                 int(compiled_source.site_bodyid[sid]))
        synthesised = dict(
            kind="add_site", scope="robot_model",
            parent_body={"name": body, "absolute": False},
            name={"name": PLANNER_SITE, "absolute": False},
            position=[float(v) for v in compiled_source.site_pos[sid]],
            quaternion_wxyz=[float(v) for v in compiled_source.site_quat[sid]],
            xyaxes=None, group=4, rgba=[0.0, 1.0, 1.0, 1.0])
        del source_site

    # An operation belongs to the gripper iff it attaches to a body the graft moved. It is
    # tested on its PARENT, not on all its refs: an add_site names a site that does not
    # exist yet, so a subset test would reject exactly the ops that matter (yam's
    # planner_grasp_site, which its grasp profile and tcp_pose sensor both reference).
    operations = [o for o in spec["robot"].get("model_operations", [])
                  if o.get("scope") == "robot_model"
                  and (o.get("parent_body") or o.get("body") or {}).get("name") in moved]
    if synthesised is not None:
        operations = list(operations) + [synthesised]
        profile = dict(profile)
        profile["grasp_frame"] = {"kind": "site", "name": {"name": PLANNER_SITE, "absolute": False}}
    for operation in operations:                     # what those ops create is gripper-side too
        created = operation.get("name")
        if isinstance(created, dict) and created.get("name"):
            moved.add(created["name"])

    # A sensor belongs to the gripper iff every name it references is gripper-side. That
    # excludes the arm's joint_state sensors without needing to enumerate them.
    def gripper_side(fragment) -> bool:
        refs = collect_refs(fragment, set())
        return bool(refs) and refs <= moved

    sensors = [s for s in spec["robot"].get("sensors", []) if gripper_side(s)]

    # Does the graft root double as one of the source's own arm links? yam's does (link_6), so its
    # gripper carries the flange link and the arm must not dress that link as well. Derived, so a
    # new catalog entry cannot forget to declare it.
    claims_flange = (resource_key in GEOM_SOURCES
                     and graft_root in {link.body for link in harvest(resource_key)})

    group = {k: v for k, v in group.items() if k != "parent_group"}
    return dict(
        name=name,
        resource_key=resource_key,
        resource_version=DATA_TYPE_TO_SOURCE_TO_VERSION["robots"][resource_key],
        mjcf=model["mjcf_path"],
        graft_root=graft_root,
        drop_joints=drop,
        keep_meshes=keep,
        claims_flange_link=bool(claims_flange),
        move_group=prefix_names(group, moved),
        controller=prefix_names(controller, moved),
        grasp_profile_id=profile_id,
        grasp_profile=prefix_names(profile, moved),
        initial_qpos=spec["robot_config"].get("initial_qpos", {}).get(group_name, []),
        sensors=prefix_names(sensors, moved),
        model_operations=prefix_names(operations, moved),
        wrist_camera=VENDOR_WRIST_CAMERA.get(name)
        or wrist_camera_entry(compiled_source, profile, frame_name),
        **({"actuator_override": ACTUATOR_OVERRIDE[name]} if name in ACTUATOR_OVERRIDE else {}),
        **({"mount_quat": MOUNT_QUAT[name]} if name in MOUNT_QUAT else {}),
    )


# Assets whose own frame does not put the fingers along the mount axis. The catalog convention is
# fingers along flange +z; floating_rum's point +x, so sampled arms wore it sideways. graft()
# composes this between the caller's mount pose and the root-pose cancellation. The roll about the
# corrected axis is arbitrary: rum is handheld, there is no arm assembly to match.
MOUNT_QUAT = {
    "rum": [0.7071068, 0.0, -0.7071068, 0.0],   # Ry(-90): +x -> +z
}



# Where the shipped wrist cameras sit, MEASURED in each gripper's own grasp frame:
#
#   robotiq  gripper/wrist_camera  pos [-0.074, +0.031, -0.133]  0.155 m back  aim.approach +0.94  fovy 57
#   rum      gripper/wrist_cam     pos [+0.086,  0.000, -0.176]  0.196 m back  aim.approach +0.97  fovy 53
#
# The convention both follow: set back 0.13-0.18 m along -approach, offset 0.07-0.09 m to one side,
# aimed within ~20 degrees of the approach axis so the jaws and the object are both in frame, at a
# 53-57 degree field of view (the ZED Mini's 52 degrees, which FrankaOmniPurposeCameraSystem
# declares, sits just below that). The lateral SIGN is arbitrary -- the two shipped assets disagree,
# robotiq mounting left and rum right.
#
# yam_jaw ships no camera, so one is synthesized at the midpoint of that measured convention rather
# than at a chosen pose. Expressed in the grasp frame, it tracks the jaws automatically and the same
# derivation serves any future gripper that lacks a camera.
DERIVED_WRIST_CAMERA = dict(
    source=None,
    pos=[0.080, 0.0, -0.155],
    fovy=55.0,
)

# What cannot be measured off OUR installed assets, hand-set from the vendor's published models,
# with provenance. Merged over the measured/derived entry in build().
#
# yam_jaw's camera is the vendor's D405 mount for the CRANK -- the gripper our shipped i2rt_yam
# actually carries -- from i2rt robot_models/gripper/crank_4310/crank_4310_d405.xml body 'camera',
# re-expressed in the grasp frame with the optical->MuJoCo flip. MolmoAct2's real rig runs the
# LINEAR_4310 instead: the real-footage calibration lives on the linear_4310 entry, and yam_jaw
# keeps the crank extrinsic as the shipped-robot reference.
VENDOR_WRIST_CAMERA = {
    "yam_jaw": dict(
        source="i2rt:crank_4310_d405.xml body 'camera'",
        pos=[0.0815, 0.0704, -0.0687],
        quat=[0.200039, 0.364881, 0.797347, 0.437131],
        fovy=58.0,
        aim=0.5378,
    ),
}

# yam_jaw finger servo: the shipped asset's kp=100 in FINGER space is a stability stand-in, not a
# transmission-mapped value -- stall force tops out at 4.7 N, while i2rt's SDK drives the real
# DM4310 to its 50 N clog-limit clamp (doc.i2rt.com; the vendor's own model XMLs are
# kinematics-only and carry no actuation at all). kp=5000 makes the CLAMP define grip strength,
# as on hardware: measured on the graft, 51.7 N stall on a 24 mm rod, actuator saturated at 50 N,
# no residual oscillation. kv=250 sets the clamp-limited FREE-CLOSING speed, 50/250 = 0.2 m/s --
# kv=100 closed the full stroke inside one 15 Hz policy frame, a teleport no camera ever saw
# mid-motion -- while leaving the stall force untouched (kv contributes nothing at rest). The
# shared versioned asset cannot be edited from this repo, so the correction is recorded here,
# next to the pin it departs from.
ACTUATOR_OVERRIDE = {
    "yam_jaw": {"gripper": dict(gainprm=[5000.0], biasprm=[0.0, -5000.0, -250.0],
                                forcerange=[-50.0, 50.0])},
}


def build_linear_4310() -> dict:
    """The i2rt LINEAR_4310 + D405, the gripper MolmoAct2's bimanual-YAM rig actually runs.

    No shipped robot carries it, so there is no source EmbodimentSpec to lift from: the model is
    the vendor's own linear_4310_d405.xml bundled under embodiment_gen/vendor/ (see its header
    for provenance and the authored actuation), and the semantics below are authored here in the
    same shapes yam_jaw's entry lifts from i2rt_yam.yaml. What CAN be derived is derived:
    keep_meshes from a trial graft, and the wrist camera from the vendor file's own `camera`
    body -- the station-README extrinsic, +Z optical axis in the ROS/OpenCV convention --
    composed with the OpenCV->MuJoCo flip diag(1,-1,-1) and re-expressed in the grasp frame.
    Verified against real MolmoAct2 D405 footage (allenai/28112025-block-02): silhouette IoU
    0.849 as-designed, 0.955 after a 6-DOF refine moving 4 mm / 1.1 deg -- one unit's mounting
    tolerance, so the CAD pose is what ships.
    """
    import numpy as np

    mjcf = "vendor/linear_4310/linear_4310_d405.xml"
    model_path = REPO / "molmo_spaces" / "embodiment_gen" / mjcf
    # the adapter body, not the vendor's own root: it flips the vendor's fingers-down frame to
    # the catalog's fingers-along-flange-+z mount convention, and graft() cancels whatever pose
    # the graft ROOT carries, so the flip must sit above the root it cancels
    graft_root = "graft_mount"

    compiled, _ = trial_graft(model_path, graft_root, drop=[])
    keep = sorted(used_meshes(compiled))
    pruned, _ = trial_graft(model_path, graft_root, drop=[], keep=set(keep))
    assert pruned.nu == compiled.nu and pruned.neq == compiled.neq, (
        f"linear_4310: pruning changed the model ({compiled.nu}->{pruned.nu} actuators, "
        f"{compiled.neq}->{pruned.neq} equality constraints)")

    source = MjSpec.from_file(str(model_path))
    camera = source.body("camera")
    site = source.site("grasp_site")
    def mat(quat):
        out = np.zeros(9)
        mujoco.mju_quat2Mat(out, np.asarray(quat, float) / np.linalg.norm(quat))
        return out.reshape(3, 3)
    camera_rotation = mat(camera.quat) @ GL_FROM_CV[:3, :3]   # OpenCV cam -> MuJoCo (GL) cam
    grasp_rotation = mat(site.quat)
    position = grasp_rotation.T @ (np.asarray(camera.pos) - np.asarray(site.pos))
    orientation_matrix = grasp_rotation.T @ camera_rotation
    orientation = np.zeros(4)
    mujoco.mju_mat2Quat(orientation, orientation_matrix.flatten())
    approach = [0.0, 0.0, 1.0]
    aim = float((orientation_matrix @ [0.0, 0.0, -1.0]) @ approach)

    def named(name):
        return {"name": name, "absolute": False}

    planner_site = dict(
        kind="add_site", scope="robot_model",
        parent_body=named(GRIPPER_PREFIX + "gripper"),
        name=named(GRIPPER_PREFIX + PLANNER_SITE),
        # the vendor grasp_site pose: between the pads, +z toward the workspace, y along the jaws
        position=[float(v) for v in site.pos],
        quaternion_wxyz=[float(v) for v in site.quat],
        xyaxes=None, group=4, rgba=[0.0, 1.0, 1.0, 1.0])

    finger = lambda side: {  # noqa: E731
        "geom_names": [], "body_names": [],
        "include_descendants_of": [named(GRIPPER_PREFIX + f"tip_{side}")],
        "exclude_descendants_of": []}

    return dict(
        name="linear_4310",
        resource_key="",                      # bundled with the repo, see GripperAsset.model_path
        resource_version="i2rt-robotics/i2rt@1276f63d (2026-08-07)",
        mjcf=mjcf,
        graft_root=graft_root,
        drop_joints=[],                       # the root body carries no joint of its own
        keep_meshes=keep,
        # The housing + D405 bracket IS the flange covering: the root body sits at the flange
        # frame carrying the wrist plate's visual and its 0.553 kg. Letting the arm dress that
        # link too parks a segment geom dead-centre in the wrist view -- the camera looks back
        # past the flange at 25 deg, and the occlusion cost a fingertip (real-frame silhouette
        # IoU 0.83 -> 0.42) -- where the real rig shows clean workspace.
        claims_flange_link=True,
        move_group=dict(
            kind="gripper", role="tool", side="center",
            root_body=named(GRIPPER_PREFIX + "gripper"),
            joint_names=[named(GRIPPER_PREFIX + "left_finger"),
                         named(GRIPPER_PREFIX + "right_finger")],
            actuator_names=[named(GRIPPER_PREFIX + "gripper")],
            leaf_frame={"kind": "site", "name": named(GRIPPER_PREFIX + PLANNER_SITE)},
            open_command=[0.04695],
            closed_command=[0.0],
            aperture=dict(
                kind="joint_expression",
                # both slides carry positive=open ranges, so the pad gap is their SUM
                # (plus the vendor's 0.06 mm closed gap, below aperture_threshold)
                terms=[
                    {"joint": named(GRIPPER_PREFIX + "left_finger"),
                     "coefficient": 1.0, "transform": "identity"},
                    {"joint": named(GRIPPER_PREFIX + "right_finger"),
                     "coefficient": 1.0, "transform": "identity"},
                ],
                offset=0.0, clamp=[0.0, 0.094])),
        controller=dict(kind="joint_position", move_group="gripper"),
        grasp_profile_id="linear_4310_parallel_jaw_v1",
        grasp_profile=dict(
            profile_version="1",
            tool_type="parallel_jaw",
            grasp_frame={"kind": "site", "name": named(GRIPPER_PREFIX + PLANNER_SITE)},
            approach_axis_in_grasp_frame=[0.0, 0.0, 1.0],
            closing_axis_in_grasp_frame=[0.0, 1.0, 0.0],
            symmetries=[{"position": [0.0, 0.0, 0.0],
                         "quaternion_wxyz": [0.0, 0.0, 0.0, 1.0]}],
            open_configuration=[0.04695],
            closed_configuration=[0.0],
            contact_groups=dict(
                left_finger=finger("left"),
                right_finger=finger("right"),
                palm={"geom_names": [],
                      "body_names": [named(GRIPPER_PREFIX + "gripper")],
                      "include_descendants_of": [], "exclude_descendants_of": []}),
            grasp_rule=dict(required_contact_groups=["left_finger", "right_finger"],
                            forbidden_contact_groups=["palm"],
                            minimum_contact_count=1, aperture_threshold=0.002),
            release_rule=dict(required_contact_groups=[], forbidden_contact_groups=[],
                              minimum_contact_count=0, aperture_threshold=None)),
        initial_qpos=[0.0, 0.0],
        sensors=[dict(kind="frame_pose", uuid="tcp_pose",
                      frame={"kind": "site", "name": named(GRIPPER_PREFIX + PLANNER_SITE)},
                      relative_to_group="base",
                      output_format="xyz_quaternion_wxyz")],
        model_operations=[planner_site],
        wrist_camera=dict(
            source="derived from the bundled vendor model's `camera` body (i2rt station-README "
                   "extrinsic, +Z optical axis, OpenCV convention), OpenCV->MuJoCo flip, "
                   "re-expressed in the grasp frame; VERIFIED on real MolmoAct2 D405 footage "
                   "(allenai/28112025-block-02): silhouette IoU 0.849 as-designed, 0.955 after "
                   "a 6-DOF refine moving 4 mm / 1.1 deg; fovy 58 = published D405 vertical FoV",
            pos=[round(float(v), 6) for v in position],
            quat=[round(float(v), 6) for v in orientation],
            fovy=58.0,
            aim=round(aim, 4)),
    )


def wrist_camera_entry(compiled, profile, frame_name: str | None) -> dict:
    """The gripper's wrist camera in its grasp frame, or the derived one when it ships none.

    Measured off the SOURCE model rather than assumed, so a gripper that moves its camera between
    resource versions is picked up by re-running this script.

    ``frame_name`` must be the site name as the SOURCE model spells it, not the rewritten uniform
    one: passing gripper/planner_grasp_site here resolved nothing for any gripper, because that site
    is created by a spec operation and the prefix is added later."""
    import numpy as np

    site = -1
    for candidate in (frame_name, "grasp_site"):
        if candidate:
            site = mujoco.mj_name2id(compiled, mujoco.mjtObj.mjOBJ_SITE, candidate)
            if site >= 0:
                break
    if site < 0 or not profile:
        return dict(DERIVED_WRIST_CAMERA)
    data = mujoco.MjData(compiled)
    mujoco.mj_forward(compiled, data)
    rotation = np.asarray(data.site_xmat[site], float).reshape(3, 3)
    origin = np.asarray(data.site_xpos[site], float)
    # a camera counts as the wrist camera if it looks roughly along the approach axis: rum also ships
    # top_camera and back_camera, which do not
    approach = np.asarray(profile["approach_axis_in_grasp_frame"], float)
    best = None
    for index in range(compiled.ncam):
        name = mujoco.mj_id2name(compiled, mujoco.mjtObj.mjOBJ_CAMERA, index) or ""
        camera_rotation = np.asarray(data.cam_xmat[index], float).reshape(3, 3)
        look = camera_rotation @ [0.0, 0.0, -1.0]
        aim = float((rotation.T @ look) @ approach)
        if aim < 0.7:
            continue
        position = rotation.T @ (np.asarray(data.cam_xpos[index], float) - origin)
        # the DESIGNED orientation, not just the position: re-deriving the aim from the
        # position alone pointed every wrist stream at the grasp point between the jaws,
        # 15.5 deg off the designed view and dominated by the gripper's own hardware
        orientation = np.zeros(4)
        mujoco.mju_mat2Quat(orientation, (rotation.T @ camera_rotation).flatten())
        if best is None or aim > best["aim"]:
            best = dict(source=name, pos=[round(float(v), 6) for v in position],
                        quat=[round(float(v), 6) for v in orientation],
                        fovy=round(float(compiled.cam_fovy[index]), 3), aim=round(aim, 4))
    if best is None:
        return dict(DERIVED_WRIST_CAMERA)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--specs", type=Path, default=REPO / "docs" / "embodiment_specs")
    ap.add_argument("--only", nargs="*", choices=sorted(SOURCES), default=sorted(SOURCES))
    ap.add_argument("--check", action="store_true",
                    help="fail if a committed entry differs instead of rewriting it")
    args = ap.parse_args()

    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    failures = []
    for name in args.only:
        if SOURCES[name] is None:
            entry = build_linear_4310()
        else:
            entry = build(name, args.specs / SOURCES[name])
        text = yaml.dump(entry, sort_keys=False, default_flow_style=False, width=100)
        out = CATALOG_DIR / f"{name}.yaml"
        if args.check:
            same = out.exists() and out.read_text() == text
            print(f"{'ok  ' if same else 'DIFF'} {out.relative_to(REPO)}")
            if not same:
                failures.append(name)
        else:
            out.write_text(text)
            print(f"wrote {out.relative_to(REPO)}: root={entry['graft_root']} "
                  f"claims_flange={entry['claims_flange_link']} "
                  f"drop={entry['drop_joints']} meshes={len(entry['keep_meshes'])} "
                  f"profile={entry['grasp_profile_id']} sensors={len(entry['sensors'])} "
                  f"ops={len(entry['model_operations'])}")
    if failures:
        raise SystemExit(f"catalog out of date: {', '.join(failures)}")


if __name__ == "__main__":
    main()
