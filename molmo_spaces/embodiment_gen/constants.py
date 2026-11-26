"""Generator-wide constants. Nothing embodiment-specific belongs here -- per-robot facts
live in the sampled params, per-gripper facts in ``gripper_catalog/*.yaml``."""

from __future__ import annotations

from pathlib import Path

# Bump when sampling or realization changes MEANING, i.e. when the same seed would produce
# a different robot. Recorded in every emitted spec so a frozen dataset can detect drift.
#
# 2: the sampler stopped drawing cameras (placement belongs to the camera system, not the robot)
#    and the builder started emitting a derived wrist_camera. BOTH halves qualify. The sampling half
#    is the one that bites: dropping those draws shifts every later draw in the same stream, so a
#    seed maps to a DIFFERENT robot than it did at version 1 -- measured on five seeds recorded as
#    task-successful, three now build a different gripper. Any seed list from version 1 is stale.
# 3: the sampled yam-style gripper is linear_4310 instead of yam_jaw -- MolmoAct2's bimanual-YAM
#    rig runs the LINEAR_4310 (its repo pins GripperType.LINEAR_4310, and the vendor D405
#    extrinsic reprojects onto the real wrist streams at IoU 0.955), while yam_jaw is the crank
#    our shipped i2rt_yam asset carries. Same draw POSITION in the stream, so only the gripper
#    identity changes per seed; yam_jaw stays in the catalog as the shipped-robot reference.
#    Also two mount corrections, same version since 3 never shipped: linear_4310 clocks -90 deg
#    about the flange axis (the vendor bolts it a quarter-turn from the crank; a YAM-geometry arm
#    now reproduces the station's qpos-to-pose mapping and wrist-limit arc exactly), and rum's
#    graft gains mount_quat so its fingers point along flange +z instead of sideways (+x).
# 4: lite6_gripper_wide joins GRIPPERS -- a fourth draw target reshuffles every seed's gripper.
GENERATOR_VERSION = 4

CATALOG_DIR = Path(__file__).resolve().parent / "gripper_catalog"

# The grippers the sampler draws from, declared EXPLICITLY rather than read from CATALOG_DIR:
# sampling must not depend on directory contents, or adding a catalog entry would silently
# reassign the gripper of every seed. Changing one is a deliberate GENERATOR_VERSION bump.
# A test asserts this matches the catalog.
GRIPPERS = ("robotiq_2f85", "rum", "linear_4310", "lite6_gripper_wide")

# The graft prefix. Every name the gripper contributes lands under it, which is the
# convention MolmoSpaces' own composed robots already use (franka_droid, franka_cap ship
# gripper/base, gripper/left_driver_joint, gripper/grasp_site).
GRIPPER_PREFIX = "gripper/"

# Body names the generated arm always defines. The flange is where the gripper attaches.
BASE_BODY = "base_link"
FLANGE_BODY = "flange"

# The tool plate site, named as franka_droid names it so an MjcfCameraConfig or a spec operation
# can reference it by the conventional name.
ATTACHMENT_SITE = "attachment_site"

# The site every gripper's grasp profile points at, so a planner never has to ask which gripper it
# is holding. Each catalog entry carries an add_site operation creating it at the pose the shipped
# profile named, which makes the name uniform without moving the frame. Not uniform, this cost a
# grasp outright: yam's profile named planner_grasp_site while its MJCF only has grasp_site, 49.7 mm
# away and 44.1 mm from the jaws, so a demo aiming at the MJCF site closed on nothing.
PLANNER_SITE = "planner_grasp_site"
# The canonical TCP, declared per gripper in the catalog as a second add_site: a rigid frame on the
# gripper root body at the midpoint of the pad line when the gripper is FULLY CLOSED, z = approach,
# x = image-up in the wrist camera, y = closing axis toward the image-right pad.
TCP_SITE = "tcp_frame"

# The wrist camera every robot declares, named uniformly for the same reason PLANNER_SITE is: an
# MjcfCameraConfig resolves its camera BY NAME, and the grippers disagree (robotiq ships
# wrist_camera, rum ships wrist_cam, yam_jaw ships none). GenericOmniPurposeCameraSystem names this
# one; a test asserts the two agree, since a mismatch is a hard failure at scene setup.
WRIST_CAMERA = "wrist_camera"
WRIST_CAMERA_MOUNT = "wrist_camera_mount"

MJCF_NAME = "model.xml"
NODE_SPEC_NAME = "node_spec.json"
SPEC_NAME = "spec.yaml"
PARAMS_NAME = "params.json"

# Library layout: entries are bucketed by ``seed // BUCKET_SIZE`` so directories stay listable.
# Sized against the ~1/1000 acceptance rate: 1e6 seeds per bucket keeps ~1e3 KEPT entries in each.
# At 1e3 a bucket held ~1 entry, so a billion-seed sweep made ~1e6 near-empty buckets. BUCKET_PAD
# fixes the bucket-index width so a directory/prefix listing sorts numerically (a min-width pad goes
# non-uniform once the index gains a digit); 6 digits covers seeds up to ~1e12.
BUCKET_SIZE = 1_000_000
BUCKET_PAD = 6
