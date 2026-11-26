import mujoco
import pytest
from mujoco import MjSpec

import numpy as np

from molmo_spaces.embodiment_gen.constants import GRIPPER_PREFIX, PLANNER_SITE, TCP_SITE
from molmo_spaces.embodiment_gen.grippers import graft, load_catalog

CATALOG = sorted(load_catalog())


def _stub_arm():
    """A bare flange to graft onto -- the smallest thing that stands in for a sampled arm."""
    arm = MjSpec()
    flange = arm.worldbody.add_body(name="flange", pos=[0, 0, 0.3])
    flange.add_geom(name="flange_geom", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.03, 0.01, 0])
    return arm, flange


def _names(model, objtype, count):
    return [mujoco.mj_id2name(model, objtype, i) for i in range(count)]


def test_catalog_is_populated():
    assert set(CATALOG) >= {"robotiq_2f85", "yam_jaw", "linear_4310"}


@pytest.mark.parametrize("name", CATALOG)
def test_graft_compiles_and_keeps_gripper_semantics(name):
    """The graft must carry the gripper's actuator and its coupling constraints. That is
    what lets a generated robot inherit gripper behaviour instead of re-deriving it -- and
    it is why a generated gripper is never 2 independent finger joints."""
    arm, flange = _stub_arm()
    graft(arm, flange, name)
    model = arm.compile()

    assert model.nu == 1, f"{name}: expected one gripper actuator, got {model.nu}"
    assert model.neq >= 1, f"{name}: coupling constraints did not survive the graft"

    for objtype, count in ((mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
                           (mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu)):
        for element in _names(model, objtype, count):
            assert element.startswith(GRIPPER_PREFIX), f"{name}: {element} escaped the prefix"


@pytest.mark.parametrize("name", CATALOG)
def test_one_command_closes_both_fingers(name):
    """A single ctrl input must move every declared finger joint: the 1-DoF coupling lives
    in the shipped asset, so nothing in the generator has to reproduce it."""
    arm, flange = _stub_arm()
    graft(arm, flange, name)
    model = arm.compile()
    joints = [j["name"] for j in load_catalog()[name].move_group["joint_names"]]

    data = mujoco.MjData(model)
    data.ctrl[:] = model.actuator_ctrlrange[:, 1]
    for _ in range(3000):
        mujoco.mj_step(model, data)

    for joint in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        assert jid >= 0, f"{name}: move group names {joint}, absent from the grafted model"
        assert abs(float(data.qpos[model.jnt_qposadr[jid]])) > 1e-3, (
            f"{name}: {joint} did not move under a closing command")


@pytest.mark.parametrize("name", CATALOG)
def test_grasp_and_leaf_frames_are_reachable(name):
    """Every frame the entry names must be resolvable: present in the grafted MJCF, or created
    by one of the entry's own model operations.

    The grasp frame and the move group's leaf frame are NOT required to be the same site. Every
    catalog grasp profile names ``planner_grasp_site`` -- uniformly, so a planner never has to ask
    which gripper it is holding -- while the leaf frame stays whatever the shipped move group
    declares. Asserting the two were equal is what this test used to do, and it only passed because
    two of the three grippers happened to agree."""
    asset = load_catalog()[name]
    arm, flange = _stub_arm()
    graft(arm, flange, name)
    model = arm.compile()

    def resolvable(site):
        added = any(operation.get("kind") == "add_site"
                    and operation["name"]["name"] == site
                    for operation in asset.model_operations)
        return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site) >= 0 or added

    grasp = asset.grasp_profile["grasp_frame"]["name"]["name"]
    leaf = asset.move_group["leaf_frame"]["name"]["name"]
    assert grasp == f"{GRIPPER_PREFIX}{PLANNER_SITE}", (
        f"{name}: grasp profile names {grasp}, not the uniform planner frame")
    for site in (grasp, leaf):
        assert resolvable(site), f"{name}: nothing supplies {site}"


@pytest.mark.parametrize("name", CATALOG)
def test_source_spec_is_reusable(name):
    """attach_body consumes its argument, so every graft must get a fresh copy. Two grafts
    of the same gripper onto different arms must both compile."""
    for _ in range(2):
        arm, flange = _stub_arm()
        graft(arm, flange, name)
        assert arm.compile().nu == 1


# Pad geoms per gripper whose distal face defines the pad line (the most distal one per finger is used).
_PADS = {
    "robotiq_2f85": (["gripper/left_pad1", "gripper/left_pad2"], ["gripper/right_pad1", "gripper/right_pad2"]),
    "rum": (["gripper/left_fingertip"], ["gripper/right_fingertip"]),
    "linear_4310": (["gripper/tip_left_pad_lo", "gripper/tip_left_pad_hi"],
                    ["gripper/tip_right_pad_lo", "gripper/tip_right_pad_hi"]),
    # yam's jaw geoms are unnamed: every geom under the finger body (the jaw, 11 geoms) is a candidate
    "yam_jaw": ("body:gripper/link_left_finger", "body:gripper/link_right_finger"),
}


def _pad_geoms(model, candidates):
    """Geom ids for a _PADS entry: a list of geom names, or ``body:<name>`` for a whole subtree."""
    if isinstance(candidates, str):
        root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, candidates.removeprefix("body:"))
        def under(body):
            while body > 0 and body != root:
                body = int(model.body_parentid[body])
            return body == root
        return [g for g in range(model.ngeom) if under(int(model.geom_bodyid[g]))]
    return [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in candidates]


def _distal_face_centre(model, data, gid, approach):
    """Centre of the geom's AABB face furthest along ``approach`` (its 'lower' edge, gripper pointing down)."""
    xmat = data.geom_xmat[gid].reshape(3, 3)
    aabb = model.geom_aabb[gid]
    centre = data.geom_xpos[gid] + xmat @ aabb[:3]
    local = xmat.T @ approach
    k = int(np.argmax(np.abs(local)))
    face = np.zeros(3)
    face[k] = np.sign(local[k]) * aabb[3 + k]
    return centre + xmat @ face


@pytest.mark.parametrize("name", sorted(_PADS))
def test_tcp_frame_sits_at_the_closed_pad_midpoint(name):
    """The catalog's second add_site, ``tcp_frame``, is the canonical TCP: a rigid frame on the gripper
    root whose origin is the midpoint of the pad line when the gripper is FULLY CLOSED, GL-style like a
    camera: x toward the right pad, y wrist-image-up, z back into the wrist (the approach is -z). It is a
    per-family constant because the gripper is grafted verbatim; this checks the
    constant against the grafted model itself, so a re-measured asset cannot drift away unnoticed."""
    asset = load_catalog()[name]
    op = next(o for o in asset.model_operations if o["name"]["name"] == f"{GRIPPER_PREFIX}{TCP_SITE}")
    # A flange with a (stiff) joint, unlike _stub_arm's welded one: MuJoCo's parent-child contact filter
    # treats bodies welded to the WORLD as the world and never filters them, so on a welded flange rum's
    # fingers jam against their own gripper body. On a real arm the gripper base moves with the wrist and
    # finger-vs-base contacts are filtered out; the joint reproduces that.
    arm = MjSpec()
    flange = arm.worldbody.add_body(name="flange", pos=[0, 0, 0.3])
    flange.add_joint(name="flange_stub", type=mujoco.mjtJoint.mjJNT_HINGE, axis=[0, 0, 1],
                     stiffness=1e5, damping=1e2)
    graft(arm, flange, name)
    parent = arm.body(op["parent_body"]["name"])
    parent.add_site(name=op["name"]["name"], pos=op["position"], quat=op["quaternion_wxyz"])
    model = arm.compile()
    model.opt.gravity[:] = 0.0
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, op["name"]["name"])

    def settle(ctrl):
        """Pads' distal-face centres and the site pose after driving the actuator to ``ctrl``."""
        data = mujoco.MjData(model)
        for _ in range(3000):
            data.ctrl[:] = ctrl
            mujoco.mj_step(model, data)
        origin, frame = data.site_xpos[sid].copy(), data.site_xmat[sid].reshape(3, 3).copy()
        approach = -frame[:, 2]
        pads = []
        for candidates in _PADS[name]:
            faces = [_distal_face_centre(model, data, g, approach) for g in _pad_geoms(model, candidates)]
            pads.append(max(faces, key=lambda p: float(p @ approach)))
        return origin, frame, pads

    # "fully closed" is whichever ctrl extreme brings the pads together -- the raw asset's ctrl sign
    # is not uniform across grippers (the emitted spec normalises it, the bare graft does not)
    origin, frame, pads = min((settle(c) for c in model.actuator_ctrlrange[0]),
                              key=lambda r: float(np.linalg.norm(r[2][1] - r[2][0])))
    gap = float(np.linalg.norm(pads[1] - pads[0]))
    assert gap < 0.006, f"{name}: gripper did not close (pad gap {gap * 1000:.1f} mm)"
    offset = frame.T @ (0.5 * (pads[0] + pads[1]) - origin)
    assert np.linalg.norm(offset) < 0.001, (
        f"{name}: tcp_frame is {offset * 1000} mm from the closed pad midpoint (site frame)")
    # the closing axis is +/-x of the frame, so the pads never leave the frame's x axis
    assert abs(float(frame[:, 1] @ (pads[1] - pads[0]))) < 0.002, f"{name}: pad line not along the frame's x axis"
