"""Convert a datagen run into wods items.

One wods ITEM per episode, one SHARD per camera. That is the natural mapping: an item is a thing that
happened, a shard is a view of it, and the trajectory-level tensors (proprio, action, target) are the
same for every view while video, intrinsics and c2w are per view.

The pipeline already writes each camera's frames as an mp4 next to the h5, and the h5's
``obs/sensor_data/<camera>`` row holds that filename rather than pixels -- so the video is copied in
RAW, via ``video_raw=``, to avoid a decode/re-encode round trip that would cost quality for nothing.

Camera frames follow the convention the consuming pipeline already established (virtual_vla
``sim_replay/common.CV_TO_WODS``, used by ``add_transfer_queries_to_wods``): wods ``c2w`` is
camera-to-world where the CAMERA-LOCAL axes are ``+x`` right, ``+y`` up, ``+z`` forward (the look
direction), and the WORLD frame is the simulator's own, untouched.

That is a camera-local axis flip, not a world change, and WHICH flip depends on what the producer
records. MolmoSpaces records OpenGL cameras (y up, looking along -z), so the flip is the z column,
``diag(1, 1, -1)``. virtual_vla starts from ManiSkill's OpenCV cameras (y down, +z forward) and flips
y instead. Both land on the same invariant, which is what this file asserts: wods ``+z`` IS the look
direction, ``+y`` is up, the position is unchanged, and the resulting frame is left-handed in a
right-handed world (``det = -1``).

An earlier version of this file got that wrong in the most dangerous way: it LEFT-multiplied by a
world-side y/z swap, on the theory that "wods is y-up" described the world. It does not -- it
describes the camera. That version silently rotated every camera and moved every camera position,
which no downstream consumer would have flagged.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import torch

from molmo_spaces.embodiment_gen.representations import (
    Kinematics, camera_frame, c2w_from_cam2world, ctrl_to_qpos, node_graph_tensors,
    driver_semantic_qpos_map_to_semantic_proprio_map, proprio_to_semantic_node_proprio,
    semantic_node_proprio_to_abstract_node_proprio, relative_node_action, pose_from_points)

def _unpack(row) -> object:
    """Rows the recorder packs as null-padded JSON in a fixed-width uint8 field."""
    raw = np.asarray(row).tobytes().rstrip(b"\x00")
    return json.loads(raw.decode()) if raw else None


def _packed_string(row) -> str:
    return np.asarray(row).tobytes().rstrip(b"\x00").decode(errors="ignore")


# The canonical order of move groups within a flattened proprio/action vector. Declared, not sorted:
# virtual_vla's transfer-query pipeline aligns "the first n_arm action dims 1:1 with the first n_arm
# revolute nodes, then the gripper drivers" (NodeSpec.link_order / n_arm / driver_qpos), so a vector
# whose arm block does not come first cannot be consumed by that convention. Alphabetical sorting
# happens to give arm-before-gripper but would put "base" between them, and would silently reorder for
# any future group name.
CANONICAL_GROUPS = ("arm", "gripper")


def canonical_group_order(entry: dict | None) -> list[str]:
    """Move groups in the canonical order: arm, gripper, then anything else the robot declares.

    Trailing groups are sorted for determinism and kept rather than dropped -- a fixed-base arm's
    "base" group is degenerate, but silently discarding a group would make the vector width depend on
    which groups a robot happens to have, with no record of it."""
    if not entry:
        return []
    leading = [g for g in CANONICAL_GROUPS if g in entry]
    return leading + sorted(g for g in entry if g not in CANONICAL_GROUPS)


def _flatten_groups(entry: dict | None, order: list[str]) -> list[float]:
    """A per-move-group dict (``{"arm": [...], "gripper": [...]}``) as one flat vector.

    The order is ``canonical_group_order`` and is recorded in the sidecar index, so a consumer can
    slice the vector back apart -- without it a flattened vector is uninterpretable."""
    if not entry:
        return []
    out: list[float] = []
    for group in order:
        value = entry.get(group)
        if value is None:
            continue
        out.extend(np.atleast_1d(np.asarray(value, dtype=np.float64)).ravel().tolist())
    return out


def group_widths(entry: dict | None, order: list[str]) -> dict[str, int]:
    """Per-group element count, from a populated frame. Needed because the groups are NOT equal width
    -- a fixed-base arm records ``{arm: 7, base: 0, gripper: 1}`` -- so a flat vector cannot be split
    by dividing its length."""
    if not entry:
        return {}
    return {g: int(np.atleast_1d(np.asarray(entry[g])).size) for g in order if g in entry}


def _merge_action(arm: np.ndarray, arm_order: list[str], arm_widths: dict[str, int],
                  full: np.ndarray, full_order: list[str],
                  full_widths: dict[str, int]) -> tuple[np.ndarray, list[str]]:
    """``joint_pos``'s arm block, then ``commanded_action``'s remaining groups, spliced by NAME."""
    if not full_order or arm.shape[1] == 0:
        return (full, full_order) if full_order else (arm, arm_order)
    bounds, offset = {}, 0
    for group in full_order:
        span = full_widths.get(group, 0)
        bounds[group] = (offset, offset + span)
        offset += span
    keep = [g for g in full_order if g not in arm_order and full_widths.get(g, 0) > 0]
    if not keep:
        return arm, arm_order
    extra = [full[:, slice(*bounds[g])] for g in keep]
    return np.concatenate([arm, *extra], axis=1), list(arm_order) + keep


def first_populated(dataset) -> dict | None:
    """The first frame that actually carries a value.

    The group order cannot be read off frame 0: at the first step no action has been commanded yet, so
    ``commanded_action[0]`` is empty and an order derived from it is empty too -- which silently
    produced a zero-width action array."""
    for i in range(dataset.shape[0]):
        entry = _unpack(dataset[i])
        if entry:
            return entry
    return None


def _stack_packed(dataset, order: list[str], hold: bool = False) -> np.ndarray:
    """``(T, width)`` from the packed rows, in ``order``.

    ``hold`` forward-fills a frame that carries no value from the previous frame, and back-fills the
    leading gap from the first frame that does. That is the right reading for a COMMAND -- a step that
    re-commands nothing means "keep the current target" -- whereas zero-filling would record a command
    to drive every joint to 0. Measurements (raw qpos) are never held: a missing measurement is a gap,
    not a repeat."""
    rows = [_flatten_groups(_unpack(dataset[i]), order) for i in range(dataset.shape[0])]
    width = max((len(r) for r in rows), default=0)
    if not width:
        return np.zeros((len(rows), 0), dtype=np.float32)
    out = np.zeros((len(rows), width), dtype=np.float32)
    previous: list[float] | None = None
    for i, row in enumerate(rows):
        if len(row) == width:
            previous = row
        elif hold and previous is not None:
            row = previous
        out[i, :len(row)] = row
    if hold:
        first = next((i for i, r in enumerate(rows) if len(r) == width), None)
        if first:
            out[:first] = out[first]
    return out


def _action_target_rows(action: np.ndarray, rows: list[dict], spec: dict) -> list[dict]:
    """Per-frame qpos dicts at the ACTION TARGET, for the action-geometry FK.

    ``action`` is the flat merged absolute action: the arm block (from ``actions/joint_pos``)
    followed by the raw gripper command (from ``commanded_action``). The command is converted into
    each DRIVER's qpos through the entry's measured two-point calibration -- the same conversion the
    relative action label uses, because inferring it from declared joint limits was wrong by 0.78 rad
    on rum and direction-inverted on yam's mirrored finger."""
    ctrl = spec.get("gripper_ctrl_range")
    at_ctrl = spec.get("gripper_qpos_at_ctrl")
    n_arm_spec = int(spec.get("n_arm", 0))
    # each driver's COLUMN inside the gripper move-group (driver_qpos is full-model, possibly non-contiguous,
    # so address - n_arm mis-indexed the move-group vector and skipped robotiq's right driver)
    driver_cols = [c - n_arm_spec for c in driver_semantic_qpos_map_to_semantic_proprio_map(spec["driver_qpos"], n_arm_spec)] \
        if spec.get("driver_qpos") else []
    out = []
    for index, row in enumerate(rows):
        arm = list(row.get("arm") or [])
        gripper = list(row.get("gripper") or [])
        n_arm = len(arm)
        target_arm = list(np.asarray(action[index, :n_arm], float)) if action.shape[1] >= n_arm \
            else arm
        target_gripper = list(gripper)
        if action.shape[1] > n_arm and ctrl and at_ctrl and gripper:
            for position, col in enumerate(driver_cols):
                if 0 <= col < len(target_gripper) and position < len(at_ctrl[0]):
                    target_gripper[col] = ctrl_to_qpos(
                        float(action[index, n_arm]), ctrl, at_ctrl, position)
        out.append({"arm": target_arm, "gripper": target_gripper})
    return out


# ---------------------------------------------------------------------------------------------------
# Post-add_transfer_queries fields. This converter emits what virtual_vla's Stage 2
# (scripts/add_transfer_queries_to_wods.py) emits, so its output is directly trainable rather than
# needing that pass re-run: raw `proprio`, per-node centered `node_proprio`, per-node RELATIVE
# `action`, and the static `node_graph`.
#
# Units stay NATIVE (radians and metres). The prismatic x10 into decimetres is a MODEL-boundary
# transform -- joint_types.METERS_TO_DECIMETERS applied by to_model_joint_units, called from
# transfer_query/dataset.py:211,350,357 and eval_transfer_query.py:571 -- so applying it on disk here
# would make the dataset layer scale twice.


def convert_run(run_dir: Path, dataset: str, seed: int, task_name: str = "", scene: str = "",
                successful_only: bool = True, node_spec_path: Path | None = None,
                entry: Path | None = None) -> list[str]:
    """Write every episode under ``run_dir`` as a wods item. Returns the keys written.

    ``dataset`` is an fsspec target as a STRING and is only ever touched through the wods API, which
    is fsspec-native. ``run_dir``, ``node_spec_path`` and ``entry`` are LOCAL paths: they are read with
    h5py and compiled by MuJoCo, neither of which can open a remote object.

    Keys are ``gen_<seed>/<task>/<scene>_house_<N>_traj<NN>`` -- embodiment, then task, then
    trajectory. wods keys are paths, so this is a directory hierarchy on disk and groups a robot's
    episodes under the robot, which is how a consumer asks "what can this robot do". It also matches
    the shape virtual_vla's emb-in-KEY layout uses (``<task>_traj<NN>/<emb>``) in kind, differing
    only in which axis leads.

    ``scene`` names the scene dataset (and split) the episode ran in, e.g.
    ``procthor-objaverse-train``. It has to come from the caller because the run directory only
    records the house INDEX (``house_12``), and an index means nothing without the dataset it
    indexes into -- ithor house 12 and procthor house 12 are unrelated scenes (docs/assets.md)."""
    from wods.v2 import (write_wods_key_done, write_wods_metadata, write_wods_shard,
                         write_wods_shard_done)

    spec = json.loads(node_spec_path.read_text()) if node_spec_path else None
    kinematics = Kinematics(entry, spec) if entry and spec else None
    written: list[str] = []
    for h5_path in sorted(run_dir.rglob("*.h5")):
        house_dir = h5_path.parent
        with h5py.File(h5_path, "r") as handle:
            for index, name in enumerate(sorted(handle)):
                traj = handle[name]
                success = bool(np.asarray(traj["success"]).any())
                if successful_only and not success:
                    continue
                index_label = name.removeprefix("traj_")
                parts = [f"gen_{seed}"]
                if task_name:
                    parts.append(task_name)
                scene_house = f"{scene}_{house_dir.name}" if scene else house_dir.name
                parts.append(f"{scene_house}_traj{int(index_label):02d}"
                             if index_label.isdigit() else f"{scene_house}_{name}")
                key = "/".join(parts)
                if _write_item(traj, house_dir, dataset, key, seed, index, spec, kinematics,
                               write_wods_metadata, write_wods_shard, write_wods_shard_done,
                               write_wods_key_done):
                    written.append(key)
    return written


def _write_item(traj, house_dir: Path, dataset: str, key: str, seed: int, index: int,
                spec: dict | None, kinematics,
                write_metadata, write_shard, shard_done, key_done) -> bool:
    scene = np.asarray(traj["obs_scene"]).item()
    scene = json.loads(scene.decode() if isinstance(scene, bytes) else scene)
    frames = int(traj["success"].shape[0])

    # Trajectory-level tensors, shared by every view.
    #
    # proprio is RAW native qpos and action is the ABSOLUTE joint-position target, which is what this
    # stage is supposed to emit: virtual_vla's add_transfer_queries consumes exactly that and derives
    # the centered node_proprio (qpos minus the per-joint mid-range (lo+hi)/2) and the per-node
    # RELATIVE action (target minus current qpos) itself. Centering or differencing here would make
    # that stage subtract twice, and the prismatic decimetre scaling it applies is a model-boundary
    # concern, not a disk format.
    qpos_order = canonical_group_order(first_populated(traj["obs/agent/qpos"]))
    proprio = _stack_packed(traj["obs/agent/qpos"], qpos_order)
    # The action is assembled from BOTH recorded fields, because neither alone is complete, and each
    # is emitted by a sensor with a documented meaning (molmo_spaces/env/sensors.py):
    #   actions/joint_pos        LastCommandedJointPosSensor -> _cmd_joint_pos(robot), the position
    #                            controller's CURRENT TARGET. Emitted every frame; {} only when
    #                            task.is_terminal() (a sentinel).
    #   actions/commanded_action LastActionSensor -> task.last_action. {} "if no action has been set
    #                            yet (padding value)" -- measured empty at exactly frame [0].
    # So frame 0's gap is a documented padding value, not missing data, and the back-fill below fills
    # precisely that.
    #
    # But action[i] is the command that ACHIEVED state[i] (sensors polled at end of task.step), not the
    # command to reach state[i+1] (=action[i+1]) -- re-sim confirms action[i+1] from state[i] reproduces
    # state[i+1] exactly; |a[i]-q[i+1]|<|a[i]-q[i]| was the servo tracking lead, not causation. Shift below.
    #
    # An earlier version read the arm from commanded_action and forward-filled the gap. That was
    # papering over a field choice, not a real gap: the frame-0 arm action was recorded all along.
    # Only the frame-0 GRIPPER is genuinely absent, and it is back-filled from frame 1 -- the gripper
    # is not commanded until the first step, so "the same as the next frame" is what was in effect.
    arm_first = first_populated(traj["actions/joint_pos"])
    arm_order = canonical_group_order(arm_first)
    # hold on joint_pos too: LastCommandedJointPosSensor returns {} when task.is_terminal(), a
    # sentinel. Zero-filling that row would record "drive every joint to 0" on the final step.
    arm_action = _stack_packed(traj["actions/joint_pos"], arm_order, hold=True)
    full_first = first_populated(traj["actions/commanded_action"])
    grip_order = canonical_group_order(full_first)
    grip_action = _stack_packed(traj["actions/commanded_action"], grip_order, hold=True)
    action, action_order = _merge_action(
        arm_action, arm_order, group_widths(arm_first, arm_order),
        grip_action, grip_order, group_widths(full_first, grip_order))
    # "target" is the pose the action aims at, which the recorder stores as the phase target
    target = np.asarray(traj["obs/extra/phase_target_pose"], dtype=np.float32) \
        if "obs/extra/phase_target_pose" in traj else None

    # Shift action + target up one so proprio[i] pairs with the command driving state[i]->state[i+1];
    # proprio / node geometry stay (state i); terminal holds its last command.
    if len(action) > 1:
        action = np.concatenate([action[1:], action[-1:]])
        if target is not None and getattr(target, "size", 0):
            target = np.concatenate([target[1:], target[-1:]])

    # The post-Stage-2 fields, when the entry declared its node set. Absent it, the item still carries
    # raw proprio and absolute actions and a consumer can run add_transfer_queries_to_wods itself.
    world_anchors = action_anchors = pose_targets = pose_states = None
    if kinematics is not None:
        if "obs/extra/robot_base_pose" not in traj:
            raise ValueError("node geometry needs obs/extra/robot_base_pose to place base-local "
                             "FK in the recorder's world frame; this trajectory lacks it")
        base = np.asarray(traj["obs/extra/robot_base_pose"], np.float64)
        rows = [_unpack(traj["obs/agent/qpos"][i]) or {} for i in range(frames)]
        world_anchors = np.stack([kinematics.anchors(row, base[i])
                                  for i, row in enumerate(rows)])             # (T, N, 2, 3)
        # The TCP node's ACHIEVED frame (the other rows have no frame and come out zero) as four world
        # points (centre, +x, +y, +z), so the per-view projection below is the one the anchors get.
        pose_states = np.stack([kinematics.poses(row, base[i])
                                for i, row in enumerate(rows)])                # (T, N, 4, 3)
        if spec:
            # The ACTION-TARGET anchors: the same FK, evaluated at the pose the executed action
            # commands. The target arm block is the absolute joint-position action; the gripper
            # DRIVER entries are the raw command mapped through the entry's measured ctrl->qpos
            # calibration; the gripper's passive linkage joints keep their CURRENT values, since the
            # command says nothing about them and zeroing them would move the 4-bar bodies.
            targets = _action_target_rows(action, rows, spec)
            action_anchors = np.stack([kinematics.anchors(row, base[i])
                                       for i, row in enumerate(targets)])
            # The TCP node's COMMANDED frame (other rows zero) at the same action-target configuration. Kept in
            # HOUSE-WORLD (the recorder's frame, where base_pose and cam2world live) -- a label does not depend
            # on which camera looks at it -- so the same (T, N, 3, 4) track is written into every shard.
            pose_targets = np.stack([kinematics.poses(row, base[i])
                                     for i, row in enumerate(targets)])            # (T, N, 4, 3) world points

    node_proprio = node_action = graph = None
    if spec:
        driver_qpos = spec["driver_qpos"]
        n_arm_i, n_nodes = int(spec["n_arm"]), len(spec["nodes"])
        dpi = driver_semantic_qpos_map_to_semantic_proprio_map(driver_qpos, n_arm_i)
        center = proprio_to_semantic_node_proprio(
            np.asarray(spec["qpos_center"], np.float32)[None], driver_qpos, n_arm_i, n_nodes)[0]
        node_proprio = semantic_node_proprio_to_abstract_node_proprio(
            proprio_to_semantic_node_proprio(proprio, dpi, n_arm_i, n_nodes), center)
        node_action = relative_node_action(action, proprio, spec, dpi,
                                           spec.get("gripper_ctrl_range"),
                                           spec.get("gripper_qpos_at_ctrl"))
        graph = node_graph_tensors(spec)

    # The caption is the task's own natural-language instruction -- what a caption is for.
    caption = scene.get("task_description") or scene.get("text") or ""
    write_metadata(dataset, key, caption=caption)

    # No sidecar. Everything a consumer needs is inside the wods item or recoverable from the KEY:
    #
    #   caption               -> write_wods_metadata
    #   node names/graph      -> node_graph in every shard
    #   embodiment, task      -> the key itself, gen_<seed>/<task>/<house>_traj<NN>
    #   proprio/action layout -> node_graph.node_names for the per-node tensors
    #
    # Three things genuinely leave the dataset, and are recoverable only via the library:
    #
    #   * the RAW ``proprio`` group layout. It is a flattened {arm, gripper, base} dict and the widths
    #     are not in the item. The key names the embodiment, so <library>/gen_<seed>/node_spec.json
    #     gives n_arm and driver_qpos, which is what splits it. The per-node tensors do not have this
    #     problem -- node_graph names every column.
    #   * the per-episode success flag. Only successful episodes are converted by default, so the
    #     dataset's existence is the flag; a mixed dataset would lose the distinction.
    #   * the pickup object and place receptacle names. The caption names them in prose but not as
    #     identifiers, so an exact object-level query is no longer possible from the item alone.
    shards = 0
    for camera in sorted(traj["obs/sensor_data"]):
        if camera.endswith("_depth"):
            continue                      # depth rides along with its colour shard below
        video_name = _packed_string(traj[f"obs/sensor_data/{camera}"])
        video_path = house_dir / video_name
        if not video_path.is_file():
            continue
        params = traj.get(f"obs/sensor_param/{camera}")
        if params is None:
            continue

        intrinsics = np.asarray(params["intrinsic_cv"], dtype=np.float32)
        c2w = c2w_from_cam2world(np.asarray(params["cam2world_gl"], dtype=np.float32))
        # timestamps from the policy step, which the recorder writes in ms
        step_ms = float(scene.get("policy_dt_ms") or 0.0) or None
        count = min(len(intrinsics), len(c2w), frames)
        timestamps = (torch.arange(count, dtype=torch.float32) * (step_ms / 1000.0)
                      if step_ms else torch.arange(count, dtype=torch.float32))

        if world_anchors is not None:
            cam2world = np.asarray(params["cam2world_gl"], float)[:count]
            payload_geometry = camera_frame(world_anchors[:count], cam2world)
            payload_action_geometry = (camera_frame(action_anchors[:count], cam2world)
                                       if action_anchors is not None else None)
            payload_action_pose = (pose_from_points(pose_targets[:count])
                                   if pose_targets is not None else None)           # (T, N, 3, 4) commanded, WORLD
            payload_pose = (pose_from_points(camera_frame(pose_states[:count], cam2world))
                            if pose_states is not None else None)                   # (T, N, 3, 4) achieved, camera
        else:
            payload_geometry = payload_action_geometry = payload_action_pose = payload_pose = None

        payload = {
            "video_raw": video_path.read_bytes(),
            "timestamps": timestamps,
            "intrinsics": torch.from_numpy(intrinsics[:count]),
            "c2w": torch.from_numpy(c2w[:count]),
        }
        if proprio.size:
            payload["proprio"] = torch.from_numpy(proprio[:count])
        if node_action is not None:
            # the wods `action` field IS the regression label, so it holds the per-node RELATIVE
            # action -- the absolute target is recoverable as node_action + node_proprio + center
            payload["action"] = torch.from_numpy(node_action[:count])
        elif action.size:
            payload["action"] = torch.from_numpy(action[:count])
        if node_proprio is not None:
            payload["node_proprio"] = torch.from_numpy(node_proprio[:count])
        if graph is not None:
            payload["node_graph"] = graph
        if payload_geometry is not None:
            payload["node_geometry"] = torch.from_numpy(payload_geometry.astype(np.float32))
        if payload_action_geometry is not None:
            payload["node_action_geometry"] = torch.from_numpy(
                payload_action_geometry.astype(np.float32))
        if payload_pose is not None:
            payload["node_pose"] = torch.from_numpy(payload_pose.astype(np.float32))
        if payload_action_pose is not None:
            payload["node_action_pose"] = torch.from_numpy(payload_action_pose.astype(np.float32))
        if target is not None and target.size:
            payload["target"] = torch.from_numpy(target[:count])

        write_shard(path=dataset, key=key, shard=camera, **payload)
        shard_done(dataset, key, camera)
        shards += 1

    if not shards:
        return False
    key_done(dataset, key)       # LAST: the marker is what makes the item readable
    return True
