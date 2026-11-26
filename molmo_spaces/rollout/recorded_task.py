"""Restore a replay bundle's recorded task targets after exact scene reconstruction."""
from __future__ import annotations

import base64
import copy
import json
from pathlib import Path
import pickle


def recover_legacy_color_targets(task, bundle):
    """Recover a missing target only when the recorded reward identifies it uniquely."""
    import mujoco
    import numpy as np

    from molmo_spaces.env.data_views import create_mlspaces_body
    from molmo_spaces.utils.mj_model_and_data_utils import body_aabb, descendant_geoms
    from molmo_spaces.utils.pose import pose_mat_to_7d

    bundle = Path(bundle)
    labels = json.loads((bundle / "meta.json").read_bytes())["per_step"]
    rewards = labels["reward"]
    if not any(labels.get("success", [])):
        raise ValueError("Cannot recover a legacy target without recorded success")
    with np.load(bundle / "states.npz") as stored:
        states, spec = stored["states"], int(stored["state_spec"])
    valid = [i for i, r in enumerate(rewards) if r is not None and i < len(states)]
    probes = np.asarray(valid)[np.unique(np.linspace(0, len(valid) - 1, min(9, len(valid)), dtype=int))]
    model = task.env.current_model
    data = task.env.mj_datas[getattr(task.env, "current_batch_index", 0)]
    robot_root = task.env.current_robot.robot_view.base.root_body_id
    free_bodies = sorted(set(int(b) for b in model.jnt_bodyid[model.jnt_type == mujoco.mjtJoint.mjJNT_FREE]
                             if model.body_rootid[b] != robot_root
                             and not model.body(int(b)).name.startswith("grasp_collision")))
    receptacles = [b for b in free_bodies if model.body(b).name.startswith("place_receptacle/")]
    pickups = [b for b in free_bodies if b not in receptacles]
    if not pickups or not receptacles:
        raise ValueError("No target candidates in the recorded color-task scene")
    errors = np.zeros((len(pickups), len(receptacles)))
    for i in probes:
        mujoco.mj_setState(model, data, np.asarray(states[i]), spec)
        mujoco.mj_forward(model, data)
        centers = np.asarray([body_aabb(model, data, b)[0] for b in receptacles])
        predicted = np.exp(-np.linalg.norm(data.xpos[pickups, None, :] - centers[None, :, :], axis=-1))
        errors = np.maximum(errors, np.abs(predicted - rewards[i]))
    candidates = np.argwhere(errors <= 1e-7)
    if len(candidates) != 1:
        raise ValueError(f"Legacy target is ambiguous: {len(candidates)} reward-matching pairs; minimum error {float(errors.min())}")
    pickup_index, receptacle_index = candidates[0]
    pickup = model.body(pickups[pickup_index]).name
    receptacle = model.body(receptacles[receptacle_index]).name
    cfg = task.config.task_config
    changed = (cfg.pickup_obj_name, cfg.place_receptacle_name) != (pickup, receptacle)
    mujoco.mj_setState(model, data, np.asarray(states[0]), spec)
    mujoco.mj_forward(model, data)
    cfg.pickup_obj_name = pickup
    cfg.place_receptacle_name = receptacle
    cfg.place_target_name = receptacle
    sampled_start_pose = cfg.place_receptacle_start_pose
    cfg.place_receptacle_start_pose = pose_mat_to_7d(create_mlspaces_body(data, receptacle).pose).tolist()
    cfg.pickup_obj_start_pose = pose_mat_to_7d(create_mlspaces_body(data, pickup).pose).tolist()
    if changed:
        # Keep color metadata consistent with the recipe-restored visual model.
        # The model's geometry/materials and the policy's WODS prompt are untouched.
        for body in receptacles:
            geoms = descendant_geoms(model, body, visible_only=True)
            if len(geoms):
                cfg.object_colors[model.body(body).name] = model.geom_rgba[geoms[0]].tolist()
    return {"method": "unique_recorded_reward_match", "target_changed": changed,
            "sampled_receptacle_start_pose": sampled_start_pose,
            "recorded_receptacle_start_pose": cfg.place_receptacle_start_pose,
            "candidate_pairs": len(pickups) * len(receptacles),
            "probe_frames": probes.tolist(), "probe_max_abs_error": float(errors[pickup_index, receptacle_index])}


def restore_recorded_task_config(config, sampler, task, meta, *, bundle=None):
    encoded = (meta.get("obs_scene") or {}).get("frozen_config")
    fields = ("pickup_obj_name", "place_receptacle_name", "joint_index")
    before = {name: getattr(task.config.task_config, name, None) for name in fields}
    recovery = None
    if encoded:
        frozen = pickle.loads(base64.b64decode(encoded))
        if type(frozen.task_config) is not type(task.config.task_config):
            raise TypeError("Recorded and reconstructed task-config classes differ")
        task_class = type(task)
        task_class_name = task_class.__module__ + "." + task_class.__name__
        recorded_class_name = getattr(frozen, "task_cls_str", None)
        if recorded_class_name and recorded_class_name != task_class_name:
            raise TypeError("Recorded and reconstructed task classes differ")
        recorded = copy.deepcopy(frozen.task_config)
        # SavedEpisode strips the callable class while preserving its import
        # path. Reset freezes the config again, so restore the verified class.
        recorded.task_cls = task_class
        task_type = getattr(frozen, "task_type", None) or meta.get("task_type") or config.task_type
        if task_type != config.task_type:
            raise TypeError("Recorded and reconstructed task types differ")
        configs = [config, task.config, sampler.config]
        if hasattr(task, "exp_config"):
            configs.append(task.exp_config)
        for current in configs:
            current.task_config = recorded
            current.task_type = task_type
    elif config.task_type == "pick_and_place_color":
        if bundle is None:
            raise ValueError("Legacy color-task target recovery requires its replay bundle")
        recovery = recover_legacy_color_targets(task, bundle)

    # The evaluated policy supplies all controls. Sensors and cached articulation
    # objects must be bound to the restored target, before reset reads state zero.
    task._registered_policy = None
    if task.config.task_config.use_sensors:
        task._sensor_suite = task._create_sensor_suite_from_config(task.config)
        task._sensor_suite.extend(task.env.current_robot.create_robot_sensors())
    else:
        task._sensor_suite = None
    if hasattr(task, "_get_articulation_objects"):
        task.articulation_objects = task._get_articulation_objects()
    task._recorded_task_audit = {
        "restored_frozen_config": bool(encoded),
        "task_type": task.config.task_type,
        "sampled_targets": before,
        "evaluated_targets": {name: getattr(task.config.task_config, name, None) for name in fields},
        "legacy_target_recovery": recovery,
    }
