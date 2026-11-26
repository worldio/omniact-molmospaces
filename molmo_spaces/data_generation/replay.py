"""Sim-replay capture for embodiment datagen: a per-episode bundle to re-simulate and re-render one
episode from the shared asset library.

    dr_deltas.npz / dr_texture_paths.json  lean DR recipe (small arrays + per-geom texture paths);
                  rebuilds the model on a fresh compile (dr_deltas.py, replay_episode.py); tex_data is
                  reloaded from the paths.
    states.npz    full per-frame integration state (mjSTATE_INTEGRATION); setState + mj_forward
                  reproduces the kinematics/render bit-for-bit.
    cameras.json  per-frame camera render inputs (pos/forward/up/fov) + derived intrinsics.
    meta.json     instruction, per-step outcome, provenance, and meta['recon'] (fresh-compile inputs
                  + a dr_fingerprint the reconstruction self-verifies).

Capture is side-effect-free on the rollout (reads only; the texture-path log still calls the original),
gated by ``MlSpacesExpConfig.capture_replay``.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import mujoco

from molmo_spaces.data_generation.dr_deltas import capture_dr_deltas, dr_fingerprint
from molmo_spaces.utils.pose import GL_FROM_CV

# mjSTATE_INTEGRATION: the full physics state (qpos/qvel/act, ctrl, mocap, eq_active, warmstart,
# applied forces, time) -- setState + mj_forward reproduces the kinematics/render bit-for-bit.
_STATE_SPEC = int(mujoco.mjtState.mjSTATE_INTEGRATION)


def render_rgb(
    renderer,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pos: np.ndarray,
    forward: np.ndarray,
    up: np.ndarray,
    fov: float,
) -> np.ndarray:
    """Render one RGB frame from an explicit free-camera pose.

    A faithful copy of ``CPUMujocoEnv._render_frame`` (RGB path): set the global fovy for this camera,
    build the scene with a FREE camera, then overwrite the scene camera's pos/forward/up. Kept in sync
    with env.py by construction so a replay renders the same pixels the datagen did.
    """
    prev_fov = model.vis.global_.fovy
    try:
        model.vis.global_.fovy = fov
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        renderer.update(data, cam)
        for scene_cam in renderer.scene.camera:
            scene_cam.pos = pos
            scene_cam.forward = forward
            scene_cam.up = up
        return renderer.render()
    finally:
        model.vis.global_.fovy = prev_fov


def camera_params(pos, forward, up, fov: float, resolution) -> dict:
    """Derive the cam.json-style intrinsics/extrinsics from render inputs.

    Replicates ``Camera.get_pose`` + ``CameraParameterSensor.get_observation`` so a consumer of the
    bundle alone gets the same extrinsic_cv/cam2world_gl/intrinsic_cv the datagen wrote next to the h5.
    """
    forward = np.asarray(forward, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    fn = np.linalg.norm(forward)
    un = np.linalg.norm(up)
    if fn < 1e-6 or un < 1e-6:
        return {}
    forward = forward / fn
    up = up / un
    right = np.cross(forward, up)
    rn = np.linalg.norm(right)
    if rn < 1e-6:
        return {}
    right = right / rn
    up = np.cross(right, forward)  # re-orthogonalize (matches get_pose)

    pose = np.eye(4)  # cam2world, OpenCV convention: columns (right, -up, forward)
    pose[:3, 0] = right
    pose[:3, 1] = -up
    pose[:3, 2] = forward
    pose[:3, 3] = np.asarray(pos, dtype=np.float64)

    extrinsic_cv = np.linalg.inv(pose)[:3, :]
    cam2world_gl = pose @ GL_FROM_CV
    width, height = int(resolution[0]), int(resolution[1])
    focal = (height / 2.0) / np.tan(np.radians(float(fov) / 2.0))
    intrinsic_cv = [[focal, 0, width / 2.0], [0, focal, height / 2.0], [0, 0, 1]]
    return {
        "cam2world_gl": cam2world_gl.tolist(),
        "extrinsic_cv": extrinsic_cv.tolist(),
        "intrinsic_cv": intrinsic_cv,
    }


def _jsonable(obj: Any) -> Any:
    """Best-effort convert numpy/paths/pydantic to JSON-serializable, never raising."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    for attr in ("model_dump", "dict"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return _jsonable(fn())
            except Exception:
                break
    return str(obj)


def _strip_frozen_config_robot_dir(obs_scene: Any) -> Any:
    """Null robot_dir in the pickled episode config (obs_scene['frozen_config']): it is an absolute
    path into the datagen's temp workspace, replay never reads it, and it is the one machine path in
    meta. Nulling it also makes meta reproducible (it was the only field that varied run-to-run)."""
    if not isinstance(obs_scene, dict) or not obs_scene.get("frozen_config"):
        return obs_scene
    try:
        import base64
        import pickle

        ep = pickle.loads(base64.b64decode(obs_scene["frozen_config"]))
        rc = getattr(ep, "robot_config", None)
        if rc is not None:
            try:
                rc.robot_dir = None
            except Exception:
                rc.__dict__["robot_dir"] = None
        return {**obs_scene, "frozen_config": base64.b64encode(pickle.dumps(ep)).decode("utf-8")}
    except Exception:
        return obs_scene


class ReplayRecorder:
    """Accumulates a replay bundle (recipe + per-frame state + camera poses) over one rollout, then
    writes it next to the trajectory h5 if the episode is kept."""

    def __init__(self, task, episode_seed: int, task_sampler=None):
        env = task.env
        self._model = env.mj_model
        self._data = env.current_data
        self._state_size = mujoco.mj_stateSize(self._model, _STATE_SPEC)

        # Capture the DR recipe from the FINAL rollout model (arrays captured mid-randomize can be
        # stale) + the texture paths the sampler logged during randomize_scene.
        paths = task_sampler.dr_texture_paths if task_sampler is not None else {}
        self._dr_deltas = capture_dr_deltas(self._model, paths)
        self._dr_fingerprint = dr_fingerprint(self._model)

        renderer = getattr(env, "_renderer", None)
        self._resolution = (
            (int(renderer.width), int(renderer.height)) if renderer is not None else (0, 0)
        )

        # Camera registry ref + static per-camera fov; poses are captured per frame.
        self._registry = env.camera_manager.registry
        self._cam_names = [cam.name for cam in self._registry]
        self._cam_fov = {cam.name: float(cam.fov) for cam in self._registry}
        self._cam_pos = {n: [] for n in self._cam_names}
        self._cam_forward = {n: [] for n in self._cam_names}
        self._cam_up = {n: [] for n in self._cam_names}

        self._states: list[np.ndarray] = []
        self._rewards: list[Any] = []
        self._successes: list[Any] = []
        self._terminals: list[Any] = []

        self._episode_seed = int(episode_seed)
        self._task = task
        # Eager snapshot: the shared exp_config.task_config is overwritten in place by the next episode's
        # sample_task, so a lazy read at write() (which runs after the whole batch) would serialize the
        # last episode's text into every bundle. Capture now, while config holds this episode's values.
        self._instruction_snapshot = self._instruction()

        # Frame 0: the reset state, matching the frame-0 observation the task records.
        self.capture()

    def capture(self, reward=None, terminal=None, success=None) -> None:
        """Snapshot the current frame: integration state + each camera's render pose + outcome."""
        arr = np.zeros(self._state_size, dtype=np.float64)
        mujoco.mj_getState(self._model, self._data, arr, _STATE_SPEC)
        self._states.append(arr)
        for cam in self._registry:
            self._cam_pos[cam.name].append(np.asarray(cam.pos, dtype=np.float64).copy())
            self._cam_forward[cam.name].append(np.asarray(cam.forward, dtype=np.float64).copy())
            self._cam_up[cam.name].append(np.asarray(cam.up, dtype=np.float64).copy())
        self._rewards.append(None if reward is None else float(np.asarray(reward).reshape(-1)[0]))
        self._terminals.append(None if terminal is None else bool(np.asarray(terminal).reshape(-1)[0]))
        self._successes.append(None if success is None else bool(success))

    @property
    def n_frames(self) -> int:
        return len(self._states)

    @property
    def episode_seed(self) -> int:
        return self._episode_seed

    def _instruction(self) -> dict:
        task = self._task
        out: dict[str, Any] = {}
        for key, getter in (
            ("instruction", "get_task_description"),
            ("referral_expressions", "get_referral_expressions"),
            ("obs_scene", "get_obs_scene"),
        ):
            fn = getattr(task, getter, None)
            if callable(fn):
                try:
                    value = fn()
                    if key == "obs_scene":
                        value = _strip_frozen_config_robot_dir(value)
                    out[key] = _jsonable(value)
                except Exception:
                    pass
        cfg = getattr(task, "config", None)
        tcfg = getattr(cfg, "task_config", None)
        if tcfg is not None and "referral_expressions" not in out:
            out["referral_expressions"] = _jsonable(getattr(tcfg, "referral_expressions", None))
        return out

    def write(self, out_dir: Path, *, exp_config=None, house_id=None, episode_idx=None) -> Path:
        """Serialize the bundle into ``out_dir`` (created)."""
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # The DR recipe rebuilds the model on a fresh compile.
        np.savez_compressed(out_dir / "dr_deltas.npz", **self._dr_deltas["arrays"])
        (out_dir / "dr_texture_paths.json").write_text(
            json.dumps(self._dr_deltas["geom_texture_paths"])
        )

        states = np.asarray(self._states, dtype=np.float64)
        np.savez_compressed(
            out_dir / "states.npz",
            states=states,
            state_spec=np.int64(_STATE_SPEC),
            state_size=np.int64(self._state_size),
            nq=np.int64(self._model.nq),
            nv=np.int64(self._model.nv),
            na=np.int64(self._model.na),
        )

        width, height = self._resolution
        cameras = {"img_resolution": [width, height], "cameras": []}
        for name in self._cam_names:
            pos = np.asarray(self._cam_pos[name])
            fwd = np.asarray(self._cam_forward[name])
            up = np.asarray(self._cam_up[name])
            fov = self._cam_fov[name]
            params = [camera_params(pos[t], fwd[t], up[t], fov, self._resolution)
                      for t in range(len(pos))]
            cameras["cameras"].append({
                "name": name,
                "fov": fov,
                "resolution": [width, height],
                "pos": pos.tolist(),
                "forward": fwd.tolist(),
                "up": up.tolist(),
                "params": params,
            })
        (out_dir / "cameras.json").write_text(json.dumps(cameras))

        meta = {
            "episode_seed": self._episode_seed,
            "episode_idx": None if episode_idx is None else int(episode_idx),
            "house_id": _jsonable(house_id),
            "n_frames": self.n_frames,
            "state_spec": _STATE_SPEC,
            "state_size": self._state_size,
            "dr_fingerprint": self._dr_fingerprint,
            "model": {"nq": int(self._model.nq), "nv": int(self._model.nv),
                      "na": int(self._model.na), "ngeom": int(self._model.ngeom),
                      "ncam": int(self._model.ncam)},
            "cameras": self._cam_names,
            "resolution": [width, height],
            "per_step": {"reward": self._rewards, "success": self._successes,
                         "terminal": self._terminals},
        }
        meta.update(self._instruction_snapshot)
        if exp_config is not None:
            meta["scene_dataset"] = exp_config.scene_dataset
            meta["data_split"] = exp_config.data_split
            meta["task_type"] = exp_config.task_type
            try:
                meta["fps"] = float(exp_config.fps)
            except Exception:
                pass
            # Inputs to fresh-compile the base scene + a fingerprint to self-verify the recon.
            recon = dict(exp_config.replay_recon_info or {})
            recon.update(
                scene_dataset=meta["scene_dataset"],
                data_split=meta["data_split"],
                house_id=_jsonable(house_id),
                dr_fingerprint=self._dr_fingerprint,
            )
            # Copy the emitted embodiment into the bundle; its model.xml already carries portable
            # ${ROOT} tokens (see spec_emit). Guard that no absolute asset path slipped through.
            spec_src = recon.get("embodiment_spec")
            if spec_src and Path(spec_src).exists():
                shutil.copytree(Path(spec_src).parent, out_dir / "embodiment", dirs_exist_ok=True)
                recon["embodiment_spec"] = f"embodiment/{Path(spec_src).name}"
                model_xml = out_dir / "embodiment" / "model.xml"
                if model_xml.exists():
                    stray = re.search(r'file="(/[^"]*)"', model_xml.read_text())
                    if stray:
                        raise AssertionError(f"model.xml has an absolute asset path: {stray[1]}")
            # Embed the asset-blacklist contents (not its path) so replay is deterministic + portable.
            blacklist = recon.pop("asset_blacklist_path", None)
            if blacklist and Path(blacklist).exists():
                shutil.copy(Path(blacklist), out_dir / "asset_blacklist.txt")
            meta["recon"] = recon
        (out_dir / "meta.json").write_text(json.dumps(_jsonable(meta)))
        return out_dir
