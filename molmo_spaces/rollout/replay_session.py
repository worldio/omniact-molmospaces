"""Exact-replay eval session.

``reset(idx)`` reconstructs bundle ``idx``'s EXACT datagen scene + RNG (``episode.reconstruct_task_exact`` --
replay_episode's frozen-config pin + ``seed_task_sampling``) and sets the recorded initial state
(``states[0]``); ``step(action)`` runs real ``task.step`` physics (inherited from ``EmbodimentRollout``). On
the datagen CPU class (AMD) + osmesa, ground-truth actions reproduce the recorded trajectory + frames
bit-exactly; a policy's actions make it closed-loop eval on the exact recorded scene."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from molmo_spaces.embodiment_gen.representations import Kinematics, NodeSpec

from .episode import reconstruct_task_exact
from .session import EmbodimentRollout


def _pin_cameras(env, bundle) -> int:
    """Reproduce datagen's camera poses from the recorded ``cameras.json`` frame-0.

    datagen domain-randomizes cameras: world/exocentric cams get a randomized absolute pose (setup_cameras),
    and robot-mounted (wrist) cams get a randomized MOUNT (``camera_offset``/``camera_quaternion`` noise in
    ``_setup_robot_mounted_camera``). The eval reconstruction re-draws different noise (RNG diverges), so
    neither matches capture. ``replay_episode`` sidesteps this by rendering from ``cameras.json`` directly;
    here we plug the recorded values back into the camera objects:
      * world cams (static) -> pin the absolute pos/forward/up (holds through stepping);
      * wrist cams (dynamic) -> recover the constant body-frame MOUNT from the recorded frame-0 pose and the
        reference body's pose at ``states[0]`` (mount = inv(body_pose) @ world_cam), so ``update_pose`` tracks
        the datagen mount for EVERY frame (reset, GT-stepping, and real policy eval).
    Must run after ``mj_setState(states[0])`` + forward, so the wrist reference body is at capture. Returns
    the number pinned."""
    import json

    from scipy.spatial.transform import Rotation as R

    from molmo_spaces.env.camera_manager import RobotMountedCamera

    cams_path = Path(bundle) / "cameras.json"
    if not cams_path.exists():
        return 0
    rec = {c["name"]: c for c in json.loads(cams_path.read_text())["cameras"]}
    registry = env.camera_manager.registry
    n = 0
    for name in list(registry.keys()):
        cam = registry[name]
        r = rec.get(name)
        if r is None:
            continue
        if isinstance(cam, RobotMountedCamera):
            body, _bn = cam._find_reference_body(env)   # reference body at states[0]
            if body is None:
                continue
            Rb = np.asarray(body.pose[:3, :3], np.float64); tb = np.asarray(body.pose[:3, 3], np.float64)
            fwd = np.asarray(r["forward"][0], np.float64); fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            right = np.cross(fwd, np.asarray(r["up"][0], np.float64)); right = right / (np.linalg.norm(right) + 1e-12)
            up = np.cross(right, fwd)                    # orthonormalize (matches Camera.get_pose)
            world_R = np.column_stack([right, up, -fwd]) # world_rotation: col0=right, col1=up, col2=-forward
            world_t = np.asarray(r["pos"][0], np.float64)
            # update_pose does world_cam = body_pose @ mount; invert for the DR'd mount and plug it in.
            cam.camera_offset = (Rb.T @ (world_t - tb)).astype(np.float32)
            cam.camera_quaternion = R.from_matrix(Rb.T @ world_R).as_quat(scalar_first=True).astype(np.float32)
            cam.fov = float(r["fov"])
            cam._last_reference_pose = None              # force recompute from the recovered mount
            cam.update_pose(env)
        else:
            cam.pos = np.asarray(r["pos"][0], np.float32)
            cam.forward = np.asarray(r["forward"][0], np.float32)
            cam.up = np.asarray(r["up"][0], np.float32)
            cam.fov = float(r["fov"])
        n += 1
    return n


class EmbodimentReplay(EmbodimentRollout):
    """One env, one replay bundle. ``reset`` puts the model, initial state, and RNG at the datagen
    first-step state; ``step`` steps physics. Owns the temp dir the client-sent bundle was written to."""

    def __init__(self, bundles, *, seed=None, cameras=None, prompt=None, task_horizon=None,
                 continue_on_ik_failure=True,
                 end_on_success=True, _tempdir=None) -> None:
        super().__init__(bundles, seed=seed, cameras=cameras, prompt=prompt,
                         continue_on_ik_failure=continue_on_ik_failure, end_on_success=end_on_success)
        self._task_horizon = task_horizon   # None keeps the task config's own cap
        self._tempdir = _tempdir   # TemporaryDirectory holding the sent bundle; cleaned on close()
        self._cached_idx = None    # episode currently reconstructed; reused across repeat resets of it

    def reset(self, episode_idx: int) -> dict:
        bundle = self._bundles[episode_idx]
        if self._cached_idx != episode_idx or self._task is None:
            # First reset of this episode: reconstruct the exact scene (the slow part). Cache it so a repeat
            # reset of the SAME episode reuses the reconstructed model + scene and only re-applies the frame-0
            # state + cameras below -- byte-identical obs (same model, same states[0], same pins), in seconds.
            super().close()        # close the previous episode's task sampler; KEEP the bundle temp dir
            _cfg, ts, task, _meta, entry = reconstruct_task_exact(
                bundle, seed=self._seed, task_horizon=self._task_horizon)
            self._ts, self._task, self._cached_idx = ts, task, episode_idx
            if self._kin is None:  # robot geometry -- identical across houses/episodes of this embodiment
                node_spec_path = Path(entry) / "node_spec.json"
                if not node_spec_path.exists():
                    raise FileNotFoundError(
                        f"{node_spec_path} missing; the bundle must embed the embodiment's node_spec.json")
                self._spec = NodeSpec.from_json(node_spec_path)
                self._kin = Kinematics(Path(entry), self._spec.raw)
        task = self._task
        # Set the exact recorded frame-0 state BEFORE task.reset(), so its cached first obs / sensor
        # baselines read it. task.reset() is task/sensor/policy bookkeeping ONLY -- BaseMujocoTask does not
        # reset the sim (env.reset is commented out, task.py:228) and draws no RNG; the scene, robot
        # placement, initial state, and rollout-start RNG all come from reconstruct_task_exact's sample_task.
        # datagen's states[0] is exactly this post-reset frame-0 state (pipeline.py:776).
        with np.load(bundle / "states.npz") as sd:
            states, state_spec = sd["states"], int(sd["state_spec"])
        model = task.env.current_model
        data = task.env.mj_datas[getattr(task.env, "current_batch_index", 0)]
        mujoco.mj_setState(model, data, np.asarray(states[0]), state_spec)
        mujoco.mj_forward(model, data)
        task.reset()               # bookkeeping + first-obs cache, read from the states[0] we just set
        from molmo_spaces.embodiment_gen.ik import TcpIK
        self._ik = TcpIK(self._kin)                                          # per episode (clamp counter)
        # Cameras: refresh from the states[0] robot, then reproduce datagen's DR'd poses from cameras.json --
        # world cams pinned absolute, wrist cams' randomized MOUNT recovered (see _pin_cameras).
        task.env.camera_manager.registry.update_all_cameras(task.env)
        _pin_cameras(task.env, bundle)
        return self._observation()

    def close(self) -> None:
        super().close()            # task sampler
        if self._tempdir is not None:
            self._tempdir.cleanup()
            self._tempdir = None


def replay_session_from_bundle(bundle: dict, options: dict):
    """``eval_server`` session_factory. ``bundle``: ``{relative_path: bytes}`` -- the whole replay bundle,
    sent by the client and written to a private temp dir the session owns, so the server needs no shared
    filesystem and no path (only its MLSPACES asset corpus + blacklist). The client folds the WODS metadata into
    the bundle as ``wods_meta.json``; we read its authoritative per-episode ``caption`` for the observation
    prompt. ``options`` (strings): ``seed`` -- override the reconstruction work-item seed (default: the bundle's
    ``episode_seed - episode_idx``); ``cameras`` -- ``;``-separated camera subset; ``continue_on_ik_failure`` --
    ``1`` (default) keeps the episode running through IK thrashing / infeasibility, ``0`` ends it there;
    ``task_horizon`` -- override the task config's own step cap (base pick config: 500), for measuring
    whether a task is horizon-limited rather than capability-limited;
    ``end_on_success`` -- ``1`` (default) ends the episode on its first successful step, as data generation did,
    ``0`` runs on to the task horizon so only the final step counts."""
    import json
    import tempfile

    if not bundle:
        raise ValueError("Open request must include the bundle files (relative_path -> content)")
    tmp = tempfile.TemporaryDirectory(prefix="eval_bundle_")
    root = Path(tmp.name)
    for rel, content in bundle.items():
        dst = root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(content)
    seed = int(options["seed"]) if options.get("seed") else None
    cameras = options["cameras"].split(";") if options.get("cameras") else None
    continue_on_ik_failure = _flag(options, "continue_on_ik_failure", default=True)
    end_on_success = _flag(options, "end_on_success", default=True)
    task_horizon = int(options["task_horizon"]) if options.get("task_horizon") else None
    # The authoritative per-episode caption, a required bundle member (folded in by the client's resolve_bundle);
    # a missing/blank one is a bad sample -> raise.
    caption = json.loads((root / "wods_meta.json").read_bytes()).get("caption")
    if not caption or not caption.strip():
        raise ValueError("bundle's wods_meta.json has no non-blank 'caption'; the eval requires the "
                         "authoritative per-episode WODS caption")
    return EmbodimentReplay([str(root)], seed=seed, cameras=cameras, prompt=caption,
                            task_horizon=task_horizon,
                            continue_on_ik_failure=continue_on_ik_failure, end_on_success=end_on_success,
                            _tempdir=tmp)


def _flag(options: dict, name: str, *, default: bool) -> bool:
    """A boolean Open option (strings on the wire): unset / blank -> ``default``; anything else must be a clear yes/no."""
    value = str(options.get(name, "")).strip().lower()
    if not value:
        return default
    if value in ("1", "true", "yes", "on"):
        return True
    if value in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"option {name}={options[name]!r} is not a boolean (use 1/0, true/false, yes/no, on/off)")
