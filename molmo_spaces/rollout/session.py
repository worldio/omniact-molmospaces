"""``EmbodimentRollout`` — shared base for the eval session: env wiring + obs assembly + action inversion.

Drives a datagen ``BaseMujocoTask`` through the representation pipeline (see ``docs/REPRESENTATIONS.md`` and
``embodiment_gen/representations.py``):

* each observation is ``EnvProprio.to_semantic().to_abstract()`` -> the WODS-style per-node obs
  (node_geometry / node_proprio / node_graph / node_pose, the TCP node identified by joint type 3) plus the rendered views + prompt;
* each action is the client's ``AbstractAction`` -- the per-node RELATIVE ``node_action`` and/or the absolute
  ``node_action_pose`` ((N, 3, 4), the wods label: house-world <- WODS-style TCP axes; the server reads the TCP
  row) -> ``to_semantic(current)`` -> ``to_env(spec, current, ik, action_mask, node_action_pose_mask)`` -> the
  ``{"arm","gripper"}`` controls ``task.step`` consumes. ``to_semantic`` is a pure structure-preserving inverse
  (relative -> absolute, the pose head carried through); ``to_env`` is the env commit, where the two masks (the
  execution inputs) and the pose IK apply. Both heads are always present (zeros for a head you do not command);
  the masks say what each commands: ``action_mask`` the relative ``action`` entries, ``node_action_pose_mask`` the pose row (only the
  TCP row) whose arm joints go through IK (``embodiment_gen/ik.py``). An ``action_mask`` that also covers the
  arm makes it hybrid, its arm target the solver's posture reference and the heads' disagreement in ``info``;
  masks that command nothing are an error. Thrashing / infeasible IK ends the episode (``info["ik_outcome"]``)
  unless ``continue_on_ik_failure`` (the default): the best-effort / clamped arm target is still commanded, the
  outcome still reported, and the episode runs on until the task terminates or truncates. A successful step ends
  the episode unless ``end_on_success`` is off, which is what data generation recorded: its pipeline breaks on
  success for the pick-and-place configs, and for pick its registered lift policy makes ``is_terminal`` end the
  episode on the current-step success. A replay session gets neither -- it registers no policy, and the datagen
  config it reconstructs has no ``terminate_upon_success`` -- so without this an episode runs its whole
  ``task_horizon`` past a success the demonstration never continued past.
  See the ``embodiment_gen/ik.py`` module docstring.

``reset`` (building the exact steppable task) is the subclass's job -- see ``EmbodimentReplay``. All FK /
unit / frame conversions live in ``representations``; this file is only env wiring."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from molmo_spaces.embodiment_gen.ik import IK_FAILURES
from molmo_spaces.embodiment_gen.representations import AbstractAction, EnvProprio, c2w_per_view


class EmbodimentRollout:
    """Shared base for ``EmbodimentReplay``: env wiring, obs assembly, action inversion. The subclass
    provides ``reset`` (which builds the task + ``self._spec``/``self._kin``); one env, ``n_episodes`` bundles."""

    def __init__(self, bundles, *, seed: int | None = None, cameras=None, prompt: str | None = None,
                 continue_on_ik_failure: bool = True, end_on_success: bool = True) -> None:
        self._bundles = [Path(b) for b in bundles]
        if not self._bundles:
            raise ValueError("EmbodimentRollout needs at least one episode bundle")
        self._seed = None if seed is None else int(seed)
        self._camera_filter = list(cameras) if cameras else None
        self._prompt_text = str(prompt) if prompt else ""   # observation prompt (the caller-supplied WODS caption)
        self._continue_on_ik_failure = bool(continue_on_ik_failure)   # IK thrashing / infeasible does not end the episode
        self._end_on_success = bool(end_on_success)                   # as data generation ends its episodes
        self._spec = None          # NodeSpec    -- per embodiment, built once
        self._kin = None           # Kinematics  -- per embodiment (robot geometry only), built once
        self._ik = None            # TcpIK       -- per episode (reset builds it; the clamp counter is episodic)
        self._ts = None
        self._task = None

    @property
    def n_episodes(self) -> int:
        return len(self._bundles)

    def step(self, action, node_action_pose, action_mask, node_action_pose_mask):
        """All four required (a head you do not command is zeros with an all-False mask): the relative joint
        ``action`` (N,) and the absolute ``node_action_pose`` (N, 3, 4), each with its (N,) bool mask of the dims
        that are commands. The pose row's arm joints go through IK; an ``action_mask`` that also covers the arm
        makes it hybrid. Masks that command nothing are an error. The conversion is the representation pipeline's
        C -> B -> A; the masks and the pose IK apply at the env commit (to_env)."""
        env = self._env_proprio()                         # the LIVE state, read before the step
        to = torch.as_tensor
        abstract = AbstractAction(node_action=to(action), node_action_pose=to(node_action_pose))
        env_action = abstract.to_semantic(env).to_env(self._spec, current=env, ik=self._ik,
                                                      action_mask=to(action_mask), node_action_pose_mask=to(node_action_pose_mask))
        move = env_action.move_groups
        _obs, reward, terminated, truncated, _info = self._task.step(move)
        info = dict(env_action.info)
        ik_failed = info.get("ik_outcome") in IK_FAILURES
        success = bool(self._task.judge_success())
        done = ((ik_failed and not self._continue_on_ik_failure)
                or bool(np.ravel(terminated)[0] or np.ravel(truncated)[0])
                or (self._end_on_success and success))
        info.update(success=success, reward=float(np.ravel(reward)[0]))
        return self._observation(), done, info

    def close(self) -> None:
        ts, self._ts, self._task = self._ts, None, None
        if ts is not None:            # the sampler owns the env (task.close leaves it), so close it here
            try:
                ts.close()
            except Exception:
                pass

    # --- internals ---
    def _camera_names(self):
        names = list(self._task.env.camera_manager.registry.keys())   # config order (== training views)
        if self._camera_filter is not None:
            names = [n for n in names if n in self._camera_filter]
        return names

    def _env_proprio(self) -> EnvProprio:
        from molmo_spaces.utils.pose import pose_mat_to_7d
        rv = self._task.env.robots[0].robot_view
        registry = self._task.env.camera_manager.registry
        cam2world = {name: np.asarray(registry[name].get_pose(), float) for name in self._camera_names()}
        return EnvProprio(qpos=rv.get_qpos_dict(), base_pose=pose_mat_to_7d(rv.base.pose),
                          cam2world=cam2world, spec=self._spec, kin=self._kin)

    def _observation(self) -> dict:
        env_proprio = self._env_proprio()
        abstract = env_proprio.to_semantic().to_abstract()
        views = [np.asarray(self._task.env.render_rgb_frame(name), np.uint8) for name in self._camera_names()]
        return {
            "views": torch.from_numpy(np.stack(views, axis=0)),        # (V, H, W, C) uint8
            "node_geometry": abstract.node_geometry,                   # (N, 2, V, 3) f32, WODS camera frame
            "node_proprio": abstract.node_proprio,                     # (N,) f32, centered
            "node_pose": abstract.node_pose,                           # (N, V, 3, 4) f32, achieved TCP frame, other rows 0 (== wods node_pose)
            "c2w": torch.from_numpy(c2w_per_view(env_proprio.cam2world)),  # (V, 3, 4) WODS camera->world; world TCP = c2w · node_pose
            "edges": abstract.edges,
            "axis_mask": abstract.axis_mask,
            "joint_type": abstract.joint_type,
            "node_names": abstract.node_names,
            "camera_names": self._camera_names(),                      # the V-axis order of views/node_geometry
            "prompt": self._prompt(),
        }

    def _prompt(self) -> str:
        return self._prompt_text
