"""Reconstruct a steppable eval episode from a datagen replay bundle.

A bundle's ``meta['recon']`` holds the fresh-compile inputs (config_selector, scene_dataset, data_split,
house, embodiment_spec, control_profile). ``sample_task`` gives a steppable ``BaseMujocoTask`` (so the eval
loop is the datagen rollout with the remote policy in place of the scripted expert), and its live model's DR
is then overwritten with the datagen-EXACT model ``molmo_spaces.data_generation.recon.build_model_from_recipe``
rebuilds from the recorded recipe -- so physics + frames match capture on any CPU."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


class BlacklistedBodiesInScene(RuntimeError):
    """The scene this episode was recorded in contains bodies on the static asset blacklist.

    Those bodies are deleted before every compile (they hit MuJoCo's mass/inertia limit), so a scene rebuilt
    today is missing objects the recording contains -- the replay is not the recorded episode, and its success
    or failure says nothing about the policy. Raised instead of evaluating, so the caller skips the trajectory
    and reports it as excluded rather than scoring it.

    Deliberately not a RetriableError/HouseInvalidForTask: the K-search must not treat this as "try the next
    seed", because every seed rebuilds the same deficient scene."""


def load_meta(bundle_dir) -> dict:
    return json.loads((Path(bundle_dir) / "meta.json").read_text())


def _close_quietly(ts) -> None:
    """Free a discarded K-search task sampler's env; best-effort (close() has a known __del__ quirk)."""
    try:
        ts.close()
    except Exception:
        pass


def _rebuild_env_renderer(env) -> None:
    """Rebuild the env's renderer so its MjrContext reflects the current (recipe-corrected) model textures.
    The renderer uploads tex_data at construction; after we overwrite the model's DR the baked context is
    stale, and a fresh ``MjOpenGLRenderer`` re-uploads -- the same fresh-renderer path recon renders through."""
    from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

    r = getattr(env, "_renderer", None)
    if r is None:
        return
    w, h = int(r.width), int(r.height)
    r.close()
    env._renderer = MjOpenGLRenderer(model=env.mj_model, width=w, height=h)


_MAX_EXCLUSION_CANDIDATES = 8   # cap the recon-time asset-exclusion search (each candidate is fingerprint-gated)


def _exclusion_search(ksearch, episode_idx, cands, model, meta, task_sampler_module):
    """Recover a bundle whose datagen run DYNAMICALLY blacklisted an asset (assets that hit MuJoCo's "mass and
    inertia of moving bodies must be larger than mjMINVAL" compile error are run-blacklisted) but never persisted
    that blacklist -- so the recon rebuilds extra bodies the recording lacks and no seed fingerprint-matches.

    `model` is the recon's full-set build (more bodies than the recording). Compute the (nq, ngeom) deficit vs
    meta['model'], enumerate candidate asset-UID subsets whose removal closes it EXACTLY, and re-run the seed
    K-search with each blacklisted -- the fingerprint (verified inside `ksearch`) picks the truly-excluded asset
    among equal-dimension candidates. Degenerate-inertia UIDs (the mjMINVAL culprits) are tried first. The winning
    K is blacklist-independent, so probe every candidate at K=episode_idx (the common no-rejection case) before
    any full-K fallback -- bounds the usual recovery to one compile per candidate.

    Cache-clear: the blacklist is read once and cached in task_sampler._STATIC_ASSET_BLACKLIST, so re-pointing the
    env var is ignored; we set the global directly per candidate and restore it in `finally`. Returns (ts, task)
    or (None, None)."""
    import itertools
    import re

    import mujoco as mj
    import numpy as np

    rec = meta.get("model") or {}
    rec_nq, rec_geom = rec.get("nq"), rec.get("ngeom")
    if rec_nq is None or rec_geom is None:
        return None, None
    d_nq, d_geom = int(model.nq) - int(rec_nq), int(model.ngeom) - int(rec_geom)
    if d_nq <= 0 or d_geom <= 0:    # an exclusion can only help when the recon has MORE than the recording
        return None, None

    # Per-asset-UID tally (skip grasp-planner probe bodies). Body names are <category>_<32hexUID>_<i>_<j>_<k>.
    uid_re = re.compile(r"_([0-9a-f]{32})_")
    geom_by_body = [0] * model.nbody
    for g in range(model.ngeom):
        geom_by_body[model.geom_bodyid[g]] += 1
    jq = {int(mj.mjtJoint.mjJNT_FREE): 7, int(mj.mjtJoint.mjJNT_BALL): 4,
          int(mj.mjtJoint.mjJNT_SLIDE): 1, int(mj.mjtJoint.mjJNT_HINGE): 1}
    nq_by_body = [0] * model.nbody
    for j in range(model.njnt):
        nq_by_body[model.jnt_bodyid[j]] += jq.get(int(model.jnt_type[j]), 0)
    per_uid: dict[str, dict] = {}
    for b in range(model.nbody):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, b) or ""
        if name.startswith("grasp_collision"):
            continue
        m = uid_re.search(name)
        if m is None:
            continue
        u = per_uid.setdefault(m.group(1), {"nq": 0, "geom": 0, "min_inertia": float("inf")})
        u["nq"] += nq_by_body[b]
        u["geom"] += geom_by_body[b]
        u["min_inertia"] = min(u["min_inertia"], float(np.min(model.body_inertia[b])))

    uids = sorted(per_uid, key=lambda u: per_uid[u]["min_inertia"])   # degenerate-inertia (mjMINVAL) first
    subsets = [(u,) for u in uids] + list(itertools.combinations(uids, 2))
    candidates = [s for s in subsets
                  if sum(per_uid[u]["nq"] for u in s) == d_nq and sum(per_uid[u]["geom"] for u in s) == d_geom]
    if not candidates:
        return None, None

    candidates = candidates[:_MAX_EXCLUSION_CANDIDATES]
    rest = [k for k in cands if k != episode_idx]
    passes = [[episode_idx]] + ([rest] if rest else [])   # fast winning-K probe, then the rare rejected-attempt K's
    task_sampler_module.get_static_asset_blacklist()       # ensure the committed blacklist is loaded/cached
    base = set(task_sampler_module._STATIC_ASSET_BLACKLIST or ())
    prev = task_sampler_module._STATIC_ASSET_BLACKLIST
    try:
        for ks in passes:
            for subset in candidates:
                task_sampler_module._STATIC_ASSET_BLACKLIST = base | set(subset)   # in-process cache-clear
                ts_k, task_k = ksearch(ks)
                if task_k is not None:
                    print(f"eval recon exclusion-search: recovered by blacklisting {list(subset)} "
                          f"(unpersisted datagen dynamic-blacklist; deficit nq={d_nq} geom={d_geom})", file=sys.stderr)
                    return ts_k, task_k
    finally:
        task_sampler_module._STATIC_ASSET_BLACKLIST = prev    # restore the module-cached blacklist
    return None, None


def reconstruct_task_exact(bundle_dir, *, seed: int | None = None,
                           task_horizon: int | None = None):
    """Reconstruct the episode's EXACT steppable scene, robust to this node's CPU:

    * ``sample_task`` builds the steppable ``BaseMujocoTask`` (env + robot controller + cameras), with the
      spawned objects PINNED to the bundle's ``frozen_config`` (seeding alone can't reproduce a non-first
      episode of a multi-episode work item; positions come from the caller's ``states[0]``).
    * Its live model's DR is then OVERWRITTEN with the datagen-exact model ``build_model_from_recipe`` rebuilds
      from the recorded recipe. That path writes capture's recorded dynamics/material BYTES (+ a seeded uint8
      texture RNG + render-role fixes), which is CPU-independent; re-seeding the randomizers here would instead
      RECOMPUTE the float dynamics, which diverges across the mixed-CPU fleet (fingerprint mismatch). The two
      models share one structure (same scene + same pinned objects), so the DR arrays + tex_data copy across.

    The renderer is rebuilt so frames render from the corrected model. Returns
    ``(exp_config, task_sampler, task, meta, entry)``."""
    import contextlib

    # recon's pinning + the proven recipe application, shared with the replay_episode.py CLI
    from molmo_spaces.data_generation.recon import (
        apply_recipe_to_compiled_model, build_model_from_recipe, datagen_max_attempts, frozen_spawned_uids,
        pin_spawned_choices,
    )
    from molmo_spaces.data_generation.dr_deltas import restore_dr_base, snapshot_dr_base
    from molmo_spaces.data_generation.main import auto_import_configs, create_experiment_config

    bundle = Path(bundle_dir)
    meta = load_meta(bundle)
    recon = meta.get("recon")
    if not recon or not recon.get("config_selector"):
        raise ValueError(f"bundle {bundle} has no recon recipe (meta['recon'])")

    spec_path = recon.get("embodiment_spec")
    if spec_path is not None and not Path(spec_path).is_absolute():
        spec_path = str(bundle / spec_path)
    blacklist = bundle / "asset_blacklist.txt"
    if blacklist.exists():
        os.environ.setdefault("MLSPACES_ASSET_BLACKLIST_PATH", str(blacklist))

    auto_import_configs()
    exp_config, _ = create_experiment_config(
        recon["config_selector"], spec_path, recon.get("control_profile")
    )
    if task_horizon is not None:
        # The task config, not the client, decides when an episode is truncated: the base pick config caps
        # it at 500 steps, so a client asking for more is silently ignored. Override it here to measure
        # whether a task is horizon-limited (pick-and-place successes crowd the 500-step wall).
        exp_config.task_horizon = int(task_horizon)
    exp_config.scene_dataset = recon["scene_dataset"]
    exp_config.data_split = recon["data_split"]
    house = int(str(recon["house_id"]).rsplit("_", 1)[-1])
    exp_config.task_sampler_config.house_inds = [house]

    # Eval server is a deterministic replay: no action noise (its RNG isn't reproduced). Eval-path only.
    if exp_config.robot_config.action_noise_config is not None:
        exp_config.robot_config.action_noise_config.enabled = False
    # capture_replay makes sample_task snapshot the clean post-compile DR base (ts.dr_base_snapshot), so we
    # apply the recorded recipe onto sample_task's OWN compiled model below -- no second scene compile.
    exp_config.capture_replay = True
    # Skip sample_task's robot-placement IK search (all task samplers go through env.place_robot_near):
    # states[0] restores the recorded base pose, so it's wasted work and can fail cross-CPU.
    exp_config.skip_robot_placement = True

    # Pin the world (exocentric) cameras to their RECORDED frame-0 pose by converting each randomized
    # exocentric config to a fixed one, so setup_cameras takes the deterministic fixed path instead of the
    # cross-CPU-sensitive, visibility-constrained placement search -- which can FAIL on some scenes and
    # whose pose is overwritten by cameras.json at replay anyway (replay_session._pin_world_cameras).
    if exp_config.camera_config is not None:
        from molmo_spaces.configs.camera_configs import (
            FixedExocentricCameraConfig, RandomizedExocentricCameraConfig,
        )
        rec_cams = {c["name"]: c for c in json.loads((bundle / "cameras.json").read_text()).get("cameras", [])
                    if c.get("pos") and c.get("forward") and c.get("up")}
        exp_config.camera_config.cameras = [
            FixedExocentricCameraConfig(name=cam.name, fov=float(rec_cams[cam.name]["fov"]),
                                        pos=rec_cams[cam.name]["pos"][0], forward=rec_cams[cam.name]["forward"][0],
                                        up=rec_cams[cam.name]["up"][0])
            if isinstance(cam, RandomizedExocentricCameraConfig) and cam.name in rec_cams else cam
            for cam in exp_config.camera_config.cameras
        ]

    episode_seed = int(meta.get("episode_seed", 0))
    episode_idx = int(meta.get("episode_idx", 0))
    frozen = frozen_spawned_uids(meta)
    # Old bundles store only episode_seed, not the scene current_seed (= episode_seed - episode_idx -
    # rejected_attempt_count). Re-sampling at a fixed seed draws the WRONG scene when datagen rejected attempts
    # -> geom-count broadcast or a fingerprint miss. K-search the offset like build_model_from_recipe: K=
    # episode_idx first (no-rejection == the common case, one sample_task), then 0,1,2,..; accept the first seed
    # whose apply_recipe_to_compiled_model returns -- it fingerprint-VERIFIES internally (raises on mismatch, so
    # a wrong seed is caught below) then applies the datagen-EXACT recipe onto sample_task's OWN compiled model
    # (capture_replay -> single compile). NB: don't re-check the fingerprint here -- apply's tail (texture-role
    # fixes) mutates the model on purpose, so a post-hoc dr_fingerprint no longer equals want. Assumes
    # REPLAY_FINGERPRINT_WARN is unset (default), so a mismatch raises rather than passing silently.
    from molmo_spaces.tasks.task_sampler_errors import HouseInvalidForTask, RetriableError

    import molmo_spaces.tasks.task_sampler as _task_sampler
    # Search the full datagen attempt window (== recon.py's texture K-search), not a fixed +8: episode_idx
    # counts only WINNING episodes, but rejected attempts push the winning K up to max_attempts (seen K=9).
    max_attempts = datagen_max_attempts(exp_config)
    cands = ([episode_idx] + [k for k in range(max_attempts) if k != episode_idx]) if seed is None else [None]
    last_model: dict = {}   # newest successfully-sampled model, kept for the exclusion search's deficit/tally

    def _ksearch(ks=None):
        """One seed K-search pass at the CURRENT asset blacklist; returns (ts, task) or (None, None). Accepts the
        first seed whose apply_recipe_to_compiled_model returns -- it fingerprint-VERIFIES internally (raises on
        mismatch) then applies the datagen-EXACT recipe onto sample_task's OWN compiled model. `ks` overrides the
        K-list (the exclusion search probes a single K, since the winning K is blacklist-independent)."""
        for K in (cands if ks is None else ks):
            current_seed = int(seed) if seed is not None else (episode_seed - K)
            ts_k = exp_config.task_sampler_config.task_sampler_class(exp_config)
            ts_k.seed_task_sampling(current_seed)    # np.random/random/torch -> the work-item stream start
            try:
                with (pin_spawned_choices(frozen) if frozen else contextlib.nullcontext()):
                    task_k = ts_k.sample_task(house_index=house)
            # Mirror datagen's per-attempt handling: an unsamplable scene at this seed (bad house, un-placeable
            # robot/object/camera, or no valid asset draw) is a rejected attempt -> try the next K, not a hard fail.
            except (HouseInvalidForTask, RetriableError, ValueError):
                _close_quietly(ts_k); continue
            if task_k is None:
                _close_quietly(ts_k); continue
            live = task_k.env.current_model
            last_model["m"] = live   # remember for the exclusion search if every seed misses
            base = getattr(ts_k, "dr_base_snapshot", None)
            try:
                if base is not None:
                    apply_recipe_to_compiled_model(live, base, bundle, meta, exp_config, ts_k._current_house_scene_path())
                else:
                    restore_dr_base(live, snapshot_dr_base(build_model_from_recipe(bundle, meta)))
            except (ValueError, RuntimeError):   # geom-count broadcast or fingerprint mismatch -> wrong seed
                _close_quietly(ts_k); continue
            if seed is None and K != episode_idx:
                print(f"eval recon K-search: matched at K={K} (seed {current_seed})", file=sys.stderr)
            return ts_k, task_k
        return None, None

    ts, task = _ksearch()
    # No seed matched on the full asset set: the bundle may have been generated under datagen's dynamic asset
    # blacklist (unpersisted). Discover the missing exclusion from the model-dimension deficit + fingerprint.
    if task is None and seed is None and last_model.get("m") is not None:
        ts, task = _exclusion_search(_ksearch, episode_idx, cands, last_model["m"], meta, _task_sampler)
    if task is None:
        raise RuntimeError(f"reconstruction failed: no episode_seed-K (K in {cands}) fingerprint-matched for {bundle}")
    deleted = getattr(ts, "blacklisted_bodies_deleted", 0)
    if deleted and os.environ.get("MLSPACES_REPLAY_ALLOW_BLACKLISTED", "").lower() not in ("1", "true", "yes"):
        _close_quietly(ts)
        raise BlacklistedBodiesInScene(
            f"{bundle}: {deleted} blacklisted body(ies) deleted from this scene, so the rebuild does not match "
            f"the recording; skipping. Set MLSPACES_REPLAY_ALLOW_BLACKLISTED=1 to evaluate it anyway.")
    _rebuild_env_renderer(task.env)    # fresh MjrContext so frames render from the corrected model

    from .recorded_task import restore_recorded_task_config
    restore_recorded_task_config(exp_config, ts, task, meta, bundle=bundle)

    entry = Path(spec_path).parent if spec_path is not None else bundle / "embodiment"
    return exp_config, ts, task, meta, entry
