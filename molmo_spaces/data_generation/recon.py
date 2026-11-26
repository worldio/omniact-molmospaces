"""Shared episode-reconstruction helpers: rebuild an episode's ``MjModel`` from a sim-replay bundle's lean
DR recipe (see ``data_generation/replay.py``) and pin the datagen's spawned objects so the scene is
reproduced exactly. Imported by both the ``scripts/embodiments/replay_episode.py`` CLI (offline render
verification) and the eval reconstruction (``molmo_spaces/rollout/episode.py``) -- the one home for this
logic so neither reaches into the other. All ``molmo_spaces`` imports are lazy (inside the functions) so the
module loads without a GL/asset context."""

from __future__ import annotations

import base64
import contextlib
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import mujoco


def frozen_spawned_uids(meta: dict) -> list[str]:
    """The exact objects the datagen SPAWNED -- receptacles, added pickups, distractors, whatever the task
    draws -- recovered from the episode's frozen_config so the reconstruction can PIN them instead of
    re-drawing. Re-drawing is not reproducible: the spawn ``np.random.choice`` runs at ``work_item_seed``
    (= episode_seed minus the rejected-attempt count, which the bundle does not store) over a catalog whose
    sort order can drift. Spawned objects are keyed ``<namespace>/<slot>_<inst>/<uid>`` in the frozen
    SavedEpisode; native house objects use a flat ``<name>_<hash>_<inst>_<room>`` key (no ``/``) and are
    skipped. Regex the keys out of the base64 pickle (cheap, no unpickle/asset load) and return the uids
    ordered by (namespace, slot, inst) -- the datagen's insertion order. Task-agnostic: any spawn namespace
    matches, so no per-task rule is baked in. Empty if the task spawns nothing."""
    fc = (meta.get("obs_scene") or {}).get("frozen_config")
    if not fc:
        return []
    blob = base64.b64decode(fc)
    seen: dict[tuple[str, int, int], str] = {}
    for m in re.finditer(rb"([A-Za-z_][A-Za-z0-9_]*)/(\d+)_(\d+)/([A-Za-z0-9_]+)", blob):
        seen.setdefault((m.group(1).decode(), int(m.group(2)), int(m.group(3))), m.group(4).decode())
    return [seen[k] for k in sorted(seen)]


@contextlib.contextmanager
def pin_spawned_choices(uids: list[str]):
    """Pin the sampler's object-spawn ``np.random.choice`` draws to the frozen record. For each draw over a
    pool, return the frozen uids that are IN that pool -- in frozen (insertion) order -- and consume them;
    draws whose pool holds no frozen uid (numeric draws, joint-name draws, ...) are left untouched. The
    frozen record, not a per-task rule, decides each draw, so this is restore-side and task-agnostic.
    Handles both set draws (``size==k`` -> the k pooled frozen uids) and scalar draws (``size is None`` ->
    the one pooled frozen uid). NOTE: a multi-stage draw (e.g. pick_task's pre-select at 341 then select at
    369) is only partly reproducible this way -- the intermediate pool is itself randomized, so the final
    pool may not even contain the frozen uids; that case needs the capture-side RNG state. Patch removed on
    exit."""
    if not uids:
        yield
        return
    orig = np.random.choice
    remaining = list(uids)

    def _pooled(a):
        try:
            pool = set(a)
        except TypeError:
            return []
        return [u for u in remaining if u in pool]

    def choice(a, size=None, replace=True, p=None):
        if remaining and hasattr(a, "__len__"):
            matched = _pooled(a)
            if size is None and len(matched) == 1:  # scalar draw -> the one spawned uid
                remaining.remove(matched[0])
                return matched[0]
            if isinstance(size, (int, np.integer)) and matched and size == len(matched):
                for u in matched:
                    remaining.remove(u)
                return np.array(matched, dtype=object)
        return orig(a, size=size, replace=replace, p=p)

    np.random.choice = choice
    try:
        yield
    finally:
        np.random.choice = orig


# Fallback K-search span when the datagen attempt cap can't be read from the config.
_RNG_REAPPLY_K_SPAN_FALLBACK = 128


def datagen_max_attempts(exp_config) -> int:
    """Datagen's per-work-item attempt cap = samples_per_house * max_total_attempts_multiplier (pipeline.py:675):
    the exact upper bound for the winning episode's attempt index K. samples_per_house is int|None on the base
    config, so guard it and fall back when unset."""
    samples_per_house = getattr(exp_config.task_sampler_config, "samples_per_house", None)
    attempts_multiplier = getattr(exp_config.task_sampler_config, "max_total_attempts_multiplier", None)
    return (
        samples_per_house * attempts_multiplier
        if samples_per_house and attempts_multiplier
        else _RNG_REAPPLY_K_SPAN_FALLBACK
    )


def _rng_reapply_textures(model, base, arrays, scene_path, all_categories, episode_seed, episode_idx, want,
                          max_attempts):
    """Recover textures the gtp recipe can't (empty randomizer materials no geom logged a path for, e.g. a
    wall that shares/overwrites a material): replay the datagen's own seeded TextureRandomizer -- random
    state RandomState(current_seed + 2), current_seed = episode_seed - K (task_sampler.py:1054/1061) -- which
    reproduces EVERY pick, shared/unlogged included, and is CPU-independent (np.random). Restores the recorded
    non-tex DR arrays so only tex_data comes from the RNG, and K-searches the rejected-attempt offset,
    verified by dr_fingerprint. Returns True (model updated in place) on a match, else False. Cheap: one
    scene compile already done; each K only re-randomizes textures. K is the datagen attempt index of the
    winning episode (episode_seed = current_seed + episode_idx, pipeline.py:735; episode_idx counts every
    attempt incl failures), so K in [0, max_attempts) with max_attempts = samples_per_house * multiplier."""
    from molmo_spaces.data_generation.dr_deltas import (
        restore_dr_base, snapshot_dr_base, dr_fingerprint, DR_ARRAYS,
    )
    from molmo_spaces.env.arena.randomization.texture import TextureRandomizer
    from molmo_spaces.utils.scene_metadata_utils import get_scene_metadata
    post_apply = snapshot_dr_base(model)  # the apply_dr_deltas (gtp) result -- keep it if no K matches
    tr = TextureRandomizer(
        model=model, random_state=np.random.RandomState(0), randomize_geom_rgba=True,
        randomize_material_rgba=True, randomize_material_specular=True, randomize_material_shininess=True,
        randomize_texture=True, texture_paths=None, scene_metadata=get_scene_metadata(scene_path),
        rgba_perturbation_size=0.2,
    )
    data = mujoco.MjData(model)
    for K in [episode_idx] + [k for k in range(max_attempts) if k != episode_idx]:
        restore_dr_base(model, base)
        tr.random_state = np.random.RandomState((episode_seed - K) + 2)  # datagen's texture stream
        tr._next_empty_material_index = 0; tr._geom_to_empty_material = {}
        tr._geom_to_original_material = {}; tr._geom_to_original_texture = {}
        (tr.randomize if all_categories else tr.randomize_by_category)(data)
        for name in DR_ARRAYS:  # non-tex DR arrays exact from the recipe; only tex_data is RNG
            getattr(model, name)[:] = arrays[name]
        if dr_fingerprint(model) == want:
            print(f"RNG texture reconstruct: gtp lossy; matched dr_fingerprint at K={K} "
                  f"(current_seed {episode_seed - K})", file=sys.stderr)
            return True
    restore_dr_base(model, post_apply)  # search exhausted: keep the gtp result, not the last K's RNG garbage
    return False


def _fix_texture_role(model) -> int:
    """Recover the render-truth texture for empty-randomizer materials the RNG restored. datagen's
    TextureRandomizer._set_texture_bitmap_by_id writes the applied texture to mat_texid role 0
    (mjTEXROLE_USER), NOT the RGB role (1) the renderer samples -- so the fingerprint-exact saved model
    strands the texture at role 0 and renders flat color, while datagen drew the texture (a byte-exact
    reconstruction verified against dr_fingerprint still mis-renders these). Copy role 0 -> RGB role for
    empty-randomizer materials that carry a role-0 texture but no RGB texture. Render-only (mat_texid doesn't
    affect mj_step); runs after the fingerprint check. Color-path geoms have role 0 == -1 (the randomizer's
    color branch clears all roles) so they're left as color; native/apply-bound materials already set the RGB
    role, so this is a precise no-op for them. Returns the number of materials fixed."""
    mtx = model.mat_texid
    if getattr(mtx, "ndim", 1) != 2:
        return 0
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB.value)
    n = 0
    for m in range(model.nmat):
        if int(mtx[m, 0]) >= 0 and int(mtx[m, rgb]) < 0:
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, m) or ""
            if name.startswith("__TEXTURE_RANDOMIZER_MAT_"):
                model.mat_texid[m, rgb] = model.mat_texid[m, 0]
                n += 1
    return n


_EMPTY_MAT_PREFIX = "__TEXTURE_RANDOMIZER_MAT_"
_EMPTY_TEX_PREFIX = "__TEXTURE_RANDOMIZER_TEX_"


def _fix_empty_material_context(model) -> int:
    """Match the datagen WODS render for empty-randomizer materials randomize_scene left COLOR-path. datagen's
    env renderer bakes its MjrContext at scene-load, binding every __TEXTURE_RANDOMIZER_MAT_N__ to its
    compile-paired slot __TEXTURE_RANDOMIZER_TEX_N__; it is never rebuilt after randomize_scene, so the video
    renders that slot's CURRENT content even for a material whose mat_texid the randomizer cleared to color
    (mat_texid all -1). A fresh renderer honors mat_texid=-1 and draws the flat color instead, diverging where
    the slot was overwritten with an orphaned texture (a shared-pool leftover). Rebind the name-matched slot so
    the fresh render reproduces the video. No-op when the slot still holds the white placeholder (white*rgba ==
    rgba). Render-only, after the fingerprint check. Returns the number of materials rebound."""
    mtx = model.mat_texid
    if getattr(mtx, "ndim", 1) != 2:
        return 0
    rgb = int(mujoco.mjtTextureRole.mjTEXROLE_RGB.value)
    n = 0
    for m in range(model.nmat):
        if (np.asarray(mtx[m]) >= 0).any():  # already textured (incl. role-fixed) -> leave it
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MATERIAL, m) or ""
        if not name.startswith(_EMPTY_MAT_PREFIX):
            continue
        tex_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_TEXTURE, _EMPTY_TEX_PREFIX + name[len(_EMPTY_MAT_PREFIX):]
        )
        if tex_id >= 0:
            model.mat_texid[m, 0] = tex_id
            model.mat_texid[m, rgb] = tex_id
            n += 1
    return n


def _check_geometry(model, bundle: Path) -> bool:
    """Belt-and-suspenders geometry cross-check (the fingerprint covers names/textures/DR arrays but NOT geom
    positions/sizes -- it verifies positions only indirectly, via the shared spawn RNG stream). Load the
    recorded reset state and verify it fits the reconstructed model (same mjSTATE_INTEGRATION size) and that
    every geom world position is finite and in a sane range. A mismatch means the chosen scene is structurally
    inconsistent with the recorded trajectory (e.g. a best-geom-count fallback that isn't the real scene, or a
    decoupled spawn), which would silently apply the recorded qpos to the wrong bodies at replay. Warns
    loudly and returns False; True when consistent (or when there's nothing to check)."""
    try:
        with np.load(bundle / "states.npz") as sd:
            states = sd["states"]; state_spec = int(sd["state_spec"])
    except Exception:
        return True
    if getattr(states, "ndim", 0) != 2 or len(states) == 0:
        return True
    exp = mujoco.mj_stateSize(model, state_spec)
    if states.shape[1] != exp:
        print(f"RECON WARNING: recorded state size {states.shape[1]} != reconstructed model's {exp} for "
              f"{bundle.name} -- the reconstructed scene structure does not match the recording; the replay "
              f"would apply the recorded state to the wrong model.", file=sys.stderr)
        return False
    data = mujoco.MjData(model)
    mujoco.mj_setState(model, data, states[0], state_spec)
    mujoco.mj_forward(model, data)
    xpos = np.asarray(data.geom_xpos)
    if not np.isfinite(xpos).all() or (xpos.size and float(np.abs(xpos).max()) > 1e3):
        print(f"RECON WARNING: reset-state geometry check failed for {bundle.name} (non-finite or "
              f"out-of-range geom positions) -- the reconstructed scene may be wrong.", file=sys.stderr)
        return False
    return True


def build_model_from_recipe(bundle: Path, meta: dict):
    """Rebuild the episode's MjModel from the recipe: fresh-compile the base scene via the gen task
    sampler's setup_robot_scene (DR off, empty-material slots forced on), then reapply dr_deltas.npz +
    dr_texture_paths.json. Spawned objects are PINNED to the frozen_config record by pool membership
    (deterministic, task-agnostic; seeding alone can't reproduce them). Self-verifies against
    meta['recon']['dr_fingerprint']."""
    from molmo_spaces.data_generation.dr_deltas import (
        apply_dr_deltas, dr_fingerprint, snapshot_dr_base,
    )
    from molmo_spaces.data_generation.main import auto_import_configs, create_experiment_config
    from molmo_spaces.env.arena.arena_utils import modify_mjmodel_thor_articulated
    from molmo_spaces.utils.lazy_loading_utils import install_scene_with_objects_and_grasps_from_path

    recon = meta.get("recon")
    if not recon or not recon.get("config_selector"):
        raise ValueError(
            "bundle has no reconstruction recipe (meta['recon']); "
            "regenerate with a datagen build that captures dr_deltas."
        )
    # The bundle embeds the asset blacklist; point the loader at the in-bundle copy.
    blacklist = Path(bundle) / "asset_blacklist.txt"
    if blacklist.exists():
        os.environ.setdefault("MLSPACES_ASSET_BLACKLIST_PATH", str(blacklist))
    elif recon.get("asset_blacklist_path"):  # older bundles stored a path
        os.environ.setdefault("MLSPACES_ASSET_BLACKLIST_PATH", recon["asset_blacklist_path"])

    # The bundle's embodiment/model.xml carries ${ROOT} tokens; the assembler materializes them on load.
    spec_path = recon.get("embodiment_spec")
    if spec_path is not None and not Path(spec_path).is_absolute():
        spec_path = str(Path(bundle) / spec_path)

    auto_import_configs()
    exp_config, _ = create_experiment_config(
        recon["config_selector"], spec_path, recon.get("control_profile")
    )
    exp_config.scene_dataset = recon["scene_dataset"]
    exp_config.data_split = recon["data_split"]
    house = int(str(recon["house_id"]).rsplit("_", 1)[-1])
    exp_config.task_sampler_config.house_inds = [house]
    exp_config.task_sampler_config.randomize_lighting = False
    exp_config.task_sampler_config.randomize_textures = False
    exp_config.task_sampler_config.randomize_dynamics = False

    dd = np.load(bundle / "dr_deltas.npz")
    recorded_ngeom = int(dd["geom_friction"].shape[0])
    frozen = frozen_spawned_uids(meta)
    episode_seed = int(meta.get("episode_seed", 0)); episode_idx = int(meta.get("episode_idx", 0))

    _shared = {}

    def _build_scene(seed_override, pin):
        ts = exp_config.task_sampler_config.task_sampler_class(exp_config)
        scene_path = ts._current_house_scene_path()
        _shared["scene_path"] = scene_path
        install_scene_with_objects_and_grasps_from_path(scene_path)
        if seed_override is not None:
            ts.seed_task_sampling(seed_override)  # reproduces the datagen's spawn draws
        ctx = pin_spawned_choices(frozen) if pin else contextlib.nullcontext()
        with ctx:
            m = ts.setup_robot_scene(
                robot_config=exp_config.robot_config,
                scene_file_path=scene_path,
                randomize_textures=True,  # FORCE empty material/texture slots so tex_data slots exist
                environment_light_intensity=exp_config.environment_light_intensity,
            )
        modify_mjmodel_thor_articulated(m, mujoco.MjData(m))
        return m

    arrays = {a: dd[a] for a in dd.files}
    paths = json.loads((bundle / "dr_texture_paths.json").read_text())
    want = recon.get("dr_fingerprint")
    all_categories = getattr(exp_config.task_sampler_config, "randomize_textures_all", False)
    no_rng = os.environ.get("REPLAY_NO_RNG_FALLBACK") == "1"
    # K (the winning episode's datagen attempt index) is bounded by the loop cap (pipeline.py:675).
    max_attempts = datagen_max_attempts(exp_config)

    def _reconstruct(seed_override, pin):
        """Build the scene at this seed and reapply the recipe (apply_dr_deltas, then the seeded RNG texture
        fallback when apply is lossy). Returns (model, fp_ok); model is None when the geom count doesn't even
        match the recording (a clearly-wrong scene)."""
        m = _build_scene(seed_override, pin)
        if m.ngeom != recorded_ngeom:
            return None, False
        # A different scene can collide on ngeom yet differ in nbody/nmat: the recipe's DR arrays would then
        # broadcast-fail in apply_dr_deltas. Skip such seeds instead of crashing the whole search.
        if any(getattr(m, name, None) is not None and np.asarray(getattr(m, name)).shape != np.asarray(value).shape
               for name, value in arrays.items()):
            return None, False
        base = snapshot_dr_base(m)  # clean post-compile state, for the RNG texture fallback
        apply_dr_deltas(m, {"arrays": arrays, "geom_texture_paths": paths})
        # gtp is lossy for empty randomizer materials no geom logged a path for; recover the exact tex_data by
        # replaying the datagen's seeded TextureRandomizer (byte-exact, fingerprint-verified).
        if want is not None and dr_fingerprint(m) != want and not no_rng:
            _rng_reapply_textures(m, base, arrays, _shared.get("scene_path"), all_categories,
                                  episode_seed, episode_idx, want, max_attempts)
        return m, (want is None or dr_fingerprint(m) == want)

    # First attempt: pin spawned objects from frozen_config when present (a no-op pin otherwise).
    model, ok = _reconstruct(None, pin=True)
    if not ok:  # first build failed the fingerprint (or, with no want, failed the geom count) -> search seeds
        # The scene didn't reconstruct to the recorded fingerprint (e.g. a pre-fix bundle with no obs_scene,
        # so nothing pins and the unpinned spawn draws diverge). The spawn np.random.choice runs at
        # episode_seed - episode_idx - rejected_attempt_count (the bundle stores only episode_seed); search
        # that offset K and take the seed whose FULL reconstruction fingerprint-matches. This is fp-gated,
        # not ngeom-gated: geom count alone collides (a different scene with the same count), which silently
        # yielded the wrong objects before. Fall back to the best geom-count match if nothing fp-matches.
        cand = [episode_idx] + [k for k in range(episode_idx + 8) if k != episode_idx]
        fallback = model
        for K in cand:
            m, ok2 = _reconstruct(episode_seed - K, pin=False)
            if m is None:
                continue
            if fallback is None:
                fallback = m
            if ok2:
                model, ok = m, True
                print(f"K-search: fingerprint-matched at K={K} (seed {episode_seed - K})", file=sys.stderr)
                break
        if not ok:
            model = fallback  # no fp-match; keep the best geom-count match (the check below warns/raises)
            print(f"RECON WARNING: no fingerprint-matching seed in the search range (K in {cand}) for "
                  f"{bundle.name}; falling back to the best geom-count match -- the reconstructed scene may "
                  f"have the WRONG spawned objects/positions (unpinnable bundle whose true seed is outside the "
                  f"range, or a scene that no longer compiles identically).", file=sys.stderr)
        if model is None:
            raise RuntimeError(f"reconstruction failed: no seed episode_seed-K (K in {cand}) yields ngeom={recorded_ngeom}")

    if want is not None:
        got = dr_fingerprint(model)
        if got != want:
            msg = (
                f"reconstruction fingerprint mismatch (got {got[:16]}, want {want[:16]}); "
                "the base scene did not compile identically -- check asset/config/blacklist versions."
            )
            if os.environ.get("REPLAY_FINGERPRINT_WARN"):
                print(f"WARNING: {msg}", file=sys.stderr)
            else:
                raise RuntimeError(msg)

    _check_geometry(model, bundle)  # (c) direct geom-position cross-check; warns if the scene is inconsistent

    # Render-fidelity pass (after the fingerprint check, render-only): datagen's texture randomizer wrote the
    # applied texture to the wrong mat_texid role, so the fingerprint-exact saved model strands some empty-
    # randomizer textures at role 0 and renders flat color where datagen drew the texture. Restore the RGB
    # role from role 0 for those materials. No-op for apply-matched/native materials (RGB role already set).
    if os.environ.get("REPLAY_NO_ROLE_FIX") != "1":  # knob to render the raw (unfixed) model for validation
        n = _fix_texture_role(model)
        if n:
            print(f"texture-role fix: restored the RGB role from role 0 for {n} empty-randomizer material(s) "
                  f"the RNG left stranded (datagen TextureRandomizer wrote role 0, not the RGB role)",
                  file=sys.stderr)
        n2 = _fix_empty_material_context(model)
        if n2:
            print(f"empty-material context rebind: rebound the scene-load texture slot on {n2} color-path "
                  f"empty material(s) that the datagen env MjrContext still renders (stale scene-load binding)",
                  file=sys.stderr)
    return model


def apply_recipe_to_compiled_model(model, base, bundle, meta, exp_config, scene_path):
    """Apply the recorded DR recipe onto an ALREADY-COMPILED model (no fresh compile), given its clean
    post-compile DR ``base`` (``snapshot_dr_base``). Same recipe path as build_model_from_recipe's per-scene
    tail -- restore the base, ``apply_dr_deltas``, recover empty-material textures via the seeded RNG reapply,
    verify the fingerprint, then the render-role fixes -- but on a model the caller already has. For the eval
    server, which owns sample_task's compiled model + (with capture_replay on) its ``dr_base_snapshot``, so it
    need not compile the scene a second time."""
    from molmo_spaces.data_generation.dr_deltas import apply_dr_deltas, dr_fingerprint, restore_dr_base

    bundle = Path(bundle)
    dd = np.load(bundle / "dr_deltas.npz")
    arrays = {a: dd[a] for a in dd.files}
    paths = json.loads((bundle / "dr_texture_paths.json").read_text())
    want = (meta.get("recon") or {}).get("dr_fingerprint")
    all_categories = getattr(exp_config.task_sampler_config, "randomize_textures_all", False)
    max_attempts = datagen_max_attempts(exp_config)
    episode_seed = int(meta.get("episode_seed", 0))
    episode_idx = int(meta.get("episode_idx", 0))

    restore_dr_base(model, base)                                        # clean post-compile state
    apply_dr_deltas(model, {"arrays": arrays, "geom_texture_paths": paths})
    if want is not None and dr_fingerprint(model) != want and os.environ.get("REPLAY_NO_RNG_FALLBACK") != "1":
        _rng_reapply_textures(model, base, arrays, scene_path, all_categories,
                              episode_seed, episode_idx, want, max_attempts)
    if want is not None:
        got = dr_fingerprint(model)
        if got != want:
            msg = (f"apply_recipe_to_compiled_model fingerprint mismatch (got {got[:16]}, want {want[:16]}) "
                   f"for {bundle.name}: the sampled model did not match the recipe")
            if os.environ.get("REPLAY_FINGERPRINT_WARN"):
                print(f"WARNING: {msg}", file=sys.stderr)
            else:
                raise RuntimeError(msg)
    if os.environ.get("REPLAY_NO_ROLE_FIX") != "1":
        _fix_texture_role(model)
        _fix_empty_material_context(model)
    return model
