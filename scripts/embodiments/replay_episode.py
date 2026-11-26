#!/usr/bin/env python3
"""Reconstruct an episode's renders from a sim-replay bundle alone (see data_generation/replay.py):

    dr_deltas.npz / dr_texture_paths.json  lean DR recipe; the model is rebuilt by fresh-compiling
                   the base scene and reapplying it -- see build_model_from_recipe.
    states.npz     per-frame MuJoCo integration state (mjSTATE_INTEGRATION)
    cameras.json   per-frame camera render inputs (pos/forward/up/fov) + intrinsics
    meta.json      instruction, per-step outcome, provenance, meta['recon'] (fresh-compile inputs)

Per frame: mj_setState + mj_forward, then re-render each camera as the datagen did -> the same pixels.
Needs the shared asset library (scenes/textures/embodiment spec) to fresh-compile.

The model-rebuild + spawn-pinning helpers live in molmo_spaces/data_generation/recon.py (shared with the
eval reconstruction in molmo_spaces/rollout/episode.py).

Usage::

    replay_episode.py BUNDLE_DIR --out OUT_DIR                 # save reconstructed mp4s + frames.npz
    replay_episode.py BUNDLE_DIR --reference RAW_DIR --strict  # verify pixels match <camera>.npy exactly
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import mujoco

from molmo_spaces.data_generation.recon import build_model_from_recipe


def load_bundle(bundle_dir: Path):
    bundle = Path(bundle_dir)
    with np.load(bundle / "states.npz") as sd:
        states = sd["states"]
        state_spec = int(sd["state_spec"])
    cameras = json.loads((bundle / "cameras.json").read_text())
    meta = json.loads((bundle / "meta.json").read_text())
    model = build_model_from_recipe(bundle, meta)  # rebuild the model from the lean DR recipe
    return model, states, state_spec, cameras, meta


def wods_key_from_meta(bundle_dir, meta):
    """The bundle's exact WODS trajectory key, derived from its own recipe -- no GCS listing, no n_frames
    guessing (both fragile: n_frames collides across trajs, and listing can pick the wrong config):

        <gen>/<config_selector>/<scene_dataset>-<data_split>_<house_id>_traj<episode_idx:02d>

    episode_idx == the converter's traj_{enumerate(winners)} index (guaranteed: house_raw_histories holds
    only saved winners, and a winner always has observations so it's never dropped -- see pipeline.py). The
    gen_<seed> component is the bundle path's gen_* ancestor. Returns None if any component is missing."""
    recon = meta.get("recon") or {}
    cfg = recon.get("config_selector")
    sd = meta.get("scene_dataset") or recon.get("scene_dataset")
    ds = meta.get("data_split") or recon.get("data_split")
    house = meta.get("house_id")
    idx = meta.get("episode_idx")
    gen = next((p.name for p in Path(bundle_dir).resolve().parents if p.name.startswith("gen_")), None)
    if gen is None or any(v is None for v in (cfg, sd, ds, house, idx)):
        return None
    return f"{gen}/{cfg}/{sd}-{ds}_{house}_traj{int(idx):02d}"


def reconstruct(bundle_dir: Path, camera_names=None):
    """Return {camera_name: np.ndarray(T, H, W, 3) uint8} rendered from the bundle alone."""
    # Imported here so the module docstring/CLI is usable without a GL context on hand.
    from molmo_spaces.data_generation.replay import render_rgb
    from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer

    model, states, state_spec, cameras, meta = load_bundle(bundle_dir)
    meta = {**meta, "wods_key": wods_key_from_meta(bundle_dir, meta)}
    data = mujoco.MjData(model)
    width, height = cameras["img_resolution"]
    by_name = {c["name"]: c for c in cameras["cameras"]}
    names = list(camera_names) if camera_names else list(by_name)
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"cameras {missing} not in bundle (have {list(by_name)})")

    renderer = MjOpenGLRenderer(model=model, width=int(width), height=int(height))
    try:
        frames = {n: [] for n in names}
        n_steps = len(states)
        for t in range(n_steps):
            mujoco.mj_setState(model, data, states[t], state_spec)
            mujoco.mj_forward(model, data)
            for name in names:
                cam = by_name[name]
                frames[name].append(
                    render_rgb(
                        renderer,
                        model,
                        data,
                        np.asarray(cam["pos"][t], dtype=np.float64),
                        np.asarray(cam["forward"][t], dtype=np.float64),
                        np.asarray(cam["up"][t], dtype=np.float64),
                        float(cam["fov"]),
                    ).copy()
                )
    finally:
        renderer.close()
    return {n: np.asarray(fs, dtype=np.uint8) for n, fs in frames.items()}, meta


def _compare(recon: np.ndarray, ref: np.ndarray) -> dict:
    t = min(len(recon), len(ref))
    a = recon[:t].astype(np.int64)
    b = ref[:t].astype(np.int64)
    diff = np.abs(a - b)
    per_frame_exact = np.array([np.array_equal(recon[i], ref[i]) for i in range(t)])
    total = a.size
    ndiff = int((diff != 0).sum())
    mse = float((diff.astype(np.float64) ** 2).mean())
    psnr = float("inf") if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)
    return {
        "frames_compared": t,
        "recon_frames": len(recon),
        "ref_frames": len(ref),
        "frames_exact": int(per_frame_exact.sum()),
        "max_abs_diff": int(diff.max()) if diff.size else 0,
        "pixels_differing_frac": (ndiff / total) if total else 0.0,
        "psnr_db": psnr,
        "exact": bool(per_frame_exact.all() and len(recon) == len(ref)),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bundle", type=Path, help="replay bundle directory (dr_deltas.npz + states.npz + cameras.json + meta.json)")
    ap.add_argument("--out", type=Path, default=None, help="write reconstructed frames here (frames.npz + per-camera mp4s)")
    ap.add_argument("--cameras", nargs="*", default=None, help="subset of camera names (default: all in the bundle)")
    ap.add_argument("--reference", type=Path, default=None, help="directory of <camera>.npy reference frames to compare against")
    ap.add_argument("--strict", action="store_true", help="exit non-zero unless every compared camera is bit-identical")
    ap.add_argument("--fps", type=float, default=None, help="mp4 fps (default: meta.json fps or 30)")
    ap.add_argument("--assets-dir", default=None, help="override MLSPACES_ASSETS_DIR for this replay")
    ap.add_argument("--cache-dir", default=None, help="override MLSPACES_CACHE_DIR for this replay")
    args = ap.parse_args(argv)
    if args.assets_dir:
        os.environ["MLSPACES_ASSETS_DIR"] = args.assets_dir
    if args.cache_dir:
        os.environ["MLSPACES_CACHE_DIR"] = args.cache_dir

    frames, meta = reconstruct(args.bundle, args.cameras)
    fps = args.fps or float(meta.get("fps") or 30.0)
    print(f"bundle: {args.bundle}")
    if meta.get("wods_key"):
        print(f"  wods key (derived from bundle meta): {meta['wods_key']}")
    # meta['instruction'] is BATCH-LAST-STALE for multi-episode batches; the authoritative per-episode caption
    # is the WODS sidecar <wods_key>.meta.json 'caption' (not stored in the bundle -- read it from the WODS root).
    print(f"  instruction (bundle meta -- batch-last-stale for multi-episode batches; authoritative caption is"
          f" the WODS <wods_key>.meta.json 'caption'): {meta.get('instruction')!r}")
    print(f"  frames: {meta.get('n_frames')}  cameras: {list(frames)}  resolution: {meta.get('resolution')}")

    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.out / "frames.npz", **frames)
        try:
            import imageio
            for name, arr in frames.items():
                imageio.mimwrite(args.out / f"{name}.mp4", arr, fps=fps, macro_block_size=1)
        except Exception as e:  # noqa: BLE001
            print(f"  (mp4 write skipped: {e})")
        print(f"  wrote {args.out}")

    ok = True
    if args.reference:
        print("verification vs reference:")
        for name, arr in frames.items():
            ref_path = args.reference / f"{name}.npy"
            if not ref_path.exists():
                print(f"  {name}: no reference ({ref_path.name}) -- skipped")
                continue
            ref = np.load(ref_path)
            m = _compare(arr, ref)
            ok = ok and m["exact"]
            verdict = "EXACT" if m["exact"] else "DIFF"
            print(f"  {name}: {verdict}  frames {m['frames_exact']}/{m['frames_compared']} exact"
                  f" (recon {m['recon_frames']} / ref {m['ref_frames']})"
                  f"  max|Δ|={m['max_abs_diff']}  pixdiff={m['pixels_differing_frac']:.3e}"
                  f"  psnr={m['psnr_db']:.1f}dB")
        if args.strict and not ok:
            print("STRICT: not bit-identical", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
