"""Compact domain-randomization recipe for lean byte-exact re-sim: capture the DR outcome
as small arrays + per-geom texture paths and rebuild the model on a fresh compile. tex_data is rebuilt
by reloading each texture from its path."""
from __future__ import annotations

import hashlib
from typing import Callable

import numpy as np

# Small DR-mutated arrays stored directly; tex_data is rebuilt from the texture paths.
_LIGHT = (
    "light_pos", "light_dir", "light_specular", "light_ambient", "light_diffuse",
    "light_active", "light_castshadow",
)
_DYNAMICS = ("body_mass", "body_inertia", "geom_friction")
_TEXTURE_META = ("geom_matid", "mat_texid", "geom_rgba", "mat_rgba", "mat_specular", "mat_shininess")
DR_ARRAYS = _LIGHT + _DYNAMICS + _TEXTURE_META


class TextureApplyLog:
    """Context manager recording per geom the latest texture path the randomizer applied (wraps
    TextureRandomizer._apply_material_to_geom). Pass ``into`` a dict to accumulate across calls."""

    def __init__(self, texture_randomizer, into: dict | None = None) -> None:
        self._tr = texture_randomizer
        self.geom_texture_paths: dict[int, str] = {} if into is None else into
        self._orig: Callable | None = None

    def __enter__(self) -> "TextureApplyLog":
        tr = self._tr
        if tr is None:
            return self
        self._orig = tr._apply_material_to_geom
        mat_to_texture = tr.MAT_TO_TEXTURE

        def _hook(material_name, geom_id, mat_id):
            result = self._orig(material_name, geom_id, mat_id)
            if (result[0] if isinstance(result, tuple) else bool(result)):
                path = mat_to_texture.get(material_name, {}).get("_MainTex")
                if path:
                    self.geom_texture_paths[int(geom_id)] = str(path)
            return result

        tr._apply_material_to_geom = _hook
        return self

    def __exit__(self, *exc) -> None:
        if self._tr is not None and self._orig is not None:
            self._tr._apply_material_to_geom = self._orig


def snapshot_dr_base(model) -> dict:
    """Snapshot the post-compile, pre-randomize DR arrays + tex_data (for the per-episode reset)."""
    snap = {name: np.asarray(getattr(model, name)).copy() for name in DR_ARRAYS}
    snap["tex_data"] = np.asarray(model.tex_data).copy()
    return snap


def restore_dr_base(model, snapshot: dict) -> None:
    """Restore DR arrays + tex_data from a :func:`snapshot_dr_base`."""
    for name in DR_ARRAYS:
        getattr(model, name)[:] = snapshot[name]
    model.tex_data[:] = snapshot["tex_data"]


def dr_fingerprint(model) -> str:
    """Stable hash of names + tex_data + DR arrays; byte-reproducible across compiles (excludes the
    ephemeral placeholder path), so a reconstruction can self-verify against it."""
    h = hashlib.sha256()
    h.update(bytes(model.names))
    h.update(np.ascontiguousarray(np.asarray(model.tex_data)).tobytes())
    for name in DR_ARRAYS:
        h.update(np.ascontiguousarray(np.asarray(getattr(model, name))).tobytes())
    return h.hexdigest()


def capture_dr_deltas(model, geom_texture_paths: dict) -> dict:
    """Build the recipe {"arrays": {name: ndarray}, "geom_texture_paths": [[gid, path]]}, sorted by geom."""
    return {
        "arrays": {name: np.asarray(getattr(model, name)).copy() for name in DR_ARRAYS},
        "geom_texture_paths": sorted([int(gid), str(path)] for gid, path in geom_texture_paths.items()),
    }


def _write_texture_slot(model, tex_id: int, bitmap: np.ndarray) -> None:
    """Write a bitmap into a texture slot (channel-adjust + LANCZOS resize), matching the randomizer."""
    from PIL import Image

    width = int(model.tex_width[tex_id])
    height = int(model.tex_height[tex_id])
    nchannel = int(model.tex_nchannel[tex_id])
    adr = int(model.tex_adr[tex_id])

    if bitmap.ndim == 2:
        bitmap = bitmap[:, :, None]
    if bitmap.shape[2] != nchannel:
        if nchannel == 3 and bitmap.shape[2] == 4:
            bitmap = bitmap[:, :, :3]
        elif nchannel == 4 and bitmap.shape[2] == 3:
            alpha = np.full((*bitmap.shape[:2], 1), 255, bitmap.dtype)
            bitmap = np.concatenate([bitmap, alpha], axis=2)
        elif bitmap.shape[2] > nchannel:
            bitmap = bitmap[:, :, :nchannel]
        else:
            pad = np.full((*bitmap.shape[:2], nchannel - bitmap.shape[2]), 255, bitmap.dtype)
            bitmap = np.concatenate([bitmap, pad], axis=2)
    if bitmap.shape[:2] != (height, width):
        bitmap = np.array(
            Image.fromarray(bitmap.astype(np.uint8)).resize((width, height), Image.Resampling.LANCZOS)
        )
    flat = np.clip(bitmap, 0, 255).astype(np.uint8).flatten()
    size = height * width * nchannel
    model.tex_data[adr:adr + size] = flat[:size]


def load_thor_texture(texture_path: str) -> np.ndarray | None:
    """Load a texture bitmap as TextureRandomizer._load_single_texture_bitmap does (resolve under
    TEXTURE_FOLDER_PATH, PIL -> RGB uint8), for byte-exact tex_data."""
    import os
    from pathlib import Path

    from PIL import Image

    from molmo_spaces.env.arena.randomization.texture import TEXTURE_FOLDER_PATH

    resolved = (
        Path(texture_path)
        if os.path.isabs(texture_path)
        else TEXTURE_FOLDER_PATH / Path(texture_path).name
    )
    if not resolved.exists():
        return None
    img = Image.open(str(resolved))
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def apply_dr_deltas(
    model, deltas: dict, load_texture: Callable[[str], np.ndarray | None] | None = None
) -> None:
    """Reapply the recipe to a freshly compiled model (compiled with randomize_textures=True so the empty
    material/slot pool exists): set the DR arrays, then rebuild the textures.

    The datagen SHARES empty materials across geoms (the randomizer cycles a finite empty-material pool),
    so a shared material renders only its LAST-applied texture -- but geom_texture_paths (gtp) is per-geom
    and only logs a geom whose OWN randomizer pass applied a texture. A geom that shares a material with a
    textured geom but was itself given a color (or textured earlier and then overwritten) is absent from
    gtp, yet still renders the shared material's final texture (e.g. room_2_visual_0 is a solid-color geom
    sharing a material with a WoodGrains1.jpg geom, so it renders wood grain). Three things must line up:

    1. Last-wins texture. Give each RECORDED material the LAST path among its gtp geoms (gtp is stored
       sorted by geom id == the datagen's apply order, so the last entry per material is what the shared
       material actually rendered), not each geom's stale per-geom path.
    2. Compile-linked binding. Bind ONE fresh empty material per recorded material via the randomizer's own
       primitive _set_texture_bitmap_by_id (fresh compile-linked empty material + LANCZOS resize). The
       recorded mat_texid slot indices do NOT survive a fresh compile, so a manual slot write mis-binds;
       only the randomizer's fresh allocation renders. Binding per-GEOM instead would allocate one material
       per geom and wrap the ~200 pool, silently mis-binding ~30% of geoms; per-material binding fits.
    3. Full sharing. Point EVERY geom that recorded this material -- gtp and non-gtp alike -- at the fresh
       bound material. Reassigning only the gtp geoms leaves the non-gtp sharers (e.g. room_2) on the old,
       now-untextured material, rendering the flat placeholder instead of the shared texture.

    Reproduces what the datagen rendered from gtp + geom_matid alone -- no dependence on the (unstored)
    randomizer seed."""
    for name, value in deltas["arrays"].items():
        getattr(model, name)[:] = value
    bind_gtp_textures(
        model, np.asarray(deltas["arrays"]["geom_matid"]),
        deltas["geom_texture_paths"], load_texture=load_texture,
    )


def bind_gtp_textures(
    model, geom_matid, geom_texture_paths, load_texture=None
) -> int:
    """Bind each recorded material's last-wins gtp texture onto a fresh compile-linked empty material and
    point every recording geom (gtp and non-gtp) at it -- what apply_dr_deltas' docstring describes.
    Returns the number of materials (re)bound."""
    import mujoco
    from collections import OrderedDict

    rgb_role = int(mujoco.mjtTextureRole.mjTEXROLE_RGB.value)  # the role the renderer samples (1, not 0)
    # Every geom recording each material (gtp AND non-gtp) must follow it to the freshly bound replacement.
    geoms_by_material: dict[int, list[int]] = {}
    for gid in range(len(geom_matid)):
        mat_id = int(geom_matid[gid])
        if mat_id >= 0:
            geoms_by_material.setdefault(mat_id, []).append(gid)
    # gtp geoms per recorded material (for the last-wins texture); gtp is sorted by geom id == apply order.
    gtp_by_material: "OrderedDict[int, list]" = OrderedDict()
    for gid, path in geom_texture_paths:
        mat_id = int(geom_matid[gid])
        if mat_id >= 0:
            gtp_by_material.setdefault(mat_id, []).append((int(gid), path))
    from molmo_spaces.env.arena.randomization.texture import TextureRandomizer
    tr = TextureRandomizer(
        model,
        randomize_geom_rgba=False,
        randomize_material_rgba=False,
        randomize_material_specular=False,
        randomize_material_shininess=False,
        randomize_texture=True,
    )
    n_bound = 0
    for mat_id, entries in gtp_by_material.items():
        rep_gid = entries[0][0]
        texture_path = entries[-1][1]  # last write wins -> the shared material's final texture
        bitmap = load_texture(texture_path) if load_texture is not None else tr._load_texture_from_path(texture_path)
        if bitmap is None:
            continue
        if not tr._set_texture_bitmap_by_id(rep_gid, np.asarray(bitmap)):
            continue
        shared_mat = int(model.geom_matid[rep_gid])  # fresh compile-linked empty material just bound
        tex_slot = int(model.mat_texid[shared_mat, 0])  # _set_...by_id writes role 0; point RGB role at it
        if tex_slot >= 0:
            model.mat_texid[shared_mat, rgb_role] = tex_slot
        for gid in geoms_by_material.get(mat_id, ()):  # every recorder of this material follows it
            if gid != rep_gid:
                model.geom_matid[gid] = shared_mat
                tr._geom_to_empty_material[gid] = shared_mat
        n_bound += 1
    return n_bound
