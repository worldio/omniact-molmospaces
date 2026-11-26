"""Portable ${ROOT} tokens for generated-robot MJCF asset paths.

emit() writes tokens instead of resolved absolute paths, and the loader (the MjcfAssembler) materializes
them, so a model.xml carries no machine-specific absolute path yet compiles anywhere the asset library
is provisioned. Roots are resolved at runtime, honoring MLSPACES_ASSETS_DIR.
"""
from __future__ import annotations

from pathlib import Path

_CACHE_ROOT: dict[str, str] = {}


def _asset_cache_root() -> str | None:
    """Physical robot-asset cache root. Its location is baked into on-disk symlinks under
    ASSETS_DIR/robots (not any env var), so discover it by resolving one installed robot."""
    from molmo_spaces.molmo_spaces_constants import ASSETS_DIR

    key = str(ASSETS_DIR)
    if key not in _CACHE_ROOT:
        robots = Path(ASSETS_DIR) / "robots"
        root = None
        if robots.is_dir():
            for child in sorted(robots.iterdir()):
                model_xml = child / "model.xml"
                try:
                    if not model_xml.exists():
                        continue
                    parts = model_xml.resolve().parts
                except OSError:
                    continue
                for i in range(len(parts) - 1):
                    if parts[i] == "robots" and parts[i + 1] == child.name:
                        root = str(Path(*parts[:i]))
                        break
                if root:
                    break
        if root is None:
            return None  # not installed yet; re-probe next call rather than cache a miss
        _CACHE_ROOT[key] = root
    return _CACHE_ROOT[key]


def _named_roots() -> list[tuple[str, str]]:
    """Named machine roots (resolved), longest first so a nested root wins."""
    from molmo_spaces.molmo_spaces_constants import (
        ABS_PATH_OF_TOP_LEVEL_MOLMO_SPACES_DIR as _INSTALL,
        ASSETS_DIR as _ASSETS,
    )

    named = [
        ("ASSET_CACHE", _asset_cache_root()),
        ("ASSETS", str(Path(_ASSETS).resolve())),
        ("INSTALL", str(Path(_INSTALL).resolve())),
    ]
    return sorted(((n, r) for n, r in named if r), key=lambda nr: -len(nr[1]))


def to_tokens(text: str) -> str:
    """Rewrite machine asset roots in MJCF text to ${ROOT} tokens."""
    for name, root in _named_roots():
        text = text.replace(root + "/", "${" + name + "}/")
    return text


def from_tokens(text: str) -> str:
    """Resolve ${ROOT} tokens back to the runtime asset roots."""
    for name, root in _named_roots():
        text = text.replace("${" + name + "}/", root + "/")
    return text


def resolve_spec_asset_files(spec) -> None:
    """Materialize tokens on a freshly-loaded MjSpec's mesh/texture file paths (no-op if untokenized)."""
    for item in list(spec.meshes) + list(spec.textures):
        if item.file and "${" in item.file:
            item.file = from_tokens(item.file)
