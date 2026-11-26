"""Generated MJCFs store portable ${ROOT} tokens instead of resolved absolute asset paths, and the
loader materializes them (molmo_spaces.utils.asset_tokens). These pin that contract: the round-trip is
lossless, paths outside the known roots are left alone (so the write-side guardrail can catch them), a
freshly-loaded MjSpec's file paths get resolved, and the robot-asset cache root is discovered by
resolving the on-disk symlink under ASSETS_DIR/robots -- not read from an env var.
"""

import os

import mujoco
import pytest

import molmo_spaces.molmo_spaces_constants as constants
from molmo_spaces.utils import asset_tokens


ROOTS = [("ASSET_CACHE", "/srv/dlcache"), ("ASSETS", "/srv/assets"), ("INSTALL", "/opt/repo")]


@pytest.fixture
def fixed_roots(monkeypatch):
    monkeypatch.setattr(asset_tokens, "_named_roots", lambda: list(ROOTS))


def test_tokenize_roundtrip_and_foreign_paths(fixed_roots):
    xml = (
        '<mesh file="/srv/dlcache/robots/arm/link0.obj"/>'
        '<texture file="/opt/repo/vendor/g/pad.png"/>'
        '<mesh file="/somewhere/else/loose.obj"/>'  # under no known root
    )
    tokenized = asset_tokens.to_tokens(xml)
    assert '${ASSET_CACHE}/robots/arm/link0.obj' in tokenized
    assert '${INSTALL}/vendor/g/pad.png' in tokenized
    assert '/somewhere/else/loose.obj' in tokenized  # foreign path untouched -> guardrail will flag it
    assert asset_tokens.from_tokens(tokenized) == xml  # lossless round-trip


def test_longest_root_wins(monkeypatch):
    # A nested root must win over its parent so the token is unambiguous.
    monkeypatch.setattr(asset_tokens, "_named_roots",
                        lambda: [("INNER", "/a/b/c"), ("OUTER", "/a")])
    assert asset_tokens.to_tokens('file="/a/b/c/x.obj"') == 'file="${INNER}/x.obj"'


def test_resolve_spec_asset_files(fixed_roots):
    spec = mujoco.MjSpec()
    spec.add_mesh(name="m", file="${ASSET_CACHE}/robots/arm/link0.obj")
    asset_tokens.resolve_spec_asset_files(spec)
    assert spec.meshes[0].file == "/srv/dlcache/robots/arm/link0.obj"


def test_asset_cache_root_discovers_symlink(tmp_path, monkeypatch):
    cache_ver = tmp_path / "dlcache" / "robots" / "widget" / "20240101"
    cache_ver.mkdir(parents=True)
    (cache_ver / "model.xml").write_text("<mujoco/>")
    robots = tmp_path / "assets" / "robots" / "widget"
    robots.mkdir(parents=True)
    os.symlink(cache_ver / "model.xml", robots / "model.xml")  # ASSETS_DIR/robots/<r>/model.xml -> cache

    monkeypatch.setattr(constants, "ASSETS_DIR", tmp_path / "assets")
    asset_tokens._CACHE_ROOT.clear()
    assert asset_tokens._asset_cache_root() == str(tmp_path / "dlcache")


def test_asset_cache_root_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(constants, "ASSETS_DIR", tmp_path / "empty")
    asset_tokens._CACHE_ROOT.clear()
    assert asset_tokens._asset_cache_root() is None
