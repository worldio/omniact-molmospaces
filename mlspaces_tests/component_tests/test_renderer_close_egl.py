"""Closing one renderer must not disturb another one's EGL context, including when __del__ does the closing."""

import gc

import numpy as np
import pytest
from mujoco import MjModel

pytest.importorskip("mujoco.egl")

from molmo_spaces.renderer.opengl_context import EGL  # noqa: E402
from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer  # noqa: E402

XML = "<mujoco><worldbody><light pos='0 0 3'/><geom type='box' size='.2 .2 .2' rgba='.8 .2 .2 1'/></worldbody></mujoco>"


@pytest.fixture(scope="module")
def model():
    return MjModel.from_xml_string(XML)


def renderer(model):
    return MjOpenGLRenderer(model=model, device_id=0, height=64, width=64)


def current_address():
    context = EGL.eglGetCurrentContext()
    return context.address if context else None


def test_closing_another_renderer_leaves_ours_current_and_rendering_unchanged(model):
    ours = renderer(model)
    expected = ours.render().copy()
    other = renderer(model)                 # creating it makes ITS context current
    ours._gl_context.make_current()         # ... and now we are the live renderer, mid-render
    other.close()
    assert current_address() == ours._gl_context._context.address
    assert np.array_equal(ours.render(), expected)
    ours.close()


def test_garbage_collecting_another_renderer_leaves_ours_current(model):
    ours = renderer(model)
    expected = ours.render().copy()
    doomed = renderer(model)
    ours._gl_context.make_current()
    del doomed                              # __del__ -> close(), at a moment we do not control
    gc.collect()
    assert current_address() == ours._gl_context._context.address
    assert np.array_equal(ours.render(), expected)
    ours.close()


def test_closing_the_current_renderer_leaves_no_context_current(model):
    ours = renderer(model)
    ours.render()
    ours.close()
    assert current_address() is None


def test_renderers_close_in_any_order(model):
    first, second = renderer(model), renderer(model)
    first.close()
    second.close()
    assert current_address() is None
