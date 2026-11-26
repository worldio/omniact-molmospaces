"""MjOpenGLRenderer.close() frees the MjrContext under its own GL context and puts back the one it displaced.

mjr_freeContext deletes GL object names in whichever context is current on the thread, and EGLGLContext.free ends
with eglReleaseThread, which drops the thread's current context. close() also runs from __del__, i.e. possibly while
another renderer is mid-render, so both halves matter: free under our own context, restore the caller's afterwards.
"""

from types import SimpleNamespace

import pytest

from molmo_spaces.renderer.opengl_rendering import MjOpenGLRenderer


class FakeMjr:
    def __init__(self, calls):
        self._calls = calls

    def free(self):
        self._calls.append("mjr.free")


class FakeGL:
    """A context whose free() takes `restore`, like EGLGLContext."""

    def __init__(self, calls, displaced=None):
        self._calls, self._displaced = calls, displaced

    def make_current(self):
        self._calls.append("gl.make_current")
        return self._displaced

    def free(self, restore=None):
        self._calls.append(f"gl.free(restore={restore})")


class MujocoStyleGL:
    """A context whose free() takes no arguments and whose make_current() returns None, like mujoco's GLContext."""

    def __init__(self, calls):
        self._calls = calls

    def make_current(self):
        self._calls.append("gl.make_current")

    def free(self):
        self._calls.append("gl.free()")


def close(**attrs):
    """Run the real close() over a stub renderer, returning (call order, the stub)."""
    renderer = SimpleNamespace(**attrs)
    MjOpenGLRenderer.close(renderer)
    return renderer


def test_frees_the_mjr_context_under_its_own_context_and_restores_the_displaced_one():
    calls = []
    close(_gl_context=FakeGL(calls, displaced="other"), _mjr_context=FakeMjr(calls))
    assert calls == ["gl.make_current", "mjr.free", "gl.free(restore=other)"]


def test_does_not_restore_when_nothing_was_current():
    calls = []
    close(_gl_context=FakeGL(calls, displaced=None), _mjr_context=FakeMjr(calls))
    assert calls == ["gl.make_current", "mjr.free", "gl.free(restore=None)"]


def test_context_without_a_restore_argument_is_freed_plainly():
    calls = []
    close(_gl_context=MujocoStyleGL(calls), _mjr_context=FakeMjr(calls))
    assert calls == ["gl.make_current", "mjr.free", "gl.free()"]


def test_no_context_is_made_current_when_there_is_no_mjr_context_to_free():
    calls = []
    close(_gl_context=FakeGL(calls, displaced="other"), _mjr_context=None)
    assert calls == ["gl.free(restore=None)"]


def test_mjr_context_without_a_gl_context_is_still_freed():
    calls = []
    close(_gl_context=None, _mjr_context=FakeMjr(calls))
    assert calls == ["mjr.free"]


@pytest.mark.parametrize("attrs", [{}, {"_gl_context": None}, {"_gl_context": None, "_mjr_context": None}])
def test_close_of_a_partially_constructed_renderer_is_a_no_op(attrs):
    renderer = close(**attrs)
    assert renderer._gl_context is None and renderer._mjr_context is None


def test_close_is_idempotent():
    calls = []
    renderer = SimpleNamespace(_gl_context=FakeGL(calls, displaced="other"), _mjr_context=FakeMjr(calls))
    MjOpenGLRenderer.close(renderer)
    MjOpenGLRenderer.close(renderer)
    assert calls == ["gl.make_current", "mjr.free", "gl.free(restore=other)"]
