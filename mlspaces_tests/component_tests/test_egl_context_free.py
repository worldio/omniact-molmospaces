"""EGLGLContext.make_current reports the context it displaced, and free puts it back after eglReleaseThread.

The EGL calls are faked here; mlspaces_tests/component_tests/test_renderer_close_egl.py drives the real thing.
"""

from types import SimpleNamespace

import pytest

from molmo_spaces.renderer import opengl_context as ctx


class Handle:
    def __init__(self, address):
        self.address = address

    def __bool__(self):
        return bool(self.address)


class FakeEGL:
    """Records the calls a free()/make_current() makes, and reports whichever context it left current."""

    EGL_NO_SURFACE = "no-surface"
    EGL_NO_CONTEXT = Handle(0)

    def __init__(self, current=None, destroy_raises=False):
        self.current, self.calls, self._destroy_raises = current or self.EGL_NO_CONTEXT, [], destroy_raises

    def eglGetCurrentContext(self):
        return self.current

    def eglMakeCurrent(self, display, draw, read, context):
        self.calls.append(("makeCurrent", context.address))
        self.current = context
        return True

    def eglDestroyContext(self, display, context):
        self.calls.append(("destroy", context.address))
        if self._destroy_raises:
            raise ctx.EGLError(err=0)

    def eglReleaseThread(self):
        self.calls.append(("releaseThread",))
        self.current = self.EGL_NO_CONTEXT      # EGL spec: releases the thread's current context

    def eglGetError(self):
        return 0


@pytest.fixture
def egl(monkeypatch):
    fake = FakeEGL()
    monkeypatch.setattr(ctx, "EGL", fake)
    monkeypatch.setattr(ctx, "EGL_DISPLAY_INITIALIZED", True)
    monkeypatch.setattr(ctx, "EGL_DISPLAY", "display")
    return fake


@pytest.fixture(autouse=True)
def stubs():
    """Hands out EGLGLContext stubs and defuses their __del__, which would reach the real EGL with a fake handle."""
    made = []

    def context(address):
        obj = ctx.EGLGLContext.__new__(ctx.EGLGLContext)
        obj._context = Handle(address)
        made.append(obj)
        return obj

    yield context
    for obj in made:
        obj._context = None


def test_make_current_returns_the_displaced_context(egl, stubs):
    egl.current = Handle(7)
    assert stubs(1).make_current().address == 7


def test_make_current_returns_none_when_no_context_was_current(egl, stubs):
    assert stubs(1).make_current() is None


def test_free_rebinds_the_displaced_context_after_releasing_the_thread(egl, stubs):
    egl.current = Handle(7)
    stubs(1).free(restore=Handle(7))
    assert egl.calls == [("destroy", 1), ("releaseThread",), ("makeCurrent", 7)]
    assert egl.current.address == 7


def test_free_unbinds_its_own_context_first_then_rebinds_the_other(egl, stubs):
    egl.current = Handle(1)                     # ours is current, e.g. we just rendered
    stubs(1).free(restore=Handle(7))
    assert egl.calls == [("makeCurrent", 0), ("destroy", 1), ("releaseThread",), ("makeCurrent", 7)]


def test_free_never_rebinds_the_context_it_just_destroyed(egl, stubs):
    egl.current = Handle(1)
    stubs(1).free(restore=Handle(1))          # the displaced context was ours
    assert egl.calls == [("makeCurrent", 0), ("destroy", 1), ("releaseThread",)]
    assert egl.current.address == 0


def test_free_without_restore_leaves_the_thread_unbound(egl, stubs):
    stubs(1).free()
    assert egl.calls == [("destroy", 1), ("releaseThread",)]


def test_free_swallows_egl_errors_and_clears_the_handle(monkeypatch, stubs):
    fake = FakeEGL(current=Handle(1), destroy_raises=True)
    monkeypatch.setattr(ctx, "EGL", fake)
    monkeypatch.setattr(ctx, "EGL_DISPLAY_INITIALIZED", True)
    monkeypatch.setattr(ctx, "EGL_DISPLAY", "display")
    obj = stubs(1)
    obj.free(restore=Handle(7))
    assert obj._context is None


def test_free_is_a_no_op_before_the_display_is_initialized(monkeypatch, stubs):
    fake = FakeEGL()
    monkeypatch.setattr(ctx, "EGL", fake)
    monkeypatch.setattr(ctx, "EGL_DISPLAY_INITIALIZED", False)
    stubs(1).free(restore=Handle(7))
    assert fake.calls == []
