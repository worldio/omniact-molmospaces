"""Route cuRobo's kernel imports to the CPU backend.

cuRobo reaches its compiled CUDA extensions from exactly five wrapper modules, each of which does
``from curobo.curobolib import <name>_cu`` at import time and then calls through that reference
(``curobolib/geom.py:20``, ``kinematics.py:20``, ``opt.py:20``, ``ls.py:19``, ``tensor_step.py:19``).
Substituting those five names is therefore the whole job -- ``MotionGen``, ``TrajOptSolver``,
``ArmReacher``, the graph planner and every cost term are untouched Python above that line.

Two orders have to work, because import order is not ours to control:

* **before** cuRobo is imported -- seed ``sys.modules`` so the wrappers pick up the shim on their
  own ``from ... import`` and never load the ``.so``;
* **after** -- rebind the attribute on each wrapper module that already imported the real one.

Doing only the first breaks whenever something imported cuRobo earlier (a config module, a sensor);
doing only the second still loads the CUDA extension, which fails outright on a machine with no
driver. So both.

Entry points that must not silently produce a wrong answer raise instead: the voxel collision
variants, the pose-distance pair and the three unused ``tensor_step`` modes are absent from the
plan's measured call set, and a stub that returned zeros would look like a working planner.
"""

from __future__ import annotations

import os
import sys
import types

import torch

from molmo_spaces.planner.curobo_cpu import geom, kinematics_fused, lbfgs_step, line_search, tensor_step

#: extension name -> {attribute: implementation}. Mirrors the ten entry points a joint-space plan
#: actually calls, as measured by ``scripts/curobo/profile_kernel_calls.py``.
_BACKEND: dict[str, dict[str, object]] = {
    "geom_cu": {
        "closest_point": geom.closest_point,
        "swept_closest_point": geom.swept_closest_point,
        "self_collision_distance": geom.self_collision_distance,
    },
    "kinematics_fused_cu": {
        "forward": kinematics_fused.forward,
        "backward": kinematics_fused.backward,
        "matrix_to_quaternion": kinematics_fused.matrix_to_quaternion,
    },
    "lbfgs_step_cu": {"forward": lbfgs_step.forward},
    "line_search_cu": {
        "line_search": line_search.line_search,
        "update_best": line_search.update_best,
    },
    "tensor_step_cu": {
        "step_idx_position2": tensor_step.step_idx_position2,
        "step_position_backward2": tensor_step.step_position_backward2,
    },
}

#: Present in the CUDA extensions but never reached on this path. Kept as explicit failures.
_UNPORTED: dict[str, tuple[str, ...]] = {
    "geom_cu": (
        "closest_point_voxel",
        "swept_closest_point_voxel",
        "pose_distance",
        "pose_distance_backward",
    ),
    "tensor_step_cu": (
        "step_position2",
        "step_acceleration",
        "step_acceleration_idx",
    ),
}

#: Which wrapper module holds each extension reference, for the after-the-fact rebind.
_WRAPPERS = {
    "geom_cu": "curobo.curobolib.geom",
    "kinematics_fused_cu": "curobo.curobolib.kinematics",
    "lbfgs_step_cu": "curobo.curobolib.opt",
    "line_search_cu": "curobo.curobolib.ls",
    "tensor_step_cu": "curobo.curobolib.tensor_step",
}


def _unsupported(extension: str, attribute: str):
    def raiser(*_args, **_kwargs):
        raise NotImplementedError(
            f"{extension}.{attribute} has no CPU implementation. It is not called by a joint-space "
            "plan, so it was left out rather than guessed at; port it if a new code path needs it."
        )

    return raiser


def _build_module(extension: str) -> types.ModuleType:
    module = types.ModuleType(f"curobo.curobolib.{extension}")
    module.__doc__ = f"CPU stand-in for {extension} (molmo_spaces.planner.curobo_cpu)."
    for attribute, implementation in _BACKEND[extension].items():
        setattr(module, attribute, implementation)
    for attribute in _UNPORTED.get(extension, ()):
        setattr(module, attribute, _unsupported(extension, attribute))
    return module


def _patch_warp_streams() -> None:
    """Make ``wp.stream_from_torch`` tolerate a CPU device.

    CUDA streams do not exist on CPU, and warp's helper raises ``ValueError: Expected a cuda
    device``. cuRobo is inconsistent about this: ``geom/transform.py:666`` guards with
    ``stream=None if not position.is_cuda else wp.stream_from_torch(...)``, but
    ``rollout/cost/bound_cost.py:783`` and its neighbours call it unconditionally. Returning ``None``
    for a CPU device is precisely what the guarded call sites do, and warp then uses the device's
    default stream.

    Patched here rather than in site-packages so the fix travels with this repo.
    """
    import warp

    # Returning None above is only half of it: warp then falls back to its *global default* device,
    # which is cuda:0 whenever a GPU is present, and the launch fails the other way round ("trying
    # to launch on device='cuda:0', but input array ... is on device=cpu"). cuRobo's warp launches
    # (warp_primitives.py:82, transform.py:666) pass no explicit device and rely entirely on the
    # stream, so the default has to be CPU for the inference to land there.
    warp.init()
    warp.set_device("cpu")

    for module in (warp, getattr(warp, "_src", None) and warp._src.torch):
        if module is None:
            continue
        original = getattr(module, "stream_from_torch", None)
        if original is None or getattr(original, "_cpu_tolerant", False):
            continue

        def tolerant(device=None, __original=original):
            if device is not None and "cuda" not in str(device):
                return None
            import torch

            if device is None and not torch.cuda.is_available():
                return None
            return __original(device)

        tolerant._cpu_tolerant = True
        module.stream_from_torch = tolerant


def _patch_cuda_synchronize() -> None:
    """Make ``torch.cuda.synchronize(device=cpu)`` a no-op instead of an error.

    cuRobo synchronises before reading its own stage timings. Most call sites are guarded by
    ``sync_cuda_time`` (``opt_base.py:173``, ``trajopt.py:1331``), but
    ``motion_gen.py:3114`` and ``:3262`` are unconditional, so setting the flag is not enough.

    Waiting for a CPU device to finish is vacuous -- the work already happened synchronously -- so
    returning immediately is the right semantics, not a workaround.
    """
    if getattr(torch.cuda.synchronize, "_cpu_tolerant", False):
        return
    original = torch.cuda.synchronize

    def tolerant(device=None):
        if device is not None and "cuda" not in str(device):
            return None
        if not torch.cuda.is_available():
            return None
        return original(device)

    tolerant._cpu_tolerant = True
    torch.cuda.synchronize = tolerant


def _patch_default_device() -> None:
    """Make cuRobo's default ``TensorDeviceType()`` mean CPU.

    ``curobo/types/base.py:21`` hardcodes ``device: torch.device = torch.device("cuda", 0)``, and
    cuRobo constructs ``TensorDeviceType()`` with no argument in many internal config dataclasses.
    Passing a CPU ``tensor_args`` into ``RobotConfig``/``MotionGenConfig`` therefore is not enough:
    anything built from an internal default still allocates on CUDA, and the first symptom is warp
    refusing to launch a CPU kernel with a ``cuda:0`` input ("input array for argument
    'retract_config' is on device=cuda:0").

    Rewriting the default rather than the call sites fixes the whole class at once. Explicit
    arguments still win, so a caller can still ask for CUDA.
    """
    from curobo.types.base import TensorDeviceType

    if getattr(TensorDeviceType.__init__, "_cpu_default", False):
        return
    original = TensorDeviceType.__init__

    def patched(self, device=None, dtype=None, collision_geometry_dtype=None,
                collision_gradient_dtype=None, collision_distance_dtype=None, **kwargs):
        if device is None or "cuda" in str(device):
            device = torch.device("cpu")
        supplied = {
            "dtype": dtype,
            "collision_geometry_dtype": collision_geometry_dtype,
            "collision_gradient_dtype": collision_gradient_dtype,
            "collision_distance_dtype": collision_distance_dtype,
        }
        original(self, device=device, **{k: v for k, v in supplied.items() if v is not None},
                 **kwargs)

    patched._cpu_default = True
    TensorDeviceType.__init__ = patched


def install() -> list[str]:
    """Point cuRobo at the CPU kernels. Returns the extension names that were substituted."""
    _patch_warp_streams()
    _patch_cuda_synchronize()
    _patch_default_device()
    substituted = []
    for extension, wrapper_name in _WRAPPERS.items():
        module = _build_module(extension)
        sys.modules[f"curobo.curobolib.{extension}"] = module

        # If curobo.curobolib is already imported, the wrappers resolve the name through the package
        # attribute, so set that too.
        package = sys.modules.get("curobo.curobolib")
        if package is not None:
            setattr(package, extension, module)

        # And if the wrapper module itself already ran, it holds its own reference to the real
        # extension; rebind that or it keeps calling CUDA.
        wrapper = sys.modules.get(wrapper_name)
        if wrapper is not None:
            setattr(wrapper, extension, module)
        substituted.append(extension)
    return substituted


def maybe_install() -> bool:
    """Install when asked to, or when there is no GPU to fall back on.

    ``CUROBO_CPU`` forces it either way (``0`` disables), so a GPU machine can exercise the CPU
    path and a CPU machine needs no configuration at all.
    """
    requested = os.environ.get("CUROBO_CPU")
    if requested is not None:
        if not bool(int(requested)):
            return False
    else:
        import torch

        if torch.cuda.is_available():
            return False
    install()
    return True
