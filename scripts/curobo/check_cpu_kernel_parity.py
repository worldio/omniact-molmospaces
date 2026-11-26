"""Compare the CPU kernel backend against cuRobo's compiled CUDA kernels, for accuracy and speed.

Inputs are not reconstructed by hand -- every entry point is wrapped during a **real planning
query**, and the first call's arguments are cloned. Hand-built inputs test the layout I believe
cuRobo uses; captured inputs test the one it does use, and the gap between those is exactly where a
port goes wrong. It is also how the surface was established: a production plan calls ten entry
points, not the eighteen the extensions expose, and ``pose_distance`` is not among them.

Accuracy is judged against a tolerance, never equality. cuRobo does not reproduce its own GPU
trajectories bitwise (measured: 4.7e-4 rad worst pairwise deviation over five replays at the
shipped budget), so bitwise agreement is neither achievable nor required. The bar is that the CPU
kernel differs from CUDA by far less than CUDA differs from itself.

The timing half decides whether a full CPU port is a datagen path or only a verification harness.

    python scripts/curobo/check_cpu_kernel_parity.py [--production] [--iters 20]
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

from molmo_spaces.planner import curobo_cpu  # noqa: E402

DEFAULT_SPEC = Path("docs/embodiment_specs/i2rt_yam.yaml")
DEFAULT_FAILURE = Path(
    "ignore/runs/barrier25_fine2x_20260803_014500/on/"
    "GenericPickAndPlaceOmniCamCuroboBarrierConfig__i2rt_yam.single.fixed/20260803_003327/"
    "curobo_plans/FAILED_pregrasp.json"
)
TOLERANCE = 1e-5

# entry point -> CPU implementation. Only what a real plan calls; see the module docstring.
IMPLEMENTATIONS = {
    "kinematics_fused_cu.forward": ("kinematics_fused", "forward"),
    "kinematics_fused_cu.matrix_to_quaternion": ("kinematics_fused", "matrix_to_quaternion"),
    "tensor_step_cu.step_idx_position2": ("tensor_step", "step_idx_position2"),
    "tensor_step_cu.step_position_backward2": ("tensor_step", "step_position_backward2"),
    "geom_cu.self_collision_distance": ("geom", "self_collision_distance"),
    "geom_cu.closest_point": ("geom", "closest_point"),
    "line_search_cu.line_search": ("line_search", "line_search"),
    "line_search_cu.update_best": ("line_search", "update_best"),
    "lbfgs_step_cu.forward": ("lbfgs_step", "forward"),
    "kinematics_fused_cu.backward": ("kinematics_fused", "backward"),
    "geom_cu.swept_closest_point": ("geom", "swept_closest_point"),
}

MODULES = ("geom_cu", "kinematics_fused_cu", "lbfgs_step_cu", "line_search_cu", "tensor_step_cu")


def clone(value):
    return value.detach().clone() if torch.is_tensor(value) else value


def capture(spec: Path, failure: Path, production: bool, with_world: bool):
    """Record the first call's arguments for every kernel entry point during one real plan."""
    import curobo.curobolib as curobolib

    captured: dict[str, tuple] = {}
    originals: dict[str, callable] = {}

    for module_name in MODULES:
        module = __import__(f"curobo.curobolib.{module_name}", fromlist=["*"])
        setattr(curobolib, module_name, module)
        for attribute in dir(module):
            if attribute.startswith("_") or not callable(getattr(module, attribute)):
                continue
            name = f"{module_name}.{attribute}"
            originals[name] = getattr(module, attribute)

            def wrapper(*args, __name=name, __fn=originals[name]):
                if __name not in captured:
                    captured[__name] = tuple(clone(a) for a in args)
                return __fn(*args)

            setattr(module, attribute, wrapper)

    from curobo.geom.types import Cuboid

    from molmo_spaces.planner.curobo_bridge import build_curobo_robot_model
    from molmo_spaces.planner.curobo_planner import CuroboPlanner, CuroboPlannerConfig
    from molmo_spaces.robots.configured.config import ConfiguredRobotConfig

    diagnostic = json.loads(failure.read_text())
    bridged = build_curobo_robot_model(
        ConfiguredRobotConfig.from_file(spec), Path("ignore/curobo_determinism")
    )
    planner = CuroboPlanner(
        CuroboPlannerConfig(
            robot_cfg=bridged.robot_cfg,
            num_trajopt_seeds=8,
            trajopt_tsteps=32,
            max_attempts=8,
            check_start_validity=False,
            enable_graph_attempt=2,
            grad_trajopt_iters=10000 if production else None,
            finetune_trajopt_iters=80000 if production else None,
        )
    )
    obstacles = diagnostic["obstacles"] if with_world else []
    planner.set_world(
        [Cuboid(name=o["name"], pose=o["pose"], dims=o["dims"]) for o in obstacles]
    )
    captured.clear()  # discard warmup; capture only the query below
    planner.plan_to_joint_config(
        list(map(float, diagnostic["start_config"])),
        list(map(float, diagnostic["goal_config"])),
    )

    # Restore, so the benchmark below times the kernel and not the wrapper.
    for name, original in originals.items():
        module_name, attribute = name.split(".")
        setattr(__import__(f"curobo.curobolib.{module_name}", fromlist=["*"]), attribute, original)
    return captured, originals


def bench(fn, iters: int, synchronize: bool) -> float:
    """Median seconds per call. Median, not mean: one cold call would dominate a mean."""
    timings = []
    for _ in range(iters):
        began = time.perf_counter()
        fn()
        if synchronize:
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - began)
    timings.sort()
    return timings[len(timings) // 2]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--with-world", action="store_true")
    args = parser.parse_args()

    print("capturing real kernel arguments from a planning query...")
    captured, originals = capture(args.spec, args.failure, args.production, args.with_world)
    print(f"  captured {len(captured)} entry points: {', '.join(sorted(captured))}\n")

    everything_passed = True
    unported = []
    trivial = []
    for name in sorted(captured):
        if name not in IMPLEMENTATIONS:
            unported.append(name)
            continue
        module_name, function_name = IMPLEMENTATIONS[name]
        cpu_fn = getattr(getattr(curobo_cpu, module_name), function_name)
        cuda_fn = originals[name]

        cuda_args = tuple(clone(a) for a in captured[name])
        cpu_args = tuple(
            a.cpu() if torch.is_tensor(a) else a for a in (clone(v) for v in captured[name])
        )
        cuda_out = [clone(t) for t in cuda_fn(*cuda_args)]
        cpu_out = cpu_fn(*cpu_args)

        # Relative, not absolute. These tensors span wildly different scales -- link positions are
        # metres, but the position-clique gradients reach 2.9e+04 because ``traj_dt`` holds 1/dt and
        # the jerk term carries dt^3 = 125000. A fixed absolute bound fails a kernel that is correct
        # to 0.68 ULP, which is exactly what it did before this was changed.
        # Compare the *mutated arguments* as well as the returned values. Several of these kernels
        # carry state forward in their inputs -- lbfgs_step rolls its curvature history and moves
        # the x_0/grad_0 anchors, self_collision rewrites a sparsity mask -- and a port that gets
        # the return right while getting those wrong looks correct for exactly one iteration.
        #
        # The two groups are compared separately. Concatenating them and zipping is wrong whenever
        # CUDA returns a different number of tensors than the CPU version does: the lists shift
        # against each other and an output gets compared to an argument, which reads as a wildly
        # broken kernel (measured: 1.1e+04 on an lbfgs port that was exact).
        worst = 0.0
        magnitude = 0.0
        for cuda_group, cpu_group in ((cuda_out, cpu_out), (cuda_args, cpu_args)):
            for a, b in zip(cuda_group, cpu_group, strict=False):
                if not torch.is_tensor(a) or not torch.is_tensor(b) or not a.numel():
                    continue
                reference = a.float().cpu()
                magnitude = max(magnitude, float(reference.abs().max()))
                scale = max(float(reference.abs().max()), 1.0)
                worst = max(worst, float((reference - b.float().cpu()).abs().max()) / scale)

        # A comparison against an all-zero reference proves nothing: the gradient kernels are handed
        # zero gradients unless the query actually collides, so without --with-world they "pass" by
        # both sides returning zero. Say so rather than banking the result.
        if magnitude == 0.0:
            trivial.append(name)
        passed = worst <= TOLERANCE
        everything_passed &= passed
        cuda_seconds = bench(lambda: cuda_fn(*cuda_args), args.iters, synchronize=True)
        cpu_seconds = bench(lambda: cpu_fn(*cpu_args), args.iters, synchronize=False)
        print(
            f"{name:44s} {'PASS' if passed else 'FAIL'}{' (trivial)' if magnitude == 0.0 else ''} worst={worst:.2e}  "
            f"CUDA {cuda_seconds * 1e3:7.3f} ms  CPU {cpu_seconds * 1e3:8.3f} ms  "
            f"{cpu_seconds / cuda_seconds:6.1f}x"
        )

    if trivial:
        print(
            "\nWARNING: the reference output was all zeros for these, so the comparison proves "
            "nothing.\n         Re-run with --with-world to exercise them:"
        )
        for name in trivial:
            print(f"  {name}")

    if unported:
        print("\nnot yet ported (captured, no CPU implementation):")
        for name in unported:
            print(f"  {name}")

    print(f"\n{'ALL PASS' if everything_passed else 'FAILURES ABOVE'}")
    return 0 if everything_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
