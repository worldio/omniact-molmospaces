"""Which cuRobo CUDA kernels does a real plan actually call, how often, and for how long?

Ports are expensive and the five extensions expose eighteen entry points, but a single planning
query does not touch all of them, and the ones it touches differ in cost by orders of magnitude.
Enumerating the surface from the source says what *could* be called; this says what *is*, so the
port can be ordered by measured share of runtime rather than by line count.

Every entry point on every extension module is wrapped with a counter and a synchronised timer,
then one query is replayed from a dumped diagnostic.

    python scripts/curobo/profile_kernel_calls.py --failure <FAILED_x.json> [--production]
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

MODULES = (
    "geom_cu",
    "kinematics_fused_cu",
    "lbfgs_step_cu",
    "line_search_cu",
    "tensor_step_cu",
)


def instrument():
    """Wrap every public callable on each extension with a counter and timer."""
    counts: dict[str, int] = defaultdict(int)
    seconds: dict[str, float] = defaultdict(float)
    import curobo.curobolib as curobolib

    for module_name in MODULES:
        module = getattr(curobolib, module_name, None)
        if module is None:
            module = __import__(f"curobo.curobolib.{module_name}", fromlist=["*"])
        for attribute in dir(module):
            if attribute.startswith("_"):
                continue
            original = getattr(module, attribute)
            if not callable(original):
                continue

            def wrapper(*args, __name=f"{module_name}.{attribute}", __fn=original):
                began = time.perf_counter()
                result = __fn(*args)
                torch.cuda.synchronize()
                seconds[__name] += time.perf_counter() - began
                counts[__name] += 1
                return result

            setattr(module, attribute, wrapper)
    return counts, seconds


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure", type=Path, required=True)
    parser.add_argument("--spec", type=Path, default=Path("docs/embodiment_specs/i2rt_yam.yaml"))
    parser.add_argument("--no-world", action="store_true")
    parser.add_argument("--production", action="store_true")
    args = parser.parse_args()

    counts, seconds = instrument()

    from curobo.geom.types import Cuboid

    from molmo_spaces.planner.curobo_bridge import build_curobo_robot_model
    from molmo_spaces.planner.curobo_planner import CuroboPlanner, CuroboPlannerConfig
    from molmo_spaces.robots.configured.config import ConfiguredRobotConfig

    failure = json.loads(args.failure.read_text())
    robot_config = ConfiguredRobotConfig.from_file(args.spec)
    bridged = build_curobo_robot_model(robot_config, Path("ignore/curobo_determinism"))
    planner = CuroboPlanner(
        CuroboPlannerConfig(
            robot_cfg=bridged.robot_cfg,
            num_trajopt_seeds=8,
            trajopt_tsteps=32,
            max_attempts=8,
            check_start_validity=False,
            enable_graph_attempt=2,
            grad_trajopt_iters=10000 if args.production else None,
            finetune_trajopt_iters=80000 if args.production else None,
        )
    )
    obstacles = [] if args.no_world else failure["obstacles"]
    planner.set_world([Cuboid(name=o["name"], pose=o["pose"], dims=o["dims"]) for o in obstacles])

    # Reset after construction: warmup runs a full plan and would otherwise dominate the profile.
    counts.clear()
    seconds.clear()
    began = time.perf_counter()
    waypoints, _, result = planner.plan_to_joint_config(
        list(map(float, failure["start_config"])), list(map(float, failure["goal_config"]))
    )
    total = time.perf_counter() - began

    print(f"plan: success={bool(waypoints)} waypoints={len(waypoints)} wall={total:.2f}s")
    print(f"{'entry point':42s} {'calls':>10s} {'seconds':>9s} {'% wall':>7s}  {'us/call':>9s}")
    for name in sorted(seconds, key=lambda k: -seconds[k]):
        print(
            f"{name:42s} {counts[name]:10d} {seconds[name]:9.2f} "
            f"{100 * seconds[name] / total:6.1f}% {1e6 * seconds[name] / counts[name]:9.1f}"
        )
    measured = sum(seconds.values())
    print(f"{'TOTAL in kernels':42s} {sum(counts.values()):10d} {measured:9.2f} "
          f"{100 * measured / total:6.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
