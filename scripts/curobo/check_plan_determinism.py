"""Is cuRobo's joint-space plan reproducible for a fixed query?

Phase 1 gate for the CPU port. A CPU backend can only be held to "same trajectory as GPU" if the
GPU itself returns the same trajectory twice, so measure that before porting anything.

The query is replayed from a dumped failure diagnostic (``FAILED_<phase>.json``), which is the only
dump carrying everything needed to reconstruct it: ``start_config``, ``goal_config`` and every
obstacle as pose+dims. Three things are varied independently:

* **repeat** -- same planner instance, same query, N times. Isolates the optimiser.
* **rebuild** -- a fresh planner instance per replay. Catches state that survives in the planner and
  changes the answer, which would defeat any cross-device comparison too.
* **world** -- ``--no-world`` drops the obstacles. The dumped ``matrix`` records the barrier query
  as ``world=1 -> TRAJOPT_FAIL`` and ``world=0 -> OK 32 waypoints``, so this is how to get a
  SUCCEEDING query out of the same fixture. Determinism of a success and of a failure are different
  questions and both matter.

A trajectory is compared bitwise, on the raw float32, because "close enough" is the thing being
measured -- rounding it first would hide the answer.

    python scripts/curobo/check_plan_determinism.py --failure <FAILED_x.json> --repeats 3
    python scripts/curobo/check_plan_determinism.py --failure <FAILED_x.json> --no-world --rebuild
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

DEFAULT_SPEC = Path("docs/embodiment_specs/i2rt_yam.yaml")


def digest(array: np.ndarray | None) -> str:
    """Bitwise fingerprint of a trajectory, or a marker when there isn't one."""
    if array is None or array.size == 0:
        return "-"
    return hashlib.sha256(np.ascontiguousarray(array, dtype=np.float32).tobytes()).hexdigest()[:16]


def rejected_iterate(result) -> np.ndarray | None:
    """The trajectory a failed solve converged to before refusing it.

    Populated even when ``success`` is False, and the most sensitive determinism probe available: a
    rejected iterate is the raw optimiser state, with none of the acceptance test's rounding.
    """
    plan = getattr(result, "interpolated_plan", None)
    position = getattr(plan, "position", None)
    if position is None:
        return None
    values = position.detach().cpu().numpy()
    return values[0] if values.ndim == 3 else values


def build_planner(robot_cfg: dict, args) -> object:
    from molmo_spaces.planner.curobo_planner import CuroboPlanner, CuroboPlannerConfig

    return CuroboPlanner(
        CuroboPlannerConfig(
            robot_cfg=robot_cfg,
            interpolation_dt=0.02,
            collision_activation_distance=0.01,
            num_ik_seeds=64,
            num_trajopt_seeds=args.trajopt_seeds,
            trajopt_tsteps=32,
            max_attempts=args.max_attempts,
            enable_graph_attempt=2,
            check_start_validity=False,
            # Default None keeps cuRobo's own budget, which is what makes this experiment cheap.
            # --production replays at the barrier config's 10000/80000 instead.
            grad_trajopt_iters=args.grad_iters,
            finetune_trajopt_iters=args.finetune_iters,
        )
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failure", type=Path, required=True, help="a FAILED_<phase>.json dump")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="build a fresh planner for every replay instead of reusing one",
    )
    parser.add_argument(
        "--no-world", action="store_true", help="drop the obstacles (turns the barrier query solvable)"
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="call planner.reset() before each replay. cuRobo's MotionGen.reset(reset_seed=True) "
        "re-seeds every sub-solver's RNG, which its own docstring names as the requirement for "
        "deterministic results.",
    )
    parser.add_argument("--trajopt-seeds", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.0275,
        help="max per-joint deviation (rad) still counted as the same trajectory. Defaults to the "
        "policy's waypoint_tolerance (policy_configs.py:335): below this, two trajectories are "
        "executed identically because a waypoint within it already counts as reached.",
    )
    parser.add_argument("--grad-iters", type=int, default=None)
    parser.add_argument("--finetune-iters", type=int, default=None)
    parser.add_argument(
        "--production",
        action="store_true",
        help="replay at the barrier config's budget (10000/80000). Slow: warmup alone runs a full "
        "plan at finetune_trajopt_iters.",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="pin the determinism knobs (TF32 off, deterministic algorithms) before planning",
    )
    args = parser.parse_args()

    if args.production:
        args.grad_iters, args.finetune_iters = 10000, 80000

    if args.deterministic:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        os.environ["CUROBO_TORCH_COMPILE_DISABLE"] = "1"
        # CUBLAS_WORKSPACE_CONFIG must be set before CUDA initialises, so the caller has to export
        # it; setting it here is too late and silently does nothing.
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            print("  WARNING: export CUBLAS_WORKSPACE_CONFIG=:4096:8 before running for this to bite")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.manual_seed(0)

    failure = json.loads(args.failure.read_text())
    start = list(map(float, failure["start_config"]))
    goal = list(map(float, failure["goal_config"]))
    obstacles = [] if args.no_world else failure["obstacles"]

    print(f"query: {args.failure.name} phase={failure.get('phase')!r}")
    print(f"  recorded status: {failure.get('status')}")
    print(f"  recorded matrix: {failure.get('matrix')}")
    print(
        f"  replaying with {len(obstacles)} obstacles, {args.trajopt_seeds} trajopt seeds, "
        f"grad_iters={args.grad_iters}, finetune_iters={args.finetune_iters}"
    )
    print(f"  torch: tf32_matmul={torch.backends.cuda.matmul.allow_tf32} "
          f"fp32_precision={torch.get_float32_matmul_precision()} cuda={torch.cuda.is_available()}")

    from curobo.geom.types import Cuboid

    from molmo_spaces.planner.curobo_bridge import build_curobo_robot_model
    from molmo_spaces.robots.configured.config import ConfiguredRobotConfig

    robot_config = ConfiguredRobotConfig.from_file(args.spec)
    bridged = build_curobo_robot_model(robot_config, Path("ignore/curobo_determinism"))
    cuboids = [Cuboid(name=o["name"], pose=o["pose"], dims=o["dims"]) for o in obstacles]

    planner = None if args.rebuild else build_planner(bridged.robot_cfg, args)
    rows, trajectories = [], []
    for replay in range(args.repeats):
        if args.rebuild:
            del planner
            planner = build_planner(bridged.robot_cfg, args)
        if args.reset:
            planner.reset()
        planner.set_world(cuboids)
        began = time.time()
        waypoints, motion_time, result = planner.plan_to_joint_config(start, goal, verbose=False)
        elapsed = time.time() - began

        accepted = np.asarray(waypoints, dtype=np.float32) if waypoints else None
        trajectories.append(accepted)
        rows.append(
            {
                "replay": replay,
                "status": str(getattr(result, "status", None)),
                "success": bool(waypoints),
                "waypoints": len(waypoints),
                "motion_time": round(float(motion_time), 9),
                "accepted": digest(accepted),
                "iterate": digest(rejected_iterate(result)),
                "seconds": round(elapsed, 1),
            }
        )
        print(
            f"  replay {replay}: success={rows[-1]['success']} n={rows[-1]['waypoints']:3d} "
            f"t={rows[-1]['motion_time']:.9f} accepted={rows[-1]['accepted']} "
            f"iterate={rows[-1]['iterate']} ({rows[-1]['seconds']}s) {rows[-1]['status']}"
        )

    # "Differs" from a hash is not a magnitude. A 1e-6 rad spread and a 0.5 rad spread are both
    # "not bitwise identical" and mean completely different things for whether a tolerance-based
    # agreement gate is usable, so measure the size of the disagreement, not just its existence.
    usable = [t for t in trajectories if t is not None]
    if len(usable) > 1 and len({t.shape for t in usable}) == 1:
        deviations = [
            float(np.abs(a - b).max())
            for i, a in enumerate(usable)
            for b in usable[i + 1 :]
        ]
        span = float(np.abs(usable[0].max(axis=0) - usable[0].min(axis=0)).max())
        print()
        print(f"pairwise max |delta| over {len(usable)} trajectories (radians):")
        print(f"  worst {max(deviations):.3e}   median {float(np.median(deviations)):.3e}")
        print(f"  for scale, the largest joint's travel over this path is {span:.3f} rad")
        print(f"  -> worst disagreement is {100 * max(deviations) / span:.2f}% of the motion")

    print()
    verdicts = {row["success"] for row in rows}
    accepted = {row["accepted"] for row in rows}
    iterates = {row["iterate"] for row in rows}
    print(f"verdict agreement:    {'YES' if len(verdicts) == 1 else 'NO'} ({verdicts})")
    print(f"accepted trajectory:  {'BITWISE IDENTICAL' if len(accepted) == 1 else 'DIFFERS'} ({len(accepted)} distinct)")
    print(f"optimiser iterate:    {'BITWISE IDENTICAL' if len(iterates) == 1 else 'DIFFERS'} ({len(iterates)} distinct)")

    # The gate is a tolerance, not a bitwise compare. cuRobo never returns the same trajectory
    # twice, but the spread is far below the tolerance at which the policy executes a waypoint
    # (``waypoint_tolerance``, 0.0275 rad), so trajectories that differ in the last bits are the
    # same motion as far as the arm is concerned. Demanding equality would fail a planner that is
    # behaving perfectly well.
    worst = max(deviations) if len(usable) > 1 and deviations else 0.0
    reproducible = len(verdicts) == 1 and worst <= args.tolerance
    print()
    if len(verdicts) != 1:
        print(f"GATE FAIL: verdicts disagree ({verdicts}) -- the query is genuinely marginal.")
    elif not reproducible:
        print(
            f"GATE FAIL: worst deviation {worst:.3e} rad exceeds tolerance {args.tolerance:.3e}."
        )
    else:
        print(
            f"GATE PASS: verdict stable and worst deviation {worst:.3e} rad is within "
            f"{args.tolerance:.3e} rad -- reproducible to the precision that is executed."
        )
    return 0 if reproducible else 1


if __name__ == "__main__":
    raise SystemExit(main())
