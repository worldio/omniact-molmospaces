"""Do CPU and GPU agree on a plan? The acceptance gate for the CPU kernel backend.

Two claims are tested, and they are not the same claim:

1. **Verdict** -- the same queries succeed. This is the requirement, so it fails the run.
2. **Trajectory** -- for queries that succeed on both, the paths agree to within the envelope the
   planner reproduces *itself* to. Measured on this machine: 4.7e-4 rad worst pairwise deviation
   over five GPU replays at the shipped budget, and 1.3e-2 rad at cuRobo's default budget. Holding
   CPU to anything tighter than GPU-vs-GPU would be holding it to a standard GPU does not meet.

Each device runs in its own process. The CPU path installs global patches (warp's default device,
cuRobo's default ``TensorDeviceType``, ``torch.cuda.synchronize``), so sharing an interpreter with
the GPU run would contaminate it and quietly compare CPU against CPU.

    python scripts/curobo/check_cpu_gpu_agreement.py --repeats 3
    python scripts/curobo/check_cpu_gpu_agreement.py --with-world --production
"""

from __future__ import annotations

import os

os.environ.setdefault("MUJOCO_GL", "egl")

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

DEFAULT_SPEC = Path("docs/embodiment_specs/i2rt_yam.yaml")
DEFAULT_FAILURE = Path(
    "ignore/runs/barrier25_fine2x_20260803_014500/on/"
    "GenericPickAndPlaceOmniCamCuroboBarrierConfig__i2rt_yam.single.fixed/20260803_003327/"
    "curobo_plans/FAILED_pregrasp.json"
)
# The GPU-vs-GPU envelope, and the point below which two trajectories execute identically anyway:
# the policy counts a waypoint reached at waypoint_tolerance = 0.0275 rad (policy_configs.py:335).
DEFAULT_TOLERANCE = 0.0275


def _plan_on_device(args, device: str) -> list[dict]:
    """Run the query `repeats` times on one device, in a fresh interpreter, and return the results."""
    script = f"""
import json, time, sys
from pathlib import Path
device = {device!r}
if device == "cpu":
    from molmo_spaces.planner.curobo_cpu.install import install
    install()
import torch
from curobo.geom.types import Cuboid
from molmo_spaces.planner.curobo_bridge import build_curobo_robot_model
from molmo_spaces.planner.curobo_planner import CuroboPlanner, CuroboPlannerConfig
from molmo_spaces.robots.configured.config import ConfiguredRobotConfig

diagnostic = json.loads(Path({str(args.failure)!r}).read_text())
bridged = build_curobo_robot_model(
    ConfiguredRobotConfig.from_file(Path({str(args.spec)!r})), Path("ignore/curobo_agreement")
)
planner = CuroboPlanner(CuroboPlannerConfig(
    robot_cfg=bridged.robot_cfg, device=device,
    num_trajopt_seeds={args.trajopt_seeds}, trajopt_tsteps=32, max_attempts={args.max_attempts},
    check_start_validity=False, enable_graph_attempt=2,
    grad_trajopt_iters={10000 if args.production else None},
    finetune_trajopt_iters={80000 if args.production else None},
))
obstacles = diagnostic["obstacles"] if {args.with_world} else []
planner.set_world([Cuboid(name=o["name"], pose=o["pose"], dims=o["dims"]) for o in obstacles])
start = list(map(float, diagnostic["start_config"]))
goal = list(map(float, diagnostic["goal_config"]))
results = []
for _ in range({args.repeats}):
    began = time.time()
    waypoints, motion_time, result = planner.plan_to_joint_config(start, goal)
    results.append({{
        "success": bool(waypoints),
        "waypoints": waypoints,
        "motion_time": float(motion_time),
        "seconds": time.time() - began,
        "status": str(getattr(result, "status", None)),
    }})
print("@@RESULTS@@" + json.dumps(results))
"""
    environment = dict(os.environ, PYTHONPATH=os.getcwd())
    if device == "cpu":
        environment["CUROBO_CPU"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=environment
    )
    marker = [line for line in completed.stdout.splitlines() if line.startswith("@@RESULTS@@")]
    if not marker:
        print(completed.stdout[-2000:])
        print(completed.stderr[-3000:], file=sys.stderr)
        raise SystemExit(f"{device} run produced no results")
    return json.loads(marker[0][len("@@RESULTS@@") :])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--failure", type=Path, default=DEFAULT_FAILURE)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--trajopt-seeds", type=int, default=8)
    parser.add_argument("--max-attempts", type=int, default=8)
    parser.add_argument("--with-world", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--tolerance", type=float, default=DEFAULT_TOLERANCE)
    args = parser.parse_args()

    print(f"query {args.failure.name}, world={'on' if args.with_world else 'off'}, "
          f"budget={'production' if args.production else 'default'}, repeats={args.repeats}")

    outcomes = {}
    for device in ("cuda", "cpu"):
        began = time.time()
        outcomes[device] = _plan_on_device(args, device)
        elapsed = time.time() - began
        for index, run in enumerate(outcomes[device]):
            print(
                f"  {device:4s} replay {index}: success={run['success']} n={len(run['waypoints']):3d} "
                f"motion_time={run['motion_time']:.9f} ({run['seconds']:.1f}s) {run['status']}"
            )
        print(f"  {device:4s} total {elapsed:.1f}s")

    gpu_verdicts = {run["success"] for run in outcomes["cuda"]}
    cpu_verdicts = {run["success"] for run in outcomes["cpu"]}
    verdicts_agree = gpu_verdicts == cpu_verdicts and len(gpu_verdicts) == 1

    print()
    print(f"verdict: GPU {gpu_verdicts} vs CPU {cpu_verdicts} -> "
          f"{'AGREE' if verdicts_agree else 'DISAGREE'}")

    import numpy as np

    worst = None
    both_succeeded = [
        (g, c)
        for g in outcomes["cuda"]
        for c in outcomes["cpu"]
        if g["success"] and c["success"] and len(g["waypoints"]) == len(c["waypoints"])
    ]
    if both_succeeded:
        deviations = [
            float(np.abs(np.asarray(g["waypoints"]) - np.asarray(c["waypoints"])).max())
            for g, c in both_succeeded
        ]
        worst = max(deviations)
        span = float(
            np.abs(
                np.asarray(outcomes["cuda"][0]["waypoints"]).max(axis=0)
                - np.asarray(outcomes["cuda"][0]["waypoints"]).min(axis=0)
            ).max()
        )
        print(f"trajectory: worst |delta| over {len(deviations)} CPU/GPU pairs = {worst:.3e} rad "
              f"({100 * worst / span:.2f}% of the {span:.3f} rad motion)")
        print(f"            tolerance {args.tolerance:.3e} rad (the policy's waypoint_tolerance)")
    elif any(run["success"] for run in outcomes["cuda"] + outcomes["cpu"]):
        print("trajectory: not comparable -- waypoint counts differ between devices")

    passed = verdicts_agree and (worst is None or worst <= args.tolerance)
    print()
    print("GATE PASS: CPU and GPU agree." if passed else "GATE FAIL: see above.")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
