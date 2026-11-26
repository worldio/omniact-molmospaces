#!/usr/bin/env python
"""Step 1 -- turn a seed into the frozen parameters of one robot.

Pure function of ``(seed, GENERATOR_VERSION)``: no MuJoCo, no assets, no I/O beyond the file this
writes. Useful on its own for ablations -- diff two seeds' parameters, or check what a seed asks for
before paying to build it.

    embodiment-sample --seed 43
    embodiment-sample --seeds 0 10 --out params/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import params_json, write_json                     # noqa: E402
from molmo_spaces.embodiment_gen.sampling import sample_params   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int, help="a single seed")
    group.add_argument("--seeds", type=int, nargs=2, metavar=("START", "STOP"),
                       help="a half-open seed range")
    parser.add_argument("--out", type=Path,
                        help="directory to write <seed>.json into; omit to print a summary")
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(*args.seeds))
    for seed in seeds:
        params = sample_params(seed)
        if args.out:
            path = args.out / f"gen_{seed}.json"
            write_json(path, params_json(params))
            print(f"{seed}\t{path}")
        else:
            print(f"seed {seed}\t{params.gripper}\t{len(params.arm.joints)}-dof"
                  f"\tgrammar={params.arm.grammar}"
                  f"\tpedestal={params.arm.pedestal_height:.3f}m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
