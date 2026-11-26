#!/usr/bin/env python
"""Step 2 -- realize sampled parameters as one compiled MuJoCo model.

Builds the arm, grafts the gripper, compiles. A compile failure is a REJECTION, not a crash: the
command reports it and exits non-zero for that seed rather than aborting a sweep.

Mesh references point into the installed asset tree, so every consumer of the written model must
resolve assets at the same path.

    embodiment-build --seed 43 --out model.xml
    embodiment-build --seeds 0 50 --out models/ --quiet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_screen_flags, screens                                   # noqa: E402
from molmo_spaces.embodiment_gen.builder import BuildError, build                # noqa: E402
from molmo_spaces.embodiment_gen.constants import MJCF_NAME                      # noqa: E402
from molmo_spaces.embodiment_gen.sampling import sample_params                   # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--seeds", type=int, nargs=2, metavar=("START", "STOP"))
    parser.add_argument("--out", type=Path, required=True,
                        help=f"a file for --seed, or a directory (one {MJCF_NAME} per seed) "
                             f"for --seeds")
    parser.add_argument("--wrist-camera", action="store_true",
                        help="declare the canonical wrist_camera the generic camera system names; "
                             "step 3 always does this, so pass it here to match what ships")
    parser.add_argument("--quiet", action="store_true", help="only report failures")
    add_screen_flags(parser)
    args = parser.parse_args()

    seeds = [args.seed] if args.seed is not None else list(range(*args.seeds))
    built = 0
    for seed in seeds:
        try:
            xml, model = build(sample_params(seed), wrist_camera=args.wrist_camera,
                               **screens(args))
        except (BuildError, ValueError, RuntimeError) as error:
            print(f"{seed}\tREJECTED\t{error}", file=sys.stderr)
            continue
        path = args.out if args.seed is not None else args.out / f"gen_{seed}" / MJCF_NAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(xml)
        built += 1
        if not args.quiet:
            print(f"{seed}\t{path}\tnbody={model.nbody} nu={model.nu} ngeom={model.ngeom} "
                  f"ncam={model.ncam}")
    if not args.quiet and len(seeds) > 1:
        print(f"built {built}/{len(seeds)}", file=sys.stderr)
    return 0 if built else 1


if __name__ == "__main__":
    raise SystemExit(main())
