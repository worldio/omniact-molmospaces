#!/usr/bin/env python
"""Step 3 -- emit the EmbodimentSpec that makes a model mean something.

Writes a complete entry: ``model.xml`` (the robot), ``spec.yaml`` (move groups, controllers,
kinematics, sensors, grasp profiles -- the contract a planner drives it through) and ``params.json``
(what produced both, so an entry can be regenerated or diffed without re-running the sampler).

The entry is location-independent: it declares ``root_path: .`` and a relative ``mjcf_path``, which
the loader resolves against the spec file's own directory. Copy, move or mount an entry elsewhere and
it still loads.

    embodiment-emit --seed 43 --out embodiments/gen_43
    embodiment-emit --seeds 0 1000 --library embodiments/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_screen_flags, entry_dir, screens                          # noqa: E402
from molmo_spaces.embodiment_gen.builder import BuildError                         # noqa: E402
from molmo_spaces.embodiment_gen.sampling import sample_params                     # noqa: E402
from molmo_spaces.embodiment_gen.spec_emit import emit                             # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--seed", type=int)
    group.add_argument("--seeds", type=int, nargs=2, metavar=("START", "STOP"))
    place = parser.add_mutually_exclusive_group(required=True)
    place.add_argument("--out", type=Path, help="the entry directory (single --seed only)")
    place.add_argument("--library", type=Path,
                       help="a library root; entries land at <library>/<bucket>/gen_<seed>/")
    parser.add_argument("--flat", action="store_true",
                        help="with --library, write <library>/gen_<seed>/ without the bucket level")
    parser.add_argument("--quiet", action="store_true", help="only report failures")
    add_screen_flags(parser)
    args = parser.parse_args()

    if args.out and args.seed is None:
        parser.error("--out takes a single --seed; use --library for a range")

    seeds = [args.seed] if args.seed is not None else list(range(*args.seeds))
    emitted = 0
    for seed in seeds:
        directory = args.out or Path(entry_dir(args.library, seed, bucketed=not args.flat))
        try:
            path = emit(sample_params(seed), directory, **screens(args))
        except (BuildError, ValueError, RuntimeError) as error:
            print(f"{seed}\tREJECTED\t{error}", file=sys.stderr)
            continue
        emitted += 1
        if not args.quiet:
            print(f"{seed}\t{path}")
    if not args.quiet and len(seeds) > 1:
        print(f"emitted {emitted}/{len(seeds)}", file=sys.stderr)
    return 0 if emitted else 1


if __name__ == "__main__":
    raise SystemExit(main())
