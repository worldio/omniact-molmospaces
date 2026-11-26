#!/usr/bin/env python
"""Step 4 -- validate emitted entries, cheapest tier first.

    T1  structural    the spec loads, the model compiles, the names resolve
    T2  reachability  MolmoSpaces' OWN IK solves a workspace pose bank to the gripper's leaf frame

T2 deliberately uses the same solver the planner uses at plan time. A screen built on any other
solver measures a different robot: the geometric screen it replaces correlated with real planner
success at rho = +0.05 across 36 accepted embodiments, so it gated usefully but ranked nothing.

A failing entry is reported, never raised -- generation is a filter.

    embodiment-validate embodiments/gen_43
    embodiment-validate --library embodiments/ --tiers T1        # structural sweep, milliseconds
    embodiment-validate --library embodiments/ --json verdicts.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from molmo_spaces.embodiment_gen.constants import SPEC_NAME                      # noqa: E402
from molmo_spaces.embodiment_gen.validate import validate                        # noqa: E402


def entries(args: argparse.Namespace) -> list[Path]:
    if args.entry:
        return [Path(e) for e in args.entry]
    return sorted(p.parent for p in Path(args.library).rglob(SPEC_NAME))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("entry", nargs="*", help="entry directories to validate")
    parser.add_argument("--library", type=Path,
                        help=f"validate every directory under this root holding a {SPEC_NAME}")
    parser.add_argument("--tiers", nargs="+", default=["T1", "T2"], choices=["T1", "T2"],
                        help="which tiers to run, in order (default: both)")
    parser.add_argument("--min-ik-reach", type=float, default=0.30,
                        help="T2 threshold: fraction of the pose bank that must solve "
                             "(default 0.30)")
    parser.add_argument("--control-profile",
                        help="control profile to assemble with; default is the spec's own")
    parser.add_argument("--json", type=Path, help="also write one JSON verdict per line here")
    args = parser.parse_args()

    if not args.entry and not args.library:
        parser.error("give entry directories or --library")

    targets = entries(args)
    if not targets:
        print("no entries found", file=sys.stderr)
        return 1

    passed, handle = 0, args.json.open("w") if args.json else None
    try:
        for entry in targets:
            verdict = validate(entry, control_profile=args.control_profile,
                               min_ik_reach=args.min_ik_reach, tiers=tuple(args.tiers))
            passed += bool(verdict)
            print(verdict.line(), flush=True)
            if handle:
                handle.write(json.dumps({"entry": str(entry), "tier": verdict.tier,
                                         "passed": verdict.passed, "reason": verdict.reason,
                                         **verdict.details}) + "\n")
    finally:
        if handle:
            handle.close()
    print(f"passed {passed}/{len(targets)}", file=sys.stderr)
    return 0 if passed == len(targets) else 1


if __name__ == "__main__":
    raise SystemExit(main())
