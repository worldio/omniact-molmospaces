#!/usr/bin/env python
"""Step 5 -- the generation fleet: sample, build, emit, validate, export, one seed at a time.

Data-parallel by striding: run the identical command in WORLD_SIZE processes with RANK set, and each
takes ``seeds[RANK::WORLD_SIZE]``. No GPU, no inter-process coordination.

Resumable and idempotent. A per-seed marker is written only AFTER the entry is durable, so a crash
writes no marker and re-running the identical command redoes exactly the work that was lost. Nothing
is mutated in place: a regenerated library is a new --out.

Rejections are expected, not errors. Acceptance is roughly 1 seed in 1,000, so ask for far more
seeds than the entries you want, and read the rejection tally rather than assuming a range yields
entries.

    embodiment-generate --seeds 0 100000 --out library/
    RANK=$R WORLD_SIZE=128 embodiment-generate --seeds 0 1000000 --out library/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_screen_flags, entry_dir, screens, stamp                  # noqa: E402
from molmo_spaces.embodiment_gen.builder import BuildError                        # noqa: E402
from molmo_spaces.embodiment_gen.sampling import sample_params                    # noqa: E402
from molmo_spaces.embodiment_gen.spec_emit import emit                            # noqa: E402
from molmo_spaces.embodiment_gen.validate import validate                         # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs=2, metavar=("START", "STOP"), required=True)
    parser.add_argument("--out", type=Path, required=True, help="library root")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--markers", type=Path,
                        help="where per-seed markers live (default <out>/.markers); put this on "
                             "scratch if the library itself must stay clean")
    parser.add_argument("--tiers", nargs="*", default=["T1", "T2"], choices=["T1", "T2"],
                        help="validation tiers to run; pass none to skip step 4 entirely")
    parser.add_argument("--min-ik-reach", type=float, default=0.30)
    parser.add_argument("--flat", action="store_true",
                        help="write <out>/gen_<seed>/ without the bucket level")
    parser.add_argument("--force", action="store_true", help="ignore markers and redo every seed")
    add_screen_flags(parser)
    args = parser.parse_args()

    markers = args.markers or args.out / ".markers"
    markers.mkdir(parents=True, exist_ok=True)
    seeds = list(range(*args.seeds))[args.rank::args.world_size]
    tally: Counter = Counter()

    for seed in seeds:
        marker = markers / f"gen_{seed}.json"
        if marker.exists() and not args.force:
            tally["skipped"] += 1
            continue
        directory = Path(entry_dir(args.out, seed, bucketed=not args.flat))
        record = {"seed": seed, **stamp()}
        try:
            emit(sample_params(seed), directory, **screens(args))
        except (BuildError, ValueError, RuntimeError) as error:
            tally["rejected_build"] += 1
            record |= {"accepted": False, "stage": "emit", "reason": str(error)[:200]}
            marker.write_text(json.dumps(record))
            continue
        except Exception:                                    # unexpected: record and keep going
            tally["error"] += 1
            record |= {"accepted": False, "stage": "emit", "reason": traceback.format_exc()[-400:]}
            marker.write_text(json.dumps(record))
            continue

        if args.tiers:
            verdict = validate(directory, min_ik_reach=args.min_ik_reach, tiers=tuple(args.tiers))
            record |= {"stage": verdict.tier, "reason": verdict.reason, **verdict.details}
            if not verdict:
                tally[f"rejected_{verdict.tier}"] += 1
                record["accepted"] = False
                # the entry is not usable, so it must not remain in the library
                for path in sorted(directory.rglob("*"), reverse=True):
                    path.unlink() if path.is_file() else path.rmdir()
                directory.rmdir()
                marker.write_text(json.dumps(record))
                continue

        tally["accepted"] += 1
        record["accepted"] = True
        # marker LAST, and only now: it is the durability record, so a crash before this point
        # leaves no marker and the seed is redone verbatim on resume
        marker.write_text(json.dumps(record))
        print(f"{seed}\t{directory}", flush=True)

    summary = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
    print(f"rank {args.rank}/{args.world_size}  seeds={len(seeds)}  {summary}", file=sys.stderr)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
