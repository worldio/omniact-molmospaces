#!/usr/bin/env python
"""Step 6 -- stamp the library index.

A library is enumerated through ``ids.txt``, never globbed: globbing and parsing a million
``spec.yaml`` costs tens of minutes per process and every rank repeats it on every resume.

Run ONCE, after every rank of step 5 has finished. Readers short-circuit to the frozen key list, so
an index stamped mid-run permanently hides everything generated after it -- which is why this is a
separate command rather than something the fleet does when it thinks it is done.

Paths are recorded relative to the library root, not bare ids, so the producer chooses the layout
(flat for a small library, bucketed for a million) and no reader has to know which.

    embodiment-index library/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from molmo_spaces.embodiment_gen.constants import SPEC_NAME


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("library", type=Path, help="a finished library root")
    parser.add_argument("--name", default="ids.txt", help="index filename (default ids.txt)")
    parser.add_argument("--force", action="store_true", help="overwrite an existing index")
    args = parser.parse_args()

    index = args.library / args.name
    if index.exists() and not args.force:
        print(f"{index} exists; pass --force to replace it", file=sys.stderr)
        return 1

    def seed_of(path: Path) -> int:
        name = path.parent.name
        return int(name.removeprefix("gen_")) if name.startswith("gen_") else -1

    specs = sorted(args.library.rglob(SPEC_NAME), key=seed_of)
    if not specs:
        print(f"no {SPEC_NAME} under {args.library}", file=sys.stderr)
        return 1

    lines = [str(p.parent.relative_to(args.library)) for p in specs]
    index.write_text("\n".join(lines) + "\n")
    print(f"{index}\t{len(lines)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
