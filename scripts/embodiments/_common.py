"""Shared plumbing for the embodiment-generation commands.

Every step is its own command so a single seed can be inspected at the stage it went wrong, rather
than only through the fleet that runs all of them. The flags below are therefore identical across
commands: the same ``--max-self-collision`` means the same thing in step 2, 4 and 5, so a seed
accepted by one is accepted by all.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from molmo_spaces.embodiment_gen.builder import (
    GRASP_FEASIBILITY_LIMIT,
    SELF_COLLISION_LIMIT,
)
from molmo_spaces.embodiment_gen.constants import BUCKET_PAD, BUCKET_SIZE, GENERATOR_VERSION


def add_screen_flags(parser: argparse.ArgumentParser) -> None:
    """The acceptance screens, spelled the same way everywhere. ``none`` disables one."""
    parser.add_argument(
        "--max-self-collision", type=float, default=SELF_COLLISION_LIMIT,
        help=f"reject a robot whose links overlap in more than this fraction of sampled poses "
             f"(default {SELF_COLLISION_LIMIT}; pass a value above 1 to disable)")
    parser.add_argument(
        "--min-grasp-feasibility", type=float, default=GRASP_FEASIBILITY_LIMIT,
        help=f"reject a robot that can complete a whole pick at fewer than this fraction of "
             f"workspace targets (default {GRASP_FEASIBILITY_LIMIT}; pass 0 to disable)")


def screens(args: argparse.Namespace) -> dict:
    """``build``/``emit`` keyword arguments for the screen flags, with disabling folded in."""
    return {
        "max_self_collision": None if args.max_self_collision > 1.0 else args.max_self_collision,
        "min_grasp_feasibility": None if args.min_grasp_feasibility <= 0.0
        else args.min_grasp_feasibility,
    }


def add_seed_range(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seeds", type=int, nargs=2, metavar=("START", "STOP"), required=True,
                        help="half-open seed range, as in range(START, STOP)")


def join(target, *parts: str) -> str:
    """Join an fsspec target with path parts: ``parent + "/" + sub``.

    Never ``pathlib``. ``Path`` is for the local filesystem and mangles a URL --
    ``Path("gs://b/x") / "y"`` collapses the double slash to ``gs:/b/x/y`` -- and joining two ``str``
    with ``/`` raises outright, which is how this was first noticed."""
    return "/".join([str(target).rstrip("/"), *(str(p).strip("/") for p in parts)])


def entry_dir(library, seed: int, bucketed: bool = True) -> str:
    """Where entry ``seed`` lives, as a string path.

    Bucketed by ``seed // BUCKET_SIZE`` so a million-entry library has listable directories; flat when
    it is small enough not to need it."""
    parts = [f"{seed // BUCKET_SIZE:0{BUCKET_PAD}d}"] if bucketed else []
    return join(library, *parts, f"gen_{seed}")


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=list))


def params_json(params) -> dict:
    return asdict(params)


def stamp() -> dict:
    """Provenance every command records, so an artefact says which generator made it."""
    return {"generator_version": GENERATOR_VERSION}
