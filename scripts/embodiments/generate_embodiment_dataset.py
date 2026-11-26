#!/usr/bin/env python
"""One command: procedural embodiments in, an embodiment library and a wods dataset out.

Fuses steps 1-5 of docs/embodiment_generation.md and then drives each surviving robot through real
data generation on a randomly chosen task:

    seed ──▶ sample ──▶ build ──▶ emit ──▶ validate (T1 structural, T2 IK reach)
                                              │
                                    reject ◀──┴──▶ accept
                                                     │
                                     for attempt in 1..--task-attempts:
                                         pick a task the embodiment can actually run
                                         run the real datagen pipeline on it
                                         if an episode succeeded: convert to wods, stop
                                     else: record the embodiment as unplannable

Two outputs, both resumable:

    <out>/library/<bucket>/gen_<seed>/{model.xml, spec.yaml, params.json}   the embodiment library
    <out>/wods/<key>/...                                                   one wods item per episode
    <out>/library/ids.txt                                                  written by step 6

A task is chosen at RUNTIME from the configs that accept this embodiment, rather than from a
hardcoded list, because that set is a property of the robot: ``create_experiment_config`` rejects a
config whose camera system names a camera the embodiment lacks, and whose task the embodiment cannot
control. Discovering it keeps this script correct as generic configs are added, and honest about the
ones that do not exist yet -- ``--list-tasks`` prints what a given embodiment can actually be asked
to do.

Retrying matters: a single (task, house, episode) draw fails for reasons that have nothing to do with
the robot -- the task sampler finds no valid object in that house, the place receptacle is too small
for the object it drew. Declaring an embodiment unplannable on one attempt conflates that with a real
kinematic failure, so the default is several attempts across independently drawn tasks.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile
import traceback
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from _common import add_screen_flags, entry_dir, join, screens, stamp  # noqa: E402
from molmo_spaces.embodiment_gen.constants import NODE_SPEC_NAME  # noqa: E402

# ---------------------------------------------------------------------------------------------------
# Task-type registry.
#
# The point of this script is NOT to know about particular tasks. It knows about task TYPES -- and
# only what it cannot discover: whether a fixed-base arm could do such a task at all. Everything else
# (which config classes exist, which cameras they name, whether the embodiment can be controlled
# through them) is asked of the framework at runtime by discover_tasks.
#
# Adding a task type here does not make it available; a registered config with a generic camera system
# has to exist for it. --list-tasks reports what a given embodiment can actually be asked to do, and
# the gap between the two is the honest answer to "which tasks are supported".
TASK_TYPES: dict[str, dict] = {
    "pick":                   {"reachable": True},
    "pick_and_place":         {"reachable": True},
    "pick_and_place_color":   {"reachable": True},
    "pick_and_place_next_to": {"reachable": True},
    "open":                   {"reachable": True},
    "close":                  {"reachable": True},
    # Navigation needs an SE(2) base. validate_robot_task_compatibility rejects a fixed-base
    # embodiment outright, so no config of this type can ever accept one -- it is excluded by
    # construction, not by preference.
    "nav_to_obj":             {"reachable": False, "why": "requires an SE(2) mobile base"},
}

REACHABLE_TASK_TYPES = tuple(k for k, v in TASK_TYPES.items() if v["reachable"])

# Config-name substrings that register as configs but are not TASKS. Barrier variants insert a wall
# between the object and its destination specifically so a straight lift-and-carry fails -- they are
# A/B fixtures for comparing planners, so drawing one makes an embodiment look unplannable for a
# reason that has nothing to do with the embodiment. cuRobo variants need per-robot collision configs,
# which is fatal at scale.
NOT_TASKS = ("barrier", "curobo", "ablation", "debug")

# ---------------------------------------------------------------------------------------------------
# Scene-dataset registry.
#
# The scene corpus is ALL of MolmoSpaces' scene datasets (docs/assets.md), not one dataset and not a
# hand-picked list of houses. These are the ``get_scenes`` names of every non-debug dataset; which
# houses each one holds is asked of the manifest at runtime, so a dataset that grows does not need
# this script to change. Generation draws from the train split only -- val and test exist to hold
# out SCENES, and a training set that consumed them would poison every scene-level benchmark
# downstream.
SCENE_DATASETS: tuple[str, ...] = ("ithor", "procthor-10k", "procthor-objaverse",
                                   "holodeck-objaverse")
SCENE_SPLIT = "train"


def scene_corpus(datasets: tuple[str, ...]) -> list[tuple[str, list[int]]]:
    """``[(dataset, [house indices...]), ...]`` for every house with a scene file in the manifest.

    Index maps differ in shape by dataset -- ithor maps an index straight to a path, the procthor
    family maps it to a ``{variant: path}`` dict whose ``base`` entry is the scene -- and both pad
    missing indices with ``None``, which is why membership is checked here rather than assumed
    contiguous."""
    from molmo_spaces.molmo_spaces_constants import get_scenes

    corpus = []
    for name in datasets:
        houses = get_scenes(name, SCENE_SPLIT)[SCENE_SPLIT]
        usable = sorted(index for index, value in houses.items()
                        if (value.get("base") if isinstance(value, dict) else value) is not None)
        if usable:
            corpus.append((name, usable))
    return corpus


def discover_tasks(spec: Path, task_types: tuple[str, ...] = REACHABLE_TASK_TYPES,
                   allow_curobo: bool = False) -> list[str]:
    """Registered configs this embodiment can actually run, asked of the framework itself.

    Instantiating through ``create_experiment_config`` is the only reliable test: it runs the same
    robot/task and robot/camera validators the real entry point runs, so a config that survives here
    survives at generation time."""
    from molmo_spaces.data_generation import config_registry as reg
    from molmo_spaces.data_generation.main import auto_import_configs, create_experiment_config

    auto_import_configs()
    usable = []
    for name in sorted(reg.list_available_configs()):
        excluded = tuple(t for t in NOT_TASKS if not (allow_curobo and t == "curobo"))
        if any(token in name.lower() for token in excluded):
            continue
        try:
            config, _ = create_experiment_config(name, spec)
        except Exception:
            continue                       # incompatible camera system, or an uncontrollable task
        # "Validates" is NOT the bar -- the config must CARRY the generic camera system. A robot with
        # the robotiq gripper inherits gripper/wrist_camera from the shared 2F-85 asset, so Franka's
        # own configs happen to validate against it, and one run recorded a third of its episodes
        # (8/24) under Franka's 4-camera rig with end_on_success=False. An exact type check, not a
        # name convention; NOT_TASKS still applies on top, because the barrier/curobo variants
        # subclass the generic configs and carry the same camera system.
        from molmo_spaces.configs.camera_configs import GenericOmniPurposeCameraSystem

        if type(config.camera_config) is not GenericOmniPurposeCameraSystem:
            continue
        if config.task_type in task_types:
            usable.append(name)
    return usable


def task_type_of(spec: Path, config_name: str) -> str:
    """The task type a config declares, with the embodiment injected (some configs vary by robot)."""
    from molmo_spaces.data_generation.main import create_experiment_config

    config, _ = create_experiment_config(config_name, spec)
    return str(config.task_type)


def _covered_types(spec: Path, config_names: list[str]) -> set[str]:
    return {task_type_of(spec, name) for name in config_names}


def _surface_datagen_errors(text: str, seed: int, detail: str) -> None:
    """Surface the distinct error reasons behind a non-winning datagen run to stdout, so a 0/0 shows
    WHY in the fleet log rather than being silent. Logging only -- behaviour is unchanged."""
    import re
    seen: list[str] = []
    for reason in re.findall(r"(?:task sampling error|rollout error|HouseInvalidForTask): (.+)", text):
        reason = reason.strip()
        if reason and reason not in seen:
            seen.append(reason)
    if seen:
        print(f"[datagen seed={seed} {detail}] reasons:", flush=True)
        for reason in seen[:6]:
            print(f"  - {reason[:200]}", flush=True)


_FORKSERVER = None


def _forkserver_ctx():
    """A process-lifetime forkserver that has already imported the datagen stack, so each attempt forks
    from a warm interpreter instead of re-importing mujoco/cuRobo/configs (~26s) on every run."""
    global _FORKSERVER
    if _FORKSERVER is None:
        import multiprocessing

        ctx = multiprocessing.get_context("forkserver")
        ctx.set_forkserver_preload(["molmo_spaces.data_generation.main"])
        _FORKSERVER = ctx
    return _FORKSERVER


def _warn_while_running(proc, warn_every: int, seed: int) -> None:
    """Wait for the attempt, warning every warn_every seconds it is still running. Warn-only: a
    progressing attempt runs to its own episode/horizon limits and nothing is killed -- the fork is proven
    single-threaded (safe), the resource locks are handled, so there is no deadlock to abort. The warnings
    just make a genuinely-stuck attempt visible in the log for a human to investigate."""
    import time

    start = time.monotonic()
    warned = 0
    while proc.is_alive():
        proc.join(min(30, warn_every))
        if not proc.is_alive():
            break
        elapsed = time.monotonic() - start
        if elapsed >= warn_every * (warned + 1):
            warned += 1
            print(f"[datagen seed={seed}] attempt still running after {int(elapsed)}s", flush=True)


def run_datagen(config_name: str, spec: Path, output_dir: Path, seed: int, scene: str, house: int,
                episodes: int, timeout: int) -> tuple[bool, str]:
    """Run the real pipeline in a forkserver-dispatched process, and report whether an episode succeeded.

    A separate process on purpose: the pipeline seeds process-global RNG, spawns workers and installs a
    renderer, and a segfault in MuJoCo or a stuck worker must not take this sweep with it. Forking from a
    forkserver rather than re-exec'ing python lets each attempt inherit the datagen imports the server
    loaded once, so the isolation costs a fork, not a ~26s cold start.
    """
    import re

    argv = [
        config_name,
        "--embodiment-spec", str(spec),
        "--scene-dataset", scene,
        "--data-split", SCENE_SPLIT,
        "--house-ind", str(house),
        "--samples-per-house", str(episodes),
        "--seed", str(seed),
        "--output-dir", str(output_dir),
        # Always capture the sim-replay bundle: --replay-out is a required output, so every kept
        # episode must have one. The capture draws no RNG and mutates nothing, so trajectories/videos
        # are byte-identical with or without it; only successful episodes actually write a bundle.
        "--replay",
    ]
    log = output_dir / "datagen.log"
    log.parent.mkdir(parents=True, exist_ok=True)

    from molmo_spaces.data_generation.main import run_redirected

    proc = _forkserver_ctx().Process(target=run_redirected, args=(argv, str(log)),
                                     name=f"datagen_gen_{seed}")
    try:
        proc.start()
    except Exception as error:                        # server import failure, fork failure, ...
        return False, f"dispatch error: {str(error)[:200]}"
    _warn_while_running(proc, timeout, seed)  # warn-only: a progressing attempt runs to its own limits

    text = log.read_text(errors="ignore")
    hits = re.findall(r"Success count: (\d+), Total count: (\d+)", text)
    if not hits:
        _surface_datagen_errors(text, seed, "no result line")
        # child aborted before the summary; surface its exit code + log tail (crash/OOM/abort)
        tail = [ln for ln in text.splitlines() if ln.strip()][-8:]
        if tail:
            print(f"[datagen seed={seed} no result line] child log tail:", flush=True)
            for ln in tail:
                print(f"  | {ln[:200]}", flush=True)
        return False, f"no result line (child exit {proc.exitcode}): run did not reach the end"
    wins, total = int(hits[-1][0]), int(hits[-1][1])
    if total == 0:
        _surface_datagen_errors(text, seed, f"{wins}/{total}")
    return wins > 0, f"{wins}/{total} episodes"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seeds", type=int, nargs=2, metavar=("START", "STOP"), required=True)
    parser.add_argument("--library", required=True,
                        help="where the embodiment library goes; any fsspec target (local path, "
                             "gs://, s3://, ...)")
    parser.add_argument("--trajectories", required=True,
                        help="where the wods dataset goes; any fsspec target")
    parser.add_argument("--markers", required=True,
                        help="where per-seed progress markers go; any fsspec target. Its own "
                             "directory, not a corner of an output: it is coordination state, it "
                             "must be shared by every rank, and deleting it just means the next run "
                             "redoes the work")
    parser.add_argument("--replay-out", required=True,
                        help="where per-episode sim-replay bundles go; any fsspec target. One bundle "
                             "per saved episode (DR recipe + full per-step MuJoCo integration "
                             "state + camera poses + meta), sufficient to deterministically re-sim "
                             "and re-render it standalone. Keyed by seed like the library")
    parser.add_argument("--rank", type=int, default=int(os.environ.get("RANK", 0)))
    parser.add_argument("--world-size", type=int, default=int(os.environ.get("WORLD_SIZE", 1)))
    parser.add_argument("--task-attempts", type=int, default=4,
                        help="independently drawn tasks to try before calling an embodiment "
                             "unplannable (default 4)")
    parser.add_argument("--episodes-per-attempt", type=int, default=2)
    parser.add_argument("--scene-datasets", nargs="+", default=list(SCENE_DATASETS),
                        choices=list(SCENE_DATASETS),
                        help="scene datasets to draw houses from (default: all of them; "
                             "see docs/assets.md)")
    parser.add_argument("--tasks", nargs="+",
                        help="restrict to these config names instead of discovering them")
    parser.add_argument("--task-types", nargs="+", default=list(REACHABLE_TASK_TYPES),
                        choices=list(TASK_TYPES),
                        help="task types to draw from; unreachable ones are rejected at discovery")
    parser.add_argument("--allow-curobo", action="store_true")
    parser.add_argument("--datagen-timeout", type=int, default=3000,
                        help="warn-only watchdog interval in seconds: log a line every this many seconds an "
                             "attempt is still running. Nothing is killed -- attempts run to their own "
                             "episode/horizon limits; the warnings just surface a genuinely stuck attempt")
    parser.add_argument("--min-ik-reach", type=float, default=0.30)
    parser.add_argument("--tiers", nargs="*", default=["T1", "T2"], choices=["T1", "T2"])
    parser.add_argument("--flat", action="store_true")
    parser.add_argument("--force", action="store_true", help="ignore markers, redo every seed")
    parser.add_argument("--no-trajectories", action="store_true",
                        help="produce the library only, skip dataset conversion")
    parser.add_argument("--list-tasks", type=Path, metavar="SPEC",
                        help="print the tasks one existing entry can run, then exit")
    add_screen_flags(parser)
    args = parser.parse_args()

    if args.list_tasks:
        found = discover_tasks(args.list_tasks, tuple(args.task_types), args.allow_curobo)
        for name in found:
            print(name)
        covered = _covered_types(args.list_tasks, found)
        missing = [t for t in args.task_types if t not in covered and TASK_TYPES[t]["reachable"]]
        if missing:
            print(f"# no config with a generic camera system exists for: {', '.join(missing)}",
                  file=sys.stderr)
        for task, entry in TASK_TYPES.items():
            if not entry["reachable"]:
                print(f"# {task}: unreachable -- {entry['why']}", file=sys.stderr)
        return 0

    from molmo_spaces.embodiment_gen.builder import BuildError
    from molmo_spaces.embodiment_gen.sampling import sample_params
    from molmo_spaces.embodiment_gen.spec_emit import emit
    from molmo_spaces.embodiment_gen.validate import validate

    # Both outputs are fsspec targets held as STRINGS. Path is used only for the local temporary
    # workspace; it mangles a URL and cannot be joined with another str.
    dataset: str = str(args.trajectories)
    seeds = range(*args.seeds)[args.rank::args.world_size]
    tally: Counter = Counter()

    # Everything local is DISPOSABLE and lives in a temporary directory: staging for an entry before
    # it is published, and the tree the datagen subprocess writes. It has to be local because the
    # producers cannot speak fsspec -- MuJoCo compiles an MJCF from a path, h5py needs a seekable
    # file, and datagen writes a directory -- and it has to be disposable because nothing in it is
    # output. Coordination state does NOT live here; see markers below.
    with tempfile.TemporaryDirectory(prefix="embodiment_gen_") as workspace:
        staging, runs = Path(workspace) / "staging", Path(workspace) / "runs"
        _run(args, seeds, tally, staging, runs, dataset)

    summary = "  ".join(f"{k}={v}" for k, v in sorted(tally.items()))
    print(f"rank {args.rank}/{args.world_size}  seeds={len(seeds)}  {summary}", file=sys.stderr)
    print(f"library      {args.library}\ntrajectories {dataset}\nmarkers      {args.markers}\n"
          f"replay-out   {args.replay_out}", file=sys.stderr)
    return 0


class _NewlineBar:
    """Turn tqdm's '\\r' bar rewrites into '\\n' lines -- beaker only surfaces newline-terminated
    output, so an in-place '\\r' bar looks frozen once datagen output goes sparse."""

    def __init__(self, fp):
        self._fp = fp

    def write(self, s):
        return self._fp.write(s.replace("\r", "\n"))

    def __getattr__(self, name):
        return getattr(self._fp, name)


def _run(args, seeds, tally, staging: Path, runs: Path, dataset: str) -> None:
    from molmo_spaces.embodiment_gen.builder import BuildError
    from molmo_spaces.embodiment_gen.sampling import sample_params
    from molmo_spaces.embodiment_gen.spec_emit import emit
    from molmo_spaces.embodiment_gen.validate import validate

    corpus = scene_corpus(tuple(args.scene_datasets))
    print("scene corpus: " + "  ".join(f"{name}={len(houses)}" for name, houses in corpus),
          file=sys.stderr)

    from tqdm import tqdm

    # miniters=1 + _NewlineBar: one newline-terminated progress line per seed (beaker drops '\r' bars).
    bar_fp = _NewlineBar(sys.stderr)
    progress = tqdm(total=len(seeds), desc=f"rank {args.rank}/{args.world_size}", unit="seed",
                    file=bar_fp, dynamic_ncols=True, smoothing=0.05, miniters=1)
    # Seeds a previous lease already decided are walked, not worked, and a rank cannot finish its
    # slice inside one lease, so every resume re-walks a prefix that grows with the campaign. The
    # marker check therefore comes first, before the tally scan, the postfix and the bar refresh,
    # which is a line on unbuffered stderr. The bar takes a whole skipped run in one update once
    # real work resumes, so its count stays right.
    skipped_run = 0
    for seed in seeds:
        if not args.force and _marker_exists(args.markers, seed):
            tally["skipped"] += 1
            skipped_run += 1
            continue
        if skipped_run:
            progress.update(skipped_run)
            skipped_run = 0
        progress.update(1)
        rej = sum(v for k, v in tally.items()
                  if k.startswith(("rejected_", "error_")) or k == "no_task")
        progress.set_postfix_str(f"planned={tally['planned']} unplan={tally['unplannable']} "
                                 f"skip={tally['skipped']} rej={rej}", refresh=False)
        record = {"seed": seed, **stamp()}
        directory = Path(entry_dir(staging, seed, bucketed=not args.flat))

        # ---- steps 1-3: sample, build, emit --------------------------------------------------
        try:
            spec = emit(sample_params(seed), directory, **screens(args))
        except (BuildError, ValueError, RuntimeError) as error:
            tally["rejected_screen"] += 1
            _write_marker(args.markers, seed, record | {"accepted": False, "stage": "emit",
                                                   "reason": str(error)[:200]})
            continue
        except Exception:
            tally["error_emit"] += 1
            _write_marker(args.markers, seed, record | {"accepted": False, "stage": "emit",
                                                   "reason": traceback.format_exc()[-400:]})
            continue

        # ---- step 4: validate ---------------------------------------------------------------
        if args.tiers:
            verdict = validate(directory, min_ik_reach=args.min_ik_reach, tiers=tuple(args.tiers))
            record |= {"stage": verdict.tier, "reason": verdict.reason, **verdict.details}
            if not verdict:
                tally[f"rejected_{verdict.tier}"] += 1
                _remove(directory)
                _write_marker(args.markers, seed, record | {"accepted": False})
                continue

        # ---- task sampling and planning ------------------------------------------------------
        pool = args.tasks or discover_tasks(spec, tuple(args.task_types), args.allow_curobo)
        if not pool:
            tally["no_task"] += 1
            _write_marker(args.markers, seed, record | {"accepted": True, "planned": False,
                                                   "reason": "no task accepts this embodiment"})
            continue

        # one RNG per seed, so the tasks an embodiment is asked to do are a property of the
        # embodiment and reproduce on a rerun rather than depending on scheduling
        rng = random.Random(seed)
        attempts, planned = [], False
        for attempt in range(args.task_attempts):
            task = rng.choice(pool)
            # Dataset first, uniformly, THEN a house within it. The corpus is four datasets whose
            # sizes differ by three orders of magnitude (48 ithor floor plans next to ~100k houses
            # apiece for the objaverse pair, docs/assets.md), so a draw uniform over houses would be
            # >95% procthor/holodeck and the hand-crafted scenes would effectively never appear.
            # What the retries are meant to vary is the KIND of scene as much as the scene itself.
            scene, houses = rng.choice(corpus)
            house = rng.choice(houses)
            episode_seed = seed * 1000 + attempt
            # <embodiment>/<task>/<attempt>: the embodiment is the outer key because a consumer asks
            # "what can this robot do", and the task name is the natural second level. The attempt
            # index is last so retries of one task sit together rather than interleaving tasks.
            run_dir = runs / f"gen_{seed}" / task / f"attempt{attempt}"
            won, detail = run_datagen(task, spec, run_dir, episode_seed, scene, house,
                                      args.episodes_per_attempt, args.datagen_timeout)
            attempts.append({"task": task, "scene_dataset": scene, "house": house,
                             "seed": episode_seed, "result": detail, "success": won})
            tqdm.write(f"{seed}\t{task}\t{scene}/house_{house}\t{detail}\t{'WIN' if won else '-'}",
                       file=bar_fp)
            if won:
                planned = True
                break

        record |= {"accepted": True, "planned": planned, "attempts": attempts}
        tally["planned" if planned else "unplannable"] += 1

        # ---- wods conversion ----------------------------------------------------------------
        if planned and not args.no_trajectories:
            from embodiment_wods import convert_run                     # noqa: PLC0415
            try:
                # node_spec_path and entry are REQUIRED for the post-Stage-2 fields: without them
                # convert_run silently emits raw absolute actions and no node_proprio/node_geometry,
                # because the spec is an optional argument. That is exactly what this call used to do.
                items = convert_run(runs / f"gen_{seed}" / task, dataset, seed=seed,
                                    task_name=task, scene=f"{scene}-{SCENE_SPLIT}",
                                    successful_only=True,
                                    node_spec_path=directory / NODE_SPEC_NAME, entry=directory)
                record["wods_items"] = items
                tally["wods_items"] += len(items)
            except Exception:
                record["wods_error"] = traceback.format_exc()[-400:]
                tally["error_wods"] += 1
        # publish the sim-replay bundles the datagen wrote under runs/ before the tree is removed.
        # Keyed by seed like the library; only the winning attempt produced any (bundles are written
        # for saved episodes only, and earlier attempts had no success).
        if planned:
            n_replay = _publish_replays(runs / f"gen_{seed}" / task, seed, args)
            record["replay_bundles"] = n_replay
            tally["replay_bundles"] += n_replay

        # publish LAST, after conversion: the library and the dataset then agree, and a run that
        # crashed mid-conversion leaves no library entry claiming trajectories that do not exist
        _publish(directory, entry_dir(args.library, seed, bucketed=not args.flat))
        _remove(runs / f"gen_{seed}")
        _write_marker(args.markers, seed, record)

    # A rank whose remaining slice is all decided never reaches the update above, so close the bar
    # out here, or it would stop short of its total.
    if skipped_run:
        progress.update(skipped_run)
    progress.close()


def _marker_path(markers: str, seed: int) -> str:
    return join(markers, f"gen_{seed}.json")


def _marker_exists(markers: str, seed: int) -> bool:
    import fsspec

    filesystem, path = fsspec.core.url_to_fs(_marker_path(markers, seed))
    return filesystem.exists(path)


def _write_marker(markers: str, seed: int, record: dict) -> None:
    """Per-seed outcome, written to its OWN fsspec directory, after every durable effect.

    A separate directory from either output, because it is neither: it is coordination state that
    every rank must see, and mixing it into the library would make a re-run's bookkeeping part of the
    artefact being published. Existence means the seed has been DECIDED, not that it succeeded --
    rejections are recorded too, and their ``reason`` is the yield data, the only record of why 999 of
    every 1000 seeds are gone.

    Written last, after the trajectories and the published entry, so a crash before it leaves no
    marker and the seed is simply redone. Redoing is safe: wods rewrites the same key with the same
    bytes and _publish overwrites."""
    import fsspec

    filesystem, path = fsspec.core.url_to_fs(_marker_path(markers, seed))
    filesystem.makedirs(path.rsplit("/", 1)[0], exist_ok=True)
    with filesystem.open(path, "w") as handle:
        handle.write(json.dumps(record))


def _publish(local: Path, target: str) -> None:
    """Copy a staged entry from the local workspace to its fsspec destination, file by file.

    ``local`` is a Path because it is a real directory; ``target`` is a string because it may be a
    bucket. Staging first is not a convenience: MuJoCo compiles an MJCF from a filesystem path and
    h5py needs a seekable file, so the producers cannot write to a remote store at all."""
    import fsspec

    if not local.is_dir():
        return
    filesystem, root = fsspec.core.url_to_fs(target)
    for path in sorted(local.rglob("*")):
        if not path.is_file():
            continue
        destination = join(root, path.relative_to(local).as_posix())
        filesystem.makedirs(destination.rsplit("/", 1)[0], exist_ok=True)
        with path.open("rb") as source, filesystem.open(destination, "wb") as sink:
            sink.write(source.read())


def _publish_replays(task_run_dir: Path, seed: int, args) -> int:
    """Publish every sim-replay bundle the datagen wrote under this task's run tree to --replay-out.

    The pipeline writes one bundle per SAVED episode at ``<output_dir>/**/replay/seed_<n>/`` (a
    directory holding meta.json + dr_deltas.npz + dr_texture_paths.json + states.npz + cameras.json).
    Only successful episodes
    produce one, so walk whatever exists -- same rglob idiom as the wods conversion -- and copy each to
    ``<replay-out>/<bucket>/gen_<seed>/seed_<n>/``, keyed by seed like the library. Independent of the
    wods conversion, so replay data is captured even under --no-trajectories. Must run before the run
    tree is removed."""
    root = entry_dir(args.replay_out, seed, bucketed=not args.flat)
    count = 0
    for meta in sorted(task_run_dir.rglob("replay/*/meta.json")):
        _publish(meta.parent, join(root, meta.parent.name))
        count += 1
    return count


def _remove(path: Path) -> None:
    """Delete a directory tree. A rejected entry must not stay in the library, or step 6 indexes it."""
    if not path.exists():
        return
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_file():
            child.unlink()
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_dir() and not any(child.iterdir()):
            child.rmdir()
    if not any(path.iterdir()):
        path.rmdir()


if __name__ == "__main__":
    raise SystemExit(main())
