# Procedural Embodiment Generation — Step-by-Step Pipeline

How a library of procedurally generated robots is built for MolmoSpaces, from RNG seeds to an
indexed library that `load_embodiment_spec` can open like any shipped robot. Every step is a
self-contained, resumable command; nothing is mutated in place.

```
 shipped robot assets ──▶ [0] make-gripper-catalog ──▶ gripper_catalog/*.yaml  (committed, 3 entries)
                                                              │
       seed ──▶ [1] sample-embodiment ──▶ params.json         │
                                            │                 │
                                            ▼                 ▼
                              [2] build-embodiment ───────────┘ ──▶ model.xml (arm + grafted gripper)
                                            │
                                            ▼
                              [3] emit-embodiment ──▶ spec.yaml + model.xml + params.json
                                            │
                                            ▼
                              [4] validate-embodiment ──▶ verdict (T1 structural ▸ T2 IK reach)
                                            │
                                            ▼
                              [5] generate-embodiments ──▶ <library>/<bucket>/gen_<seed>/
                                            │
                                            ▼
                              [6] write-embodiment-index ──▶ <library>/ids.txt
```

| step | command | status |
|---|---|---|
| **0** gripper catalog | `make_gripper_catalog.py` | implemented; `--check` reproduces the committed entries |
| **1** parameter sampling | `sample_embodiment.py` | implemented |
| **2** arm build and graft | `build_embodiment.py` | implemented |
| **3** spec emission | `emit_embodiment.py` | implemented; specs are self-contained, the gripper-fragment split described in step 3 is **not yet done** |
| **4** validation | `validate_embodiment.py` | T1 and T2 implemented; **T3 not yet** — run it by hand via `data_generation/main.py` |
| **5** generation fleet | `generate_embodiments.py` | implemented, resumable |
| **6** library index | `write_embodiment_index.py` | implemented |

Every step is its own command, so a single seed can be inspected at the stage it went wrong rather
than only through the fleet that runs all of them. Steps 1–4 are also importable library functions.

Embodiment identity is a pure function of `(seed, GENERATOR_VERSION)`. A dataset row therefore
traces back to the robot that produced it without shipping the robot alongside it — and a
`GENERATOR_VERSION` bump is the only thing that may change what a seed means.

---

## Conventions (read once)

**Naming.** An embodiment id is `gen_<seed>`. Everything the gripper contributes to a model lives
under the prefix `gripper/` — the convention MolmoSpaces' own composed robots already use
(`franka_droid` and `franka_cap` ship `gripper/base`, `gripper/left_driver_joint`,
`gripper/grasp_site`). The arm contributes `base_link`, `flange`, `joint<i>`, `link<i>`.

**Composition is `MjsFrame.attach_body`**, the same mechanism `molmo_spaces.robots.franka` uses to
put a robot into a scene. It carries actuators, equality constraints, tendons, sites and mesh
assets across specs. A gripper grafted onto an arm is **bit-identical** to the same gripper
compiled standalone (verified: `max |Δqpos| = 0.0`). Four things bite, all handled in
`embodiment_gen/grippers.py`:

- `attach_body` **consumes** the child spec — every graft needs a fresh `copy()`.
- Source **keyframes must be deleted**; they go stale the moment the root joint is dropped
  (*"Keyframe 'home' has invalid qpos size, got 8, should be 7"*).
- The graft root's **own joint must be dropped** — the floating grippers are rooted on a free joint
  (*"free joint can only be used on top level"*), and yam's gripper is rooted on the arm's last
  driven link.
- **Unreferenced meshes ride along** and must be pruned, keyed on `name or Path(file).stem`:
  meshes declared by `file` alone have an **empty `name`** until compile time, so keying on `name`
  silently deletes the entire asset block. rby1's hand drags in 1112 meshes (204 KB) unpruned.

**Why grippers are grafted, never synthesized.** The gripper's 1-DoF coupling, its actuator and
its tuned gains live *inside* the shipped asset. Grafting inherits them, so nothing in the
generator re-derives gripper semantics — and because MolmoSpaces' grasp libraries are selected by
*object* library (`thor -> droid`, `objaverse -> droid_objaverse`) with each robot supplying its
own `grasp_profiles` adapter, a grafted gripper inherits working grasp annotations for free.

**How to run each kind of job.** Steps 0 and 6 run **once, single process**. Steps 1–4 take either
`--seed` (one robot) or `--seeds START STOP` (a range), and are what you reach for when investigating
a specific seed. Step 5 is the production path and is **data-parallel**: set `RANK`/`WORLD_SIZE` and
run the identical command in `WORLD_SIZE` processes, which stride the range as
`seeds[RANK::WORLD_SIZE]`. It is resumable and idempotent — a per-seed marker is written **only
after** the entry is durable, so a crash writes no marker and re-running redoes exactly the lost work.

**Acceptance is roughly 1 seed in 1,000.** Ask for far more seeds than the entries you want, and read
the rejection tally each command prints rather than assuming a range yields entries.

**GPU.** Nothing in this pipeline needs one. Tiers 1–2 of validation are mesh-free
(`molmo_spaces.kinematics.mujoco_kinematics` and the warp batch path both load with
`strip_meshes=True`), and tier 3 runs headless physics. Only rendered datagen downstream needs a
graphics stack.

**Environment.** Point both of these at durable storage before anything touches assets:

```bash
export MLSPACES_ASSETS_DIR=/path/to/durable/assets
export MLSPACES_CACHE_DIR=/path/to/durable/download-cache
```

`MLSPACES_ASSETS_DIR` replaces the default `~/.cache/molmospaces/assets/<b64 of install path>`;
`MLSPACES_CACHE_DIR` replaces the download cache. Two reasons this matters beyond disk space. On a
container or preemptible node the home directory is often **ephemeral**, so assets are re-downloaded
every run and a virtualenv whose interpreter lives there stops working after the node changes. And
the default download cache has shipped a **stale manifest** that fails `get_robot_path` for 3 of 4
robots (*"Directory path exists on disk but is not recorded in the cache manifest"*); a fresh
`MLSPACES_CACHE_DIR` fixes it. Assets download lazily on first use — there is no separate fetch step.

Choose the paths once and use the same pair everywhere: mesh references inside an emitted `model.xml`
point into the asset tree, so **every rank and every downstream consumer must resolve assets at the
same path**.

**Index + immutability.** A library is enumerated through its `ids.txt` index, **never globbed** —
globbing and parsing a million `spec.yaml` costs ~30 min per process and every rank repeats it on
every resume. Stamp the index **once, after every rank has finished**; publishing it early freezes
an incomplete library. Entries are append-only: a regenerated library is a new `--out`, never an
in-place edit.

---

## The fused pipeline (`generate_embodiment_dataset.py`) --- the one command to run

Steps 1–5 plus task-driven data generation and wods conversion, as one resumable command. This is
the production path; the per-step CLIs above remain for investigating a single seed.

```bash
RANK=$R WORLD_SIZE=64 python scripts/embodiments/generate_embodiment_dataset.py \
    --seeds 0 200000 \
    --library      <fsspec target>   # entries + published only after their trajectories exist
    --trajectories <fsspec target>   # wods items, keyed gen_<seed>/<task>/<scene>_house_<N>_traj<NN>
    --markers      <fsspec target>   # one json per DECIDED seed; delete to force a redo
    --replay-out   <fsspec target>   # one sim-replay bundle per saved episode, keyed by seed
```

- All four outputs are fsspec targets (local path, `gs://`, `s3://`). Local file I/O happens in a
  `TemporaryDirectory` and is published through fsspec afterwards — MuJoCo compiles from a path and
  h5py needs a seekable file, so the producers cannot write to a bucket directly.
- **Scenes are drawn from the whole MolmoSpaces corpus** — every non-debug scene dataset in
  [assets.md](assets.md) (`ithor`, `procthor-10k`, `procthor-objaverse`, `holodeck-objaverse`),
  train split only so the val/test scenes stay held out. Each attempt picks a dataset uniformly and
  then a house uniformly within it; dataset-first because the corpus sizes span three orders of
  magnitude (48 usable ithor floor plans next to ~100k houses apiece for the objaverse pair), so a
  draw uniform over houses would almost never produce a hand-crafted scene. `--scene-datasets`
  restricts the pool when debugging one dataset; the underlying per-run override is
  `main.py --scene-dataset <name> --data-split <split>`. Houses download on demand on first touch —
  for a fleet-sized run, prefetch with `scripts/assets/hf_download.py` (see
  [assets.md](assets.md#bulk-download)) rather than paying per-house HTTP latency mid-run.
- The **task pool** is discovered per robot but restricted to configs that *carry*
  `GenericOmniPurposeCameraSystem` — validation alone is not the bar, because a robot with the
  robotiq gripper inherits `gripper/wrist_camera` from the shared 2F-85 and Franka's own configs
  happen to validate against it.
- Markers are written **last**, after the wods items and the published entry, so a crash costs one
  in-flight seed and a re-run redoes exactly that. Rejections are recorded with their reason; the
  markers are the yield data.
- Each wods item carries `video_raw` per camera, `timestamps`, `intrinsics`, `c2w` (wods camera
  convention), raw `proprio`, per-node relative `action`, mid-range-centred `node_proprio`,
  `node_graph`, and **exact** `node_geometry` + `node_action_geometry` — FK of the same qpos row
  (and its action target) that became the proprio in the same shard, on the assembled robot — plus
  the **TCP node** (after the drivers, before the terminals — breadth-first, a sideless child of the wrist node; the gripper's `tcp_frame` site, the catalog's second `add_site` — closed-pad
  midpoint, GL-style: x toward the right pad, y = wrist-image-up, z back into the wrist; `joint_type` 3,
  `axis_mask` False, parent = the last arm node, anchor axis = the approach, −z) and two `(T, N, 3, 4)`
  per-node SE(3) tracks `[R | t]` over WODS-style TCP axes (x right, y up, z forward into the fingertips):
  `node_pose` at the recorded state, WODS camera ← TCP (det +1; observation, pairs with `node_geometry`), and
  `node_action_pose` at the action-target configuration, **house-world** ← TCP (det −1 like `c2w`; label,
  pairs with `node_action_geometry`; row `i` aligns with `action[i]`; the same track in every shard; `c2w⁻¹`
  takes it into any view). Only the TCP row (joint type 3) carries
  a frame; every other row is all zero, a body/geom frame being the asset's own convention rather than a
  quantity to transmit. A dataset converted before the TCP node existed gets it from the post-run fix chain
  (branch `embgen-postrun-fix`, Fixes 5 + 6: the node inserted into every per-node field and both tracks
  recomputed exactly from each trajectory's replay bundle, into a new prefix; the shipped data is read-only).
  Replay bundles keep their `spec.yaml` and `model.xml`: `Kinematics` injects the catalog site into its own
  FK model when a spec has none, and exact replay hashes the model names.
- Each saved episode also gets a **sim-replay bundle** under `--replay-out` (keyed by seed): a compact
  domain-randomization recipe plus the full per-step MuJoCo integration state and camera poses. It
  re-simulates and re-renders that episode byte-identically on a fresh compile via
  `scripts/embodiments/replay_episode.py`, independently of the wods conversion (so bundles are
  captured even under `--no-trajectories`).

After it finishes, you still need to write the wods index and the embodiment library index.

Below are the old step by step commands for reference and debugging:

---

## Step 0 — Gripper catalog (`make_gripper_catalog`)

Make MolmoSpaces' shipped grippers graftable. Every field of an entry is **derived**, never
hand-authored, so refreshing after an upstream asset or spec bump is one command.

```bash
python scripts/embodiments/make_gripper_catalog.py
python scripts/embodiments/make_gripper_catalog.py --check    # CI: committed entries still reproduce?
```

- **Input**: the shipped robot MJCF (`get_robot_path(<resource_key>)/<mjcf_path>`) and its
  `docs/embodiment_specs/<robot>.yaml`.
- **Output**: `molmo_spaces/embodiment_gen/gripper_catalog/<name>.yaml`, one per gripper,
  **committed to the repo** (~120–165 lines each). No MJCF is copied or checked in: the graft reads
  the installed asset, so mesh paths resolve against the source file's own directory.
- **Run on**: one process, once. Requires the source assets fetched.
- **Derivation** — nothing is measured by hand:

  | field | derived from |
  |---|---|
  | `graft_root` | the gripper move group's `root_body` |
  | `drop_joints` | that body's **own** joints |
  | `keep_meshes` | the meshes a trial graft's geoms actually reference |
  | `move_group`, `controller`, `grasp_profile`, `sensors`, `model_operations` | lifted verbatim from the source spec, every name rewritten under `gripper/` |

- **Gate**: the pruned graft must compile and preserve the unpruned graft's actuator and equality
  counts, else the entry is rejected.
- **Which sensors and operations belong to the gripper**: an operation is tested on its
  **`parent_body`**, not on all its refs — an `add_site` names a site that does not exist yet, so a
  subset test would reject exactly the ops that matter. What those ops create is then added to the
  gripper-side name set, after which a sensor qualifies iff *every* name it references is
  gripper-side. That is what carries yam's `planner_grasp_site` through to its leaf frame, its grasp
  profile and its `tcp_pose` sensor.
- **Result** — four entries:

  | entry | source | root (dropped joint) | nu | neq | nten | meshes | grasp profile |
  |---|---|---|---|---|---|---|---|
  | `robotiq_2f85` | `floating_robotiq` | `base` (`robotiq_2f85_free_joint`) | 1 | 3 | 1 | 8 | `robotiq_2f85_v1` |
  | `rum` | `floating_rum` | `base` (`rum_free_joint`) | 1 | 1 | 1 | 4 | `rum_parallel_jaw_v1` |
  | `yam_jaw` | `i2rt_yam` | `link_6` (`joint6`) | 1 | 1 | 0 | 6 | `yam_parallel_jaw_v1` |
  | `linear_4310` | bundled vendor model | `graft_mount` (none) | 1 | 1 | 0 | 7 | `linear_4310_parallel_jaw_v1` |

  `linear_4310` is the i2rt LINEAR_4310 + D405 — the gripper MolmoAct2's real bimanual-YAM rig
  runs — bundled verbatim (MIT) under `embodiment_gen/vendor/` because no shipped asset carries
  it; its entry is authored by `build_linear_4310()` rather than lifted from a spec, its
  `graft_mount` adapter flips the vendor's fingers-down frame to the catalog's
  fingers-along-flange-+z convention (clocked to the vendor station's joint6 zero), and its
  wrist camera is the vendor's D405 extrinsic, verified against real MolmoAct2 footage
  (silhouette IoU 0.849 as-designed, 0.955 after a 4 mm / 1.1° refine). It replaced `yam_jaw`
  in the sampled `GRIPPERS` tuple at generator version 3; `yam_jaw` — the crank on our shipped
  `i2rt_yam` — stays in the catalog as the reference-fidelity anchor.

  `franka_droid` and `franka_cap` are deliberately absent: they reuse `floating_robotiq`'s 2F-85 and
  `floating_rum`'s jaw. **`rby1_hand` is excluded** — 64 KB/entry after pruning (8× the others,
  ≈64 GB at 1M), it double-prefixes to `gripper/robot_0/…` because its source is pre-namespaced, and
  its fingers barely move under raw `ctrl` without the spec's `actuator_overrides`.

  Note the 2F-85 is a **6-DoF linkage** (3 equalities + a tendon), not two finger joints; only its
  two driver joints are *declared* in the move group. This matters for DoF accounting.

- **`tcp_frame`, the second `add_site`** (all four entries). The canonical TCP: a rigid frame on the graft
  root body whose origin is the midpoint of the two pads' distal faces with the gripper **fully closed**,
  `z` = approach, `x` = image-up in the wrist camera, `y` = closing axis toward the image-right pad. It is a
  per-family constant because the gripper is grafted verbatim (identical across embodiments to 0.1 mm),
  **measured** (2026-09-04) by settling each gripper closed on a jointed stub flange and read off the
  planner site: rum `(-0.2, 0, -2.6)` mm, robotiq `(0, 0, +18.3)`, linear_4310 `(+0.6, -0.2, +12.4)`,
  yam_jaw `(-5.5, -0.1, +16.7)`, with a `Rz(pi)` on robotiq and linear_4310 whose planner-site `+x` pointed
  down in the wrist image. `mlspaces_tests/.../test_embodiment_gen_grippers.py::test_tcp_frame_sits_at_the_closed_pad_midpoint`
  re-derives it from the grafted model. **Hand-added to the yaml** (like the `lr_swap` note), so
  `make_gripper_catalog.py --check` reports these entries as DIFF until the generator learns the frame.
  Entries emitted before it lack the op; `Kinematics` injects it into its FK model, never into a spec file.

---

## Step 1 — Parameter sampling (`sample_embodiment.py`)

Turn one integer into a complete, frozen description of a robot. Pure function, no MuJoCo, no assets
— so ablations can hold any axis fixed and the whole space is testable in isolation.

```bash
python scripts/embodiments/sample_embodiment.py --seed 43
python scripts/embodiments/sample_embodiment.py --seeds 0 10 --out params/
```

Without `--out` it prints a one-line summary per seed (gripper, DoF, grammar, pedestal height) —
enough to see what a seed asks for before paying to build it.

- **Input**: an integer `seed`.
- **Output**: an in-memory frozen `EmbodimentParams` (**no files**), serialized later as
  `params.json` for provenance.
- **Axes**, independent by construction:
  - `ArmParams` — DoF ∈ {6, 7, 8}, twist-token grammar (`alternating` | `ur` | `mixed`), per-segment
    link length and lateral offset, pedestal height/radius, per-joint ranges, gain family.
  - **gripper choice** — a catalog name from step 0.
  - `DressingParams` — a **mixture** of `primitive` (capsule/cylinder geoms) and `shell` (meshes
    drawn from installed robot assets), sampled ~0.65/0.35. Dressing is **pixels-only**: collision
    is always a slim capsule along the bone plus a joint housing, so swapping dressing changes the
    image and not the behaviour.
- **Rejection**: a sampled twist sequence is checked **numerically** for axis degeneracy (≥3 distinct
  directions; no consecutive collinear pair about a vertical axis, which would be a coaxial
  roll-roll — a parallel *horizontal* pair is a legitimate UR-style planar section). Rejected
  samples redraw within the same seed, so the seed→robot map stays total.
- **Design constraint**: the space is defined so **every sample is spec-valid by construction**
  rather than generated freely and validated after. The `EmbodimentSpec` schema is strict, and it is
  far cheaper to satisfy it structurally than to discover violations at emission time.

---

## Step 2 — Arm build and graft (`build_embodiment.py`)

Realize the sampled parameters as one MuJoCo model: build the arm, graft the gripper, compile.

```bash
python scripts/embodiments/build_embodiment.py --seed 43 --out model.xml --wrist-camera
python scripts/embodiments/build_embodiment.py --seeds 0 50 --out models/ --quiet
```

`--wrist-camera` declares the canonical `wrist_camera` that `GenericOmniPurposeCameraSystem` names.
Step 3 always declares it, so pass it here to build what actually ships.

- **Input**: `EmbodimentParams` + the chosen `GripperAsset` (step 0).
- **Output**: `model.xml` — a single MJCF containing arm and gripper, with compiled inertials, the
  arm's **actuators emitted directly in the model** (matching every shipped robot, whose specs only
  *override* actuators), and mesh references resolving into the installed asset tree.
- **Sequence**: `build_arm(params) -> MjSpec` ▸ `dressing` applies geoms ▸
  `grippers.graft(arm, flange, name)` ▸ `compile()` (the validity gate and inertia computation) ▸
  write.
- **Gate**: a compile failure is a **build rejection**, not a crash — the seed is recorded as
  rejected and the fleet moves on.
- **Cost note**: `attach_body` consumes its child, so the gripper's parsed spec is cached once per
  process and `copy()`-ed per embodiment rather than re-read from disk 1M times.
- **Shared-path constraint**: mesh refs point into the installed asset tree, so **every rank and
  every downstream consumer must see `MLSPACES_ASSETS_DIR` at the same path**.

---

## Step 3 — Spec emission (`emit_embodiment.py`)

Emit the `EmbodimentSpec` that makes the MJCF *mean* something: move groups, controllers,
kinematics, sensors, grasp profiles.

```bash
python scripts/embodiments/emit_embodiment.py --seed 43 --out embodiments/gen_43
python scripts/embodiments/emit_embodiment.py --seeds 0 1000 --library embodiments/
```

A complete entry is four files — `model.xml`, `spec.yaml`, `params.json`, `node_spec.json` — and
loads through the ordinary `load_embodiment_spec`, like any shipped robot.

`node_spec.json` is the morphology-agnostic interface: typed nodes (`{"kind": "body"|"geom",
"name", "centre_offset", "axis"}`), edges, `joint_type` (terminal 0 / revolute 1 / prismatic 2),
`n_arm`, `driver_qpos`, the per-joint mid-range `qpos_center`, and the gripper's **measured**
ctrl→qpos calibration (`gripper_qpos_at_ctrl`, obtained by driving the actuator to each extreme on the
bare catalog gripper on a rigid mount in zero gravity, once per family: the servo's setpoint, a property of the gripper rather than of any arm — the declared joint limits are wrong by up to 0.78 rad and direction-inverted on a mirrored finger).
Node centres sit at the **joint** (`centre_offset` is `jnt_pos`, non-zero where a hinge is offset
from its body origin), driver axes are **signed toward opening**, and a terminal is a rigid off-axis
observable — a child body where one exists, a named geom where the finger is a leaf.

- **Input**: `EmbodimentParams`, the compiled model, the `GripperAsset`.
- **Output**: two files, at different rates —
  - `<library>/<bucket>/gen_<seed>/spec.yaml` — per entry, small: `extends:` the gripper fragment
    and overrides only what is arm-specific.
  - `<library>/_grippers/<gripper>.yaml` — **once per library**, holding the gripper's
    `move_groups.gripper`, its `grasp_profiles` block and its controller binding.
- **Why the split**: `loader._merge_mappings` is a **recursive deep merge**, so a child's
  `move_groups.arm` lands beside the parent's `move_groups.gripper` without clobbering it. Measured
  on a 6-DoF + `yam_jaw` spike: **7.6 KB inlined → 4.9 KB entry + 2.7 KB shared once**, so ~36%
  smaller (≈7.6 GB → ≈4.9 GB at 1M entries). The saving is bounded by the lists that cannot move
  into the fragment.
- **Limit to respect**: the merge recurses into dicts but **replaces lists wholesale**. `robot.sensors`
  and `robot.model_operations` are lists, so they **cannot** be split across parent and child — each
  entry emits its *full* sensor list (the gripper's contact/aperture sensors plus the arm's
  `joint_state` and `frame_pose` sensors) and its full operation list. Only dict-shaped blocks live
  in the fragment.
- **Location independence**: the entry declares `source: path`, `root_path: .`, `mjcf_path: model.xml`.
  `loader.py` resolves a relative `root_path` against the spec file's own directory, so an entry
  survives being copied, moved or mounted elsewhere with no rewriting.
- **Conditioning comes free**: per-node anchors are declared as `frame_pose` sensors
  (`relative_to_group: base`) and recorded per frame by the ordinary recorder, which also already
  writes `actions/joint_pos_rel` ("delta between commanded joint pos and current joint pos"). There
  is therefore **no separate geometry-annotation stage** — no second FK pass, no per-(task, robot)
  environment build, and none of the stale-handle failure modes that come with one.
- **Gate**: the emitted pair must pass `load_embodiment_spec` (pydantic) before anything is written
  durably.

---

## Step 4 — Validation (`validate_embodiment.py`)

Tiers, cheapest first, so the expensive one only ever sees survivors. A failing entry is a **verdict,
not an exception** — generation is a filter, and a seed that cannot be driven is dropped rather than
crashing a fleet.

```bash
python scripts/embodiments/validate_embodiment.py embodiments/gen_43
python scripts/embodiments/validate_embodiment.py --library embodiments/ --tiers T1
python scripts/embodiments/validate_embodiment.py --library embodiments/ --json verdicts.jsonl
```

| tier | what it checks | cost |
|---|---|---|
| **T1** structural | `load_embodiment_spec` + MJCF compile; the model the spec names exists and loads | ms |
| **T2** rest pose + reachability | the emitted `initial_qpos` has **zero structural contacts on the assembled robot**, then MolmoSpaces' **own** IK solves a workspace pose bank to the gripper's leaf frame | seconds |
| **T3** task gauntlet | one scripted pick in one house through the real datagen path | seconds |

The contact check runs on the **assembled** robot because the pedestal only exists after assembly —
the bare build model cannot see arm-vs-pedestal interpenetration, and one shipped robot started with
a link 30 mm inside its pedestal, interpenetrated it across 398/400 of sampled joint space, and
scored the *highest* IK-reach of its cohort (IK knows nothing of collision) before failing every
planner episode. Structural means any contact not internal to the gripper: gripper-internal contacts
are the asset's own (the shipped i2rt_yam rests with 6 penetrating jaw contacts, and every yam_jaw
graft inherits exactly those).

**T2 must use MolmoSpaces' own DLS solver**, not an independent one, because `check_feasible_ik`
routes through `kinematics.ik` (single) and `parallel_kinematics.ik` (batch) at plan time —
`base_object_manipulation_planner_policy.py:1304`. This is measured, not assumed: an earlier screen
used a damped-least-squares solver local to `builder`, and its score correlated with real planner
success at **rho = +0.05 across 36 accepted embodiments**. It gated usefully — the accepted population
does the task far more often than the general one — but it ranked nothing, which is exactly how a
mismatched screen fails.

Three details T2 has to get right, each of which produced a wrong answer first:

- **Solve on the assembled robot, not the raw MJCF.** The frame the planner aims at is the gripper
  move group's `leaf_frame`, and for `yam_jaw` that is `gripper/planner_grasp_site` — created by the
  entry's own `add_site` operation at load time. It is absent from `model.xml`, so aiming at the
  compiled model's `grasp_site` instead lands 49.7 mm away.
- **Assemble under the config's own namespace.** A `ConfiguredRobot` rejects any other prefix
  (*"insertion prefix '' must match configured namespace 'robot_0/'"*), and the robot view has to be
  built under the same one or every joint lookup misses.
- **Pass the view's current qpos as `q0`, as the planner does** — not `robot_config.init_qpos`, which
  omits the base group, so the solver rejects the call outright (*"q0 keys must match move group
  ids"*). And never count a raised exception as an unreachable pose: doing so once reported a
  plausible-looking 0/30 that was entirely a harness fault.

The pose bank is a box of positions around the arm's **rest** gripper pose, holding its orientation,
in metres rather than scaled by reach — the workspace a task occupies is tabletop-sized and does not
grow with the arm. Anchoring on the rest pose is what makes it robot-agnostic: `base.pose` is at the
**floor**, not at the arm root, so "table height" is not `z = 0` in that frame. Assuming it was put
every target at ankle height, which every arm failed.

Holding the rest orientation rather than forcing tool-down is deliberate: the screen asks whether the
planner's solver can move the arm around its workspace, and forcing an orientation folds in a second
question — whether the wrist can reach that orientation at all.

**T3 is not implemented as a tier.** Run it by hand for now, which is also how you reproduce a
planner result for a single entry:

```bash
python molmo_spaces/data_generation/main.py GenericPickAndPlaceOmniCamConfig \
    --embodiment-spec embodiments/gen_43/spec.yaml \
    --house-ind 0 --samples-per-house 2 --seed 43 --output-dir runs/gen_43
```

Give every concurrent run its **own** `--output-dir`. Runs otherwise collide, and because camera
systems share exocentric camera *names*, one silently overwrites another's videos.

**Use the non-curobo planner policies throughout.** The `curobo_*` variants need per-robot collision
configs, which is fatal at a million robots.

## Step 5 — Generation fleet (`generate_embodiments.py`)

One unit of work = one seed: sample ▸ build ▸ emit ▸ validate ▸ export.

```bash
# single process
python scripts/embodiments/generate_embodiments.py --seeds 0 100000 --out library/

# data-parallel: identical command in every process, RANK strides the range
RANK=$REPLICA_RANK WORLD_SIZE=128 \
    python scripts/embodiments/generate_embodiments.py --seeds 0 1000000 --out library/
```

- **Output**: `<library>/<seed//1_000_000, 6-digit>/gen_<seed>/{model.xml, spec.yaml, params.json}` for every
  surviving seed, plus a per-seed marker under `<library>/.markers` (relocatable with `--markers`,
  for when the library itself must stay clean).
- **Run on**: `WORLD_SIZE` CPU processes, identical command, `RANK` striding the range. No GPU.
- **Resumability**: the marker is written **after** the entry is durable, so a crash writes no marker
  and re-running the identical command redoes exactly the lost seed. `--force` ignores markers.
- **A rejected entry is removed, not left behind.** If validation fails after emission, the partial
  directory is deleted before the marker is written — otherwise the library would contain entries the
  index would then publish.
- **Markers record why**, not just that: each holds the seed, the generator version, the stage that
  rejected it and the reason, so the yield of a run can be analysed without re-running it.
- **Rejections are silent and expected.** The library is therefore *sparse* in seed space, which is
  why step 6 records paths rather than assuming `gen_<seed>` exists.
- Every rank prints a tally: `accepted=N rejected_build=N rejected_T2=N skipped=N`.

## Step 6 — Library index (`write_embodiment_index.py`)

```bash
python scripts/embodiments/write_embodiment_index.py library/
```

- **Input**: a finished library.
- **Output**: `<library>/ids.txt` — one path per line, **relative to the library root**, in seed
  order.
- **Run on**: one process, **once, after every rank of step 5 has finished**.
- **Why relative paths, not just ids**: recording the path is what lets the producer choose the
  layout — a small library can be flat (`--flat` in steps 3 and 5), a million-robot library is
  bucketed (`001234/gen_1234000567`) — and no reader has to know which.
- **Never publish early**: readers short-circuit to the frozen key list, so an index stamped mid-run
  permanently hides everything generated after it. The command refuses to overwrite an existing index
  without `--force`, for that reason.
