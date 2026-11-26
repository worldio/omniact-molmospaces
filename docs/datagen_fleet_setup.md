# Embodiment datagen: clean-clone / new-box setup

What has to be true before launching an embodiment-generation datagen fleet
(`scripts/embodiments/generate_embodiment_dataset.py`) from a fresh checkout on new hardware, and
**why** — most of these exist to make the run deadlock-free and provably fork-safe. Do them in order;
the last section is a copy-paste checklist.

The driver forks one attempt per house draw through a forkserver, runs the MuJoCo/cuRobo pipeline under
osmesa (software) rendering with `--gpus 0`, and publishes trajectories to GCS. Almost everything below
is about not hanging: every historical stall has been a `FileLock` on a network filesystem that a
SIGKILL'd rank poisoned, or a non-fork-safe fork.

---

## 1. System packages + Python env

Software (osmesa) rendering + the mujoco extra:

```bash
apt-get install -y --no-install-recommends cmake build-essential libosmesa6 libgl1 libjemalloc2
uv pip install -e ".[mujoco]" gcsfs
```

## 2. Environment variables

| var | value | why |
|---|---|---|
| `MLSPACES_ASSETS_DIR` | a persistent shared/network filesystem | asset/symlink root. **Never leave it on the ephemeral `/root` default** — it ships a stale manifest. |
| `MLSPACES_CACHE_DIR` | persistent shared dir (same store) | download/extract cache. |
| `MUJOCO_GL` | `osmesa` | CPU software rendering (no GPU on `--gpus 0`). Read at `import mujoco`, so it must be set before the process starts. |
| `PYOPENGL_PLATFORM` | `osmesa` | matches `MUJOCO_GL`. |
| `LD_PRELOAD` | `/usr/lib/x86_64-linux-gnu/libjemalloc.so.2:/usr/lib/x86_64-linux-gnu/libstdc++.so.6` | jemalloc as the allocator (§5b), then the system libstdc++ osmesa/llvmpipe needs. |
| `NLTK_DATA` | pre-warmed dir with `wordnet` + `wordnet2022` | task sampling uses wordnet; pre-warm to avoid a per-rank download race. |
| **`OMP_NUM_THREADS`** | **`1`** | **fork-safety (see §5).** |
| **`OPENBLAS_NUM_THREADS`** | **`1`** | fork-safety. |
| **`MKL_NUM_THREADS`** | **`1`** | fork-safety (torch-MKL builds). |
| **`NUMEXPR_NUM_THREADS`** | **`1`** | fork-safety. |
| **`VECLIB_MAXIMUM_THREADS`** | **`1`** | fork-safety. |
| `LP_NUM_THREADS` | `8` | osmesa/llvmpipe render throughput (ablation: 1→8 ≈ +66% node throughput; 8→16 ≈ +7%). This is a *render* pool spawned in the worker at run time, so it does **not** affect fork-safety. |
| `MLSPACES_RESOURCE_LOCK_FREE` | `True` | skip the shared-filesystem resource install locks — **only after §3**, else on-demand download breaks. |
| `CUROBO_CPU` | `0` | skip the cuRobo CPU-patch — `maybe_install()` returns early, so cuRobo is never imported and need not be installed (datagen plans with warp IK; `.[mujoco]` suffices). Without this, on a no-GPU box `maybe_install` tries to patch cuRobo and fails with `No module named 'curobo'`. Run `--gpus 0`; no GPU also keeps CUDA uninitialized for the §5 fork check. |
| GCS creds | `GOOGLE_APPLICATION_CREDENTIALS` | trajectory upload. |

## 3. Pre-download every asset (required for lock-free)

The run must not download anything at runtime — on-demand download needs a shared-filesystem lock, and a
SIGKILL'd holder poisons it (network-FS locks aren't reclaimed on death), deadlocking every peer in
`install_grasps_for_scene`. So pre-install **all** pinned scenes/objects/grasps once, then run with
`MLSPACES_RESOURCE_LOCK_FREE=True`. Built-in mechanism:

```bash
MLSPACES_DOWNLOAD_EXTRACT_ALL_SCENES_OBJECTS_GRASPS=True \
python -c "from molmo_spaces.molmo_spaces_constants import get_resource_manager; get_resource_manager(force_post_setup=True)"
```

- Covers the full pinned set: ~232k scenes + ~130k objects + ~48k grasps (~500 GB–1 TB). Already-cached
  packages are skipped (verify-only), so it's a top-up.
- Single writer, or shard by **source** across parallel runs (disjoint files/manifests → no race):
  `python -c "…get_resource_manager().install_all_for_source('objects','objaverse')"`, etc.
- Scenes are cached with `skip_linking=True` and linked on demand at runtime; that linking works
  lock-free (verified), so cache-only is sufficient.

**Verify** (sample a few installed packages, re-download fresh, byte-compare) before trusting the cache —
there is no checksum in the resource manager, so a truncated file would otherwise pass the existence
check. A clean sample ⇒ gap-fill only; a mismatch ⇒ force a re-extract.

## 4. Pre-build the objathor-metadata LMDB

`get_db()` builds a ~1.4 GB LMDB at `ASSETS_DIR/.lmdb/objathor_metadata/<version>` by merging
`clip_features.pkl` + `objects_metadata.json.gz` (per-object CLIP features + annotations, used for the
referral-expression ranking during task sampling). It is a **derived** artifact — the pre-download does
**not** build it. If it's missing at run time the first rank builds it under a shared-filesystem `FileLock`
while the rest wait; a SIGKILL there poisons the lock. Build it once, offline:

```bash
python -c "from molmo_spaces.utils.object_metadata import get_db; get_db(); print('lmdb ready')"
```

Confirm `ASSETS_DIR/.lmdb/objathor_metadata/<pinned_version>` exists. (The build lock now has a
timeout, so a poisoned lock fails fast instead of hanging — but pre-building avoids the build entirely.)

## 5. Fork-safety check (must be single-threaded at import)

The driver forks each attempt from a forkserver that has `import`ed the datagen stack. Forking a
multi-threaded process is unsafe (a native thread holding a lock at `fork()` → the child inherits a
locked mutex with no owner → deadlock). So the process that gets forked must be **single-threaded** when
it forks.

With the `*_NUM_THREADS=1` env from §2 **and** no CUDA (`--gpus 0`, so importing cuRobo/torch doesn't
init a CUDA driver thread), importing the stack is single-threaded. Verify on the target box:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
python -c "import os, molmo_spaces.data_generation.main; print('OS threads:', len(os.listdir('/proc/self/task')))"
# expect: OS threads: 1
```

If it prints `> 1`, forking is not provably safe — find the extra thread (`/proc/self/task/*/comm`); a
`cuda…` thread means CUDA is initializing (a GPU/driver is visible — it must not be), a BLAS-pool thread
means a `*_NUM_THREADS` cap is missing. (On a GPU dev box the count is 255+; that's expected there and
irrelevant to the `--gpus 0` fleet.)

Given this holds, the pipeline runs `num_workers=1` (in-process, no worker subprocess), so there is no
other fork and no shared-`counter_lock`/hung-worker path in play.

## 5b. Driver memory + allocator

Screening recompiles each generated MJCF many times in-process (`emit`→`build`); under glibc `malloc`
the freed compile buffers fragment a rank to tens of GB over a shard, driving nodes into
swap/NUMA-reclaim thrash. Preload **jemalloc** (`LD_PRELOAD=…/libjemalloc.so.2`, §2): it holds a rank
near its working set (~1.6 GB peak vs glibc-default's ~2.7 GB over the same screening) **and** is faster
than the `MALLOC_TRIM_THRESHOLD_=0` glibc tunable it supersedes — that tunable forced a trim on every
`free()`, which cost ~1.3× screening throughput at 64-way concurrency (~2.8× single-process). Byte-identical
output (allocator behaviour only). It's a preload read before `main`, so it must be in the launch env,
not set from Python; keep libstdc++ in the same `LD_PRELOAD`, jemalloc first.

## 6. Launch

Only after 3–5: run with `MLSPACES_RESOURCE_LOCK_FREE=True` and the §2 env. The per-attempt watchdog is
**warn-only** — `--datagen-timeout` just logs "still running after Ns" periodically; nothing is killed,
because with the locks handled and the fork proven safe there is no deadlock to abort, and a legit
attempt can legitimately run up to `max_attempts`(8) × `task_horizon`(500) steps.

---

## Version pinning

The pinned versions live in `DATA_TYPE_TO_SOURCE_TO_VERSION`. §3 (download) and §4 (LMDB) are
**per-version** — bumping any `scenes`/`objects`/`grasps`/`objathor_metadata` version means redoing that
step for the new version before the next launch, or the run falls back to runtime download/build (the
lock path we're avoiding).

## Checklist

- [ ] `apt` deps + `uv pip install -e ".[mujoco]" gcsfs`
- [ ] env: asset dirs (persistent), `MUJOCO_GL/PYOPENGL_PLATFORM=osmesa`, `LD_PRELOAD` (jemalloc + libstdc++), `NLTK_DATA`
- [ ] env: `OMP/OPENBLAS/MKL/NUMEXPR/VECLIB_NUM_THREADS=1`, `LP_NUM_THREADS=8`
- [ ] pre-download all scenes/objects/grasps (§3) **+ verify a sample byte-for-byte**
- [ ] pre-build the objathor-metadata LMDB (§4)
- [ ] fork-safety check prints `OS threads: 1` (§5)
- [ ] launch with `MLSPACES_RESOURCE_LOCK_FREE=True`, `--gpus 0`
