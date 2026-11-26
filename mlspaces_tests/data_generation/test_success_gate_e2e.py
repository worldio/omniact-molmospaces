"""End-to-end: the datagen success gate agrees with an independent offline replay of the same episodes.

Runs the real pipeline (data_generation/main.py) with ``--replay``. The gate prints a ``SUCCESS_GATE {json}`` verdict
for every success it judges and rejects the ones with reasons; the Generic OmniCam configs save failed episodes too
(filter_for_successful_trajectories=False), so rejected episodes still write a replay bundle. Each verdict is then
re-derived from its bundle independently of the live gate: the scene is rebuilt from the recipe
(``recon.build_model_from_recipe``), the recorded states are replayed for robot<->target contacts, and the target's
free joint is tracked in qpos. The live gate and the replay must agree on the success frame, whether the robot
touched the target before it (first-contact frame within one frame: the live read sees the state one physics substep
before the recorded one), the target displacement, and the verdict -- which must also match what the case expects.

Cases (embodiments and houses from embgen_ms_1m, checked on video):
  pick_kept     a real pick (lift) -- robot-caused, kept
  close_lid     the toilet whose lid falls shut by step 5 before the arm arrives -- no_contact

Slow (a few minutes per case, a scene rebuild per episode) and needs the full datagen environment, so opt-in:

    EMBGEN_DATAGEN_E2E=1 MLSPACES_ASSETS_DIR=... MLSPACES_CACHE_DIR=... [MUJOCO_GL=osmesa ...] \\
        pytest mlspaces_tests/data_generation/test_success_gate_e2e.py
"""

import base64
import io
import json
import os
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
LIBRARY = Path(os.environ.get("EMBGEN_E2E_LIBRARY", "/gpfs/scrubbed/jy050706/embgen_ms_1m_fixed/library"))

pytestmark = pytest.mark.skipif(
    not (os.environ.get("EMBGEN_DATAGEN_E2E") == "1" and os.path.isdir(os.environ.get("MLSPACES_ASSETS_DIR", ""))
         and LIBRARY.is_dir()),
    reason="datagen e2e needs EMBGEN_DATAGEN_E2E=1, MLSPACES_ASSETS_DIR and the embgen_ms_1m library",
)

CASES = {
    "pick_kept": dict(config="GenericPickOmniCamConfig", gen=100000085, scene="ithor", house=11, expect=[]),
    "close_lid": dict(config="GenericCloseOmniCamConfig", gen=237002488, scene="procthor-objaverse", house=84140,
                      expect=["no_contact"]),
}


# ---- independent offline replay of one bundle ------------------------------------------------------------------
class _Stub:
    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        self.__dict__["_state"] = state


class _StubUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.split(".")[0] in ("builtins", "collections", "copyreg", "_codecs", "numpy"):
            return super().find_class(module, name)
        return type(name, (_Stub,), {})


def _task_config(meta):
    """frozen_config.task_config fields, unpickled without importing molmo_spaces classes."""
    state = _StubUnpickler(io.BytesIO(base64.b64decode(meta["obs_scene"]["frozen_config"]))).load().__dict__["_state"]
    task_config = state["__dict__"]["task_config"].__dict__["_state"]
    return task_config.get("__dict__", task_config)


def _success_start(meta):
    succ = [bool(s) for s in meta["per_step"]["success"]]
    start = max(i for i, s in enumerate(succ) if s)
    while start > 0 and succ[start - 1]:
        start -= 1
    return start


def _replay(bundle: Path, meta: dict, upto: int) -> dict:
    """First robot<->target contact frame in [0, upto] and the target's free-joint displacement over it."""
    import mujoco

    from molmo_spaces.data_generation.recon import build_model_from_recipe

    task_config = _task_config(meta)
    model = build_model_from_recipe(bundle, meta)
    data = mujoco.MjData(model)
    with np.load(bundle / "states.npz") as z:
        states, spec, nq = z["states"], int(z["state_spec"]), int(z["nq"])

    robot = np.array([(model.body(i).name or "").startswith("robot_0/") for i in range(model.nbody)])
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, task_config["pickup_obj_name"])
    target = np.zeros(model.nbody, dtype=bool)
    target[root] = True
    for body in range(root + 1, model.nbody):   # depth-first body order: children follow parents
        target[body] = target[model.body_parentid[body]]
    first = None
    for frame in range(upto + 1):
        mujoco.mj_setState(model, data, states[frame], spec)
        mujoco.mj_forward(model, data)
        pairs = model.geom_bodyid[data.contact.geom[: data.ncon]]
        if data.ncon and ((robot[pairs[:, 0]] & target[pairs[:, 1]]) | (robot[pairs[:, 1]] & target[pairs[:, 0]])).any():
            first = frame
            break

    moved = None
    pose = (task_config.get("object_poses") or {}).get(task_config["pickup_obj_name"])
    if pose is not None:
        qpos = states[: upto + 1, 1: 1 + nq]
        hits = np.flatnonzero(np.abs(np.lib.stride_tricks.sliding_window_view(qpos[0], 7) - np.asarray(pose)).max(1) < 1e-5)
        if len(hits) == 1:
            moved = float(np.linalg.norm(qpos[:, hits[0]: hits[0] + 3] - qpos[0, hits[0]: hits[0] + 3], axis=1).max())
    return {"first_contact": first, "target_pos": moved}


# ---- test -------------------------------------------------------------------------------------------------------
def _run_datagen(case, out_dir: Path) -> str:
    spec = LIBRARY / f"{case['gen'] // 1_000_000:06d}" / f"gen_{case['gen']}" / "spec.yaml"
    argv = [sys.executable, str(REPO / "molmo_spaces/data_generation/main.py"), case["config"],
            "--embodiment-spec", str(spec), "--scene-dataset", case["scene"], "--data-split", "train",
            "--house-ind", str(case["house"]), "--samples-per-house", "2", "--seed", str(case["gen"]),
            "--num-workers", "1", "--output-dir", str(out_dir), "--replay"]
    env = {**os.environ, "PYTHONPATH": str(REPO)}
    proc = subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=3600)
    log = proc.stdout + proc.stderr
    (out_dir / "datagen.log").write_text(log)
    assert proc.returncode == 0, log[-3000:]
    return log


@pytest.mark.parametrize("name", sorted(CASES))
def test_live_gate_matches_offline_replay(name, tmp_path):
    case = CASES[name]
    log = _run_datagen(case, tmp_path)
    verdicts = [json.loads(line[len("SUCCESS_GATE "):]) for line in log.splitlines() if line.startswith("SUCCESS_GATE ")]
    assert verdicts, f"no success was judged (see {tmp_path / 'datagen.log'})"
    assert "success gate init failed" not in log and "Success gate disabled" not in log
    if any(v["reasons"] for v in verdicts):
        assert "the robot did not cause it" in log, "a rejected success must be marked a failure"

    for verdict in verdicts:
        bundles = list(tmp_path.rglob(f"replay/seed_{verdict['episode_seed']}"))
        assert len(bundles) == 1, f"no single replay bundle for episode_seed {verdict['episode_seed']}"
        meta = json.loads((bundles[0] / "meta.json").read_text())
        s = _success_start(meta)
        assert (meta["n_frames"], s) == (verdict["n_frames"], verdict["success_start"]), verdict

        offline = _replay(bundles[0], meta, s)
        assert (verdict["first_contact"] is None) == (offline["first_contact"] is None), (verdict, offline)
        if verdict["first_contact"] is not None:
            assert abs(verdict["first_contact"] - offline["first_contact"]) <= 1, (verdict, offline)
        if verdict["target_pos"] is not None:
            assert offline["target_pos"] is not None and abs(offline["target_pos"] - verdict["target_pos"]) < 5e-3, (
                verdict, offline)
        assert verdict["reasons"] == case["expect"], verdict
