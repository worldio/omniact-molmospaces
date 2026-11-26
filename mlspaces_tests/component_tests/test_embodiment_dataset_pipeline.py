"""The fused pipeline's scene-corpus draw and its conversion plumbing, tested without running
datagen. Three of these pin behaviour that broke for real: the two index-map shapes (ithor maps an
index straight to a path, the procthor family to a variant dict, both padded with None), the seed
fold (house indices run to ~100k and seed + house_id * 1_000_003 overflowed numpy's 2**32 legacy
seeding limit -- invisible while house_inds never left 0..3), and the scene-qualified wods key
(a house INDEX means nothing without the dataset it indexes into).
"""

import sys
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "embodiments"))

import embodiment_wods  # noqa: E402
import generate_embodiment_dataset as pipeline  # noqa: E402

from molmo_spaces.tasks.task_sampler import BaseMujocoTaskSampler  # noqa: E402


def test_scene_registry_is_every_nondebug_dataset():
    """docs/assets.md lists exactly these four; the -debug aliases must never be drawn."""
    assert set(pipeline.SCENE_DATASETS) == {"ithor", "procthor-10k", "procthor-objaverse",
                                            "holodeck-objaverse"}
    assert pipeline.SCENE_SPLIT == "train"


def test_scene_corpus_reads_both_index_map_shapes(monkeypatch):
    maps = {
        # procthor-family shape: index -> {variant: path_or_None}, missing indices padded
        "procthor-10k": {0: {"base": "h0.xml"}, 1: {"base": None}, 2: {"base": "h2.xml"}},
        # ithor shape: index -> path_or_None
        "ithor": {1: "FloorPlan1.xml", 2: None, 5: "FloorPlan5.xml"},
    }
    monkeypatch.setattr("molmo_spaces.molmo_spaces_constants.get_scenes",
                        lambda name, split: {split: maps[name]})
    corpus = pipeline.scene_corpus(("ithor", "procthor-10k"))
    assert corpus == [("ithor", [1, 5]), ("procthor-10k", [0, 2])]


def test_scene_corpus_drops_empty_datasets(monkeypatch):
    monkeypatch.setattr("molmo_spaces.molmo_spaces_constants.get_scenes",
                        lambda name, split: {split: {0: None}})
    assert pipeline.scene_corpus(("ithor",)) == []


def test_derived_work_item_seed_folds_into_numpy_range():
    """The pipeline seeds workers with seed + house_id * 1_000_003 + batch; at house 99999 that is
    ~1e11 and numpy's legacy MT19937 refuses anything >= 2**32. The fold lives in
    seed_task_sampling, and current_seed must keep the folded value because per-episode seeds are
    derived from it by small offsets and come back through the same method."""
    sampler = object.__new__(BaseMujocoTaskSampler)
    derived = 5 + 99999 * 1_000_003 + 1  # a real work-item construction from the corpus edge
    sampler.seed_task_sampling(derived)  # raised ValueError before the fold
    assert sampler.current_seed == derived % 2**32

    sampler.seed_task_sampling(42)
    assert sampler.current_seed == 42  # in-range seeds pass through unchanged


def test_convert_run_keys_are_scene_qualified_and_success_filtered(tmp_path, monkeypatch):
    house_dir = tmp_path / "attempt0" / "house_3"
    house_dir.mkdir(parents=True)
    with h5py.File(house_dir / "episodes.h5", "w") as handle:
        handle.create_group("traj_0").create_dataset("success", data=np.array([False, False]))
        handle.create_group("traj_1").create_dataset("success", data=np.array([False, True]))

    written = []
    monkeypatch.setattr(embodiment_wods, "_write_item",
                        lambda traj, house, dataset, key, *a, **k: written.append(key) or True)

    keys = embodiment_wods.convert_run(tmp_path, "unused-dataset", seed=7, task_name="SomeTask",
                                       scene="ithor-train")
    assert keys == written == ["gen_7/SomeTask/ithor-train_house_3_traj01"]

    written.clear()
    keys = embodiment_wods.convert_run(tmp_path, "unused-dataset", seed=7, task_name="SomeTask")
    assert keys == ["gen_7/SomeTask/house_3_traj01"]  # no scene given -> legacy key shape


def test_run_datagen_pins_the_scene_on_the_subprocess(tmp_path, monkeypatch):
    # run_datagen dispatches the datagen argv to run_redirected via a forkserver Process; intercept
    # that Process so nothing actually runs and assert the scene/split/house flags land in the argv.
    calls = []

    class _FakeProc:
        def start(self):
            pass

        def is_alive(self):
            return False

        def join(self, *a, **k):
            pass

    class _FakeCtx:
        def Process(self, target=None, args=(), name=None):
            calls.append(args[0])          # args == (argv, log_path)
            Path(args[1]).write_text("")   # run_redirected would create this log
            return _FakeProc()

    monkeypatch.setattr(pipeline, "_forkserver_ctx", lambda: _FakeCtx())
    won, detail = pipeline.run_datagen("SomeConfig", Path("spec.yaml"), tmp_path / "run", seed=9,
                                       scene="holodeck-objaverse", house=86243, episodes=2,
                                       timeout=10)
    (command,) = calls
    for flag, value in (("--scene-dataset", "holodeck-objaverse"), ("--data-split", "train"),
                        ("--house-ind", "86243")):
        assert value == command[command.index(flag) + 1]
    assert not won and detail.startswith("no result line")
