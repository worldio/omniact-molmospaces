import gzip
import os
import json
import logging
from pathlib import Path
import pickle
from typing import TYPE_CHECKING
from collections.abc import Mapping
from functools import cache, lru_cache
from itertools import chain

import numpy as np
from filelock import FileLock, Timeout
from tqdm import tqdm

from molmo_spaces.utils.lazy_loading_utils import UserAssetLibraryIndexEntry, get_user_library_index

from molmospaces_resources import PickleLMDBMap

from molmo_spaces.molmo_spaces_constants import (
    ASSETS_DIR,
    DATA_TYPE_TO_SOURCE_TO_VERSION,
    USER_ASSET_LIBRARIES,
    get_resource_manager,
)

log = logging.getLogger(__name__)


def get_metadata_lmdb_dir():
    metadata_version = DATA_TYPE_TO_SOURCE_TO_VERSION.get("objects", {}).get(
        "objathor_metadata", None
    )
    if metadata_version is None:
        return None
    return ASSETS_DIR / ".lmdb" / "objathor_metadata" / metadata_version


if TYPE_CHECKING:
    from molmo_spaces.tasks.task import BaseMujocoTask

DEFAULT_CLIP_MODEL = "ViT-L-14"
DEFAULT_CLIP_PRETRAIN = "laion2b_s32b_b82k"
DEFAULT_DEVICE = "cpu"

_CLIP = None
_DB = None


def _get_clip_features():
    with open(ASSETS_DIR / "objects" / "objathor_metadata" / "clip_features.pkl", "rb") as f:
        return pickle.load(f)


def get_annotation():
    with gzip.open(ASSETS_DIR / "objects" / "objathor_metadata" / "objects_metadata.json.gz") as f:
        return json.load(f)


def _install_mmap_checkpoint_loader() -> None:
    """Load the CLIP weights memory-mapped and assign them, instead of copying into fresh tensors.

    open_clip reads the checkpoint with torch.load() and then load_state_dict()s it, so every
    process that builds the model holds a private ANON copy of the weights -- and anon is exactly
    what the cgroup OOM killer counts and what swap will not rescue.

    Measured on the 1.59 GB ViT-L-14 checkpoint:
        torch.load(mmap=False)  +1.60 GB resident
        torch.load(mmap=True)   +0.00 GB resident   (pages stay file-backed)

    File-backed pages are held ONCE by the page cache for every process on the node that maps the
    same file, and are reclaimable under pressure rather than fatal. `assign=True` is what keeps
    them mapped: without it load_state_dict copies straight back into anon.

    Best-effort: any failure leaves open_clip's own loader in place, since a slightly fatter
    process is much better than no CLIP.
    """
    try:
        import torch
        from open_clip import factory

        if getattr(factory.load_checkpoint, "_mmap_patched", False):
            return
        original = factory.load_checkpoint

        def load_checkpoint(model, checkpoint_path, strict=True, weights_only=True, device="cpu"):
            try:
                if str(checkpoint_path).endswith(".safetensors"):
                    return original(model, checkpoint_path, strict=strict,
                                    weights_only=weights_only, device=device)
                state_dict = torch.load(checkpoint_path, map_location=device,
                                        weights_only=weights_only, mmap=True)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # open_clip strips this prefix itself for older checkpoints
                if next(iter(state_dict), "").startswith("module."):
                    state_dict = {k[7:]: v for k, v in state_dict.items()}
                return model.load_state_dict(state_dict, strict=strict, assign=True)
            except Exception as error:  # noqa: BLE001 -- never fail the run over an optimisation
                log.warning(f"mmap CLIP load failed ({type(error).__name__}: {error}); "
                            f"falling back to open_clip's loader")
                return original(model, checkpoint_path, strict=strict,
                                weights_only=weights_only, device=device)

        load_checkpoint._mmap_patched = True
        factory.load_checkpoint = load_checkpoint
    except Exception as error:  # noqa: BLE001
        log.warning(f"could not install the mmap CLIP loader: {type(error).__name__}: {error}")


def get_clip_model():
    global _CLIP

    if _CLIP is None:
        # Imported HERE, not at module scope. open_clip pulls in transformers, and that import cost
        # ~8 minutes of a profiled rank's first 10 -- 93% of the samples -- because this module is
        # reached transitively from validate.tier1() via robots.configured -> configs -> task ->
        # object_manager. Screening is 99.66% of the datagen workload and never calls this function,
        # so every rank was paying a CLIP-sized import for nothing, on every process start.
        try:
            import open_clip
        except ImportError as error:
            raise ImportError("Try `pip install open-clip-torch` for open_clip") from error

        # MLSPACES_CLIP_CACHE lets a fleet point every rank at ONE checkpoint on shared storage.
        # That matters twice over: the node-local default re-downloads 1.6 GB per node, and pages
        # from a shared file are held once by the page cache for every process that maps it.
        # Falls back to node-local /tmp, where a lock on a network FS isn't reclaimed on SIGKILL.
        clip_cache = Path(os.environ.get("MLSPACES_CLIP_CACHE") or (Path("/tmp") / "mlspaces_open_clip"))
        clip_cache.mkdir(parents=True, exist_ok=True)

        def _load():
            model, _, preprocess = open_clip.create_model_and_transforms(
                DEFAULT_CLIP_MODEL,
                pretrained=DEFAULT_CLIP_PRETRAIN,
                device=DEFAULT_DEVICE,
                cache_dir=str(clip_cache),
            )
            return model, preprocess

        _install_mmap_checkpoint_loader()

        try:
            with FileLock(str(clip_cache / "download.lock"), timeout=300):
                clip_model, clip_preprocess = _load()
        except Timeout:
            log.warning("open_clip download lock timed out after 300s; loading without it")
            clip_model, clip_preprocess = _load()

        clip_tokenizer = open_clip.get_tokenizer(DEFAULT_CLIP_MODEL)
        _CLIP = dict(model=clip_model, preprocess=clip_preprocess, tokenizer=clip_tokenizer)

    return _CLIP


class UserLibraryMetadata(Mapping):
    """
    Class which provides dict-like access to a user library metadata.
    """

    def __init__(
        self,
        user_library_path: Path,
        user_library_index: dict[str, UserAssetLibraryIndexEntry],
        lru_cache_size: int = 1000,
    ):
        self._user_library_path = user_library_path
        self._user_library_index = user_library_index

        @lru_cache(maxsize=lru_cache_size)
        def _get(key):
            metadata_entry = self._user_library_index[key]
            metadata_rel_path = metadata_entry.metadata_path
            with open(self._user_library_path / metadata_rel_path, "r") as f:
                metadata: dict = json.load(f)

            if metadata_entry.metadata_npz_path is not None:
                metadata_npz = np.load(self._user_library_path / metadata_entry.metadata_npz_path)

                for npz_key, npz_value in metadata_npz.items():
                    d = metadata
                    key_parts = npz_key.split("/")
                    for k_part in key_parts[:-1]:
                        d = d.setdefault(k_part, {})
                    k = key_parts[-1]
                    assert k not in d, (
                        f"Key {npz_key} from npz metadata already exists in metadata for uid={key}"
                    )
                    d[k] = npz_value

            return metadata

        self._get = _get

    def __contains__(self, key):
        return key in self._user_library_index

    def __getitem__(self, key):
        return self._get(key)

    def __len__(self):
        return len(self._user_library_index)

    def __iter__(self):
        return iter(self._user_library_index)


class DictUnion(Mapping):
    """
    Union of multiple nonoverlapping dictionaries.
    This will not check for key collisions between dictionaries!

    Args:
        *dicts: The dictionaries to union.
        raise_on_missing: Whether to raise an error if a key is not found in any of the dictionaries.
    """

    def __init__(self, *dicts, raise_on_missing: bool = False):
        self._dicts = list(dicts)
        self._raise_on_missing = raise_on_missing

    def __contains__(self, key):
        return any(key in d for d in self._dicts)

    def get(self, key, default=None):
        for d in self._dicts:
            if key in d:
                return d[key]
        return default

    def __getitem__(self, key):
        for d in self._dicts:
            if key in d:
                return d[key]
        if self._raise_on_missing:
            raise KeyError(f"Key {key} not found in any of the dictionaries")
        return None

    def __len__(self):
        return sum(len(d) for d in self._dicts)

    def __iter__(self):
        return chain.from_iterable(self._dicts)


def get_db():
    global _DB

    lmb_dir = get_metadata_lmdb_dir()
    if lmb_dir is None:
        return None

    if _DB is None:
        if not PickleLMDBMap.database_exists(lmb_dir):
            lmb_dir.mkdir(parents=True, exist_ok=True)
            # Timeout so a poisoned shared-fs lock fails fast (raises) instead of hanging forever -- this
            # LMDB should be pre-built offline, and the lock only guards the rare one-time build. Kept on
            # the shared filesystem, not node-local: two nodes building the same LMDB concurrently corrupt it.
            with FileLock(str(lmb_dir / ".creation_lock"), timeout=1800):
                if not PickleLMDBMap.database_exists(lmb_dir):
                    # Ensure metadata are installed
                    get_resource_manager()

                    clip = _get_clip_features()
                    annotation = get_annotation()

                    assert len(clip["uids"]) == len(annotation), (
                        f"# clip entries {len(clip['uids'])} != # annotation entries {len(annotation)}"
                    )

                    # First, combine annotation and features under a single dict
                    combined = {}
                    for it, uid in tqdm(enumerate(clip["uids"]), "Aggregating metadata for LMDB"):
                        combined[uid] = annotation[uid]
                        combined[uid]["clip_img_features"] = clip["img_features"][it].copy()
                        combined[uid]["clip_text_features"] = clip["text_features"][it].copy()

                    clip.clear()
                    annotation.clear()

                    PickleLMDBMap.from_dict(combined, lmb_dir)

        _DB = PickleLMDBMap(lmb_dir)

    dicts = [_DB]
    for user_library_dir in USER_ASSET_LIBRARIES.values():
        dicts.append(_get_user_library_metadata(user_library_dir))
    return DictUnion(*dicts)


@cache
def _get_user_library_metadata(user_library_path: Path) -> "UserLibraryMetadata":
    return UserLibraryMetadata(user_library_path, get_user_library_index(user_library_path))


def compute_text_clip(text_list: str | list[str] | list[list[str]]):
    import torch

    if isinstance(text_list, str):
        text_list = [text_list]
    elif isinstance(text_list, list) and isinstance(text_list[0], list):
        text_list = sum(text_list, [])

    clip = get_clip_model()
    with torch.no_grad():
        return (
            clip["model"]
            .encode_text(clip["tokenizer"](text_list).to(DEFAULT_DEVICE))
            .cpu()
            .numpy()
            .astype("float16")
        )


def clip_sim(clip_img, clip_text, normalize=True, num_views=3):
    assert clip_img.shape[-2] == num_views, f"expected {num_views} feature vectors for img modality"

    if normalize:
        clip_text = clip_text / np.linalg.norm(clip_text, axis=-1, keepdims=True)
        clip_img = clip_img / np.linalg.norm(clip_img, axis=-1, keepdims=True)

    if clip_img.ndim == 2:
        clip_img = clip_img[None, ...]

    if clip_text.ndim == 1:
        clip_text = clip_text[None, :]

    assert clip_img.ndim == 3 and clip_text.ndim == 2, (
        f"Received clip_img shape {clip_img.shape} and clip_text shape {clip_text.shape}"
    )

    def func(i):
        return np.einsum("od,qd->oq", clip_img[:, i, :], clip_text)

    # We store 2 x o x q similarity scores, instead of worst case 3 x o x q - irrelevant
    res = func(0)
    for i in range(1, num_views):
        res = np.maximum(res, func(i))

    return res


class ObjectMeta:
    @staticmethod
    def all_uids() -> list[str]:
        return list(get_db().keys())

    @staticmethod
    def annotation(asset_ids: str | list[str] | None = None) -> list[dict | None] | dict | None:
        container = get_db()

        if asset_ids is None:
            return container

        if isinstance(asset_ids, str):
            return container.get(asset_ids, None)

        return [container[asset_id] for asset_id in asset_ids]

    @classmethod
    def get_short_description(cls, object_uid):
        # TODO ported from PromptSampler late at night, might need clean-up
        short_descriptions = cls.short_descriptions(object_uid)
        if len(short_descriptions) == 0:
            return [object_uid.split("_")[0]] * 4
        return short_descriptions

    @staticmethod
    def get_target_object_uid(task) -> str:
        # TODO ported from PromptSampler late at night, might need clean-up
        scene_metadata = task.env.current_scene_metadata["objects"]
        pickup_obj_name = task.config.task_config.pickup_obj_name
        object_metadata = scene_metadata[pickup_obj_name]
        asset_uid = object_metadata["asset_id"]
        return asset_uid

    @classmethod
    def clean_object_name(cls, task: "BaseMujocoTask") -> str:
        # TODO ported from PromptSampler late at night, might need clean-up
        pickup_obj_name = task.config.task_config.pickup_obj_name

        # Check if this is a custom object with a provided name
        if pickup_obj_name.startswith("custom_object/"):
            eval_params = task.config.eval_runtime_params
            if eval_params and eval_params.custom_object_name:
                # Return the provided custom object name directly
                return eval_params.custom_object_name.capitalize()

        return cls.get_short_description(cls.get_target_object_uid(task))[0]

    @classmethod
    def short_descriptions(cls, asset_ids: str | list[str]):
        def func(asset_id):
            return list(cls.annotation(asset_id)["description_short"].values())

        if isinstance(asset_ids, str):
            if asset_ids in cls.annotation():
                return func(asset_ids)
            return []

        return [func(asset_id) for asset_id in asset_ids]

    @classmethod
    def all_descriptions(cls, asset_ids: str | list[str]):
        def func(asset_id):
            anno = cls.annotation(asset_id)
            return cls.short_descriptions(asset_id) + [
                anno["description"],
                anno["description_long"],
            ]

        if isinstance(asset_ids, str):
            return func(asset_ids)

        return [func(asset_id) for asset_id in asset_ids]

    @classmethod
    def get_features(cls, feature_type_str: str, asset_ids: str | list[str] | None = None):
        db = get_db()

        if asset_ids is None:
            return np.stack([item[feature_type_str] for item in db.values()], axis=0)

        if isinstance(asset_ids, str):
            if asset_ids in db:
                return np.array([db[asset_ids][feature_type_str]])
            raise KeyError(f"Missing {asset_ids} for {feature_type_str}")

        if any(asset_id not in db for asset_id in asset_ids):
            raise KeyError(f"Missing some of {len(asset_ids)} asset_ids for {feature_type_str}")

        return np.stack([db[asset_id][feature_type_str] for asset_id in asset_ids], axis=0)

    @classmethod
    def img_features(cls, asset_ids: str | list[str] | None = None):
        return cls.get_features("clip_img_features", asset_ids)

    @classmethod
    def description_text_features(cls, asset_ids: str | list[str] | None = None):
        return cls.get_features("clip_text_features", asset_ids)


if __name__ == "__main__":
    texts = [
        "a tree",
        "a watch",
        "a tower clock",
        "a clock",
        "an alarm clock",
        "a blue alarm clock",
    ]

    print(texts)

    text_clips = compute_text_clip(texts)
    assert text_clips.shape == (len(texts), 768)

    asset_id = "Alarm_Clock_1"

    for text, text_clip in zip(texts, text_clips):
        print(text, clip_sim(ObjectMeta.img_features(asset_id), text_clip).item())

    all_sims = clip_sim(ObjectMeta.img_features(asset_id), text_clips).squeeze()
    print("All user", all_sims, all_sims.dtype)

    all_descs = ObjectMeta.all_descriptions(asset_id)
    print(all_descs)
    all_sims = clip_sim(ObjectMeta.img_features(asset_id), compute_text_clip(all_descs)).squeeze()
    print("All any", all_sims, all_sims.dtype)

    description = ObjectMeta.description_text_features(asset_id)
    all_sims = clip_sim(ObjectMeta.img_features(asset_id), description).squeeze()
    print("Description", all_sims, all_sims.dtype)

    asset_ids = ["Alarm_Clock_1", "Wall_Decor_Photo_9"]
    img = ObjectMeta.img_features(asset_ids)

    descriptions = list(ObjectMeta.annotation(asset_id)["description"] for asset_id in asset_ids)
    print("\n".join(descriptions))
    print("Descriptions", clip_sim(img, ObjectMeta.description_text_features(asset_ids)))
    print("Recomputed", clip_sim(img, compute_text_clip(descriptions)))

    all_descriptions = ObjectMeta.all_descriptions(asset_ids)
    print("\n".join(sum(all_descriptions, [])))
    print("Recomputed", clip_sim(img, compute_text_clip(all_descriptions)))
