"""YAML loading for configured robot embodiment specifications."""

from pathlib import Path

import yaml
from pydantic import ValidationError

from molmo_spaces.robots.configured.spec import EmbodimentSpec


def _merge_mappings(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_mappings(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_mapping(source_path: Path, seen: set[Path]) -> dict:
    if source_path in seen:
        raise ValueError(f"Embodiment spec inheritance cycle includes {source_path}")
    seen.add(source_path)
    try:
        with source_path.open(encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
    except OSError as error:
        raise OSError(f"Could not read embodiment spec {source_path}: {error}") from error
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML in embodiment spec {source_path}: {error}") from error

    if not isinstance(data, dict):
        raise ValueError(
            f"Embodiment spec {source_path} must contain a YAML mapping at the document root"
        )

    extends = data.pop("extends", None)
    if extends is not None:
        if not isinstance(extends, str) or not extends:
            raise ValueError(f"Embodiment spec {source_path} extends must be a nonempty path")
        parent_path = (source_path.parent / extends).resolve()
        data = _merge_mappings(_load_mapping(parent_path, seen), data)

    model = data.get("robot_config", {}).get("model", {})
    root_path = model.get("root_path")
    if model.get("source") == "path" and root_path is not None:
        root_path = Path(root_path).expanduser()
        if not root_path.is_absolute():
            model["root_path"] = source_path.parent / root_path
    seen.remove(source_path)
    return data


def load_embodiment_spec(path: str | Path) -> EmbodimentSpec:
    """Safely load, compose, and validate an :class:`EmbodimentSpec` YAML file."""
    source_path = Path(path).expanduser().resolve()
    data = _load_mapping(source_path, set())

    try:
        return EmbodimentSpec.model_validate(data)
    except ValidationError as error:
        error.add_note(f"Loaded from {source_path}")
        raise
