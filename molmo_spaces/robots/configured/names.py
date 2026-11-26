"""Namespace-aware MuJoCo name resolution for configured robots."""

from __future__ import annotations

from typing import Literal

from mujoco import MjModel

from molmo_spaces.robots.configured.spec import FrameRef, ModelNameRef


def resolve_model_name(reference: ModelNameRef, namespace: str) -> str:
    """Resolve a model reference against a compiled robot namespace."""
    return reference.name if reference.absolute else f"{namespace}{reference.name}"


def resolve_model_id(
    model: MjModel,
    object_type: Literal["actuator", "body", "geom", "joint", "site"],
    reference: ModelNameRef,
    namespace: str,
) -> int:
    """Resolve a reference and fail with a useful error when it is absent."""
    name = resolve_model_name(reference, namespace)
    try:
        return int(getattr(model, object_type)(name).id)
    except KeyError as error:
        raise ValueError(f"Configured robot {object_type} {name!r} was not found") from error


def resolve_frame_id(model: MjModel, frame: FrameRef, namespace: str) -> int:
    """Resolve a configured body or site frame."""
    return resolve_model_id(model, frame.kind, frame.name, namespace)
