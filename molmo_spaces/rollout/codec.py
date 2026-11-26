"""Pack/unpack tensors, observations, and info dicts across the rollout gRPC wire.

Mirror of the client-side codec (virtual_vla): an observation is a schema-free map of named dense
tensors (+ node_names + prompt), so the server owns which keys it fills and the client reads what it
needs."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .proto import rollout_pb2


def tensor_to_proto(t: torch.Tensor) -> rollout_pb2.Tensor:
    arr = np.ascontiguousarray(torch.as_tensor(t).detach().cpu().numpy())
    return rollout_pb2.Tensor(shape=list(arr.shape), dtype=str(arr.dtype), data=arr.tobytes())


def proto_to_tensor(p: rollout_pb2.Tensor) -> torch.Tensor:
    arr = np.frombuffer(p.data, dtype=np.dtype(p.dtype)).reshape(tuple(p.shape))
    return torch.from_numpy(arr.copy())          # frombuffer is read-only; copy so torch owns it


def obs_to_proto(obs: dict[str, Any]) -> rollout_pb2.Observation:
    tensors, node_names, camera_names, prompt = {}, [], [], ""
    for key, value in obs.items():
        if key == "prompt":
            prompt = value or ""
        elif key == "node_names":
            node_names = list(value)
        elif key == "camera_names":
            camera_names = list(value)
        elif value is not None:
            tensors[key] = tensor_to_proto(value)
    return rollout_pb2.Observation(
        tensors=tensors, node_names=node_names, camera_names=camera_names, prompt=prompt)


def proto_to_obs(p: rollout_pb2.Observation) -> dict[str, Any]:
    obs: dict[str, Any] = {k: proto_to_tensor(v) for k, v in p.tensors.items()}
    if p.node_names:
        obs["node_names"] = list(p.node_names)
    if p.camera_names:
        obs["camera_names"] = list(p.camera_names)
    obs["prompt"] = p.prompt
    return obs


def info_to_proto(info: dict[str, Any]) -> dict[str, rollout_pb2.Value]:
    out = {}
    for key, value in info.items():
        cell = rollout_pb2.Value()
        if isinstance(value, bool):              # bool before int (bool is an int subclass)
            cell.b = value
        elif isinstance(value, int):
            cell.i = value
        elif isinstance(value, float):
            cell.f = value
        else:
            cell.s = str(value)
        out[key] = cell
    return out


def proto_to_info(mapping) -> dict[str, Any]:
    out = {}
    for key, cell in mapping.items():
        which = cell.WhichOneof("v")
        out[key] = getattr(cell, which) if which else None
    return out
