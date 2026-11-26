"""Closed-loop rollout server — the env side of the RolloutSession protocol.

molmospaces hosts the sim + FK and serves episodes over gRPC (see ``proto/rollout.proto``); an external
client (e.g. virtual_vla) sends a replay bundle and drives the loop with its own policy. Nothing here knows
about the policy. ``episode.reconstruct_task_exact`` rebuilds the bundle's EXACT scene + RNG, and
``replay_session.EmbodimentReplay`` (base ``session.EmbodimentRollout``: obs assembly + action inversion)
steps it. ``server.serve(replay_session_from_bundle)`` runs it."""
from .episode import load_meta, reconstruct_task_exact
from .replay_session import EmbodimentReplay, replay_session_from_bundle
from .server import RolloutServicer, serve
from .session import EmbodimentRollout

__all__ = [
    "reconstruct_task_exact", "load_meta",
    "EmbodimentRollout", "EmbodimentReplay", "replay_session_from_bundle",
    "RolloutServicer", "serve",
]
