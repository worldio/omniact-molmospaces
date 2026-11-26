"""gRPC server hosting ``EmbodimentRollout`` sessions behind the Rollout service.

``Open`` builds a session (an embodiment + task -> a set of eval episodes); ``Reset``/``Step``/``Close``
drive it through the shared codec. The service is embodiment-generic -- it knows nothing about the policy
or its consumer; the client (e.g. virtual_vla) drives the loop and runs the policy locally.

One env per process: the sim is single-env and not thread-safe, so a process hosts one session at a time
(``max_workers=1`` serializes RPCs). Parallelize with multiple PROCESSES, each serving a slice of
episodes, not threads."""
from __future__ import annotations

import uuid
from concurrent import futures
from typing import Any, Callable

import grpc

from . import codec
from .episode import BlacklistedBodiesInScene
from .proto import rollout_pb2, rollout_pb2_grpc
from ..tasks.task_sampler_errors import HouseInvalidForTask

# A scene the server cannot rebuild for this recording. The request is well-formed and the server is
# healthy, so this is neither INVALID_ARGUMENT nor a transport failure: it is FAILED_PRECONDITION, which
# lets a client skip the episode and keep its panel running instead of dying on an UNKNOWN.
UNRECONSTRUCTABLE = (BlacklistedBodiesInScene, HouseInvalidForTask)

# session_factory(bundle, options) -> a RolloutSession-shaped object:
#   n_episodes: int; reset(idx) -> obs dict; step(action, tcp_pose) -> (obs, done, info); close().
# bundle: {relative_path: bytes} -- the whole replay bundle, sent by the client; options: {str: str}.
SessionFactory = Callable[[dict, dict], Any]

# Bundles (states.npz etc.) ride in the Open request; lift gRPC's 4 MB default well above bundle size.
_MAX_MSG = 512 * 1024 * 1024


class RolloutServicer(rollout_pb2_grpc.RolloutServicer):
    def __init__(self, session_factory: SessionFactory) -> None:
        self._factory = session_factory
        self._sessions: dict[str, Any] = {}

    def Open(self, request, context):
        try:
            session = self._factory(dict(request.bundle), dict(request.options))
        except UNRECONSTRUCTABLE as error:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        session_id = uuid.uuid4().hex
        self._sessions[session_id] = session
        return rollout_pb2.Session(session_id=session_id, n_episodes=int(session.n_episodes))

    def Reset(self, request, context):
        try:
            obs = self._session(request.session_id, context).reset(int(request.episode_idx))
        except UNRECONSTRUCTABLE as error:   # the scene is rebuilt lazily, so this surfaces here
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(error))
        return codec.obs_to_proto(obs)

    def Step(self, request, context):
        required = ("action", "node_action_pose", "action_mask", "node_action_pose_mask")
        missing = [f for f in required if not request.HasField(f)]
        if missing:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT,
                          f"StepRequest is missing required field(s): {', '.join(missing)} "
                          f"(a head you do not command is zeros with an all-False mask)")
        args = [codec.proto_to_tensor(getattr(request, f)) for f in required]
        try:
            obs, done, info = self._session(request.session_id, context).step(*args)
        except ValueError as error:                       # masks command nothing / a row the server cannot run
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(error))
        return rollout_pb2.StepResult(
            obs=codec.obs_to_proto(obs), done=bool(done), info=codec.info_to_proto(info)
        )

    def Close(self, request, context):
        session = self._sessions.pop(request.session_id, None)
        if session is not None:
            session.close()
        return rollout_pb2.Empty()

    def _session(self, session_id: str, context):
        session = self._sessions.get(session_id)
        if session is None:
            context.abort(grpc.StatusCode.NOT_FOUND, f"unknown session {session_id!r}")
        return session


def serve(session_factory: SessionFactory, *, port: int = 50051, max_workers: int = 1) -> grpc.Server:
    """Start (non-blocking) a Rollout server; returns the started ``grpc.Server`` (call ``wait_for_termination``)."""
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=max_workers),
        options=[("grpc.max_receive_message_length", _MAX_MSG),
                 ("grpc.max_send_message_length", _MAX_MSG)],
    )
    rollout_pb2_grpc.add_RolloutServicer_to_server(RolloutServicer(session_factory), server)
    server.add_insecure_port(f"[::]:{port}")
    server.start()
    return server
