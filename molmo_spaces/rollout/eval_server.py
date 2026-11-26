"""CLI: host the exact-replay eval server.

    python -m molmo_spaces.rollout.eval_server --port 50051

Serves the Rollout gRPC service (Open/Reset/Step/Close). Each Open receives the whole replay bundle as bytes
(the client sends it; the server needs no shared filesystem or path -- only its MLSPACES asset corpus +
blacklist), writes it to a temp dir, builds an ``EmbodimentReplay``, reconstructs the datagen episode
(``replay_episode``'s recipe method -> exact model + initial state, RNG at the datagen first-step state) and
PHYSICS-STEPS it, so ground-truth actions reproduce the recorded trajectory + frames bit-exactly on the
datagen CPU class (AMD) + osmesa. One env per process (``max_workers=1``); parallelize with PROCESSES.
"""
from __future__ import annotations

import argparse

from .replay_session import replay_session_from_bundle
from .server import serve


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--port", type=int, default=50051)
    parser.add_argument("--max-workers", type=int, default=1,
                        help="RPC threads; the sim is single-env, so keep 1 and scale with processes")
    args = parser.parse_args(argv)

    server = serve(replay_session_from_bundle, port=args.port, max_workers=args.max_workers)
    print(f"[eval_server] Rollout service on :{args.port} (exact-replay sessions); Ctrl-C to stop", flush=True)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
