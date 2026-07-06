"""Serve a configured DART engine over HTTP (OpenAI-compatible /v1/completions).

    python scripts/run_server.py --config configs/target_qwen.yaml --port 8400

Requires a target with a tokenizer (i.e. an hf: target, not tiny:).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dart.runtime.factory import engine_from_yaml
from dart.runtime.server import serve


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8400)
    args = ap.parse_args()
    serve(engine_from_yaml(args.config), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
