from __future__ import annotations

import json
from dataclasses import asdict, replace
from io import BytesIO
from types import SimpleNamespace

import pytest

from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_engine
from dart.runtime.server import make_handler
from dart.runtime.types import DecodeConfig
from tests.conftest import random_prompt


class MemorySocket:
    def __init__(self, request):
        self.request = BytesIO(request)
        self.response = BytesIO()

    def makefile(self, *args):
        return self.request

    def sendall(self, data):
        self.response.write(data)


def post(handler, endpoint, payload):
    body = json.dumps(payload).encode()
    headers = (
        f"POST {endpoint} HTTP/1.0\r\n"
        "Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n"
    ).encode()
    connection = MemorySocket(headers + body)
    handler(connection, ("127.0.0.1", 0), SimpleNamespace())
    head, _, response = connection.response.getvalue().partition(b"\r\n\r\n")
    assert head.splitlines()[0] == b"HTTP/1.0 200 OK", response.decode()
    return json.loads(response)


@pytest.mark.parametrize("endpoint", ["/v1/completions", "/v1/dart_demo"])
def test_http_requests_keep_configured_defaults(monkeypatch, endpoint):
    defaults = DecodeConfig(
        mode="exact_sampling", max_new_tokens=7, full_kv_residency="cpu_pinned",
        allowed_lanes=["target_only"], draft_len=2, canvas_len=8,
        prompt_kind="code", seed=31, temperature=0.7, target_model="configured-target",
    )
    engine = build_engine({
        "target": {"model": "tiny:1234"}, "lanes": {"ngram": {}},
        "decode": asdict(defaults),
    })
    engine.target.eos_token_ids = frozenset()
    seen = []
    generate = engine.generate

    def capture(prompt, config):
        seen.append(config)
        return generate(prompt, config)

    monkeypatch.setattr(engine, "generate", capture)
    handler = make_handler(engine)
    with engine:
        for override in ({}, {"max_tokens": 3}, {}):
            payload = {"prompt_ids": random_prompt(307, 8), **override}
            response = post(handler, endpoint, payload)
            expected = replace(defaults, max_new_tokens=override.get("max_tokens", 7))
            if endpoint == "/v1/dart_demo":
                expected.mode = "exact"  # The demo compares against greedy output.
                assert response["identical"] is True
            else:
                assert response["dart"]["mode"] == expected.mode
                assert response["dart"]["full_kv_residency"] == "cpu_pinned"
                assert response["dart"]["tokens_generated"] == expected.max_new_tokens
            assert seen[-1] == expected
            assert seen[-1] is not engine.default_decode_config
            assert engine.default_decode_config == defaults


@pytest.mark.parametrize("endpoint,budget", [("/v1/completions", 9), ("/v1/dart_demo", 64)])
def test_http_fallback_budgets_without_config(monkeypatch, tiny_target, endpoint, budget):
    seen = []
    with DartEngine(tiny_target) as engine:
        generate = engine.generate

        def capture(prompt, config):
            seen.append(config)
            return generate(prompt, config)

        monkeypatch.setattr(engine, "generate", capture)
        post(make_handler(engine, default_max_tokens=9), endpoint, {
            "prompt_ids": random_prompt(308, 8),
        })
    assert seen[0].max_new_tokens == budget
    assert seen[0].mode == "exact"
