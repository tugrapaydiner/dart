from __future__ import annotations

from types import SimpleNamespace

import pytest

from dart.runtime import server
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig, GenerationOutput
from tests.test_server import post


@pytest.mark.parametrize("device", ["cpu", "cuda:1"])
def test_demo_times_decoding_and_waits_for_device_work(monkeypatch, device):
    clock = [0.0]
    pending = [0.1 if device.startswith("cuda") else 0.0]
    waits = []
    tokens = [7, 2]

    def work(seconds):
        if device.startswith("cuda"):
            pending[0] += seconds
        else:
            clock[0] += seconds

    def synchronize(selected):
        waits.append(str(selected))
        clock[0] += pending[0]
        pending[0] = 0.0

    def decode(ids, skip_special_tokens=False):
        clock[0] += 0.003
        return "7 2"

    def greedy(prompt, max_new_tokens):
        work(0.01)
        return tokens

    def generate(prompt, config):
        work(0.02)
        return GenerationOutput(tokens, decode(tokens), {
            "output_hash": output_hash(tokens), "acceptance_rate": 0.5,
            "avg_accept_run": 1.0, "lanes_used": {"ngram": 1},
        })

    monkeypatch.setattr(server.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr("torch.cuda.synchronize", synchronize)
    target = SimpleNamespace(device=device, tokenizer=SimpleNamespace(decode=decode),
                             greedy_generate=greedy, eos_token_ids=frozenset({2}))
    engine = SimpleNamespace(target=target, generate=generate, default_decode_config=DecodeConfig(max_new_tokens=2))
    response = post(server.make_handler(engine), "/v1/dart_demo", {"prompt_ids": [3, 4]})
    assert response["baseline"]["wall_ms"] == 13
    assert response["dart"]["wall_ms"] == 23
    assert waits == (["cuda:1"] * 3 if device.startswith("cuda") else [])
    assert pending[0] == 0
    assert response["identical"] is True
    assert response["baseline"]["tokens"] == response["dart"]["tokens"] == 2
    assert response["reference"] == "DART target greedy"
    assert response["baseline"]["output_hash"] == response["dart"]["output_hash"] == output_hash(tokens)
    assert "single" in response["timing_scope"] and "decoding" in response["timing_scope"]
