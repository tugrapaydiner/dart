"""Shared fixtures: a tiny random-weight Llama-style target model.

Exactness tests do not need trained weights: the contract is
"DART output == target greedy output" for *whatever* the target computes,
so a seeded random model is a perfect (and fast, download-free) test
subject.  Trained models only matter for acceptance-rate/speed numbers.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.target import TargetModel

VOCAB = 211
EOS_ID = 2
FIRST_SAFE_ID = 5  # prompt tokens drawn from [FIRST_SAFE_ID, VOCAB) avoid specials


def build_tiny_target(seed: int = 1234) -> TargetModel:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=EOS_ID,
        pad_token_id=0,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg)
    return TargetModel(model, tokenizer=None, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="session")
def tiny_target() -> TargetModel:
    return build_tiny_target()


def random_prompt(seed: int, length: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        FIRST_SAFE_ID, VOCAB, (length,), generator=g
    ).tolist()


@pytest.fixture
def study(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from benchmarks import execution_analysis as analysis
    from benchmarks import execution_protocol as protocol
    from dart.data.corpus import read_records

    rows = read_records(analysis.PROMPTS)
    monkeypatch.setattr(protocol, "encode_prompt", lambda *args: [7, 8, 9])
    monkeypatch.setattr(protocol, "tokenizer_hash", lambda tokenizer: "a" * 64)
    monkeypatch.setattr(protocol, "model_hashes", lambda model: dict.fromkeys(protocol.MODEL_FILES, "b" * 64))
    names = ("benchmarks/execution_protocol.py", "benchmarks/execution_bench.py", "benchmarks/hf_replay.py",
             "dart/models/replay.py", "dart/runtime/engine.py")
    monkeypatch.setattr(protocol, "source_hashes", lambda: dict.fromkeys(names, "c" * 64))
    monkeypatch.setattr(protocol, "version", lambda name: "1.0")

    def build(split="test", candidate=(50, 50, 50)):
        plan = protocol.make_protocol(rows, SimpleNamespace(eos_token_id=2), 1024, tmp_path, split=split)
        reports, processes = [], []

        def output(latency):
            return {"token_ids": [7, 8, 2], "output_hash": protocol.digest([7, 8, 2]),
                    "latency_ms": latency, "eos": True, "early_eos": True}

        for session in range(plan["sessions"]):
            checks = protocol.session_checks(plan, session)
            report = {"schema": "dart-execution-session-v1", "status": "complete", "session": session,
                      "protocol_hash": plan["protocol_hash"], "pid": 100 + session, "parent_pid": 1,
                      "started_at": f"2026-01-01T00:{session:02d}:00+00:00", "elapsed_seconds": 50,
                      "eos_token_ids": [2], "vocab_size": 32, "peak_allocated_bytes": 1000,
                      "output_hash_schema": "canonical-json-sha256", "target_parameters": 1000,
                      "extra_learned_parameters": 0, "resident_model_copies": 2,
                      "memory_scope": plan["memory_scope"], "setup_scope": "synthetic test data",
                      "setup_ms": {"reference_model": 1, "native_model": 1, "replay": 10, "static": 1},
                      "capture_and_warmup_ms": 9,
                      "environment": {"python": "3.12", "system": "Synthetic", "torch": "1.0", "cuda": "12.0",
                                      "device": "Synthetic", "device_memory_bytes": 10_000, "threads": 1, "matmul_precision": "highest"},
                      "references": {key: output(100) for key in checks},
                      "checks": [{"case": key, "match": True, **output(candidate[session])} for key in checks],
                      "warmups": [], "samples": []}
            means = dict.fromkeys(protocol.METHODS, 80)
            means.update(current_dart=100, static_dart=120, replay_dart=candidate[session])
            for call in protocol.schedule(plan, session):
                key = "samples" if call["phase"] == "timed" else "warmups"
                report[key].append({**call, "match": True, **output(means[call["method"]])})
            reports.append(report)
            processes.append({"session": session, "pid": report["pid"], "worker_pid": report["pid"],
                              "returncode": 0, "timed_out": False, "elapsed_seconds": 51})
        status = {"schema": "dart-execution-run-v1", "status": "complete", "protocol_hash": plan["protocol_hash"],
                  "timeout_seconds": plan["timeout_seconds"], "processes": processes, "elapsed_seconds": 52 * len(reports)}
        return plan, status, reports

    return build

