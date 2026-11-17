from __future__ import annotations

from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

from dart.runtime.types import DecodeConfig
from dart.verification import synthetic


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def outputs(monkeypatch):
    seen = {}

    def decoder(engine):
        def decode(case):
            result = engine.generate(list(case.prompt), DecodeConfig(max_new_tokens=case.max_new_tokens,
                                                                     draft_len=case.draft_len))
            seen.setdefault(id(engine), []).append(result)
            return result.token_ids
        return decode

    monkeypatch.setattr(synthetic, "engine_decoder", decoder)
    return seen


@pytest.mark.parametrize("seed", [17, 29])
def test_adaptive_candidate_switches_lengths_and_verifiers(outputs, seed):
    report = synthetic.run_synthetic_check(seed)
    assert report.passed, report.failures
    assert {row.backend for row in report.results} >= {"adaptive"}
    adaptive = [result for results in outputs.values() for result in results
                if any(step.lane == "target_only" for step in result.rounds)
                and any(step.lane == "synthetic" for step in result.rounds)]
    rounds = [step for result in adaptive for step in result.rounds if step.lane == "synthetic"]
    assert {step.verifier for step in rounds} == {"block", "tree"}
    assert {step.n_draft_proposed for step in rounds} >= {2, 4}
    assert any(step.n_draft_accepted > 0 for step in rounds)


@pytest.mark.skipif(Version(version("transformers")) < Version("5.16.1"),
                    reason="static execution requires transformers >= 5.16.1")
def test_static_candidates_cover_both_buckets_and_adaptive_block_rounds(outputs):
    report = synthetic.run_synthetic_check(execution="static")
    assert report.passed, report.failures
    capacities = {result.metadata.get("cache_capacity") for results in outputs.values() for result in results
                  if result.token_ids}
    assert {16, synthetic.CONTEXT_LENGTH} <= capacities
    verifiers = {step.verifier for results in outputs.values() for result in results for step in result.rounds}
    assert verifiers == {"block"}
