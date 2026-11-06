from __future__ import annotations

import json

import pytest

from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import VOCAB, build_tiny_target, random_prompt
from tests.test_exactness import OracleDrafter


class NoisyOracle(OracleDrafter):
    """Correct drafts except every third round, so skipped rounds can diverge."""

    def __init__(self, target):
        super().__init__(target)
        self.calls = 0

    def propose(self, committed, max_tokens):
        drafts = list(super().propose(committed, max_tokens))
        self.calls += 1
        if drafts and self.calls % 3 == 0:
            drafts[-1] = (drafts[-1] + 1) % VOCAB
        return drafts


@pytest.mark.parametrize("mode, verify_every", [("exact", 1), ("approx", 1), ("approx", 2), ("approx", 3)])
def test_trace_records_every_round_and_committed_token(tmp_path, mode, verify_every):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    path = tmp_path / "trace.jsonl"
    with DartEngine(target, NoisyOracle(target), trace_path=str(path)) as engine:
        output = engine.generate(random_prompt(71, 6), DecodeConfig(mode=mode, max_new_tokens=30, draft_len=3,
                                                                    verify_every=verify_every))
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == len(output.rounds)
    assert [record["round_idx"] for record in records] == [step.round_idx for step in output.rounds]
    assert [token for record in records for token in record["committed_ids"]] == output.token_ids[1:]
    assert [record["n_committed"] for record in records] == [len(record["committed_ids"]) for record in records]
    skipped = [record for record in records if record["verifier"] == "none"]
    assert bool(skipped) == (mode == "approx" and verify_every > 1)
