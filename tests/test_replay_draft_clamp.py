from __future__ import annotations

from importlib.metadata import version

import pytest
from packaging.version import Version

from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target, random_prompt

pytestmark = pytest.mark.skipif(Version(version("transformers")) < Version("5.16.1"),
                                reason="static execution requires transformers >= 5.16.1")


@pytest.fixture
def base():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    return target


def static(base, kind):
    from dart.models.replay import CapacityBuckets, CudaGraphTarget

    if kind == "single":
        return CudaGraphTarget(base, capacity=128, max_block_tokens=5, replay=False)
    return CapacityBuckets(base, (64, 128), max_block_tokens=5, replay=False)


@pytest.mark.parametrize("kind", ["single", "buckets"])
def test_drafts_longer_than_prepared_blocks_are_clamped(base, kind):
    prompt = random_prompt(3, 6) * 4  # Repetition gives the n-gram lane long proposals.
    target = static(base, kind)
    try:
        with DartEngine(target, NGramDrafter()) as engine:
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=30, draft_len=8))
    finally:
        target.close()
    assert output.token_ids == base.greedy_generate(prompt, 30)
    assert max(step.n_draft_proposed for step in output.rounds) == 4  # Pending token plus four drafts fill a block.


def test_cost_table_arms_above_the_block_size_run_clamped(base):
    prompt = random_prompt(4, 5) * 5
    target = static(base, "single")
    scheduler = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]), draft_lengths=(2, 8),
                                   warmup_rounds=1, retry_interval=3)
    try:
        with DartEngine(target, lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
            output = engine.generate(prompt, DecodeConfig(max_new_tokens=40))
    finally:
        target.close()
    assert output.token_ids == base.greedy_generate(prompt, 40)
    assert "ngram:8" in scheduler.counts and all(step.n_draft_proposed <= 4 for step in output.rounds)
