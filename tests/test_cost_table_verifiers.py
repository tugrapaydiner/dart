from __future__ import annotations

from importlib.metadata import version

import pytest
from packaging.version import Version

from dart.drafters.ngram import NGramDrafter
from dart.runtime import factory
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler
from dart.runtime.types import DecodeConfig, DecodeState, VerifyResult
from tests.conftest import build_tiny_target, random_prompt


def state():
    return DecodeState(16, 4, float("inf"), 1, "code", True, 0.0, {}, 0.0, 0.0)


def result(committed):
    return VerifyResult(committed=[7] * committed, n_draft_proposed=committed - 1,
                        n_draft_accepted=committed - 1, all_accepted=True, finished=False)


def planner(**options):
    config = SchedulerConfig(available_lanes=["ngram"], ewma_alpha=1.0)
    return CostTableScheduler(config, draft_lengths=(2, 4), verifiers=("block", "tree"), warmup_rounds=1,
                              retry_interval=50, **options)


def test_arms_pair_each_length_with_each_verifier():
    timings = {"target_only": (10.0, 1), "ngram:2:block": (12.0, 2), "ngram:2:tree": (13.0, 2),
               "ngram:4:block": (14.0, 3), "ngram:4:tree": (15.0, 5)}
    scheduler, chosen = planner(), []
    for _ in range(8):
        action = scheduler.choose_action(state())
        key = "target_only" if action.lane == "target_only" else f"{action.lane}:{action.draft_len}:{action.verifier}"
        chosen.append(key)
        ms, committed = timings[key]
        scheduler.observe(action.lane, result(committed), round_ms=ms)
    assert chosen[:5] == list(timings)
    assert chosen[5:] == ["ngram:4:tree"] * 3  # 3 ms per token is the cheapest arm.
    assert set(scheduler.counts) == set(timings)


def test_without_a_verifier_menu_actions_keep_the_engine_verifier():
    scheduler = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]), draft_lengths=(2,), warmup_rounds=1)
    scheduler.choose_action(state())  # Warm-up target step.
    scheduler.observe("target_only", result(1), round_ms=10.0)
    action = scheduler.choose_action(state())
    assert (action.lane, action.draft_len, action.verifier) == ("ngram", 2, None)


@pytest.mark.parametrize("verifiers", [(), ("block", "block"), ["tree"], ("beam",), ("reference",)])
def test_rejects_invalid_verifier_menus(verifiers):
    with pytest.raises(ValueError, match="verifiers"):
        CostTableScheduler(SchedulerConfig(), verifiers=verifiers)


def test_engine_mixes_block_and_tree_arms_and_matches_greedy():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(61, 7) * 4
    scheduler = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]), draft_lengths=(2, 4),
                                   verifiers=("block", "tree"), warmup_rounds=1, retry_interval=5)
    with DartEngine(target, lanes={"ngram": NGramDrafter(tree_width=3)}, scheduler=scheduler) as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=40))
    assert output.token_ids == target.greedy_generate(prompt, 40)
    assert {"ngram:2:block", "ngram:2:tree", "ngram:4:block", "ngram:4:tree"} <= set(scheduler.counts)
    assert {step.verifier for step in output.rounds} == {"block", "tree"}


@pytest.mark.skipif(Version(version("transformers")) < Version("5.16.1"),
                    reason="static execution requires transformers >= 5.16.1")
def test_replay_targets_reject_tree_arms_before_execution(monkeypatch):
    from dart.models.replay import CudaGraphTarget

    replay = CudaGraphTarget(build_tiny_target(), capacity=64, replay=False)
    calls = []
    monkeypatch.setattr(replay, "prefill", lambda *a, **k: calls.append(1))
    scheduler = CostTableScheduler(SchedulerConfig(available_lanes=["ngram"]), verifiers=("block", "tree"))
    try:
        with DartEngine(replay, lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
            with pytest.raises(ValueError, match="block verification only"):
                engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=4))
        assert calls == []
    finally:
        replay.close()


def test_factory_reads_the_verifier_menu():
    engine = factory.build_engine({"target": {"model": "tiny:0"}, "lanes": {"ngram": {}},
                                   "scheduler": {"type": "cost_table", "draft_lengths": [2], "verifiers": ["block", "tree"]}})
    with engine:
        assert engine.scheduler.verifiers == ("block", "tree")
        engine.generate([5, 6, 7, 5, 6, 7], DecodeConfig(max_new_tokens=6))
