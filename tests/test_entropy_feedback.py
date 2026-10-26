from functools import wraps
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from dart.drafters.base import NullDrafter
from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.time_scheduler import TimeScheduler
from dart.runtime.types import DecodeConfig
from dart.verification.block_verify import verify_block_greedy
from tests.conftest import build_tiny_target, random_prompt


@pytest.mark.parametrize("pending,drafts,eos", [
    ([4], [], set()),
    ([4], [7, 7], set()),
    ([4], [1, 7], set()),
    ([4], [1, 2], set()),
    ([4], [1, 2], {2}),
    ([4], [1, 2], {3}),
    ([4, 6], [2, 7], set()),
])
def test_optional_entropy_keeps_decisions_and_cache(pending, drafts, eos):
    class Target:
        eos_token_ids = eos

        def forward_block(self, state, tokens):
            state.seq_len += len(tokens)
            return torch.eye(8)[1:len(tokens) + 1] * 4

        def crop(self, state, length):
            state.seq_len = length

    target = Target()
    enabled_state = SimpleNamespace(seq_len=9)
    disabled_state = SimpleNamespace(seq_len=9)
    expected = verify_block_greedy(target, enabled_state, pending, drafts)
    with patch.object(torch, "softmax", side_effect=AssertionError("unused entropy")):
        actual = verify_block_greedy(target, disabled_state, pending, drafts, compute_entropy=False)
    assert actual == expected
    assert actual.hidden_index == expected.hidden_index
    assert disabled_state.seq_len == enabled_state.seq_len
    assert actual.entropy == 0.0
    if not (actual.finished and actual.n_draft_accepted == len(actual.committed)):
        assert expected.entropy > 0


@pytest.mark.parametrize("drafter", [NullDrafter, NGramDrafter])
@pytest.mark.parametrize("budget", [1, 16])
def test_fixed_lanes_skip_unused_entropy(drafter, budget):
    target = build_tiny_target()
    prompt = random_prompt(411, 12)
    expected = target.greedy_generate(prompt, budget)
    with DartEngine(target, drafter()) as engine:
        with patch.object(torch, "softmax", side_effect=AssertionError("unused entropy")):
            actual = engine.generate(prompt, DecodeConfig(max_new_tokens=budget, draft_len=4))
    assert actual.token_ids == expected
    assert actual.metadata["greedy_verified"] is True


class RecordingDrafter:
    name = "recording"

    def __init__(self):
        self.entropies = []

    def propose(self, committed, max_tokens):
        return []

    def observe(self, result):
        self.entropies.append(result.entropy)


def test_custom_feedback_defaults_to_entropy_and_can_opt_out():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafter = RecordingDrafter()
    prompt = random_prompt(412, 10)
    expected = target.greedy_generate(prompt, 8)
    with DartEngine(target, drafter) as engine:
        for flag in (None, False, True, False):
            if flag is not None:
                drafter.requires_entropy = flag
            drafter.entropies.clear()
            actual = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
            assert actual.token_ids == expected
            assert len(drafter.entropies) == 7
            assert all((value > 0) == (flag is not False) for value in drafter.entropies)


def test_scheduler_receives_prefill_entropy_and_live_feedback():
    class Scheduler(HeuristicScheduler):
        def reset(self):
            super().reset()
            self.states, self.entropies = [], []

        def choose_action(self, state, allowed=None):
            self.states.append(state.entropy_ewma)
            return super().choose_action(state, allowed)

        def observe(self, lane, result, **timing):
            self.entropies.append(result.entropy)
            return super().observe(lane, result, **timing)

    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(413, 10)
    expected = target.greedy_generate(prompt, 8)
    _, logits = target.prefill(prompt)
    probs = logits.float().softmax(-1)
    entropy = float(-(probs * probs.clamp_min(1e-12).log()).sum())
    scheduler = Scheduler(SchedulerConfig(available_lanes=["ngram"], entropy_gate_threshold=0.0))
    with DartEngine(target, lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
        for _ in range(2):
            actual = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
            assert actual.token_ids == expected
            assert scheduler.states[0] == pytest.approx(entropy)
            assert all(value > 0 for value in scheduler.entropies)
            assert scheduler.states[1] == pytest.approx(0.7 * entropy + 0.3 * scheduler.entropies[0])


def test_time_scheduler_only_requests_entropy_for_an_active_gate():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(415, 10)
    expected = target.greedy_generate(prompt, 8)
    config = SchedulerConfig(available_lanes=["ngram", "target_only"])
    scheduler = TimeScheduler(config)
    with DartEngine(target, lanes={"ngram": NGramDrafter(), "target_only": NullDrafter()}, scheduler=scheduler) as engine:
        for threshold in (float("inf"), 0.0, float("inf")):
            config.entropy_gate_threshold = threshold
            with patch.object(torch, "softmax", wraps=torch.softmax) as entropy:
                actual = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
            assert actual.token_ids == expected
            assert (entropy.call_count > 0) == (threshold == 0)


def test_entropy_consumer_still_gets_feedback_with_a_time_scheduler():
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafter = RecordingDrafter()
    scheduler = TimeScheduler(SchedulerConfig(available_lanes=[drafter.name]))
    prompt = random_prompt(416, 10)
    with DartEngine(target, lanes={drafter.name: drafter}, scheduler=scheduler) as engine:
        actual = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
    assert actual.token_ids == target.greedy_generate(prompt, 8)
    assert drafter.entropies and all(value > 0 for value in drafter.entropies)


@pytest.mark.parametrize("wrapped", [False, True])
def test_verifier_instrumentation_keeps_its_call_signature(wrapped):
    target = build_tiny_target()
    prompt = random_prompt(414, 10)
    expected = target.greedy_generate(prompt, 8)
    calls = []
    with DartEngine(target, NGramDrafter()) as engine:
        if wrapped:
            @wraps(verify_block_greedy)
            def verify(*args, **kwargs):
                calls.append(kwargs)
                return verify_block_greedy(*args, **kwargs)
        else:
            def verify(target, state, pending, drafts):
                calls.append({})
                return verify_block_greedy(target, state, pending, drafts)
        engine.verify = verify
        actual = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
    assert actual.token_ids == expected
    assert calls
    assert all(call == ({"compute_entropy": False} if wrapped else {}) for call in calls)
