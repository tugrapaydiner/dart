from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from dart.models.replay import CudaGraphTarget
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeAction, DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import DraftTree


class FeedbackDrafter:
    name = 'feedback'

    def __init__(self):
        self.entropies = []

    def propose(self, committed, max_tokens):
        return []

    def observe(self, result):
        self.entropies.append(result.entropy)


class SwitchingScheduler:
    requires_free_memory = False

    def __init__(self, verifiers, needs_entropy=False):
        self.verifiers = verifiers
        self.requires_entropy = needs_entropy
        self.acceptance_ewma = {}
        self.reset()

    def reset(self):
        self.states, self.entropies, self.lanes = [], [], []

    def choose_action(self, state, allowed=None):
        index = len(self.states)
        self.states.append(state.entropy_ewma)
        lane = ('target_only', 'feedback', 'target_only')[index % 3]
        self.lanes.append(lane)
        return DecodeAction(lane, 2, 0, 16, 1, 0, True,
                            verifier=self.verifiers[index % len(self.verifiers)])

    def observe(self, lane, result, **timing):
        self.entropies.append(result.entropy)


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    target, reference = _synthetic_models(17)
    target.eos_token_ids = frozenset()
    reference.config.eos_token_id = reference.generation_config.eos_token_id = None
    try:
        yield target, reference
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('verifiers', [('block',), ('tree',), ('block', 'tree')])
@pytest.mark.parametrize('passive', [False, True])
@pytest.mark.parametrize('scheduler_entropy', [False, True])
def test_feedback_follows_selected_lane(models, verifiers, passive, scheduler_entropy):
    target, reference = models
    drafter = FeedbackDrafter()  # Custom lanes request entropy by default.
    if passive:
        drafter.after_verify = drafter.observe
    scheduler = SwitchingScheduler(verifiers, scheduler_entropy)
    expected = hf_decoder(reference)(GenerationCase('switching', (5, 7, 9), 8))
    with DartEngine(target, lanes={'feedback': drafter}, scheduler=scheduler) as engine:
        for enabled in (True, False, True):
            drafter.requires_entropy = enabled
            drafter.entropies.clear()
            with patch.object(torch, 'softmax', wraps=torch.softmax) as softmax:
                output = engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=8))
            needs = [scheduler_entropy or (enabled and (passive or lane == 'feedback'))
                     for lane in scheduler.lanes]
            assert output.token_ids == expected
            assert len(output.rounds) == len(needs) == 7
            assert softmax.call_count == int(scheduler_entropy) + sum(needs)
            assert [(value > 0) for value in scheduler.entropies] == needs
            assert len(drafter.entropies) == (7 if passive else scheduler.lanes.count('feedback'))
            assert all((value > 0) == (scheduler_entropy or enabled) for value in drafter.entropies)
            if scheduler_entropy:
                assert scheduler.states[0] > 0
                for index in range(1, len(needs)):
                    assert scheduler.states[index] == pytest.approx(
                        .7 * scheduler.states[index - 1] + .3 * scheduler.entropies[index - 1])


def test_branching_observer_receives_live_entropy(models):
    target, reference = models
    drafter = FeedbackDrafter()
    drafter.propose_tree = lambda committed, budget: DraftTree.from_branches([[3], [4]])
    scheduler = SwitchingScheduler(('tree',))
    expected = hf_decoder(reference)(GenerationCase('branches', (5, 7, 9), 12))
    with DartEngine(target, lanes={'feedback': drafter}, scheduler=scheduler) as engine:
        with patch.object(target, 'forward_tree', wraps=target.forward_tree) as forward:
            output = engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=12))
    assert output.token_ids == expected and forward.call_count > 0
    assert drafter.entropies and all(value > 0 for value in drafter.entropies)
    assert all(value == 0 for lane, value in zip(scheduler.lanes, scheduler.entropies) if lane == 'target_only')


@pytest.mark.parametrize('passive', [False, True])
def test_disallowed_lane_only_needs_entropy_for_passive_feedback(models, passive):
    target, reference = models
    drafter = FeedbackDrafter()
    if passive:
        drafter.after_verify = drafter.observe
    expected = hf_decoder(reference)(GenerationCase('allowed', (5, 7, 9), 8))
    with DartEngine(target, drafter) as engine:
        with patch.object(torch, 'softmax', wraps=torch.softmax) as softmax:
            output = engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=8, allowed_lanes=['target_only']))
    assert output.token_ids == expected
    assert softmax.call_count == (7 if passive else 0)
    assert len(drafter.entropies) == (7 if passive else 0)


def test_static_rounds_restore_compact_predictions(models):
    base, reference = models
    prompt, budget = [5, 7, 9], 8
    expected = hf_decoder(reference)(GenerationCase('static', tuple(prompt), budget))
    scheduler = SwitchingScheduler(('block',))
    drafter = FeedbackDrafter()
    with ExitStack() as cleanup:
        target = CudaGraphTarget(base, capacity=64, max_block_tokens=3, replay=False, capture_predictions=True)
        cleanup.callback(target.close)
        engine = cleanup.enter_context(DartEngine(target, lanes={'feedback': drafter}, scheduler=scheduler))
        with patch.object(target, 'forward_predictions', wraps=target.forward_predictions) as predictions:
            with patch.object(target, 'forward_block', wraps=target.forward_block) as logits:
                output = engine.generate(prompt, DecodeConfig(max_new_tokens=budget))
        assert output.token_ids == expected
        assert predictions.call_count == scheduler.lanes.count('target_only') == 5
        assert logits.call_count == scheduler.lanes.count('feedback') == 2
        state = predictions.call_args.args[0]
        with torch.inference_mode():
            ids = torch.tensor([prompt + expected[:-1]])
            native = reference(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)
            assert state.seq_len == ids.shape[1]
            for index, layer in enumerate(native.past_key_values.layers):
                torch.testing.assert_close(state.cache.storage[index, 0, ..., :state.seq_len, :],
                                           layer.keys, rtol=2e-5, atol=2e-6)
                torch.testing.assert_close(state.cache.storage[index, 1, ..., :state.seq_len, :],
                                           layer.values, rtol=2e-5, atol=2e-6)
            followup = reference(input_ids=torch.tensor([[expected[-1]]]),
                                 past_key_values=native.past_key_values, use_cache=True).logits[0].float()
            torch.testing.assert_close(target.forward_block(state, [expected[-1]]), followup,
                                       rtol=2e-5, atol=2e-6)
