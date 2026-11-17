from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from dart.drafters.ngram import NGramDrafter
from dart.models.replay import CudaGraphTarget
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import BanditScheduler, HeuristicScheduler, SchedulerConfig
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_checks import _cache_failures, _reference_forward
from dart.verification.synthetic import _synthetic_models


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


@pytest.mark.parametrize('policy', [HeuristicScheduler, BanditScheduler])
@pytest.mark.parametrize('execution', ['block', 'tree', 'static'])
@pytest.mark.parametrize('repeated', [False, True])
@torch.inference_mode()
def test_disabled_gate_avoids_entropy_work_and_keeps_native_state(models, policy, execution, repeated):
    base, reference = models
    prompt, budget = [5, 7, 9, 5, 7, 9], 10
    if repeated:
        base.model.lm_head.weight.zero_()
        reference.lm_head.weight.zero_()
        prompt = [0, 0, 0, 0]
    expected = hf_decoder(reference)(GenerationCase('entropy-gate', tuple(prompt), budget))
    options = SchedulerConfig(available_lanes=['ngram'], draft_len=4)
    scheduler = policy(options)
    with ExitStack() as cleanup:
        target = base
        if execution == 'static':
            target = CudaGraphTarget(base, capacity=64, max_block_tokens=5, replay=False, capture_predictions=True)
            cleanup.callback(target.close)
        engine = cleanup.enter_context(DartEngine(target, lanes={'ngram': NGramDrafter()}, scheduler=scheduler,
                                                 verifier='tree' if execution == 'tree' else 'block'))
        for threshold in (float('inf'), 0., float('inf')):
            options.entropy_gate_threshold = threshold
            with ExitStack() as observed:
                softmax = observed.enter_context(patch.object(torch, 'softmax', wraps=torch.softmax))
                crop = observed.enter_context(patch.object(target, 'crop', wraps=target.crop))
                choices = observed.enter_context(patch.object(scheduler, 'choose_action', wraps=scheduler.choose_action))
                feedback = observed.enter_context(patch.object(scheduler, 'observe', wraps=scheduler.observe))
                drafting = observed.enter_context(patch.object(engine, '_round_drafts', wraps=engine._round_drafts))
                if execution == 'static':
                    predictions = observed.enter_context(patch.object(target, 'forward_predictions', wraps=target.forward_predictions))
                    logits = observed.enter_context(patch.object(target, 'forward_block', wraps=target.forward_block))
                output = engine.generate(prompt, DecodeConfig(max_new_tokens=budget))
            assert output.token_ids == expected and output.metadata['greedy_verified']
            assert output.rounds and crop.call_count
            if repeated:
                assert any(row.n_draft_accepted for row in output.rounds)
            state = crop.call_args.args[0]
            live = prompt + expected[:-1]
            native = _reference_forward(reference, live)
            assert not _cache_failures(state, native.past_key_values, len(live), atol=2e-6, rtol=2e-5)
            followup = target.forward_block(state, [expected[-1]])[-1]
            next_native = _reference_forward(reference, live + [expected[-1]])
            torch.testing.assert_close(followup, next_native.logits[0, -1], atol=2e-6, rtol=2e-5)
            assert not _cache_failures(state, next_native.past_key_values, len(live) + 1, atol=2e-6, rtol=2e-5)
            enabled = threshold == 0
            assert softmax.call_count == (len(output.rounds) + 1 if enabled else 0)
            assert all((call.args[0].entropy_ewma > 0) == enabled for call in choices.call_args_list)
            assert all((call.args[1].entropy > 0) == enabled for call in feedback.call_args_list)
            if execution == 'static':
                assert predictions.call_count == (0 if enabled else len(output.rounds))
                assert logits.call_count == (len(output.rounds) if enabled else 0)
            if repeated:
                assert drafting.call_args_list[0].args[2] == (2 if enabled else 4)


@pytest.mark.parametrize('policy', [HeuristicScheduler, BanditScheduler])
@pytest.mark.parametrize('declared', [None, False, True])
def test_custom_scheduler_keeps_entropy_unless_it_opts_out(models, policy, declared):
    target, reference = models
    states, results = [], []

    class Custom(policy):
        def choose_action(self, state, allowed=None):
            states.append(state.entropy_ewma)
            return super().choose_action(state, allowed)

        def observe(self, lane, result, **timing):
            results.append(result.entropy)
            return super().observe(lane, result, **timing)

    if declared is not None:
        Custom.requires_entropy = declared
    scheduler = Custom(SchedulerConfig(available_lanes=['ngram']))
    expected = hf_decoder(reference)(GenerationCase('custom-gate', (5, 7, 9), 8))
    with DartEngine(target, lanes={'ngram': NGramDrafter()}, scheduler=scheduler) as engine:
        output = engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=8))
    assert output.token_ids == expected and len(states) == len(results) == len(output.rounds)
    assert states and all((value > 0) == (declared is not False) for value in states + results)


@pytest.mark.parametrize('policy', [HeuristicScheduler, BanditScheduler])
@pytest.mark.parametrize('passive', [False, True])
def test_drafter_feedback_still_requests_entropy(models, policy, passive):
    target, reference = models
    values = []
    observer = NGramDrafter()
    observer.requires_entropy = True
    lanes = {'ngram': observer}
    if passive:
        observer.after_verify = lambda result: values.append(result.entropy)
        lanes = {'ngram': NGramDrafter(), 'observer': observer}
    else:
        observer.observe = lambda result: values.append(result.entropy)
    scheduler = policy(SchedulerConfig(available_lanes=['ngram']))
    expected = hf_decoder(reference)(GenerationCase('observer-gate', (5, 7, 9), 8))
    with DartEngine(target, lanes=lanes, scheduler=scheduler) as engine:
        with patch.object(torch, 'softmax', wraps=torch.softmax) as softmax:
            output = engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=8))
    assert output.token_ids == expected and len(values) == len(output.rounds)
    assert values and all(value > 0 for value in values)
    assert softmax.call_count == len(output.rounds)


@pytest.mark.parametrize('policy', [HeuristicScheduler, BanditScheduler])
def test_seeded_sampling_keeps_its_distributions(models, policy):
    target, _ = models
    scheduler = policy(SchedulerConfig(available_lanes=['ngram']))
    config = DecodeConfig(mode='exact_sampling', max_new_tokens=8, seed=29, temperature=.7)
    with DartEngine(target, lanes={'ngram': NGramDrafter()}, scheduler=scheduler) as engine:
        with patch.object(policy, 'requires_entropy', True, create=True):
            expected = engine.generate([5, 7, 9], config)
        with patch.object(torch, 'softmax', wraps=torch.softmax) as softmax:
            actual = engine.generate([5, 7, 9], config)
    assert actual.token_ids == expected.token_ids and len(actual.rounds) == len(expected.rounds)
    assert softmax.call_count > 0 and actual.metadata['distribution_exact']
