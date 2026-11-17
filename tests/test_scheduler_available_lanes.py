from contextlib import ExitStack
from functools import wraps
from unittest.mock import patch

import pytest
import torch

from dart.drafters.ngram import NGramDrafter
from dart.models.replay import CudaGraphTarget
from dart.runtime import engine as engine_module
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import BanditScheduler, HeuristicScheduler, SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler, TimeScheduler
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
    with torch.no_grad():
        target.model.lm_head.weight.zero_()
        reference.lm_head.weight.zero_()
    try:
        yield target, reference
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('policy', [TimeScheduler, CostTableScheduler])
@pytest.mark.parametrize('execution', ['block', 'tree', 'static'])
@pytest.mark.parametrize('restricted', [False, True])
@torch.inference_mode()
def test_missing_lane_cannot_spend_the_only_probe(models, policy, execution, restricted, monkeypatch):
    base, reference = models
    verifier = 'tree' if execution == 'tree' else 'block'
    options = {'draft_lengths': (2,), 'verifiers': (verifier,)} if policy is CostTableScheduler else {}
    settings = SchedulerConfig(available_lanes=['missing', 'ngram'], draft_len=2, ewma_alpha=1)
    scheduler = policy(settings, warmup_rounds=1, retry_interval=99, max_probe_rounds=1, **options)
    allowed = ['missing', 'ngram'] if restricted else None
    config = DecodeConfig(max_new_tokens=12, allowed_lanes=allowed)
    prompt = [0, 0, 0, 0]
    expected = hf_decoder(reference)(GenerationCase('available', tuple(prompt), config.max_new_tokens))
    drafter, lanes = NGramDrafter(), {}
    with ExitStack() as cleanup:
        target = base
        if execution == 'static':
            target = CudaGraphTarget(base, capacity=64, max_block_tokens=3, replay=False, capture_predictions=True)
            cleanup.callback(target.close)
        engine = cleanup.enter_context(DartEngine(target, lanes=lanes, scheduler=scheduler, verifier=verifier))
        clock, drafting = [0.0], [False]
        pick, verify = engine._pick_lane, engine.verify

        def choose(*args, **kwargs):
            choice = pick(*args, **kwargs)
            drafting[0] = choice[0] is drafter
            return choice

        @wraps(verify)
        def measured(*args, **kwargs):
            result = verify(*args, **kwargs)
            clock[0] += .002 if drafting[0] else .010  # Policy feedback, not a benchmark.
            return result

        monkeypatch.setattr(engine, '_pick_lane', choose)
        monkeypatch.setattr(engine, 'verify', measured)
        monkeypatch.setattr(engine_module.time, 'perf_counter', lambda: clock[0])
        for available in (True, False, True):
            lanes.clear()
            if available:
                lanes['ngram'] = drafter
            with patch.object(target, 'crop', wraps=target.crop) as crop:
                output = engine.generate(prompt, config)
            assert output.token_ids == expected and output.metadata['greedy_verified']
            state = crop.call_args.args[0]
            live = prompt + expected[:-1]
            native = _reference_forward(reference, live)
            assert not _cache_failures(state, native.past_key_values, len(live), atol=2e-6, rtol=2e-5)
            followup = target.forward_block(state, [expected[-1]])[-1]
            next_native = _reference_forward(reference, live + [expected[-1]])
            torch.testing.assert_close(followup, next_native.logits[0, -1], atol=2e-6, rtol=2e-5)
            assert not _cache_failures(state, next_native.past_key_values, len(live) + 1, atol=2e-6, rtol=2e-5)
            if available:
                assert [row.lane for row in output.rounds[:3]] == ['target_only', 'ngram', 'ngram']
                arm = f'ngram:2:{verifier}' if policy is CostTableScheduler else 'ngram'
                assert set(scheduler.counts) == {'target_only', arm}
                assert scheduler.counts[arm] > 1 and scheduler.probe_rounds == 1
                assert any(row.n_draft_accepted for row in output.rounds)
            else:
                assert {row.lane for row in output.rounds} == {'target_only'}
                assert scheduler.counts == {'target_only': len(output.rounds)}
                assert scheduler.probe_rounds == 0
            assert sum(scheduler.counts.values()) == len(output.rounds)
            assert settings.available_lanes == ['missing', 'ngram']
            assert config.allowed_lanes == (['missing', 'ngram'] if restricted else None)


@pytest.mark.parametrize('policy', [HeuristicScheduler, BanditScheduler, TimeScheduler, CostTableScheduler])
@pytest.mark.parametrize('allowed', [[], ['missing'], ['target_only']])
def test_available_lane_still_respects_request_restrictions(models, policy, allowed):
    target, reference = models
    options = {'warmup_rounds': 1, 'max_probe_rounds': 1} if issubclass(policy, TimeScheduler) else {}
    scheduler = policy(SchedulerConfig(available_lanes=['missing', 'ngram']), **options)
    expected = hf_decoder(reference)(GenerationCase('restricted', (0, 0, 0, 0), 8))
    with DartEngine(target, lanes={'ngram': NGramDrafter()}, scheduler=scheduler) as engine:
        output = engine.generate([0, 0, 0, 0], DecodeConfig(max_new_tokens=8, allowed_lanes=allowed))
    assert output.token_ids == expected
    assert output.metadata['draft_tokens_proposed'] == 0
    assert {row.lane for row in output.rounds} == {'target_only'}
    if isinstance(scheduler, TimeScheduler):
        assert scheduler.counts == {'target_only': len(output.rounds)}
        assert scheduler.probe_rounds == 0
