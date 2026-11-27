import json
from contextlib import ExitStack
from functools import wraps

import pytest
import torch

from dart.drafters.base import NullDrafter
from dart.models.replay import CudaGraphTarget
from dart.runtime import engine as engine_module
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler, TimeScheduler
from dart.runtime.types import DecodeAction, DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models


class Worker(NullDrafter):
    name = 'worker'


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


@pytest.mark.parametrize('policy', [TimeScheduler, CostTableScheduler])
@pytest.mark.parametrize('execution', ['block', 'tree', 'static'])
def test_timed_alias_learns_the_selected_arm(models, policy, execution, monkeypatch, tmp_path):
    base, reference = models
    verifier = 'tree' if execution == 'tree' else 'block'
    options = {'draft_lengths': (2,), 'verifiers': (verifier,)} if policy is CostTableScheduler else {}
    scheduler = policy(SchedulerConfig(available_lanes=['fast'], draft_len=2, ewma_alpha=1),
                       warmup_rounds=1, retry_interval=99, max_probe_rounds=1, **options)
    drafter = Worker()
    trace_path = tmp_path / 'rounds.jsonl'
    outputs = []
    with ExitStack() as cleanup:
        target = base
        if execution == 'static':
            target = CudaGraphTarget(base, capacity=64, max_block_tokens=3, replay=False, capture_predictions=True)
            cleanup.callback(target.close)
        engine = cleanup.enter_context(DartEngine(target, lanes={'fast': drafter}, scheduler=scheduler,
                                                  verifier=verifier, trace_path=str(trace_path)))
        clock, drafting = [0.0], [False]
        pick, verify = engine._pick_lane, engine.verify

        def choose(*args, **kwargs):
            choice = pick(*args, **kwargs)
            drafting[0] = choice[0] is drafter
            return choice

        @wraps(verify)
        def measured(*args, **kwargs):
            result = verify(*args, **kwargs)
            clock[0] += .002 if drafting[0] else .010  # Deterministic policy feedback, not a benchmark.
            return result

        monkeypatch.setattr(engine, '_pick_lane', choose)
        monkeypatch.setattr(engine, 'verify', measured)
        monkeypatch.setattr(engine_module.time, 'perf_counter', lambda: clock[0])
        for prompt in ((5, 7, 9), (11, 13)):
            expected = hf_decoder(reference)(GenerationCase('lane', prompt, 8))
            output = engine.generate(list(prompt), DecodeConfig(max_new_tokens=8))
            outputs.append(output)
            assert output.token_ids == expected
            assert [row.lane for row in output.rounds[:3]] == ['target_only', 'fast', 'fast']
            arm = f'fast:2:{verifier}' if policy is CostTableScheduler else 'fast'
            assert set(scheduler.counts) == {'target_only', arm}
            assert scheduler.counts[arm] > 1 and scheduler.probe_rounds == 1
            assert sum(scheduler.counts.values()) == len(output.rounds)
            assert output.metadata['lanes_used'] == ['fast', 'target_only']
            assert drafter.name == 'worker'
    traces = [json.loads(line) for line in trace_path.read_text(encoding='utf-8').splitlines()]
    assert [row['lane'] for row in traces] == [step.lane for output in outputs for step in output.rounds]


class ScriptedScheduler:
    requires_entropy = False
    requires_free_memory = False

    def __init__(self):
        self.acceptance_ewma, self.observed, self.index = {}, [], 0

    def choose_action(self, state, allowed=None):
        lane = ('left', 'right', 'alias', 'missing')[self.index % 4]
        self.index += 1
        return DecodeAction(lane, 2, 0, 16, 1, 0, state.exact_required)

    def observe(self, lane, result, **timing):
        self.observed.append(lane)


class RecordingWorker(Worker):
    def __init__(self):
        self.active, self.passive = 0, 0

    def propose(self, committed, max_tokens):
        return [0] * max_tokens

    def observe(self, result):
        self.active += 1

    def after_verify(self, result):
        self.passive += 1


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('mode', ['exact', 'approx'])
def test_lane_keys_stay_distinct_from_shared_resource_names(models, shared, mode):
    target, reference = models
    with torch.no_grad():
        target.model.lm_head.weight.zero_()
        reference.lm_head.weight.zero_()
    first = RecordingWorker()
    second = first if shared else RecordingWorker()
    lanes = {'left': first, 'right': second, 'alias': first}
    scheduler = ScriptedScheduler()
    expected = hf_decoder(reference)(GenerationCase('shared', (5, 7, 9), 25))
    with DartEngine(target, lanes=lanes, scheduler=scheduler) as engine:
        output = engine.generate([5, 7, 9], DecodeConfig(mode=mode, max_new_tokens=25, verify_every=2))
    wanted = [('left', 'right', 'alias', 'target_only')[i % 4] for i in range(len(output.rounds))]
    assert output.token_ids == expected
    assert [row.lane for row in output.rounds] == wanted
    verified = [row for row in output.rounds if row.verifier != 'none']
    assert scheduler.observed == [row.lane for row in verified]
    assert output.metadata['lanes_used'] == sorted(set(wanted))
    if mode == 'approx':
        assert len(verified) < len(output.rounds)
    for drafter in {id(first): first, id(second): second}.values():
        active = sum(lanes.get(row.lane) is drafter for row in verified)
        assert drafter.active == active and drafter.passive == len(verified) - active
        assert drafter.name == 'worker'


@pytest.mark.parametrize('policy', [TimeScheduler, CostTableScheduler])
def test_missing_lane_feedback_still_describes_target_fallback(models, policy):
    target, reference = models
    scheduler = policy(SchedulerConfig(available_lanes=['missing']), warmup_rounds=1, max_probe_rounds=1)
    expected = hf_decoder(reference)(GenerationCase('fallback', (5, 7, 9), 8))
    with DartEngine(target, lanes={}, scheduler=scheduler) as engine:
        output = engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=8))
    assert output.token_ids == expected
    assert scheduler.counts == {'target_only': len(output.rounds)}
    assert scheduler.probe_rounds == 0
    assert {row.lane for row in output.rounds} == {'target_only'}
