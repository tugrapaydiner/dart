from types import SimpleNamespace

import pytest
import torch

from dart.drafters.base import NullDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeAction, DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models


class HookDrafter(NullDrafter):
    __hash__ = None

    def __init__(self, name):
        self.name = name
        self.calls = []

    def __eq__(self, other):
        return isinstance(other, HookDrafter)  # Equal objects still own separate state.

    def seed_request(self, seed):
        self.calls.append(('seed', seed))

    def after_prefill(self):
        self.calls.append(('prefill', None))

    def observe(self, result):
        self.calls.append(('observe', tuple(result.committed)))

    def after_verify(self, result):
        self.calls.append(('feedback', tuple(result.committed)))

    def finish_request(self):
        self.calls.append(('finish', None))

    def close(self):
        self.calls.append(('close', None))


class LaneScheduler:
    requires_entropy = False
    requires_free_memory = False

    def __init__(self, verifier):
        self.verifier = verifier
        self.acceptance_ewma = {}
        self.reset()

    def reset(self):
        self.index = 0

    def choose_action(self, state, allowed=None):
        lane = ('first', 'second', 'alias', 'target_only')[self.index % 4]
        self.index += 1
        return DecodeAction(lane, 0, 0, 16, 1, 0, True, verifier=self.verifier)

    def observe(self, *args, **kwargs):
        pass


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


@pytest.mark.parametrize('verifier', ['block', 'tree', 'reference'])
@pytest.mark.parametrize('budget', [1, 10])
def test_shared_lanes_receive_one_hook_per_request_and_round(models, verifier, budget):
    target, reference = models
    first, second = HookDrafter('first'), HookDrafter('second')
    lanes = {'first': first, 'alias': first, 'second': second, 'other_alias': second}
    engine = DartEngine(target, lanes=lanes, scheduler=LaneScheduler(verifier), owns_drafters=True)
    seeds = torch.Generator().manual_seed(23)
    expected_seeds = [int(torch.randint(2**62, (), generator=seeds)) for _ in range(2)]
    with engine:
        for prompt in ([5, 7, 9], [11, 13], [5, 7, 9]):
            expected = hf_decoder(reference)(GenerationCase('hooks', tuple(prompt), budget))
            for drafter in (first, second):
                drafter.calls.clear()
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=budget, seed=23))
            assert result.token_ids == expected
            for drafter, seed in zip((first, second), expected_seeds):
                calls = [('seed', seed), ('prefill', None)]
                for index, step in enumerate(result.rounds):
                    kind = 'observe' if step.lane == drafter.name else 'feedback'
                    calls.append((kind, (expected[index + 1],)))
                assert drafter.calls == calls + [('finish', None)]
    assert first.calls[-1] == second.calls[-1] == ('close', None)
    assert sum(kind == 'close' for kind, _ in first.calls) == 1
    assert sum(kind == 'close' for kind, _ in second.calls) == 1


class Cancelled(BaseException):
    pass


@pytest.mark.parametrize('stage', ['after_prefill', 'observe', 'after_verify'])
@pytest.mark.parametrize('error_type', [RuntimeError, Cancelled])
def test_shared_hook_errors_propagate_and_cleanup_once(models, monkeypatch, stage, error_type):
    target, _ = models
    first, second = HookDrafter('first'), HookDrafter('second')
    engine = DartEngine(target, lanes={'first': first, 'alias': first, 'second': second},
                        scheduler=LaneScheduler('block'))
    failed = second if stage == 'after_verify' else first
    error = error_type('hook failed')
    invoked = []

    def fail(*args):
        invoked.append(stage)
        raise error

    with monkeypatch.context() as patch:
        patch.setattr(failed, stage, fail)
        with pytest.raises(error_type) as caught:
            engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=3))
    assert caught.value is error and invoked == [stage]
    for drafter in (first, second):
        assert sum(kind == 'finish' for kind, _ in drafter.calls) == 1
        drafter.calls.clear()
    engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=3))
    for drafter in (first, second):
        assert sum(kind == 'prefill' for kind, _ in drafter.calls) == 1
        assert sum(kind == 'finish' for kind, _ in drafter.calls) == 1


def test_lane_replacement_between_requests_is_visible(models):
    target, _ = models
    first, second = HookDrafter('first'), HookDrafter('second')
    lanes = {'first': first, 'alias': first}
    engine = DartEngine(target, lanes=lanes, scheduler=SimpleNamespace(requires_entropy=False))
    engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=1))
    first.calls.clear()
    lanes['first'] = lanes['alias'] = second
    engine.generate([5, 7, 9], DecodeConfig(max_new_tokens=1))
    assert first.calls == []
    assert second.calls == [('prefill', None), ('finish', None)]
