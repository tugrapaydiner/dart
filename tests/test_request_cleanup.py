import copy
import threading

import pytest

from dart.drafters.ngram import NGramDrafter
from dart.kv.prefetch import ThreadPrefetcher
from dart.kv.residency import ResidencyStats
from dart.runtime import engine as runtime
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder


@pytest.mark.parametrize("stage", ["draft", "verify", "observe", "trace", "offload"])
def test_failed_request_closes_offload_worker_and_next_request_succeeds(tiny_target, monkeypatch, stage):
    workers = []

    class Worker(ThreadPrefetcher):
        closed = False

        def __init__(self):
            super().__init__()
            workers.append(self)

        def close(self):
            super().close()
            self.closed = True

    def fail(*args, **kwargs):
        raise RuntimeError("request failure")

    monkeypatch.setattr(runtime, "ThreadPrefetcher", Worker)
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    config = DecodeConfig(max_new_tokens=4, full_kv_residency="cpu_pinned")
    with DartEngine(tiny_target) as engine:
        try:
            with monkeypatch.context() as patch:
                if stage == "offload":
                    def move(cache, device, pin=False):
                        if pin:
                            fail()
                        return ResidencyStats()

                    patch.setattr(runtime, "move_cache", move)
                else:
                    obj, attribute = {"draft": (engine.drafter, "propose"), "verify": (engine, "verify"),
                                      "observe": (engine.drafter, "observe"), "trace": (engine.trace, "log_round")}[stage]
                    patch.setattr(obj, attribute, fail)
                with pytest.raises(RuntimeError, match="request failure"):
                    engine.generate([5, 6, 7], config)
            assert workers[0].closed
            with pytest.raises(RuntimeError, match="cannot schedule new futures"):
                workers[0].prefetch(lambda: None)
            expected = tiny_target.greedy_generate([5, 6, 7], 4)
            assert engine.generate([5, 6, 7], config).token_ids == expected
            assert workers[-1].closed
        finally:
            for worker in workers:
                worker.close()


def test_interrupted_request_waits_for_inflight_offload(tiny_target, monkeypatch):
    started, release, finished = (threading.Event() for _ in range(3))
    workers = []

    class Interrupted(BaseException):
        pass

    class Worker(ThreadPrefetcher):
        def __init__(self):
            super().__init__()
            workers.append(self)

        def close(self):
            release.set()
            super().close()

    def move(cache, device, pin=False):
        if pin:
            started.set()
            assert release.wait(5)
            finished.set()
        return ResidencyStats()

    failure = Interrupted("cancelled")
    calls = 0

    def propose(*args):
        nonlocal calls
        calls += 1
        if calls == 2:
            assert started.wait(5)
            raise failure
        return []

    monkeypatch.setattr(runtime, "ThreadPrefetcher", Worker)
    monkeypatch.setattr(runtime, "move_cache", move)
    monkeypatch.setattr(tiny_target, "eos_token_ids", frozenset())
    with DartEngine(tiny_target) as engine:
        monkeypatch.setattr(engine.drafter, "propose", propose)
        try:
            with pytest.raises(Interrupted) as caught:
                engine.generate([5, 6, 7], DecodeConfig(max_new_tokens=4, full_kv_residency="cpu_pinned"))
            assert caught.value is failure
            assert finished.is_set()
            with pytest.raises(RuntimeError, match="cannot schedule new futures"):
                workers[0].prefetch(lambda: None)
        finally:
            release.set()
            for worker in workers:
                worker.close()


@pytest.mark.parametrize('verifier,mode', [('block', 'exact'), ('tree', 'exact'), ('block', 'exact_sampling')])
@pytest.mark.parametrize('stop', ['budget', 'one_round', 'verified_eos', 'prefill_eos'])
def test_completed_request_does_not_park_its_cache(tiny_target, monkeypatch, verifier, mode, stop):
    prompt = [5, 6, 7]
    budget = 2 if stop == 'one_round' else 7
    config = DecodeConfig(mode=mode, max_new_tokens=budget, draft_len=2, seed=17)
    monkeypatch.setattr(tiny_target, 'eos_token_ids', frozenset())

    def generate(residency):
        config.full_kv_residency = residency
        with DartEngine(tiny_target, NGramDrafter(min_ngram=64, max_ngram=64), verifier=verifier) as engine:
            return engine.generate(prompt, config)

    expected = generate('gpu')
    if stop in {'verified_eos', 'prefill_eos'}:
        index = 0 if stop == 'prefill_eos' else next(
            i for i, token in enumerate(expected.token_ids) if i and token not in expected.token_ids[:i]
        )
        monkeypatch.setattr(tiny_target, 'eos_token_ids', frozenset([expected.token_ids[index]]))
        expected = generate('gpu')
        assert expected.metadata['finished_eos'] and len(expected.token_ids) == index + 1
    if mode == 'exact':
        reference = copy.deepcopy(tiny_target.model)
        eos = list(tiny_target.eos_token_ids) or None
        reference.config.eos_token_id = reference.generation_config.eos_token_id = eos
        assert expected.token_ids == hf_decoder(reference)(GenerationCase('offload', tuple(prompt), budget))

    moves = []
    move = runtime.move_cache

    def record(cache, device, pin=False):
        result = move(cache, device, pin=pin)
        moves.append((pin, cache.get_seq_length()))
        return result

    monkeypatch.setattr(runtime, 'move_cache', record)
    actual = generate('cpu_pinned')
    assert actual.token_ids == expected.token_ids
    assert actual.metadata['finished_eos'] == expected.metadata['finished_eos']
    assert sum(pin for pin, _ in moves) == max(0, len(actual.rounds) - 1)
    assert sum(not pin for pin, _ in moves) == len(actual.rounds)
    assert actual.metadata['kv_residency_moves'] == actual.metadata['kv_residency_moved_mb'] == 0
    assert generate('cpu_pinned').token_ids == expected.token_ids


def test_unverified_eos_still_drains_the_previous_offload(tiny_target, monkeypatch):
    prompt = [5, 6, 7]
    monkeypatch.setattr(tiny_target, 'eos_token_ids', frozenset())
    first = tiny_target.greedy_generate(prompt, 2)
    eos = next(token for token in range(5, tiny_target.model.config.vocab_size) if token not in first)
    monkeypatch.setattr(tiny_target, 'eos_token_ids', frozenset([eos]))
    workers, offloads = [], []
    move = runtime.move_cache

    class Worker(ThreadPrefetcher):
        closed = False

        def __init__(self):
            super().__init__()
            workers.append(self)

        def close(self):
            super().close()
            self.closed = True

    def record(cache, device, pin=False):
        result = move(cache, device, pin=pin)
        if pin:
            offloads.append(cache.get_seq_length())
        return result

    monkeypatch.setattr(runtime, 'ThreadPrefetcher', Worker)
    monkeypatch.setattr(runtime, 'move_cache', record)
    with DartEngine(tiny_target) as engine:
        monkeypatch.setattr(engine.drafter, 'propose', lambda committed, limit: [eos] * limit)
        result = engine.generate(prompt, DecodeConfig(mode='approx', verify_every=2, max_new_tokens=8,
                                                       draft_len=2, full_kv_residency='cpu_pinned'))
    assert result.token_ids == [*first, eos]
    assert result.metadata['finished_eos'] and result.metadata['trailing_unverified'] == 1
    assert len(result.rounds) == 2 and len(offloads) == 1
    assert workers[0].closed
    with pytest.raises(RuntimeError, match='cannot schedule new futures'):
        workers[0].prefetch(lambda: None)


@pytest.mark.parametrize('early_eos', [False, True])
def test_accepted_drafts_can_finish_without_an_offload(tiny_target, monkeypatch, early_eos):
    prompt = [5, 6, 7]
    monkeypatch.setattr(tiny_target, 'eos_token_ids', frozenset())
    reference = copy.deepcopy(tiny_target.model)
    reference.config.eos_token_id = reference.generation_config.eos_token_id = None
    expected = hf_decoder(reference)(GenerationCase('accepted', tuple(prompt), 5))
    if early_eos:
        monkeypatch.setattr(tiny_target, 'eos_token_ids', frozenset([expected[3]]))
        expected = expected[:4]
    move = runtime.move_cache

    def no_park(cache, device, pin=False):
        assert not pin, 'completed request submitted an offload'
        return move(cache, device, pin=pin)

    monkeypatch.setattr(runtime, 'move_cache', no_park)
    with DartEngine(tiny_target) as engine:
        monkeypatch.setattr(engine.drafter, 'propose', lambda committed, limit: expected[1:1 + limit])
        result = engine.generate(prompt, DecodeConfig(max_new_tokens=5, draft_len=4,
                                                       full_kv_residency='cpu_pinned'))
    assert result.token_ids == expected
    assert len(result.rounds) == 1 and result.rounds[0].n_draft_accepted == 3
