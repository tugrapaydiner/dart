from __future__ import annotations

from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeAction, DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _ReferenceDrafter, _synthetic_models

static_only = pytest.mark.skipif(Version(version("transformers")) < Version("5.16.1"),
                                 reason="static execution requires transformers >= 5.16.1")

MIXED = [(0, "block"), (3, "block"), (2, "tree"), (4, "tree"), (1, "block"), (4, "reference"),
         (4, "block"), (0, "tree"), (3, "tree"), (2, "reference")]
LINEAR = [(0, "block"), (3, "block"), (1, "block"), (4, "block"), (2, "block")]
CASES = [((5, 9, 12, 7), 24), ((30, 31, 32, 33, 34, 35, 36, 37), 17), ((40,), 9)]


class ScriptedScheduler:
    """Cycle through fixed (draft length, verifier) actions on one lane."""

    requires_entropy = False

    def __init__(self, script, lane="reference"):
        self.script, self.lane, self.acceptance_ewma, self.index = script, lane, {}, 0

    def reset(self):
        self.index = 0

    def choose_action(self, state, allowed=None):
        draft_len, verifier = self.script[self.index % len(self.script)]
        self.index += 1
        return DecodeAction(self.lane, draft_len, 0, 16, 1, 0, state.exact_required, verifier=verifier)

    def observe(self, *args, **kwargs):
        pass


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    target, reference = _synthetic_models(17)
    yield target, reference
    torch.set_num_threads(threads)


def reference_runs(reference):
    decode = hf_decoder(reference)
    outputs = [decode(GenerationCase(f"case-{i}", prompt, budget)) for i, (prompt, budget) in enumerate(CASES)]
    return outputs, [prompt + tuple(output) for (prompt, _), output in zip(CASES, outputs)]


def run(engine):
    return [engine.generate(list(prompt), DecodeConfig(max_new_tokens=budget)) for prompt, budget in CASES]


def scripted_engine(target, sequences, script, **options):
    lanes = {"reference": _ReferenceDrafter(sequences)}
    return DartEngine(target, lanes=lanes, scheduler=ScriptedScheduler(script), **options)


def static_target(target):
    from dart.models.replay import CudaGraphTarget

    return CudaGraphTarget(target, capacity=64, max_block_tokens=5, replay=False)


def test_switching_verifiers_each_round_matches_reference(models):
    target, reference = models
    expected, sequences = reference_runs(reference)
    with scripted_engine(target, sequences, MIXED) as engine:
        results = run(engine)
    assert [result.token_ids for result in results] == expected
    rounds = [step for result in results for step in result.rounds]
    assert {step.verifier for step in rounds} == {"block", "tree", "reference"}
    assert any(step.verifier == "tree" and step.n_draft_accepted > 0 for step in rounds)
    assert any(step.verifier == "block" and step.n_draft_accepted == step.n_draft_proposed > 0 for step in rounds)
    first = results[0]
    assert [step.verifier for step in first.rounds] == [MIXED[i % len(MIXED)][1] for i in range(len(first.rounds))]
    assert first.metadata["verifiers_used"] == ["block", "reference", "tree"]


@static_only
def test_static_target_switches_draft_lengths_and_matches_reference(models):
    target, reference = models
    expected, sequences = reference_runs(reference)
    replay = static_target(target)
    try:
        with scripted_engine(replay, sequences, LINEAR) as engine:
            results = run(engine)
    finally:
        replay.close()
    assert [result.token_ids for result in results] == expected
    proposed = {step.n_draft_proposed for result in results for step in result.rounds}
    assert {0, 1, 3, 4} <= proposed


@static_only
def test_static_target_rejects_tree_rounds_before_drafting(models):
    target, reference = models
    expected, sequences = reference_runs(reference)
    replay = static_target(target)
    try:
        with scripted_engine(replay, sequences, [(2, "block"), (2, "tree")]) as engine:
            with pytest.raises(ValueError, match="block verification only"):
                engine.generate(list(CASES[0][0]), DecodeConfig(max_new_tokens=CASES[0][1]))
            assert engine.lanes["reference"].rounds == 1  # The tree round never drafted.
            engine.scheduler.script = LINEAR
            assert [result.token_ids for result in run(engine)] == expected
    finally:
        replay.close()


@pytest.mark.parametrize("mode", ["exact_sampling", "approx"])
def test_tree_actions_require_exact_mode(models, mode):
    target, reference = models
    _, sequences = reference_runs(reference)
    with scripted_engine(target, sequences, [(2, "block"), (2, "tree")]) as engine:
        with pytest.raises(ValueError, match="tree verifier requires exact mode"):
            engine.generate([5, 9, 12, 7], DecodeConfig(mode=mode, max_new_tokens=12, seed=1))


def test_unknown_round_verifier_is_rejected(models):
    target, reference = models
    _, sequences = reference_runs(reference)
    with scripted_engine(target, sequences, [(2, "beam")]) as engine:
        with pytest.raises(ValueError, match="unknown verifier 'beam'"):
            engine.generate([5, 9, 12, 7], DecodeConfig(max_new_tokens=6))


def test_actions_without_a_verifier_keep_the_engine_verifier(models):
    target, reference = models
    expected, sequences = reference_runs(reference)
    with scripted_engine(target, sequences, [(3, None), (1, None)], verifier="tree") as engine:
        results = run(engine)
    assert [result.token_ids for result in results] == expected
    assert {step.verifier for result in results for step in result.rounds} == {"tree"}
