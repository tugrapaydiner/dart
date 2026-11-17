from __future__ import annotations

import copy
import json
import weakref
from dataclasses import asdict, replace
from functools import wraps

import pytest
import torch

from dart.models.replay import CudaGraphTarget
from dart.verification import state_checks
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_checks import (
    RoundCase,
    check_verifier_state,
    compare_verifier_states,
)
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import DraftTree


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    target, reference = _synthetic_models(17)
    target.eos_token_ids = frozenset()
    reference.config.eos_token_id = reference.generation_config.eos_token_id = None
    yield target, reference
    torch.set_num_threads(previous)


def make_case(reference, shape="chain", rejection=None):
    prefix = (5, 6, 7)
    oracle = hf_decoder(reference)(GenerationCase("oracle", prefix, 5))
    drafts = oracle[1:4]
    if rejection is not None:
        drafts[rejection] = (drafts[rejection] + 1) % reference.config.vocab_size
    branches = [] if shape == "empty" else [drafts]
    if shape == "branch":
        branches.append([(oracle[1] + 1) % reference.config.vocab_size, 13, 17])
    tree = DraftTree.from_branches(branches)
    return RoundCase("compare", prefix, oracle[0], tuple(tree.tokens), tuple(tree.parent))


@pytest.mark.parametrize("inspect", [False, True])
@pytest.mark.parametrize("shape,rejection", [("empty", None), ("chain", None), ("chain", 0),
                                            ("chain", 2), ("branch", None), ("branch", 1)])
def test_comparison_matches_separate_checks_with_fewer_native_calls(models, monkeypatch, shape, rejection, inspect):
    target, reference = models
    case = make_case(reference, shape, rejection)
    candidates = {"first": target, "second": copy.deepcopy(target)}
    generation, forwards = [], []
    generate, forward = reference.generate, state_checks._reference_forward
    config = reference.generation_config

    @wraps(generate)
    def count_generation(*args, **kwargs):
        generation.append(args[0].tolist())
        return generate(*args, **kwargs)

    def count_forward(model, tokens):
        forwards.append(tuple(tokens))
        return forward(model, tokens)

    monkeypatch.setattr(reference, "generate", count_generation)
    monkeypatch.setattr(state_checks, "_reference_forward", count_forward)
    separate = {name: check_verifier_state(candidate, reference, case, inspect_tree=inspect)
                for name, candidate in candidates.items()}
    old_forwards, old_generation = list(forwards), list(generation)
    forwards.clear()
    generation.clear()
    together = compare_verifier_states(candidates, reference, case, inspect_tree=inspect)
    assert together == separate and all(result.passed for result in together.values())
    assert list(together) == list(candidates)
    assert len(generation) == 1 and old_generation == generation * 2
    assert len(forwards) == len(old_forwards) - 2
    assert forwards[:2] == old_forwards[:2]  # Same full-prefix reference inputs.
    assert len(forwards) == 2 + (2 * (len(case.drafts) + 1) if inspect else 0)
    assert reference.generation_config is config
    json.dumps({name: asdict(result) for name, result in together.items()})


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("verifier", ["tree", "block"])
def test_eos_and_static_cache_match_native_checks(models, position, verifier):
    target, reference = models
    case = make_case(reference)
    eos = case.drafts[position]
    target.eos_token_ids = frozenset([eos])
    reference.generation_config.eos_token_id = eos
    static = CudaGraphTarget(target, capacity=64, max_block_tokens=4, replay=False)
    try:
        results = compare_verifier_states({"eager": target, "static": static}, reference, case, verifier=verifier)
        assert results["eager"] == results["static"]
        assert results["eager"].passed, results
        first_eos = case.drafts.index(eos)
        assert results["eager"].actual == case.drafts[:first_eos + 1]
        assert results["eager"].cache_length == len(case.prefix) + first_eos + 1
    finally:
        static.close()


@pytest.mark.parametrize("fault", ["exception", "cache", "followup", "branch"])
def test_one_fault_does_not_hide_the_next_candidate(models, monkeypatch, fault):
    target, reference = models
    good = copy.deepcopy(target)
    case = make_case(reference, "branch")
    if fault == "exception":
        def fail(*args):
            raise RuntimeError("injected")
        monkeypatch.setattr(target, "prefill", fail)
        kind = "exception"
    elif fault == "cache":
        gather = target.gather_tree_tail
        monkeypatch.setattr(target, "gather_tree_tail", lambda state, keep, path: gather(state, keep, [0] * len(path)))
        kind = "cache_values"
    elif fault == "followup":
        forward = target.forward_block
        monkeypatch.setattr(target, "forward_block", lambda *args: forward(*args) + 1)
        kind = "followup_logits"
    else:
        forward = target.forward_tree

        def wrong_branch(*args):
            logits = forward(*args).clone()
            logits[4] += 1
            return logits

        monkeypatch.setattr(target, "forward_tree", wrong_branch)
        kind = "tree_logits"
    results = compare_verifier_states({"faulty": target, "good": good}, reference, case, inspect_tree=True)
    assert kind in {failure.kind for failure in results["faulty"].failures}
    assert results["good"].passed, results["good"].failures


@pytest.mark.parametrize("invalid", ["shared", "training", "reference_training", "eos", "context", "vocab"])
def test_all_candidates_are_validated_before_any_forward(models, monkeypatch, invalid):
    target, reference = models
    case = make_case(reference)
    bad = copy.deepcopy(target)
    if invalid == "shared":
        bad.model = reference
    elif invalid == "training":
        bad.model.train()
    elif invalid == "reference_training":
        reference.train()
    elif invalid == "eos":
        bad.eos_token_ids = frozenset([2])
    elif invalid == "context":
        bad.model.config.max_position_embeddings = 4
    else:
        bad.model.config.vocab_size = 1
    for model in (target.model, bad.model, reference):
        monkeypatch.setattr(model, "forward", lambda *a, **k: pytest.fail("invalid comparison ran a model"))
    with pytest.raises(ValueError):
        compare_verifier_states({"first": target, "invalid": bad}, reference, case)


@pytest.mark.parametrize("candidates", [{}, {"": None}, {1: None}, [], [("a", None)]])
def test_invalid_candidate_names_are_rejected(models, candidates):
    with pytest.raises(ValueError, match="candidate"):
        compare_verifier_states(candidates, models[1], make_case(models[1]))


@pytest.mark.parametrize("options", [{"verifier": "other"}, {"atol": -1}, {"rtol": float("nan")},
                                   {"inspect_tree": 1}, {"inspect_tree": True, "verifier": "block"}])
def test_invalid_options_fail_before_reference_generation(models, monkeypatch, options):
    target, reference = models
    case = make_case(reference)
    monkeypatch.setattr(reference, "generate", lambda *a, **k: pytest.fail("invalid options ran reference"))
    with pytest.raises(ValueError):
        compare_verifier_states({"a": target}, reference, case, **options)


@pytest.mark.parametrize("stage", ["generation", "cache", "followup", "tree"])
def test_reference_errors_abort_instead_of_becoming_candidate_failures(models, monkeypatch, stage):
    target, reference = models
    case = make_case(reference, "branch")
    forward = state_checks._reference_forward
    calls = 0

    def fail(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == {"generation": 1, "cache": 1, "followup": 2, "tree": 3}[stage]:
            raise RuntimeError("reference failed")
        return forward(*args, **kwargs)

    monkeypatch.setattr(reference if stage == "generation" else state_checks,
                        "generate" if stage == "generation" else "_reference_forward", fail)
    with pytest.raises(RuntimeError, match="reference failed"):
        compare_verifier_states({"a": target, "b": target}, reference, case, inspect_tree=True)


def test_reused_target_has_fresh_states_and_reference_work_per_call(models, monkeypatch):
    target, reference = models
    first = make_case(reference)
    second = replace(first, name="changed", prefix=(11, 13))
    forward = state_checks._reference_forward
    tokens, retained = [], []

    def observe(model, prefix):
        output = forward(model, prefix)
        tokens.append(tuple(prefix))
        retained.extend((weakref.ref(output.logits), weakref.ref(output.past_key_values)))
        return output

    monkeypatch.setattr(state_checks, "_reference_forward", observe)
    one = compare_verifier_states({"a": target, "b": target}, reference, first)
    two = compare_verifier_states({"a": target, "b": target}, reference, second)
    assert one["a"] == one["b"] and two["a"] == two["b"]
    assert one["a"].passed and two["a"].passed
    assert len(tokens) == 4 and tokens[0] != tokens[2]
    assert all(item() is None for item in retained)


def test_later_candidate_is_revalidated_after_an_earlier_one(models, monkeypatch):
    target, reference = models
    later = copy.deepcopy(target)
    case = make_case(reference)
    prefill = target.prefill

    def change(*args):
        later.model.train()
        return prefill(*args)

    monkeypatch.setattr(target, "prefill", change)
    monkeypatch.setattr(later, "prefill", lambda *a: pytest.fail("changed candidate executed"))
    with pytest.raises(ValueError, match="eval"):
        compare_verifier_states({"first": target, "later": later}, reference, case)


def test_cancellation_propagates(models, monkeypatch):
    target, reference = models
    case = make_case(reference)

    def cancel(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(target, "prefill", cancel)
    with pytest.raises(KeyboardInterrupt):
        compare_verifier_states({"a": target}, reference, case)
