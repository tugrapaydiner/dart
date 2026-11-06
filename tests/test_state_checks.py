from __future__ import annotations

import copy
from dataclasses import replace
from functools import wraps

import pytest
import torch

from dart.verification import state_checks
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_checks import RoundCase, check_verifier_state
from dart.verification.tree_verify import DraftTree
from tests.conftest import build_tiny_target


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    target = build_tiny_target(seed=37)
    target.eos_token_ids = frozenset()
    target.model.config.eos_token_id = None
    target.model.generation_config.eos_token_id = None
    reference = copy.deepcopy(target.model)
    yield target, reference
    torch.set_num_threads(previous)


def sample(models, rejection=None):
    _, reference = models
    prefix = (5, 6, 7, 8)
    pending = hf_decoder(reference)(GenerationCase("pending", prefix, 1))[0]
    oracle = hf_decoder(reference)(GenerationCase("draft", (*prefix, pending), 5))
    drafts = oracle[:4]
    if rejection is not None:
        drafts[rejection] = (drafts[rejection] + 1) % reference.config.vocab_size
    return RoundCase("chain", prefix, pending, tuple(drafts), (-1, 0, 1, 2))


@pytest.mark.parametrize("rejection", [0, 1, 2, 3, None])
def test_chain_and_tree_match_reference_at_every_rejection(models, rejection):
    case = sample(models, rejection)
    block = check_verifier_state(*models, case, verifier="block")
    tree = check_verifier_state(*models, case)
    assert block.passed and tree.passed, (block.failures, tree.failures)
    assert block.actual == tree.actual
    assert block.cache_length == tree.cache_length == len(case.prefix) + len(block.actual)
    assert len(block.actual) == (5 if rejection is None else rejection + 1)


@pytest.mark.parametrize("reverse", [False, True])
def test_shared_prefix_and_sibling_order_preserve_state(models, reverse):
    case = sample(models)
    a, b, c, _ = case.drafts
    branches = [[a, (b + 1) % 211], [a, b, c], [(a + 1) % 211, 9]]
    if reverse:
        branches.reverse()
    tree = DraftTree.from_branches(branches)
    case = replace(case, drafts=tuple(tree.tokens), parents=tuple(tree.parent))
    result = check_verifier_state(*models, case)
    assert result.passed, result.failures
    assert result.actual[:3] == (a, b, c)
    assert len(tree.tokens) < sum(map(len, branches))


@pytest.mark.parametrize("position", [0, 1, 2, 3])
@pytest.mark.parametrize("verifier", ["block", "tree"])
def test_eos_removes_last_accepted_token_from_cache(models, position, verifier):
    target, reference = models
    case = sample(models)
    eos = case.drafts[position]
    assert eos not in case.drafts[:position]
    target.eos_token_ids = frozenset([eos])
    reference.generation_config.eos_token_id = eos
    result = check_verifier_state(target, reference, case, verifier=verifier)
    assert result.passed, result.failures
    assert result.actual == case.drafts[:position + 1]
    assert result.cache_length == len(case.prefix) + position + 1


def test_empty_tree_checks_plain_greedy_state(models):
    case = replace(sample(models), drafts=(), parents=())
    result = check_verifier_state(*models, case)
    assert result.passed and len(result.actual) == 1


def test_missing_rollback_is_detected_in_real_block_path(models, monkeypatch):
    case = sample(models, rejection=0)
    monkeypatch.setattr(models[0], "crop", lambda *args: None)
    result = check_verifier_state(*models, case, verifier="block")
    assert not result.passed
    assert result.actual == result.expected
    assert "cache_length" in {failure.kind for failure in result.failures}


def test_wrong_gather_with_correct_length_and_output_is_detected(models, monkeypatch):
    target, _ = models
    case = sample(models)
    tree = DraftTree.from_branches([[(case.drafts[0] + 1) % 211, 9], list(case.drafts)])
    case = replace(case, drafts=tuple(tree.tokens), parents=tuple(tree.parent))
    gather = target.gather_tree_tail
    monkeypatch.setattr(target, "gather_tree_tail", lambda state, keep, path: gather(state, keep, [0] * len(path)))
    result = check_verifier_state(*models, case)
    assert not result.passed and result.actual == result.expected
    assert result.cache_length == result.expected_cache_length
    assert "cache_values" in {failure.kind for failure in result.failures}


def test_sibling_attention_fault_is_detected(models, monkeypatch):
    target, _ = models
    case = sample(models)
    tree = DraftTree.from_branches([list(case.drafts), [11, 12, 13, 14]])
    case = replace(case, drafts=tuple(tree.tokens), parents=tuple(tree.parent))
    forward = target.model.forward
    injected = []

    @wraps(forward)
    def wrong_mask(*args, **kwargs):
        mask = kwargs.get("attention_mask")
        if mask is not None and mask.ndim == 4:
            kwargs["attention_mask"] = torch.zeros_like(mask)
            injected.append(True)
        return forward(*args, **kwargs)

    monkeypatch.setattr(target.model, "forward", wrong_mask)
    result = check_verifier_state(*models, case)
    assert injected and not result.passed
    assert any(failure.kind in {"committed", "cache_values", "cache_length"} for failure in result.failures)


def test_execution_failure_is_reported_without_mutating_reference(models, monkeypatch):
    case = sample(models)

    def fail(*args):
        raise RuntimeError("injected")

    monkeypatch.setattr(models[0], "forward_tree", fail)
    result = check_verifier_state(*models, case)
    assert not result.passed and result.actual is None
    assert result.failures[0].kind == "exception"
    assert result.failures[0].detail == "RuntimeError: injected"


def test_followup_fault_is_detected_after_a_correct_round(models, monkeypatch):
    target, _ = models
    case = sample(models)
    forward = target.forward_block

    def wrong_next(state, tokens):
        logits = forward(state, tokens).clone()
        index = (int(logits[-1].argmax()) + 1) % logits.shape[-1]
        logits[-1, index] = logits.max() + 1
        return logits

    monkeypatch.setattr(target, "forward_block", wrong_next)
    result = check_verifier_state(*models, case)
    assert result.actual == result.expected and result.cache_length == result.expected_cache_length
    assert "followup_token" in {failure.kind for failure in result.failures}


def test_boolean_tokens_cannot_pass_as_integer_ids(models, monkeypatch):
    for model in (models[0].model, models[1]):
        with torch.no_grad():
            model.lm_head.weight.zero_()
    case = sample(models)
    verify = state_checks.verify_tree_greedy

    def bool_tokens(*args):
        result = verify(*args)
        result.committed = [False] * len(result.committed)
        return result

    monkeypatch.setattr(state_checks, "verify_tree_greedy", bool_tokens)
    result = check_verifier_state(*models, case)
    assert "committed" in {failure.kind for failure in result.failures}


@pytest.mark.parametrize("change,match", [
    ({"verifier": "unknown"}, "verifier"),
    ({"atol": float("inf")}, "tolerances"),
    ({"rtol": -1}, "tolerances"),
])
def test_invalid_options_fail_before_model_execution(models, change, match):
    case = sample(models)
    with pytest.raises(ValueError, match=match):
        check_verifier_state(*models, case, **change)


def test_shared_reference_and_mismatched_eos_are_rejected(models):
    case = sample(models)
    with pytest.raises(ValueError, match="separate"):
        check_verifier_state(models[0], models[0].model, case)
    models[1].generation_config.eos_token_id = 2
    with pytest.raises(ValueError, match="EOS"):
        check_verifier_state(*models, case)


def test_context_and_branch_validation(models):
    case = sample(models)
    with pytest.raises(ValueError, match="context"):
        check_verifier_state(*models, replace(case, prefix=(5,) * 1020))
    with pytest.raises(ValueError, match="vocabulary"):
        check_verifier_state(*models, replace(case, pending=211))
    branch = replace(case, parents=(-1, -1, 0, 1))
    with pytest.raises(ValueError, match="chain"):
        check_verifier_state(*models, branch, verifier="block")
    with pytest.raises(ValueError, match="preceding"):
        replace(case, parents=(1, -1, 0, 1))
