from __future__ import annotations

from dataclasses import replace
from itertools import repeat

import pytest

from dart.verification.differential import (
    CheckFailure,
    CheckReport,
    CheckResult,
    GenerationCase,
    InvalidCheckCase,
)
from dart.verification.reduction import (
    FailureKey,
    case_size,
    failure_key,
    reduce_failure,
    smaller_cases,
)
from dart.verification.state_checks import RoundCase, RoundCheck, check_verifier_state
from dart.verification.state_sequences import (
    StateFailure,
    StateOperation,
    StateSequenceCheck,
    StateStepCheck,
    _validate_operations,
    check_state_sequence,
)
from tests.test_state_checks import models as models
from tests.test_state_checks import sample

CASE = GenerationCase("test", (13, 17, 19, 23), 8, 4)
KEY = FailureKey("tokens", "block", "test")


def test_reduces_input_and_budgets_without_changing_original():
    seen = []

    def evaluate(case):
        seen.append(case)
        return KEY if 17 in case.prompt and case.max_new_tokens >= 2 and case.draft_len >= 1 else None

    result = reduce_failure(CASE, evaluate, max_evaluations=64)
    assert result.reproduced and result.stop_reason == "search_exhausted"
    assert result.reduced == replace(CASE, prompt=(17,), max_new_tokens=2, draft_len=1)
    assert result.original == CASE and CASE.prompt == (13, 17, 19, 23)
    assert result.evaluations == len(seen) <= 64
    assert seen[-1] == result.reduced and result.accepted > 0
    assert reduce_failure(CASE, evaluate, max_evaluations=64) == result


@pytest.mark.parametrize("changed", [
    replace(KEY, kind="exception"), replace(KEY, backend="tree"),
    replace(KEY, request="another"), replace(KEY, actor="another"),
    replace(KEY, error_type="RuntimeError"),
])
def test_different_failure_cannot_replace_original(changed):
    result = reduce_failure(CASE, lambda case: KEY if case == CASE else changed)
    assert result.reproduced and result.reduced == CASE and result.accepted == 0


def test_first_failure_key_keeps_backend_request_actor_and_error_type():
    failure = CheckFailure("exception", 3, "RuntimeError: synthetic failure")
    differential = CheckReport((CheckResult(CASE, "block", (), None, failure),))
    single = RoundCheck(RoundCase("test", (5,), 7, (), ()), "block", (), None, 1, None, (failure,))
    sequence = StateSequenceCheck((StateStepCheck(StateOperation("b", "forward", (7,)), ("a", "b"),
                                                (StateFailure("a", failure),)),))
    assert failure_key(differential) == failure_key(single) == FailureKey("exception", "block", "test", "", "RuntimeError")
    assert failure_key(sequence) == FailureKey("exception", "forward", "a", "b", "RuntimeError")
    shifted = replace(differential.results[0], failure=replace(failure, index=0))
    assert failure_key(CheckReport((shifted,))) == failure_key(differential)
    assert failure_key(CheckReport((replace(shifted, failure=None),))) is None
    assert failure_key(replace(single, failures=())) is None
    assert failure_key(StateSequenceCheck((replace(sequence.steps[0], failures=()),))) is None
    with pytest.raises(ValueError, match="details"):
        failure_key(CheckReport(()))
    with pytest.raises(ValueError, match="details"):
        failure_key(StateSequenceCheck(()))


@pytest.mark.parametrize("budget", [2, 3, 7])
def test_evaluation_budget_includes_initial_and_final_checks(budget):
    seen = []

    def evaluate(case):
        seen.append(case)
        return KEY

    result = reduce_failure(CASE, evaluate, max_evaluations=budget,
                            candidates=lambda case: [replace(case, max_new_tokens=case.max_new_tokens - 1)])
    assert result.evaluations == len(seen) == budget
    assert result.stop_reason == "evaluation_budget" and result.budget_exhausted
    assert result.reproduced and seen[-1] == result.reduced


def test_infinite_duplicate_proposals_stop_without_extra_evaluations():
    result = reduce_failure(CASE, lambda case: KEY, candidates=lambda case: repeat(case), max_candidates=9)
    assert result.evaluations == 2 and result.candidates == 9 and result.accepted == 0
    assert result.stop_reason == "candidate_budget" and result.budget_exhausted and result.reproduced


def test_nonshrinking_candidates_are_never_executed():
    bigger = replace(CASE, prompt=(*CASE.prompt, 31))
    result = reduce_failure(CASE, lambda case: KEY if case == CASE else pytest.fail("executed larger case"),
                            candidates=lambda case: [bigger, CASE])
    assert result.evaluations == 2 and result.candidates == 2 and result.reduced == CASE
    assert result.stop_reason == "search_exhausted" and not result.budget_exhausted


@pytest.mark.parametrize("final", [None, replace(KEY, kind="length"), InvalidCheckCase("now invalid")])
def test_final_replay_must_reproduce_selected_failure(final):
    outcomes = iter([KEY, final])

    def evaluate(case):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    result = reduce_failure(CASE, evaluate, candidates=lambda case: [])
    assert not result.reproduced and result.evaluations == 2 and result.failure == KEY
    assert result.final_failure == (None if isinstance(final, Exception) else final)


def test_invalid_candidate_is_counted_and_skipped():
    smaller = replace(CASE, prompt=(17,))
    seen = []

    def evaluate(case):
        seen.append(case)
        if case != CASE:
            raise InvalidCheckCase("invalid reduced schedule")
        return KEY

    result = reduce_failure(CASE, evaluate, candidates=lambda case: [smaller, smaller])
    assert result.reproduced and result.reduced == CASE
    assert seen == [CASE, smaller, CASE] and result.evaluations == 3 and result.candidates == 2


@pytest.mark.parametrize("error", [RuntimeError("reference failed"), ValueError("reference config changed")])
def test_unexpected_errors_abort_instead_of_becoming_nonfailures(error):
    def evaluate(case):
        if case != CASE:
            raise error
        return KEY

    with pytest.raises(type(error), match=str(error)):
        reduce_failure(CASE, evaluate)


def test_passing_or_invalid_original_cannot_be_reduced():
    with pytest.raises(ValueError, match="does not reproduce"):
        reduce_failure(CASE, lambda case: None)

    def invalid(case):
        raise InvalidCheckCase("original is invalid")

    with pytest.raises(InvalidCheckCase, match="original is invalid"):
        reduce_failure(CASE, invalid)


@pytest.mark.parametrize("kwargs,match", [
    ({"max_evaluations": 1}, "max_evaluations"), ({"max_evaluations": True}, "max_evaluations"),
    ({"max_candidates": 0}, "max_candidates"), ({"max_candidates": 2.5}, "max_candidates"),
    ({"size": lambda case: ()}, "size"), ({"size": lambda case: (-1,)}, "size"),
    ({"size": lambda case: (True,)}, "size"),
])
def test_invalid_limits_fail_before_evaluation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        reduce_failure(CASE, lambda case: pytest.fail("evaluated with invalid limits"), **kwargs)


def test_callback_contracts_are_checked():
    with pytest.raises(TypeError, match="FailureKey"):
        reduce_failure(CASE, lambda case: True)
    with pytest.raises(TypeError, match="case type"):
        reduce_failure(CASE, lambda case: KEY, candidates=lambda case: [()])
    with pytest.raises(ValueError, match="length"):
        reduce_failure(CASE, lambda case: KEY, size=lambda case: (1,) if case == CASE else (0, 0))
    with pytest.raises(ValueError, match="kind"):
        FailureKey("")
    with pytest.raises(ValueError, match="string"):
        FailureKey("tokens", backend=1)


def test_tree_pruning_preserves_ancestry_and_remaps_indices():
    original = RoundCase("branch", (13, 17), 19, (23, 29, 31, 37, 41), (-1, 0, -1, 2, 3))
    proposals = list(smaller_cases(original))
    assert replace(original, drafts=(31, 37, 41), parents=(-1, 0, 1)) in proposals
    assert replace(original, drafts=(23, 29), parents=(-1, 0)) in proposals
    assert replace(original, drafts=(), parents=()) in proposals
    for proposed in proposals:
        assert case_size(proposed) < case_size(original)
        if proposed.drafts == original.drafts or any(token not in original.drafts for token in proposed.drafts):
            continue
        for index, token in enumerate(proposed.drafts):
            old_parent = original.parents[original.drafts.index(token)]
            expected_parent = None if old_parent == -1 else original.drafts[old_parent]
            parent = proposed.parents[index]
            assert (None if parent == -1 else proposed.drafts[parent]) == expected_parent


def test_sequence_proposals_keep_valid_request_lifetimes_and_tree_structure():
    operations = (
        StateOperation("unused", "prefill", (43,)), StateOperation("unused", "release"),
        StateOperation("a", "prefill", (5, 7)),
        StateOperation("a", "tree", (11, 13, 17, 19), (-1, -1, 1)),
        StateOperation("a", "crop", length=1), StateOperation("a", "release"),
        StateOperation("a", "prefill", (23,)), StateOperation("a", "forward", (29,)),
    )
    proposals = list(smaller_cases(operations))
    assert operations[2:] in proposals
    for proposed in proposals:
        _validate_operations(proposed)
        assert case_size(proposed) < case_size(operations)


def test_real_rollback_failure_reduces_and_replays(models, monkeypatch):
    target, _ = models
    original = sample(models, rejection=0)
    crop = target.crop
    monkeypatch.setattr(target, "crop", lambda *args: None)

    def evaluate(case):
        return failure_key(check_verifier_state(*models, case, verifier="block"))

    result = reduce_failure(original, evaluate, max_evaluations=32)
    assert result.reproduced and result.failure.kind == "cache_length"
    assert len(result.reduced.drafts) == 1 and len(result.reduced.prefix) == 1
    assert case_size(result.reduced) < case_size(original)
    assert evaluate(result.reduced) == result.failure
    monkeypatch.setattr(target, "crop", crop)
    assert evaluate(result.reduced) is None


def test_real_idle_cache_corruption_reduces_to_two_requests_and_one_forward(models, monkeypatch):
    target, _ = models
    prefill, forward = target.prefill, target.forward_block
    states = []

    def remember(tokens):
        state, logits = prefill(tokens)
        states.append(state)
        return state, logits

    def leak(state, tokens):
        logits = forward(state, tokens)
        if len(states) > 1 and state is not states[0]:
            states[0].cache.layers[0].keys.add_(0.1)
        return logits

    monkeypatch.setattr(target, "prefill", remember)
    monkeypatch.setattr(target, "forward_block", leak)
    original = (StateOperation("a", "prefill", (5, 6, 7)), StateOperation("b", "prefill", (17, 19)),
                StateOperation("unused", "prefill", (31,)), StateOperation("unused", "release"),
                StateOperation("b", "forward", (23, 29)), StateOperation("b", "release"))

    def evaluate(operations):
        states.clear()  # Each reduction trial needs fresh transient state.
        return failure_key(check_state_sequence(*models, operations))

    result = reduce_failure(original, evaluate, max_evaluations=48)
    assert result.reproduced and result.failure == FailureKey("cache_values", "forward", "a", "b")
    assert [(op.request, op.action) for op in result.reduced] == [("a", "prefill"), ("b", "prefill"), ("b", "forward")]
    assert all(len(op.tokens) == 1 for op in result.reduced)
    assert evaluate(result.reduced) == result.failure
    monkeypatch.setattr(target, "forward_block", forward)
    assert evaluate(result.reduced) is None


def test_reduced_crop_cannot_hide_a_reference_configuration_error(models):
    operations = (StateOperation("a", "prefill", (5, 7)), StateOperation("a", "crop", length=2))
    invalid = (replace(operations[0], tokens=(5,)), operations[1])
    with pytest.raises(InvalidCheckCase, match="crop"):
        check_state_sequence(*models, invalid)
    models[1].train()
    with pytest.raises(ValueError, match="eval-mode") as error:
        check_state_sequence(*models, operations)
    assert not isinstance(error.value, InvalidCheckCase)
