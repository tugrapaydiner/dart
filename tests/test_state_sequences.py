from __future__ import annotations

import gc
import weakref

import pytest
import torch

from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_sequences import StateOperation as Op
from dart.verification.state_sequences import check_state_sequence
from dart.verification.tree_verify import DraftTree
from tests.test_state_checks import models as models


def kinds(step, request):
    return {item.failure.kind for item in step.failures if item.request == request}


def test_interleaved_forward_crop_to_zero_and_release_reuse(models):
    operations = [
        Op("a", "prefill", (5, 6, 7)), Op("b", "prefill", (17, 19, 23)),
        Op("a", "forward", (11, 12)), Op("b", "crop", length=2),
        Op("b", "forward", (27,)), Op("a", "forward", (13,)),
        Op("a", "crop", length=0), Op("b", "forward", (28,)),
        Op("a", "forward", (31,)), Op("a", "release"),
        Op("a", "prefill", (37, 41)), Op("b", "release"),
        Op("a", "forward", (43,)), Op("a", "release"),
    ]
    report = check_state_sequence(*models, operations)
    assert report.passed, report
    assert len(report.steps) == len(operations)
    assert set(report.steps[7].checked_requests) == {"a", "b"}
    assert report.steps[9].checked_requests == ("b",)
    assert report.steps[-1].checked_requests == ()


def test_interleaved_block_and_branching_tree_rounds(models):
    reference = models[1]
    a, b = (5, 6, 7), (17, 19, 23)
    decode = hf_decoder(reference)
    left = decode(GenerationCase("a", a, 6))
    right = decode(GenerationCase("b", b, 6))
    tree = DraftTree.from_branches([[(right[1] + 1) % 211, 11], right[1:4]])
    operations = [Op("a", "prefill", a), Op("b", "prefill", b),
                  Op("a", "block", tuple(left[:4])),
                  Op("b", "tree", (right[0], *tree.tokens), tuple(tree.parent)),
                  Op("a", "forward", (left[4],)), Op("b", "forward", (right[4],)),
                  Op("a", "block", (left[5],)), Op("b", "tree", (right[5],)),
                  Op("a", "release"), Op("b", "release")]
    report = check_state_sequence(*models, operations)
    assert report.passed, report


def test_idle_request_value_corruption_is_detected_immediately(models, monkeypatch):
    target, _ = models
    prefill, forward = target.prefill, target.forward_block
    states = {}

    def remember(tokens):
        state, logits = prefill(tokens)
        states[tokens[0]] = state
        return state, logits

    def leak(state, tokens):
        logits = forward(state, tokens)
        if state is states[17]:
            states[5].cache.layers[0].keys.add_(0.1)
        return logits

    monkeypatch.setattr(target, "prefill", remember)
    monkeypatch.setattr(target, "forward_block", leak)
    report = check_state_sequence(*models, [Op("a", "prefill", (5, 6, 7)),
                                          Op("b", "prefill", (17, 19, 23)), Op("b", "forward", (29,))])
    assert not report.passed and all(step.passed for step in report.steps[:2])
    assert kinds(report.steps[-1], "a") == {"cache_values"}
    assert not kinds(report.steps[-1], "b")


def test_aliased_request_caches_are_detected(models, monkeypatch):
    target, _ = models
    prefill, saved = target.prefill, []

    def alias(tokens):
        state, logits = prefill(tokens)
        saved.append(state)
        return saved[0], logits

    monkeypatch.setattr(target, "prefill", alias)
    report = check_state_sequence(*models, [Op("a", "prefill", (5, 6, 7)),
                                          Op("b", "prefill", (17, 19, 23)), Op("b", "forward", (29,))])
    assert not report.passed
    assert "cache_values" in kinds(report.steps[1], "b")
    assert "cache_length" in kinds(report.steps[2], "a")


def test_expected_admission_errors_preserve_states_and_allow_continuation(models):
    report = check_state_sequence(*models, [
        Op("a", "prefill", (5, 6, 7)), Op("b", "prefill", (17, 19)),
        Op("a", "crop", length=-1, expect_error="ValueError"),
        Op("a", "crop", length=4, expect_error="ValueError"),
        Op("a", "forward", (5,) * 1024, expect_error="ValueError"),
        Op("a", "forward", (31,)), Op("b", "forward", (37,)),
    ])
    assert report.passed, report


@pytest.mark.parametrize("action", ["prefill", "forward"])
def test_missing_expected_error_is_a_failure(models, action):
    setup = [] if action == "prefill" else [Op("a", "prefill", (5, 6, 7))]
    report = check_state_sequence(*models, [*setup, Op("a", action, (17,), expect_error="ValueError")])
    assert not report.passed and "missing_exception" in kinds(report.steps[-1], "a")


def test_partial_error_is_retained_while_other_request_and_recreated_state_recover(models, monkeypatch):
    target, _ = models
    forward = target.forward_block
    calls = []

    def partial(state, tokens):
        logits = forward(state, tokens)
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("injected after cache update")
        return logits

    monkeypatch.setattr(target, "forward_block", partial)
    operations = [Op("a", "prefill", (5, 6, 7)), Op("b", "prefill", (17, 19)),
                  Op("a", "forward", (11,)), Op("b", "forward", (23,)),
                  Op("a", "release"), Op("a", "prefill", (29, 31)), Op("a", "forward", (37,))]
    report = check_state_sequence(*models, operations)
    assert not report.passed and len(report.steps) == len(operations)
    assert kinds(report.steps[2], "a") == {"exception", "cache_length"}
    assert not kinds(report.steps[3], "b")
    assert all(step.passed for step in report.steps[4:])


def test_failed_prefill_does_not_abort_remaining_operations(models, monkeypatch):
    target, _ = models
    prefill, calls = target.prefill, []

    def fail_once(tokens):
        calls.append(True)
        if len(calls) == 1:
            raise RuntimeError("injected before prefill")
        return prefill(tokens)

    monkeypatch.setattr(target, "prefill", fail_once)
    operations = [Op("a", "prefill", (5, 6)), Op("a", "block", (7,)),
                  Op("a", "release"), Op("a", "prefill", (17, 19)), Op("a", "forward", (23,))]
    report = check_state_sequence(*models, operations)
    assert not report.passed and len(report.steps) == len(operations)
    assert all(step.passed for step in report.steps[2:])


def test_verification_can_follow_crop_to_empty(models):
    report = check_state_sequence(*models, [Op("a", "prefill", (5, 6, 7)), Op("a", "crop", length=0),
                                          Op("a", "block", (11, 13)), Op("a", "release")])
    assert report.passed, report


def test_malformed_output_is_reported_and_later_steps_run(models, monkeypatch):
    target, _ = models
    forward = target.forward_block

    def missing_logits(state, tokens):
        logits = forward(state, tokens)
        return None if tokens == [11] else logits

    monkeypatch.setattr(target, "forward_block", missing_logits)
    report = check_state_sequence(*models, [Op("a", "prefill", (5, 6, 7)), Op("a", "forward", (11,)),
                                          Op("a", "forward", (13,)), Op("a", "release")])
    assert not report.passed and kinds(report.steps[1], "a") == {"output"}
    assert all(step.passed for step in report.steps[2:])


def test_reference_failure_aborts_before_candidate_execution(models, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("reference failed")

    monkeypatch.setattr(models[1], "forward", fail)
    monkeypatch.setattr(models[0], "prefill", lambda *args: pytest.fail("candidate executed"))
    with pytest.raises(RuntimeError, match="reference failed"):
        check_state_sequence(*models, [Op("a", "prefill", (5, 6, 7))])


def test_unbounded_iterators_are_rejected_without_consumption(models):
    def operations():
        pytest.fail("consumed an unbounded iterator")
        yield Op("a", "prefill", (5,))

    with pytest.raises(ValueError, match="sequence"):
        check_state_sequence(*models, operations())


def test_results_do_not_retain_released_states_or_write_files(models, monkeypatch, tmp_path):
    target, _ = models
    prefill, seen = target.prefill, []

    def remember(tokens):
        state, logits = prefill(tokens)
        seen.append(weakref.ref(state))
        return state, logits

    monkeypatch.setattr(target, "prefill", remember)
    monkeypatch.chdir(tmp_path)
    rng = torch.random.get_rng_state().clone()
    operations = [Op("a", "prefill", (5, 6, 7)), Op("a", "release")]
    first = check_state_sequence(*models, operations)
    second = check_state_sequence(*models, operations)
    gc.collect()
    assert first == second and first.passed
    assert all(reference() is None for reference in seen)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("operations", [
    [], [Op("a", "release")], [Op("a", "forward", (5,))],
    [Op("a", "prefill", (5,))] * 2,
    [Op(str(index), "prefill", (5,)) for index in range(9)],
    [Op(str(index), "prefill", (), expect_error="ValueError") for index in range(9)],
    [None],
    [Op("a", "prefill", (5,)), *[Op("a", "forward", (5,)) for _ in range(64)]],
])
def test_bad_schedules_fail_before_execution(models, monkeypatch, operations):
    monkeypatch.setattr(models[0], "prefill", lambda *args: pytest.fail("executed an invalid schedule"))
    with pytest.raises(ValueError):
        check_state_sequence(*models, operations)


@pytest.mark.parametrize("kwargs", [
    {"action": "unknown"}, {"request": ""}, {"tokens": [True]}, {"tokens": [-1]},
    {"length": 0}, {"parents": [0]}, {"expect_error": "anything"},
])
def test_invalid_operation_fields(kwargs):
    with pytest.raises(ValueError):
        Op(**{"request": "a", "action": "prefill", "tokens": (5,), **kwargs})
