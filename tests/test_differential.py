from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dart.verification.differential import (
    CheckReport,
    GenerationCase,
    run_differential,
)


def run(actual, expected=(5, 6, 7), budget=3, eos=(2, 3)):
    return run_differential(
        [GenerationCase("sample", (8, 9), budget)],
        lambda case: expected, {"candidate": lambda case: actual}, eos_token_ids=eos,
    )


@pytest.mark.parametrize("tokens,budget", [([], 0), ([5, 6, 7], 3), ([5, 2], 8), ([3], 4)])
def test_complete_matches_include_zero_budget_and_multiple_eos(tokens, budget):
    report = run(tokens, tokens, budget)
    assert report.passed and not report.failures
    assert report.results[0].expected == report.results[0].actual == tuple(tokens)


@pytest.mark.parametrize("actual,expected,budget,kind,index", [
    ([5, 4, 7], [5, 6, 7], 3, "tokens", 1),
    ([5, 6], [5, 6, 7], 3, "length", 2),
    ([5, 6, 7, 8], [5, 6, 7], 3, "budget", 3),
    ([5, 6, 7], [5, 2], 3, "eos", 1),
    ([5, 2], [5, 6, 7], 3, "eos", 1),
    ([5, 2, 7], [5, 2], 3, "eos", 1),
    ([5, 3], [5, 2], 3, "tokens", 1),
    ([5], [], 0, "budget", 0),
])
def test_reports_first_failure_location(actual, expected, budget, kind, index):
    report = run(actual, expected, budget)
    assert not report.passed and len(report.failures) == 1
    failure = report.failures[0].failure
    assert (failure.kind, failure.index) == (kind, index)
    assert report.failures[0].case.prompt == (8, 9)


@pytest.mark.parametrize("output", [[True, 6, 7], [5.0, 6, 7], [-1, 6, 7], None, "", b"", iter([5, 6, 7])])
def test_invalid_candidate_output_fails(output):
    report = run(output)
    assert not report.passed and report.failures[0].failure.kind == "output"
    assert report.failures[0].actual is None


@pytest.mark.parametrize("expected", [[], [5, 6], [5, 2, 7], [5, 6, 7, 8], [False, 6, 7]])
def test_shared_invalid_outputs_cannot_pass(expected):
    with pytest.raises(ValueError):
        run(expected, expected)


def test_candidate_error_does_not_hide_later_cases_or_backends():
    cases = [GenerationCase(str(index), (5,), 1) for index in range(2)]

    def broken(case):
        raise RuntimeError("injected")

    report = run_differential(cases, lambda case: [7], {"broken": broken, "ok": lambda case: [7]})
    assert len(report.results) == 4 and len(report.failures) == 2
    assert all(row.failure.kind == "exception" for row in report.failures)
    assert report.failures[0].failure.detail == "RuntimeError: injected"
    assert [row.passed for row in report.results] == [False, True, False, True]


def test_reference_error_aborts_before_candidate_execution():
    def reference(case):
        raise RuntimeError("reference failed")

    with pytest.raises(RuntimeError, match="reference failed"):
        run_differential([GenerationCase("a", (5,), 1)], reference,
                         {"unused": lambda case: pytest.fail("candidate ran")})


def test_interrupts_are_not_reported_as_mismatches():
    def interrupt(case):
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_differential([GenerationCase("a", (5,), 1)], lambda case: [7], {"interrupt": interrupt})


def test_snapshots_are_owned_and_case_cannot_be_mutated():
    prompt, output = [8, 9], [5, 6, 7]
    case = GenerationCase("a", prompt, 3)
    report = run_differential([case], lambda case: output, {"a": lambda case: output})
    prompt.clear()
    output.clear()
    row = report.results[0]
    assert row.case.prompt == (8, 9) and row.expected == row.actual == (5, 6, 7)
    with pytest.raises(FrozenInstanceError):
        case.max_new_tokens = 0


@pytest.mark.parametrize("kwargs", [
    {"name": ""}, {"prompt": []}, {"prompt": [True]}, {"prompt": [-1]},
    {"max_new_tokens": -1}, {"max_new_tokens": True}, {"draft_len": 1.5},
])
def test_invalid_case_rejected(kwargs):
    values = {"name": "a", "prompt": (5,), "max_new_tokens": 1, **kwargs}
    with pytest.raises(ValueError):
        GenerationCase(**values)


@pytest.mark.parametrize("cases,candidates", [
    ([], {"a": lambda case: [7]}),
    ([GenerationCase("a", (5,), 1)] * 2, {"a": lambda case: [7]}),
    ([GenerationCase("a", (5,), 1)], {}),
    ([GenerationCase("a", (5,), 1)], {"": lambda case: [7]}),
])
def test_empty_or_ambiguous_suites_rejected(cases, candidates):
    with pytest.raises(ValueError):
        run_differential(cases, lambda case: [7], candidates)
    assert not CheckReport(()).passed
