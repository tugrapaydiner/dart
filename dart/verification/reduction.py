"""Bounded reduction of synthetic cases, preserving a selected failure."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import islice

from dart.verification.differential import CheckReport, GenerationCase, InvalidCheckCase
from dart.verification.state_checks import RoundCase, RoundCheck
from dart.verification.state_sequences import (
    StateOperation,
    StateSequenceCheck,
    _validate_operations,
)

Case = GenerationCase | RoundCase | tuple[StateOperation, ...]


@dataclass(frozen=True)
class FailureKey:
    kind: str
    backend: str = ""
    request: str = ""
    actor: str = ""
    error_type: str = ""

    def __post_init__(self):
        if not self.kind or any(not isinstance(value, str) for value in
                                (self.kind, self.backend, self.request, self.actor, self.error_type)):
            raise ValueError("failure keys need a kind and string fields")


@dataclass(frozen=True)
class ReductionResult:
    original: Case
    reduced: Case
    failure: FailureKey
    final_failure: FailureKey | None
    evaluations: int
    candidates: int
    accepted: int
    stop_reason: str

    @property
    def reproduced(self) -> bool:
        return self.failure == self.final_failure

    @property
    def budget_exhausted(self) -> bool:
        return self.stop_reason in {"evaluation_budget", "candidate_budget"}


def failure_key(report: CheckReport | RoundCheck | StateSequenceCheck) -> FailureKey | None:
    """Select the first failure, retaining its backend, request and error type."""
    if report.passed:
        return None
    if isinstance(report, CheckReport) and report.failures:
        row = report.failures[0]
        failure, backend, request, actor = row.failure, row.backend, row.case.name, ""
    elif isinstance(report, RoundCheck) and report.failures:
        failure, backend, request, actor = report.failures[0], report.verifier, report.case.name, ""
    elif isinstance(report, StateSequenceCheck):
        step = next((step for step in report.steps if step.failures), None)
        if step is None:
            raise ValueError("report has no failure details")
        item = step.failures[0]
        failure, backend, request, actor = item.failure, step.operation.action, item.request, step.operation.request
    else:
        raise ValueError("report has no failure details")
    error_type = ""
    if failure.kind in {"exception", "output", "cache_error"}:
        prefix, separator, _ = failure.detail.partition(":")
        if separator and prefix.isidentifier():
            error_type = prefix
    return FailureKey(failure.kind, backend, request, actor, error_type)


def _cuts(length, minimum=0):
    size = length - minimum
    while size > 0:
        for start in range(0, length, size):
            end = min(length, start + size)
            if length - (end - start) >= minimum:
                yield start, end
        size //= 2


def _numbers(value):
    sign = -1 if value < 0 else 1
    for smaller in sorted({0, abs(value) // 2, abs(value) - 1}):
        if 0 <= smaller < abs(value):
            yield sign * smaller


def _token_cases(tokens, minimum=0):
    for start, end in _cuts(len(tokens), minimum):
        yield tokens[:start] + tokens[end:]
    if any(tokens):
        yield (0,) * len(tokens)
    for index, token in enumerate(tokens):
        for smaller in _numbers(token):
            yield (*tokens[:index], smaller, *tokens[index + 1:])


def _pruned_trees(tokens, parents):
    for start, end in _cuts(len(tokens)):
        removed = set(range(start, end))
        for index, parent in enumerate(parents):
            if parent in removed:
                removed.add(index)
        kept = [index for index in range(len(tokens)) if index not in removed]
        remap = {old: new for new, old in enumerate(kept)}
        yield (tuple(tokens[index] for index in kept),
               tuple(-1 if parents[index] == -1 else remap[parents[index]] for index in kept))


def _smaller_operation(operation):
    if operation.action in {"prefill", "forward"}:
        minimum = 0 if operation.expect_error else 1
        for tokens in _token_cases(operation.tokens, minimum):
            yield replace(operation, tokens=tokens)
    elif operation.action in {"block", "tree"}:
        drafts = operation.tokens[1:]
        parents = operation.parents if operation.action == "tree" else tuple(range(-1, len(drafts) - 1))
        for tokens, pruned in _pruned_trees(drafts, parents):
            yield replace(operation, tokens=(operation.tokens[0], *tokens),
                          parents=pruned if operation.action == "tree" else ())
        for index, token in enumerate(operation.tokens):
            for smaller in _numbers(token):
                yield replace(operation, tokens=(*operation.tokens[:index], smaller, *operation.tokens[index + 1:]))
    elif operation.action == "crop":
        for length in _numbers(operation.length):
            yield replace(operation, length=length)


def smaller_cases(case: Case) -> Iterable[Case]:
    """Try deletions and simpler IDs; tree pruning removes descendants too."""
    if isinstance(case, GenerationCase):
        for prompt in _token_cases(case.prompt, 1):
            yield replace(case, prompt=prompt)
        for field in ("max_new_tokens", "draft_len"):
            for value in _numbers(getattr(case, field)):
                yield replace(case, **{field: value})
    elif isinstance(case, RoundCase):
        for drafts, parents in _pruned_trees(case.drafts, case.parents):
            yield replace(case, drafts=drafts, parents=parents)
        for prefix in _token_cases(case.prefix, min(1, len(case.prefix))):
            yield replace(case, prefix=prefix)
        for pending in _numbers(case.pending):
            yield replace(case, pending=pending)
        for index, token in enumerate(case.drafts):
            for smaller in _numbers(token):
                yield replace(case, drafts=(*case.drafts[:index], smaller, *case.drafts[index + 1:]))
    elif isinstance(case, tuple) and all(isinstance(operation, StateOperation) for operation in case):
        proposals = []
        for request in dict.fromkeys(operation.request for operation in case):
            proposals.append(tuple(operation for operation in case if operation.request != request))
        for start, end in _cuts(len(case), 1):
            proposals.append(case[:start] + case[end:])
        for proposed in proposals:
            if proposed:
                try:
                    _validate_operations(proposed)
                except InvalidCheckCase:
                    continue
                yield proposed
        for index, operation in enumerate(case):
            for smaller in _smaller_operation(operation):
                yield (*case[:index], smaller, *case[index + 1:])
    else:
        raise TypeError("unsupported reduction case")


def case_size(case: Case) -> tuple[int, ...]:
    if isinstance(case, GenerationCase):
        return (len(case.prompt) + case.max_new_tokens + case.draft_len,
                len(case.prompt), case.max_new_tokens, case.draft_len, sum(case.prompt))
    if isinstance(case, RoundCase):
        return (len(case.prefix) + len(case.drafts), len(case.prefix), len(case.drafts),
                sum(case.prefix) + case.pending + sum(case.drafts))
    if isinstance(case, tuple) and all(isinstance(operation, StateOperation) for operation in case):
        return (len(case), sum(len(operation.tokens) for operation in case),
                sum(abs(operation.length or 0) for operation in case),
                sum(sum(operation.tokens) for operation in case))
    raise TypeError("unsupported reduction case")


def reduce_failure(
    case: Case,
    evaluate: Callable[[Case], FailureKey | None],
    *,
    max_evaluations: int = 32,
    max_candidates: int = 1024,
    candidates: Callable[[Case], Iterable[Case]] = smaller_cases,
    size: Callable[[Case], tuple[int, ...]] = case_size,
) -> ReductionResult:
    """Keep strictly smaller cases with the same failure, then replay the result.

    Evaluation counts include the initial check, invalid candidates and final
    replay. The proposal cap also bounds duplicate/non-shrinking candidates.
    Callbacks are not preempted. Keep model weights/config fixed and reset
    transient state in evaluate. This greedy search does not prove minimality.
    """
    for name, value, minimum in (("max_evaluations", max_evaluations, 2), ("max_candidates", max_candidates, 1)):
        if type(value) is not int or value < minimum:
            raise ValueError(f"{name} must be an integer >= {minimum}")
    evaluations, proposals, accepted = 0, 0, 0

    def measure(value):
        result = size(value)
        if not isinstance(result, tuple) or not result or any(type(item) is not int or item < 0 for item in result):
            raise ValueError("size must return a nonempty tuple of nonnegative integers")
        return result

    def check(value, required=False):
        nonlocal evaluations
        evaluations += 1
        try:
            result = evaluate(value)
        except InvalidCheckCase:
            if required:
                raise
            return None
        if result is not None and not isinstance(result, FailureKey):
            raise TypeError("evaluate must return a FailureKey or None")
        return result

    current, current_size = case, measure(case)
    seen = {case}
    failure = check(case, required=True)
    if failure is None:
        raise ValueError("original case does not reproduce a failure")
    while True:
        if evaluations >= max_evaluations - 1:
            reason = "evaluation_budget"
            break
        if proposals >= max_candidates:
            reason = "candidate_budget"
            break
        changed = False
        for candidate in islice(candidates(current), max_candidates - proposals):
            proposals += 1
            if type(candidate) is not type(current):
                raise TypeError("candidate must keep the case type")
            candidate_size = measure(candidate)
            if len(candidate_size) != len(current_size):
                raise ValueError("size tuple length must stay fixed")
            if candidate in seen or candidate_size >= current_size:
                continue
            seen.add(candidate)
            if check(candidate) == failure:
                current, current_size = candidate, candidate_size
                accepted += 1
                changed = True
                break
            if evaluations >= max_evaluations - 1:
                break
        if not changed and evaluations < max_evaluations - 1 and proposals < max_candidates:
            reason = "search_exhausted"
            break
    final_failure = check(current)
    return ReductionResult(case, current, failure, final_failure, evaluations, proposals, accepted, reason)
