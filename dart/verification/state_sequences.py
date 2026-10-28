"""Replay bounded request operations and check every live cache after each step."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dart.verification.block_verify import verify_block_greedy
from dart.verification.differential import CheckFailure, InvalidCheckCase
from dart.verification.state_checks import (
    RoundCase,
    _cache_failures,
    _cache_pairs,
    _reference_forward,
    _result_failures,
    _round_expectation,
    _validate_tolerances,
)
from dart.verification.tree_verify import DraftTree, verify_tree_greedy


@dataclass(frozen=True)
class StateOperation:
    request: str
    action: str
    tokens: tuple[int, ...] = ()
    parents: tuple[int, ...] = ()
    length: int | None = None
    expect_error: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "tokens", tuple(self.tokens))
        object.__setattr__(self, "parents", tuple(self.parents))
        if not isinstance(self.request, str) or not self.request:
            raise ValueError("request name must be nonempty")
        if self.action not in {"prefill", "forward", "crop", "block", "tree", "release"}:
            raise ValueError("unknown state operation")
        if self.expect_error not in {None, "ValueError", "RuntimeError", "IndexError"}:
            raise ValueError("unsupported expected exception")
        if any(type(token) is not int or token < 0 for token in self.tokens):
            raise ValueError("tokens must be nonnegative integer IDs")
        if self.action in {"block", "tree"} and not self.tokens:
            raise ValueError("verification needs a pending token")
        if self.action in {"prefill", "forward"} and not self.tokens and self.expect_error is None:
            raise ValueError("prefill and forward need tokens")
        if self.action == "crop":
            if type(self.length) is not int:
                raise ValueError("crop needs an integer length")
        elif self.length is not None:
            raise ValueError("length is only used by crop")
        if self.action in {"crop", "release"} and self.tokens:
            raise ValueError("crop and release do not take tokens")
        if self.action != "tree" and self.parents:
            raise ValueError("parents are only used by tree verification")
        if self.action == "tree":
            RoundCase(self.request, (0,), self.tokens[0], self.tokens[1:], self.parents)
        if self.action == "release" and self.expect_error is not None:
            raise ValueError("release cannot expect an exception")


@dataclass(frozen=True)
class StateFailure:
    request: str
    failure: CheckFailure


@dataclass(frozen=True)
class StateStepCheck:
    operation: StateOperation
    checked_requests: tuple[str, ...]
    failures: tuple[StateFailure, ...]

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class StateSequenceCheck:
    steps: tuple[StateStepCheck, ...]

    @property
    def passed(self) -> bool:
        return bool(self.steps) and all(step.passed for step in self.steps)


def _validate_operations(operations):
    if any(not isinstance(operation, StateOperation) for operation in operations):
        raise InvalidCheckCase("provide StateOperation values")
    if len({operation.request for operation in operations}) > 8:
        raise InvalidCheckCase("at most eight request names are allowed")
    live = set()
    for operation in operations:
        if operation.action == "prefill":
            if operation.request in live:
                raise InvalidCheckCase("release a request before prefilling it again")
            if operation.expect_error is None:
                live.add(operation.request)
        elif operation.request not in live:
            raise InvalidCheckCase("operation uses a request that is not live")
        elif operation.action == "release":
            live.remove(operation.request)


@torch.inference_mode()
def check_state_sequence(target, reference_model, operations: Sequence[StateOperation], *,
                         atol: float = 2e-6, rtol: float = 2e-5) -> StateSequenceCheck:
    """Check interleaving, rollback and release/reuse with independent HF state.

    A block/tree operation takes pending first, then drafts. An expected error
    must leave all live states unchanged. Unexpected errors remain failures;
    later operations still run so recovery can be inspected. Release drops the
    request state; it does not call a backend's close method. No files are written.
    """
    if not isinstance(operations, Sequence) or not 1 <= len(operations) <= 64:
        raise InvalidCheckCase("provide a sequence of between 1 and 64 operations")
    operations = tuple(operations)
    _validate_operations(operations)
    _validate_tolerances(atol, rtol)
    if target.model is reference_model or target.model.training or reference_model.training:
        raise ValueError("use separate eval-mode target and reference models")
    context = min(target.model.config.max_position_embeddings, reference_model.config.max_position_embeddings)
    vocab = min(target.model.config.vocab_size, reference_model.config.vocab_size)
    states, prefixes, reference_caches, steps = {}, {}, {}, []
    for operation in operations:
        request, action = operation.request, operation.action
        failures = []

        def record(kind, detail, request=request, failures=failures):
            failures.append(StateFailure(request, CheckFailure(kind, detail=detail)))

        if action == "release":
            states.pop(request, None)
            prefixes.pop(request, None)
            reference_caches.pop(request, None)
        else:
            prefix = prefixes.get(request, ())
            next_prefix, reference_cache, wanted, expected, accepted = prefix, None, None, None, None
            reference_available = action == "prefill" or request in prefixes
            if operation.expect_error is None and reference_available:
                if max(operation.tokens, default=0) >= vocab:
                    raise InvalidCheckCase("operation contains a token outside the model vocabulary")
                if action == "crop":
                    if not 0 <= operation.length <= len(prefix):
                        raise InvalidCheckCase("crop length is outside the reference prefix")
                    next_prefix = prefix[:operation.length]
                elif action in {"prefill", "forward"}:
                    next_prefix = operation.tokens if action == "prefill" else prefix + operation.tokens
                else:
                    parents = operation.parents if action == "tree" else tuple(range(-1, len(operation.tokens) - 2))
                    case = RoundCase(request, prefix, operation.tokens[0], operation.tokens[1:], parents)
                    expected, accepted = _round_expectation(target, reference_model, case, action)
                    next_prefix = (*prefix, operation.tokens[0], *expected[:-1])
                if len(next_prefix) > context:
                    raise InvalidCheckCase("operation exceeds the reference context limit")
                if next_prefix:
                    output = _reference_forward(reference_model, next_prefix)
                    reference_cache = output.past_key_values
                    if action in {"prefill", "forward"}:
                        count = 1 if action == "prefill" else len(operation.tokens)
                        wanted = output.logits[0, -count:].float()
                else:
                    reference_cache = [(key[..., :0, :], value[..., :0, :])
                                       for key, value in _cache_pairs(reference_caches[request])]
            try:
                if action == "prefill":
                    state, logits = target.prefill(list(operation.tokens))
                    states[request] = state
                    logits = logits.unsqueeze(0)
                elif request not in states or not reference_available:
                    raise RuntimeError("request state is unavailable after an earlier failure")
                elif action == "forward":
                    logits = target.forward_block(states[request], list(operation.tokens))
                elif action == "crop":
                    target.crop(states[request], operation.length)
                elif action == "block":
                    result = verify_block_greedy(target, states[request], operation.tokens[0], list(operation.tokens[1:]))
                else:
                    tree = DraftTree(list(operation.tokens[1:]), list(operation.parents))
                    result = verify_tree_greedy(target, states[request], operation.tokens[0], tree)
            except Exception as error:
                if type(error).__name__ != operation.expect_error:
                    record("exception", f"{type(error).__name__}: {error}")
            else:
                if operation.expect_error is not None:
                    record("missing_exception", f"expected {operation.expect_error}")
                else:
                    prefixes[request], reference_caches[request] = tuple(next_prefix), reference_cache
                    try:
                        if wanted is not None:
                            if logits.shape != wanted.shape:
                                record("logits", "output shape differs")
                            else:
                                if logits.argmax(-1).tolist() != wanted.argmax(-1).tolist():
                                    record("tokens", "next-token IDs differ")
                                if not torch.allclose(logits.to(wanted.device), wanted, atol=atol, rtol=rtol):
                                    record("logits", "output differs beyond tolerance")
                        if expected is not None:
                            failures.extend(StateFailure(request, failure) for failure in
                                            _result_failures(result, expected, accepted, len(operation.tokens) - 1,
                                                             target.eos_token_ids))
                    except Exception as error:
                        record("output", f"{type(error).__name__}: {error}")
        for name, state in states.items():
            if name not in reference_caches:
                failures.append(StateFailure(name, CheckFailure("unexpected_state", detail="state exists after an expected error")))
                continue
            try:
                failures.extend(StateFailure(name, failure) for failure in
                                _cache_failures(state, reference_caches[name], len(prefixes[name]), atol=atol, rtol=rtol))
            except Exception as error:
                failures.append(StateFailure(name, CheckFailure("cache_error", detail=f"{type(error).__name__}: {error}")))
        steps.append(StateStepCheck(operation, tuple(states), tuple(failures)))
    return StateSequenceCheck(tuple(steps))
