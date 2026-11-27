"""Compare a cached target block with independent full and cached HF paths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch

from dart.verification.differential import CheckFailure, InvalidCheckCase, _tokens
from dart.verification.state_checks import (
    _cache_pairs,
    _check_reference_storage,
    _reference_forward,
    _validate_tolerances,
)


@dataclass(frozen=True)
class ForwardComparison:
    failures: tuple[CheckFailure, ...]
    logits_bitwise: bool = False
    cache_bitwise: bool = False

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class PredictionComparison:
    failures: tuple[CheckFailure, ...]
    cache_bitwise: bool = False

    @property
    def passed(self) -> bool:
        return not self.failures


@dataclass(frozen=True)
class ForwardCheck:
    candidate_full: ForwardComparison
    candidate_incremental: ForwardComparison
    incremental_full: ForwardComparison
    predictions_full: PredictionComparison | None = None
    predictions_incremental: PredictionComparison | None = None

    @property
    def passed(self) -> bool:
        return all(check is None or check.passed for check in (
            self.candidate_full, self.candidate_incremental, self.incremental_full,
            self.predictions_full, self.predictions_incremental,
        ))


@dataclass
class _Snapshot:
    logits: torch.Tensor
    cache: tuple[tuple[torch.Tensor, torch.Tensor], ...]
    length: int
    prefill_logits: torch.Tensor


def _snapshot(logits, cache, length, prefill_logits):
    pairs = _cache_pairs(cache)
    if not pairs:
        raise ValueError("forward checks require a populated KV cache")
    return _Snapshot(logits.detach().clone(),
                     tuple(tuple(tensor.detach().clone() for tensor in pair) for pair in pairs), length, prefill_logits)


def _reference_snapshot(output, width, length, vocab, *, prefill_logits=None):
    full = prefill_logits is None
    if output.logits.shape != (1, length if full else width, vocab):
        raise ValueError("reference returned an invalid logit shape")
    cache = output.past_key_values
    if hasattr(cache, "get_seq_length") and int(cache.get_seq_length()) != length:
        raise ValueError("reference returned an invalid cache length")
    if full:
        prefill_logits = output.logits[0, -width - 1].detach().clone()
    snapshot = _snapshot(output.logits[0, -width:], cache, length, prefill_logits)
    if (snapshot.logits.shape != (width, vocab)
            or any(tensor.ndim != 4 or tensor.shape[-2] != length for pair in snapshot.cache for tensor in pair)):
        raise ValueError("reference returned invalid logit or cache shapes")
    return snapshot


def _equal(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False, False
    actual = actual.to(expected.device)
    bitwise = torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
    close = (bool(torch.isfinite(actual).all()) and bool(torch.isfinite(expected).all())
             and torch.allclose(actual, expected, atol=atol, rtol=rtol))
    return close, bitwise


def _compare_cache(pairs, length, expected, *, atol, rtol):
    failures = []
    if length != expected.length:
        failures.append(CheckFailure("cache_length", detail="cache lengths differ"))
    cache_bitwise = len(pairs) == len(expected.cache)
    if not cache_bitwise:
        failures.append(CheckFailure("cache_layers", detail="cache layer counts differ"))
    for index, (a, e) in enumerate(zip(pairs, expected.cache)):
        for name, got, wanted in zip(("keys", "values"), a, e):
            close, bitwise = _equal(got, wanted, atol=atol, rtol=rtol)
            cache_bitwise = cache_bitwise and bitwise
            if not close:
                failures.append(CheckFailure("cache_values", index, f"{name} shape, dtype or values differ"))
    return failures, cache_bitwise


def _compare_logits(actual, expected, *, atol, rtol, prefix=""):
    failures = []
    close, bitwise = _equal(actual, expected, atol=atol, rtol=rtol)
    if actual.shape != expected.shape:
        failures.append(CheckFailure(prefix + "logits_shape", detail="logit shapes differ"))
    else:
        if actual.argmax(-1).tolist() != expected.argmax(-1).tolist():
            failures.append(CheckFailure(prefix + "tokens", detail="next-token IDs differ"))
        if not close:
            failures.append(CheckFailure(prefix + "logits", detail="logit dtype, finiteness or values differ"))
    return failures, bitwise


def _compare(actual, expected, *, atol, rtol):
    failures, logits_bitwise = _compare_logits(actual.logits, expected.logits, atol=atol, rtol=rtol)
    prompt, _ = _compare_logits(actual.prefill_logits, expected.prefill_logits, atol=atol, rtol=rtol, prefix="prefill_")
    cache_failures, cache_bitwise = _compare_cache(actual.cache, actual.length, expected, atol=atol, rtol=rtol)
    return ForwardComparison((*prompt, *failures, *cache_failures), logits_bitwise, cache_bitwise)


def _compare_predictions(ids, prefill_logits, pairs, length, expected, *, atol, rtol):
    prompt, _ = _compare_logits(prefill_logits, expected.prefill_logits, atol=atol, rtol=rtol, prefix="prefill_")
    failures, cache_bitwise = _compare_cache(pairs, length, expected, atol=atol, rtol=rtol)
    wanted = tuple(expected.logits.argmax(-1).tolist())
    if ids != wanted:
        first = next((i for i, (a, b) in enumerate(zip(ids, wanted)) if a != b), min(len(ids), len(wanted)))
        failures.append(CheckFailure("tokens", first, "compact next-token IDs differ"))
    return PredictionComparison((*prompt, *failures), cache_bitwise)


def _candidate_prefill(target, prefix):
    state, logits = target.prefill(list(prefix))
    return state, logits.detach().clone()  # The next forward can reuse the score buffer.


def _validate(target, reference, prefix, tokens):
    for values in (prefix, tokens):
        if (not isinstance(values, Sequence) or isinstance(values, (str, bytes)) or not values
                or any(type(token) is not int or token < 0 for token in values)):
            raise InvalidCheckCase("provide nonempty sequences of nonnegative integer token IDs")
    if len(tokens) > 16:
        raise InvalidCheckCase("forward checks support at most 16 block tokens")
    if target.model is reference or target.model.training or reference.training:
        raise ValueError("use separate eval-mode target and reference models")
    _check_reference_storage(target.model, reference)
    context = min(target.model.config.max_position_embeddings, reference.config.max_position_embeddings)
    if len(prefix) + len(tokens) > context:
        raise InvalidCheckCase("prefix and block exceed the model context limit")
    vocab = min(target.model.config.vocab_size, reference.config.vocab_size)
    if max((*prefix, *tokens)) >= vocab:
        raise InvalidCheckCase("token ID is outside the model vocabulary")


@torch.inference_mode()
def check_forward_paths(target, reference_model, prefix: Sequence[int], tokens: Sequence[int], *,
                        atol: float = 2e-6, rtol: float = 2e-5) -> ForwardCheck:
    """Check one block of 1..16 tokens with three independent cache paths.

    Supply idle, separate eval-mode models with matching weights and settings.
    The reference computes full-prefix and native cached forwards; the target
    uses its own prefill/forward methods. Each pair checks the prompt's next-token
    logits and winning ID as well as the block. The logits_bitwise flag describes
    the block rows. Every comparison must pass.
    Targets with compact predictions enabled also check their IDs and KV from
    a fresh prefill against the same two references. Logit and cache dtypes
    are preserved and must match within each pair.
    Reference errors propagate; candidate errors remain failed comparisons.
    Results retain no tensors and write no files. This is a local forward check,
    not a generation, EOS, tree, performance or all-input correctness check.
    """
    _validate_tolerances(atol, rtol)
    _validate(target, reference_model, prefix, tokens)
    prefix, tokens = tuple(prefix), tuple(tokens)
    length, width = len(prefix) + len(tokens), len(tokens)
    output = _reference_forward(reference_model, (*prefix, *tokens))
    full = _reference_snapshot(output, width, length, reference_model.config.vocab_size)
    del output
    prefill = _reference_forward(reference_model, prefix)
    if prefill.logits.shape != (1, len(prefix), reference_model.config.vocab_size):
        raise ValueError("reference returned an invalid prefill logit shape")
    prefill_logits = prefill.logits[0, -1].detach().clone()
    cache = prefill.past_key_values
    del prefill
    ids = torch.tensor([tokens], dtype=torch.long, device=reference_model.device)
    mask = torch.ones((1, length), dtype=torch.long, device=reference_model.device)
    output = reference_model(input_ids=ids, attention_mask=mask, past_key_values=cache, use_cache=True)
    incremental = _reference_snapshot(output, width, length, reference_model.config.vocab_size,
                                      prefill_logits=prefill_logits)
    del cache, output
    reference_check = _compare(incremental, full, atol=atol, rtol=rtol)
    try:
        state, prefill_logits = _candidate_prefill(target, prefix)
        logits = target.forward_block(state, list(tokens))
        candidate = _snapshot(logits, state.cache, state.seq_len, prefill_logits)
        against_full = _compare(candidate, full, atol=atol, rtol=rtol)
        against_incremental = _compare(candidate, incremental, atol=atol, rtol=rtol)
    except Exception as error:
        failure = ForwardComparison((CheckFailure("exception", detail=f"{type(error).__name__}: {error}"),))
        against_full = against_incremental = failure
    predictions_full = predictions_incremental = None
    if getattr(target, "capture_predictions", False):
        try:
            state, prefill_logits = _candidate_prefill(target, prefix)
            ids = _tokens(target.forward_predictions(state, list(tokens)))
            pairs = _cache_pairs(state.cache, read_only=True)
            predictions_full = _compare_predictions(ids, prefill_logits, pairs, state.seq_len, full, atol=atol, rtol=rtol)
            predictions_incremental = _compare_predictions(ids, prefill_logits, pairs, state.seq_len, incremental,
                                                          atol=atol, rtol=rtol)
        except Exception as error:
            failure = PredictionComparison((CheckFailure("exception", detail=f"{type(error).__name__}: {error}"),))
            predictions_full = predictions_incremental = failure
    return ForwardCheck(against_full, against_incremental, reference_check, predictions_full, predictions_incremental)
