"""Check verifier rounds, packed tree rows and surviving caches against native HF."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from itertools import chain

import torch

from dart.models.replay import ReplayCache
from dart.verification.block_verify import verify_block_greedy
from dart.verification.differential import (
    CheckFailure,
    GenerationCase,
    InvalidCheckCase,
    hf_decoder,
)
from dart.verification.tree_verify import DraftTree, verify_tree_greedy


@dataclass(frozen=True)
class RoundCase:
    name: str
    prefix: tuple[int, ...]
    pending: int
    drafts: tuple[int, ...]
    parents: tuple[int, ...]

    def __post_init__(self) -> None:
        for name in ("prefix", "drafts", "parents"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("case name must be nonempty")
        if any(type(token) is not int or token < 0 for token in (*self.prefix, self.pending, *self.drafts)):
            raise ValueError("prefix, pending and drafts must be nonnegative integer token IDs")
        if len(self.parents) != len(self.drafts):
            raise ValueError("each draft needs a parent")
        if any(type(parent) is not int or parent < -1 or parent >= index
               for index, parent in enumerate(self.parents)):
            raise ValueError("parents must be -1 or a preceding node index")


@dataclass(frozen=True)
class RoundCheck:
    case: RoundCase
    verifier: str
    expected: tuple[int, ...]
    actual: tuple[int, ...] | None
    expected_cache_length: int
    cache_length: int | None
    failures: tuple[CheckFailure, ...]
    tree_rows_checked: int = 0

    @property
    def passed(self) -> bool:
        return not self.failures


def _expected_path(drafts, parents, continuation, eos):
    committed, accepted, frontier = [], 0, {-1}
    for token in continuation:
        committed.append(token)
        frontier = {index for index, (draft, parent) in enumerate(zip(drafts, parents))
                    if parent in frontier and draft == token}
        if not frontier:
            break
        accepted += 1
        if token in eos:
            break
    return tuple(committed), accepted


@torch.inference_mode()
def _reference_forward(model, tokens, **kwargs):
    ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True, **kwargs)


def _round_cache(model, tokens, record_hidden):
    if not record_hidden:
        return _reference_forward(model, tokens).past_key_values, None
    output = _reference_forward(model, tokens, output_hidden_states=True)
    hidden = output.hidden_states[-1]
    if hidden.ndim != 3 or hidden.shape[:2] != (1, len(tokens)):
        raise ValueError("reference returned an invalid hidden state shape")
    return output.past_key_values, hidden[0, -1].clone()  # Keep only the retained prefix's final row.


def _hidden_failures(target, result, expected, *, atol, rtol):
    if expected is None:
        return []
    block, index = target.last_hidden_block, result.hidden_index
    if not isinstance(block, torch.Tensor) or block.ndim != 2:
        return [CheckFailure("hidden_state", detail="expected a matrix of hidden rows")]
    if type(index) is not int or not 0 <= index < len(block):
        return [CheckFailure("hidden_index", detail="index outside the recorded hidden rows")]
    row = block[index].to(expected.device)
    if (row.shape != expected.shape or row.dtype != expected.dtype or not torch.isfinite(row).all()
            or not torch.allclose(row, expected, atol=atol, rtol=rtol)):
        return [CheckFailure("hidden_state", index, "selected row differs from the retained prefix")]
    return []


def _check_reference_storage(model, reference):
    devices = {}
    for owner, module in enumerate((model, reference)):
        for tensor in chain(module.parameters(), module.buffers()):
            if tensor.numel():
                storage = tensor.untyped_storage()
                start = storage.data_ptr()
                devices.setdefault(tensor.device, []).append((start, start + storage.nbytes(), owner))

    for ranges in devices.values():
        ends = [0, 0]
        for start, end, owner in sorted(ranges):
            if start < ends[1 - owner]:
                raise ValueError("target and reference must not share tensor storage")
            ends[owner] = max(ends[owner], end)  # Views within one model may overlap.


def _cache_pairs(cache, *, read_only=False):
    if isinstance(cache, ReplayCache):
        if read_only and type(cache) is ReplayCache:
            storage = cache._buffer.tensor  # These views stay inside the comparison.
        else:
            storage = cache.storage
        if (not isinstance(storage, torch.Tensor) or storage.ndim != 6
                or storage.shape[1:3] != (2, 1) or any(size <= 0 for size in storage.shape)):
            raise ValueError("replay cache has invalid storage shape")
        length = cache.get_seq_length()
        if type(length) is not int or not 0 <= length <= storage.shape[-2]:
            raise ValueError("replay cache has invalid prefix length")
        return [(layer[0, ..., :length, :], layer[1, ..., :length, :]) for layer in storage]
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        try:
            return list(zip(cache.key_cache, cache.value_cache, strict=True))
        except ValueError as error:
            raise ValueError("invalid legacy cache key/value layers") from error
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]
    raise TypeError("state checks require an HF-compatible cache or ReplayCache")


def _cache_failures(state, expected, length, *, atol, rtol):
    if state.seq_len != length:
        return [CheckFailure("cache_length", detail=f"expected {length}, got {state.seq_len}")]
    actual_pairs = _cache_pairs(state.cache, read_only=True)
    expected_pairs = _cache_pairs(expected, read_only=True)
    if len(actual_pairs) != len(expected_pairs):
        return [CheckFailure("cache_layers", detail="cache layer counts differ")]
    failures = []
    for index, (actual, wanted) in enumerate(zip(actual_pairs, expected_pairs)):
        for label, got, want in zip(("keys", "values"), actual, wanted):
            if got.shape != want.shape or got.dtype != want.dtype:
                failures.append(CheckFailure("cache_values", index, f"{label} shape or dtype differs"))
            elif not torch.allclose(got.to(want.device), want, atol=atol, rtol=rtol):
                failures.append(CheckFailure("cache_values", index, f"{label} differ"))
    return failures


def _validate_tolerances(atol, rtol):
    for value in (atol, rtol):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("tolerances must be finite and nonnegative")


def _inspect_tree_rows(target, reference_model, case, *, atol, rtol):
    paths = [(case.pending,)]
    for token, parent in zip(case.drafts, case.parents):
        paths.append((*paths[parent + 1], token))
    packed_length = len(case.prefix) + len(paths)
    failures, checked = [], 0
    try:
        state, _ = target.prefill(list(case.prefix))
        if case.drafts:
            logits = target.forward_tree(state, [case.pending], list(case.drafts), list(case.parents))
        else:
            logits = target.forward_block(state, [case.pending])
        if logits.shape != (len(paths), reference_model.config.vocab_size):
            return [CheckFailure("tree_logits_shape", detail="packed logits shape differs")], 0
        if state.seq_len != packed_length:
            return [CheckFailure("tree_cache_length", detail=f"expected {packed_length}, got {state.seq_len}")], 0
        pairs = _cache_pairs(state.cache, read_only=True)
    except Exception as error:
        return [CheckFailure("exception", detail=f"{type(error).__name__}: {error}")], 0

    for row, path in enumerate(paths):
        reference = _reference_forward(reference_model, (*case.prefix, *path))
        expected_pairs = _cache_pairs(reference.past_key_values)
        wanted = reference.logits[0, -1].float()
        try:
            actual = logits[row].to(wanted.device)
            if actual.argmax().item() != wanted.argmax().item():
                failures.append(CheckFailure("tree_token", row, "packed row argmax differs"))
            if not torch.allclose(actual, wanted, atol=atol, rtol=rtol):
                failures.append(CheckFailure("tree_logits", row, "packed row logits differ beyond tolerance"))
            if len(pairs) != len(expected_pairs):
                failures.append(CheckFailure("tree_cache_layers", row, "cache layer counts differ"))
            for layer, (actual_pair, expected_pair) in enumerate(zip(pairs, expected_pairs)):
                for name, got, want in zip(("keys", "values"), actual_pair, expected_pair):
                    if got.shape[-2] != packed_length:
                        failures.append(CheckFailure("tree_cache_values", row, f"layer {layer} {name} packed length differs"))
                        continue
                    if row == 0:
                        got = got[..., :len(case.prefix) + 1, :]  # Check the shared prefix once.
                    else:
                        position = len(case.prefix) + row
                        got, want = got[..., position:position + 1, :], want[..., -1:, :]
                    if got.shape != want.shape or got.dtype != want.dtype:
                        failures.append(CheckFailure("tree_cache_values", row, f"layer {layer} {name} shape or dtype differs"))
                    elif not torch.allclose(got.to(want.device), want, atol=atol, rtol=rtol):
                        failures.append(CheckFailure("tree_cache_values", row, f"layer {layer} {name} differ"))
            checked += 1
        except Exception as error:
            failures.append(CheckFailure("exception", row, f"{type(error).__name__}: {error}"))
            break
    return failures, checked


def _round_parameters(target, reference_model, case, verifier):
    if target.model is reference_model:
        raise ValueError("use a separate reference model")
    if target.model.training or reference_model.training:
        raise ValueError("both models must be in eval mode")
    if verifier not in {"block", "tree"}:
        raise ValueError("verifier must be block or tree")
    if verifier == "block" and case.parents != tuple(range(-1, len(case.drafts) - 1)):
        raise InvalidCheckCase("block verification requires a chain")
    tokens = (*case.prefix, case.pending, *case.drafts)
    vocab = min(target.model.config.vocab_size, reference_model.config.vocab_size)
    if max(tokens) >= vocab:
        raise InvalidCheckCase("case contains a token outside the model vocabulary")
    depths = []
    for parent in case.parents:
        depths.append(0 if parent == -1 else depths[parent] + 1)
    lookahead = max(depths, default=-1) + 2
    context = min(target.model.config.max_position_embeddings, reference_model.config.max_position_embeddings)
    if len(case.prefix) + 1 + lookahead > context:
        raise InvalidCheckCase("case and reference lookahead exceed the context limit")
    eos = reference_model.generation_config.eos_token_id
    if eos is None:
        eos = reference_model.config.eos_token_id
    eos = frozenset(() if eos is None else [eos] if isinstance(eos, int) else eos)
    if target.eos_token_ids != eos:
        raise ValueError("reference and target EOS settings differ")
    _check_reference_storage(target.model, reference_model)
    return lookahead, eos


def _round_expectation(target, reference_model, case, verifier):
    lookahead, eos = _round_parameters(target, reference_model, case, verifier)
    continuation = hf_decoder(reference_model)(
        GenerationCase(case.name, (*case.prefix, case.pending), lookahead)
    )
    return _expected_path(case.drafts, case.parents, continuation, eos)


def _result_failures(result, expected, accepted, draft_count, eos):
    failures = []
    actual = tuple(result.committed)
    if actual != expected or any(type(token) is not int for token in actual):
        failures.append(CheckFailure("committed", detail="committed token sequences differ"))
    if (result.n_draft_accepted != accepted or result.n_draft_proposed != draft_count
            or result.all_accepted != (accepted == draft_count) or result.pending_divergence != 0):
        failures.append(CheckFailure("acceptance", detail="verification counts differ"))
    if result.finished != (expected[-1] in eos):
        failures.append(CheckFailure("eos", detail="finished flag differs"))
    return failures


def _validate_state_check(case, verifier, atol, rtol, inspect_tree):
    if not case.prefix:
        raise InvalidCheckCase("standalone round checks need a nonempty prefill prefix")
    _validate_tolerances(atol, rtol)
    if type(inspect_tree) is not bool or (inspect_tree and verifier != "tree"):
        raise ValueError("inspect_tree must be boolean and requires the tree verifier")
    if inspect_tree and len(case.drafts) > 32:
        raise InvalidCheckCase("inspect_tree supports at most 32 draft nodes")


@torch.inference_mode()
def check_verifier_state(
    target,
    reference_model,
    case: RoundCase,
    *,
    verifier: str = "tree",
    atol: float = 2e-6,
    rtol: float = 2e-5,
    inspect_tree: bool = False,
) -> RoundCheck:
    """Check one real round, cache values, and one subsequent target step.

    Use eval-mode models with independent parameter and buffer storage, matching weights and EOS settings.
    This supports HF-compatible caches and ReplayCache. The reference uses native
    generation and fresh full-prefix forwards, never target cache operations.
    Targets that record hidden states also check the row selected for drafter feedback.
    Reference errors propagate; candidate exceptions are reported as failures.
    inspect_tree checks every packed row in a separate pass, including rejected
    branches, with at most 32 draft nodes. Each reference branch is run separately.
    """
    _validate_state_check(case, verifier, atol, rtol, inspect_tree)
    expected, accepted = _round_expectation(target, reference_model, case, verifier)
    cached_tokens = (*case.prefix, case.pending, *expected[:-1])
    reference_cache, hidden = _round_cache(reference_model, cached_tokens, getattr(target, "record_hidden", False))
    reference_next = _reference_forward(reference_model, (*cached_tokens, expected[-1]))
    return _check_round_state(target, reference_model, case, verifier, expected, accepted,
                              reference_cache, hidden, reference_next, atol, rtol, inspect_tree)


@torch.inference_mode()
def compare_verifier_states(
    candidates: Mapping[str, object],
    reference_model,
    case: RoundCase,
    *,
    verifier: str = "tree",
    atol: float = 2e-6,
    rtol: float = 2e-5,
    inspect_tree: bool = False,
) -> dict[str, RoundCheck]:
    """Check named implementations using one set of native round references.

    Candidates run in mapping order with separate request states and matching weights and EOS settings.
    They must keep the reference unmodified and use independent parameter and buffer storage.
    All candidates are validated before inference. Native generation and the two
    full-prefix cache forwards run once per call; nothing is cached across calls.
    Optional packed-row inspections still run independently for each candidate.
    Reference errors abort; candidate exceptions remain individual failures.
    """
    if not isinstance(candidates, Mapping):
        raise ValueError("provide a mapping of named candidates")
    candidates = dict(candidates)
    if not candidates or any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("provide at least one named candidate")
    _validate_state_check(case, verifier, atol, rtol, inspect_tree)
    for target in candidates.values():
        _round_parameters(target, reference_model, case, verifier)
    first = next(iter(candidates.values()))
    expected, accepted = _round_expectation(first, reference_model, case, verifier)
    cached_tokens = (*case.prefix, case.pending, *expected[:-1])
    recording = {name for name, target in candidates.items() if getattr(target, "record_hidden", False)}
    reference_cache, hidden = _round_cache(reference_model, cached_tokens, bool(recording))
    reference_next = _reference_forward(reference_model, (*cached_tokens, expected[-1]))
    results = {}
    for name, target in candidates.items():
        _round_parameters(target, reference_model, case, verifier)  # Recheck before each candidate runs.
        results[name] = _check_round_state(target, reference_model, case, verifier, expected, accepted,
                                           reference_cache, hidden if name in recording else None,
                                           reference_next, atol, rtol, inspect_tree)
    return results


def _check_round_state(target, reference_model, case, verifier, expected, accepted,
                       reference_cache, hidden, reference_next, atol, rtol, inspect_tree):
    cached_tokens = (*case.prefix, case.pending, *expected[:-1])
    failures, checked = [], 0
    if inspect_tree:
        failures, checked = _inspect_tree_rows(target, reference_model, case, atol=atol, rtol=rtol)
    actual, length = None, None
    try:
        state, _ = target.prefill(list(case.prefix))
        if verifier == "tree":
            result = verify_tree_greedy(target, state, case.pending, DraftTree(list(case.drafts), list(case.parents)))
        else:
            result = verify_block_greedy(target, state, case.pending, list(case.drafts))
        actual, length = tuple(result.committed), int(state.seq_len)
        failures.extend(_result_failures(result, expected, accepted, len(case.drafts), target.eos_token_ids))
        failures.extend(_hidden_failures(target, result, hidden, atol=atol, rtol=rtol))
        failures.extend(_cache_failures(state, reference_cache, len(cached_tokens), atol=atol, rtol=rtol))
        if actual:
            logits = target.forward_block(state, [actual[-1]])[-1]
            wanted = reference_next.logits[0, -1].float()
            if logits.argmax().item() != wanted.argmax().item():
                failures.append(CheckFailure("followup_token", detail="next token differs after cache reuse"))
            if logits.shape != wanted.shape or not torch.allclose(logits.to(wanted.device), wanted, atol=atol, rtol=rtol):
                failures.append(CheckFailure("followup_logits", detail="next-step logits differ beyond tolerance"))
            failures.extend(_cache_failures(state, reference_next.past_key_values, len(cached_tokens) + 1,
                                            atol=atol, rtol=rtol))
    except Exception as error:
        failures.append(CheckFailure("exception", detail=f"{type(error).__name__}: {error}"))
    return RoundCheck(case, verifier, expected, actual, len(cached_tokens), length, tuple(failures), checked)
