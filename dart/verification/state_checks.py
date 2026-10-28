"""Check a verifier round and its surviving cache against native HF."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from dart.verification.block_verify import verify_block_greedy
from dart.verification.differential import CheckFailure, GenerationCase, hf_decoder
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
        GenerationCase(self.name, self.prefix, 0)
        if any(type(token) is not int or token < 0 for token in (self.pending, *self.drafts)):
            raise ValueError("pending and drafts must be nonnegative integer token IDs")
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

    @property
    def passed(self) -> bool:
        return not self.failures


def _expected_path(case, continuation, eos):
    committed, accepted, parent = [], 0, -1
    for token in continuation:
        committed.append(token)
        matches = [index for index, (draft, ancestor) in enumerate(zip(case.drafts, case.parents))
                   if ancestor == parent and draft == token]
        if not matches:
            break
        accepted += 1
        parent = matches[0]
        if token in eos:
            break
    return tuple(committed), accepted


@torch.inference_mode()
def _reference_forward(model, tokens):
    ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
    return model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=True)


def _cache_pairs(cache):
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    if isinstance(cache, (tuple, list)):
        return [(layer[0], layer[1]) for layer in cache]
    raise TypeError("state checks require an HF-compatible cache")


def _cache_failures(state, expected, length, *, atol, rtol):
    if state.seq_len != length:
        return [CheckFailure("cache_length", detail=f"expected {length}, got {state.seq_len}")]
    actual_pairs, expected_pairs = _cache_pairs(state.cache), _cache_pairs(expected)
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


@torch.inference_mode()
def check_verifier_state(
    target,
    reference_model,
    case: RoundCase,
    *,
    verifier: str = "tree",
    atol: float = 2e-6,
    rtol: float = 2e-5,
) -> RoundCheck:
    """Check one real round, cache values, and one subsequent target step.

    Supply separate eval-mode models with matching weights and EOS settings.
    This currently supports HF-compatible caches. The reference uses native
    generation and fresh full-prefix forwards, never target cache operations.
    Reference errors propagate; candidate exceptions are reported as failures.
    """
    if target.model is reference_model:
        raise ValueError("use a separate reference model")
    if target.model.training or reference_model.training:
        raise ValueError("both models must be in eval mode")
    if verifier not in {"block", "tree"}:
        raise ValueError("verifier must be block or tree")
    if verifier == "block" and case.parents != tuple(range(-1, len(case.drafts) - 1)):
        raise ValueError("block verification requires a chain")
    for value in (atol, rtol):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError("tolerances must be finite and nonnegative")
    tokens = (*case.prefix, case.pending, *case.drafts)
    vocab = min(target.model.config.vocab_size, reference_model.config.vocab_size)
    if max(tokens) >= vocab:
        raise ValueError("case contains a token outside the model vocabulary")
    depths = []
    for parent in case.parents:
        depths.append(0 if parent == -1 else depths[parent] + 1)
    lookahead = max(depths, default=-1) + 2
    context = min(target.model.config.max_position_embeddings, reference_model.config.max_position_embeddings)
    if len(case.prefix) + 1 + lookahead > context:
        raise ValueError("case and reference lookahead exceed the context limit")
    eos = reference_model.generation_config.eos_token_id
    if eos is None:
        eos = reference_model.config.eos_token_id
    eos = frozenset(() if eos is None else [eos] if isinstance(eos, int) else eos)
    if target.eos_token_ids != eos:
        raise ValueError("reference and target EOS settings differ")
    continuation = hf_decoder(reference_model)(
        GenerationCase(case.name, (*case.prefix, case.pending), lookahead)
    )
    expected, accepted = _expected_path(case, continuation, eos)
    cached_tokens = (*case.prefix, case.pending, *expected[:-1])
    reference_cache = _reference_forward(reference_model, cached_tokens).past_key_values
    reference_next = _reference_forward(reference_model, (*cached_tokens, expected[-1]))
    failures, actual, length = [], None, None
    try:
        state, _ = target.prefill(list(case.prefix))
        if verifier == "tree":
            result = verify_tree_greedy(target, state, case.pending, DraftTree(list(case.drafts), list(case.parents)))
        else:
            result = verify_block_greedy(target, state, case.pending, list(case.drafts))
        actual, length = tuple(result.committed), int(state.seq_len)
        if actual != expected or any(type(token) is not int for token in actual):
            failures.append(CheckFailure("committed", detail="committed token sequences differ"))
        if (result.n_draft_accepted != accepted or result.n_draft_proposed != len(case.drafts)
                or result.all_accepted != (accepted == len(case.drafts)) or result.pending_divergence != 0):
            failures.append(CheckFailure("acceptance", detail="verification counts differ"))
        if result.finished != (expected[-1] in eos):
            failures.append(CheckFailure("eos", detail="finished flag differs"))
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
    return RoundCheck(case, verifier, expected, actual, len(cached_tokens), length, tuple(failures))
