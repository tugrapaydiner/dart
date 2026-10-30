"""Compare complete greedy outputs from independent decoding paths."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass


class InvalidCheckCase(ValueError):
    """Input cannot be evaluated as a valid comparison case."""


@dataclass(frozen=True)
class GenerationCase:
    name: str
    prompt: tuple[int, ...]
    max_new_tokens: int
    draft_len: int = 4

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt", tuple(self.prompt))
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("case name must be nonempty")
        if not self.prompt or any(type(token) is not int or token < 0 for token in self.prompt):
            raise ValueError("prompt must contain nonnegative integer token IDs")
        for name in ("max_new_tokens", "draft_len"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")


Decoder = Callable[[GenerationCase], Sequence[int]]


@dataclass(frozen=True)
class CheckFailure:
    kind: str
    index: int | None = None
    detail: str = ""


@dataclass(frozen=True)
class CheckResult:
    case: GenerationCase
    backend: str
    expected: tuple[int, ...]
    actual: tuple[int, ...] | None
    failure: CheckFailure | None

    @property
    def passed(self) -> bool:
        return self.failure is None


@dataclass(frozen=True)
class CheckReport:
    results: tuple[CheckResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.results) and all(result.passed for result in self.results)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(result for result in self.results if not result.passed)


def _tokens(output: Sequence[int]) -> tuple[int, ...]:
    if (not isinstance(output, Sequence) or isinstance(output, (str, bytes))
            or any(type(token) is not int or token < 0 for token in output)):
        raise ValueError("decoder must return a sequence of nonnegative integer token IDs")
    return tuple(output)


def _eos_index(tokens: tuple[int, ...], eos: frozenset[int]) -> int | None:
    return next((index for index, token in enumerate(tokens) if token in eos), None)


def _compare(case: GenerationCase, expected: tuple[int, ...], actual: tuple[int, ...],
             eos: frozenset[int]) -> CheckFailure | None:
    if len(actual) > case.max_new_tokens:
        return CheckFailure("budget", case.max_new_tokens, "output exceeds the token budget")
    expected_eos, actual_eos = _eos_index(expected, eos), _eos_index(actual, eos)
    if actual_eos is not None and actual_eos != len(actual) - 1:
        return CheckFailure("eos", actual_eos, "output continues after EOS")
    if expected_eos != actual_eos:
        index = min(index for index in (expected_eos, actual_eos) if index is not None)
        return CheckFailure("eos", index, "EOS positions differ")
    for index, (wanted, got) in enumerate(zip(expected, actual)):
        if wanted != got:
            return CheckFailure("tokens", index, "token IDs differ")
    if len(actual) != len(expected):
        return CheckFailure("length", min(len(expected), len(actual)), "output lengths differ")
    return None


def run_differential(
    cases: Sequence[GenerationCase],
    reference: Decoder,
    candidates: Mapping[str, Decoder],
    *,
    eos_token_ids: Sequence[int] = (),
) -> CheckReport:
    """Run each case once per backend, retaining failures without writing files.

    Callers supply an independent reference with matching weights and decoding
    rules. Reference errors abort the check; candidate errors are failed cases.
    Empty suites are rejected. This checks observed outputs, not all inputs.
    """
    cases, candidates = tuple(cases), dict(candidates)
    if not cases or len({case.name for case in cases}) != len(cases):
        raise ValueError("provide nonempty cases with unique names")
    if not candidates or any(not isinstance(name, str) or not name for name in candidates):
        raise ValueError("provide at least one named candidate")
    eos = frozenset(_tokens(eos_token_ids))
    results = []
    for case in cases:
        expected = _tokens(reference(case))
        stop = _eos_index(expected, eos)
        if (len(expected) > case.max_new_tokens
                or (stop is not None and stop != len(expected) - 1)
                or (len(expected) < case.max_new_tokens and stop is None)):
            raise ValueError(f"invalid reference output for case {case.name!r}")
        for name, decode in candidates.items():
            actual = None
            try:
                output = decode(case)
            except Exception as error:
                failure = CheckFailure("exception", detail=f"{type(error).__name__}: {error}")
            else:
                try:
                    actual = _tokens(output)
                except ValueError as error:
                    failure = CheckFailure("output", detail=str(error))
                else:
                    failure = _compare(case, expected, actual, eos)
            results.append(CheckResult(case, name, expected, actual, failure))
    return CheckReport(tuple(results))


def engine_decoder(engine) -> Decoder:
    """Adapt an existing engine without resetting it between cases."""
    from dart.runtime.types import DecodeConfig

    def decode(case: GenerationCase) -> list[int]:
        return engine.generate(
            list(case.prompt),
            DecodeConfig(max_new_tokens=case.max_new_tokens, draft_len=case.draft_len),
        ).token_ids

    return decode


def hf_decoder(model) -> Decoder:
    """Use native HF greedy generation on an idle, eval-mode reference model.

    The model must not be used concurrently. Its generation defaults are
    replaced during each call and restored even if generation fails.
    """
    import torch
    from transformers import GenerationConfig

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = model.config.eos_token_id

    @torch.inference_mode()
    def decode(case: GenerationCase) -> list[int]:
        limit = model.config.max_position_embeddings
        if len(case.prompt) + case.max_new_tokens > limit:
            raise InvalidCheckCase("case exceeds the model context limit")
        if case.max_new_tokens == 0:
            return []
        ids = torch.tensor([case.prompt], device=model.device, dtype=torch.long)
        config = GenerationConfig(
            do_sample=False, num_beams=1, use_cache=True,
            max_new_tokens=case.max_new_tokens, eos_token_id=eos,
            pad_token_id=model.config.pad_token_id, bos_token_id=model.config.bos_token_id,
        )
        previous = model.generation_config
        model.generation_config = config  # HF fills unspecified options from model defaults.
        try:
            output = model.generate(ids, attention_mask=torch.ones_like(ids), generation_config=config)
        finally:
            model.generation_config = previous
        return output[0, len(case.prompt):].tolist()

    return decode
