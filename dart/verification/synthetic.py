"""Small CPU checks using random weights, with no checkpoint or download."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import replace

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.verification.differential import (
    CheckReport,
    Decoder,
    GenerationCase,
    InvalidCheckCase,
    engine_decoder,
    hf_decoder,
    run_differential,
)
from dart.verification.state_sequences import (
    StateOperation,
    StateSequenceCheck,
    _validate_operations,
    check_state_sequence,
)
from dart.verification.tree_verify import DraftTree

VOCAB_SIZE = 67
CONTEXT_LENGTH = 64


def synthetic_model_config() -> dict:
    """Return the fixed tiny model configuration used by the synthetic check."""
    return {
        "vocab_size": VOCAB_SIZE, "hidden_size": 32, "intermediate_size": 64,
        "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
        "max_position_embeddings": CONTEXT_LENGTH, "bos_token_id": 1,
        "eos_token_id": 2, "pad_token_id": 0, "attn_implementation": "eager",
    }


def synthetic_execution_config(execution: str = "eager") -> dict:
    """Return the fixed CPU execution settings used by synthetic checks."""
    if not isinstance(execution, str) or execution not in {"eager", "static"}:
        raise ValueError("execution must be eager or static")
    if execution == "static":
        return {"backend": "static", "capacity": CONTEXT_LENGTH,
                "max_block_tokens": 5, "capture_predictions": True}
    return {"backend": "eager"}


def _execution_target(target, execution, cleanup):
    if execution == "static":
        from dart.models.replay import CudaGraphTarget

        options = synthetic_execution_config(execution)
        options.pop("backend")
        target = CudaGraphTarget(target, replay=False, **options)
        cleanup.callback(target.close)
    return target


class _ProbeDrafter:
    name = "synthetic"
    requires_entropy = False

    def propose(self, committed, max_tokens):
        return [(committed[-1] + index + 1) % VOCAB_SIZE for index in range(max_tokens)]

    def propose_tree(self, committed, max_tokens):
        first = self.propose(committed, max_tokens)
        second = [(token + 1) % VOCAB_SIZE for token in first]
        return DraftTree.from_branches([first, second])

    def observe(self, result):
        pass


def _generator(seed):
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    return torch.Generator(device="cpu").manual_seed(seed)


def synthetic_cases(seed: int = 17) -> tuple[GenerationCase, ...]:
    """Six bounded cases, including zero output and a full context boundary."""
    generator = _generator(seed)
    cases = []
    for index, (length, budget) in enumerate(((1, 0), (4, 1), (7, 2), (8, 5), (51, 13), (8, 8))):
        prompt = torch.randint(3, VOCAB_SIZE, (length,), generator=generator).tolist()
        cases.append(GenerationCase(f"case-{index}", tuple(prompt), budget))
    return tuple(cases)


def synthetic_state_operations(seed: int = 17, *, execution: str = "eager") -> tuple[StateOperation, ...]:
    """Bounded operations covering interleaving, rollback and context errors."""
    synthetic_execution_config(execution)
    generator = _generator(seed)

    def tokens(count):
        return tuple(torch.randint(3, VOCAB_SIZE, (count,), generator=generator).tolist())

    operations = (
        StateOperation("a", "prefill", tokens(4)),
        StateOperation("b", "prefill", tokens(7)),
        StateOperation("a", "forward", tokens(3)),
        StateOperation("b", "block", tokens(4)),
        StateOperation("a", "tree", tokens(5), (-1, -1, 0, 1)),
        StateOperation("b", "forward", tokens(2)),
        StateOperation("a", "crop", length=0),
        StateOperation("b", "forward", tokens(1)),
        StateOperation("a", "block", tokens(3)),
        StateOperation("a", "release"),
        StateOperation("a", "prefill", tokens(CONTEXT_LENGTH + 1), expect_error="ValueError"),
        StateOperation("a", "prefill", tokens(CONTEXT_LENGTH - 1)),
        StateOperation("a", "forward", tokens(1)),
        StateOperation("a", "forward", tokens(1), expect_error="ValueError"),
        StateOperation("a", "crop", length=CONTEXT_LENGTH - 1),
        StateOperation("a", "forward", tokens(1)),
        StateOperation("a", "release"),
        StateOperation("a", "prefill", tokens(3)),
        StateOperation("b", "crop", length=-1, expect_error="ValueError"),
        StateOperation("b", "crop", length=CONTEXT_LENGTH + 1, expect_error="ValueError"),
        StateOperation("b", "forward", tokens(CONTEXT_LENGTH), expect_error="ValueError"),
        StateOperation("b", "forward", tokens(2)),
        StateOperation("a", "tree", tokens(5), (-1, 0, -1, 2)),
        StateOperation("b", "release"),
        StateOperation("a", "forward", tokens(1)),
        StateOperation("a", "release"),
    )
    if execution == "eager":
        return operations
    steps = []
    for operation in operations:
        if operation.action == "tree":
            steps.append(replace(operation, expect_error="ValueError"))  # Static execution rejects trees.
            steps.append(StateOperation(operation.request, "block", operation.tokens))
        else:
            steps.append(operation)
    return tuple(steps)


def _synthetic_models(seed):
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        config = LlamaConfig(**synthetic_model_config())
        reference = LlamaForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
    target = TargetModel(copy.deepcopy(reference), device="cpu", dtype=torch.float32)
    return target, reference


def _bounded_state_operations(operations):
    if not isinstance(operations, Sequence) or not 1 <= len(operations) <= 64:
        raise InvalidCheckCase("provide between 1 and 64 synthetic state operations")
    operations = tuple(operations)
    _validate_operations(operations)
    for operation in operations:
        if len(operation.request) > 64 or len(operation.tokens) > CONTEXT_LENGTH + 1:
            raise InvalidCheckCase("synthetic operation exceeds the name or token limit")
        if any(token >= VOCAB_SIZE for token in operation.tokens):
            raise InvalidCheckCase("synthetic token is outside the vocabulary")
        if operation.length is not None and abs(operation.length) > CONTEXT_LENGTH + 1:
            raise InvalidCheckCase("synthetic crop exceeds the length limit")
    return operations


def run_synthetic_state_check(
    seed: int = 17, *, operations: Sequence[StateOperation] | None = None,
    execution: str = "eager",
) -> StateSequenceCheck:
    """Check every live cache against independent HF state, without a checkpoint.

    Release drops request state, not the target. Static execution closes its
    workspace after the check. Neither CPU mode validates CUDA graph capture.
    """
    synthetic_execution_config(execution)
    _generator(seed)
    operations = _bounded_state_operations(synthetic_state_operations(seed, execution=execution)
                                           if operations is None else operations)
    target, reference = _synthetic_models(seed)
    with ExitStack() as cleanup:
        target = _execution_target(target, execution, cleanup)
        return check_state_sequence(target, reference, operations)


def run_synthetic_check(
    seed: int = 17,
    *,
    adapter_factory: Callable[[TargetModel], Mapping[str, Decoder]] | None = None,
    execution: str = "eager",
) -> CheckReport:
    """Check tiny CPU paths against a separate, unmodified HF model.

    A custom factory receives only the candidate target. It can expose another
    eager decoder using the same weights. Eager checks greedy, block and tree;
    static checks greedy and block. This is not a performance benchmark.
    """
    synthetic_execution_config(execution)
    if execution != "eager" and adapter_factory is not None:
        raise ValueError("adapter_factory requires eager execution")
    cases = synthetic_cases(seed)
    target, reference = _synthetic_models(seed)
    with ExitStack() as cleanup:
        target = _execution_target(target, execution, cleanup)
        if adapter_factory is None:
            engines = {"greedy": cleanup.enter_context(DartEngine(target)),
                       "block": cleanup.enter_context(DartEngine(target, _ProbeDrafter()))}
            if execution == "eager":
                engines["tree"] = cleanup.enter_context(DartEngine(target, _ProbeDrafter(), verifier="tree"))
            candidates = {name: engine_decoder(engine) for name, engine in engines.items()}
        else:
            candidates = adapter_factory(target)
        return run_differential(cases, hf_decoder(reference), candidates, eos_token_ids=[2])
