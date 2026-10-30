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


class _ReferenceDrafter:
    """Propose the reference continuation with one wrong token per round.

    The wrong position cycles through every draft offset plus one clean round,
    so drafts are rejected at each position and sometimes fully accepted.
    """

    name = "synthetic"
    requires_entropy = False

    def __init__(self, sequences):
        self.sequences = tuple(tuple(sequence) for sequence in sequences)
        self.rounds = 0

    def propose(self, committed, max_tokens):
        committed = tuple(committed)
        drafts = next((list(sequence[len(committed):len(committed) + max_tokens]) for sequence in self.sequences
                       if sequence[:len(committed)] == committed), None)
        if drafts is None:  # Off the reference path; any proposal is still checked.
            return _ProbeDrafter().propose(committed, max_tokens)
        wrong = self.rounds % (max_tokens + 1)
        self.rounds += 1
        if wrong < len(drafts):
            drafts[wrong] = (drafts[wrong] + 1) % VOCAB_SIZE
        return drafts

    def propose_tree(self, committed, max_tokens):
        drafts = self.propose(committed, max_tokens)
        if not drafts:
            return DraftTree([], [])
        decoy = [(drafts[0] + 2) % VOCAB_SIZE, *drafts[1:]]  # A rejected sibling ahead of the real branch.
        return DraftTree.from_branches([decoy, drafts])

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
    if execution != "eager":
        steps = []
        for operation in operations:
            if operation.action == "tree":
                steps.append(replace(operation, expect_error="ValueError"))  # Static execution rejects trees.
                steps.append(StateOperation(operation.request, "block", operation.tokens))
            else:
                steps.append(operation)
        operations = tuple(steps)
    return _reference_drafts(operations, seed)


def _reference_drafts(operations, seed):
    """Draft from the reference continuation so verification accepts tokens.

    Block drafts alternate between one wrong token and full acceptance. Tree
    drafts place the real branch behind a rejected first root, and every other
    tree also gets a wrong final node.
    """
    reference = _reference_model(seed)
    prefixes, steps, blocks, trees = {}, [], 0, 0

    @torch.inference_mode()
    def continuation(prefix, count):
        ids, out = list(prefix), []
        for _ in range(count):
            out.append(int(reference(torch.tensor([ids])).logits[0, -1].argmax()))
            ids.append(out[-1])
        return out

    def accepted(tokens, parents, cont):
        children = {}
        for index, parent in enumerate(parents):
            children.setdefault(parent, []).append(index)
        committed, node = [], -1
        while True:
            want = cont[len(committed)]
            node = next((c for c in children.get(node, []) if tokens[c] == want), None)
            if node is None or want == 2:  # A miss commits the correction; EOS ends the round.
                return committed + [want]
            committed.append(want)

    def wrong(token):
        return (token + 1) % VOCAB_SIZE

    for operation in operations:
        request, action = operation.request, operation.action
        if operation.expect_error is not None:
            steps.append(operation)
            continue
        if action in {"block", "tree"}:
            pending, drafts = operation.tokens[0], list(operation.tokens[1:])
            parents = list(operation.parents) if action == "tree" else list(range(-1, len(drafts) - 1))
            depths = []
            for parent in parents:
                depths.append(0 if parent < 0 else depths[parent] + 1)
            cont = continuation((*prefixes[request], pending), max(depths, default=-1) + 2)
            path, node = [], len(drafts) - 1
            while node >= 0:  # The real branch ends at the last packed node.
                path.append(node)
                node = parents[node]
            for node in path:
                drafts[node] = cont[depths[node]]
            for node, parent in enumerate(parents):
                if node not in path and (parent in path or parent < 0) and drafts[node] == cont[depths[node]]:
                    drafts[node] = wrong(drafts[node])  # Siblings of the real branch must miss.
            if action == "block" and blocks % 2 == 0 and len(drafts) > 1:
                drafts[1] = wrong(drafts[1])
            if action == "tree" and trees % 2 == 1 and path:
                drafts[path[0]] = wrong(drafts[path[0]])
            blocks, trees = blocks + (action == "block"), trees + (action == "tree")
            operation = replace(operation, tokens=(pending, *drafts))
            committed = accepted(drafts, parents, cont)
            prefixes[request] = (*prefixes[request], pending, *committed[:-1])
        elif action == "prefill":
            prefixes[request] = operation.tokens
        elif action == "forward":
            prefixes[request] = (*prefixes[request], *operation.tokens)
        elif action == "crop":
            prefixes[request] = prefixes[request][:operation.length]
        else:
            prefixes.pop(request, None)
        steps.append(operation)
    return tuple(steps)


def _reference_model(seed):
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        config = LlamaConfig(**synthetic_model_config())
        return LlamaForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()


def _synthetic_models(seed):
    reference = _reference_model(seed)
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
    static checks greedy and block. Drafts follow the reference output with one
    wrong token per round. This is not a performance benchmark.
    """
    synthetic_execution_config(execution)
    if execution != "eager" and adapter_factory is not None:
        raise ValueError("adapter_factory requires eager execution")
    cases = synthetic_cases(seed)
    target, reference = _synthetic_models(seed)
    with ExitStack() as cleanup:
        target = _execution_target(target, execution, cleanup)
        decode = hf_decoder(reference)
        expected = {case.name: tuple(decode(case)) for case in cases}
        sequences = [case.prompt + expected[case.name] for case in cases]
        if adapter_factory is None:
            engines = {"greedy": cleanup.enter_context(DartEngine(target)),
                       "block": cleanup.enter_context(DartEngine(target, _ReferenceDrafter(sequences)))}
            if execution == "eager":
                engines["tree"] = cleanup.enter_context(
                    DartEngine(target, _ReferenceDrafter(sequences), verifier="tree"))
            candidates = {name: engine_decoder(engine) for name, engine in engines.items()}
        else:
            candidates = adapter_factory(target)
        return run_differential(cases, lambda case: list(expected[case.name]), candidates, eos_token_ids=[2])
