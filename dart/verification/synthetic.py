"""Small CPU checks using random weights, with no checkpoint or download."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping

import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.verification.differential import (
    CheckReport,
    Decoder,
    GenerationCase,
    engine_decoder,
    hf_decoder,
    run_differential,
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


def synthetic_cases(seed: int = 17) -> tuple[GenerationCase, ...]:
    """Six bounded cases, including zero output and a full context boundary."""
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer in [0, 2**63)")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    cases = []
    for index, (length, budget) in enumerate(((1, 0), (4, 1), (7, 2), (8, 5), (51, 13), (8, 8))):
        prompt = torch.randint(3, VOCAB_SIZE, (length,), generator=generator).tolist()
        cases.append(GenerationCase(f"case-{index}", tuple(prompt), budget))
    return tuple(cases)


def run_synthetic_check(
    seed: int = 17,
    *,
    adapter_factory: Callable[[TargetModel], Mapping[str, Decoder]] | None = None,
) -> CheckReport:
    """Check tiny eager CPU paths against a separate, unmodified HF model.

    A custom factory receives only the candidate target. It can expose another
    decoder using the same weights. The default exercises greedy, block and
    tree engines. This is a small output check, not a performance benchmark.
    """
    cases = synthetic_cases(seed)
    with torch.random.fork_rng(devices=[]), torch.device("cpu"):
        torch.random.default_generator.manual_seed(seed)
        config = LlamaConfig(**synthetic_model_config())
        reference = LlamaForCausalLM(config).to(device="cpu", dtype=torch.float32).eval()
    target = TargetModel(copy.deepcopy(reference), device="cpu", dtype=torch.float32)
    if adapter_factory is None:
        candidates = {
            "greedy": engine_decoder(DartEngine(target)),
            "block": engine_decoder(DartEngine(target, _ProbeDrafter())),
            "tree": engine_decoder(DartEngine(target, _ProbeDrafter(), verifier="tree")),
        }
    else:
        candidates = adapter_factory(target)
    return run_differential(cases, hf_decoder(reference), candidates, eos_token_ids=[2])
