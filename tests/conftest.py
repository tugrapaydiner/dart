"""Shared fixtures: a tiny random-weight Llama-style target model.

Exactness tests do not need trained weights: the contract is
"DART output == target greedy output" for *whatever* the target computes,
so a seeded random model is a perfect (and fast, download-free) test
subject.  Trained models only matter for acceptance-rate/speed numbers.
"""

from __future__ import annotations

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models.target import TargetModel

VOCAB = 211
EOS_ID = 2
FIRST_SAFE_ID = 5  # prompt tokens drawn from [FIRST_SAFE_ID, VOCAB) avoid specials


def build_tiny_target(seed: int = 1234) -> TargetModel:
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=VOCAB,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=1024,
        bos_token_id=1,
        eos_token_id=EOS_ID,
        pad_token_id=0,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg)
    return TargetModel(model, tokenizer=None, device="cpu", dtype=torch.float32)


@pytest.fixture(scope="session")
def tiny_target() -> TargetModel:
    return build_tiny_target()


def random_prompt(seed: int, length: int) -> list[int]:
    g = torch.Generator().manual_seed(seed)
    return torch.randint(
        FIRST_SAFE_ID, VOCAB, (length,), generator=g
    ).tolist()
