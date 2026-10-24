from __future__ import annotations

import pytest
import torch

from dart.drafters.base import NullDrafter
from dart.drafters.mtp import MTPDrafter, MTPHeads
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference
from dart.verification.sampling import verify_block_sampling
from dart.verification.tree_verify import DraftTree, verify_tree_greedy
from tests.conftest import VOCAB, build_tiny_target, random_prompt
from tests.test_drafter_state import AlternatingScheduler


class RecordingHeads(MTPHeads):
    def __init__(self):
        super().__init__(64, VOCAB, 2)
        self.inputs = []

    def forward(self, hidden):
        self.inputs.append(hidden.clone())
        return super().forward(hidden)


@pytest.mark.parametrize("verifier", ["block", "reference", "tree", "sampling"])
@pytest.mark.parametrize("pending_len", [1, 3])
@pytest.mark.parametrize("finish", [False, True])
def test_mtp_features_match_the_verified_prefix(verifier, pending_len, finish):
    target = build_tiny_target()
    target.record_hidden = True
    target.eos_token_ids = frozenset()
    prompt = random_prompt(350, 8)
    continuation = target.greedy_generate(prompt, max_new_tokens=6)
    pending = continuation[:pending_len]
    drafts = continuation[pending_len:pending_len + 2]
    if finish:
        target.eos_token_ids = frozenset([drafts[-1]])
    state, _ = target.prefill(prompt)
    heads = RecordingHeads()
    drafter = MTPDrafter(target, heads)
    drafter.after_prefill()
    if verifier == "tree":
        wrong = [(t + 1) % VOCAB for t in drafts]
        result = verify_tree_greedy(target, state, pending, DraftTree.from_branches([wrong, drafts]))
    elif verifier == "sampling":
        result = verify_block_sampling(
            target, state, pending, drafts, generator=torch.Generator().manual_seed(3),
            temperature=1e-5 if finish else 1.0,
        )
    else:
        verify = verify_block_greedy if verifier == "block" else verify_greedy_reference
        result = verify(target, state, pending, drafts)
    assert result.finished is finish
    committed = prompt + pending + result.committed
    drafter.observe(result)
    drafter.propose(committed, 2)
    target.prefill(committed[:-1])
    torch.testing.assert_close(heads.inputs[-1], target.last_hidden_block[-1], atol=1e-5, rtol=1e-5)


def test_mtp_features_refresh_after_another_lane_runs(monkeypatch):
    target = build_tiny_target()
    target.record_hidden = True
    target.eos_token_ids = frozenset()
    heads = RecordingHeads()
    drafter = MTPDrafter(target, heads)
    contexts = []
    original = drafter.propose

    def propose(committed, max_tokens):
        contexts.append(list(committed))
        return original(committed, max_tokens)

    monkeypatch.setattr(drafter, "propose", propose)
    lanes = {"mtp": drafter, "target_only": NullDrafter()}
    prompt = random_prompt(351, 8)
    expected = target.greedy_generate(prompt, max_new_tokens=12)
    with DartEngine(target, lanes=lanes, scheduler=AlternatingScheduler("mtp")) as engine:
        result = engine.generate(prompt, DecodeConfig(max_new_tokens=12))
    assert result.token_ids == expected
    assert len(contexts) > 1
    for committed, hidden in zip(contexts, heads.inputs):
        target.prefill(committed[:-1])
        torch.testing.assert_close(hidden, target.last_hidden_block[-1], atol=1e-5, rtol=1e-5)
