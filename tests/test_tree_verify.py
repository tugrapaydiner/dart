"""Tree verification (docs/design.md §16 roadmap item, now implemented).

Anchoring strategy: a chain-shaped tree must reproduce the block verifier
decision-for-decision (committed tokens AND final KV length), which ties
the tree path to the block/reference verifier pair the exactness contract
already rests on.  Branching trees must then (a) stay byte-exact end to
end under adversarial branch sets and (b) actually accept a correct branch
packed among garbage ones.
"""

from __future__ import annotations

import pytest
import torch

from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.block_verify import verify_block_greedy
from dart.verification.tree_verify import DraftTree, verify_tree_greedy
from tests.conftest import VOCAB, random_prompt

PROMPTS = [(311, 6), (312, 12), (313, 20), (314, 9)]


# --------------------------- DraftTree ------------------------------- #
def test_drafttree_validates_topological_parents():
    with pytest.raises(ValueError):
        DraftTree([1, 2], [1, -1])  # parent after child
    with pytest.raises(ValueError):
        DraftTree([1], [-2])
    with pytest.raises(ValueError):
        DraftTree([1, 2], [-1])  # length mismatch


def test_drafttree_chain_shape():
    t = DraftTree.chain([7, 8, 9])
    assert t.parent == [-1, 0, 1]
    assert t.depths == [0, 1, 2]


def test_drafttree_from_branches_shares_prefixes():
    t = DraftTree.from_branches([[5, 6, 7], [5, 6, 8], [9]])
    # trie: 5,6 shared; 7 and 8 fork under 6; 9 is its own root child
    assert len(t) == 5
    assert t.tokens == [5, 6, 7, 8, 9]
    assert t.parent == [-1, 0, 1, 1, -1]
    assert t.depths == [0, 1, 2, 2, 0]


# ------------------- chain == block verifier ------------------------- #
def _drafts_for(kind: str, prompt: list[int], target, k: int) -> list[int]:
    if kind == "oracle":
        return target.greedy_generate(prompt, max_new_tokens=k)
    if kind == "garbage":
        g = torch.Generator().manual_seed(sum(prompt))
        return torch.randint(5, VOCAB, (k,), generator=g).tolist()
    return [7] * k  # constant


@pytest.mark.parametrize("kind", ["oracle", "garbage", "constant"])
@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_chain_tree_equals_block_verifier(tiny_target, kind, seed, plen):
    prompt = random_prompt(seed, plen)
    drafts = _drafts_for(kind, prompt + [11], tiny_target, 5)

    state_a, logits_a = tiny_target.prefill(prompt)
    pending = int(torch.argmax(logits_a).item())
    res_block = verify_block_greedy(tiny_target, state_a, pending, drafts)

    state_b, _ = tiny_target.prefill(prompt)
    res_tree = verify_tree_greedy(
        tiny_target, state_b, pending, DraftTree.chain(drafts)
    )

    assert res_tree.committed == res_block.committed
    assert res_tree.n_draft_accepted == res_block.n_draft_accepted
    assert res_tree.finished == res_block.finished
    assert state_b.seq_len == state_a.seq_len  # identical KV postcondition


# ------------------- branching trees, single round ------------------- #
@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_tree_accepts_oracle_branch_among_garbage(tiny_target, seed, plen):
    prompt = random_prompt(seed, plen)
    state, logits = tiny_target.prefill(prompt)
    pending = int(torch.argmax(logits).item())

    oracle = tiny_target.greedy_generate(prompt + [pending], max_new_tokens=3)
    g = torch.Generator().manual_seed(seed)
    garbage = [
        torch.randint(5, VOCAB, (3,), generator=g).tolist() for _ in range(2)
    ]
    tree = DraftTree.from_branches(garbage[:1] + [oracle] + garbage[1:])

    res = verify_tree_greedy(tiny_target, state, pending, tree)
    # the walk must find the oracle branch and ride it to its leaf + bonus
    # (or stop early only at an EOS inside the branch)
    if not res.finished:
        assert res.committed[:3] == oracle
        assert res.n_draft_accepted == 3
    # postcondition: cache covers all committed tokens except the new pending
    assert state.seq_len == plen + 1 + len(res.committed) - 1


@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_tree_all_garbage_branches_still_commits_target_token(
    tiny_target, seed, plen
):
    prompt = random_prompt(seed, plen)
    state, logits = tiny_target.prefill(prompt)
    pending = int(torch.argmax(logits).item())
    base = tiny_target.greedy_generate(prompt + [pending], max_new_tokens=1)

    tree = DraftTree.from_branches([[3, 4], [4, 3]])  # near-certainly wrong
    res = verify_tree_greedy(tiny_target, state, pending, tree)
    if res.n_draft_accepted == 0:
        assert res.committed == base  # pure correction == target's own token


# ------------------- engine end-to-end exactness --------------------- #
class FanTreeDrafter:
    """Adversarial tree drafter: random branches, one of which is
    occasionally the truth (via a planted oracle branch)."""

    name = "fan_tree"

    def __init__(self, target, plant_oracle: bool, seed: int = 0):
        self.target = target
        self.plant = plant_oracle
        self.g = torch.Generator().manual_seed(seed)

    def propose(self, committed, max_tokens):  # linear fallback path
        if max_tokens <= 0:
            return []
        return torch.randint(5, VOCAB, (max_tokens,), generator=self.g).tolist()

    def propose_tree(self, committed, max_tokens):
        if max_tokens <= 0:
            return DraftTree([], [])
        k = min(3, max_tokens)
        branches = [
            torch.randint(5, VOCAB, (k,), generator=self.g).tolist()
            for _ in range(3)
        ]
        if self.plant:
            branches[1] = self.target.greedy_generate(
                committed, max_new_tokens=k
            )
        return DraftTree.from_branches(branches)

    def observe(self, result):
        pass


@pytest.mark.parametrize("plant_oracle", [False, True])
@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_engine_tree_verifier_is_byte_exact(tiny_target, plant_oracle, seed, plen):
    prompt = random_prompt(seed, plen)
    n = 32
    base = tiny_target.greedy_generate(prompt, max_new_tokens=n)
    engine = DartEngine(
        tiny_target,
        FanTreeDrafter(tiny_target, plant_oracle, seed=seed),
        verifier="tree",
    )
    out = engine.generate(prompt, DecodeConfig(max_new_tokens=n, draft_len=4))
    assert out.token_ids == base
    assert out.metadata["exact_greedy_equivalent"] is True
    assert out.metadata["verifier"] == "tree"
    if plant_oracle:
        # planted truth must actually be exploited
        assert out.metadata["draft_tokens_accepted"] > 0


@pytest.mark.parametrize("seed,plen", PROMPTS)
def test_engine_tree_verifier_with_plain_drafter_chains(tiny_target, seed, plen):
    """Drafters without propose_tree degrade to chains and stay exact."""
    prompt = random_prompt(seed, plen)
    n = 24

    class LinearGarbage:
        name = "lin"

        def __init__(self):
            self.g = torch.Generator().manual_seed(99)

        def propose(self, committed, max_tokens):
            if max_tokens <= 0:
                return []
            return torch.randint(5, VOCAB, (max_tokens,), generator=self.g).tolist()

        def observe(self, result):
            pass

    base = tiny_target.greedy_generate(prompt, max_new_tokens=n)
    out = DartEngine(tiny_target, LinearGarbage(), verifier="tree").generate(
        prompt, DecodeConfig(max_new_tokens=n, draft_len=4)
    )
    assert out.token_ids == base


def test_engine_ngram_tree_lane_exact(tiny_target):
    # repetitive prompt so the n-gram lane has real branches to offer
    span = random_prompt(881, 7)
    prompt = span + random_prompt(882, 3) + span
    n = 32
    base = tiny_target.greedy_generate(prompt, max_new_tokens=n)
    out = DartEngine(
        tiny_target, NGramDrafter(max_ngram=4, tree_width=3), verifier="tree"
    ).generate(prompt, DecodeConfig(max_new_tokens=n, draft_len=4))
    assert out.token_ids == base


def test_tree_verifier_rejects_non_exact_modes(tiny_target):
    engine = DartEngine(tiny_target, NGramDrafter(), verifier="tree")
    with pytest.raises(ValueError, match="exact"):
        engine.generate([5, 6, 7], DecodeConfig(mode="exact_sampling"))
    with pytest.raises(ValueError, match="exact"):
        engine.generate([5, 6, 7], DecodeConfig(mode="approx", verify_every=2))


# ------------------- multi-round KV integrity ------------------------ #
def test_tree_rounds_leave_reusable_cache(tiny_target):
    """After a tree round, continuing with plain block rounds must still
    match the baseline - i.e. the gathered cache is genuinely correct,
    not just long enough."""
    prompt = random_prompt(400, 10)
    n = 40
    base = tiny_target.greedy_generate(prompt, max_new_tokens=n)

    class AlternatingDrafter(FanTreeDrafter):
        name = "alt"

    out = DartEngine(
        tiny_target,
        AlternatingDrafter(tiny_target, plant_oracle=True, seed=5),
        verifier="tree",
    ).generate(prompt, DecodeConfig(max_new_tokens=n, draft_len=3))
    assert out.token_ids == base
