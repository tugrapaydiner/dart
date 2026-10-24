"""Tree verification: check several candidate branches with one forward.

The block verifier (verification/block_verify.py) checks one linear draft
block per target pass.  A *tree* verifier packs multiple candidate
continuations that share prefixes — e.g. two plausible next tokens, each
with its own tail — into a single forward, using a tree-structured
attention mask (attention/verifier_mask.tree_mask) and depth-based
position ids.  The target's argmax walk then selects at most one
root-to-leaf path; the longest matching path is committed plus the usual
correction (on mismatch) or bonus token (on full acceptance).

Exactness is preserved by the same argument as the block verifier: every
committed token is the target's own full-precision argmax at its position.
A packed node attends to exactly the committed prefix plus its ancestor
chain, and its position id is its depth — so its logits row equals the row
a linear forward over that branch would produce (up to the reduction-order
caveat of docs/exactness_contract.md, shared with block verification).

KV handling: the forward appends cache entries for *all* packed nodes.
Because a node's K/V depend only on tokens the mask lets it see, the
accepted path's entries are exactly the entries a linear pass would have
written; ``TargetModel.gather_tree_tail`` keeps those, drops the rest, and
re-establishes the round invariant.

A chain (each node parented on the previous one) makes tree verification
coincide with block verification decision-for-decision; the test suite
enforces that equivalence, which anchors the tree path to the block/
reference verifier pair.

Requires a target whose HF model honors custom 4D attention masks and
explicit ``position_ids`` (Llama-family models with eager attention do;
the suite validates the tiny reference target).  Greedy/exact mode only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from dart.models.target import TargetModel, TargetState
from dart.runtime.types import VerifyResult


@dataclass
class DraftTree:
    """Packed draft tree.  ``parent[i]`` is the index of node i's parent
    (-1 = child of the committed prefix).  Parents must precede children
    in the packing (topological order)."""

    tokens: list[int]
    parent: list[int]
    _depths: list[int] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        if len(self.tokens) != len(self.parent):
            raise ValueError("tokens and parent must have equal length")
        for i, p in enumerate(self.parent):
            if p >= i or p < -1:
                raise ValueError(
                    f"node {i}: parent {p} must be -1 or a preceding index"
                )
        d: list[int] = []
        for p in self.parent:
            d.append(0 if p < 0 else d[p] + 1)
        self._depths = d

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def depths(self) -> list[int]:
        return self._depths

    def truncate(self, max_tokens: int) -> "DraftTree":
        if not self.tokens or max(self._depths) < max_tokens:
            return self
        kept = [i for i, depth in enumerate(self._depths) if depth < max_tokens]
        remap = {old: new for new, old in enumerate(kept)}
        return DraftTree(
            [self.tokens[i] for i in kept],
            [-1 if self.parent[i] < 0 else remap[self.parent[i]] for i in kept],
        )

    @classmethod
    def chain(cls, tokens: list[int]) -> "DraftTree":
        """A linear block as a degenerate tree (parent = previous node)."""
        return cls(list(tokens), [i - 1 for i in range(len(tokens))])

    @classmethod
    def from_branches(cls, branches: list[list[int]]) -> "DraftTree":
        """Pack independent candidate continuations, sharing common prefixes.

        Branches that start with the same tokens share nodes (a trie), so
        proposing [[a, b], [a, c]] costs three packed nodes, not four.
        """
        tokens: list[int] = []
        parent: list[int] = []
        # trie: (parent_idx, token) -> node_idx
        seen: dict[tuple[int, int], int] = {}
        for branch in branches:
            p = -1
            for tok in branch:
                key = (p, int(tok))
                idx = seen.get(key)
                if idx is None:
                    tokens.append(int(tok))
                    parent.append(p)
                    idx = len(tokens) - 1
                    seen[key] = idx
                p = idx
        return cls(tokens, parent)


@torch.inference_mode()
def verify_tree_greedy(
    target: TargetModel,
    state: TargetState,
    pending: int | list[int],
    draft_tree: "DraftTree | list[int]",
) -> VerifyResult:
    """Greedy tree verification.  Same state contract as the block verifier:
    on entry the cache covers everything before the pending block; on exit
    it covers all committed tokens except the new pending one."""
    if not isinstance(draft_tree, DraftTree):
        draft_tree = DraftTree.chain(list(draft_tree))
    pending_block = [pending] if isinstance(pending, int) else list(pending)
    if not pending_block:
        raise ValueError("pending block must contain at least one token")
    p_len = len(pending_block)
    n = len(draft_tree)
    base_len = state.seq_len

    if n == 0:
        # No drafts: identical to a plain target step via the block path.
        from dart.verification.block_verify import verify_block_greedy

        return verify_block_greedy(target, state, pending_block, [])

    logits = target.forward_tree(
        state, pending_block, draft_tree.tokens, draft_tree.parent
    )
    preds = torch.argmax(logits, dim=-1).tolist()

    # Divergence audit of unverified pending tokens (approx-mode metadata;
    # 0 in exact mode by construction) - matches the block verifier.
    divergence = sum(
        1 for j in range(1, p_len) if pending_block[j] != preds[j - 1]
    )

    children: dict[int, list[int]] = {}
    for i, p in enumerate(draft_tree.parent):
        children.setdefault(p, []).append(i)

    # Greedy argmax walk from the prefix down the tree.
    path: list[int] = []          # accepted node indices, root-first
    cur_parent = -1
    cur_row = p_len - 1           # row deciding the next position
    accepted_eos = False
    correction: int | None = None
    while True:
        want = preds[cur_row]
        nxt = next(
            (c for c in children.get(cur_parent, [])
             if draft_tree.tokens[c] == want),
            None,
        )
        if nxt is None:
            correction = want  # target's own token; may terminate the walk
            break
        path.append(nxt)
        if draft_tree.tokens[nxt] in target.eos_token_ids:
            accepted_eos = True
            break
        cur_parent = nxt
        cur_row = p_len + nxt

    m = len(path)
    if accepted_eos:
        committed = [draft_tree.tokens[i] for i in path]
        finished = True
    else:
        committed = [draft_tree.tokens[i] for i in path] + [int(correction)]
        finished = committed[-1] in target.eos_token_ids

    if accepted_eos:
        entropy = 0.0
    else:
        dist = torch.softmax(logits[cur_row].float(), dim=-1)
        entropy = float(-(dist * dist.clamp_min(1e-12).log()).sum())

    # Keep the accepted path's KV entries, drop every other packed node,
    # then align with the block verifier's postcondition (the final
    # committed token is the new pending one and is never in cache).
    target.gather_tree_tail(state, base_len + p_len, path)
    target.crop(state, base_len + p_len - 1 + len(committed))

    return VerifyResult(
        committed=committed,
        n_draft_proposed=n,
        n_draft_accepted=m,
        all_accepted=(m == n),
        finished=finished,
        pending_divergence=divergence,
        entropy=entropy,
        hidden_index=cur_row,
    )
