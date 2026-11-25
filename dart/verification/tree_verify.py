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

import operator
from collections.abc import Iterable
from dataclasses import dataclass

import torch

from dart.models.target import TargetModel, TargetState
from dart.runtime.types import VerifyResult
from dart.verification.greedy import _cpu_greedy_indices, _restore_failed_round


def _check_parent(index: int, parent: int) -> None:
    if type(parent) is not int or not -1 <= parent < index:
        raise ValueError(f"node {index}: parent {parent} must be -1 or a preceding index")


@dataclass
class DraftTree:
    """Packed draft tree.  ``parent[i]`` is the index of node i's parent
    (-1 = child of the committed prefix).  Parents must precede children
    in the packing (topological order)."""

    tokens: list[int]
    parent: list[int]

    def __post_init__(self) -> None:
        self._check_size()
        for index, parent in enumerate(self.parent):
            _check_parent(index, parent)

    def _check_size(self) -> None:
        if len(self.tokens) != len(self.parent):
            raise ValueError("tokens and parent must have equal length")

    def __len__(self) -> int:
        return len(self.tokens)

    @property
    def depths(self) -> list[int]:
        self._check_size()
        depths: list[int] = []
        for index, parent in enumerate(self.parent):
            _check_parent(index, parent)
            depths.append(0 if parent < 0 else depths[parent] + 1)
        return depths

    def truncate(self, max_tokens: int, *, max_nodes: int | None = None) -> "DraftTree":
        """Keep nodes within the depth budget, in packing order, up to the node limit."""
        self._check_size()
        if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 0):
            raise ValueError("max_nodes must be a nonnegative integer")
        count = len(self) if max_nodes is None else min(len(self), max_nodes)
        if max_tokens >= count:
            if count == len(self):
                for index, parent in enumerate(self.parent):
                    _check_parent(index, parent)
            return self.limit(count)  # A path cannot be longer than the number of kept nodes.
        if max_tokens <= 0:
            return DraftTree([], []) if self.tokens else self

        slots = {-1: (-1, 0)}  # Packed parent and depth of its children.
        parents = []
        for index, parent in enumerate(self.parent):
            _check_parent(index, parent)
            kept_parent = slots.get(parent)
            if kept_parent is None or not kept_parent[1] < max_tokens:
                continue
            new_parent, depth = kept_parent
            slots[index] = (len(parents), depth + 1)
            parents.append(new_parent)
            if len(parents) == count:
                break
        if len(parents) == len(self):
            return self
        return DraftTree([self.tokens[i] for i in slots if i != -1], parents)

    def limit(self, max_nodes: int) -> "DraftTree":
        """Keep the first ``max_nodes`` packed nodes; parents precede children."""
        if len(self.tokens) <= max_nodes:
            return self
        return DraftTree(self.tokens[:max_nodes], self.parent[:max_nodes])

    @classmethod
    def chain(cls, tokens: list[int]) -> "DraftTree":
        """A linear block as a degenerate tree (parent = previous node)."""
        return cls(list(tokens), [i - 1 for i in range(len(tokens))])

    @classmethod
    def from_branches(cls, branches: Iterable[Iterable[int]], *, max_nodes: int | None = None) -> "DraftTree":
        """Pack independent candidate continuations, sharing common prefixes.

        Branches that start with the same tokens share nodes (a trie), so
        proposing [[a, b], [a, c]] costs three packed nodes, not four.
        With a node limit, stop reading branches as soon as the tree is full.
        """
        if max_nodes is not None:
            if type(max_nodes) is not int or max_nodes < 0:
                raise ValueError("max_nodes must be a nonnegative integer")
            if max_nodes == 0:
                return cls([], [])
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
                    if len(tokens) == max_nodes:
                        return cls(tokens, parent)
                    idx = len(tokens) - 1
                    seen[key] = idx
                p = idx
        return cls(tokens, parent)


def _is_chain(tree: DraftTree) -> bool:
    return len(tree.parent) == len(tree) and all(
        type(parent) is int and parent == index - 1 for index, parent in enumerate(tree.parent)
    )



def _scalar_edge_id(token) -> int | None:
    if isinstance(token, int) and type(token) is not bool:
        if type(token).__eq__ is int.__eq__ and type(token).__int__ is int.__int__:
            return int(token)
    if (type(token) is torch.Tensor and token.device.type == "cpu" and token.ndim == 0
            and token.layout == torch.strided and token.dtype != torch.bool):
        try:
            return operator.index(token)
        except TypeError:
            pass
    return None


def _index_tree_edges(tree: DraftTree) -> tuple[DraftTree, dict[tuple[int, int], int] | None]:
    if len(tree.parent) != len(tree.tokens):
        return tree, None
    edges = {}
    slots = []
    for index, (parent, token) in enumerate(zip(tree.parent, tree.tokens)):
        if type(parent) is not int or not -1 <= parent < index:
            return tree, None
        if type(token) is not int:
            token = _scalar_edge_id(token)
            if token is None:
                return tree, None  # Keep the target's normal input checks.
        parent = slots[parent] if parent >= 0 else -1
        slots.append(edges.setdefault((parent, token), len(edges)))
    if len(edges) < len(tree):
        tree = DraftTree([token for _, token in edges], [parent for parent, _ in edges])
    return tree, edges


def _live_tree_rows(target, tree: DraftTree, context_len: int):
    eos = target.eos_token_ids
    if not eos or not any(type(token) is int and token in eos for token in tree.tokens):
        return None
    config = getattr(getattr(target, "model", None), "config", None)
    vocab = getattr(config, "vocab_size", None)
    if type(vocab) is not int or len(tree.parent) != len(tree.tokens) or any(
        type(token) is not int or not 0 <= token < vocab for token in tree.tokens
    ):
        return None  # Keep normal admission for unknown or invalid inputs.
    depths = []
    for index, parent in enumerate(tree.parent):
        if type(parent) is not int or not -1 <= parent < index:
            return None
        depths.append(0 if parent < 0 else depths[parent] + 1)
    length = context_len + max(depths) + 1
    for limit in (getattr(config, "max_position_embeddings", None), getattr(target, "context_limit", None)):
        if limit is not None and (type(limit) is not int or length > limit):
            return None
    tokens, parents, slots = [], [], {}
    for index, (token, parent) in enumerate(zip(tree.tokens, tree.parent)):
        if token in eos or (parent >= 0 and parent not in slots):
            continue  # EOS and its descendants never supply a decision row.
        slots[index] = len(tokens)
        tokens.append(token)
        parents.append(-1 if parent < 0 else slots[parent])
    return tokens, parents, slots


def _fits_causal_block(target, pending_len: int, parents: list[int]) -> bool:
    limit = getattr(target, "max_block_tokens", None)
    if limit is not None and (type(limit) is not int or pending_len + len(parents) > limit):
        return False
    return callable(getattr(target, "forward_block", None)) and all(p == i - 1 for i, p in enumerate(parents))


@torch.inference_mode()
def verify_tree_greedy(
    target: TargetModel,
    state: TargetState,
    pending: int | list[int],
    draft_tree: "DraftTree | list[int]",
    *,
    compute_entropy: bool = True,
) -> VerifyResult:
    """Greedy tree verification.  Same state contract as the block verifier:
    on entry the cache covers everything before the pending block; on exit
    it covers all committed tokens except the new pending one.
    Set compute_entropy=False when no consumer needs that signal.
    """
    if not isinstance(draft_tree, DraftTree):
        draft_tree = DraftTree.chain(list(draft_tree))
    pending_block = [pending] if isinstance(pending, int) else list(pending)
    if not pending_block:
        raise ValueError("pending block must contain at least one token")
    n = len(draft_tree)

    chain = n == 0 or _is_chain(draft_tree)
    if not chain:
        draft_tree, edges = _index_tree_edges(draft_tree)
        chain = len(draft_tree) < n and _is_chain(draft_tree)
    if chain:
        from dart.verification.block_verify import verify_block_greedy

        result = verify_block_greedy(target, state, pending_block, draft_tree.tokens, compute_entropy=compute_entropy)
        if result.n_draft_proposed != n:
            result.n_draft_proposed = n  # Count the submitted nodes.
            result.all_accepted = False
        return result

    base_len = state.seq_len
    try:
        return _verify_branches(target, state, pending_block, draft_tree, n, edges, base_len, compute_entropy)
    except BaseException:
        _restore_failed_round(target, state, base_len)
        raise


def _verify_branches(target, state, pending_block, draft_tree, n, edges, base_len, compute_entropy):
    p_len = len(pending_block)
    live = _live_tree_rows(target, draft_tree, base_len + p_len)
    tokens, parents, slots = (draft_tree.tokens, draft_tree.parent, None) if live is None else live
    causal = live is not None and _fits_causal_block(target, p_len, parents)
    path_only = False
    if causal and not compute_entropy and getattr(target, "capture_predictions", False):
        preds = target.forward_predictions(state, pending_block + tokens)
    else:
        logits = (target.forward_block(state, pending_block + tokens) if causal
                  else target.forward_tree(state, pending_block, tokens, parents))
        path_only = logits.device.type == "cpu" and logits.size(-1) >= 32_000 and torch.get_num_threads() == 1
        if path_only:
            preds = _cpu_greedy_indices(logits[:p_len]).tolist()
        else:
            preds = torch.argmax(logits, dim=-1).tolist()

    # Divergence audit of unverified pending tokens (approx-mode metadata;
    # 0 in exact mode by construction) - matches the block verifier.
    divergence = sum(
        1 for j in range(1, p_len) if pending_block[j] != preds[j - 1]
    )

    # Greedy argmax walk from the prefix down the tree.
    path: list[int] = []          # accepted node indices, root-first
    children: dict[int, list[int]] | None = None
    cur_parent = -1
    cur_row = p_len - 1           # row deciding the next position
    accepted_eos = False
    committed = []
    want = int(preds[cur_row])
    while True:
        committed.append(want)
        if edges is not None:
            nxt = edges.get((cur_parent, want))
        else:
            if children is None:  # Some scalar IDs compare by value but hash by identity.
                children = {}
                for i, p in enumerate(draft_tree.parent):
                    children.setdefault(p, []).append(i)
            nxt = next((c for c in children.get(cur_parent, ()) if draft_tree.tokens[c] == want), None)
        if nxt is None:
            break
        path.append(nxt)
        if want in target.eos_token_ids:
            accepted_eos = True
            break
        cur_parent = nxt
        cur_row = p_len + (nxt if slots is None else slots[nxt])
        want = int(_cpu_greedy_indices(logits[cur_row]) if path_only else preds[cur_row])

    m = len(path)
    finished = committed[-1] in target.eos_token_ids

    if accepted_eos or not compute_entropy:
        entropy = 0.0
    else:
        dist = torch.softmax(logits[cur_row].float(), dim=-1)
        entropy = float(-(dist * dist.clamp_min(1e-12).log()).sum())

    # Keep the accepted path's KV entries, drop every other packed node,
    # then align with the block verifier's postcondition (the final
    # committed token is the new pending one and is never in cache).
    if not causal:  # A causal block keeps a contiguous prefix.
        kept_path = path[:-1] if accepted_eos else path  # EOS stays pending.
        if slots is not None:
            kept_path = [slots[index] for index in kept_path]
        target.gather_tree_tail(state, base_len + p_len, kept_path)
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
