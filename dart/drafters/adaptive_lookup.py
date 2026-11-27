"""Choose greedy, linear or branched lookup from measured verification costs."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Collection, Mapping
from numbers import Real
from threading import local

from dart.verification.tree_verify import DraftTree


def _costs(values: Mapping[int, float], minimum_rows: int) -> dict[int, float]:
    if not isinstance(values, Mapping):
        raise ValueError("verification costs must map row counts to elapsed times")
    result = {}
    for rows, cost in values.items():
        if type(rows) is not int or not minimum_rows <= rows <= 16:
            raise ValueError(f"verification costs require integer row counts from {minimum_rows} to 16")
        if isinstance(cost, bool) or not isinstance(cost, Real) or not math.isfinite(cost) or cost <= 0:
            raise ValueError("verification costs must be positive and finite")
        result[rows] = float(cost)
    return result


class _History:
    def __init__(self, eos: frozenset[int]) -> None:
        self.eos = eos
        self.tokens: list[int] = []
        self.positions: dict[int, list[int]] = defaultdict(list)
        self.counts: Counter[int] = Counter()
        self.survival = defaultdict(lambda: [0, 0])
        self.pending = []
        self.context = (0, False)
        self.offers = {}

    def _continuations(self):
        tokens = self.tokens
        positions = self.positions[tokens[-1]]
        for length in range(min(4, len(tokens) - 1), 0, -1):
            suffix = tokens[-length:]
            branches = []
            for index in range(len(positions) - 2, -1, -1):
                end = positions[index]
                if end < length - 1:
                    break
                if tokens[end - length + 1:end + 1] != suffix:
                    continue
                path = tuple(tokens[end + 1:end + 9])
                if path not in branches:
                    branches.append(path)
                    if len(branches) == 2:
                        break
            if branches:
                return length, branches
        return 0, []

    def _offers(self):
        length, branches = self._continuations()
        context = (min(length, 2), len(branches) == 2)
        if branches:
            paths = []
            for branch in branches:
                stop = next((i for i, token in enumerate(branch) if token in self.eos), len(branch))
                if stop:
                    paths.append(branch[:stop])  # A matching EOS already ends the target step.
            offers = {f"copy{n}": (paths[0][:n],) for n in (2, 4, 8) if paths}
            pair = tuple(dict.fromkeys(path[:4] for path in paths))
            if len(pair) == 2:
                offers["pair"] = pair
        else:
            ranked = sorted((t for t in self.counts if t not in self.eos),
                            key=lambda t: (self.counts[t], self.positions[t][-1]), reverse=True)
            roots = [(t,) for t in ranked[:15]]
            offers = {f"roots{n}": tuple(roots[:n]) for n in (2, 4, 8, 15) if roots}
        return context, offers

    def append(self, token: int) -> None:
        pending = []
        for context, name, depth, paths in self.pending:
            hits = [p for p in paths if p[0] == token]
            for key in (context, None):
                count = self.survival[key, name, depth]
                count[0] += bool(hits)
                count[1] += 1
            tails = tuple(p[1:] for p in hits if len(p) > 1)
            if tails:
                pending.append((context, name, depth + 1, tails))
        self.positions[token].append(len(self.tokens))
        self.counts[token] += 1
        self.tokens.append(token)
        self.context, self.offers = self._offers()
        pending.extend((self.context, name, 1, paths) for name, paths in self.offers.items())
        self.pending = pending

    def expected_advance(self, name, depth):
        survival = advance = 1.0
        for step in range(1, depth + 1):
            hits, seen = self.survival.get((None, name, step), (0, 0))
            prior = (hits + 0.5) / (seen + 1)
            hits, seen = self.survival.get((self.context, name, step), (0, 0))
            survival *= (hits + prior) / (seen + 1)
            advance += survival
        return advance


class AdaptiveLookupDrafter:
    """Price recent continuations and frequent roots using confirmed tokens.

    Costs are elapsed times in any common unit, keyed by forward row count
    including the pending token. Supply block row 1 for greedy fallback.
    Missing shapes are skipped. With the tree verifier the drafter can choose
    several branches. The block verifier uses linear candidates. Both support
    up to eight draft positions and fifteen nodes. Requests longer than
    max_history use greedy fallback.
    """

    name = "adaptive_lookup"
    requires_entropy = False

    def __init__(self, block_costs: Mapping[int, float], tree_costs: Mapping[int, float], *,
                 max_history: int = 1024, eos_token_ids: Collection[int] = ()) -> None:
        self._block = _costs(block_costs, 1)
        self._tree = _costs(tree_costs, 3)
        if 1 not in self._block:
            raise ValueError("block_costs must include one row for greedy decoding")
        if type(max_history) is not int or not 1 <= max_history <= 1024:
            raise ValueError("max_history must be an integer from 1 to 1024")
        self._max_history = max_history
        self._eos = frozenset(eos_token_ids)
        self._request = local()

    def after_prefill(self) -> None:
        self._request.history = None

    def finish_request(self) -> None:
        if hasattr(self._request, "history"):
            del self._request.history

    def observe(self, result) -> None:
        pass  # The next committed prefix supplies all feedback.

    def _choose(self, committed, max_tokens, max_nodes, eos, allow_tree):
        if len(committed) > self._max_history:
            if hasattr(self._request, "history"):
                self._request.history = None
            return ()
        if not committed or max_tokens <= 0 or max_nodes == 0:
            return ()
        history = getattr(self._request, "history", None)
        if history is None or history.eos != eos or committed[:len(history.tokens)] != history.tokens:
            history = _History(eos)
        for token in committed[len(history.tokens):]:
            history.append(token)
        if hasattr(self._request, "history"):
            self._request.history = history

        best = (self._block[1], 1)
        choice = ()
        for name, branches in history.offers.items():
            paths = tuple(dict.fromkeys(path[:max_tokens] for path in branches))
            depth = max(map(len, paths))
            nodes = len({path[:i] for path in paths for i in range(1, len(path) + 1)})
            if nodes > max_nodes or (nodes != depth and not allow_tree):
                continue
            cost = (self._block if nodes == depth else self._tree).get(nodes + 1)
            if cost is not None:
                rank = (cost / history.expected_advance(name, depth), nodes + 1)
                if rank < best:
                    best, choice = rank, paths
        return choice

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        paths = self._choose(committed, max_tokens, 15, self._eos, False)
        return list(max(paths, key=len)) if paths else []

    def propose_tree(self, committed: list[int], max_tokens: int, *, max_nodes: int | None = None,
                     eos_token_ids: Collection[int] | None = None) -> DraftTree:
        if max_nodes is not None and (type(max_nodes) is not int or max_nodes < 0):
            raise ValueError("max_nodes must be a nonnegative integer or None")
        limit = 15 if max_nodes is None else min(15, max_nodes)
        eos = self._eos if eos_token_ids is None else frozenset(eos_token_ids)
        paths = self._choose(committed, min(max_tokens, limit), limit, eos, True)
        return DraftTree.from_branches(paths)
