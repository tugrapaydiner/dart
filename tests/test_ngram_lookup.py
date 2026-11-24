import random

import pytest

from dart.drafters.ngram import NGramDrafter
from dart.verification.tree_verify import DraftTree


def reference_branches(sequence, budget, minimum, maximum, scan, width):
    if budget <= 0:
        return []
    lower = 0 if scan is None else max(0, len(sequence) - scan)
    for size in range(min(maximum, len(sequence) - 1), minimum - 1, -1):
        suffix = tuple(sequence[-size:])
        matches = []
        for start in range(lower, len(sequence) - size):
            if tuple(sequence[start:start + size]) == suffix:
                matches.append(sequence[start + size:start + size + budget])
        branches = []
        for branch in reversed(matches):
            if branch not in branches:
                branches.append(branch)
            if len(branches) == width:
                break
        if branches:
            return branches
    return []


@pytest.mark.parametrize("length", [0, 1, 2, 7, 31, 128, 1024])
@pytest.mark.parametrize("vocab", [1, 3, 41])
@pytest.mark.parametrize("scan", [None, 1, 9, 64])
def test_lookup_matches_exhaustive_search(length, vocab, scan):
    rng = random.Random(1000 + length + vocab)
    sequence = [rng.randrange(vocab) for _ in range(length)]
    saved = list(sequence)
    for minimum, maximum in [(1, 1), (1, 4), (2, 6), (9, 12)]:
        drafter = NGramDrafter(min_ngram=minimum, max_ngram=maximum, max_scan=scan, tree_width=3)
        for budget in [0, 1, 4]:
            expected = reference_branches(sequence, budget, minimum, maximum, scan, 3)
            assert drafter.propose(sequence, budget) == (expected[0] if expected else [])
            assert drafter.propose_tree(sequence, budget) == DraftTree.from_branches(expected)
    assert sequence == saved


def test_lookup_keeps_longest_match_recency_overlap_and_distinct_branches():
    drafter = NGramDrafter(max_ngram=2, tree_width=3)
    sequence = [1, 2, 7, 8, 1, 2, 7, 8, 1, 2, 9, 8, 1, 2]
    assert drafter.propose(sequence, 2) == [9, 8]
    assert drafter.propose_tree(sequence, 2) == DraftTree.from_branches([[9, 8], [7, 8]])
    assert drafter.propose([1, 1, 1], 4) == [1]
    assert drafter.propose([1, 2, 8, 9, 2, 7, 1, 2], 2) == [8, 9]


def test_lookup_honors_scan_boundary_and_reused_mutable_inputs():
    drafter = NGramDrafter(min_ngram=2, max_ngram=2, max_scan=4)
    sequence = [1, 2, 7, 1, 2]
    assert drafter.propose(sequence, 2) == []
    drafter.max_scan = 5
    assert drafter.propose(sequence, 2) == [7, 1]
    sequence[:] = [4, 5, 6, 4, 5]
    assert drafter.propose(sequence, 2) == [6, 4]
    sequence.append(3)
    assert drafter.propose(sequence, 2) == []
    assert drafter.propose([1, 2, 7, 1, 2], 2) == [7, 1]


@pytest.mark.parametrize("tree", [False, True])
def test_suffix_search_avoids_rescanning_a_missing_token(tree):
    compared = []

    class LastToken(int):
        def __eq__(self, other):
            compared.append(other)
            return super().__eq__(other)

    history = [*range(127), LastToken(127)]
    drafter = NGramDrafter(max_ngram=4)
    if tree:
        assert drafter.propose_tree(history, 2).tokens == []
    else:
        assert drafter.propose(history, 2) == []
    assert sorted(compared) == list(range(127))


@pytest.mark.parametrize("tree", [False, True])
def test_suffix_search_avoids_lengths_that_cannot_fit_at_the_last_match(tree):
    suffix_lengths = []

    class History(list):
        def __getitem__(self, key):
            items = super().__getitem__(key)
            if isinstance(key, slice) and key.stop is None:
                suffix_lengths.append(len(items))
            return items

    history = History(range(1024))
    history[-1] = history[-8]
    drafter = NGramDrafter(max_ngram=1023, max_scan=12)
    if tree:
        assert drafter.propose_tree(history, 2) == DraftTree.chain([1017, 1018])
    else:
        assert drafter.propose(history, 2) == [1017, 1018]
    assert max(suffix_lengths, default=0) <= 5


@pytest.mark.parametrize("cap", [0, 1, 3, 5, 8, 15, 25])
def test_bounded_lookup_keeps_the_first_packed_nodes(cap):
    rng = random.Random(37)
    drafter = NGramDrafter(tree_width=12)
    for sequence in ([rng.randrange(4) for _ in range(96)] for _ in range(20)):
        for depth in (0, 1, 4, 8):
            branches = reference_branches(sequence, depth, 1, 4, None, 12)
            expected = DraftTree.from_branches(branches).limit(cap)
            assert drafter.propose_tree(sequence, depth, max_nodes=cap) == expected


def test_bounded_packing_stops_at_the_last_retained_token():
    def final_branch():
        yield from (1, 4, 5)
        pytest.fail("read past the last retained token")

    def branches():
        yield [1, 2, 3]
        yield [1, 2, 3]
        yield final_branch()
        pytest.fail("read past the last retained branch")

    assert DraftTree.from_branches(branches(), max_nodes=5) == DraftTree([1, 2, 3, 4, 5], [-1, 0, 1, 0, 3])
    assert DraftTree.from_branches(branches(), max_nodes=0) == DraftTree([], [])


def test_bounded_lookup_stops_searching_older_matches():
    reads = []

    class History(list):
        def __getitem__(self, key):
            if isinstance(key, slice) and key.start is not None and key.stop == key.start + 8:
                reads.append(key.start)
            return super().__getitem__(key)

    sequence = History()
    suffix = [1, 2, 3, 4]
    for token in range(10, 42):
        sequence.extend(suffix + [token] * 8)
    sequence.extend(suffix)
    drafter = NGramDrafter(tree_width=32)
    assert len(drafter.propose_tree(sequence, 8, max_nodes=15)) == 15
    assert reads == [len(sequence) - 12, len(sequence) - 24]


@pytest.mark.parametrize("cap", [-1, True, 1.5])
def test_bounded_packing_rejects_invalid_caps_before_lookup(cap):
    def branches():
        pytest.fail("read a branch before validating the cap")
        yield []

    with pytest.raises(ValueError, match="max_nodes"):
        DraftTree.from_branches(branches(), max_nodes=cap)
