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
