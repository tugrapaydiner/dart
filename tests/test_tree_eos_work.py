from contextlib import closing
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dart.models.replay import CudaGraphTarget
from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference
from dart.verification.sampling import verify_block_sampling
from dart.verification.state_checks import _cache_failures, _reference_forward
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import DraftTree, verify_tree_greedy
from tests.test_block_eos_work import CountingTarget
from tests.test_tree_linear_work import TreeTarget


@pytest.mark.parametrize('tokens,parents,eos,live,live_parents,branch', [
    ([11, 12, 13, 50, 51], [-1, 0, 1, -1, 3], [11], [50, 51], [-1, 0], [11, 12, 13]),
    ([11, 12, 13, 14, 50, 51], [-1, 0, 1, 2, -1, 4], [13], [11, 12, 50, 51], [-1, 0, -1, 2], [11, 12, 13, 14]),
    ([50, 51, 52, 11, 12, 13], [-1, 0, 1, -1, 3, 4], [50], [11, 12, 13], [-1, 0, 1], [11, 12, 13]),
    ([11, 50], [-1, -1], [11, 50], [], [], [11]),
    ([50, 11, 51, 12, 13, 52, 14], [-1, -1, 0, 1, 3, 2, 4], [50], [11, 12, 13, 14], [-1, 0, 1, 2], [11, 12, 13, 14]),
    ([11, 12, 13, 50, 51], [-1, 0, 1, 0, 3], [12, 66], [11, 50, 51], [-1, 0, 1], [11, 12, 13]),
    ([50, 51, 2, 3], [-1, 0, -1, 2], [2], [50, 51], [-1, 0], []),
])
@pytest.mark.parametrize('pending', [[10], [7, 9, 10]])
@pytest.mark.parametrize('entropy', [False, True])
def test_terminal_rows_are_not_executed(tokens, parents, eos, live, live_parents, branch, pending, entropy):
    target, reference = TreeTarget(eos, False), CountingTarget(eos, False)
    state, expected_state = SimpleNamespace(seq_len=3), SimpleNamespace(seq_len=3)
    tree = DraftTree(tokens, parents)
    original = deepcopy(tree)
    expected = verify_greedy_reference(reference, expected_state, pending, branch)
    actual = verify_tree_greedy(target, state, pending, tree, compute_entropy=entropy)
    assert actual.committed == expected.committed and actual.finished == expected.finished
    assert actual.n_draft_accepted == expected.n_draft_accepted
    assert actual.n_draft_proposed == len(tokens)
    assert actual.all_accepted == (actual.n_draft_accepted == len(tokens))
    assert actual.pending_divergence == expected.pending_divergence
    assert state.seq_len == expected_state.seq_len and tree == original
    causal = live_parents == [i - 1 for i in range(len(live))]
    assert target.trees == ([] if causal else [(live, live_parents)])
    assert target.calls == [pending + live]
    assert len(target.gathers) == (0 if causal else 1)
    if causal:
        assert actual.hidden_index == len(pending) + len(expected.committed) - 2
    if not entropy or (actual.finished and actual.n_draft_accepted == len(actual.committed)):
        assert actual.entropy == 0
    else:
        assert actual.entropy > 0


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield _synthetic_models(17)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('mismatch', [None, 0, 1, 3])
@pytest.mark.parametrize('eos_index', [0, 2])
@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('entropy', [False, True])
@torch.inference_mode()
def test_pruned_tree_matches_native_cache_hidden_and_followup(models, monkeypatch, mismatch, eos_index, reverse, entropy):
    target, reference = models
    target.record_hidden = True
    prompt, pending = [5, 7, 9], [4, 6]
    oracle, prefix = [], prompt + pending
    for _ in range(5):
        token = int(_reference_forward(reference, prefix).logits[0, -1].argmax())
        oracle.append(token)
        prefix = prefix + [token]
    eos = oracle[eos_index]
    assert eos not in oracle[:eos_index]
    target.eos_token_ids = frozenset([eos])
    wrong = next(token for token in range(67) if token not in oracle)
    drafts = oracle[:4]
    if mismatch is not None:
        drafts[mismatch] = wrong
    branches = [[wrong, eos, 8], drafts]
    tree = DraftTree.from_branches(branches[::-1] if reverse else branches)
    original = deepcopy(tree)
    expected = oracle[:min(eos_index + 1, 5 if mismatch is None else mismatch + 1)]
    state, _ = target.prefill(prompt)
    calls, forward, block = [], target.forward_tree, target.forward_block

    def record_block(state, tokens):
        calls.append((list(tokens[len(pending):]), None))
        return block(state, tokens)

    def record(state, pending, tokens, parents):
        calls.append((list(tokens), list(parents)))
        return forward(state, pending, tokens, parents)

    monkeypatch.setattr(target, 'forward_tree', record)
    monkeypatch.setattr(target, 'forward_block', record_block)
    result = verify_tree_greedy(target, state, pending, tree, compute_entropy=entropy)
    assert result.committed == expected and result.finished == (expected[-1] == eos)
    assert result.n_draft_proposed == len(original) and tree == original
    assert len(calls) == 1 and len(calls[0][0]) < len(tree)
    assert eos not in calls[0][0]
    live = prompt + pending + expected[:-1]
    native = _reference_forward(reference, live)
    assert not _cache_failures(state, native.past_key_values, len(live), atol=2e-6, rtol=2e-5)
    hidden = reference(input_ids=torch.tensor([live]), attention_mask=torch.ones((1, len(live)), dtype=torch.long),
                       use_cache=False, output_hidden_states=True).hidden_states[-1][0, -1]
    torch.testing.assert_close(target.last_hidden_block[result.hidden_index], hidden, atol=2e-6, rtol=2e-5)
    if entropy and not (result.finished and result.n_draft_accepted == len(expected)):
        probabilities = native.logits[0, -1].float().softmax(-1)
        wanted_entropy = float(-(probabilities * probabilities.clamp_min(1e-12).log()).sum())
        assert result.entropy == pytest.approx(wanted_entropy, abs=2e-6)
    else:
        assert result.entropy == 0
    followup = target.forward_block(state, [expected[-1]])[-1]
    native_next = _reference_forward(reference, live + [expected[-1]])
    torch.testing.assert_close(followup, native_next.logits[0, -1], atol=2e-6, rtol=2e-5)
    assert not _cache_failures(state, native_next.past_key_values, len(live) + 1, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('bad', [-1, 67, 2**80, 67.9])
@torch.inference_mode()
def test_invalid_dead_tokens_still_fail_before_model_execution(models, monkeypatch, bad):
    target, _ = models
    target.eos_token_ids = frozenset([2])
    state, _ = target.prefill([5, 7, 9])
    tree = DraftTree([2, bad, 11], [-1, 0, -1])
    monkeypatch.setattr(target.model, 'forward', lambda **kwargs: pytest.fail('invalid token reached the model'))
    with pytest.raises(ValueError, match='outside the model vocabulary'):
        verify_tree_greedy(target, state, 10, tree)
    assert not state._failed and state.seq_len == 3


@pytest.mark.parametrize('parents', [[-1, False, -1], [-1, 3, -1], [-1, 0], [-1, 0, 1, 2]])
@torch.inference_mode()
def test_invalid_dead_parents_still_fail_before_model_execution(models, monkeypatch, parents):
    target, _ = models
    target.eos_token_ids = frozenset([2])
    state, _ = target.prefill([5, 7, 9])
    tree = DraftTree([2, 5, 11], [-1, 0, -1])
    tree.parent = parents
    monkeypatch.setattr(target.model, 'forward', lambda **kwargs: pytest.fail('invalid parent reached the model'))
    with pytest.raises(ValueError, match='equal length|tree parents'):
        verify_tree_greedy(target, state, 10, tree)
    assert not state._failed and state.seq_len == 3


@torch.inference_mode()
def test_original_logical_depth_is_still_admitted(models, monkeypatch):
    target, _ = models
    target.eos_token_ids = frozenset([2])
    target.model.config.max_position_embeddings = 8
    state, _ = target.prefill([5, 7, 9])
    tree = DraftTree.from_branches([[2, 5, 6, 7, 8, 9], [11]])
    monkeypatch.setattr(target.model, 'forward', lambda **kwargs: pytest.fail('long tree reached the model'))
    with pytest.raises(ValueError, match='context limit'):
        verify_tree_greedy(target, state, 10, tree)
    assert not state._failed and state.seq_len == 3


@pytest.mark.parametrize('vocabulary', [None, True, '67'])
def test_unknown_vocabulary_keeps_original_forward(vocabulary):
    target = TreeTarget([11], False)
    target.model.config.vocab_size = vocabulary
    tree = DraftTree.from_branches([[11, 12], [50]])
    verify_tree_greedy(target, SimpleNamespace(seq_len=3), 10, tree)
    assert target.trees == [(tree.tokens, tree.parent)]


@pytest.fixture(scope='module', params=[1, 4])
def prediction_threads(request):
    previous = torch.get_num_threads()
    torch.set_num_threads(request.param)
    try:
        yield request.param
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('vocab', [512, 32000])
@pytest.mark.parametrize('pending', [[10], [7, 9, 10]])
@pytest.mark.parametrize('branches,eos,committed,accepted,path', [
    ([[40, 41], [50, 51]], [2], [11], 0, []),
    ([[50, 51], [11, 12, 13]], [2], [11, 12, 13, 14], 3, [2, 3, 4]),
    ([[50, 51], [11, 40]], [2], [11, 12], 1, [2]),
    ([[50, 51], [11, 12, 13]], [12], [11, 12], 2, [2]),
    ([[50, 51], [11, 12]], [11, 50], [11], 1, []),
    ([[11, 12], [50, 51]], [13], [11, 12, 13], 2, [0, 1]),
])
def test_cpu_tree_prediction_rows(monkeypatch, prediction_threads, vocab, pending, branches, eos, committed, accepted, path):
    target = TreeTarget(eos, False)
    target.model.config.vocab_size = vocab
    state = SimpleNamespace(seq_len=3)
    tree = DraftTree.from_branches(branches)
    visited = []
    physical = []

    def forward(state, pending, tokens, parents):
        physical.extend(pending + tokens)
        state.seq_len += len(physical)
        logits = torch.zeros((len(physical), 2 * vocab))[:, ::2]
        logits[:, -1] = 1  # The smaller token wins the tie.
        logits[torch.arange(len(physical)), [token + 1 for token in physical]] = 1
        return logits

    argmax = torch.argmax

    def record(rows, *args, **kwargs):
        start = rows.storage_offset() // (2 * vocab)
        count = rows.numel() // vocab
        visited.extend(range(start, start + count))
        return argmax(rows, *args, **kwargs)

    monkeypatch.setattr(target, 'forward_tree', forward)
    monkeypatch.setattr(target, 'forward_block', lambda state, ids: forward(state, [], ids, []))
    monkeypatch.setattr(torch, 'argmax', record)
    result = verify_tree_greedy(target, state, pending, tree, compute_entropy=False)

    expected_rows = list(range(len(pending))) + [len(pending) + offset for offset in path]
    path_only = vocab == 32000 and prediction_threads == 1
    assert visited == (expected_rows if path_only else list(range(len(physical))))
    assert result.committed == committed and result.n_draft_accepted == accepted
    assert result.n_draft_proposed == len(tree) and not result.all_accepted
    assert result.hidden_index == expected_rows[-1]
    assert result.finished == (committed[-1] in eos)
    assert result.pending_divergence == sum(b != a + 1 for a, b in zip(pending, pending[1:]))
    assert state.seq_len == 3 + len(pending) + len(committed) - 1


@pytest.mark.parametrize('peak', [2., float('inf'), float('nan')])
@pytest.mark.parametrize('pending', [[4], [4, 5]])
def test_wide_tree_winners_follow_the_accepted_path(monkeypatch, peak, pending):
    vocab = 32_003
    tree = DraftTree.from_branches([[19], [7, 9]])
    target = TreeTarget([], False)
    target.model.config.vocab_size = vocab
    state = SimpleNamespace(seq_len=3)
    reached = []
    maximum = torch.max

    def forward(state, prefix, tokens, parents):
        size = len(prefix)
        state.seq_len += size + len(tokens)
        scores = torch.zeros((size + len(tokens), vocab))
        scores[0, 5] = 1
        for row, token in ((size - 1, 7), (size + 1, 9), (size + 2, 21)):
            scores[row, [token, vocab - 1]] = peak  # The first tied token wins.
        return scores

    def record(rows, dim):
        reached.append((rows.storage_offset() // vocab, rows.numel() // vocab))
        return maximum(rows, dim=dim)

    monkeypatch.setattr(target, 'forward_tree', forward)
    monkeypatch.setattr(torch, 'get_num_threads', lambda: 1)
    monkeypatch.setattr(torch, 'max', record)
    result = verify_tree_greedy(target, state, pending, tree, compute_entropy=False)
    size = len(pending)
    assert result.committed == [7, 9, 21] and result.n_draft_accepted == 2
    assert result.pending_divergence == 0 and result.hidden_index == size + 2
    assert state.seq_len == 3 + size + 2 and target.gathers == [[1, 2]]
    assert reached == [(0, size), (size + 1, 1), (size + 2, 1)]


@pytest.mark.parametrize('layout', ['accepted', 'correction', 'bonus', 'empty'])
@pytest.mark.parametrize('compact,entropy', [(False, True), (True, False), (True, True)])
@torch.inference_mode()
def test_pruned_static_block_matches_native(models, monkeypatch, layout, compact, entropy):
    base, reference = models
    prompt, pending = [5, 7, 9], [4, 6]
    oracle = []
    for _ in range(3):
        logits = _reference_forward(reference, prompt + pending + oracle).logits
        oracle.append(int(logits[0, -1].argmax()))
    first, second, _ = oracle
    assert first != second
    dead, miss = [token for token in range(67) if token not in oracle][:2]
    eos = {dead, second}
    branches, expected, accepted = [[dead, 7], [first, miss]], oracle[:2], 1
    if layout == 'accepted':
        branches.append([first, second, 8])
        accepted = 2
    elif layout == 'bonus':
        eos = {dead}
        branches, expected, accepted = [[dead, 7], [first, second]], oracle, 2
    elif layout == 'empty':
        eos = {dead, first}
        branches, expected, accepted = [[dead, 7], [first, 8]], oracle[:1], 1
    tree = DraftTree.from_branches(branches)
    with closing(CudaGraphTarget(base, capacity=32, max_block_tokens=4, replay=False,
                                 capture_predictions=compact)) as target:
        target.eos_token_ids = frozenset(eos)
        state, _ = target.prefill(prompt)
        calls = []
        forward, predict = target.forward_block, target.forward_predictions

        def block(state, tokens):
            calls.append(('block', list(tokens)))
            return forward(state, tokens)

        def predictions(state, tokens):
            calls.append(('predictions', list(tokens)))
            return predict(state, tokens)

        monkeypatch.setattr(target, 'forward_block', block)
        monkeypatch.setattr(target, 'forward_predictions', predictions)
        result = verify_tree_greedy(target, state, pending, tree, compute_entropy=entropy)
        live = [] if layout == 'empty' else branches[1]
        assert calls == [('predictions' if compact and not entropy else 'block', pending + live)]
        assert result.committed == expected and result.finished == (expected[-1] in eos)
        assert result.n_draft_proposed == len(tree) and result.n_draft_accepted == accepted
        assert not result.all_accepted and result.hidden_index == len(pending) + len(expected) - 2
        retained = prompt + pending + expected[:-1]
        native = _reference_forward(reference, retained)
        assert not _cache_failures(state, native.past_key_values, len(retained), atol=2e-6, rtol=2e-5)
        if entropy and layout in ('correction', 'bonus'):
            prob = native.logits[0, -1].float().softmax(-1)
            assert result.entropy == pytest.approx(float(-(prob * prob.log()).sum()), abs=2e-6)
        else:
            assert result.entropy == 0
        next_logits = forward(state, expected[-1:])[-1]
        next_ref = _reference_forward(reference, retained + expected[-1:])
        torch.testing.assert_close(next_logits, next_ref.logits[0, -1], atol=2e-6, rtol=2e-5)
        assert not _cache_failures(state, next_ref.past_key_values, len(retained) + 1, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('constraint', ['narrow', 'tree_only', 'context'])
def test_pruned_block_limit_preserves_tree_admission(monkeypatch, constraint):
    target = TreeTarget([50], False)
    state = SimpleNamespace(seq_len=3)
    tree = DraftTree.from_branches([[50, 53, 54], [11, 12]])
    target.model.config.max_position_embeddings = 64
    if constraint == 'narrow':
        target.max_block_tokens = 2
    elif constraint == 'tree_only':
        monkeypatch.setattr(target, 'forward_block', None)
    else:
        target.context_limit = 6

    def packed(state, pending, tokens, parents):
        target.trees.append((list(tokens), list(parents)))
        if constraint == 'context' and 50 in tokens:
            raise ValueError('exceeds target context')
        return CountingTarget.forward_block(target, state, pending + tokens)

    monkeypatch.setattr(target, 'forward_tree', packed)
    if constraint == 'context':
        with pytest.raises(ValueError, match='exceeds target context'):
            verify_tree_greedy(target, state, 10, tree)
        assert target.trees == [(tree.tokens, tree.parent)] and state.seq_len == 3
    else:
        result = verify_tree_greedy(target, state, 10, tree)
        assert result.committed == [11, 12, 13] and target.trees == [([11, 12], [-1, 0])]


@pytest.mark.parametrize("verifier", ["reference", "block", "tree", "sampling"])
@pytest.mark.parametrize("tensor_ids", [False, True])
@pytest.mark.parametrize("terminal", [False, True])
def test_accepted_scalar_ids_match_integer_proposals(models, verifier, tensor_ids, terminal):
    class UnhashableToken(int):
        __hash__ = None

    target, reference = models
    prefix, pending = [5, 7, 9], [4, 6]
    tokens = []
    for _ in range(4):
        logits = _reference_forward(reference, prefix + pending + tokens).logits[0, -1]
        tokens.append(int(logits.argmax()))
    assert tokens[2] not in tokens[:2]
    target.eos_token_ids = frozenset([tokens[2]]) if terminal else frozenset()

    def run(drafts):
        state, _ = target.prefill(prefix)
        if verifier == "tree":
            wrong = (tokens[0] + 1) % target.model.config.vocab_size
            tree = DraftTree([wrong, *drafts], [-1, -1, 1, 2])
            result = verify_tree_greedy(target, state, pending, tree)
        elif verifier == "sampling":
            result = verify_block_sampling(target, state, pending, drafts, temperature=1e-5,
                                           generator=torch.Generator().manual_seed(19))
        else:
            verify = verify_block_greedy if verifier == "block" else verify_greedy_reference
            result = verify(target, state, pending, drafts)
        return state, result

    _, expected = run(tokens[:3])
    wrap = torch.tensor if tensor_ids else UnhashableToken
    state, actual = run([wrap(token) for token in tokens[:3]])
    assert actual == expected and actual.finished == terminal
    assert actual.committed == (tokens[:3] if terminal else tokens)
    assert all(type(token) is int for token in actual.committed)
    live = prefix + pending + actual.committed[:-1]
    native = _reference_forward(reference, live)
    assert not _cache_failures(state, native.past_key_values, len(live), atol=2e-6, rtol=2e-5)
