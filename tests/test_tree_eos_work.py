from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dart.verification.greedy import verify_greedy_reference
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
    assert target.trees == [(live, live_parents)]
    assert target.calls == [pending + live]
    assert len(target.gathers) == 1
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
    calls, forward = [], target.forward_tree

    def record(state, pending, tokens, parents):
        calls.append((list(tokens), list(parents)))
        return forward(state, pending, tokens, parents)

    monkeypatch.setattr(target, 'forward_tree', record)
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
