from contextlib import closing
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from dart.models.replay import CudaGraphTarget
from dart.verification.greedy import verify_greedy_reference
from dart.verification.state_checks import _cache_failures, _reference_forward
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import DraftTree, verify_tree_greedy
from tests.test_block_eos_work import CountingTarget


class TreeTarget(CountingTarget):
    def __init__(self, eos, compact):
        super().__init__(eos, compact)
        self.trees, self.gathers = [], []

    def forward_tree(self, state, pending, tokens, parents):
        self.trees.append((list(tokens), list(parents)))
        return self.forward_block(state, pending + tokens)

    def gather_tree_tail(self, state, keep_len, path):
        self.gathers.append(list(path))
        self.crop(state, keep_len + len(path))


@pytest.mark.parametrize('drafts,eos', [
    ([], [2]), ([11], [2]), ([11, 12, 13, 14], [2]),
    ([50, 12, 13, 14], [2]), ([11, 12, 50, 14], [2]),
    ([11, 12, 13, 14], [11, 66]), ([11, 12, 13, 14], [13, 66]),
    ([50, 12, 13, 14], [13, 66]),
])
@pytest.mark.parametrize('pending', [[10], [7, 9, 10]])
@pytest.mark.parametrize('compact,entropy', [(False, True), (True, False), (True, True)])
def test_linear_tree_uses_one_causal_forward(drafts, eos, pending, compact, entropy):
    target, reference = TreeTarget(eos, compact), CountingTarget(eos, False)
    state, expected_state = SimpleNamespace(seq_len=3), SimpleNamespace(seq_len=3)
    tree = DraftTree.chain(drafts)
    original = deepcopy(tree)
    expected = verify_greedy_reference(reference, expected_state, pending, drafts)
    actual = verify_tree_greedy(target, state, pending, tree, compute_entropy=entropy)
    assert actual == expected and state.seq_len == expected_state.seq_len
    assert actual.hidden_index == len(pending) + len(actual.committed) - 2
    assert tree == original
    stop = next((i for i, token in enumerate(drafts) if token in eos), len(drafts))
    assert target.calls == [pending + drafts[:stop]]
    assert target.trees == target.gathers == []
    if not entropy or actual.finished:
        assert actual.entropy == 0
    else:
        assert actual.entropy > 0


@pytest.mark.parametrize('parents', [[-1, -1], [-1, 0, 0], [-1, 0, -1, 2]])
def test_branches_keep_tree_execution(parents):
    target = TreeTarget([2], False)
    state = SimpleNamespace(seq_len=3)
    tree = DraftTree(list(range(11, 11 + len(parents))), parents)
    verify_tree_greedy(target, state, 10, tree)
    assert target.trees == [(tree.tokens, parents)] and len(target.gathers) == 1


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield _synthetic_models(17)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('rejection', range(5))
@pytest.mark.parametrize('eos_index', [None, 0, 2])
@pytest.mark.parametrize('entropy', [False, True])
@torch.inference_mode()
def test_chain_matches_native_outputs_cache_hidden_and_continuation(models, monkeypatch, rejection, eos_index, entropy):
    target, reference = models
    target.record_hidden = True
    prompt = [5, 7, 9]
    oracle, prefix = [], list(prompt)
    for _ in range(6):
        token = int(_reference_forward(reference, prefix).logits[0, -1].argmax())
        oracle.append(token)
        prefix.append(token)
    target.eos_token_ids = frozenset() if eos_index is None else frozenset([oracle[1 + eos_index]])
    drafts = oracle[1:5]
    if rejection < 4:
        drafts[rejection] = (drafts[rejection] + 1) % 67
    expected = oracle[1:rejection + 2]
    if eos_index is not None:
        expected = expected[:eos_index + 1]
    state, _ = target.prefill(prompt)
    rows, forward = [], target.forward_block

    def record(state, tokens):
        rows.append(list(tokens))
        return forward(state, tokens)

    monkeypatch.setattr(target, 'forward_block', record)
    monkeypatch.setattr(target, 'forward_tree', lambda *args: pytest.fail('linear tree used a tree mask'))
    monkeypatch.setattr(target, 'gather_tree_tail', lambda *args: pytest.fail('linear tree gathered KV'))
    result = verify_tree_greedy(target, state, oracle[0], DraftTree.chain(drafts), compute_entropy=entropy)
    assert result.committed == expected and result.n_draft_proposed == 4
    assert result.finished is (expected[-1] in target.eos_token_ids)
    assert len(rows) == 1
    live = prompt + [oracle[0]] + expected[:-1]
    wanted = _reference_forward(reference, live)
    assert not _cache_failures(state, wanted.past_key_values, len(live), atol=2e-6, rtol=2e-5)
    hidden = reference(
        input_ids=torch.tensor([live]), attention_mask=torch.ones((1, len(live)), dtype=torch.long),
        use_cache=False, output_hidden_states=True,
    ).hidden_states[-1][0, -1]
    torch.testing.assert_close(target.last_hidden_block[result.hidden_index], hidden, atol=2e-6, rtol=2e-5)
    if entropy and not (result.finished and result.n_draft_accepted == len(expected)):
        dist = wanted.logits[0, -1].float().softmax(-1)
        expected_entropy = float(-(dist * dist.clamp_min(1e-12).log()).sum())
        assert result.entropy == pytest.approx(expected_entropy, abs=2e-6)
    else:
        assert result.entropy == 0
    actual_next = target.forward_block(state, [expected[-1]])[-1]
    next_expected = _reference_forward(reference, live + [expected[-1]])
    torch.testing.assert_close(actual_next, next_expected.logits[0, -1], atol=2e-6, rtol=2e-5)
    assert int(actual_next.argmax()) == int(next_expected.logits[0, -1].argmax())
    assert not _cache_failures(state, next_expected.past_key_values, len(live) + 1, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('parents', [[-1], [-1, 0, 1], [-1, False], [-1, -2], [-1, 1]])
@torch.inference_mode()
def test_changed_parents_still_reject_before_forward(models, monkeypatch, parents):
    target, _ = models
    state, _ = target.prefill([5, 7, 9])
    tree = DraftTree.chain([11, 12])
    tree.parent = parents
    monkeypatch.setattr(target.model, 'forward', lambda **kwargs: pytest.fail('invalid tree reached the model'))
    with pytest.raises(ValueError, match='equal length|tree parents'):
        verify_tree_greedy(target, state, 10, tree)
    assert not state._failed and state.seq_len == 3


@pytest.mark.parametrize('bad', [-1, 67, 2**80, 67.9])
@torch.inference_mode()
def test_invalid_ids_after_eos_remain_rejected(models, monkeypatch, bad):
    target, _ = models
    target.eos_token_ids = frozenset([2])
    state, _ = target.prefill([5, 7, 9])
    tree = DraftTree.chain([2, bad])
    monkeypatch.setattr(target.model, 'forward', lambda **kwargs: pytest.fail('invalid ID reached the model'))
    with pytest.raises(ValueError, match='outside the model vocabulary'):
        verify_tree_greedy(target, state, 10, tree)
    assert not state._failed and state.seq_len == 3


@pytest.mark.parametrize('compact,entropy', [(False, True), (True, False), (True, True)])
@torch.inference_mode()
def test_linear_tree_can_use_static_block_execution(models, compact, entropy):
    base, reference = models
    prompt = [5, 7, 9]
    first = int(_reference_forward(reference, prompt).logits[0, -1].argmax())
    next_token = int(_reference_forward(reference, prompt + [first]).logits[0, -1].argmax())
    with closing(CudaGraphTarget(base, capacity=64, max_block_tokens=4, replay=False,
                                  capture_predictions=compact)) as target:
        target.eos_token_ids = frozenset([next_token])
        state, _ = target.prefill(prompt)
        result = verify_tree_greedy(target, state, first, [next_token, 5, 7], compute_entropy=entropy)
        assert result.committed == [next_token] and result.finished
        assert result.n_draft_accepted == 1 and result.n_draft_proposed == 3
        expected = _reference_forward(reference, prompt + [first])
        assert not _cache_failures(state, expected.past_key_values, len(prompt) + 1, atol=2e-6, rtol=2e-5)
