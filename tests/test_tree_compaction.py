from __future__ import annotations

import pytest
import torch

from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_checks import (
    _cache_failures,
    _cache_pairs,
    _reference_forward,
)
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import DraftTree


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        target, reference = _synthetic_models(17)
        target.eos_token_ids = frozenset()
        reference.config.eos_token_id = reference.generation_config.eos_token_id = None
        yield target, reference
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize('path', [[], [0], [0, 1], [0, 1, 2], [3], [3, 4]])
def test_tree_compaction_keeps_hf_cache_and_avoids_contiguous_prefix_copies(models, monkeypatch, path):
    target, reference = models
    prefix, pending = [0, 5, 0, 6], [7]
    tree = DraftTree.from_branches([[11, 13, 17], [19, 23]])
    state, _ = target.prefill(prefix)
    idle, _ = target.prefill([29, 31, 37])
    idle_before = [(k.clone(), v.clone()) for k, v in _cache_pairs(idle.cache)]
    target.forward_tree(state, pending, tree.tokens, tree.parent)
    before = [tensor for pair in _cache_pairs(state.cache) for tensor in pair]
    calls, select = [], torch.Tensor.index_select

    def record(tensor, *args, **kwargs):
        calls.append(tensor.shape)
        return select(tensor, *args, **kwargs)

    with monkeypatch.context() as spy:
        spy.setattr(torch.Tensor, 'index_select', record)
        target.gather_tree_tail(state, len(prefix) + 1, path)
    tokens = prefix + pending + [tree.tokens[index] for index in path]
    expected = _reference_forward(reference, tokens)
    assert not _cache_failures(state, expected.past_key_values, len(tokens), atol=1e-5, rtol=1e-4)
    after = [tensor for pair in _cache_pairs(state.cache) for tensor in pair]
    if path == list(range(len(path))):
        assert calls == []
        assert all(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() for a, b in zip(before, after))
    next_logits = target.forward_block(state, [41])[-1]
    next_reference = _reference_forward(reference, tokens + [41])
    torch.testing.assert_close(next_logits, next_reference.logits[0, -1], atol=1e-5, rtol=1e-4)
    assert next_logits.argmax().item() == next_reference.logits[0, -1].argmax().item()
    assert not _cache_failures(state, next_reference.past_key_values, len(tokens) + 1, atol=1e-5, rtol=1e-4)
    for tensor in [tensor for pair in _cache_pairs(state.cache) for tensor in pair]:
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    for pair, wanted in zip(_cache_pairs(idle.cache), idle_before):
        for got, old in zip(pair, wanted):
            torch.testing.assert_close(got, old, atol=0, rtol=0)


@pytest.mark.parametrize('keep_len', [0, 4])
def test_empty_path_can_keep_none_or_all_without_copying(models, monkeypatch, keep_len):
    target, reference = models
    tokens = [0, 5, 6, 7]
    state, _ = target.prefill(tokens)
    before = [tensor for pair in _cache_pairs(state.cache) for tensor in pair]
    with monkeypatch.context() as spy:
        spy.setattr(torch.Tensor, 'index_select', lambda *a, **k: pytest.fail('copied a contiguous prefix'))
        target.gather_tree_tail(state, keep_len, [])
    assert state.seq_len == keep_len
    after = [tensor for pair in _cache_pairs(state.cache) for tensor in pair]
    assert all(a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr() for a, b in zip(before, after))
    actual = target.forward_block(state, [11])[-1]
    expected = _reference_forward(reference, tokens[:keep_len] + [11])
    torch.testing.assert_close(actual, expected.logits[0, -1], atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize('mode', ['reject', 'partial', 'accept', 'other_branch'])
def test_complete_tree_outputs_match_hf_after_compaction(models, mode):
    target, reference = models
    decode = hf_decoder(reference)
    expected = decode(GenerationCase('expected', (0, 5, 0, 6), 11))
    paths, gather = [], target.gather_tree_tail

    class Drafter:
        name = 'synthetic'
        requires_entropy = False

        def propose_tree(self, committed, max_tokens):
            correct = decode(GenerationCase('draft', tuple(committed), max_tokens))
            wrong = [(token + 1) % 67 for token in correct]
            if mode == 'reject':
                return DraftTree.chain(wrong)
            if mode == 'partial':
                return DraftTree.chain(correct[:1] + wrong[1:])
            if mode == 'other_branch':
                return DraftTree.from_branches([wrong, correct])
            return DraftTree.chain(correct)

        def observe(self, result):
            pass

    def record(state, keep_len, path):
        paths.append(list(path))
        return gather(state, keep_len, path)

    target.gather_tree_tail = record
    with DartEngine(target, Drafter(), verifier='tree') as engine:
        for _ in range(2):
            assert engine.generate([0, 5, 0, 6], DecodeConfig(max_new_tokens=11, draft_len=4)).token_ids == expected
    assert paths
    if mode == 'other_branch':
        assert any(path and path[0] != 0 for path in paths)
    else:
        assert all(path == list(range(len(path))) for path in paths)
