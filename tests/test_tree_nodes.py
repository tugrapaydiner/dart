from __future__ import annotations

import json
from dataclasses import asdict, replace
from functools import wraps

import pytest
import torch

from dart.verification import state_checks
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.reduction import failure_key, reduce_failure
from dart.verification.state_checks import RoundCase, check_verifier_state
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import DraftTree


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    target, reference = _synthetic_models(17)
    target.eos_token_ids = frozenset()
    reference.config.eos_token_id = None
    reference.generation_config.eos_token_id = None
    yield target, reference
    torch.set_num_threads(threads)


def branching_case(reference):
    prefix = (5, 6, 7)
    pending = hf_decoder(reference)(GenerationCase('pending', prefix, 1))[0]
    oracle = hf_decoder(reference)(GenerationCase('drafts', (*prefix, pending), 3))
    wrong = (oracle[0] + 1) % reference.config.vocab_size
    tree = DraftTree.from_branches([oracle, [wrong, 13, 17]])
    return RoundCase('branches', prefix, pending, tuple(tree.tokens), tuple(tree.parent))


@pytest.mark.parametrize('fault', ['logits', 'cache', 'mask'])
def test_inspection_finds_faults_on_a_rejected_branch(models, monkeypatch, fault):
    target, reference = models
    case = branching_case(reference)
    forward = target.forward_tree
    model_forward = target.model.forward
    injected = []

    def wrong_row(state, pending, tokens, parents):
        logits = forward(state, pending, tokens, parents)
        if fault == 'logits':
            logits = logits.clone()
            logits[len(pending) + 3] += 1  # The same argmax hides this from output checks.
        else:
            state.cache.layers[1].values[..., len(case.prefix) + len(pending) + 3, :].add_(1)
        injected.append(True)
        return logits

    @wraps(model_forward)
    def wrong_mask(*args, **kwargs):
        mask = kwargs.get('attention_mask')
        if mask is not None and mask.ndim == 4:
            mask = mask.clone()
            mask[..., 4, len(case.prefix) + 1] = 0  # One rejected root sees an unrelated sibling.
            kwargs['attention_mask'] = mask
            injected.append(True)
        return model_forward(*args, **kwargs)

    with monkeypatch.context() as patch:
        if fault == 'mask':
            patch.setattr(target.model, 'forward', wrong_mask)
        else:
            patch.setattr(target, 'forward_tree', wrong_row)
        ordinary = check_verifier_state(target, reference, case)
        assert injected and ordinary.passed, ordinary.failures
        inspected = check_verifier_state(target, reference, case, inspect_tree=True)
        assert not inspected.passed and inspected.actual == inspected.expected
        kinds = {failure.kind for failure in inspected.failures}
        assert ('tree_cache_values' if fault == 'cache' else 'tree_logits') in kinds
        assert inspected.tree_rows_checked == len(case.drafts) + 1
    assert check_verifier_state(target, reference, case, inspect_tree=True).passed


@pytest.mark.parametrize('order', ['original', 'breadth', 'reverse_siblings'])
@pytest.mark.parametrize('shape', ['chain', 'shared', 'fan', 'empty'])
def test_every_packed_row_matches_an_independent_branch_after_reordering(models, order, shape):
    target, reference = models
    case = branching_case(reference)
    a, b, c = case.drafts[:3]
    branches = {
        'chain': [[a, b, c]],
        'shared': [[a, b, c], [a, (b + 1) % 67, 13], [(a + 1) % 67, 17]],
        'fan': [[a], [(a + 1) % 67], [(a + 2) % 67]],
        'empty': [],
    }[shape]
    tree = DraftTree.from_branches(branches)
    indices = list(range(len(tree)))
    if order != 'original':
        indices.sort(key=lambda index: (tree.depths[index], -index if order == 'reverse_siblings' else index))
    remap = {old: new for new, old in enumerate(indices)}
    case = replace(case, drafts=tuple(tree.tokens[index] for index in indices),
                   parents=tuple(-1 if tree.parent[index] == -1 else remap[tree.parent[index]] for index in indices))
    report = check_verifier_state(target, reference, case, inspect_tree=True)
    assert report.passed, report.failures
    assert report.tree_rows_checked == 1 + len(case.drafts)
    json.dumps(asdict(report))


def test_tree_storage_can_exceed_the_logical_context_limit(models):
    case = RoundCase('wide', (5,) * 60, 7, tuple(range(10, 30)), (-1,) * 20)
    assert len(case.prefix) + 1 + len(case.drafts) > 64
    report = check_verifier_state(*models, case, inspect_tree=True)
    assert report.passed and report.tree_rows_checked == 21, report.failures


@pytest.mark.parametrize('options', [{'inspect_tree': 1}, {'inspect_tree': None},
                                    {'inspect_tree': True, 'verifier': 'block'}])
def test_invalid_inspection_options_are_rejected_before_forward(models, monkeypatch, options):
    case = branching_case(models[1])
    for model in (models[0].model, models[1]):
        monkeypatch.setattr(model, 'forward', lambda *a, **k: pytest.fail('invalid option reached a model'))
    with pytest.raises(ValueError, match='inspect_tree'):
        check_verifier_state(*models, case, **options)


def test_inspection_node_limit_is_checked_before_model_work(models, monkeypatch):
    case = RoundCase('wide', (5, 6), 7, (13,) * 33, (-1,) * 33)
    for model in (models[0].model, models[1]):
        monkeypatch.setattr(model, 'forward', lambda *a, **k: pytest.fail('oversized inspection reached a model'))
    with pytest.raises(ValueError, match='32'):
        check_verifier_state(*models, case, inspect_tree=True)


def test_inspection_reference_error_is_not_recorded_as_a_candidate_failure(models, monkeypatch):
    case = branching_case(models[1])
    reference_forward = state_checks._reference_forward
    calls = 0

    def fail(model, tokens):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError('reference inspection failed')
        return reference_forward(model, tokens)

    monkeypatch.setattr(state_checks, '_reference_forward', fail)
    with pytest.raises(RuntimeError, match='reference inspection failed'):
        check_verifier_state(*models, case, inspect_tree=True)


def test_bad_packed_shape_is_reported(models, monkeypatch):
    target, _ = models
    case = branching_case(models[1])
    forward = target.forward_tree
    monkeypatch.setattr(target, 'forward_tree', lambda *args: forward(*args)[:-1])
    report = check_verifier_state(*models, case, inspect_tree=True)
    assert not report.passed and report.tree_rows_checked == 0
    assert 'tree_logits_shape' in {failure.kind for failure in report.failures}


def test_node_inspection_failure_can_be_reduced_and_replayed(models, monkeypatch):
    target, reference = models
    case = branching_case(reference)
    forward = target.forward_tree

    def shifted(*args):
        return forward(*args) + 1

    with monkeypatch.context() as patch:
        patch.setattr(target, 'forward_tree', shifted)
        result = reduce_failure(case, lambda candidate: failure_key(check_verifier_state(
            target, reference, candidate, inspect_tree=True,
        )), max_evaluations=12)
        assert result.reproduced and result.failure.kind == 'tree_logits'
        assert result.evaluations <= 12 and len(result.reduced.drafts) < len(case.drafts)
        assert not check_verifier_state(target, reference, result.reduced, inspect_tree=True).passed
    assert check_verifier_state(target, reference, result.reduced, inspect_tree=True).passed
