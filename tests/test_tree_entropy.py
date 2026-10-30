from functools import partial

import pytest
import torch

from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from dart.verification import state_checks
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.state_checks import RoundCase, check_verifier_state
from dart.verification.synthetic import _ProbeDrafter, _synthetic_models
from dart.verification.tree_verify import DraftTree, verify_tree_greedy


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


def setup_case(models, rejection, reverse):
    target, reference = models
    prefix = (0, 5, 0, 6)  # Padding IDs are real prompt tokens here.
    tokens = hf_decoder(reference)(GenerationCase('oracle', prefix, 6))
    pending, oracle = tokens[0], tokens[1:]
    drafts = oracle[:4]
    if rejection < 4:
        drafts[rejection] = (drafts[rejection] + 1) % 67
    branches = [[(oracle[0] + 1) % 67, 0], drafts]
    tree = DraftTree.from_branches(branches[::-1] if reverse else branches)
    case = RoundCase('entropy', prefix, pending, tuple(tree.tokens), tuple(tree.parent))
    return case, oracle


def check_without_entropy(models, case, monkeypatch):
    expected = check_verifier_state(*models, case, verifier='tree', inspect_tree=True)
    assert expected.passed, expected.failures
    monkeypatch.setattr(state_checks, 'verify_tree_greedy', partial(verify_tree_greedy, compute_entropy=False))
    monkeypatch.setattr(torch, 'softmax', lambda *a, **k: pytest.fail('computed unused entropy'))
    actual = check_verifier_state(*models, case, verifier='tree', inspect_tree=True)
    assert actual.passed, actual.failures
    assert actual.actual == expected.actual and actual.cache_length == expected.cache_length
    assert actual.tree_rows_checked == len(case.drafts) + 1
    return actual


@pytest.mark.parametrize('rejection', range(5))
@pytest.mark.parametrize('reverse', [False, True])
def test_all_rejection_positions_preserve_hf_outputs_and_live_cache(models, monkeypatch, rejection, reverse):
    case, oracle = setup_case(models, rejection, reverse)
    report = check_without_entropy(models, case, monkeypatch)
    assert report.actual == tuple(oracle[:rejection + 1])


@pytest.mark.parametrize('offset', range(3))
@pytest.mark.parametrize('accepted', [False, True])
def test_accepted_and_corrected_eos_preserve_cache_and_followup(models, monkeypatch, offset, accepted):
    case, oracle = setup_case(models, 4 if accepted else offset, True)
    eos = oracle[offset]
    assert eos not in [case.pending, *oracle[:offset]]
    target, reference = models
    target.eos_token_ids = frozenset({eos})
    reference.config.eos_token_id = reference.generation_config.eos_token_id = eos
    report = check_without_entropy(models, case, monkeypatch)
    assert report.actual == tuple(oracle[:offset + 1])


@pytest.mark.parametrize('prompt', [(0, 5, 0, 6), (5, 0, 6, 0)])
def test_engine_skips_entropy_for_nonempty_trees_and_matches_hf(models, monkeypatch, prompt):
    target, reference = models
    expected = hf_decoder(reference)(GenerationCase('tree-output', prompt, 9))
    calls, forward = [], target.forward_tree

    def record(state, pending, tokens, parents):
        calls.append(len(tokens))
        return forward(state, pending, tokens, parents)

    monkeypatch.setattr(target, 'forward_tree', record)
    monkeypatch.setattr(torch, 'softmax', lambda *a, **k: pytest.fail('computed unused entropy'))
    with DartEngine(target, _ProbeDrafter(), verifier='tree') as engine:
        output = engine.generate(list(prompt), DecodeConfig(max_new_tokens=9, draft_len=4))
    assert output.token_ids == expected
    assert calls and all(count > 0 for count in calls)
