from __future__ import annotations

from importlib.metadata import version

import pytest
import torch
from packaging.version import Version

from dart.verification import state_sequences, synthetic


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def record(monkeypatch):
    rounds = []
    for name in ("verify_block_greedy", "verify_tree_greedy"):
        original = getattr(state_sequences, name)

        def verify(*args, _original=original, _name=name, **kwargs):
            result = _original(*args, **kwargs)
            rounds.append((_name, result.n_draft_proposed, result.n_draft_accepted))
            return result

        monkeypatch.setattr(state_sequences, name, verify)
    return rounds


@pytest.mark.parametrize("seed", [17, 29, 101])
def test_state_check_accepts_drafts_in_block_and_tree_operations(seed, monkeypatch):
    rounds = record(monkeypatch)
    report = synthetic.run_synthetic_state_check(seed)
    assert report.passed, report
    blocks = [(proposed, accepted) for name, proposed, accepted in rounds if name == "verify_block_greedy"]
    trees = [accepted for name, _, accepted in rounds if name == "verify_tree_greedy"]
    assert any(0 < accepted < proposed for proposed, accepted in blocks)
    assert any(accepted == proposed > 0 for proposed, accepted in blocks)
    assert len(trees) == 2 and all(accepted >= 1 for accepted in trees)


@pytest.mark.skipif(Version(version("transformers")) < Version("5.16.1"),
                    reason="static execution requires transformers >= 5.16.1")
def test_static_state_check_accepts_block_drafts(monkeypatch):
    rounds = record(monkeypatch)
    report = synthetic.run_synthetic_state_check(17, execution="static")
    assert report.passed, report
    assert any(accepted == proposed > 0 for _, proposed, accepted in rounds)


def test_operation_template_keeps_its_shape():
    operations = synthetic.synthetic_state_operations(17)
    tokens = [operation.tokens for operation in operations if operation.action not in {"block", "tree"}]
    assert len(operations) == 26
    assert tokens == [operation.tokens for operation in synthetic.synthetic_state_operations(17)
                      if operation.action not in {"block", "tree"}]
    assert all(len(operation.tokens) in (3, 4, 5) for operation in operations if operation.action in {"block", "tree"})
