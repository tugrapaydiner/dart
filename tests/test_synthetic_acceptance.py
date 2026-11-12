from __future__ import annotations

from collections import defaultdict

import pytest
import torch

from dart.runtime import engine as engine_module
from dart.verification.synthetic import run_synthetic_check


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def record_rounds(monkeypatch, wrap=None):
    rounds = defaultdict(list)
    for name in ("block", "tree"):
        original = engine_module._VERIFIERS[name]

        def verify(*args, _name=name, _original=original, **kwargs):
            result = _original(*args, **kwargs)
            rounds[_name].append((result.n_draft_proposed, result.n_draft_accepted))
            return wrap(result, args[3]) if wrap else result

        monkeypatch.setitem(engine_module._VERIFIERS, name, verify)
    return rounds


@pytest.mark.parametrize("seed", [17, 29, 101])
def test_default_check_accepts_drafts_and_rejects_at_every_offset(seed, monkeypatch):
    rounds = record_rounds(monkeypatch)
    report = run_synthetic_check(seed)
    assert report.passed, report.failures
    block = rounds["block"]
    assert any(proposed > 0 and accepted == proposed for proposed, accepted in block)
    assert {accepted for proposed, accepted in block if proposed == 4} == {0, 1, 2, 3, 4}
    assert any(accepted > 0 for proposed, accepted in rounds["tree"])


def test_accepting_a_rejected_draft_is_detected(monkeypatch):
    def accept_wrong(result, drafts):
        drafts = list(drafts.tokens if hasattr(drafts, "tokens") else drafts)
        if result.n_draft_accepted < len(drafts) and not result.finished:
            result.committed[-1] = drafts[result.n_draft_accepted]  # Keep the rejected draft instead of the correction.
        return result

    record_rounds(monkeypatch, accept_wrong)
    report = run_synthetic_check()
    assert not report.passed
    assert {row.backend for row in report.failures} == {"block", "tree", "adaptive"}
    assert all(row.failure.kind in {"tokens", "eos"} for row in report.failures)
