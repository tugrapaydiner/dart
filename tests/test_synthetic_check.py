from __future__ import annotations

import pytest
import torch

from dart.models.target import TargetModel
from dart.runtime import engine as engine_module
from dart.runtime.engine import DartEngine
from dart.verification.differential import GenerationCase, engine_decoder, hf_decoder
from dart.verification.synthetic import (
    CONTEXT_LENGTH,
    run_synthetic_check,
    synthetic_cases,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize("seed", [17, 29, 101])
def test_default_paths_match_separate_hf_and_exercise_drafts(seed, monkeypatch):
    calls = {"block": [], "tree": []}
    block, tree = TargetModel.forward_block, TargetModel.forward_tree

    def block_spy(self, state, tokens):
        calls["block"].append(len(tokens))
        return block(self, state, tokens)

    def tree_spy(self, state, pending, tokens, parents):
        calls["tree"].append(tuple(parents))
        return tree(self, state, pending, tokens, parents)

    monkeypatch.setattr(TargetModel, "forward_block", block_spy)
    monkeypatch.setattr(TargetModel, "forward_tree", tree_spy)
    monkeypatch.setattr(TargetModel, "greedy_generate", lambda *a, **k: pytest.fail("shared reference used"))
    report = run_synthetic_check(seed)
    assert report.passed, report.failures
    assert len(report.results) == 18
    assert {row.backend for row in report.results} == {"greedy", "block", "tree"}
    assert max(calls["block"]) > 1
    assert any(parents.count(-1) > 1 for parents in calls["tree"])
    assert any(len(row.case.prompt) + row.case.max_new_tokens == CONTEXT_LENGTH for row in report.results)


def test_repeatable_without_changing_rng_or_writing_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    rng = torch.random.get_rng_state().clone()
    first, second = run_synthetic_check(), run_synthetic_check()
    assert first == second and first.passed
    assert torch.equal(rng, torch.random.get_rng_state())
    assert list(tmp_path.iterdir()) == []


def test_real_verifier_fault_is_detected(monkeypatch):
    original = engine_module._VERIFIERS["block"]

    def wrong_token(*args, **kwargs):
        result = original(*args, **kwargs)
        result.committed[-1] = (result.committed[-1] + 1) % 67
        return result

    monkeypatch.setitem(engine_module._VERIFIERS, "block", wrong_token)
    report = run_synthetic_check()
    assert not report.passed
    assert {row.backend for row in report.failures} == {"greedy", "block"}
    assert all(row.failure.kind in {"tokens", "eos"} for row in report.failures)
    assert all(row.passed for row in report.results if row.backend == "tree")


def test_candidate_weight_changes_do_not_modify_reference():
    def factory(target):
        with torch.no_grad():
            target.model.lm_head.weight.zero_()
        return {"changed": engine_decoder(DartEngine(target))}

    report = run_synthetic_check(adapter_factory=factory)
    assert not report.passed and report.failures
    assert all(set(row.actual) <= {0} for row in report.results)
    assert any(any(token != 0 for token in row.expected) for row in report.results)


def test_hf_adapter_ignores_stale_generation_options_and_checks_context(tiny_target):
    model = tiny_target.model
    old = model.generation_config
    try:
        model.generation_config = old.__class__(
            do_sample=True, repetition_penalty=10.0, min_new_tokens=30,
            eos_token_id=2, pad_token_id=0,
        )
        decode = hf_decoder(model)
        case = GenerationCase("small", (5, 6, 7), 4)
        poisoned = model.generation_config
        actual = decode(case)
        assert model.generation_config is poisoned
        model.generation_config = old
        assert actual == hf_decoder(model)(case)
        assert decode(GenerationCase("zero", (5,), 0)) == []
        with pytest.raises(ValueError, match="context"):
            decode(GenerationCase("too-long", (5,) * model.config.max_position_embeddings, 1))
    finally:
        model.generation_config = old


def test_hf_adapter_restores_config_after_generation_error(tiny_target, monkeypatch):
    model = tiny_target.model
    original = model.generation_config

    def fail(*args, **kwargs):
        assert model.generation_config is not original
        raise RuntimeError("injected generation failure")

    monkeypatch.setattr(model, "generate", fail)
    with pytest.raises(RuntimeError, match="injected generation failure"):
        hf_decoder(model)(GenerationCase("a", (5,), 1))
    assert model.generation_config is original


@pytest.mark.parametrize("seed", [-1, True, 1.5, 2**63])
def test_invalid_seed_rejected_before_model_creation(seed, monkeypatch):
    monkeypatch.setattr("dart.verification.synthetic.LlamaForCausalLM", lambda *a: pytest.fail("built a model"))
    with pytest.raises(ValueError, match="seed"):
        run_synthetic_check(seed)


def test_seeded_cases_are_bounded_and_differ_between_seeds():
    cases = synthetic_cases(17)
    assert cases == synthetic_cases(17) and cases != synthetic_cases(29)
    assert len(cases) == 6
    assert all(len(case.prompt) + case.max_new_tokens <= CONTEXT_LENGTH for case in cases)
    assert max(case.max_new_tokens for case in cases) == 13
