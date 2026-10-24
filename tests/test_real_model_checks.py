from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from tests import test_real_model_exactness as checks


def configured_target():
    try:
        return checks.real_target.__wrapped__()
    except pytest.skip.Exception:
        pytest.fail("configured model checks must not skip loading errors")


def target_outputs(actual, expected):
    return SimpleNamespace(
        tokenizer=SimpleNamespace(encode=lambda text: [7, 8, 9]),
        greedy_generate=lambda ids, **kwargs: list(actual),
        model=SimpleNamespace(
            config=SimpleNamespace(eos_token_id=0),
            generate=lambda ids, **kwargs: torch.cat(
                (ids, torch.tensor([expected], dtype=ids.dtype)), dim=1,
            ),
        ),
    )


@pytest.mark.parametrize(
    "actual,expected",
    [([11], [11, 22]), ([11, 22], [11]), ([11, 22], [11, 22, 0]),
     ([11, 22, 0, 0], [11, 22, 0]), ([], [0]), ([11, 99], [11, 22])],
    ids=["truncated", "extra", "missing_eos", "after_eos", "empty", "wrong_token"],
)
def test_real_model_check_rejects_output_mismatches(actual, expected):
    with pytest.raises(AssertionError):
        checks.test_dart_baseline_matches_hf_generate_on_real_model(
            target_outputs(actual, expected),
        )


@pytest.mark.parametrize("tokens", [[11, 22], [11, 22, 0], [0]])
def test_real_model_check_accepts_complete_matches(tokens):
    checks.test_dart_baseline_matches_hf_generate_on_real_model(
        target_outputs(tokens, tokens),
    )


def test_real_model_fixture_skips_only_when_unconfigured(monkeypatch):
    monkeypatch.delenv("DART_TEST_MODEL", raising=False)

    def unexpected_load(*args, **kwargs):
        raise AssertionError("unconfigured checks must not fetch a model")

    monkeypatch.setattr(checks.TargetModel, "from_pretrained", unexpected_load)
    with pytest.raises(pytest.skip.Exception, match="DART_TEST_MODEL"):
        checks.real_target.__wrapped__()


def test_real_model_fixture_uses_only_local_files(monkeypatch, tmp_path):
    monkeypatch.setenv("DART_TEST_MODEL", str(tmp_path))
    loaded = object()

    def load(path, **kwargs):
        assert path == str(tmp_path)
        assert kwargs["local_files_only"] is True
        assert kwargs["trust_remote_code"] is False
        return loaded

    monkeypatch.setattr(checks.TargetModel, "from_pretrained", load)
    assert configured_target() is loaded


@pytest.mark.parametrize("error", [ValueError, RuntimeError, OSError])
def test_configured_model_errors_are_not_skipped(monkeypatch, tmp_path, error):
    monkeypatch.setenv("DART_TEST_MODEL", str(tmp_path))

    def broken_load(*args, **kwargs):
        raise error("model load failed")

    monkeypatch.setattr(checks.TargetModel, "from_pretrained", broken_load)
    with pytest.raises(error, match="model load failed"):
        configured_target()
