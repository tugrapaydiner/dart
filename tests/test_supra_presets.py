from __future__ import annotations

import pytest
import torch

from dart.models.target import TargetModel
from dart.presets import supra_config
from dart.runtime import factory
from dart.runtime.factory import build_engine
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
def test_local_presets_reach_both_loaders_and_preserve_complete_outputs(tmp_path, monkeypatch, device):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    config = supra_config(tmp_path, device)
    calls, threads = {}, []
    original = build_tiny_target()

    def load_model(path, **kwargs):
        calls["model"] = (path, kwargs)
        original.model.set_attn_implementation(kwargs["attn_implementation"])
        return original.model

    def load_tokenizer(path, **kwargs):
        calls["tokenizer"] = (path, kwargs)
        return None

    monkeypatch.setattr("dart.models.target.AutoModelForCausalLM.from_pretrained", load_model)
    monkeypatch.setattr("dart.models.target.AutoTokenizer.from_pretrained", load_tokenizer)
    monkeypatch.setattr(factory.torch, "set_num_threads", threads.append)
    with build_engine(config) as engine:
        assert threads == [1]
        assert engine.scheduler is None
        assert engine.drafter.name == "ngram"
        assert engine.target.device == device
        assert engine.target.dtype == torch.float32
        assert engine.target.model.config._attn_implementation == ("sdpa" if device == "cpu" else "eager")
        assert not engine.target.record_hidden
        assert engine.default_decode_config.max_new_tokens == 32
        for prompt in ([7, 8, 9] * 2, [20, 21, 22] * 3):
            expected = engine.target.greedy_generate(prompt, 32)
            assert engine.generate(prompt).token_ids == expected
    for path, options in calls.values():
        assert path == str(tmp_path.resolve())
        assert options["local_files_only"] is True
        assert options["trust_remote_code"] is False


def test_preset_calls_do_not_share_mutable_configuration(tmp_path):
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    first = supra_config(tmp_path)
    first["target"]["precision"] = "bf16"
    first["lanes"]["ngram"]["max_ngram"] = 99
    second = supra_config(tmp_path)
    assert second["target"]["precision"] == "fp32"
    assert second["lanes"]["ngram"]["max_ngram"] == 4


@pytest.mark.parametrize("kind", ["missing", "file", "no_config"])
def test_presets_reject_missing_local_models(tmp_path, kind):
    path = tmp_path / "checkpoint"
    if kind == "file":
        path.write_text("model", encoding="utf-8")
    elif kind == "no_config":
        path.mkdir()
    with pytest.raises(ValueError, match="existing local model directory"):
        supra_config(path)


@pytest.mark.parametrize("device", ["auto", "mps", "cuda:-1", "cuda:bad", "cuda:00", None, False])
def test_presets_reject_unknown_devices(tmp_path, device):
    with pytest.raises(ValueError, match="cpu or cuda"):
        supra_config(tmp_path, device)


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "1"])
def test_invalid_thread_settings_fail_before_model_loading(monkeypatch, threads):
    def unexpected_load(spec):
        pytest.fail("model loading started with invalid settings")
    monkeypatch.setattr(factory, "build_target", unexpected_load)
    with pytest.raises(ValueError, match="threads must be"):
        build_engine({"threads": threads, "target": {"model": "tiny:1"}})


@pytest.mark.parametrize("local_only", [False, True])
def test_local_file_option_reaches_model_and_tokenizer(monkeypatch, local_only):
    calls = []
    model = build_tiny_target().model

    def load_model(*args, **kwargs):
        calls.append(kwargs["local_files_only"])
        return model

    def load_tokenizer(*args, **kwargs):
        calls.append(kwargs["local_files_only"])
        return None

    monkeypatch.setattr("dart.models.target.AutoModelForCausalLM.from_pretrained", load_model)
    monkeypatch.setattr("dart.models.target.AutoTokenizer.from_pretrained", load_tokenizer)
    TargetModel.from_pretrained("fixture", local_files_only=local_only)
    assert calls == [local_only, local_only]


def test_existing_config_does_not_change_process_thread_count(monkeypatch):
    def unexpected_change(count):
        pytest.fail("a config without threads changed the process setting")
    monkeypatch.setattr(factory.torch, "set_num_threads", unexpected_change)
    with build_engine({"target": {"model": "tiny:1"}}):
        pass
