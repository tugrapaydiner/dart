from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

from benchmarks.profile_decode import main, trace_decode
from dart.drafters.ngram import NGramDrafter
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target
from tests.test_cli_check import local_model as local_model
from tests.test_corpus_eval import records


def test_trace_records_real_operators_and_preserves_output_and_hooks(tmp_path):
    target = build_tiny_target()
    with torch.no_grad():
        target.model.lm_head.weight.zero_()
    path = tmp_path / "decode.trace.json"
    prompt, config = [7, 8, 7, 8], DecodeConfig(max_new_tokens=5, draft_len=2)
    with DartEngine(target, NGramDrafter()) as engine:
        original = (engine.verify, engine._propose, target.prefill)
        report = trace_decode(engine, prompt, config, path)
        assert (engine.verify, engine._propose, target.prefill) == original
        assert engine.generate(prompt, config).token_ids == [0] * 5
    assert report["generated_tokens"] == 5 and report["rounds"] > 0
    assert {row["output_hash"] for row in report["phases"]["samples"]} == {report["output_hash"]}
    names = {row["name"] for row in report["operations"]}
    assert {"dart.request", "dart.prefill", "dart.draft", "dart.verify", "aten::argmax"} <= names
    assert report["gpu_events"] == 0 and not report["gpu_timeline_available"]
    assert report["trace_bytes"] == path.stat().st_size > 0
    assert any("tolist" in row["name"] for row in report["python_calls"])
    assert all("/" not in row["file"] and "\\" not in row["file"] for row in report["python_calls"])
    assert json.loads(path.read_text(encoding="utf-8"))["traceEvents"]
    assert not target.model._forward_hooks and not target.model._forward_pre_hooks


@pytest.mark.parametrize("config", [
    DecodeConfig(max_new_tokens=0), DecodeConfig(max_new_tokens=33), DecodeConfig(mode="exact_sampling"),
])
def test_invalid_trace_config_fails_before_generation(tmp_path, monkeypatch, config):
    with DartEngine(build_tiny_target()) as engine:
        monkeypatch.setattr(engine, "generate", lambda *a: pytest.fail("unexpected generation"))
        with pytest.raises(ValueError):
            trace_decode(engine, [7], config, tmp_path / "trace.json")


def test_existing_trace_is_not_overwritten(tmp_path, monkeypatch):
    path = tmp_path / "trace.json"
    path.write_text("keep", encoding="utf-8")
    with DartEngine(build_tiny_target()) as engine:
        monkeypatch.setattr(engine, "generate", lambda *a: pytest.fail("unexpected generation"))
        with pytest.raises(FileExistsError):
            trace_decode(engine, [7], DecodeConfig(max_new_tokens=2), path)
    assert path.read_text() == "keep"


def test_mismatched_trace_output_is_rejected_and_wrappers_restored(tmp_path, monkeypatch):
    with DartEngine(build_tiny_target()) as engine:
        original = engine.generate
        calls = 0

        def changed(*args, **kwargs):
            nonlocal calls
            calls += 1
            output = original(*args, **kwargs)
            return replace(output, token_ids=[-1]) if calls == 3 else output

        monkeypatch.setattr(engine, "generate", changed)
        verify = engine.verify
        path = tmp_path / "failed.json"
        with pytest.raises(ValueError, match="mismatch during operator tracing"):
            trace_decode(engine, [7, 8], DecodeConfig(max_new_tokens=2), path)
        assert engine.verify == verify and "_propose" not in vars(engine)
        assert "prefill" not in vars(engine.target)
        assert not engine.target.model._forward_hooks
        assert not path.exists()


@pytest.mark.parametrize("option", [["--max-new", "33"], ["--max-new", "0"]])
def test_command_rejects_invalid_work_before_model_loading(monkeypatch, option):
    monkeypatch.setattr("benchmarks.profile_decode.build_engine", lambda *a: pytest.fail("unexpected load"))
    with pytest.raises(SystemExit) as error:
        main(["--model", "missing", "--out", "report.json", *option])
    assert error.value.code == 2


def test_command_preserves_existing_report_before_loading(tmp_path, monkeypatch):
    output = tmp_path / "report.json"
    output.write_text("keep", encoding="utf-8")
    monkeypatch.setattr("benchmarks.profile_decode.build_engine", lambda *a: pytest.fail("unexpected load"))
    with pytest.raises(SystemExit) as error:
        main(["--model", "missing", "--out", str(output)])
    assert error.value.code == 2 and output.read_text() == "keep"


def test_command_profiles_the_selected_native_norm_backend(local_model, tmp_path, monkeypatch):
    from benchmarks import profile_decode
    from dart.runtime import factory

    directory, built = local_model

    def build(config):
        engine = factory.build_engine(config)
        engine.default_decode_config = DecodeConfig(**config["decode"])
        return engine

    monkeypatch.setattr(profile_decode, "build_engine", build)
    monkeypatch.setattr(profile_decode, "load_frozen", lambda *args: records())
    output = tmp_path / "native-profile.json"
    main(["--model", str(directory), "--native-rms-norm", "--case", "a", "--max-new", "3", "--out", str(output)])
    report = json.loads(output.read_text())
    assert report["target"]["native_rms_norms"] == 5 and len(built) == 1
    assert any("rms_norm" in row["name"] for row in report["operations"])
