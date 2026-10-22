from __future__ import annotations

import json
import sys

import pytest
import torch

from benchmarks import run_bench
from dart.models.target import TargetModel
from tests.test_sampling_devices import CUDA_REQUIRED


@pytest.mark.parametrize("device,dtype", [
    ("cpu", "float32"), ("cpu", "bfloat16"),
    pytest.param("cuda", "float32", marks=CUDA_REQUIRED),
    pytest.param("cuda", "bfloat16", marks=CUDA_REQUIRED),
])
def test_tiny_benchmark_uses_requested_device_and_dtype(monkeypatch, tmp_path, device, dtype):
    built = []
    original = run_bench.build_tiny_target

    def build(**kwargs):
        target = original(**kwargs)
        built.append(target)
        return target

    monkeypatch.setattr(run_bench, "build_tiny_target", build)
    monkeypatch.setattr(sys, "argv", [
        "run_bench", "--device", device, "--dtype", dtype,
        "--lanes", "target_only", "--prompts", "1", "--max-new", "3",
        "--out", str(tmp_path), "--tag", "settings",
    ])
    run_bench.main()
    parameter = next(built[0].model.parameters())
    assert parameter.device.type == device
    assert parameter.dtype == getattr(torch, dtype)
    assert built[0].record_hidden is False
    report = json.loads((tmp_path / "settings.json").read_text())
    assert report["run"]["dtype"] == str(parameter.dtype)
    assert report["run"]["attention"] == "eager"
    assert report["run"]["threads"] == 1
    assert report["lanes"]["target_only"]["exact_match_rate"] == 1.0


def test_benchmark_attention_setting_reaches_the_model(monkeypatch, tmp_path):
    seen = []
    original = run_bench.build_tiny_target

    def build(**kwargs):
        target = original(**kwargs)
        seen.append(target.model.config._attn_implementation)
        return target

    monkeypatch.setattr(run_bench, "build_tiny_target", build)
    monkeypatch.setattr(sys, "argv", [
        "run_bench", "--attention", "sdpa", "--lanes", "target_only",
        "--prompts", "1", "--max-new", "3", "--out", str(tmp_path),
    ])
    run_bench.main()
    assert seen == ["sdpa"]


@pytest.mark.parametrize("lane", ["target_only", "mtp_untrained"])
def test_hf_benchmark_records_hidden_states_only_for_mtp(monkeypatch, tmp_path, lane):
    loaded = []

    def load(name, **kwargs):
        loaded.append(kwargs)
        return run_bench.build_tiny_target(seed=1234, spec={"record_hidden": kwargs.get("record_hidden", False)})

    monkeypatch.setattr(TargetModel, "from_pretrained", load)
    monkeypatch.setattr(sys, "argv", [
        "run_bench", "--target", "hf:local-test", "--lanes", lane,
        "--prompts", "1", "--max-new", "3", "--out", str(tmp_path),
    ])
    run_bench.main()
    assert loaded[0]["record_hidden"] is (lane == "mtp_untrained")
    assert loaded[0]["dtype"] == torch.float32
    assert loaded[0]["attn_implementation"] == "eager"


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA_REQUIRED)])
def test_tiny_ar_drafter_matches_target_device_and_dtype(device):
    target = run_bench.build_tiny_target(seed=1234)
    target = TargetModel(target.model, device=device, dtype=torch.bfloat16)
    rows = run_bench.build_rows(target, selected={"ar_draft"})
    parameter = next(rows["ar_draft"].dm.model.parameters())
    assert parameter.device.type == target.device
    assert parameter.dtype == target.dtype
