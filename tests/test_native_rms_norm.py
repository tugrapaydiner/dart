from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from benchmarks.run_bench import build_hf_method
from dart.data.teacher import target_fingerprint
from dart.drafters.ngram import NGramDrafter
from dart.models.rms_norm import TorchRMSNorm, enable_native_rms_norm
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_engine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target
from tests.test_sampling_devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("length", [1, 5, 31])
def test_native_norm_preserves_parameters_and_close_float32_values(device, length):
    torch.manual_seed(303)
    original = LlamaRMSNorm(64, eps=1e-5).to(device).eval()
    original.weight.data.uniform_(0.5, 1.5)
    native = TorchRMSNorm(original)
    values = torch.randn(1, length, 128, device=device)[..., ::2]
    with torch.inference_mode():
        expected, actual = original(values), native(values)
    assert native.weight is original.weight and not native.training
    assert actual.dtype == expected.dtype == torch.float32
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-6)
    assert dict(native.named_parameters()).keys() == {"weight"}


def test_conversion_keeps_weights_and_marks_the_execution_fingerprint():
    model = build_tiny_target().model
    weights = {name: value.clone() for name, value in model.state_dict().items()}
    parameters = {name: id(value) for name, value in model.named_parameters()}
    original = TargetModel(deepcopy(model))
    native = TargetModel(model, native_rms_norm=True)
    assert original.native_rms_norm_count == 0 and native.native_rms_norm_count == 5
    assert parameters == {name: id(value) for name, value in native.model.named_parameters()}
    assert weights.keys() == native.model.state_dict().keys()
    assert all(torch.equal(value, native.model.state_dict()[name]) for name, value in weights.items())
    before, after = target_fingerprint(original), target_fingerprint(native)
    assert "native_rms_norms" not in before
    assert after.pop("native_rms_norms") == 5 and after == before
    enable_native_rms_norm(native.model)
    assert sum(isinstance(module, TorchRMSNorm) for module in native.model.modules()) == 5
    assert TargetModel(native.model).native_rms_norm_count == 5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("early_eos", [False, True])
def test_native_norm_matches_full_hf_outputs_through_lookup_rollback_and_reuse(device, early_eos):
    model = build_tiny_target().model
    original = TargetModel(deepcopy(model), device=device, record_hidden=True)
    native = TargetModel(model, device=device, record_hidden=True, native_rms_norm=True)
    prompts = [[7, 8, 9, 7, 8, 9], [11, 12, 11, 12], [7, 8, 9, 7, 8, 9]]
    if early_eos:
        eos = original.greedy_generate(prompts[0], 1)[0]
        assert eos != prompts[0][-1]
        original.eos_token_ids = native.eos_token_ids = frozenset([eos])
    cfg = DecodeConfig(max_new_tokens=9, draft_len=4)
    for prompt in prompts:
        expected = build_hf_method(original, 9)(prompt).token_ids
        wrong = next(token for token in range(211) if token not in expected)

        class WrongDrafter:
            name = "wrong"

            def __init__(self, token):
                self.token = token

            def propose(self, committed, max_tokens):
                return [self.token] * max_tokens

            def observe(self, result):
                pass

        for drafter in (NGramDrafter(), WrongDrafter(wrong)):
            with DartEngine(native, drafter) as engine:
                output = engine.generate(prompt, cfg)
            assert output.token_ids == expected
            assert output.metadata["finished_eos"] == bool(expected and expected[-1] in original.eos_token_ids)
            if not early_eos and drafter.name == "wrong":
                assert sum(row.n_draft_proposed for row in output.rounds) > 0
                assert all(row.n_draft_accepted == 0 for row in output.rounds)
        assert build_hf_method(native, 9, prompt_lookup_num_tokens=4)(prompt).token_ids == expected
        old_state, old_logits = original.prefill(prompt)
        new_state, new_logits = native.prefill(prompt)
        assert old_state.seq_len == new_state.seq_len == len(prompt)
        torch.testing.assert_close(new_logits, old_logits, rtol=2e-5, atol=2e-6)
        torch.testing.assert_close(native.last_hidden_block, original.last_hidden_block, rtol=2e-5, atol=2e-6)


def test_factory_enables_native_norm_only_when_requested():
    for enabled in (False, True):
        with build_engine({"target": {"model": "tiny:7", "native_rms_norm": enabled}}) as engine:
            assert engine.target.native_rms_norm_count == (5 if enabled else 0)


@pytest.mark.parametrize("option", [1, "true", None])
def test_invalid_native_setting_is_rejected(option):
    with pytest.raises(ValueError, match="boolean"):
        TargetModel(build_tiny_target().model, native_rms_norm=option)


def test_unsupported_precision_and_model_fail_without_partial_conversion():
    model = build_tiny_target().model
    with pytest.raises(ValueError, match="float32"):
        TargetModel(model, dtype=torch.bfloat16, native_rms_norm=True)
    norms = [module for module in model.modules() if isinstance(module, LlamaRMSNorm)]
    norms[-1].to(torch.bfloat16)
    with pytest.raises(ValueError, match="float32"):
        enable_native_rms_norm(model)
    assert not any(isinstance(module, TorchRMSNorm) for module in model.modules())
    with pytest.raises(ValueError, match="LlamaRMSNorm"):
        enable_native_rms_norm(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match="float32"):
        TorchRMSNorm(LlamaRMSNorm(4))(torch.ones(1, 4, dtype=torch.bfloat16))


def test_missing_torch_primitive_fails_before_conversion(monkeypatch):
    model = build_tiny_target().model
    monkeypatch.setattr("dart.models.rms_norm.F.rms_norm", None)
    with pytest.raises(ValueError, match="does not provide"):
        enable_native_rms_norm(model)
    assert not any(isinstance(module, TorchRMSNorm) for module in model.modules())
