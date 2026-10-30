from __future__ import annotations

import logging

import pytest
import torch
from transformers import AutoModelForCausalLM, LlamaConfig, LlamaForCausalLM

from dart.models import target as target_module
from dart.models.target import TargetModel


@pytest.fixture
def saved_model(tmp_path):
    config = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, eos_token_id=2)
    LlamaForCausalLM(config).save_pretrained(tmp_path)
    return tmp_path


def test_loading_a_local_model_uses_the_current_dtype_option(saved_model, caplog, monkeypatch):
    monkeypatch.setattr(target_module.AutoTokenizer, "from_pretrained", lambda *a, **k: None)  # The fixture has no tokenizer.
    with caplog.at_level(logging.WARNING, logger="transformers"):
        target = TargetModel.from_pretrained(str(saved_model), device="cpu", dtype=torch.float32, local_files_only=True)
    assert target.dtype == torch.float32 and next(target.model.parameters()).dtype == torch.float32
    assert not [record for record in caplog.records if "torch_dtype" in record.getMessage()]


@pytest.mark.parametrize("installed, option", [("5.16.1", "dtype"), ("5.0.0", "dtype"), ("4.57.0", "torch_dtype")])
def test_dtype_option_follows_the_transformers_version(monkeypatch, installed, option):
    seen = {}

    class Stop(Exception):
        pass

    def load(name, **kwargs):
        seen.update(kwargs)
        raise Stop

    monkeypatch.setattr(target_module, "version", lambda package: installed)
    monkeypatch.setattr(AutoModelForCausalLM, "from_pretrained", load)
    with pytest.raises(Stop):
        TargetModel.from_pretrained("unused", dtype=torch.bfloat16)
    assert seen[option] is torch.bfloat16 and ({"dtype", "torch_dtype"} - {option}).isdisjoint(seen)
