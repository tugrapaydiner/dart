from __future__ import annotations

import json
import sys

import pytest

from benchmarks import run_bench
from dart.models.target import TargetModel
from tests.conftest import build_tiny_target, random_prompt
from tests.test_sampling_devices import DEVICES


def test_command_records_references_and_raw_timings(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", [
        "run_bench", "--prompts", "2", "--max-new", "4", "--warmup", "1",
        "--repeats", "2", "--out", str(tmp_path), "--tag", "timed",
    ])
    run_bench.main()
    report = json.loads((tmp_path / "timed.json").read_text())
    assert set(report["timing"]["summary"]) == {"hf_greedy", "target_greedy", "target_only", "ngram"}
    assert len(report["timing"]["samples"]) == 16
    assert report["timing"]["repeats"] == 2
    assert report["run"]["reference"] == "hf_greedy"
    for name, row in report["lanes"].items():
        assert row["exact_match_rate"] == 1
        assert row["speedup_wall"] == report["timing"]["summary"][name]["speedup"]
        assert len(row["ci95_speedup_vs_target_only"]) == 2
        assert len(row["ci95_speedup_vs_fastest_baseline"]) == 2
    for prompt_index in range(2):
        assert len({s["output_hash"] for s in report["timing"]["samples"] if s["prompt_index"] == prompt_index}) == 1


@pytest.mark.parametrize("setting", ["repetition_penalty", "suppress_tokens", "bad_words_ids"])
def test_hf_reference_uses_plain_greedy_settings_and_matching_eos(setting):
    target = build_tiny_target()
    prompt = random_prompt(370, 8)
    continuation = target.greedy_generate(prompt, max_new_tokens=8)
    target.eos_token_ids = frozenset([continuation[min(2, len(continuation) - 1)]])
    value = {"repetition_penalty": 3.0, "suppress_tokens": [continuation[0]], "bad_words_ids": [[continuation[0]]]}[setting]
    setattr(target.model.generation_config, setting, value)
    expected = target.greedy_generate(prompt, max_new_tokens=8)
    result = run_bench.build_hf_method(target, 8)(prompt)
    assert result.token_ids == expected
    assert result.token_ids[-1] in target.eos_token_ids
    assert getattr(target.model.generation_config, setting) == value


def test_hf_reference_returns_decoded_text():
    target = build_tiny_target()

    class Tokenizer:
        def decode(self, token_ids, skip_special_tokens=False):
            assert skip_special_tokens is False
            return str(token_ids)

    target.tokenizer = Tokenizer()
    result = run_bench.build_hf_method(target, 3)([7, 8, 9])
    assert result.text == str(result.token_ids)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prompt", [[0, 1, 2], [7, 0, 8], [0, 0, 7]])
def test_hf_reference_keeps_prompt_tokens_equal_to_the_padding_id(device, prompt):
    target = TargetModel(build_tiny_target().model, device=device)
    expected = target.greedy_generate(prompt, max_new_tokens=6)
    assert run_bench.build_hf_method(target, 6)(prompt).token_ids == expected
