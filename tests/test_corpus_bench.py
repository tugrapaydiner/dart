from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from benchmarks import corpus_bench
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target
from tests.test_corpus_eval import Tokenizer, records
from tests.test_sampling_devices import DEVICES


def build_engine(device="cpu"):
    target = TargetModel(build_tiny_target().model, tokenizer=Tokenizer(), device=device, record_hidden=True)
    return DartEngine(target, NGramDrafter(), default_decode_config=DecodeConfig(max_new_tokens=99, draft_len=2))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("early_eos", [False, True])
def test_benchmark_preserves_budgets_eos_and_separate_measurements(device, early_eos):
    with build_engine(device) as engine:
        target = engine.target
        if early_eos:
            prompt = target.tokenizer.encode("aaa", add_special_tokens=True, truncation=False)
            target.eos_token_ids = frozenset(target.greedy_generate(prompt, 1))
        progress = []
        report = corpus_bench.benchmark_corpus(engine, records(), warmup=1, repeats=2, progress=progress.append)
        assert target.record_hidden is True and engine.default_decode_config.max_new_tokens == 99
    assert len(report["timing"]["samples"]) == 20
    assert len(report["profile"]["samples"]) == 20
    assert report["hf_prompt_lookup"] == {"prompt_lookup_num_tokens": 2, "max_matching_ngram_size": 4}
    assert set(report["by_workload"]) == {"instruction", "code"}
    assert set(report["by_budget"]) == {"3", "7"}
    assert len(progress) == 2
    checks = report["correctness"]
    assert checks["summary"]["exact_match_rate"] == 1
    assert checks["summary"]["draft_tokens_proposed"] == 0
    assert checks["summary"]["acceptance_rate"] == 0
    for sample in report["timing"]["samples"] + report["profile"]["samples"]:
        output = checks["outputs"][sample["prompt_index"]]
        assert sample["output_hash"] == output["output_hash"]
        assert sample["tokens"] == output["generated_tokens"] <= output["max_new_tokens"]
    assert report["record_hidden_by_method"] == {
        "hf_greedy": False, "hf_prompt_lookup": False, "target_greedy": False, "target_only": False, "dart": True,
    }
    if early_eos:
        assert checks["outputs"][0]["early_eos"]
    assert (report["summary"]["timing"]["dart"]["peak_gpu_bytes"] > 0) == device.startswith("cuda")


def test_repeated_target_outputs_report_positive_acceptance():
    with build_engine() as engine:
        with torch.no_grad():
            engine.target.model.lm_head.weight.zero_()
        report = corpus_bench.benchmark_corpus(engine, records(), warmup=1, repeats=2)
    counts = report["correctness"]["summary"]
    assert counts["draft_tokens_proposed"] > 0
    assert counts["draft_tokens_accepted"] == counts["draft_tokens_proposed"]
    assert counts["acceptance_rate"] == 1
    assert all(set(row["token_ids"]) == {0} for row in report["correctness"]["outputs"])


def test_fastest_reference_is_selected_separately_for_each_group(monkeypatch):
    costs = {"hf_greedy": [10, 8], "hf_prompt_lookup": [4, 10],
             "target_greedy": [8, 9], "target_only": [9, 4], "dart": [2, 2]}
    original = corpus_bench.measure_methods

    def measure(*args, **kwargs):
        report = original(*args, **kwargs)
        for sample in report["samples"]:
            sample["latency_ms"] = costs[sample["method"]][sample["prompt_index"]]
        report["summary"] = corpus_bench.summarize_samples(report["samples"], "hf_greedy")
        return report

    monkeypatch.setattr(corpus_bench, "measure_methods", measure)
    with build_engine() as engine:
        report = corpus_bench.benchmark_corpus(engine, records(), warmup=1, repeats=2)
    assert report["summary"]["fastest_baseline"] == "target_only"
    assert report["summary"]["vs_fastest"]["dart"]["speedup"] == 3.25
    instruction = report["by_workload"]["instruction"]
    code = report["by_workload"]["code"]
    assert instruction["fastest_baseline"] == "hf_prompt_lookup"
    assert instruction["vs_fastest"]["dart"]["ci95_speedup"] == [2, 2]
    assert code["fastest_baseline"] == "target_only"
    assert code["vs_fastest"]["dart"]["ci95_speedup"] == [2, 2]


def test_lookup_can_be_the_fastest_aggregate_baseline():
    samples = [{"method": method, "prompt_index": 0, "repeat": repeat,
                "latency_ms": cost, "tokens": 4, "peak_gpu_bytes": 0}
               for method, cost in {"hf_greedy": 10, "hf_prompt_lookup": 2,
                                    "target_greedy": 8, "target_only": 7, "dart": 4}.items()
               for repeat in range(2)]
    report = corpus_bench._summarize(samples, 17)
    assert report["fastest_baseline"] == "hf_prompt_lookup"
    assert report["vs_fastest"]["dart"] == {"speedup": 0.5, "ci95_speedup": [0.5, 0.5]}


def test_lookup_budget_can_be_explicit_for_a_no_drafting_ablation():
    with build_engine() as engine:
        config = replace(engine.default_decode_config, draft_len=0)
        report = corpus_bench.benchmark_corpus(
            engine, records(), config, warmup=1, repeats=2, lookup_tokens=3, lookup_max_ngram=2,
        )
    assert report["hf_prompt_lookup"] == {"prompt_lookup_num_tokens": 3, "max_matching_ngram_size": 2}
    assert report["correctness"]["summary"]["draft_tokens_proposed"] == 0


@pytest.mark.parametrize("phase", ["timing", "profiling"])
def test_changed_reference_between_phases_fails_the_benchmark(monkeypatch, phase):
    name = "measure_methods" if phase == "timing" else "profile_methods"
    original = getattr(corpus_bench, name)

    def changed(*args, **kwargs):
        report = original(*args, **kwargs)
        report["samples"][0]["output_hash"] = "changed"
        return report

    monkeypatch.setattr(corpus_bench, name, changed)
    with build_engine() as engine, pytest.raises(ValueError, match="reference output changed"):
        corpus_bench.benchmark_corpus(engine, records(), warmup=1, repeats=2)
    assert engine.target.record_hidden is True


def test_changed_timing_reference_cannot_be_saved(monkeypatch):
    original = corpus_bench.measure_methods

    def changed(*args, **kwargs):
        result = original(*args, **kwargs)
        result["samples"][0]["output_hash"] = "changed"
        return result

    saved = []
    monkeypatch.setattr(corpus_bench, "measure_methods", changed)
    monkeypatch.setattr(corpus_bench, "profile_methods", lambda *a, **k: pytest.fail("profiled invalid timings"))
    with build_engine() as engine, pytest.raises(ValueError, match="reference output changed"):
        corpus_bench.benchmark_corpus(engine, records(), warmup=1, repeats=2, on_timing=saved.append)
    assert not saved


@pytest.mark.parametrize("options", [
    {"warmup": 0}, {"repeats": 1}, {"warmup": True}, {"seed": -1}, {"seed": True},
    {"lookup_tokens": 0}, {"lookup_max_ngram": True}, {"config": DecodeConfig(draft_len=0)},
])
def test_invalid_settings_fail_before_generation(monkeypatch, options):
    monkeypatch.setattr(corpus_bench, "evaluate_corpus", lambda *args: pytest.fail("invalid settings reached generation"))
    with build_engine() as engine, pytest.raises(ValueError):
        corpus_bench.benchmark_corpus(engine, records(), **options)


def test_unsupported_verifier_fails_before_generation(monkeypatch):
    monkeypatch.setattr(corpus_bench, "evaluate_corpus", lambda *args: pytest.fail("invalid verifier reached generation"))
    with build_engine() as engine:
        engine.verifier_name = "tree"
        with pytest.raises(ValueError, match="block verifier"):
            corpus_bench.benchmark_corpus(engine, records())


def test_workload_selection_is_balanced_and_independent_of_input_order():
    first, second = records()
    extra = [replace(row, id="z-" + row.id, source="z-" + row.source, text="z" + row.text) for row in (first, second)]
    assert corpus_bench.select_records([*extra, second, first], 1) == [first, second]
    assert corpus_bench.select_records([second, *extra, first], 2) == [first, second, *extra]
    with pytest.raises(ValueError, match="fewer than"):
        corpus_bench.select_records([first, second, extra[0]], 2)


@pytest.mark.parametrize("count", [0, 33, True, 1.5])
def test_invalid_selection_count_is_rejected(count):
    with pytest.raises(ValueError, match="1 to 32"):
        corpus_bench.select_records(records(), count)


def test_empty_selection_is_rejected():
    with pytest.raises(ValueError, match="empty"):
        corpus_bench.select_records([], 1)
