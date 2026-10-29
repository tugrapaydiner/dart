from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

from benchmarks import run_bench
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target


def unused_model(*args, **kwargs):
    pytest.fail("An unselected draft model was constructed")


@pytest.mark.parametrize("lanes", [None, "target_only,ngram"])
def test_baseline_benchmark_does_not_construct_learned_drafters(monkeypatch, tmp_path, lanes):
    target = build_tiny_target()
    target.record_hidden = True
    monkeypatch.setattr(run_bench, "build_tiny_target", lambda **kwargs: target)
    monkeypatch.setattr(run_bench, "DiffusionDrafterModel", unused_model)
    monkeypatch.setattr(run_bench, "MTPHeads", unused_model)
    args = ["run_bench", "--warmup", "1", "--repeats", "2", "--prompts", "1", "--max-new", "3", "--out", str(tmp_path), "--tag", "check"]
    if lanes is not None:
        args += ["--lanes", lanes]
    monkeypatch.setattr(sys, "argv", args)
    run_bench.main()
    report = json.loads((tmp_path / "check.json").read_text())
    assert set(report["lanes"]) == {"target_only", "ngram"}
    assert all(row["exact_match_rate"] == 1.0 for row in report["lanes"].values())


def test_unknown_lane_is_rejected_before_drafter_construction(monkeypatch, tmp_path):
    monkeypatch.setattr(run_bench, "DiffusionDrafterModel", unused_model)
    monkeypatch.setattr(sys, "argv", ["run_bench", "--warmup", "1", "--repeats", "2", "--lanes", "unknown", "--out", str(tmp_path)])
    with pytest.raises(SystemExit, match="unknown lanes"):
        run_bench.main()


def test_large_target_can_select_baselines_without_extra_weights(monkeypatch):
    target = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=32_000, hidden_size=512)),
        record_hidden=True,
    )
    monkeypatch.setattr(run_bench, "DiffusionDrafterModel", unused_model)
    monkeypatch.setattr(run_bench, "MTPHeads", unused_model)
    monkeypatch.setattr(run_bench, "build_tiny_target", unused_model)
    rows = run_bench.build_rows(target, selected={"ngram"})
    assert set(rows) == {"target_only", "ngram"}


def test_selected_diffusion_policies_start_with_identical_weights(monkeypatch):
    target = build_tiny_target()
    monkeypatch.setattr(run_bench.torch, "load", unused_model)
    rows = run_bench.build_rows(target, selected={"diffusion_discard", "diffusion_reconcile"})
    assert set(rows) == {"target_only", "diffusion_discard", "diffusion_reconcile"}
    first = rows["diffusion_discard"].model.state_dict()
    second = rows["diffusion_reconcile"].model.state_dict()
    assert all(torch.equal(first[key], second[key]) for key in first)
    single = run_bench.build_rows(target, selected={"diffusion_reconcile"})
    assert all(torch.equal(first[key], single["diffusion_reconcile"].model.state_dict()[key]) for key in first)


def test_all_available_rows_keep_greedy_outputs_on_the_tiny_target():
    target = run_bench.build_tiny_target(seed=1234, spec={"record_hidden": True})
    rows = run_bench.build_rows(target)
    prompt = [7, 8, 9, 7, 8, 9]
    expected = target.greedy_generate(prompt, max_new_tokens=8)
    assert {"target_only", "ngram", "ar_draft", "mtp_untrained", "diffusion_reconcile"} <= rows.keys()
    for name, drafter in rows.items():
        with DartEngine(target, drafter) as engine:
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=8, draft_len=3, canvas_len=4))
        assert result.token_ids == expected, name


def test_benchmark_keeps_cache_statistics_after_request_cleanup(monkeypatch, tmp_path):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafters = []
    original = run_bench.CompressedKVDrafter

    def build(*args, **kwargs):
        drafter = original(*args, **kwargs)
        drafters.append(drafter)
        return drafter

    monkeypatch.setattr(run_bench, "build_tiny_target", lambda **kwargs: target)
    monkeypatch.setattr(run_bench, "CompressedKVDrafter", build)
    monkeypatch.setattr(sys, "argv", ["run_bench", "--warmup", "1", "--repeats", "2",
                                    "--prompts", "1", "--max-new", "4", "--lanes", "compressed_kv_int4",
                                    "--out", str(tmp_path), "--tag", "cleanup"])
    run_bench.main()
    report = json.loads((tmp_path / "cleanup.json").read_text(encoding="utf-8"))
    stats = report["lanes"]["compressed_kv_int4"]["kv_stats"]
    assert stats["seq_len"] > 0 and stats["logical_compressed_bytes"] > 0
    assert len(drafters) == 1 and drafters[0]._state is None
    assert drafters[0].kv_stats() == {} and drafters[0].last_kv_stats == stats
