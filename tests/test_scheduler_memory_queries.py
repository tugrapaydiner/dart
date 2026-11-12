from __future__ import annotations

from importlib.metadata import version
from types import SimpleNamespace

import pytest
import torch
from packaging.version import Version

from dart.drafters.base import NullDrafter
from dart.drafters.ngram import NGramDrafter
from dart.models.replay import CudaGraphTarget
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler, TimeScheduler
from dart.runtime.types import DecodeAction, DecodeConfig, DecodeState, VerifyResult
from dart.verification.differential import GenerationCase, hf_decoder
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def memory_queries(monkeypatch):
    queries, free = [], [4.0]
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def read(device):
        queries.append(str(device))
        return int(free[0] * 1e9), 32_000_000_000

    monkeypatch.setattr(torch.cuda, "mem_get_info", read)
    return queries, free


@pytest.mark.parametrize("policy", [TimeScheduler, CostTableScheduler])
@pytest.mark.parametrize("free_gb", [0.0, 24.0])
@pytest.mark.parametrize("kv_limit", [0.5, float("inf")])
def test_timed_policy_skips_unused_memory_query_without_changing_actions(policy, free_gb, kv_limit, memory_queries):
    queries, free = memory_queries
    free[0] = free_gb
    options = SchedulerConfig(available_lanes=["ngram"], max_gpu_kv_gb=kv_limit, ewma_alpha=1.0)
    scheduler, reference = policy(options), policy(options)
    with DartEngine(SimpleNamespace(device="cuda:2"), lanes={"ngram": NGramDrafter()}, scheduler=scheduler) as engine:
        for index in range(12):
            state = DecodeState(context_len=8 + index, generated_len=index, gpu_free_gb=free_gb,
                                batch_size=1, prompt_kind="code", exact_required=True, entropy_ewma=0.0,
                                acceptance_ewma=dict(reference.acceptance_ewma), kv_transfer_ms=0.0,
                                target_step_ms=0.0, max_draft_tokens=4, max_tree_nodes=15, default_verifier="block")
            expected = reference.choose_action(state)
            drafter, length, verifier = engine._pick_lane(DecodeConfig(prompt_kind="code", max_tree_nodes=15),
                                                         8, index, max_draft_tokens=4)
            assert (drafter.name, length, verifier) == (expected.lane, expected.draft_len, expected.verifier or "block")
            result = VerifyResult([7, 8], 2, 1, False, False)
            for item in (scheduler, reference):
                item.observe(drafter.name, result, round_ms=3.0 if drafter.name == "ngram" else 10.0,
                             kv_gb=1.0 if drafter.name == "ngram" else 0.0)
        assert scheduler.counts == reference.counts and scheduler.kv_gb == reference.kv_gb
    assert queries == []


@pytest.mark.parametrize("parent", [HeuristicScheduler, TimeScheduler, CostTableScheduler])
def test_custom_policy_subclasses_still_receive_fresh_device_memory(parent, memory_queries):
    queries, free = memory_queries
    observed = []

    class Policy(parent):
        def choose_action(self, state, allowed=None):
            observed.append(state.gpu_free_gb)
            return DecodeAction(lane="ngram" if state.gpu_free_gb > 8 else "target_only", draft_len=2,
                                canvas_len=0, kv_bits=4, verify_every=1, denoise_steps=0, exact_mode=True)

    with DartEngine(SimpleNamespace(device="cuda:2"), lanes={"ngram": NGramDrafter()},
                    scheduler=Policy(SchedulerConfig())) as engine:
        choices = []
        for value in (24.0, 2.0, 12.0):
            free[0] = value
            choices.append(engine._pick_lane(DecodeConfig(), 8, 1)[0].name)
    assert observed == [24.0, 2.0, 12.0] and choices == ["ngram", "target_only", "ngram"]
    assert queries == ["cuda:2"] * 3


def test_heuristic_diffusion_canvas_keeps_its_free_memory_gate(memory_queries):
    queries, free = memory_queries
    scheduler = HeuristicScheduler(SchedulerConfig(available_lanes=["diffusion"], canvas_len=8))
    with DartEngine(SimpleNamespace(device="cuda:2"), lanes={"diffusion": NullDrafter()}, scheduler=scheduler) as engine:
        free[0] = 24.0
        assert engine._pick_lane(DecodeConfig(), 8, 1)[1] == 32
        free[0] = 2.0
        assert engine._pick_lane(DecodeConfig(), 8, 1)[1] == 8
    assert queries == ["cuda:2", "cuda:2"]


@pytest.mark.parametrize("required", [False, True])
def test_custom_policy_must_declare_its_own_memory_requirement(required, memory_queries):
    queries, _ = memory_queries

    class Policy(TimeScheduler):
        requires_free_memory = required

    with DartEngine(SimpleNamespace(device="cuda:2"), lanes={}, scheduler=Policy(SchedulerConfig())) as engine:
        engine._pick_lane(DecodeConfig(), 8, 1)
    assert queries == ([] if not required else ["cuda:2"])


@pytest.fixture
def models():
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield _synthetic_models(17)
    finally:
        torch.set_num_threads(threads)


@pytest.mark.parametrize("policy", [TimeScheduler, CostTableScheduler])
@pytest.mark.parametrize("execution", ["eager", "tree", pytest.param("static", marks=pytest.mark.skipif(
    Version(version("transformers")) < Version("5.16.1"), reason="static execution requires transformers >= 5.16.1",
))])
@pytest.mark.parametrize("case_kind", ["ordinary", "accepted", "eos"])
def test_timed_generation_still_matches_independent_hf(models, policy, execution, case_kind, monkeypatch):
    base, reference = models
    base.eos_token_ids = frozenset()
    reference.config.eos_token_id = reference.generation_config.eos_token_id = None
    if case_kind != "ordinary":
        with torch.no_grad():
            base.model.lm_head.weight.zero_()
            reference.lm_head.weight.zero_()
        if case_kind == "eos":
            base.eos_token_ids = frozenset({0})
            reference.config.eos_token_id = reference.generation_config.eos_token_id = 0
    target = CudaGraphTarget(base, capacity=32, max_block_tokens=5, replay=False,
                             capture_predictions=True) if execution == "static" else base
    config = SchedulerConfig(available_lanes=["ngram"], draft_len=4)
    options = {"draft_lengths": (2, 4), "verifiers": ("block", "tree")} if policy is CostTableScheduler and execution == "tree" else {}
    scheduler = policy(config, **options)
    queries = []

    def unexpected(*args):
        queries.append(args)
        return 4.0

    monkeypatch.setattr(DartEngine, "_gpu_free_gb", unexpected)
    try:
        with DartEngine(target, lanes={"ngram": NGramDrafter()}, scheduler=scheduler,
                        verifier="tree" if execution == "tree" else "block") as engine:
            for prompt in ((5, 6, 1, 5, 6, 1), (0, 0, 0, 0)):
                case = GenerationCase("memory-query", prompt, 12)
                expected = hf_decoder(reference)(case)
                for _ in range(2):
                    output = engine.generate(list(prompt), DecodeConfig(max_new_tokens=12))
                    assert output.token_ids == expected
                    assert sum(scheduler.counts.values()) == len(output.rounds)
                    if case_kind == "accepted":
                        assert any(row.n_draft_accepted for row in output.rounds)
                    if case_kind == "eos":
                        assert output.token_ids == [0] and not output.rounds
        assert queries == []
    finally:
        if execution == "static":
            target.close()
