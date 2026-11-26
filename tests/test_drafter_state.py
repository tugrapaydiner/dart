from __future__ import annotations

import pytest
import torch

from dart.drafters.ar_draft import ARDrafter
from dart.drafters.base import NullDrafter
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.kv.cache import QuantizedKVCache, QuantSpec
from dart.runtime.engine import DartEngine
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.time_scheduler import CostTableScheduler
from dart.runtime.types import DecodeAction, DecodeConfig, VerifyResult
from tests.conftest import VOCAB, build_tiny_target, random_prompt
from tests.test_lanes_exactness import _diffusion_drafter


@pytest.mark.parametrize("next_context", ["continue", "restart", "other_prompt", "lane_gap"])
def test_repair_only_reuses_drafts_for_their_original_context(next_context):
    drafter = _diffusion_drafter("reconcile")
    prompt = random_prompt(340, 12)
    drafts = drafter.propose(prompt, 8)
    correction = (drafts[0] + 1) % VOCAB
    drafter.observe(VerifyResult([correction], 8, 0, False, False))
    contexts = {
        "continue": prompt + [correction],
        "restart": prompt,
        "other_prompt": random_prompt(341, 13),
        "lane_gap": prompt + [correction, 10],
    }
    drafter.propose(contexts[next_context], 8)
    assert drafter.stats["warm_rounds"] == int(next_context == "continue")


class AlternatingScheduler(HeuristicScheduler):
    def __init__(self, lane):
        super().__init__(SchedulerConfig())
        self.lane = lane
        self.round = 0

    def reset(self):
        super().reset()
        self.round = 0

    def choose_action(self, state, allowed=None):
        lane = self.lane if self.round % 2 == 0 else "target_only"
        self.round += 1
        return DecodeAction(lane, 3, 3, 16, 1, 4, True)


@pytest.mark.parametrize("lane", ["ar_draft", "diffusion"])
def test_lane_switches_and_repeated_requests_keep_complete_outputs(lane):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    drafter = ARDrafter(target) if lane == "ar_draft" else _diffusion_drafter("reconcile")
    lanes = {lane: drafter, "target_only": NullDrafter()}
    with DartEngine(target, lanes=lanes, scheduler=AlternatingScheduler(lane)) as engine:
        for seed in (342, 343, 342):
            prompt = random_prompt(seed, 8)
            expected = target.greedy_generate(prompt, max_new_tokens=16)
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=16))
            assert result.token_ids == expected
            assert {r.lane for r in result.rounds} == {lane, "target_only"}
    if lane == "diffusion":
        assert drafter.stats["warm_rounds"] == 0


def track_draft_updates(monkeypatch):
    lengths = []
    update = QuantizedKVCache.update

    def record(self, keys, values, layer_idx, *args, **kwargs):
        if layer_idx == 0:
            lengths.append(keys.size(-2))
        return update(self, keys, values, layer_idx, *args, **kwargs)

    monkeypatch.setattr(QuantizedKVCache, "update", record)
    return lengths


@pytest.mark.parametrize("verifier", ["block", "tree"])
def test_verified_draft_prefix_extends_the_prompt_by_one(monkeypatch, verifier):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(205, 8)
    expected = target.greedy_generate(prompt, 8)
    lengths = track_draft_updates(monkeypatch)
    draft = CompressedKVDrafter(target, QuantSpec(window_len=16))
    with DartEngine(target, draft, verifier=verifier) as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=8, draft_len=2))
    assert output.token_ids == expected
    assert lengths[:2] == [len(prompt), 1]
    assert draft._state is None


def test_verified_draft_prefix_uses_the_current_prefix_after_a_lane_switch(monkeypatch):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(207, 8)
    lengths = track_draft_updates(monkeypatch)
    draft = CompressedKVDrafter(target, QuantSpec(window_len=16))
    lanes = {"compressed_kv": draft}
    scheduler = CostTableScheduler(SchedulerConfig(available_lanes=list(lanes)), draft_lengths=(2,), warmup_rounds=1)
    with DartEngine(target, lanes=lanes, scheduler=scheduler) as engine:
        output = engine.generate(prompt, DecodeConfig(max_new_tokens=8))
    assert output.token_ids == target.greedy_generate(prompt, 8)
    assert lengths[:2] == [len(prompt) + 1, 1]


@pytest.mark.parametrize("setting", ["window", "other_model", "sampling", "hidden", "custom_prefill", "hooks"])
def test_verified_draft_prefix_preserves_normal_prefill(monkeypatch, setting):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(209, 8)
    lengths = track_draft_updates(monkeypatch)
    model = build_tiny_target(seed=7) if setting == "other_model" else target
    model.eos_token_ids = frozenset()
    draft = CompressedKVDrafter(model, QuantSpec(window_len=9 if setting == "window" else 32))
    if setting == "hidden":
        target.record_hidden = True
    if setting == "custom_prefill":
        prefill = target.prefill
        monkeypatch.setattr(target, "prefill", lambda *args, **kwargs: prefill(*args, **kwargs))
    hook = model.model.model.layers[0].register_forward_hook(lambda *args: None) if setting == "hooks" else None
    try:
        with DartEngine(target, draft) as engine:
            cfg = DecodeConfig(mode="exact_sampling" if setting == "sampling" else "exact", max_new_tokens=6, draft_len=2)
            engine.generate(prompt, cfg)
    finally:
        if hook is not None:
            hook.remove()
    assert lengths[0] == len(prompt) + 1


def test_verified_draft_prefix_owns_its_keys_and_values():
    target = build_tiny_target()
    prompt = [7, 8, 9, 10]
    source, logits = target.prefill(prompt)
    draft = CompressedKVDrafter(target, QuantSpec(window_len=8))
    draft._reuse_verified_prefix(source, prompt + [int(logits.argmax())], 2)
    assert draft._processed == prompt and draft._state.seq_len == len(prompt)
    with torch.inference_mode():
        for layer, key, value in zip(source.cache.layers, draft._state.cache._kw, draft._state.cache._vw, strict=True):
            for original, copied in ((layer.keys, key), (layer.values, value)):
                assert torch.equal(copied, original)
                assert copied.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
                copied.add_(1)
                assert not torch.equal(copied, original)


def test_verified_draft_prefix_copy_failure_leaves_the_drafter_cold(monkeypatch):
    target = build_tiny_target()
    prompt = [7, 8, 9, 10]
    source, logits = target.prefill(prompt)
    draft = CompressedKVDrafter(target, QuantSpec(window_len=8))
    update = QuantizedKVCache.update

    def fail(self, keys, values, layer_idx, *args, **kwargs):
        if layer_idx == 1:
            raise RuntimeError("interrupted")
        return update(self, keys, values, layer_idx, *args, **kwargs)

    monkeypatch.setattr(QuantizedKVCache, "update", fail)
    with pytest.raises(RuntimeError, match="interrupted"):
        draft._reuse_verified_prefix(source, prompt + [int(logits.argmax())], 1)
    assert draft._state is None and draft._processed == []
    assert source.seq_len == len(prompt)


def test_verified_draft_prefix_is_used_during_profiling(monkeypatch):
    from benchmarks.profiling import profile_methods

    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = [7, 8, 9, 10]
    lengths = track_draft_updates(monkeypatch)
    draft = CompressedKVDrafter(target, QuantSpec(window_len=16))
    with DartEngine(target, draft) as engine:
        def run_profile(tokens):
            lengths.clear()
            return engine.generate(tokens, DecodeConfig(max_new_tokens=6, draft_len=2))

        report = profile_methods({"draft": run_profile}, [prompt], target, {"draft": engine},
                                 reference="draft", repeats=1)
    assert lengths[:2] == [len(prompt), 1]
    assert report["samples"][0]["verify_target_calls"] > 0


@pytest.mark.parametrize("verifier", ["block", "tree"])
def test_shared_ar_prefix_only_forwards_the_pending_token(verifier):
    target = build_tiny_target()
    target.eos_token_ids = frozenset()
    prompt = random_prompt(213, 8)
    expected = target.greedy_generate(prompt, 10)
    widths = []

    def record(module, args, kwargs):
        widths.append(kwargs["input_ids"].size(-1))

    record._dart_observer = True
    with target.model.register_forward_pre_hook(record, with_kwargs=True):
        with DartEngine(target, ARDrafter(target), verifier=verifier) as engine:
            result = engine.generate(prompt, DecodeConfig(max_new_tokens=10, draft_len=2))
    assert result.token_ids == expected
    assert widths == [len(prompt)] + [1, 1, 3] * 3


@pytest.mark.parametrize("setting", ["different_model", "sampling", "hidden", "hook", "custom_forward"])
def test_shared_ar_prefix_respects_draft_execution(monkeypatch, setting):
    target = build_tiny_target()
    draft_model = build_tiny_target(seed=9) if setting == "different_model" else target
    target.eos_token_ids = draft_model.eos_token_ids = frozenset()
    draft = ARDrafter(draft_model)
    prompt = random_prompt(214, 8)
    widths = []

    def record(module, args, kwargs):
        widths.append(kwargs["input_ids"].size(-1))

    record._dart_observer = setting != "hook"
    if setting == "hidden":
        target.record_hidden = True
    if setting == "custom_forward":
        forward = target.forward_block
        monkeypatch.setattr(target, "forward_block", lambda state, tokens: forward(state, tokens))
    mode = "exact_sampling" if setting == "sampling" else "exact"
    with draft_model.model.register_forward_pre_hook(record, with_kwargs=True), DartEngine(target, draft) as engine:
        engine.generate(prompt, DecodeConfig(mode=mode, max_new_tokens=6, draft_len=2, seed=17))
    assert len(prompt) + 1 in widths


def test_shared_ar_prefix_keeps_owned_cache_and_reuses_one_token_extensions(monkeypatch):
    target = build_tiny_target()
    draft = ARDrafter(target)
    prefix = random_prompt(215, 8)
    with torch.inference_mode():
        source, _ = target.prefill(prefix)
        draft._reuse_verified_prefix(source, prefix + [3], 1)
        old = draft._state
        assert draft._processed == prefix and old.seq_len == len(prefix)
        for left, right in zip(old.cache.layers, source.cache.layers, strict=True):
            for owned, original in ((left.keys, right.keys), (left.values, right.values)):
                assert torch.equal(owned, original)
                assert owned.untyped_storage().data_ptr() != original.untyped_storage().data_ptr()
        draft._reuse_verified_prefix(source, prefix + [3], 1)
        assert draft._state is old
        target.forward_block(source, [3, 4])
        clone = torch.Tensor.clone

        def interrupt(tensor, *args, **kwargs):
            if tensor is source.cache.layers[-1].keys:
                raise RuntimeError("interrupted copy")
            return clone(tensor, *args, **kwargs)

        with monkeypatch.context() as hooks:
            hooks.setattr(torch.Tensor, "clone", interrupt)
            with pytest.raises(RuntimeError, match="interrupted copy"):
                draft._reuse_verified_prefix(source, prefix + [3, 4, 5], 1)
        assert draft._state is old and old.seq_len == len(prefix) and draft._processed == prefix
        draft._reuse_verified_prefix(source, prefix + [3, 4, 5], 1)
        assert draft._state.seq_len == source.seq_len == len(prefix) + 2
        assert draft._processed == prefix + [3, 4]
