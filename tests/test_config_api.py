"""Config + API contract tests (docs/design.md §11).

These pin the three silent-no-op fixes: from_config must honor YAML decode
settings, DecodeConfig.allowed_lanes must actually restrict lanes, and
generate must accept a text prompt when a tokenizer is present.
"""

from __future__ import annotations

import pytest

from dart.drafters.base import NullDrafter
from dart.drafters.ngram import NGramDrafter
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.factory import build_engine, build_target
from dart.runtime.scheduler import HeuristicScheduler, SchedulerConfig
from dart.runtime.types import DecodeConfig
from tests.conftest import build_tiny_target, random_prompt

TINY = {"model": "tiny:1234", "device": "cpu"}


# --------------------- from_config decode defaults ------------------- #
def test_from_config_captures_decode_defaults_and_applies_them():
    eng = build_engine({
        "target": TINY, "mode": "exact", "lanes": {"ngram": {}},
        "decode": {"max_new_tokens": 7, "full_kv_residency": "cpu_pinned"},
    })
    assert eng.default_decode_config is not None
    assert eng.default_decode_config.max_new_tokens == 7
    assert eng.default_decode_config.full_kv_residency == "cpu_pinned"
    # generate() with no explicit config must use the captured defaults.
    out = eng.generate(random_prompt(1, 8))
    assert out.metadata["tokens_generated"] <= 7
    assert out.metadata["full_kv_residency"] == "cpu_pinned"


def test_full_kv_prefetch_async_alias_maps_to_residency():
    eng = build_engine({
        "target": TINY, "lanes": {"ngram": {}},
        "verification": {"full_kv_prefetch": "async"},
    })
    assert eng.default_decode_config.full_kv_residency == "cpu_pinned"


def test_config_without_decode_keys_has_no_defaults():
    eng = build_engine({"target": TINY, "lanes": {"ngram": {}}})
    assert eng.default_decode_config is None  # nothing to capture


# ------------------------- allowed_lanes ----------------------------- #
def _repetitive_prompt():
    span = random_prompt(111, 9)
    return span + random_prompt(112, 3) + span  # ngram will fire


def test_allowed_lanes_restricts_scheduled_engine():
    target = build_tiny_target()
    lanes = {"ngram": NGramDrafter(), "target_only": NullDrafter()}
    sched = HeuristicScheduler(SchedulerConfig(available_lanes=list(lanes)))
    eng = DartEngine(target, lanes=lanes, scheduler=sched, verifier="block")
    prompt = _repetitive_prompt()
    base = target.greedy_generate(prompt, max_new_tokens=24)

    restricted = eng.generate(
        prompt,
        DecodeConfig(max_new_tokens=24, draft_len=4, allowed_lanes=["target_only"]),
    )
    assert restricted.token_ids == base
    assert restricted.metadata["draft_tokens_proposed"] == 0  # ngram excluded

    free = eng.generate(prompt, DecodeConfig(max_new_tokens=24, draft_len=4))
    assert free.metadata["draft_tokens_proposed"] > 0  # control: ngram drafts


def test_allowed_lanes_restricts_single_drafter_engine():
    target = build_tiny_target()
    eng = DartEngine(target, drafter=NGramDrafter(), verifier="block")
    prompt = _repetitive_prompt()
    base = target.greedy_generate(prompt, max_new_tokens=24)
    out = eng.generate(
        prompt, DecodeConfig(max_new_tokens=24, draft_len=4, allowed_lanes=["mtp"])
    )
    assert out.token_ids == base  # disallowed lane -> target-only, still exact
    assert out.metadata["draft_tokens_proposed"] == 0


# --------------------------- text prompt ----------------------------- #
class _IntTok:
    def encode(self, text, add_special_tokens=False):
        return [int(t) for t in text.split()]

    def decode(self, ids, skip_special_tokens=False):
        return " ".join(str(i) for i in ids)


def test_generate_accepts_text_prompt_with_tokenizer():
    target = build_tiny_target()
    target.tokenizer = _IntTok()
    text = " ".join(str(t) for t in random_prompt(7, 10))
    eng = DartEngine(target, drafter=NGramDrafter(), verifier="block")
    out = eng.generate(text, DecodeConfig(max_new_tokens=12, draft_len=4))
    base = target.greedy_generate([int(t) for t in text.split()], max_new_tokens=12)
    assert out.token_ids == base


def test_generate_text_prompt_without_tokenizer_raises():
    target = build_tiny_target()  # tokenizer is None
    eng = DartEngine(target, drafter=NGramDrafter())
    with pytest.raises(ValueError, match="tokenizer"):
        eng.generate("1 2 3", DecodeConfig(max_new_tokens=4))


# ------------- design.md §11 DecodeConfig(target_model=...) ---------- #
def test_decodeconfig_accepts_target_model_and_reports_it():
    # The literal docs/design.md §11 example must construct without error.
    cfg = DecodeConfig(
        mode="exact", max_new_tokens=8, target_model="gemma-4-26b-it",
        allowed_lanes=["ngram"], draft_len=4,
    )
    target = build_tiny_target()
    eng = DartEngine(target, drafter=NGramDrafter(), verifier="block")
    out = eng.generate(random_prompt(9, 10), cfg)
    # When set, target_model is surfaced in metadata (18.2) instead of the
    # autodetected class name.
    assert out.metadata["target_model"] == "gemma-4-26b-it"


def test_target_model_defaults_to_autodetected_name():
    target = build_tiny_target()
    out = DartEngine(target, drafter=NGramDrafter()).generate(
        random_prompt(9, 10), DecodeConfig(max_new_tokens=8)
    )
    assert out.metadata["target_model"]  # non-empty autodetected name
    assert out.metadata["target_model"] != "gemma-4-26b-it"


@pytest.fixture
def local_hf_loaders(monkeypatch):
    monkeypatch.setattr(
        "dart.models.target.AutoModelForCausalLM.from_pretrained",
        lambda *args, **kwargs: build_tiny_target().model,
    )
    monkeypatch.setattr(
        "dart.models.target.AutoTokenizer.from_pretrained", lambda *args, **kwargs: None,
    )


@pytest.mark.parametrize("model", ["tiny:1234", "hf:local-test", "local-test"])
@pytest.mark.parametrize("record_hidden", [None, False, True])
def test_target_config_controls_hidden_states(local_hf_loaders, model, record_hidden):
    spec = {"model": model, "device": "cpu"}
    if record_hidden is not None:
        spec["record_hidden"] = record_hidden
    target = build_target(spec)
    assert target.record_hidden is bool(record_hidden)
    state, _ = target.prefill([5, 6, 7])
    if record_hidden:
        assert target.last_hidden_block.shape == (3, target.model.config.hidden_size)
    else:
        assert target.last_hidden_block is None
    target.forward_block(state, [8])
    if record_hidden:
        assert target.last_hidden_block.shape == (1, target.model.config.hidden_size)
    else:
        assert target.last_hidden_block is None


@pytest.mark.parametrize("record_hidden", [False, True])
def test_pretrained_target_accepts_hidden_recording(local_hf_loaders, record_hidden):
    target = TargetModel.from_pretrained("local-test", record_hidden=record_hidden)
    target.prefill([5, 6, 7])
    assert (target.last_hidden_block is not None) is record_hidden


def test_hf_config_with_mtp_generates_exact_tokens(local_hf_loaders):
    with build_engine({
        "target": {"model": "hf:local-test", "record_hidden": True},
        "lanes": {"mtp": {"num_offsets": 2}},
    }) as engine:
        prompt = random_prompt(305, 8)
        expected = engine.target.greedy_generate(prompt, max_new_tokens=12)
        result = engine.generate(prompt, DecodeConfig(max_new_tokens=12, draft_len=2))
    assert result.token_ids == expected
    assert result.metadata["draft_tokens_proposed"] > 0
