from __future__ import annotations

import pytest

from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig


@pytest.mark.parametrize("field,value", [
    ("max_new_tokens", -1), ("max_new_tokens", 1.5), ("max_new_tokens", True),
    ("draft_len", -1), ("draft_len", "4"),
    ("canvas_len", -1), ("canvas_len", 2.5),
    ("verify_every", 0), ("verify_every", -1), ("verify_every", 1.5),
    ("temperature", -0.1), ("temperature", float("nan")),
    ("temperature", float("inf")), ("temperature", "1.0"),
    ("seed", "7"), ("seed", True),
    ("mode", "unknown"), ("full_kv_residency", "unknown"),
    ("allowed_lanes", "ngram"), ("allowed_lanes", ["ngram", 4]),
])
def test_invalid_decode_settings_are_rejected(field, value):
    with pytest.raises(ValueError, match=field):
        DecodeConfig(**{field: value})


def test_zero_budgets_and_temperature_remain_valid():
    config = DecodeConfig(max_new_tokens=0, draft_len=0, canvas_len=0, temperature=0)
    assert config.max_new_tokens == config.draft_len == config.canvas_len == 0


def test_empty_lane_allowlist_remains_valid():
    assert DecodeConfig(allowed_lanes=[]).allowed_lanes == []


@pytest.mark.parametrize("field,value", [
    ("max_new_tokens", -1), ("temperature", float("nan")),
    ("full_kv_residency", "unknown"),
])
def test_changed_config_is_checked_before_prefill(monkeypatch, tiny_target, field, value):
    config = DecodeConfig()
    setattr(config, field, value)

    def unexpected_prefill(*args, **kwargs):
        pytest.fail("Invalid settings reached the model")

    monkeypatch.setattr(tiny_target, "prefill", unexpected_prefill)
    with DartEngine(tiny_target) as engine, pytest.raises(ValueError, match=field):
        engine.generate([5, 6], config)
