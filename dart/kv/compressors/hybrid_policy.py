"""Hybrid precision policy (docs/design.md §9.1 and §9.4).

The QuantizedKVCache already implements the load-bearing hybrid structure:
full-precision recent window + high-bit sinks + low-bit middle.  This
module exposes named policies over that mechanism, including the full
``precision_policy`` form of docs/design.md §9.4 (per-layer + token-class bits).
"""

from __future__ import annotations

from typing import Optional

from dart.kv.cache import PrecisionPolicy, QuantSpec


def hybrid_spec(
    default_kv_bits: int = 4,
    sink_tokens: int = 16,
    sink_tokens_bits: int = 8,
    recent_window: int = 64,
) -> QuantSpec:
    return QuantSpec(
        bits=default_kv_bits,
        sink_len=sink_tokens,
        sink_bits=sink_tokens_bits,
        window_len=recent_window,
    )


def precision_policy_from_dict(
    policy: dict, num_layers: Optional[int] = None
) -> PrecisionPolicy:
    """Build a PrecisionPolicy from a ``precision_policy`` config mapping.

    Accepts the section-14.3 field names directly::

        precision_policy:
          default_kv_bits: 4
          sink_tokens_bits: 8
          last_2_layers_bits: 8
          syntax_tokens_bits: 8
          high_entropy_bits: 8
          # DART extensions, optional:
          syntax_token_ids: [123, 124]   # ids of braces/punctuation/delimiters
          high_entropy_threshold: 2.5    # nats; omit to disable the entropy rule
    """
    last_layer_bits = policy.get("last_2_layers_bits")
    return PrecisionPolicy(
        default_bits=policy.get("default_kv_bits", policy.get("default_bits", 4)),
        sink_len=policy.get("sink_tokens", 16),
        sink_bits=policy.get("sink_tokens_bits", policy.get("sink_token_bits", 8)),
        syntax_token_ids=frozenset(policy.get("syntax_token_ids", []) or []),
        syntax_bits=policy.get("syntax_tokens_bits", 8),
        high_entropy_threshold=policy.get("high_entropy_threshold", float("inf")),
        high_entropy_bits=policy.get("high_entropy_bits", 8),
        num_layers=num_layers,
        last_layers_hi=policy.get("last_layers_hi", 2 if last_layer_bits else 0),
        layer_hi_bits=last_layer_bits or policy.get("layer_hi_bits", 8),
        first_layer_hi=bool(policy.get("first_layer_hi", False)),
    )


def from_policy_dict(policy: dict, num_layers: Optional[int] = None) -> QuantSpec:
    """Build a full QuantSpec (window structure + precision policy) from a
    config ``kv_policy`` / ``precision_policy`` mapping."""
    spec = hybrid_spec(
        default_kv_bits=policy.get("default_bits", policy.get("default_kv_bits", 4)),
        sink_tokens=policy.get("sink_tokens", 16),
        sink_tokens_bits=policy.get("sink_token_bits", policy.get("sink_tokens_bits", 8)),
        recent_window=policy.get("recent_window", 64),
    )
    spec.policy = precision_policy_from_dict(policy, num_layers=num_layers)
    return spec
