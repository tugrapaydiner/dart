"""DART: Diffusion-Augmented Reliable Transformer Serving.

An inference engine where lossy draft lanes (diffusion canvas, MTP/n-gram,
compressed KV, sparse attention) propose tokens and a full autoregressive
target model verifies them, guaranteeing greedy output identical to the
target model in exact mode.
"""

from dart.runtime.types import DecodeConfig, GenerationOutput, VerifyResult

__version__ = "0.1.0"

__all__ = [
    "DecodeConfig",
    "GenerationOutput",
    "VerifyResult",
    "DartEngine",
    "__version__",
]


def __getattr__(name):
    # Lazy import so `import dart` stays cheap and torch loads only on use.
    if name == "DartEngine":
        from dart.runtime.engine import DartEngine

        return DartEngine
    raise AttributeError(f"module 'dart' has no attribute {name!r}")
