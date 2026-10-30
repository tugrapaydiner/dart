"""KV residency policy: where full vs compressed KV lives (docs/design.md §9.2).

v0.1 implements the synchronous reference policy:

  * ``gpu``        - everything on the accelerator (default when it fits)
  * ``cpu_pinned`` - full-precision verifier KV parked in pinned host
                     memory, moved to the device per verify round

Asynchronous chunk prefetch and a GPU verifier pool are roadmap work (docs/design.md §16)
(kv/prefetch.py).  On CPU-only hosts these are no-ops, kept so configs
stay portable across machines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

try:
    from transformers.cache_utils import DynamicCache
except ImportError as e:  # pragma: no cover
    raise ImportError("dart requires `transformers>=5.16.1`") from e

Residency = Literal["gpu", "cpu_pinned"]


@dataclass
class ResidencyStats:
    moved_tensors: int = 0
    moved_bytes: int = 0


def _layer_slots(cache: DynamicCache):
    """Yield (getter, setter) pairs for every K/V tensor in the cache.

    Handles both DynamicCache layouts: the ``layers`` attribute-based layout
    (transformers >= 4.54) and the legacy ``key_cache``/``value_cache`` list
    layout.  On new transformers ``key_cache`` is a *property that returns a
    fresh list*, so writing into that list is silently lost - the setter
    returned here always writes back to the cache's real storage.  (Same
    dual-layout dispatch as ``TargetModel.gather_tree_tail``.)
    """
    layers = getattr(cache, "layers", None)
    if layers is not None:  # transformers >= 4.54 layout
        for layer in layers:
            for attr in ("keys", "values"):
                yield (
                    lambda lyr=layer, a=attr: getattr(lyr, a),
                    lambda v, lyr=layer, a=attr: setattr(lyr, a, v),
                )
        return
    for name in ("key_cache", "value_cache"):  # legacy list layout
        tensors = getattr(cache, name, None)
        if not tensors:
            continue
        for i in range(len(tensors)):
            yield (
                lambda ts=tensors, j=i: ts[j],
                lambda v, ts=tensors, j=i: ts.__setitem__(j, v),
            )


def move_cache(cache: DynamicCache, device: str | torch.device, pin: bool = False) -> ResidencyStats:
    """Synchronously move a DynamicCache's tensors; returns transfer stats.

    Stats count only tensors actually moved, so metadata like
    ``kv_residency_moved_mb`` reflects real transfers, never intent.
    """
    stats = ResidencyStats()
    target = torch.device(device)
    if not torch.cuda.is_available() and target.type != "cpu":
        return stats  # nothing to do on CPU-only hosts
    if target.type == "cuda" and target.index is None:
        target = torch.device("cuda", torch.cuda.current_device())  # Tensors report an explicit index.
    for get, set_ in _layer_slots(cache):
        t = get()
        if t is None or not isinstance(t, torch.Tensor) or t.device == target:
            continue
        moved = t.to(device, non_blocking=True)
        if pin and target.type == "cpu":
            moved = moved.pin_memory()
        set_(moved)
        stats.moved_tensors += 1
        stats.moved_bytes += t.numel() * t.element_size()
    return stats
