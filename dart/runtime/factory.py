"""Build engines from YAML configs (docs/design.md §11:
``DartEngine.from_config("configs/dart_exact.yaml")``).

Target schemes:
  * ``hf:<name-or-path>``  - any HuggingFace causal LM
  * ``tiny:<seed>``        - seeded random tiny Llama (tests/demos, no download)

Lanes are declared by name with their own blocks; unknown lanes fail
loudly rather than silently degrading to target-only.
"""

from __future__ import annotations

from typing import Optional

import torch
import yaml

from dart.diffusion.reconciliation import ReconcilePolicy
from dart.drafters.ar_draft import ARDrafter
from dart.drafters.base import NullDrafter
from dart.drafters.compressed_kv import CompressedKVDrafter
from dart.drafters.diffusion_small import (
    DiffusionDrafter,
    DiffusionDrafterConfig,
    DiffusionDrafterModel,
)
from dart.drafters.mtp import MTPDrafter, MTPHeads
from dart.drafters.ngram import NGramDrafter
from dart.drafters.sparse_attention import SparseAttentionDrafter
from dart.kv.cache import QuantSpec
from dart.kv.compressors.token_drop import TokenDropSpec
from dart.models.target import TargetModel
from dart.runtime.scheduler import (
    BanditScheduler,
    HeuristicScheduler,
    SchedulerConfig,
)
from dart.runtime.time_scheduler import TimeScheduler


def build_target(spec: dict) -> TargetModel:
    name = spec["model"]
    device = spec.get("device", "cpu")
    if name.startswith("tiny:"):
        return build_tiny_target(seed=int(name.split(":", 1)[1]), spec=spec)
    if name.startswith("hf:"):
        name = name.split(":", 1)[1]
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}.get(
        spec.get("precision", ""), None
    )
    return TargetModel.from_pretrained(
        name,
        device=device,
        dtype=dtype,
        trust_remote_code=bool(spec.get("trust_remote_code", False)),
        record_hidden=bool(spec.get("record_hidden", False)),
        attn_implementation=spec.get("attn_implementation", "eager"),
        local_files_only=bool(spec.get("local_files_only", False)),
        native_rms_norm=spec.get("native_rms_norm", False),
    )


def build_tiny_target(seed: int, spec: Optional[dict] = None) -> TargetModel:
    from transformers import LlamaConfig, LlamaForCausalLM

    spec = spec or {}
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=spec.get("vocab_size", 211),
        hidden_size=spec.get("hidden_size", 64),
        intermediate_size=spec.get("intermediate_size", 128),
        num_hidden_layers=spec.get("num_layers", 2),
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=2048,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation=spec.get("attn_implementation", "eager"),
    )
    model = LlamaForCausalLM(cfg)
    return TargetModel(
        model,
        tokenizer=None,
        device=spec.get("device", "cpu"),
        dtype={"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}.get(
            spec.get("precision", "fp32"), torch.float32
        ),
        record_hidden=bool(spec.get("record_hidden", False)),
        native_rms_norm=spec.get("native_rms_norm", False),
    )


def build_lane(name: str, spec: dict, target: TargetModel):
    if name == "ngram":
        return NGramDrafter(
            max_ngram=spec.get("max_ngram", 4),
            min_ngram=spec.get("min_ngram", 1),
            max_scan=spec.get("max_scan"),
        )
    if name == "compressed_kv":
        store_factory = None
        if spec.get("method") == "polar_vq":
            from dart.kv.compressors.turboquant_like import PolarVQStore

            store_factory = PolarVQStore
        qspec = QuantSpec(
            bits=spec.get("bits", 4),
            sink_len=spec.get("sink_tokens", 16),
            sink_bits=spec.get("sink_token_bits", 8),
            window_len=spec.get("recent_window", 32),
        )
        if spec.get("precision_policy"):
            from dart.kv.compressors.hybrid_policy import precision_policy_from_dict

            qspec.policy = precision_policy_from_dict(
                spec["precision_policy"],
                num_layers=target.model.config.num_hidden_layers,
            )
        return CompressedKVDrafter(target, qspec, store_factory=store_factory)
    if name == "sparse_kv":
        return SparseAttentionDrafter(
            target,
            TokenDropSpec(
                sink_len=spec.get("sink_tokens", 16),
                window_len=spec.get("window", 256),
            ),
        )
    if name == "block_sparse_kv":
        from dart.drafters.block_sparse_kv import BlockSparseKVDrafter

        return BlockSparseKVDrafter(
            target,
            block_size=spec.get("block_size", 16),
            top_k=spec.get("top_k", 4),
            sink_blocks=spec.get("sink_blocks", 1),
            recent_blocks=spec.get("recent_blocks", 2),
        )
    if name == "ar_draft":
        draft_target = build_target(spec)
        return ARDrafter(draft_target)
    if name == "mtp":
        heads = MTPHeads(
            hidden_size=target.model.config.hidden_size,
            vocab_size=target.model.config.vocab_size,
            num_offsets=spec.get("num_offsets", 3),
        )
        ckpt = spec.get("checkpoint")
        if ckpt:
            heads.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True)
            )
        return MTPDrafter(target, heads)
    if name == "diffusion":
        cfg = DiffusionDrafterConfig(
            vocab_size=target.model.config.vocab_size,
            hidden_size=spec.get("hidden_size", 256),
            num_layers=spec.get("num_layers", 4),
            num_heads=spec.get("num_heads", 4),
            ctx_window=spec.get("ctx_window", 64),
            max_canvas=spec.get("max_canvas", 64),
        )
        model = DiffusionDrafterModel(cfg)
        ckpt = spec.get("checkpoint")
        if ckpt:
            model.load_state_dict(
                torch.load(ckpt, map_location="cpu", weights_only=True)
            )
        policy = ReconcilePolicy(
            mode=spec.get("reconcile", "reconcile"),
            repair_steps=spec.get("repair_steps", 4),
        )
        return DiffusionDrafter(
            model,
            canvas_len=spec.get("canvas_length", 32),
            denoise_steps=spec.get("max_denoise_steps", 8),
            min_confidence=spec.get("min_confidence", 0.65),
            adaptive_stop=spec.get("adaptive_stop", True),
            reconcile=policy,
            device=target.device,
        )
    raise ValueError(f"unknown lane {name!r}")


def decode_defaults(cfg: dict):
    """DecodeConfig captured from the YAML so from_config honors decode-time
    settings (docs/design.md §11): top-level ``mode``/``max_new_tokens``,
    an optional ``decode:`` block, and the ``verification.full_kv_prefetch:
    async`` alias for cpu_pinned residency.  Unknown keys are ignored."""
    import dataclasses

    from dart.runtime.types import DecodeConfig

    dec = dict(cfg.get("decode", {}) or {})
    if "mode" in cfg:
        dec.setdefault("mode", cfg["mode"])
    if "max_new_tokens" in cfg:
        dec.setdefault("max_new_tokens", cfg["max_new_tokens"])
    ver = cfg.get("verification", {}) or {}
    if ver.get("full_kv_prefetch") == "async":
        dec.setdefault("full_kv_residency", "cpu_pinned")
    fields = {f.name for f in dataclasses.fields(DecodeConfig)}
    kw = {k: v for k, v in dec.items() if k in fields}
    return DecodeConfig(**kw) if kw else None


def build_engine(cfg: dict):
    from dart.runtime.engine import DartEngine

    threads = cfg.get("threads")
    if threads is not None:
        if type(threads) is not int or threads < 1:
            raise ValueError("threads must be an integer >= 1")
        torch.set_num_threads(threads)  # Applies to the whole process.
    target = build_target(cfg["target"])
    defaults = decode_defaults(cfg)
    lane_specs: dict = cfg.get("lanes", {})
    if isinstance(lane_specs, list):  # allow simple list form from the doc
        lane_specs = {name: {} for name in lane_specs}

    if not lane_specs:
        return DartEngine(
            target, verifier=cfg.get("verifier", "block"),
            default_decode_config=defaults,
        )

    if len(lane_specs) == 1 and not cfg.get("scheduler"):
        name, spec = next(iter(lane_specs.items()))
        return DartEngine(
            target, drafter=build_lane(name, spec or {}, target),
            verifier=cfg.get("verifier", "block"),
            default_decode_config=defaults,
        )

    lanes = {
        name: build_lane(name, spec or {}, target)
        for name, spec in lane_specs.items()
    }
    lanes.setdefault("target_only", NullDrafter())
    sched_cfg = cfg.get("scheduler", {}) or {}
    sched_type = sched_cfg.get("type", "heuristic")
    sched_cls = (
        BanditScheduler if sched_type in ("contextual_bandit", "bandit")
        else TimeScheduler if sched_type == "time"
        else HeuristicScheduler
    )
    timing_options = {
        key: sched_cfg[key] for key in ("retry_interval", "warmup_rounds", "min_time_saving")
        if key in sched_cfg
    } if sched_cls is TimeScheduler else {}
    scheduler = sched_cls(
        SchedulerConfig(
            available_lanes=list(lanes),
            long_context_threshold=sched_cfg.get("long_context_threshold", 128_000),
            min_lane_acceptance=sched_cfg.get("min_lane_acceptance", 0.30),
            mtp_acceptance_gate=sched_cfg.get("mtp_acceptance_gate", 0.65),
            ewma_alpha=sched_cfg.get("ewma_alpha", 0.2),
            draft_len=sched_cfg.get("draft_len", 4),
            canvas_len=sched_cfg.get("canvas_len", 32),
            denoise_steps=sched_cfg.get("denoise_steps", 8),
            memory_penalty=sched_cfg.get("memory_penalty", 0.0),
            max_gpu_kv_gb=sched_cfg.get("max_gpu_kv_gb", float("inf")),
            max_rejection_rate=sched_cfg.get("max_rejection_rate", 1.0),
            entropy_gate_threshold=sched_cfg.get(
                "entropy_gate_threshold", float("inf")
            ),
            entropy_gate_factor=sched_cfg.get("entropy_gate_factor", 0.5),
        ),
        **timing_options,
    )
    return DartEngine(
        target, lanes=lanes, scheduler=scheduler, verifier=cfg.get("verifier", "block"),
        default_decode_config=defaults,
    )


def engine_from_yaml(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    return build_engine(cfg)
