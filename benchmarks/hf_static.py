from __future__ import annotations

import threading
import time

import torch
from transformers import LlamaForCausalLM, StaticCache

from benchmarks.run_bench import hf_generation_settings
from benchmarks.timing import synchronize
from dart.models.rms_norm import TorchRMSNorm
from dart.runtime.types import GenerationOutput


def _geometry(target):
    model = target.model
    if not isinstance(model, LlamaForCausalLM) or any(module.training for module in model.modules()):
        raise ValueError("persistent static control requires an eval-mode Llama")
    device = model.device
    if (device.type not in ("cpu", "cuda") or target.dtype != torch.float32
            or any(tensor.device != device or (tensor.is_floating_point() and tensor.dtype != torch.float32)
                   for tensor in (*model.parameters(), *model.buffers()))):
        raise ValueError("persistent static control requires one float32 CPU or CUDA device")
    config = model.config
    if config._attn_implementation not in ("eager", "sdpa"):
        raise ValueError("persistent static control requires eager or sdpa attention")
    if device.type == "cuda" and (config._attn_implementation != "eager"
                                 or any(isinstance(module, TorchRMSNorm) for module in model.modules())):
        raise ValueError("persistent CUDA control requires eager attention and ordinary RMSNorm")
    return (config.num_hidden_layers, config.num_attention_heads, config.num_key_value_heads,
            config.head_dim, config.max_position_embeddings, config.vocab_size, str(device))


def _context(device, owner=None):
    if device.type == "cpu":
        return None
    if torch.cuda.current_device() != device.index:
        raise RuntimeError("use the owning CUDA device")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("native generation cannot run during capture")
    current = (threading.get_ident(), torch.cuda.current_stream(device).cuda_stream)
    if owner is not None and current != owner:
        raise RuntimeError("use the owning thread and CUDA stream")
    return current


class PersistentStaticControl:
    """Native greedy with one reset-cleared cache; close after the comparison."""

    @staticmethod
    def validate(target, capacity):
        shape = _geometry(target)
        if type(capacity) is not int or not 1 <= capacity <= target.model.config.max_position_embeddings:
            raise ValueError("persistent static_capacity must be within the model context limit")
        _context(target.model.device)
        return shape

    @torch.inference_mode()
    def __init__(self, target, capacity):
        self.shape = self.validate(target, capacity)
        self.device = target.model.device
        self.owner = _context(self.device)
        self.lock = threading.Lock()
        self.closed = False
        self.target, self.model, self.capacity = target, target.model, capacity
        self.settings = hf_generation_settings(target, 1, record_hidden=False)
        self.settings.update(disable_compile=True, cache_implementation=None)
        synchronize(self.device)
        start = time.perf_counter()
        self.cache = StaticCache(config=self.model.config, max_cache_len=capacity)
        self.cache.early_initialization(batch_size=1, num_heads=self.model.config.num_key_value_heads,
                                        head_dim=self.model.config.head_dim, dtype=torch.float32, device=self.device)
        synchronize(self.device)
        self.setup_ms = (time.perf_counter() - start) * 1_000
        self.cache_bytes = sum(tensor.numel() * tensor.element_size() for layer in self.cache.layers
                               for tensor in (layer.keys, layer.values, layer.cumulative_length))

    @torch.inference_mode()
    def generate(self, prompt, max_new_tokens):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("persistent static control is busy")
        try:
            if self.closed:
                raise RuntimeError("persistent static control is closed")
            _context(self.device, self.owner)
            if self.target.model is not self.model or _geometry(self.target) != self.shape:
                raise ValueError("persistent static cache geometry changed")
            if not isinstance(prompt, (list, tuple)):
                raise ValueError("prompt must contain integer token ids")
            self.target.validate_generation(len(prompt), max_new_tokens)
            if getattr(self.target, "uses_replay_cache", False):
                self.target.validate_prompt(prompt)
            if len(prompt) + max_new_tokens > self.capacity:
                raise ValueError("prompt and output budget exceed persistent static_capacity")
            if any(type(token) is not int or not 0 <= token < self.model.config.vocab_size for token in prompt):
                raise ValueError("prompt must contain integer token ids within the model vocabulary")
            try:
                self.cache.reset()
                token_ids = []
                if max_new_tokens:
                    ids = torch.tensor([prompt], dtype=torch.long, device=self.device)
                    settings = {**self.settings, "max_new_tokens": max_new_tokens}
                    output = self.model.generate(ids, attention_mask=torch.ones_like(ids),
                                                 past_key_values=self.cache, **settings)
                    token_ids = output[0, len(prompt):].tolist()
                text = self.target.tokenizer.decode(token_ids, skip_special_tokens=False) if self.target.tokenizer else None
                return GenerationOutput(token_ids, text, {})
            finally:
                try:
                    self.cache.reset()  # Cleanup stays inside request timing, including failed requests.
                except BaseException:
                    self.cache = self.target = self.model = None
                    self.closed = True
                    raise
        finally:
            self.lock.release()

    def close(self):
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("persistent static control is busy")
        try:
            if not self.closed:
                _context(self.device, self.owner)
                synchronize(self.device)
            self.cache = self.target = self.model = None
            self.closed = True
        finally:
            self.lock.release()

    def __enter__(self):
        if self.closed:
            raise RuntimeError("persistent static control is closed")
        return self

    def __exit__(self, *_):
        self.close()
