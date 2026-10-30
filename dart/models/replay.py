"""Bounded CUDA replay for batch-one Llama verification blocks."""

from __future__ import annotations

import threading
import time
from importlib.metadata import version

import torch
from packaging.version import Version
from transformers import LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from dart.models.target import TargetModel, TargetState


class ReplayCache(DynamicCache):
    """A request-owned KV snapshot; only its valid prefix can be read."""

    def __init__(self, storage: torch.Tensor, length: int, owner: object):
        super().__init__()
        self.storage = storage
        self.length = length
        self.owner = owner

    @classmethod
    def from_dynamic(cls, cache: DynamicCache, capacity: int, owner: object):
        if type(cache) is not DynamicCache or not cache.layers:
            raise ValueError("replay requires an initialized full DynamicCache")
        length = cache.get_seq_length()
        if type(capacity) is not int or not 0 < length <= capacity:
            raise ValueError("cache prefix exceeds replay capacity")
        keys = cache.layers[0].keys
        shape = keys.shape
        if keys.ndim != 4 or shape[0] != 1 or shape[-2] != length:
            raise ValueError("replay requires a batch-one cache")
        for layer in cache.layers:
            for tensor in (layer.keys, layer.values):
                if tensor.shape != shape or tensor.dtype != keys.dtype or tensor.device != keys.device:
                    raise ValueError("replay cache layers must have matching shapes, devices and dtypes")
        storage = keys.new_empty((len(cache.layers), 2, *shape[:2], capacity, shape[-1]))
        for index, layer in enumerate(cache.layers):
            storage[index, 0, ..., :length, :].copy_(layer.keys)
            storage[index, 1, ..., :length, :].copy_(layer.values)
        return cls(storage, length, owner)

    @property
    def is_compileable(self):
        return False

    def get_seq_length(self, layer_idx=0):
        return self.length

    def get_mask_sizes(self, query_length, layer_idx=0):
        return self.length + query_length, 0

    def crop(self, tokens_to_remove):
        if isinstance(tokens_to_remove, torch.Tensor):
            if tokens_to_remove.ndim != 0 or tokens_to_remove.dtype not in (torch.int32, torch.int64):
                raise ValueError("replay crop requires an integer scalar")
            tokens_to_remove = tokens_to_remove.item()
        if type(tokens_to_remove) is not int or not -self.length <= tokens_to_remove <= 0:
            raise ValueError("replay crop requires a negative removal count within the cache")
        self.length += tokens_to_remove  # Rejected slots stay masked until overwritten.

    def update(self, *args, **kwargs):
        raise ValueError("update replay caches through CudaGraphTarget")


class _Workspace:
    def __init__(self, model, capacity):
        config = model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        shape = (config.num_hidden_layers, 2, 1, config.num_key_value_heads, capacity, head_dim)
        if torch.Size(shape).numel() * 4 > 256 * 1024**2:
            raise ValueError("replay KV workspace exceeds 256 MiB")
        parameter = next(model.parameters())
        self.storage = torch.zeros(shape, device=parameter.device, dtype=parameter.dtype)
        self.positions = None

    def update(self, keys, values, layer_idx, *args, **kwargs):
        a, b = self.storage[layer_idx, 0], self.storage[layer_idx, 1]
        a.index_copy_(2, self.positions, keys)
        b.index_copy_(2, self.positions, values)
        return a, b


class _ReplayBlock:
    def __init__(self, model, workspace, length, capture, *, capture_predictions=False):
        self.forward_model = model.forward
        self.workspace = workspace
        device = workspace.storage.device
        self.ids = torch.zeros((1, length), dtype=torch.long, device=device)
        self.start = torch.zeros((), dtype=torch.long, device=device)
        self.steps = torch.arange(length, device=device)
        self.columns = torch.arange(workspace.storage.shape[-2], device=device)
        self.graph = None
        self.output = None
        self.capture_predictions = capture_predictions
        self.predictions = None
        if capture:
            stream = torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.forward()
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.device(device), torch.cuda.graph(self.graph, stream=stream):
                self.output = self.forward()
            torch.cuda.synchronize(device)

    def forward(self):
        positions = self.steps + self.start
        self.workspace.positions = positions
        mask = self.workspace.storage.new_zeros((1, 1, self.ids.shape[1], self.columns.numel()))
        mask.masked_fill_(self.columns[None, :] > positions[:, None], torch.finfo(mask.dtype).min)
        logits = self.forward_model(input_ids=self.ids, position_ids=positions[None], attention_mask=mask,
                                    past_key_values=self.workspace, use_cache=True, output_hidden_states=False).logits
        if self.capture_predictions:
            self.predictions = logits.argmax(dim=-1)
        return logits

    def run(self):
        if self.graph is None:
            return self.forward()
        self.graph.replay()
        return self.output


class CudaGraphTarget(TargetModel):
    """Opt-in FP32 Llama replay for transformers >= 5.16.1.

    Graph construction includes warmup and capture; prepare with no other CUDA work
    running in the process. Calls use the preparation stream and serialize
    workspace access. Each state and returned logit tensor owns its data.
    ``replay=False`` runs the static-cache path on CPU or CUDA without graph capture.
    ``capture_predictions=True`` also prepares argmax for entropy-free greedy calls.
    """

    uses_replay_cache = True

    @torch.inference_mode()
    def __init__(self, target: TargetModel, *, max_block_tokens=5, capacity=None, replay=True,
                 capture_predictions=False):
        if type(replay) is not bool:
            raise ValueError("replay must be a boolean")
        if type(capture_predictions) is not bool:
            raise ValueError("capture_predictions must be a boolean")
        device = torch.device(target.device).type
        if replay and (device != "cuda" or target.dtype != torch.float32):
            raise ValueError("CUDA replay requires a float32 CUDA target")
        if device not in {"cpu", "cuda"} or target.dtype != torch.float32:
            raise ValueError("static execution requires a float32 CPU or CUDA target")
        if not isinstance(target.model, LlamaForCausalLM) or Version(version("transformers")) < Version("5.16.1"):
            raise ValueError("static cache execution requires LlamaForCausalLM and transformers >= 5.16.1")
        config = target.model.config
        rope = getattr(config, "rope_parameters", None) or getattr(config, "rope_scaling", None) or {}
        if target.record_hidden or config._attn_implementation != "eager" or rope.get("rope_type", "default") != "default":
            raise ValueError("static cache execution requires eager attention, default RoPE and no hidden-state recording")
        capacity = config.max_position_embeddings if capacity is None else capacity
        if type(capacity) is not int or not 1 <= capacity <= min(4096, config.max_position_embeddings):
            raise ValueError("replay capacity must be within the model context and at most 4096")
        if type(max_block_tokens) is not int or not 1 <= max_block_tokens <= min(16, capacity):
            raise ValueError("max_block_tokens must be an integer from 1 to min(16, capacity)")
        if len({parameter.device for parameter in target.model.parameters()}) != 1 or any(
            parameter.dtype != torch.float32 or parameter.device.type != device for parameter in target.model.parameters()
        ):
            raise ValueError(f"replay model parameters must all use float32 on one {device.upper()} device")
        super().__init__(target.model, target.tokenizer, device=target.device, dtype=target.dtype)
        self.eos_token_ids = target.eos_token_ids
        self.capacity = capacity
        self.max_block_tokens = max_block_tokens
        self._capture_predictions = capture_predictions
        self.execution_backend = "cuda_graph" if replay else "static"
        self._owner = object()
        self._lock = threading.RLock()
        self._closed = False
        self._stream = torch.cuda.current_stream(next(self.model.parameters()).device) if device == "cuda" else None
        self._model_signature = self._signature()
        self._model_tensors = tuple(tensor.detach() for tensor in (*self.model.parameters(), *self.model.buffers()))
        self._workspace = _Workspace(self.model, capacity)
        started = time.perf_counter()
        self._blocks = {length: _ReplayBlock(self.model, self._workspace, length, replay,
                                            capture_predictions=capture_predictions)
                        for length in range(1, max_block_tokens + 1)}
        self.setup_ms = (time.perf_counter() - started) * 1000

    @property
    def capture_predictions(self):
        return self._capture_predictions

    def _signature(self):
        pending, seen, tensors = [self.model], set(), []
        while pending:
            module = pending.pop()
            if module is None or module in seen:
                continue
            seen.add(module)
            pending.extend(module._modules.values())
            tensors.extend(tensor for tensor in (*module._parameters.values(), *module._buffers.values())
                           if tensor is not None)  # Check live registrations without building parameter names.
        return tuple((tensor.data_ptr(), tensor.device, tensor.dtype, tuple(tensor.shape), tensor.stride())
                     for tensor in tensors)

    def _check_ready(self):
        if self._closed:
            raise RuntimeError("replay target is closed")
        if self._stream is not None and torch.cuda.current_stream(self._stream.device) != self._stream:
            raise ValueError("use the CUDA stream that prepared the replay target")
        if self.record_hidden or self.model.training or self.model.config._attn_implementation != "eager":
            raise ValueError("replay requires unchanged inference settings")
        if self._signature() != self._model_signature:
            raise ValueError("model tensor storage changed; prepare a new replay target")

    @property
    def context_limit(self):
        limit = super().context_limit
        return self.capacity if limit is None else min(limit, self.capacity)

    def _check_context_length(self, length):
        super()._check_context_length(length)
        if length > self.capacity:
            raise ValueError(f"sequence length {length} exceeds replay capacity {self.capacity}")

    def _ids(self, tokens):
        short_block = isinstance(tokens, (list, tuple)) and 0 < len(tokens) <= self.max_block_tokens
        if short_block and all(type(token) is int for token in tokens):
            if min(tokens) < 0 or max(tokens) >= self.model.config.vocab_size:
                raise ValueError("token id is outside the model vocabulary")
            return torch.tensor([tokens], dtype=torch.long, device=self.device)
        values = tokens.to(dtype=torch.long) if isinstance(tokens, torch.Tensor) else torch.as_tensor(tokens, dtype=torch.long)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[0] != 1 or values.shape[1] == 0:
            raise ValueError("replay expects nonempty [T] or [1, T] token ids")
        low, high = torch.aminmax(values)
        if low.item() < 0 or high.item() >= self.model.config.vocab_size:
            raise ValueError("token id is outside the model vocabulary")
        return values.to(device=self.device)

    @torch.inference_mode()
    def prefill(self, prompt_ids, cache=None):
        if cache is not None:
            raise ValueError("replay prefill creates its own request cache")
        return super().prefill(prompt_ids)

    def validate_prompt(self, prompt_ids):
        with self._lock:
            self._check_ready()
            super().validate_prompt(prompt_ids)

    @torch.inference_mode()
    def forward_predictions(self, state, tokens):
        """Return owned next-token IDs from a prepared greedy block."""
        if not self.capture_predictions:
            raise ValueError("prepare the target with capture_predictions=True")
        return self._forward(state, self._ids(tokens), return_predictions=True)

    def _forward(self, state, input_ids, *, last_logits_only=False, return_predictions=False):
        with self._lock:
            self._check_ready()
            state._check_usable()
            total_len = state.seq_len + input_ids.shape[1]
            self._check_context_length(total_len)
            if not isinstance(state.cache, ReplayCache) and state.seq_len == 0:
                if type(state.cache) is not DynamicCache:
                    raise ValueError("replay requires a full DynamicCache")
                logits = super()._forward(state, input_ids, last_logits_only=last_logits_only)
                return logits.argmax(dim=-1).tolist() if return_predictions else logits
            if input_ids.shape[1] not in self._blocks:
                raise ValueError(f"block exceeds prepared replay size {self.max_block_tokens}")
            cache = state.cache
            if not isinstance(cache, ReplayCache):
                cache = ReplayCache.from_dynamic(cache, self.capacity, self._owner)
            if cache.owner is not self._owner or cache.storage.shape != self._workspace.storage.shape:
                raise ValueError("cache belongs to a different replay target")
            block = self._blocks[input_ids.shape[1]]
            self._workspace.storage[..., :cache.length, :].copy_(cache.storage[..., :cache.length, :])
            block.ids.copy_(input_ids)
            block.start.fill_(cache.length)
            logits = block.run()
            cache.storage[..., cache.length:total_len, :].copy_(self._workspace.storage[..., cache.length:total_len, :])
            if return_predictions:
                output = block.predictions[0].tolist()  # Copy before another call can replay this workspace.
            else:
                output = logits[0].float().clone()  # Later replays must not overwrite a caller's logits.
            cache.length = total_len
            state.cache = cache
            return output

    def crop(self, state, length):
        with self._lock:
            if self._closed:
                raise RuntimeError("replay target is closed")
            if type(length) is not int:
                raise ValueError("cache length must be an integer")
            if isinstance(state.cache, ReplayCache) and state.cache.owner is not self._owner:
                raise ValueError("cache belongs to a different replay target")
            super().crop(state, length)

    def forward_tree(self, *args, **kwargs):
        raise ValueError("CUDA replay supports block verification, not tree verification")

    @torch.inference_mode()
    def greedy_generate(self, prompt_ids, max_new_tokens):
        ids = self._ids(prompt_ids)
        self.validate_generation(ids.shape[1], max_new_tokens)
        if max_new_tokens == 0:
            self.validate_prompt(ids)
            return []
        state = TargetState(DynamicCache())
        logits = self._forward(state, ids, last_logits_only=True)[-1]
        generated = []
        token = int(logits.argmax().item())
        for _ in range(max_new_tokens):
            generated.append(token)
            if token in self.eos_token_ids or len(generated) == max_new_tokens:
                break
            if self.capture_predictions:
                token = self.forward_predictions(state, [token])[0]
            else:
                token = int(self.forward_block(state, [token])[-1].argmax().item())
        return generated

    def close(self):
        with self._lock:
            if not self._closed:
                if self._stream is not None:
                    self._stream.synchronize()
                self._blocks.clear()
                self._workspace = None
                self._model_tensors = ()
                self._closed = True
