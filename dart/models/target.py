"""Target model wrapper: the trusted autoregressive model.

The target is the single source of truth in exact mode.  Everything else in
DART (drafters, compressed KV, diffusion canvases) is allowed to be lossy;
only tokens equal to the target argmax are ever committed.

The wrapper exposes a deliberately small surface:

  * ``prefill``        - run the prompt, return a KV state + next-token logits
  * ``forward_block``  - run T tokens against an existing KV state in one pass
  * ``crop``           - roll the KV state back after rejected drafts
  * ``greedy_generate``- the trusted full-KV greedy baseline (K=0 decode loop)

Batch size is fixed at 1 for interactive serving
(docs/design.md §1 and §15).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

try:  # transformers >= 4.40
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.cache_utils import DynamicCache
except ImportError as e:  # pragma: no cover
    raise ImportError("dart requires `transformers>=4.40`") from e


@dataclass
class TargetState:
    """Mutable KV-cache state for one sequence.

    ``cache`` covers the first ``seq_len`` tokens of the sequence.  Under the
    decode-loop invariant (see dart/runtime/types.py) this is everything but
    the pending token.

    A failed forward or cache update invalidates this state. Start a new prefill instead
    of reusing a cache whose layers may have been only partly updated.
    """

    cache: DynamicCache
    _failed: bool = field(default=False, init=False, repr=False)

    def _check_usable(self) -> None:
        if self._failed:
            raise RuntimeError("request cache is invalid after a failed forward or cache update; start a new prefill")

    @property
    def seq_len(self) -> int:
        return self.cache.get_seq_length()


class TargetModel:
    """Thin, correctness-first wrapper around a HF causal LM."""

    def __init__(
        self,
        model,
        tokenizer=None,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        record_hidden: bool = False,
        native_rms_norm: bool = False,
    ) -> None:
        from dart.models.rms_norm import TorchRMSNorm, enable_native_rms_norm

        if type(native_rms_norm) is not bool:
            raise ValueError("native_rms_norm must be a boolean")
        if native_rms_norm and dtype != torch.float32:
            raise ValueError("native RMSNorm requires float32")
        self.model = model.to(device=device, dtype=dtype)
        if native_rms_norm:
            enable_native_rms_norm(self.model)
        self.native_rms_norm_count = sum(isinstance(module, TorchRMSNorm) for module in self.model.modules())
        self.model.eval()
        self.model.requires_grad_(False)
        self.tokenizer = tokenizer
        self.device = device
        self.dtype = dtype
        self.eos_token_ids = self._collect_eos_ids()
        # Opt-in tap for drafters that condition on target hidden states
        # (MTP heads, trained diffusion drafters).  When enabled, the final
        # layer's hidden states of the most recent forward are kept in
        # ``last_hidden_block`` ([T, hidden_size], detached).  Consumers
        # index it by *accepted* position - the tail rows of a verify pass
        # belong to rejected drafts and are off-trajectory.
        self.record_hidden = record_hidden
        self.last_hidden_block: Optional[Tensor] = None
        try:
            parameter = inspect.signature(self.model.forward).parameters.get("logits_to_keep")
            self._last_prefill_logits = (
                torch.device(self.device).type == "cpu" and parameter is not None  # Only CPU showed a consistent gain.
                and parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
            )
        except (TypeError, ValueError):
            self._last_prefill_logits = False

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        device: str = "cpu",
        dtype: Optional[torch.dtype] = None,
        attn_implementation: str = "eager",
        trust_remote_code: bool = False,
        record_hidden: bool = False,
        local_files_only: bool = False,
        native_rms_norm: bool = False,
    ) -> "TargetModel":
        if dtype is None:
            dtype = torch.float32 if device == "cpu" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            name_or_path,
            torch_dtype=dtype,
            attn_implementation=attn_implementation,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            name_or_path, trust_remote_code=trust_remote_code, local_files_only=local_files_only
        )
        return cls(model, tokenizer, device=device, dtype=dtype, record_hidden=record_hidden,
                   native_rms_norm=native_rms_norm)

    def _collect_eos_ids(self) -> frozenset[int]:
        ids: set[int] = set()
        if self.tokenizer is not None and self.tokenizer.eos_token_id is not None:
            ids.add(int(self.tokenizer.eos_token_id))
        gen_eos = getattr(self.model.generation_config, "eos_token_id", None)
        if gen_eos is None:
            gen_eos = getattr(self.model.config, "eos_token_id", None)
        if gen_eos is not None:
            if isinstance(gen_eos, int):
                ids.add(gen_eos)
            else:
                ids.update(int(t) for t in gen_eos)
        return frozenset(ids)

    # ------------------------------------------------------------------ #
    # Core forward primitives
    # ------------------------------------------------------------------ #
    def _check_context_length(self, length: int) -> None:
        limit = getattr(self.model.config, "max_position_embeddings", None)
        if limit is not None and length > limit:
            raise ValueError(f"sequence length {length} exceeds context limit {limit}")

    def validate_generation(self, prompt_len: int, max_new_tokens: int) -> None:
        if prompt_len < 1:
            raise ValueError("prompt must contain at least one token")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 0:
            raise ValueError("max_new_tokens must be an integer >= 0")
        self._check_context_length(prompt_len + max_new_tokens)

    def validate_prompt(self, prompt_ids: list[int] | Tensor) -> None:
        """Check prompt admission without executing the model."""
        ids = self._ids(prompt_ids)
        self.validate_generation(ids.size(1), 0)
        self._check_vocabulary(ids)

    def _check_vocabulary(self, ids: Tensor) -> None:
        low, high = torch.aminmax(ids)
        if low.item() < 0 or high.item() >= self.model.config.vocab_size:
            raise ValueError("token id is outside the model vocabulary")

    def _ids(self, tokens: list[int] | Tensor) -> Tensor:
        if isinstance(tokens, Tensor):
            t = tokens.to(device=self.device, dtype=torch.long)
        else:
            t = torch.tensor(tokens, device=self.device, dtype=torch.long)
        if t.dim() == 1:
            t = t.unsqueeze(0)
        if t.dim() != 2 or t.size(0) != 1:
            raise ValueError(f"expected [T] or [1, T] token ids, got {tuple(t.shape)}")
        return t

    @torch.inference_mode()
    def prefill(
        self, prompt_ids: list[int] | Tensor, cache: Optional[DynamicCache] = None
    ) -> tuple[TargetState, Tensor]:
        """Run the prompt; return (state covering the prompt, logits [V]).

        ``cache`` lets draft lanes substitute a compressed cache (e.g.
        QuantizedKVCache) while reusing the same forward plumbing.
        """
        input_ids = self._ids(prompt_ids)
        if input_ids.size(1) == 0:
            raise ValueError("prompt must contain at least one token")
        self._check_vocabulary(input_ids)  # Out-of-range IDs assert inside CUDA embedding kernels.
        state = TargetState(cache=cache if cache is not None else DynamicCache())
        logits = self._forward(state, input_ids, last_logits_only=True)
        return state, logits[-1]

    @torch.inference_mode()
    def forward_block(self, state: TargetState, tokens: list[int] | Tensor) -> Tensor:
        """Feed T tokens after ``state``; return logits [T, V].

        ``logits[i]`` is the target's next-token distribution after consuming
        tokens[0..i].  The KV state is extended by T entries.
        """
        return self._forward(state, self._ids(tokens))

    def _forward(self, state: TargetState, input_ids: Tensor, *, last_logits_only: bool = False) -> Tensor:
        state._check_usable()
        past_len = state.seq_len
        total_len = past_len + input_ids.size(1)
        self._check_context_length(total_len)
        attention_mask = torch.ones(
            (1, total_len), device=self.device, dtype=torch.long
        )
        logits_options = {"logits_to_keep": 1} if last_logits_only and self._last_prefill_logits else {}
        return self._run_model(
            state,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **logits_options,
        )

    def _run_model(self, state: TargetState, **kwargs) -> Tensor:
        try:
            out = self.model(past_key_values=state.cache, use_cache=True,
                             output_hidden_states=self.record_hidden, **kwargs)
            state.cache = out.past_key_values  # Some models return a new cache.
            if self.record_hidden:
                self.last_hidden_block = out.hidden_states[-1][0].detach()
            return out.logits[0].float()
        except BaseException:
            state._failed = True
            self.last_hidden_block = None
            raise

    @torch.inference_mode()
    def forward_tree(
        self,
        state: TargetState,
        pending: list[int],
        tree_tokens: list[int],
        parent: list[int],
    ) -> Tensor:
        """Feed ``pending`` plus a packed draft tree in one pass; return
        logits [len(pending) + len(tree_tokens), V].

        ``parent[i]`` is the in-tree parent of packed node i (-1 = child of
        the committed prefix).  Each node attends to the full cache, the
        whole pending block, its ancestor chain, and itself; its position
        id is depth-based, so its logits row matches what a linear forward
        over that single branch would produce.  The KV cache is extended by
        one entry per fed token *in packed order* — callers must fix the
        tail up afterwards (``gather_tree_tail``).

        Requires a model that honors custom 4D attention masks and explicit
        ``position_ids`` (e.g. Llama-family, eager attention).
        """
        state._check_usable()
        p_len, n = len(pending), len(tree_tokens)
        if p_len == 0:
            raise ValueError("pending block must contain at least one token")
        if len(parent) != n:
            raise ValueError("tree_tokens and parent must have equal length")
        base_len = state.seq_len
        q_len = p_len + n
        kv_len = base_len + q_len
        depths: list[int] = []
        for i, p in enumerate(parent):
            if p < -1 or p >= i:
                raise ValueError("tree parents must be -1 or a preceding index")
            depths.append(0 if p < 0 else depths[p] + 1)
        self._check_context_length(base_len + p_len + max(depths, default=-1) + 1)

        allow = torch.zeros(q_len, kv_len, dtype=torch.bool, device=self.device)
        allow[:, :base_len] = True  # everything sees the committed cache
        for i in range(p_len):      # pending block: ordinary causal
            allow[i, base_len : base_len + i + 1] = True
        for i in range(n):
            row = p_len + i
            allow[row, base_len : base_len + p_len] = True  # full pending
            allow[row, base_len + row] = True               # itself
            j = parent[i]
            while j >= 0:
                allow[row, base_len + p_len + j] = True     # ancestors
                j = parent[j]

        mask = torch.zeros(
            (1, 1, q_len, kv_len), dtype=self.dtype, device=self.device
        )
        mask.masked_fill_(~allow, torch.finfo(self.dtype).min)

        positions = list(range(base_len, base_len + p_len)) + [
            base_len + p_len + d for d in depths
        ]
        position_ids = torch.tensor([positions], device=self.device)

        return self._run_model(
            state,
            input_ids=self._ids(pending + list(tree_tokens)),
            attention_mask=mask,
            position_ids=position_ids,
        )

    def gather_tree_tail(
        self, state: TargetState, keep_len: int, path: list[int]
    ) -> None:
        """Rebuild the cache tail after a tree forward: keep the first
        ``keep_len`` entries (prefix + pending) plus the packed-tree entries
        at offsets ``path`` (the accepted branch, in order), drop the rest.

        Valid because a packed node's K/V depend only on tokens its mask
        let it attend to — the prefix and its ancestors — so the surviving
        entries are exactly those of a linear forward over that branch.
        Cropping a contiguous prefix avoids a copy but retains its backing allocation.
        """
        state._check_usable()
        current_length = state.seq_len
        if type(keep_len) is not int or not 0 <= keep_len <= current_length:
            raise ValueError("keep_len must be an integer within the current cache")
        if not isinstance(path, (list, tuple)) or any(
            type(offset) is not int or not 0 <= offset < current_length - keep_len for offset in path
        ):
            raise ValueError("path must contain integer offsets within the tree tail")
        if all(offset == index for index, offset in enumerate(path)):
            self.crop(state, keep_len + len(path))
            return
        idx = torch.tensor(
            list(range(keep_len)) + [keep_len + p for p in path],
            device=self.device,
            dtype=torch.long,
        )
        cache = state.cache
        try:
            layers = getattr(cache, "layers", None)
            if layers is not None:  # transformers >= 4.54 layout
                for layer in layers:
                    layer.keys = layer.keys.index_select(2, idx)
                    layer.values = layer.values.index_select(2, idx)
            else:  # legacy list-of-tensors layout
                for i in range(len(cache.key_cache)):
                    cache.key_cache[i] = cache.key_cache[i].index_select(2, idx)
                    cache.value_cache[i] = cache.value_cache[i].index_select(2, idx)
        except BaseException:
            state._failed = True
            self.last_hidden_block = None
            raise

    @torch.inference_mode()
    def all_logits(self, tokens: list[int] | Tensor) -> Tensor:
        """One throwaway full forward over ``tokens``; return logits [T, V].

        Used by precision policies that need per-position predictive entropy
        (docs/design.md §9.4).  Runs on a fresh cache so it never disturbs a
        drafter's incremental state; row ``i`` is the distribution after
        consuming ``tokens[:i+1]``.
        """
        return self._forward(TargetState(cache=DynamicCache()), self._ids(tokens))

    def crop(self, state: TargetState, length: int) -> None:
        """Roll the KV state back so it covers only the first ``length`` tokens."""
        state._check_usable()
        current_length = state.seq_len
        if type(length) is not int:
            raise ValueError("cache length must be an integer")
        if length < 0 or length > current_length:
            raise ValueError(
                f"cannot crop cache of length {current_length} to {length}"
            )
        if length < current_length:
            try:
                state.cache.crop(length - current_length)  # HF accepts negative removal counts.
            except BaseException:
                state._failed = True
                self.last_hidden_block = None
                raise

    # ------------------------------------------------------------------ #
    # Trusted baseline
    # ------------------------------------------------------------------ #
    @torch.inference_mode()
    def greedy_generate(
        self, prompt_ids: list[int] | Tensor, max_new_tokens: int
    ) -> list[int]:
        """Full-KV incremental greedy decoding: the exactness reference.

        This is the K=0 decode loop: one forward per token, argmax commit,
        stop at EOS or the token budget.
        """
        input_ids = self._ids(prompt_ids)
        self.validate_generation(input_ids.size(1), max_new_tokens)
        if max_new_tokens == 0:
            self.validate_prompt(input_ids)
            return []
        state, logits = self.prefill(input_ids)
        generated: list[int] = []
        next_token = int(torch.argmax(logits).item())
        for _ in range(max_new_tokens):
            generated.append(next_token)
            if next_token in self.eos_token_ids:
                break
            if len(generated) == max_new_tokens:
                break
            step_logits = self.forward_block(state, [next_token])
            next_token = int(torch.argmax(step_logits[-1]).item())
        return generated
