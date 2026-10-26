"""The same replay executor behind HF's greedy and prompt-lookup loops."""

import copy
from functools import wraps

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from dart.models.replay import CudaGraphTarget
from dart.models.target import TargetState


def build_replay_generate(target: CudaGraphTarget, settings: dict):
    """Build the unpadded, batch-one HF benchmark path without copying weights."""
    if not isinstance(target, CudaGraphTarget):
        raise ValueError("HF replay requires a CudaGraphTarget")
    settings = copy.deepcopy(settings)
    if settings.get("do_sample") or settings.get("num_beams", 1) != 1 or settings.get("num_return_sequences", 1) != 1:
        raise ValueError("HF replay benchmarks require single-sequence greedy decoding")
    if settings.get("output_hidden_states") or settings.get("output_attentions") or not settings.get("use_cache", True):
        raise ValueError("HF replay requires caching and no hidden-state or attention recording")
    if settings.get("cache_implementation") is not None:
        raise ValueError("HF replay owns its cache implementation")
    lookup = settings.get("prompt_lookup_num_tokens") or 0
    if settings["max_new_tokens"] > 1 and lookup + 1 > target.max_block_tokens:
        raise ValueError("HF prompt lookup exceeds the prepared replay block size")

    model = copy.copy(target.model)  # Only the top-level wrapper is copied; weights stay shared.
    model.generation_config = copy.deepcopy(target.model.generation_config)
    original_forward = target.model.forward

    @wraps(original_forward)
    def forward(input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                use_cache=True, logits_to_keep=0, **kwargs):
        if input_ids is None or input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] == 0:
            raise ValueError("HF replay requires nonempty batch-one token ids")
        if not use_cache or kwargs.get("inputs_embeds") is not None or kwargs.get("labels") is not None:
            raise ValueError("HF replay only supports cached token-id inference")
        if kwargs.get("output_hidden_states") or kwargs.get("output_attentions"):
            raise ValueError("HF replay does not record hidden states or attentions")
        if type(logits_to_keep) is not int or logits_to_keep < 0:
            raise ValueError("HF replay expects an integer logits_to_keep")
        cached = past_key_values.get_seq_length() if past_key_values is not None else 0
        total = cached + input_ids.shape[1]
        target._check_context_length(total)
        if attention_mask is None or attention_mask.shape != (1, total):
            raise ValueError("HF replay requires the full unpadded attention mask")
        if cached == 0:
            return original_forward(input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids,
                                    past_key_values=past_key_values, use_cache=use_cache,
                                    logits_to_keep=logits_to_keep, **kwargs)
        state = TargetState(past_key_values)
        logits = target._forward(state, input_ids)  # Generated IDs already passed HF's vocabulary selection.
        if logits_to_keep:
            logits = logits[-logits_to_keep:]
        return CausalLMOutputWithPast(logits=logits[None], past_key_values=state.cache)

    model.forward = forward

    @torch.inference_mode()
    def generate(prompt):
        target.validate_generation(len(prompt), settings["max_new_tokens"])
        if lookup and settings["max_new_tokens"] > 1:
            target.validate_generation(len(prompt), settings["max_new_tokens"] + lookup - 1)
        ids = target._ids(prompt)
        with target._lock:
            target._check_ready()
            return model.generate(ids, attention_mask=torch.ones_like(ids), **settings)

    return generate
