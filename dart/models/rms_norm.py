from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.llama.modeling_llama import LlamaRMSNorm


class TorchRMSNorm(nn.Module):
    """PyTorch RMSNorm with the original float32 weight and epsilon."""

    def __init__(self, source: LlamaRMSNorm):
        super().__init__()
        self.weight = source.weight
        self.variance_epsilon = source.variance_epsilon
        self.normalized_shape = tuple(source.weight.shape)
        self.train(source.training)

    def forward(self, hidden_states):
        if hidden_states.dtype != torch.float32 or self.weight.dtype != torch.float32:
            raise ValueError("native RMSNorm requires float32 inputs and weights")
        return F.rms_norm(hidden_states, self.normalized_shape, self.weight, self.variance_epsilon)


def enable_native_rms_norm(model: nn.Module) -> None:
    """Replace known Llama norms without changing parameter names or values."""
    if not callable(getattr(F, "rms_norm", None)):
        raise ValueError("this PyTorch version does not provide rms_norm")
    norms = [(name, module) for name, module in model.named_modules() if type(module) is LlamaRMSNorm]
    if not norms and not any(isinstance(module, TorchRMSNorm) for module in model.modules()):
        raise ValueError("native RMSNorm requires LlamaRMSNorm layers")
    if any(module.weight.dtype != torch.float32 for _, module in norms):
        raise ValueError("native RMSNorm requires float32 weights")
    for name, module in norms:
        if not name:
            raise ValueError("pass the model containing the RMSNorm layers")
        model.set_submodule(name, TorchRMSNorm(module))
