import copy
import gc
import weakref
from contextlib import closing

import pytest
import torch

from dart.models.replay import CudaGraphTarget, ReplayCache
from dart.models.rms_norm import TorchRMSNorm
from dart.models.target import TargetModel
from dart.verification.differential import GenerationCase, hf_decoder


def change(model, setting, patch):
    layer = model.model.layers[0]
    choices = {
        "epsilon": (layer.input_layernorm, "variance_epsilon", 1.0),
        "scale": (layer.self_attn, "scaling", 4.0),
        "rope_scale": (model.model.rotary_emb, "attention_scaling", 2.0),
        "dropout": (layer.self_attn, "attention_dropout", 0.25),
        "training": (layer.self_attn, "training", True),
        "head_dim": (layer.self_attn, "head_dim", 8),
        "groups": (layer.self_attn, "num_key_value_groups", 1),
        "activation": (layer.mlp, "act_fn", torch.nn.Tanh().eval()),
    }
    if setting == "norm":
        patch.setitem(layer._modules, "input_layernorm", TorchRMSNorm(layer.input_layernorm))
    else:
        obj, name, value = choices[setting]
        if setting == "activation":
            patch.setitem(obj._modules, name, value)  # Restore the registered layer on undo.
        else:
            patch.setattr(obj, name, value)


def values(state):
    if isinstance(state.cache, ReplayCache):
        return tuple((row[0, ..., :state.seq_len, :], row[1, ..., :state.seq_len, :]) for row in state.cache.storage)
    return tuple((layer.keys, layer.values) for layer in state.cache.layers)


@pytest.mark.parametrize("setting", ["epsilon", "scale", "rope_scale", "dropout", "training", "head_dim", "groups", "activation", "norm"])
@pytest.mark.parametrize("operation", ["prefill", "block", "predictions", "greedy_chunk"])
@torch.inference_mode()
def test_changed_layer_settings_reject_before_execution_and_preserve_state(tiny_target, monkeypatch, setting, operation):
    base = TargetModel(copy.deepcopy(tiny_target.model))
    reference = copy.deepcopy(base.model)
    with closing(CudaGraphTarget(base, capacity=32, max_block_tokens=3, replay=False,
                                  capture_predictions=True, greedy_chunk_sizes=(2,))) as target:
        state, _ = target.prefill([5, 6, 7])
        target.forward_block(state, [8])
        saved = [(a.clone(), b.clone()) for a, b in values(state)]
        storage = tuple(tensor.data_ptr() for tensor in (*base.model.parameters(), *base.model.buffers()))
        calls = []
        hook = base.model.register_forward_pre_hook(lambda *args: calls.append(True))
        for prepared in (*target._blocks.values(), *target._greedy_blocks.values()):
            def track(run=prepared.run):
                calls.append(True)
                return run()
            monkeypatch.setattr(prepared, "run", track)
        try:
            with monkeypatch.context() as patch:
                change(base.model, setting, patch)
                assert storage == tuple(tensor.data_ptr() for tensor in (*base.model.parameters(), *base.model.buffers()))
                with pytest.raises(ValueError, match="layer settings changed"):
                    if operation == "prefill":
                        target.prefill([9, 10])
                    elif operation == "block":
                        target.forward_block(state, [9])
                    elif operation == "predictions":
                        target.forward_predictions(state, [9])
                    else:
                        target.forward_greedy(state, 9, 2)
            assert calls == [] and state.seq_len == 4
            for actual, expected in zip(values(state), saved, strict=True):
                for a, b in zip(actual, expected, strict=True):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
        finally:
            hook.remove()
        actual = target.forward_block(state, [9, 10])
        assert calls
        expected = reference(torch.tensor([[5, 6, 7, 8, 9, 10]]), use_cache=True)
        torch.testing.assert_close(actual, expected.logits[0, -2:], rtol=1e-4, atol=1e-5)
        assert actual.argmax(-1).tolist() == expected.logits[0, -2:].argmax(-1).tolist()
        for (keys, vals), layer in zip(values(state), expected.past_key_values.layers, strict=True):
            torch.testing.assert_close(keys, layer.keys, rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(vals, layer.values, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("setting", ["epsilon", "normalized_shape"])
def test_native_norm_settings_are_guarded(tiny_target, monkeypatch, setting):
    base = TargetModel(copy.deepcopy(tiny_target.model), native_rms_norm=True)
    with closing(CudaGraphTarget(base, capacity=16, replay=False)) as target:
        norm = base.model.model.layers[0].input_layernorm
        with monkeypatch.context() as patch:
            if setting == "epsilon":
                patch.setattr(norm, "variance_epsilon", 1.0)
            else:
                patch.setattr(norm, "normalized_shape", (32,))
            with pytest.raises(ValueError, match="layer settings changed"):
                target.prefill([5, 6])
        assert target.greedy_generate([5, 6], 4) == base.greedy_generate([5, 6], 4)


@pytest.mark.parametrize("setting", ["epsilon", "scale", "rope_scale", "activation", "norm"])
def test_repreparing_changed_model_restores_supported_execution(tiny_target, monkeypatch, setting):
    base = TargetModel(copy.deepcopy(tiny_target.model))
    with closing(CudaGraphTarget(base, capacity=32, replay=False)) as old:
        change(base.model, setting, monkeypatch)
        with pytest.raises(ValueError, match="layer settings changed"):
            old.greedy_generate([5, 6, 7], 5)
        reference = TargetModel(copy.deepcopy(base.model))
        with closing(CudaGraphTarget(base, capacity=32, replay=False)) as current:
            expected = hf_decoder(reference.model)(GenerationCase("changed-settings", (5, 6, 7), 5, 3))
            assert current.greedy_generate([5, 6, 7], 5) == expected


def test_close_releases_replaced_module_registration(tiny_target):
    base = TargetModel(copy.deepcopy(tiny_target.model))
    target = CudaGraphTarget(base, capacity=16, replay=False)
    activation = weakref.ref(base.model.model.layers[0].mlp.act_fn)
    base.model.model.layers[0].mlp.act_fn = torch.nn.Tanh().eval()
    target.close()
    gc.collect()
    assert activation() is None


def test_unrelated_metadata_does_not_invalidate_preparation(tiny_target):
    base = TargetModel(copy.deepcopy(tiny_target.model))
    with closing(CudaGraphTarget(base, capacity=16, replay=False)) as target:
        base.model.model.layers[0].review_note = "example"
        base.model.config.name_or_path = "example"
        assert target.greedy_generate([5, 6, 7], 3) == base.greedy_generate([5, 6, 7], 3)


def test_tensor_setting_replacement_is_rejected_without_tensor_comparison(tiny_target, monkeypatch):
    base = TargetModel(copy.deepcopy(tiny_target.model))
    with closing(CudaGraphTarget(base, capacity=16, replay=False)) as target:
        base.model.model.layers[0].self_attn.scaling = torch.tensor(0.25)
        def forbidden(*args, **kwargs):
            raise AssertionError("guard compared a tensor value")
        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "__eq__", forbidden)
            patch.setattr(torch.Tensor, "__bool__", forbidden)
            with pytest.raises(ValueError, match="layer settings changed"):
                target.validate_prompt([5, 6])


def test_mutated_native_shape_list_cannot_change_prepared_dimensions(tiny_target):
    base = TargetModel(copy.deepcopy(tiny_target.model), native_rms_norm=True)
    norm = base.model.model.layers[0].input_layernorm
    norm.normalized_shape = list(norm.normalized_shape)
    with closing(CudaGraphTarget(base, capacity=16, replay=False)) as target:
        state, _ = target.prefill([5, 6])
        width = norm.normalized_shape[0]
        norm.normalized_shape[0] = width // 2
        with pytest.raises(ValueError, match="layer settings changed"):
            target.forward_block(state, [7])
        assert state.seq_len == 2
        norm.normalized_shape[0] = width
        assert torch.isfinite(target.forward_block(state, [7])).all()
