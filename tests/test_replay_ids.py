from types import SimpleNamespace

import pytest
import torch

from dart.models.replay import CudaGraphTarget


@pytest.fixture
def target():
    result = object.__new__(CudaGraphTarget)
    result.device, result.max_block_tokens = "cpu", 5
    result.model = SimpleNamespace(config=SimpleNamespace(vocab_size=100))
    return result


@pytest.mark.parametrize("tokens", [[0], [0, 99], (1, 2, 3, 4, 5)])
def test_short_integer_blocks_do_not_need_tensor_reductions(target, monkeypatch, tokens):
    def unexpected(*args, **kwargs):
        raise AssertionError("unnecessary tensor reduction")

    monkeypatch.setattr(torch, "aminmax", unexpected)
    result = target._ids(tokens)
    assert result.dtype == torch.long and result.device.type == "cpu"
    assert result.tolist() == [list(tokens)]


@pytest.mark.parametrize("tokens", [[-1], [100], (0, 100), [2**80]])
def test_bad_python_ids_fail_before_allocation(target, monkeypatch, tokens):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid IDs reached tensor allocation")

    monkeypatch.setattr(torch, "as_tensor", unexpected)
    monkeypatch.setattr(torch, "tensor", unexpected)
    with pytest.raises(ValueError, match="outside the model vocabulary"):
        target._ids(tokens)


@pytest.mark.parametrize("tokens,expected", [
    ([1.9, 2.2], [[1, 2]]), ([True, False], [[1, 0]]), ([[1, 2]], [[1, 2]]),
    (list(range(6)), [list(range(6))]), (torch.tensor([3, 4]), [[3, 4]]),
])
def test_other_supported_inputs_keep_their_conversion(target, tokens, expected):
    assert target._ids(tokens).tolist() == expected


@pytest.mark.parametrize("tokens", [[], (), [[1], [2]], [[[]]], torch.tensor([])])
def test_invalid_shapes_still_fail(target, tokens):
    with pytest.raises(ValueError, match="nonempty"):
        target._ids(tokens)
