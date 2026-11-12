from __future__ import annotations

import random

import pytest
import torch

from dart.attention.verifier_mask import tree_mask
from tests.test_sampling_devices import DEVICES


def expected_mask(parent, prefix):
    rows = []
    for i in range(len(parent)):
        ancestors = set()
        node = i
        while node >= 0:
            ancestors.add(node)
            node = parent[node]
        rows.append([True] * prefix + [node in ancestors for node in range(len(parent))])
    return torch.tensor(rows, dtype=torch.bool).reshape(len(parent), prefix + len(parent))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("prefix", [0, 9, 1008])
@pytest.mark.parametrize("nodes", [0, 1, 4, 8, 16])
@pytest.mark.parametrize("shape", ["chain", "fan", "binary", "random"])
def test_tree_mask_matches_ancestor_oracle(device, prefix, nodes, shape):
    rng = random.Random(31)
    parents = {
        "chain": [i - 1 for i in range(nodes)],
        "fan": [-1] * nodes,
        "binary": [(i - 1) // 2 for i in range(nodes)],
        "random": [rng.randrange(-1, i) for i in range(nodes)],
    }[shape]
    before = parents.copy()
    actual = tree_mask(parents, prefix, device=device)
    assert actual.dtype == torch.bool
    assert actual.device.type == torch.device(device).type
    assert torch.equal(actual.cpu(), expected_mask(parents, prefix))
    assert parents == before
    actual.fill_(False)
    assert torch.equal(tree_mask(parents, prefix, device=device).cpu(), expected_mask(parents, prefix))


@pytest.mark.parametrize("parent", [[-2], [-1, -2]])
def test_tree_mask_rejects_negative_parent(parent):
    with pytest.raises(ValueError, match="parents"):
        tree_mask(parent, 3)


@pytest.mark.parametrize("parent", [[0], [1, -1], [-1, 1], [-1, True], [-1, 0.0]])
def test_tree_mask_rejects_non_topological_parent_before_allocation(monkeypatch, parent):
    def fail(*args, **kwargs):
        pytest.fail("invalid parent reached tensor allocation")

    monkeypatch.setattr(torch, "ones", fail)
    with pytest.raises(ValueError, match="parents"):
        tree_mask(parent, 3)


@pytest.mark.parametrize("prefix", [-1, True, 1.5])
def test_tree_mask_rejects_invalid_prefix(prefix):
    with pytest.raises(ValueError, match="prefix_len"):
        tree_mask([-1], prefix)


def test_tree_mask_uses_explicit_device_and_boolean_dtype():
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        with torch.device("meta"):
            actual = tree_mask([-1, 0, 0], 2, device="cpu")
        assert actual.device.type == "cpu" and actual.dtype == torch.bool
        assert torch.equal(actual, expected_mask([-1, 0, 0], 2))
    finally:
        torch.set_default_dtype(previous)


@pytest.mark.parametrize("pending", [[11], [11, 13, 17]])
@pytest.mark.parametrize("parents", [[], [-1], [-1, 0, 1], [-1, -1, -1], [-1, 0, 0, 2, -1]])
def test_tree_forward_preserves_pending_causality_and_branch_isolation(tiny_target, monkeypatch, pending, parents):
    state, _ = tiny_target.prefill([5, 7, 9])
    base = state.seq_len
    captured = {}

    def capture(state, **kwargs):
        captured.update(kwargs)
        return torch.empty(0)

    monkeypatch.setattr(tiny_target, "_run_model", capture)
    tiny_target.forward_tree(state, pending, [19] * len(parents), parents)
    size = len(pending) + len(parents)
    expected = torch.zeros(size, base + size, dtype=torch.bool)
    for i in range(len(pending)):
        expected[i, :base + i + 1] = True
    expected[len(pending):] = expected_mask(parents, base + len(pending))
    mask = captured["attention_mask"]
    assert mask.shape == (1, 1, size, base + size)
    assert mask.dtype == tiny_target.dtype
    assert torch.equal(mask[0, 0] == 0, expected)
    assert (mask[0, 0][~expected] == torch.finfo(mask.dtype).min).all()
    depths = []
    for i in range(len(parents)):
        depth, node = 0, parents[i]
        while node >= 0:
            depth += 1
            node = parents[node]
        depths.append(depth)
    assert captured["position_ids"].tolist() == [
        list(range(base, base + len(pending))) + [base + len(pending) + depth for depth in depths]
    ]
    assert state.seq_len == base
