import gc
import weakref
from importlib.metadata import version

import pytest
import torch
from packaging.version import Version
from transformers import LlamaConfig, LlamaForCausalLM

from dart.models import replay
from dart.models.replay import CudaGraphTarget
from dart.models.target import TargetModel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or Version(version("transformers")) < Version("5.16.1"),
                                reason="requires CUDA and transformers >=5.16.1")


def base_model(seed):
    torch.manual_seed(seed)
    config = LlamaConfig(vocab_size=97, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                         num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=64,
                         eos_token_id=None, pad_token_id=0, attention_dropout=0.0)
    config._attn_implementation = "eager"
    return TargetModel(LlamaForCausalLM(config), device="cuda")


def native(base, prompt, count):
    ids = torch.tensor([prompt], device=base.device)
    with torch.inference_mode():
        return base.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=count,
                                   do_sample=False, eos_token_id=None, use_cache=True)[0, len(prompt):].tolist()


def check_state(base, target, state, tokens, follow):
    expected, _ = base.prefill(tokens)
    assert state.seq_len == expected.seq_len
    for index, layer in enumerate(expected.cache.layers):
        torch.testing.assert_close(state.cache.storage[index, 0, ..., :state.seq_len, :], layer.keys, atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(state.cache.storage[index, 1, ..., :state.seq_len, :], layer.values, atol=2e-5, rtol=2e-4)
    assert target.forward_predictions(state, [follow]) == base.forward_block(expected, [follow]).argmax(-1).tolist()


def test_repeated_preparation_keeps_allocation_and_streams_bounded():
    base = base_model(617)
    expected = native(base, [3, 5, 7], 11)
    allocations, stream_sets = [], []
    for _ in range(5):
        target = CudaGraphTarget(base, capacity=64, max_block_tokens=9, capture_predictions=True)
        refs = [weakref.ref(target._workspace)] + [weakref.ref(block.graph) for block in target._blocks.values()]
        stream_sets.append({block._capture_stream.cuda_stream for block in target._blocks.values()})
        try:
            assert target.greedy_generate([3, 5, 7], 11) == expected
        finally:
            target.close()
        gc.collect()
        torch.cuda.synchronize()
        assert all(ref() is None for ref in refs)
        allocations.append(torch.cuda.memory_allocated())
    assert len(stream_sets[0]) == 9
    assert all(streams == stream_sets[0] for streams in stream_sets)
    assert max(allocations) - min(allocations) <= 65536


def test_live_targets_and_greedy_chunks_keep_independent_workspaces():
    first_base, second_base = base_model(617), base_model(29)
    first_truth = native(first_base, [3, 5, 7], 9)
    second_truth = native(second_base, [11, 13, 17], 9)
    side = torch.cuda.Stream()
    first = CudaGraphTarget(first_base, capacity=64, max_block_tokens=3,
                            capture_predictions=True, greedy_chunk_sizes=(2, 4))
    with torch.cuda.stream(side):
        second = CudaGraphTarget(second_base, capacity=64, max_block_tokens=3,
                                 capture_predictions=True, greedy_chunk_sizes=(2, 4))
    third = None
    try:
        first_streams = {block._capture_stream.cuda_stream for block in [*first._blocks.values(), *first._greedy_blocks.values()]}
        second_streams = {block._capture_stream.cuda_stream for block in [*second._blocks.values(), *second._greedy_blocks.values()]}
        assert first_streams.isdisjoint(second_streams)
        a, _ = first.prefill([3, 5, 7])
        with torch.cuda.stream(side):
            b, _ = second.prefill([11, 13, 17])
        first_logits = first.forward_block(a, [19, 23])
        with torch.cuda.stream(side):
            second_logits = second.forward_block(b, [29, 31])
        torch.cuda.synchronize()
        ar, _ = first_base.prefill([3, 5, 7])
        br, _ = second_base.prefill([11, 13, 17])
        torch.testing.assert_close(first_logits, first_base.forward_block(ar, [19, 23]), atol=2e-5, rtol=2e-4)
        torch.testing.assert_close(second_logits, second_base.forward_block(br, [29, 31]), atol=2e-5, rtol=2e-4)
        check_state(first_base, first, a, [3, 5, 7, 19, 23], 37)
        with torch.cuda.stream(side):
            check_state(second_base, second, b, [11, 13, 17, 29, 31], 41)
            assert second.greedy_generate([11, 13, 17], 9) == second_truth
        assert first.greedy_generate([3, 5, 7], 9) == first_truth
        first.close()
        third = CudaGraphTarget(first_base, capacity=64, max_block_tokens=3, capture_predictions=True)
        assert {block._capture_stream.cuda_stream for block in third._blocks.values()}.isdisjoint(second_streams)
        assert third.greedy_generate([3, 5, 7], 9) == first_truth
        with torch.cuda.stream(side):
            assert second.greedy_generate([11, 13, 17], 9) == second_truth
            check_state(second_base, second, b, [11, 13, 17, 29, 31, 41], 43)
    finally:
        first.close()
        second.close()
        if third is not None:
            third.close()


def test_partial_capture_preparation_releases_completed_blocks(monkeypatch):
    base = base_model(617)
    pool = replay._capture_pool(next(base.model.parameters()).device)
    active = set(pool.active)
    original = replay._ReplayBlock.forward

    def fail(self):
        if self.ids.shape[1] == 2:
            raise RuntimeError("warmup failed")
        return original(self)

    with monkeypatch.context() as patch:
        patch.setattr(replay._ReplayBlock, "forward", fail)
        with pytest.raises(RuntimeError, match="warmup failed"):
            CudaGraphTarget(base, capacity=64, max_block_tokens=3, capture_predictions=True)
    assert pool.active == active
    target = CudaGraphTarget(base, capacity=64, max_block_tokens=3, capture_predictions=True)
    try:
        assert target.greedy_generate([3, 5, 7], 7) == native(base, [3, 5, 7], 7)
    finally:
        target.close()


def test_failed_graph_reset_is_not_recycled(monkeypatch):
    base = base_model(617)
    first = CudaGraphTarget(base, capacity=64, max_block_tokens=3, capture_predictions=True)
    failed = first._blocks[2]
    stream = failed._capture_stream.cuda_stream
    graph = failed.graph
    original = graph.reset
    second = None

    def fail():
        raise RuntimeError("reset failed")

    try:
        monkeypatch.setattr(graph, "reset", fail)
        with pytest.raises(RuntimeError, match="reset failed"):
            first.close()
        second = CudaGraphTarget(base, capacity=64, max_block_tokens=1, capture_predictions=True)
        assert second._blocks[1]._capture_stream.cuda_stream != stream
        assert second.greedy_generate([3, 5, 7], 7) == native(base, [3, 5, 7], 7)
        monkeypatch.setattr(graph, "reset", original)
        first.close()
        assert second.greedy_generate([3, 5, 7], 7) == native(base, [3, 5, 7], 7)
    finally:
        monkeypatch.setattr(graph, "reset", original)
        first.close()
        if second is not None:
            second.close()
