from types import SimpleNamespace

import pytest
import torch

from dart.models.replay import CudaGraphTarget
from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference
from dart.verification.state_checks import _cache_failures, _reference_forward
from dart.verification.synthetic import _synthetic_models


class CountingTarget:
    def __init__(self, eos, compact):
        self.eos_token_ids = frozenset(eos)
        self.capture_predictions = compact
        self.model = SimpleNamespace(config=SimpleNamespace(vocab_size=67))
        self.calls = []

    def forward_block(self, state, tokens):
        self.calls.append(list(tokens))
        state.seq_len += len(tokens)
        logits = torch.full((len(tokens), 67), -8.)
        logits[torch.arange(len(tokens)), [(token + 1) % 67 for token in tokens]] = 8.
        return logits

    def forward_predictions(self, state, tokens):
        return self.forward_block(state, tokens).argmax(-1).tolist()

    def crop(self, state, length):
        assert length <= state.seq_len
        state.seq_len = length


@pytest.mark.parametrize('eos_index', [0, 2, 4])
@pytest.mark.parametrize('mismatch', [None, 0, 1])
@pytest.mark.parametrize('pending', [[10], [7, 9, 10]])
@pytest.mark.parametrize('compact,entropy', [(False, True), (True, False)])
def test_eos_work_matches_sequential_verification(eos_index, mismatch, pending, compact, entropy):
    drafts = [11, 12, 13, 14, 15]
    eos = drafts[eos_index]
    if mismatch is not None and mismatch < eos_index:
        drafts[mismatch] = 50
    target = CountingTarget([eos, 66], compact)
    reference = CountingTarget([eos, 66], False)
    state, wanted_state = SimpleNamespace(seq_len=3), SimpleNamespace(seq_len=3)
    original = list(drafts)
    wanted = verify_greedy_reference(reference, wanted_state, pending, drafts)
    result = verify_block_greedy(target, state, pending, drafts, compute_entropy=entropy)
    assert result == wanted
    assert state.seq_len == wanted_state.seq_len
    assert drafts == original and result.n_draft_proposed == 5
    assert result.hidden_index == len(pending) + len(result.committed) - 2
    assert target.calls == [pending + drafts[:eos_index]]
    if result.finished or not entropy:
        assert result.entropy == 0
    else:
        assert result.entropy > 0


@pytest.mark.parametrize('drafts,eos', [([], [2]), ([11, 12, 13], []), ([11, 12, 13], [2])])
def test_no_eos_keeps_full_block(drafts, eos):
    target = CountingTarget(eos, False)
    state = SimpleNamespace(seq_len=3)
    result = verify_block_greedy(target, state, 10, drafts)
    assert target.calls == [[10] + drafts]
    assert result.committed == drafts + [11 + len(drafts)]


@pytest.fixture
def models():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield _synthetic_models(17)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('execution', ['eager', 'static', 'predictions'])
@pytest.mark.parametrize('mismatch', [False, True])
@pytest.mark.parametrize('entropy', [False, True])
@torch.inference_mode()
def test_trimmed_round_matches_native_cache_and_continuation(models, execution, mismatch, entropy, monkeypatch):
    base, reference = models
    prompt = [5, 7, 9]
    oracle, prefix = [], list(prompt)
    for _ in range(6):
        token = int(_reference_forward(reference, prefix).logits[0, -1].argmax())
        oracle.append(token)
        prefix.append(token)
    target = base if execution == 'eager' else CudaGraphTarget(
        base, capacity=64, max_block_tokens=6, replay=False,
        capture_predictions=execution == 'predictions',
    )
    try:
        target.eos_token_ids = frozenset([oracle[3]])
        if execution == 'eager':
            target.record_hidden = True
        drafts = oracle[1:]
        if mismatch:
            drafts[0] = (oracle[1] + 1) % 67
            target.eos_token_ids = frozenset([drafts[0]])
        state, _ = target.prefill(prompt)
        rows = []
        method = 'forward_predictions' if execution == 'predictions' and not entropy else 'forward_block'
        original = getattr(target, method)

        def observe(state, tokens):
            rows.append(len(tokens))
            return original(state, tokens)

        monkeypatch.setattr(target, method, observe)
        result = verify_block_greedy(target, state, oracle[0], drafts, compute_entropy=entropy)
        eos_index = next(i for i, token in enumerate(drafts) if token in target.eos_token_ids)
        expected = [oracle[1]] if mismatch else drafts[:eos_index + 1]
        assert result.committed == expected
        assert result.finished is (not mismatch)
        assert rows == [1 + eos_index]
        live = prompt + [oracle[0]] + result.committed[:-1]
        wanted = _reference_forward(reference, live)
        assert not _cache_failures(state, wanted.past_key_values, len(live), atol=2e-6, rtol=2e-5)
        if execution == 'eager':
            hidden = reference(
                input_ids=torch.tensor([live]), attention_mask=torch.ones((1, len(live)), dtype=torch.long),
                use_cache=False, output_hidden_states=True,
            ).hidden_states[-1][0, -1]
            torch.testing.assert_close(target.last_hidden_block[result.hidden_index], hidden, atol=2e-6, rtol=2e-5)
        continued = target.forward_block(state, [result.committed[-1]])[-1]
        next_expected = _reference_forward(reference, live + [result.committed[-1]])
        torch.testing.assert_close(continued, next_expected.logits[0, -1], atol=2e-6, rtol=2e-5)
        assert int(continued.argmax()) == int(next_expected.logits[0, -1].argmax())
        assert not _cache_failures(state, next_expected.past_key_values, len(live) + 1, atol=2e-6, rtol=2e-5)
    finally:
        if execution != 'eager':
            target.close()


@pytest.mark.parametrize('bad', [-1, 67, 2**80, 67.9])
@torch.inference_mode()
def test_invalid_suffix_after_eos_is_still_rejected(models, monkeypatch, bad):
    target, reference = models
    target.eos_token_ids = frozenset([2])
    state, _ = target.prefill([5, 7, 9])
    with monkeypatch.context() as patch:
        patch.setattr(target.model, 'forward', lambda **kwargs: pytest.fail('invalid draft reached the model'))
        with pytest.raises(ValueError, match='outside the model vocabulary'):
            verify_block_greedy(target, state, 10, [2, bad])
    assert not state._failed and state.seq_len == 3
    actual = target.forward_block(state, [10])[-1]
    expected = _reference_forward(reference, [5, 7, 9, 10]).logits[0, -1]
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize('suffix', [[3.5], [True], [torch.tensor(4)]])
@torch.inference_mode()
def test_converted_suffix_keeps_existing_execution(models, monkeypatch, suffix):
    target, _ = models
    target.eos_token_ids = frozenset([2])
    state, _ = target.prefill([5, 7, 9])
    original, calls = target.forward_block, []

    def observe(state, tokens):
        calls.append(tokens)
        return original(state, tokens)

    monkeypatch.setattr(target, 'forward_block', observe)
    verify_block_greedy(target, state, 10, [2] + suffix)
    assert len(calls) == 1 and len(calls[0]) == 3
