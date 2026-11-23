from types import SimpleNamespace

import pytest
import torch

from dart.models.replay import CudaGraphTarget
from dart.verification.block_verify import verify_block_greedy
from dart.verification.greedy import verify_greedy_reference
from dart.verification.state_checks import _cache_failures, _reference_forward
from dart.verification.synthetic import _synthetic_models
from dart.verification.tree_verify import verify_tree_greedy


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


@pytest.mark.parametrize('verify', [verify_block_greedy, verify_tree_greedy])
@pytest.mark.parametrize('pending,drafts', [([10], [2, 13, 14]), ([10, 11], [12, 2, 14])])
@pytest.mark.parametrize('execution,limit', [
    ('eager', 'context'), ('static', 'capacity'), ('static', 'width'),
    ('predictions', 'capacity'), ('predictions', 'width'),
])
@torch.inference_mode()
def test_eos_preserves_full_block_limits(models, verify, pending, drafts, execution, limit):
    base, reference = models
    prompt = [5, 7, 9]
    full = pending + drafts
    total = len(prompt) + len(full)
    if execution == 'eager':
        base.model.config.max_position_embeddings = total - 1
        target = base
    else:
        target = CudaGraphTarget(
            base, capacity=total - 1 if limit == 'capacity' else total,
            max_block_tokens=len(full) - 1 if limit == 'width' else len(full),
            replay=False, capture_predictions=execution == 'predictions',
        )
    try:
        target.eos_token_ids = frozenset([2])
        state, _ = target.prefill(prompt)
        forward = target.forward_predictions if execution == 'predictions' else target.forward_block
        reason = {'context': 'context limit', 'capacity': 'replay capacity', 'width': 'prepared replay size'}[limit]
        with pytest.raises(ValueError, match=reason) as native:
            forward(state, full)
        with pytest.raises(ValueError) as error:
            verify(target, state, pending, drafts, compute_entropy=execution != 'predictions')
        assert str(error.value) == str(native.value)
        assert state.seq_len == len(prompt) and not state._failed
        expected = _reference_forward(reference, prompt)
        assert not _cache_failures(state, expected.past_key_values, len(prompt), atol=2e-6, rtol=2e-5)
        actual = target.forward_block(state, pending)[-1]
        expected = _reference_forward(reference, prompt + pending).logits[0, -1]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-5)
    finally:
        if target is not base:
            target.close()


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


@pytest.mark.parametrize('context,width,rows', [(8, 5, 3), (None, None, 3), ('8', 5, 5), (8, True, 5)])
def test_eos_shortcut_checks_the_complete_input(context, width, rows):
    target = CountingTarget([13], False)
    target.model.config.max_position_embeddings = context
    target.max_block_tokens = width
    pending, drafts = [10, 11], [12, 13, 14]
    result = verify_block_greedy(target, SimpleNamespace(seq_len=3), pending, drafts)
    assert target.calls == [(pending + drafts)[:rows]]
    assert result.committed == [12, 13] and result.n_draft_proposed == 3
    assert pending == [10, 11] and drafts == [12, 13, 14]


@pytest.mark.parametrize('pending,drafts', [([67], [2, 13, 14]), ([10], [2, 13, 67])])
def test_oversized_eos_block_preserves_token_error_precedence(models, pending, drafts):
    target, _ = models
    target.model.config.max_position_embeddings = 6
    target.eos_token_ids = frozenset([2])
    state, _ = target.prefill([5, 7, 9])
    with pytest.raises(ValueError, match='outside the model vocabulary') as native:
        target.forward_block(state, pending + drafts)
    with pytest.raises(ValueError) as error:
        verify_block_greedy(target, state, pending, drafts)
    assert str(error.value) == str(native.value)
    assert state.seq_len == 3 and not state._failed
