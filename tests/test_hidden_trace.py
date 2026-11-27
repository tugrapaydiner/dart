import pytest
import torch

from dart.models.hidden_trace import collect_trace
from dart.verification.state_checks import _reference_forward
from dart.verification.synthetic import _synthetic_models


@pytest.fixture
def models():
    target, reference = _synthetic_models(17)
    target.eos_token_ids = frozenset()
    return target, reference


@pytest.mark.parametrize('budget', [1, 2, 6])
@pytest.mark.parametrize('record_hidden', [False, True])
def test_trace_stops_at_budget_and_matches_native_rows(models, monkeypatch, budget, record_hidden):
    target, reference = models
    target.record_hidden = record_hidden
    prompt, tokens, rows, hiddens = [0, 5, 0, 6], [], [], []
    with torch.inference_mode():
        for _ in range(budget):
            output = reference(input_ids=torch.tensor([prompt + tokens]), output_hidden_states=True, use_cache=False)
            row = output.logits[0, -1]
            rows.append(row)
            hiddens.append(output.hidden_states[-1][0, -1])
            tokens.append(int(row.argmax()))
    calls, forward = [], target.forward_block

    def record(state, ids):
        calls.append(list(ids))
        return forward(state, ids)

    monkeypatch.setattr(target, 'forward_block', record)
    trace = collect_trace(target, prompt, budget)
    assert trace.prompt_ids == prompt and trace.prompt_ids is not prompt
    assert trace.continuation_ids == tokens
    assert calls == [[token] for token in tokens[:-1]]
    assert trace.logits.dtype == torch.float16 and trace.entropy.dtype == torch.float32
    torch.testing.assert_close(trace.logits, torch.stack(rows).half(), atol=2e-6, rtol=2e-5)
    logp = torch.stack(rows).log_softmax(-1)
    torch.testing.assert_close(trace.entropy, -(logp.exp() * logp).sum(-1), atol=2e-6, rtol=2e-5)
    if record_hidden:
        assert trace.hidden.dtype == torch.float16
        torch.testing.assert_close(trace.hidden, torch.stack(hiddens).half(), atol=2e-6, rtol=2e-5)
    else:
        assert trace.hidden is None


@pytest.mark.parametrize('record_hidden', [False, True])
def test_trace_stops_at_first_eos_without_feeding_it(models, monkeypatch, record_hidden):
    target, reference = models
    prompt = [5, 7, 11]
    row = _reference_forward(reference, prompt).logits[0, -1]
    eos = int(row.argmax())
    target.eos_token_ids = frozenset({eos})
    target.record_hidden = record_hidden
    monkeypatch.setattr(target, 'forward_block', lambda *a: pytest.fail('fed EOS after completion'))
    trace = collect_trace(target, prompt, 6)
    assert trace.continuation_ids == [eos]
    assert trace.logits.shape == (1, target.model.config.vocab_size)
    torch.testing.assert_close(trace.logits[0], row.half(), atol=2e-6, rtol=2e-5)
    assert (trace.hidden is not None) == record_hidden


@pytest.mark.parametrize('record_hidden', [False, True])
def test_zero_budget_validates_prompt_without_model_work(models, monkeypatch, record_hidden):
    target, _ = models
    target.record_hidden = record_hidden
    monkeypatch.setattr(target.model, 'forward', lambda *a, **k: pytest.fail('empty trace ran the model'))
    trace = collect_trace(target, [0, 5, 7], 0)
    assert trace.prompt_ids == [0, 5, 7] and trace.continuation_ids == []
    assert trace.logits.shape == (0, target.model.config.vocab_size)
    assert trace.logits.dtype == torch.float16 and trace.logits.device.type == 'cpu'
    assert trace.entropy.shape == (0,) and trace.entropy.dtype == torch.float32
    assert trace.hidden is None


@pytest.mark.parametrize('prompt,budget', [
    ([], 0), ([], 1), ([5], -1), ([5], True), ([5], False), ([5], 1.5),
    ([5], 64), ([5] * 65, 0), ([-1], 0), ([67], 0), ([67], 1),
])
def test_invalid_trace_request_fails_before_model_execution(models, monkeypatch, prompt, budget):
    target, _ = models
    monkeypatch.setattr(target.model, 'forward', lambda *a, **k: pytest.fail('invalid trace ran the model'))
    with pytest.raises(ValueError):
        collect_trace(target, prompt, budget)


def test_trace_at_context_boundary_needs_no_extra_forward(models, monkeypatch):
    target, reference = models
    prompt = [5] * 63
    expected = int(_reference_forward(reference, prompt).logits[0, -1].argmax())
    monkeypatch.setattr(target, 'forward_block', lambda *a: pytest.fail('ran an unused final forward'))
    assert collect_trace(target, prompt, 1).continuation_ids == [expected]
