from __future__ import annotations

from types import SimpleNamespace

import pytest

from dart.runtime.metrics import output_hash
from scripts import run_real_model


def configure_run(monkeypatch, hf_tokens, baseline_tokens, with_lanes):
    target = SimpleNamespace(
        model=SimpleNamespace(config=SimpleNamespace(vocab_size=16)),
        tokenizer=SimpleNamespace(encode=lambda prompt: [1]),
        greedy_generate=lambda ids, max_new_tokens: list(baseline_tokens),
    )
    monkeypatch.setattr(run_real_model.TargetModel, "from_pretrained", lambda *a, **kw: target)
    monkeypatch.setattr(run_real_model, "hf_generate_greedy", lambda *a: list(hf_tokens))
    monkeypatch.setattr(run_real_model, "load_prompts", lambda path: ["test prompt"])
    monkeypatch.setattr(
        run_real_model, "build_lanes", lambda target: {"test": object()} if with_lanes else {},
    )
    output = SimpleNamespace(token_ids=list(baseline_tokens), metadata={
        "acceptance_rate": 1.0, "output_hash": output_hash(baseline_tokens),
    })
    monkeypatch.setattr(run_real_model, "DartEngine", lambda *a, **kw: SimpleNamespace(
        generate=lambda ids, config: output,
    ))
    monkeypatch.setattr("sys.argv", ["run_real_model.py", "--model", "local-test"])


@pytest.mark.parametrize("with_lanes", [False, True])
@pytest.mark.parametrize("hf_tokens,baseline_tokens", [
    ([5, 2], [5]),
    ([5], [5, 2]),
    ([5], []),
    ([5, 2], [6, 2]),
])
def test_evaluator_rejects_incomplete_or_different_outputs(
    monkeypatch, capsys, hf_tokens, baseline_tokens, with_lanes,
):
    configure_run(monkeypatch, hf_tokens, baseline_tokens, with_lanes)
    with pytest.raises(SystemExit) as error:
        run_real_model.main()
    assert error.value.code == 1
    assert "EXACTNESS FAILURE" in capsys.readouterr().err


@pytest.mark.parametrize("with_lanes", [False, True])
def test_evaluator_accepts_complete_matching_outputs(monkeypatch, capsys, with_lanes):
    configure_run(monkeypatch, [5, 2], [5, 2], with_lanes)
    run_real_model.main()
    assert "EXACTNESS HOLDS" in capsys.readouterr().out
