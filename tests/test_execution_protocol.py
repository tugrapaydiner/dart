from __future__ import annotations

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from benchmarks import execution_protocol as protocol


@pytest.fixture
def inputs(monkeypatch, tmp_path):
    for name in protocol.MODEL_FILES:
        (tmp_path / name).write_bytes(b"synthetic file for fingerprint checks")
    rows = []
    for split in ("validation", "test"):
        for index, name in enumerate(protocol.selected_ids(split)[0]):
            rows.append(SimpleNamespace(id=name, split=split, source="source:" + name, family="family:" + name,
                                        workload=name.split("-")[-2], max_new_tokens=(16, 32, 64)[index % 3]))
    tokenizer = SimpleNamespace(eos_token_id=2)
    monkeypatch.setattr(protocol, "encode_prompt", lambda row, tokenizer, limit: [7, 8, 9])
    monkeypatch.setattr(protocol, "tokenizer_hash", lambda tokenizer: "tokenizer")
    monkeypatch.setattr(protocol, "corpus_hash", lambda rows: protocol.digest([row.id for row in rows]))
    monkeypatch.setattr(protocol, "source_hashes", lambda: {"dart/engine.py": "source"})
    monkeypatch.setattr(protocol, "version", lambda name: "1.0")
    return rows, tokenizer, 1024, tmp_path


def test_final_plan_keeps_every_case_and_balanced_budgets(inputs):
    plan = protocol.make_protocol(*inputs, split="test")
    assert len(plan["cases"]) == 48 and len(plan["timed_ids"]) == 12
    assert Counter(case["workload"] for case in plan["cases"]) == dict.fromkeys(protocol.WORKLOADS, 12)
    assert Counter(case["max_new_tokens"] for case in plan["cases"]) == {16: 16, 32: 16, 64: 16}
    assert plan["sessions"] == 3 and plan["timeout_seconds"] == 3600
    assert len(protocol.session_checks(plan, 0)) == 48
    assert protocol.session_checks(plan, 1) == plan["timed_ids"]
    assert protocol.session_checks(plan, 2) == plan["timed_ids"]
    protocol.validate_protocol(plan, *inputs)


def test_validation_is_a_small_disjoint_public_smoke(inputs):
    plan = protocol.make_protocol(*inputs)
    assert plan["sessions"] == 1 and plan["repeats"] == 2 and plan["timeout_seconds"] == 300
    assert len(plan["cases"]) == len(plan["timed_ids"]) == 4
    assert all("validation" in name for name in plan["timed_ids"])
    assert not set(plan["timed_ids"]) & set(protocol.selected_ids("test")[0])
    assert inputs[-1].as_posix() not in json.dumps(plan)


@pytest.mark.parametrize("field,value", [("repeats", 1), ("sessions", 1), ("timeout_seconds", 9999),
                                         ("schema", "unknown"), ("protocol_hash", "forged")])
def test_changed_settings_are_rejected_even_with_a_new_digest(inputs, field, value):
    plan = protocol.make_protocol(*inputs, split="test")
    plan[field] = value
    if field != "protocol_hash":
        plan["protocol_hash"] = protocol.digest({key: value for key, value in plan.items() if key != "protocol_hash"})
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_protocol(plan, *inputs)


@pytest.mark.parametrize("field,value", [("sessions", True), ("sessions", 1.0), ("repeats", 2.0)])
def test_numeric_type_changes_do_not_pass_with_the_original_fingerprint(inputs, field, value):
    plan = protocol.make_protocol(*inputs)
    plan[field] = value
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_protocol(plan, *inputs)


def test_nested_numeric_type_changes_do_not_pass_with_the_original_fingerprint(inputs):
    plan = protocol.make_protocol(*inputs)
    plan["settings"]["threads"] = True
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_protocol(plan, *inputs)


@pytest.mark.parametrize("kind", ["case", "order", "model", "source", "packages", "tokenizer", "tokens"])
def test_protocol_rejects_drift_before_any_generation(inputs, monkeypatch, kind):
    plan = protocol.make_protocol(*inputs)
    if kind == "case":
        plan["cases"].pop()
    elif kind == "order":
        plan["timed_ids"].reverse()
    elif kind == "model":
        (inputs[-1] / "model.safetensors").write_bytes(b"different synthetic weights")
    elif kind == "source":
        monkeypatch.setattr(protocol, "source_hashes", lambda: {"dart/engine.py": "changed"})
    elif kind == "packages":
        monkeypatch.setattr(protocol, "version", lambda name: "2.0")
    elif kind == "tokenizer":
        monkeypatch.setattr(protocol, "tokenizer_hash", lambda tokenizer: "changed")
    else:
        monkeypatch.setattr(protocol, "encode_prompt", lambda *args: [9, 8, 7])
    with pytest.raises(ValueError, match="changed"):
        protocol.validate_protocol(plan, *inputs)


def test_schedule_pairs_every_method_and_rotates_across_sessions(inputs):
    plan = protocol.make_protocol(*inputs, split="test")
    orders = []
    for session in range(3):
        calls = protocol.schedule(plan, session)
        timed = [call for call in calls if call["phase"] == "timed"]
        warmup = [call for call in calls if call["phase"] == "warmup"]
        assert len(timed) == 12 * 6 * 3 and len(warmup) == 12 * 6
        assert len({(call["case"], call["method"], call["repeat"]) for call in timed}) == len(timed)
        for start in range(0, len(timed), 6):
            group = timed[start:start + 6]
            assert {call["method"] for call in group} == set(protocol.METHODS)
            assert [call["position"] for call in group] == list(range(6))
            assert len({(call["case"], call["repeat"]) for call in group}) == 1
        orders.append(tuple(call["method"] for call in timed[:6]))
    assert len(set(orders)) == 3
    assert protocol.schedule(plan, 0) == protocol.schedule(copy.deepcopy(plan), 0)


@pytest.mark.parametrize("session", [-1, 3, True, 0.5])
def test_invalid_session_indices_fail(inputs, session):
    with pytest.raises(ValueError, match="session index"):
        protocol.schedule(protocol.make_protocol(*inputs, split="test"), session)


def test_eos_and_missing_cases_cannot_be_silently_dropped(inputs, monkeypatch):
    rows, tokenizer, context, model = inputs
    with pytest.raises(ValueError, match="missing or duplicate"):
        protocol.make_protocol(rows[1:], tokenizer, context, model)
    with pytest.raises(ValueError, match="missing or duplicate"):
        protocol.make_protocol(rows + rows[:1], tokenizer, context, model)
    monkeypatch.setattr(protocol, "encode_prompt", lambda *args: [7, 2])
    with pytest.raises(ValueError, match="end in EOS"):
        protocol.make_protocol(*inputs)


def test_existing_protocol_cannot_be_overwritten(inputs):
    plan = protocol.make_protocol(*inputs)
    path = inputs[-1] / "new-directory" / "protocol.json"
    protocol.write_protocol(path, plan)
    assert json.loads(path.read_text()) == plan
    with pytest.raises(FileExistsError):
        protocol.write_protocol(path, {})
    assert json.loads(path.read_text()) == plan


def test_source_inventory_detects_added_code_but_normalizes_checkout_newlines(tmp_path, monkeypatch):
    import dart

    dart_root, bench_root = tmp_path / "dart", tmp_path / "benchmarks"
    dart_root.mkdir()
    (bench_root / "fixtures").mkdir(parents=True)
    init = dart_root / "__init__.py"
    init.write_bytes(b"value = 1\r\n")
    script = bench_root / "execution_protocol.py"
    script.write_bytes(b"value = 2\n")
    (bench_root / "fixtures" / "prompts.jsonl").write_text('{"id":1}\n')
    monkeypatch.setattr(dart, "__file__", str(init))
    monkeypatch.setattr(protocol, "__file__", str(script))
    before = protocol.source_hashes()
    assert set(before) == {"dart/__init__.py", "benchmarks/execution_protocol.py", "benchmarks/fixtures/prompts.jsonl"}
    init.write_bytes(b"value = 1\n")
    assert protocol.source_hashes() == before
    (dart_root / "new_runtime.py").write_text("value = 3\n")
    assert protocol.source_hashes() != before


def test_missing_local_weights_fail_without_network_fallback(inputs):
    (inputs[-1] / "model.safetensors").unlink()
    with pytest.raises(ValueError, match="local config"):
        protocol.make_protocol(*inputs)
