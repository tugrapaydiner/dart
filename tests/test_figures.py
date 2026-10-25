from __future__ import annotations

import hashlib
import json
import xml.etree.ElementTree as ET
from copy import deepcopy
from types import SimpleNamespace

import pytest

from benchmarks import figures


def report():
    costs = {"hf_greedy": [6, 6], "hf_prompt_lookup": [2, 5], "target_greedy": [1, 4],
             "target_only": [3, 4], "dart": [2, 2], "dart_model_norm": [3, 3]}
    outputs = [{"id": str(i), "workload": kind, "max_new_tokens": 16 * (i + 1),
                "generated_tokens": 4, "output_hash": str(i)} for i, kind in enumerate(("code", "structured"))]
    samples = [{"method": name, "repeat": repeat, "prompt_index": i,
                "latency_ms": cost, "tokens": 4, "peak_gpu_bytes": 10, "output_hash": str(i)}
               for name, values in costs.items() for i, cost in enumerate(values) for repeat in range(2)]
    return {"seed": 303, "timing": {"samples": samples, "repeats": 2}, "correctness": {"outputs": outputs},
            "summary": {"speedup": 999}}


def test_figures_recompute_ratios_and_retain_the_slower_case():
    data = figures.summarize(report())
    aggregate = data["aggregate"]
    assert aggregate["timing"]["dart"]["speedup"] == 1.75
    assert aggregate["fastest_control"] == "target_greedy"
    assert aggregate["vs_fastest"]["speedup"] == 1.25
    assert aggregate["norm_ablation"]["speedup"] == 1.5
    assert data["cases"][0]["vs_fastest"]["speedup"] == .5
    assert data["cases"][0]["vs_fastest"]["ci95_speedup"] == [.5, .5]
    assert data["cases"][1]["fastest_control"] == "dart_model_norm"
    assert len(data["cases"]) == 2 and set(data["workloads"]) == {"code", "structured"}
    assert data["prompts"] == 2 and data["repeats"] == 2 and data["budgets"] == [16, 32]


@pytest.mark.parametrize("change", ["missing", "duplicate", "hash", "tokens", "latency", "id"])
def test_invalid_evidence_cannot_produce_figures(change):
    value = report()
    rows = value["timing"]["samples"]
    if change == "missing":
        rows.pop()
    elif change == "duplicate":
        rows[-1] = deepcopy(rows[0])
    elif change == "hash":
        rows[0]["output_hash"] = "different"
    elif change == "tokens":
        rows[0]["tokens"] = 3
    elif change == "latency":
        rows[0]["latency_ms"] = float("nan")
    else:
        value["correctness"]["outputs"][1]["id"] = "0"
    with pytest.raises(ValueError):
        figures.summarize(value)


def test_public_command_renders_all_figures_from_raw_measurements(tmp_path):
    source, output, previews = tmp_path / "report.json", tmp_path / "figures", tmp_path / "previews"
    source.write_text(json.dumps(report()), encoding="utf-8")
    figures.main(["--report", str(source), "--out", str(output), "--previews", str(previews)])
    audit = json.loads((output / "figure-data.json").read_text(encoding="utf-8"))
    assert audit["input_sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert set(audit["figures_sha256"]) == {"methods", "workloads", "normalization", "cases"}
    for name, digest in audit["figures_sha256"].items():
        raw = (output / (name + ".svg")).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest
        svg = ET.fromstring(raw)
        assert svg.tag.endswith("svg") and svg.attrib["width"] == "982"
        assert b"#ebcece" in raw if name == "cases" else b"#ced6bf" in raw
        if name == "methods":
            assert "2 cases × 2 repeats".encode() in raw
        if name == "workloads":
            assert b"1 fixed cases per workload" in raw and b"16 / 32-token output budgets" in raw
        assert (previews / (name + ".png")).read_bytes().startswith(b"\x89PNG")
    assert audit["data"]["cases"][0]["vs_fastest"]["speedup"] == .5
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(SystemExit):
        figures.main(["--report", str(source), "--out", str(output)])
    assert before == {p.name: p.read_bytes() for p in output.iterdir()}


def test_registered_fonts_are_kept_and_reused_without_overwriting(tmp_path, monkeypatch):
    import fontTools.ttLib
    import fontTools.varLib.instancer
    import matplotlib
    from matplotlib import font_manager

    source = tmp_path / "source.ttf"
    source.write_bytes(b"synthetic font input")
    saved, registered = [], []

    class Font:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def __contains__(self, key):
            return key == "fvar"

        def __getitem__(self, key):
            return SimpleNamespace(axes=[SimpleNamespace(axisTag=axis) for axis in ("wght", "opsz")])

    class Static:
        def __getitem__(self, key):
            return SimpleNamespace(names=[])

        def save(self, path):
            assert not path.exists()
            path.write_bytes(b"synthetic font cache")
            saved.append(path)

        def close(self):
            pass

    monkeypatch.setattr(matplotlib, "get_cachedir", lambda: str(tmp_path / "cache"))
    monkeypatch.setattr(fontTools.ttLib, "TTFont", lambda path: Font())
    monkeypatch.setattr(fontTools.varLib.instancer, "instantiateVariableFont", lambda *a, **k: Static())
    monkeypatch.setattr(font_manager.fontManager, "addfont", registered.append)
    family = figures._font(source)
    assert figures._font(source) == family
    assert len(saved) == 3 and len(registered) == 6
    assert all(path.read_bytes() == b"synthetic font cache" for path in saved)
