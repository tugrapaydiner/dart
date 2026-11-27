from __future__ import annotations

import re
from pathlib import Path

from packaging.version import Version

ROOT = Path(__file__).resolve().parents[1]


def pins():
    lines = (ROOT / "constraints-tested.txt").read_text(encoding="utf-8").splitlines()
    return dict(line.split("==") for line in lines if "==" in line and not line.startswith("#"))


def floors():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    block = text[text.index("dependencies = ["):text.index("]", text.index("dependencies = ["))]
    return dict(re.findall(r'"([A-Za-z0-9_.-]+)>=([^"]+)"', block))


def test_core_runtime_floors_are_the_tested_versions():
    tested, declared = pins(), floors()
    for package in ("torch", "transformers"):  # The runtime depends on their cache and graph internals.
        assert Version(declared[package]) == Version(tested[package]), package


def test_import_errors_name_the_declared_transformers_floor():
    expected = f"transformers>={floors()['transformers']}"
    for path in ("dart/models/target.py", "dart/kv/cache.py", "dart/kv/residency.py"):
        source = (ROOT / path).read_text(encoding="utf-8")
        assert re.findall(r"transformers>=[0-9.]+", source) == [expected] * len(re.findall(r"transformers>=", source)), path
