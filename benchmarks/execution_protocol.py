"""Freeze inputs and settings before measuring the CUDA execution paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from benchmarks.verification_workloads import load_frozen
from dart.data.corpus import corpus_hash, encode_prompt
from dart.data.storage import tokenizer_hash
from dart.presets import local_model_path

SCHEMA = "dart-execution-protocol-v2"
LEGACY_SCHEMA = "dart-execution-protocol-v1"
CONTROL_COMMIT = "63411869e4a8ed832c0cc4c1ecb560801625d84e"
METHODS = ("current_dart", "static_dart", "replay_dart", "replay_greedy",
           "replay_hf_greedy", "replay_hf_lookup")
BASELINES = ("replay_greedy", "replay_hf_greedy", "replay_hf_lookup")
WORKLOADS = ("code", "instruction", "repetition", "structured")
PILOT_INDICES = {"code": 0, "instruction": 2, "repetition": 2, "structured": 0}
MODEL_FILES = ("config.json", "model.safetensors", "tokenizer.json",
               "tokenizer_config.json", "special_tokens_map.json", "generation_config.json")
PACKAGES = ("torch", "transformers", "numpy", "tokenizers", "safetensors")


def digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def model_hashes(directory: Path) -> dict[str, str]:
    required = MODEL_FILES[:4]
    if any(not (directory / name).is_file() for name in required):
        raise ValueError("execution benchmarks need local config, tokenizer and single-file safetensors weights")
    return {name: file_hash(directory / name) for name in MODEL_FILES if (directory / name).is_file()}


def source_hashes() -> dict[str, str]:
    import dart

    roots = {"dart": Path(dart.__file__).parent, "benchmarks": Path(__file__).parent}
    result = {}
    for package, root in roots.items():
        paths = sorted(root.rglob("*.py"))
        if package == "benchmarks":
            paths += sorted((root / "fixtures").glob("*.json*"))
        for path in paths:
            content = path.read_bytes().replace(b"\r\n", b"\n")  # Git checkout line endings do not change the code.
            result[f"{package}/{path.relative_to(root).as_posix()}"] = hashlib.sha256(content).hexdigest()
    return result


def selected_ids(split: str) -> tuple[list[str], list[str]]:
    if split not in ("validation", "test"):
        raise ValueError("select validation or test explicitly")
    if split == "validation":
        ids = [f"v03-execution-validation-{workload}-{PILOT_INDICES[workload]:02d}" for workload in WORKLOADS]
        return ids, list(ids)
    checks = [f"v03-execution-test-{workload}-{index:02d}" for workload in WORKLOADS for index in range(12)]
    timed = [f"v03-execution-test-{workload}-{index:02d}" for workload in WORKLOADS for index in range(3)]
    return checks, timed


def make_protocol(rows, tokenizer, context_limit: int, model: Path, *, split="validation") -> dict:
    check_ids, _ = selected_ids(split)
    records = {row.id: row for row in rows}
    if len(records) != len(rows) or any(name not in records for name in check_ids):
        raise ValueError("verification corpus has missing or duplicate cases")
    cases = []
    for name in check_ids:
        row = records[name]
        if row.split != split:
            raise ValueError("selected case belongs to the wrong split")
        tokens = encode_prompt(row, tokenizer, context_limit - 3)
        if tokens[-1] == tokenizer.eos_token_id:
            raise ValueError("HF lookup requires prompts that do not end in EOS")
        cases.append({"id": name, "source": row.source, "family": row.family,
                      "workload": row.workload, "max_new_tokens": row.max_new_tokens,
                      "prompt_ids": tokens, "prompt_hash": digest(tokens)})
    protocol = {
        **protocol_settings(split, context_limit), "corpus_hash": corpus_hash(rows),
        "tokenizer_hash": tokenizer_hash(tokenizer), "model_hashes": model_hashes(model),
        "source_hashes": source_hashes(), "packages": {name: version(name) for name in PACKAGES},
        "cases": cases,
    }
    return {**protocol, "protocol_hash": digest(protocol)}


def protocol_settings(split: str, context_limit: int, *, schema=SCHEMA) -> dict:
    if schema not in (SCHEMA, LEGACY_SCHEMA):
        raise ValueError("unsupported execution protocol schema")
    _, timed_ids = selected_ids(split)
    final = split == "test"
    legacy = schema == LEGACY_SCHEMA
    return {
        "schema": schema, "split": split, "context_limit": context_limit, "timed_ids": timed_ids,
        "methods": list(METHODS), "baselines": list(BASELINES), "candidate": "replay_dart",
        "control": "current_dart", "control_commit": CONTROL_COMMIT if legacy else None,
        "control_scope": ("optimized eager DART path retained from the control commit" if legacy else
                          "optimized eager DART from source_hashes; all methods use the same recorded source, not a historical checkout"),
        "sessions": 3 if final else 1, "repeats": 3 if final else 2, "warmups": 1,
        "timeout_seconds": 3600 if final else 300, "seed": 307,
        "settings": {"device": "cuda", "dtype": "float32", "attention": "eager",
                     "native_rms_norm": True, "reference_native_rms_norm": False,
                     "threads": 1, "matmul_precision": "highest", "draft_len": 4,
                     "max_ngram": 4, "min_ngram": 1, "max_block_tokens": 5,
                     "cache_capacity": context_limit, "record_hidden": False,
                     "prefill": "existing device policy; HF keeps its native last-logit prefill"},
        "ordering": "rotate methods by session + case index + repeat; reverse odd session+repeat rounds",
        "correctness": "session 0 checks every case; later sessions check all timed cases; warmups and timed outputs also match",
        "timing_scope": "token IDs through generation and text decoding, synchronized around each request; setup, tokenization, checking and writes excluded",
        "memory_scope": "allocator bytes with two model copies, replay/static workspaces and request caches resident; not standalone method memory",
        "uncertainty": "paired Student-t 95% interval over three session log mean-latency ratios, df=2; fixed cases on one host",
        "gates": {"minimum_mean_latency_reduction": 0.10, "ratio_ci95_lower_exceeds": 1.0,
                  "each_session_improves": True, "beat_fastest_matched_baseline": True,
                  "maximum_workload_regression": 0.05, "maximum_peak_allocated_bytes": 6 * 1024**3,
                  "maximum_extra_learned_parameters": 5_000_000},
    }


def validate_protocol(protocol: dict, rows, tokenizer, context_limit: int, model: Path) -> None:
    expected = make_protocol(rows, tokenizer, context_limit, model, split=protocol.get("split"))
    if digest(protocol) != digest(expected):
        raise ValueError("protocol, inputs, model, packages or source changed; freeze a new protocol before measuring")


def session_checks(protocol: dict, session: int) -> list[str]:
    if type(session) is not int or not 0 <= session < protocol["sessions"]:
        raise ValueError("session index is outside the protocol")
    return [case["id"] for case in protocol["cases"]] if session == 0 else list(protocol["timed_ids"])


def schedule(protocol: dict, session: int) -> list[dict]:
    session_checks(protocol, session)
    names = protocol["methods"]
    result = []
    for phase, count in (("warmup", protocol["warmups"]), ("timed", protocol["repeats"])):
        for repeat in range(count):
            for index, name in enumerate(protocol["timed_ids"]):
                shift = (session + index + repeat) % len(names)
                order = names[shift:] + names[:shift]
                if (session + repeat) % 2:
                    order = list(reversed(order))
                result.extend({"phase": phase, "repeat": repeat, "case": name, "method": method,
                               "position": position} for position, method in enumerate(order))
    return result


def write_protocol(path: Path, protocol: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(protocol, output, indent=2, ensure_ascii=False, allow_nan=False)
        output.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="new protocol JSON file")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("protocol file already exists")
    from transformers import AutoConfig, AutoTokenizer

    model = local_model_path(args.model)
    config = AutoConfig.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
    rows = load_frozen(tokenizer, config.max_position_embeddings)
    protocol = make_protocol(rows, tokenizer, config.max_position_embeddings, model, split=args.split)
    write_protocol(args.out, protocol)
    print(json.dumps({"protocol_hash": protocol["protocol_hash"], "split": args.split,
                      "checks": len(protocol["cases"]), "timed_cases": len(protocol["timed_ids"]),
                      "sessions": protocol["sessions"], "timeout_seconds": protocol["timeout_seconds"]}))


if __name__ == "__main__":
    main()
