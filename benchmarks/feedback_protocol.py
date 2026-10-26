"""Freeze the entropy ablation and its non-repetitive confirmation target."""

from __future__ import annotations

import argparse
import json
from importlib.metadata import version
from pathlib import Path

from benchmarks import execution_protocol as execution
from benchmarks import feedback_workloads, verification_workloads
from benchmarks.execution_analysis import validate_archive_protocol
from dart.data.corpus import corpus_hash, encode_prompt, read_records
from dart.data.storage import tokenizer_hash
from dart.presets import local_model_path

SCHEMA = "dart-feedback-protocol-v1"
METHODS = ("entropy_on4", "entropy_off4", "entropy_off2", "replay_greedy",
           "replay_hf_greedy", "replay_hf_lookup", "time_scheduler")
BASELINES = ("replay_greedy", "replay_hf_greedy", "replay_hf_lookup")


def records(split):
    if split not in ("validation", "test"):
        raise ValueError("select validation or test explicitly")
    path = feedback_workloads.PROMPTS if split == "test" else verification_workloads.PROMPTS
    return [row for row in read_records(path) if row.split == split]


def settings(split, context):
    ids = [row.id for row in records(split)]
    final = split == "test"
    return {
        "schema": SCHEMA, "split": split, "context_limit": context, "timed_ids": ids,
        "methods": list(METHODS), "baselines": list(BASELINES), "candidate": "entropy_off4",
        "control": "entropy_on4", "control_commit": "2c9a6a8ad32172b3d92a557af575c4c2a71cac79",
        "control_scope": "prior entropy arithmetic enabled in the same engine with requires_entropy=True; an ablation, not a separate checkout",
        "sessions": 3 if final else 1, "repeats": 3 if final else 2, "warmups": 1,
        "timeout_seconds": 1800 if final else 300, "seed": 311,
        "settings": {"device": "cuda", "dtype": "float32", "attention": "eager", "native_rms_norm": True,
                     "reference_native_rms_norm": False, "threads": 1, "matmul_precision": "highest",
                     "draft_len": 4, "alternative_draft_len": 2, "max_ngram": 4, "min_ngram": 1,
                     "max_block_tokens": 5, "cache_capacity": context, "record_hidden": False,
                     "scheduler": {"retry_interval": 16, "warmup_rounds": 2, "min_time_saving": 0.05,
                                   "entropy_gate": "disabled; unused entropy skipped"},
                     "prefill": "existing device policy; HF retains last-logit prefill"},
        "ordering": "rotate methods by session + case index + repeat; reverse odd session+repeat rounds",
        "correctness": "every case in every session; all checks, warmups and timed outputs match complete original-norm HF outputs",
        "timing_scope": "token IDs through generation and text decoding, synchronized around each request; setup, tokenization, checking and writes excluded",
        "memory_scope": "allocator peak with two model copies, one replay workspace and request caches; not standalone method memory",
        "uncertainty": "paired Student-t 95% interval over three session log mean-latency ratios, df=2; fixed cases on one host",
        "gates": {"minimum_mean_latency_reduction": 0.05, "minimum_nonrepetitive_latency_reduction": 0.05,
                  "ratio_ci95_lower_exceeds": 1.0, "each_session_improves": True,
                  "beat_hf_nonrepetitive": True, "maximum_nonrepetitive_workload_regression": 0.05,
                  "maximum_peak_allocated_bytes": 6 * 1024**3, "maximum_extra_learned_parameters": 0},
    }


def make_protocol(tokenizer, context, model, *, split="validation"):
    loader = feedback_workloads.load_frozen if split == "test" else verification_workloads.load_frozen
    rows = [row for row in loader(tokenizer, context) if row.split == split]
    fixed = settings(split, context)
    if [row.id for row in rows] != fixed["timed_ids"]:
        raise ValueError("feedback case selection changed")
    cases = []
    for row in rows:
        tokens = encode_prompt(row, tokenizer, context - 3)
        if tokens[-1] == tokenizer.eos_token_id:
            raise ValueError("HF lookup requires prompts that do not end in EOS")
        cases.append({"id": row.id, "source": row.source, "family": row.family, "workload": row.workload,
                      "max_new_tokens": row.max_new_tokens, "prompt_ids": tokens, "prompt_hash": execution.digest(tokens)})
    plan = {**fixed, "corpus_hash": corpus_hash(rows), "tokenizer_hash": tokenizer_hash(tokenizer),
            "model_hashes": execution.model_hashes(model), "source_hashes": execution.source_hashes(),
            "packages": {name: version(name) for name in execution.PACKAGES}, "cases": cases}
    return {**plan, "protocol_hash": execution.digest(plan)}


def validate_archive(plan):
    if not isinstance(plan, dict) or plan.get("schema") != SCHEMA:
        raise ValueError("unsupported feedback protocol")
    rows = records(plan.get("split"))
    validate_archive_protocol(plan, settings=settings, rows=rows, check_ids=[row.id for row in rows],
                              required_sources=("benchmarks/feedback_protocol.py", "benchmarks/feedback_bench.py",
                                                "benchmarks/feedback_analysis.py", "dart/verification/block_verify.py"))


def read_protocol(path):
    if path.stat().st_size > 2 * 1024**2:
        raise ValueError("protocol exceeds 2 MiB")
    plan = json.loads(path.read_text(encoding="utf-8"))
    validate_archive(plan)
    if plan["source_hashes"] != execution.source_hashes():
        raise ValueError("protocol source changed; freeze a new protocol")
    return plan


def validate_protocol(plan, tokenizer, context, model):
    expected = make_protocol(tokenizer, context, model, split=plan.get("split"))
    if execution.digest(plan) != execution.digest(expected):
        raise ValueError("protocol, inputs, model, packages or source changed; freeze a new protocol")


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
    plan = make_protocol(tokenizer, config.max_position_embeddings, model, split=args.split)
    execution.write_protocol(args.out, plan)
    print(json.dumps({"protocol_hash": plan["protocol_hash"], "split": args.split, "cases": len(plan["cases"]),
                      "sessions": plan["sessions"], "timeout_seconds": plan["timeout_seconds"]}))


if __name__ == "__main__":
    main()
