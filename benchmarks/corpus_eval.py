from __future__ import annotations

from dataclasses import asdict, replace
from itertools import groupby
from statistics import mean

from benchmarks.run_bench import build_hf_method
from dart.data.corpus import Prompt, corpus_hash, encode_prompt, validate_records
from dart.data.storage import tokenizer_hash
from dart.data.teacher import target_fingerprint
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig


def output_stats(token_ids: list[int], eos_ids: frozenset[int], budget: int) -> dict:
    windows = [tuple(token_ids[i:i + 4]) for i in range(max(0, len(token_ids) - 3))]
    finished = bool(token_ids and token_ids[-1] in eos_ids)
    return {
        "generated_tokens": len(token_ids), "finished_eos": finished,
        "early_eos": finished and len(token_ids) < budget,
        "hit_token_limit": len(token_ids) == budget,
        "repeated_4gram_fraction": 1 - len(set(windows)) / len(windows) if windows else 0.0,
        "max_identical_token_run": max((sum(1 for _ in group) for _, group in groupby(token_ids)), default=0),
    }


def _summarize(rows: list[dict]) -> dict:
    proposed = sum(row["draft_tokens_proposed"] for row in rows)
    accepted = sum(row["draft_tokens_accepted"] for row in rows)
    return {
        "prompts": len(rows), "exact_match_rate": 1.0,
        "mean_generated_tokens": mean(row["generated_tokens"] for row in rows),
        "early_eos_count": sum(row["early_eos"] for row in rows),
        "eos_count": sum(row["finished_eos"] for row in rows),
        "token_limit_count": sum(row["hit_token_limit"] for row in rows),
        "mean_repeated_4gram_fraction": mean(row["repeated_4gram_fraction"] for row in rows),
        "max_identical_token_run": max(row["max_identical_token_run"] for row in rows),
        "draft_tokens_proposed": proposed, "draft_tokens_accepted": accepted,
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "rounds": sum(row["rounds"] for row in rows),
    }


def evaluate_corpus(
    engine: DartEngine, records: list[Prompt], config: DecodeConfig | None = None,
    progress=None, *, reference_target: TargetModel | None = None,
) -> dict:
    """Full-output correctness and acceptance checks, with no latency claims."""
    validate_records(records)
    config = config or engine.default_decode_config or DecodeConfig()
    config.validate()
    if config.mode != "exact" or len({row.split for row in records}) != 1:
        raise ValueError("corpus evaluation requires exact mode and one explicit split")
    target = engine.target
    if target.tokenizer is None:
        raise ValueError("corpus evaluation requires a tokenizer")
    reference_info = None
    if reference_target is not None:
        if not target.native_rms_norm_count or reference_target.native_rms_norm_count:
            raise ValueError("norm comparison requires a native target and an original-norm reference")
        selected_info = target_fingerprint(target)
        selected_info.pop("native_rms_norms")
        reference_info = target_fingerprint(reference_target)
        if selected_info != reference_info:
            raise ValueError("reference weights and decode settings must match except for normalization")
        if (target.tokenizer is not reference_target.tokenizer
                and tokenizer_hash(target.tokenizer) != tokenizer_hash(reference_target.tokenizer)):
            raise ValueError("reference tokenizers must match")
    prompts = [encode_prompt(row, target.tokenizer, target.model.config.max_position_embeddings) for row in records]
    if len({tuple(ids) for ids in prompts}) != len(prompts):
        raise ValueError("corpus contains duplicate tokenized prompts")
    references = {budget: build_hf_method(reference_target or target, budget)
                  for budget in {row.max_new_tokens for row in records}}
    results = []
    for index, (record, prompt) in enumerate(zip(records, prompts)):
        expected = references[record.max_new_tokens](prompt)
        kind = {"instruction": "chat", "code": "code", "structured": "json", "repetition": "chat"}[record.workload]
        cfg = replace(config, max_new_tokens=record.max_new_tokens, prompt_kind=kind)
        output = engine.generate(prompt, cfg)
        if output.token_ids != expected.token_ids:
            raise ValueError(f"{record.id}: complete token mismatch")
        results.append({
            "id": record.id, "source": record.source, "family": record.family,
            "workload": record.workload, "prompt_kind": kind, "prompt_tokens": len(prompt),
            "max_new_tokens": record.max_new_tokens, "output_hash": output_hash(output.token_ids),
            "token_ids": output.token_ids, "text": output.text,
            **output_stats(output.token_ids, target.eos_token_ids, record.max_new_tokens),
            **{key: output.metadata[key] for key in ("draft_tokens_proposed", "draft_tokens_accepted", "rounds")},
        })
        if progress is not None:
            progress(index + 1, len(records))
    return {
        "split": records[0].split, "selected_corpus_hash": corpus_hash(records),
        "reference": "plain HF greedy (original RMSNorm)" if reference_target is not None else "plain HF greedy",
        "decode_config": asdict(config),
        **({"reference_target": reference_info} if reference_info is not None else {}),
        "budget_scope": "each prompt's max_new_tokens overrides the decode default",
        "prompt_kind_scope": "code and structured workloads map to code/json; instruction and repetition map to chat",
        "repetition_scope": "fraction of repeated overlapping four-token windows in generated output",
        "measurement_scope": "correctness and acceptance only; latency requires repeated benchmark timing",
        "summary": _summarize(results),
        "by_workload": {
            kind: _summarize([row for row in results if row["workload"] == kind])
            for kind in sorted({row["workload"] for row in results})
        },
        "by_budget": {
            str(budget): _summarize([row for row in results if row["max_new_tokens"] == budget])
            for budget in sorted({row["max_new_tokens"] for row in results})
        },
        "outputs": results,
    }
