from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import torch

from dart.data.storage import tokenizer_hash
from dart.data.teacher import TraceDataset, _read_bounded, target_fingerprint
from dart.drafters.compact_diffusion import (
    CompactDiffusionDrafter,
    CompactDiffusionModel,
)
from dart.models.target import TargetModel
from dart.runtime.engine import DartEngine
from dart.runtime.types import DecodeConfig

SCHEMA = "dart-draft-failures-v1"
MAX_FILE_BYTES = 2 * 1024 ** 2
MAX_RECORDS = 4096


def _trace_hash(traces):
    return hashlib.sha256(json.dumps(traces.manifest, sort_keys=True).encode()).hexdigest()


def _model_hash(model):
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


def write_failures(
    target: TargetModel, model: CompactDiffusionModel, traces: TraceDataset,
    path: str | Path, *, indices: list[int] | None = None, canvas_len: int = 4,
    per_prompt: int = 8, seed: int = 17, max_bytes: int = MAX_FILE_BYTES, progress=None,
) -> dict:
    """Collect real proposal errors without storing another copy of teacher tensors."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("failure output already exists")
    if traces.manifest["split"] != "train" or model._target() is not target:
        raise ValueError("failure collection needs training traces and the shared target")
    if type(canvas_len) is not int or not 1 <= canvas_len <= min(64, model.cfg.max_canvas):
        raise ValueError("canvas_len exceeds the collection limit or model capacity")
    if type(per_prompt) is not int or not 1 <= per_prompt <= 64 or type(seed) is not int:
        raise ValueError("use per_prompt from 1 to 64 and an integer seed")
    if type(max_bytes) is not int or not 1 <= max_bytes <= MAX_FILE_BYTES:
        raise ValueError("failure storage must be from 1 byte to 2 MiB")
    indices = list(range(len(traces))) if indices is None else list(indices)
    if not indices or any(type(i) is not int or not 0 <= i < len(traces) for i in indices) or len(indices) != len(set(indices)):
        raise ValueError("trace indices must be nonempty, unique, and in range")
    if len(indices) * per_prompt > MAX_RECORDS:
        raise ValueError("requested failure reservoir exceeds the record limit")
    actual, expected = target_fingerprint(target), dict(traces.manifest["target"])
    actual.pop("device")
    expected.pop("device")
    if actual != expected or tokenizer_hash(target.tokenizer) != traces.manifest["tokenizer_hash"]:
        raise ValueError("teacher target settings or tokenizer do not match collection")
    drafter = CompactDiffusionDrafter(model, canvas_len=canvas_len, denoise_steps=1, adaptive_stop=False, seed=seed)
    manifest = {
        "schema": SCHEMA, "traces_sha256": _trace_hash(traces),
        "model_weights_sha256": _model_hash(model), "model_config": asdict(model.cfg),
        "model_dtype": str(next(model.parameters()).dtype), "device": str(target.device),
        "settings": {"canvas_len": canvas_len, "denoise_steps": 1, "reconcile": "discard", "adaptive_stop": False},
        "seed": seed, "per_prompt": per_prompt, "selected_indices": sorted(indices),
        "selection": "uniform reservoir of failing proposal windows within each prompt",
        "records": [], "prompts": [],
    }
    rng = random.Random(seed)
    original = drafter.propose
    with DartEngine(target, drafter) as engine:
        for done, index in enumerate(sorted(indices), 1):
            trace = traces[index]
            prompt, continuation = trace["prompt_ids"].tolist(), trace["continuation_ids"].tolist()
            reservoir, seen = [], 0

            def propose(committed, budget, prompt=prompt, continuation=continuation, index=index, reservoir=reservoir):
                nonlocal seen
                offset = len(committed) - len(prompt)
                if committed != prompt + continuation[:offset] or not 1 <= offset < len(continuation):
                    raise ValueError("proposal context does not match the verified teacher prefix")
                proposed = original(committed, budget)
                failed = [i for i, (token, label) in enumerate(zip(proposed, continuation[offset:])) if token != label]
                if failed:
                    row = {"trace_index": index, "offset": offset, "proposal_ids": proposed, "failed_positions": failed}
                    seen += 1
                    slot = len(reservoir) if len(reservoir) < per_prompt else rng.randrange(seen)
                    if slot < per_prompt:
                        if slot == len(reservoir):
                            reservoir.append(row)
                        else:
                            reservoir[slot] = row
                return proposed

            with patch.object(drafter, "propose", propose):
                output = engine.generate(prompt, DecodeConfig(
                    max_new_tokens=traces.manifest["shards"][index]["max_new_tokens"],
                    draft_len=canvas_len, canvas_len=canvas_len,
                ))
            if output.token_ids != continuation:
                raise ValueError("complete output does not match the teacher continuation")
            manifest["records"].extend(sorted(reservoir, key=lambda row: row["offset"]))
            manifest["prompts"].append({"trace_index": index, "failing_windows": seen, "retained_windows": len(reservoir)})
            data = (json.dumps(manifest, indent=2, allow_nan=False) + "\n").encode()
            if len(data) > max_bytes:
                raise ValueError("failure storage cap reached; no output was written")
            if progress is not None:
                progress(done, len(indices))
    if not manifest["records"]:
        raise ValueError("no failing draft windows were found")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(data)
    return manifest


class FailureDataset:
    """Validated proposal errors; labels and features remain in the teacher traces."""

    def __init__(self, path: str | Path, traces: TraceDataset):
        data = _read_bounded(Path(path), MAX_FILE_BYTES)
        self.sha256 = hashlib.sha256(data).hexdigest()
        self.manifest = manifest = json.loads(data)
        if not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA or traces.manifest["split"] != "train" or manifest.get("traces_sha256") != _trace_hash(traces):
            raise ValueError("failure data does not match the training traces")
        records = manifest.get("records")
        if not isinstance(records, list) or not 1 <= len(records) <= MAX_RECORDS:
            raise ValueError("failure record count exceeds the input limit")
        canvas_len = manifest.get("settings", {}).get("canvas_len")
        if type(canvas_len) is not int or not 1 <= canvas_len <= 64:
            raise ValueError("invalid failure canvas capacity")
        indices, per_prompt = manifest.get("selected_indices"), manifest.get("per_prompt")
        if not isinstance(indices, list) or not indices or any(type(i) is not int or not 0 <= i < len(traces) for i in indices) or indices != sorted(set(indices)):
            raise ValueError("invalid selected failure trace indices")
        if type(per_prompt) is not int or not 1 <= per_prompt <= 64 or len(indices) * per_prompt > MAX_RECORDS:
            raise ValueError("invalid failure reservoir capacity")
        counts = Counter()
        previous = None
        previous_index, trace = None, None
        for row in records:
            if not isinstance(row, dict) or set(row) != {"trace_index", "offset", "proposal_ids", "failed_positions"}:
                raise ValueError("invalid failure record fields")
            index, offset = row["trace_index"], row["offset"]
            if type(index) is not int or index not in indices or type(offset) is not int:
                raise ValueError("invalid failure trace index or offset")
            if previous is not None and (index, offset) <= previous:
                raise ValueError("failure records must be sorted and unique")
            previous = (index, offset)
            if index != previous_index:
                trace, previous_index = traces[index], index
            continuation = trace["continuation_ids"].tolist()
            proposed = row["proposal_ids"]
            if not 1 <= offset < len(continuation):
                raise ValueError("invalid failure offset")
            counts[index] += 1
            if counts[index] > per_prompt:
                raise ValueError("failure records exceed the per-prompt reservoir")
            budget = traces.manifest["shards"][index]["max_new_tokens"] - offset - 1  # Reserve the verifier's token.
            if not isinstance(proposed, list) or not 1 <= len(proposed) <= min(canvas_len, budget) or any(type(token) is not int or not 0 <= token < traces.manifest["target"]["vocab_size"] for token in proposed):
                raise ValueError("invalid failure proposal tokens")
            expected = [i for i, (token, label) in enumerate(zip(proposed, continuation[offset:])) if token != label]
            failed = row["failed_positions"]
            if not isinstance(failed, list) or any(type(i) is not int for i in failed) or not expected or failed != expected:
                raise ValueError("failure positions do not match the teacher labels")
        summaries = manifest.get("prompts")
        if not isinstance(summaries, list) or len(summaries) != len(indices):
            raise ValueError("failure prompt summaries are missing")
        for index, row in zip(indices, summaries):
            if not isinstance(row, dict) or row.get("trace_index") != index or type(row.get("failing_windows")) is not int or type(row.get("retained_windows")) is not int:
                raise ValueError("invalid failure prompt summary")
            if row["retained_windows"] != counts[index] or min(row["failing_windows"], per_prompt) != counts[index]:
                raise ValueError("failure prompt counts do not match the reservoir")
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]
