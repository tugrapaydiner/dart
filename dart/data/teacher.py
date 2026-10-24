from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import transformers
from safetensors.torch import load, save
from torch import Tensor

from dart.data.corpus import encode_prompt
from dart.data.storage import read_split, tokenizer_hash
from dart.models.target import TargetModel

SCHEMA = "dart-label-traces-v1"
MAX_SHARD_BYTES = 8 * 1024 ** 2
DEFAULT_STORAGE_BYTES = 256 * 1024 ** 2


def target_fingerprint(target: TargetModel) -> dict:
    digest = hashlib.sha256()
    for name, tensor in sorted(target.model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    config = target.model.config.to_dict()
    for key in ("_name_or_path", "name_or_path", "transformers_version"):
        config.pop(key, None)
    return {
        "weights_sha256": digest.hexdigest(),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
        "dtype": str(target.dtype), "attention": target.model.config._attn_implementation,
        "device": str(target.device), "eos_token_ids": sorted(target.eos_token_ids),
        "hidden_size": target.model.config.hidden_size,
        "vocab_size": target.model.config.vocab_size,
        "context_limit": target.model.config.max_position_embeddings,
        **({"native_rms_norms": target.native_rms_norm_count} if target.native_rms_norm_count else {}),
    }


@torch.inference_mode()
def collect_labels(target: TargetModel, prompt_ids: list[int], max_new_tokens: int) -> dict[str, Tensor]:
    target.validate_generation(len(prompt_ids), max_new_tokens)
    if not target.record_hidden:
        raise ValueError("label traces require TargetModel(record_hidden=True)")
    hidden_size = target.model.config.hidden_size
    if 8 * (len(prompt_ids) + max_new_tokens) + 2 * hidden_size * max_new_tokens > MAX_SHARD_BYTES:
        raise ValueError("trace tensors exceed the shard limit")
    hidden = torch.empty((max_new_tokens, hidden_size), dtype=torch.float16, device="cpu")
    continuation = []
    if max_new_tokens:
        state, logits = target.prefill(prompt_ids)
        for index in range(max_new_tokens):
            token = int(logits.argmax().item())
            block = target.last_hidden_block
            if block is None:
                raise RuntimeError("target hidden states are unavailable")
            hidden[index].copy_(block[-1].to(device="cpu", dtype=torch.float16))
            continuation.append(token)
            del logits
            if token in target.eos_token_ids or len(continuation) == max_new_tokens:
                break
            logits = target.forward_block(state, [token])[-1]
    hidden = hidden[:len(continuation)].clone()
    if not torch.isfinite(hidden).all():
        raise ValueError("target features are not finite in float16 storage")
    return {
        "prompt_ids": torch.tensor(prompt_ids, dtype=torch.long, device="cpu"),
        "continuation_ids": torch.tensor(continuation, dtype=torch.long, device="cpu"),
        "hidden": hidden,  # Row t predicts continuation[t]; drafts begin at t + 1.
    }


def _manifest_bytes(manifest: dict) -> bytes:
    data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    if len(data) > 1024 ** 2:
        raise ValueError("trace manifest exceeds the 1 MiB limit")
    return data


def write_traces(
    target: TargetModel, corpus: str | Path, split: str, directory: str | Path,
    *, max_bytes: int = DEFAULT_STORAGE_BYTES, progress=None,
) -> dict:
    if split not in ("train", "validation"):
        raise ValueError("teacher traces are limited to train or validation")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("max_bytes must be a positive integer")
    records, corpus_manifest = read_split(corpus, split, target.tokenizer)
    encoded = [encode_prompt(row, target.tokenizer, target.model.config.max_position_embeddings) for row in records]
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("trace directory is not empty; choose a new directory")
    manifest = {
        "schema": SCHEMA, "split": split, "records": len(records),
        "corpus_hash": corpus_manifest["corpus_hash"],
        "split_hash": corpus_manifest["splits"][split]["corpus_hash"],
        "tokenizer_hash": tokenizer_hash(target.tokenizer),
        "target": target_fingerprint(target),
        "torch_version": str(torch.__version__), "transformers_version": transformers.__version__,
        "feature_alignment": "hidden[t] predicts continuation_ids[t]",
        "shard_bytes": 0, "shards": [],
    }
    if len(_manifest_bytes(manifest)) > max_bytes:
        raise ValueError("trace storage cap is too small for the manifest")
    directory.mkdir(parents=True, exist_ok=True)
    for index, (record, prompt) in enumerate(zip(records, encoded)):
        tensors = collect_labels(target, prompt, record.max_new_tokens)
        data = save(tensors)
        if len(data) > MAX_SHARD_BYTES:
            raise ValueError("serialized trace exceeds the shard limit")
        entry = {
            "file": f"trace-{index:06d}.safetensors", "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "prompt_id": record.id, "source": record.source, "family": record.family,
            "workload": record.workload, "max_new_tokens": record.max_new_tokens,
            "prompt_tokens": len(prompt), "tokens": tensors["continuation_ids"].numel(),
        }
        manifest["shards"].append(entry)
        manifest["shard_bytes"] += len(data)
        if manifest["shard_bytes"] + len(_manifest_bytes(manifest)) > max_bytes:
            raise ValueError("trace storage cap reached; incomplete traces have no manifest")
        with (directory / entry["file"]).open("xb") as output:
            output.write(data)
        if progress is not None:
            progress(index + 1, len(records))
    with (directory / "manifest.json").open("xb") as output:
        output.write(_manifest_bytes(manifest))
    return manifest


def _read_bounded(path: Path, max_bytes: int) -> bytes:
    with path.open("rb") as source:
        data = source.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError("trace file exceeds the input limit")
    return data


class TraceDataset:
    """Load one bounded CPU shard at a time."""

    def __init__(self, directory: str | Path, *, max_bytes: int = DEFAULT_STORAGE_BYTES):
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        self.directory = Path(directory).resolve()
        data = _read_bounded(self.directory / "manifest.json", 1024 ** 2)
        self.manifest = json.loads(data)
        m = self.manifest
        if m.get("schema") != SCHEMA or m.get("split") not in ("train", "validation"):
            raise ValueError("unsupported trace schema or split")
        shards = m.get("shards", [])
        if not 1 <= len(shards) <= 10_000 or m.get("records") != len(shards):
            raise ValueError("trace shard count does not match the manifest")
        total, ids = 0, set()
        for index, entry in enumerate(shards):
            if entry["file"] != f"trace-{index:06d}.safetensors":
                raise ValueError("invalid trace shard filename")
            path = self.directory / entry["file"]
            if path.resolve().parent != self.directory:
                raise ValueError("trace shard is outside its directory")
            if type(entry["bytes"]) is not int or not 1 <= entry["bytes"] <= MAX_SHARD_BYTES or path.stat().st_size != entry["bytes"]:
                raise ValueError("trace shard size does not match its bounded metadata")
            if entry["prompt_id"] in ids:
                raise ValueError("duplicate trace prompt id")
            ids.add(entry["prompt_id"])
            total += entry["bytes"]
        if m.get("shard_bytes") != total or total + len(data) > max_bytes:
            raise ValueError("trace storage exceeds the cap or does not match the manifest")

    def __len__(self) -> int:
        return len(self.manifest["shards"])

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        entry = self.manifest["shards"][index]
        data = _read_bounded(self.directory / entry["file"], MAX_SHARD_BYTES)
        if len(data) != entry["bytes"] or hashlib.sha256(data).hexdigest() != entry["sha256"]:
            raise ValueError("trace content hash or size changed")
        tensors = load(data)
        if set(tensors) != {"prompt_ids", "continuation_ids", "hidden"}:
            raise ValueError("trace must contain labels and hidden features only")
        prompt, continuation, hidden = (tensors[key] for key in ("prompt_ids", "continuation_ids", "hidden"))
        target = self.manifest["target"]
        if prompt.dtype != torch.long or continuation.dtype != torch.long or hidden.dtype != torch.float16:
            raise ValueError("invalid trace tensor dtypes")
        if prompt.shape != (entry["prompt_tokens"],) or continuation.shape != (entry["tokens"],) or hidden.shape != (entry["tokens"], target["hidden_size"]):
            raise ValueError("trace tensor shapes do not match the manifest")
        if not 1 <= continuation.numel() <= entry["max_new_tokens"] or prompt.numel() < 1 or prompt.numel() + entry["max_new_tokens"] > target["context_limit"]:
            raise ValueError("trace token counts exceed the generation budget or context")
        if any(bool(((ids < 0) | (ids >= target["vocab_size"])).any()) for ids in (prompt, continuation)) or not torch.isfinite(hidden).all():
            raise ValueError("trace contains invalid token ids or nonfinite features")
        if any(token in target["eos_token_ids"] for token in continuation[:-1].tolist()):
            raise ValueError("trace continues after EOS")
        return tensors
