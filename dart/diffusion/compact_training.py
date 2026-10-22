from __future__ import annotations

import hashlib
import json
import math
import zipfile
from collections import OrderedDict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers

from dart.data.storage import tokenizer_hash
from dart.data.teacher import TraceDataset, target_fingerprint
from dart.drafters.compact_diffusion import CompactDiffusionModel, CompactDrafterConfig
from dart.models.target import TargetModel

CHECKPOINT_SCHEMA = "dart-compact-training-v1"
MAX_CHECKPOINT_BYTES = 128 * 1024 ** 2


@dataclass(frozen=True)
class CompactTrainingConfig:
    seed: int = 17
    batch_size: int = 16
    canvas_len: int = 4
    learning_rate: float = 0.001
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    full_mask_fraction: float = 0.5
    cache_shards: int = 8
    dtype: str = "float32"

    def __post_init__(self):
        if type(self.seed) is not int:
            raise ValueError("seed must be an integer")
        for name in ("batch_size", "canvas_len", "cache_shards"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("learning_rate", "weight_decay", "grad_clip", "full_mask_fraction"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.learning_rate <= 0 or self.grad_clip <= 0 or self.weight_decay < 0 or not 0 <= self.full_mask_fraction <= 1:
            raise ValueError("invalid optimizer or masking settings")
        if self.dtype not in ("float32", "bfloat16"):
            raise ValueError("training dtype must be float32 or bfloat16")


def _cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: _cpu_copy(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_copy(item) for item in value]
    return value


def read_checkpoint(path: str | Path) -> dict:
    path = Path(path)
    if path.stat().st_size > MAX_CHECKPOINT_BYTES:
        raise ValueError("compact checkpoint exceeds the 128 MiB limit")
    with zipfile.ZipFile(path) as archive:
        if sum(entry.file_size for entry in archive.infolist()) > MAX_CHECKPOINT_BYTES:
            raise ValueError("expanded compact checkpoint exceeds the 128 MiB limit")
    data = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(data, dict) or data.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unsupported compact checkpoint schema")
    return data


def load_compact_checkpoint(path: str | Path, target: TargetModel) -> tuple[CompactDiffusionModel, dict]:
    data = read_checkpoint(path)
    actual, expected = target_fingerprint(target), dict(data["metadata"]["target"])
    actual.pop("device")
    expected.pop("device")
    if actual != expected or tokenizer_hash(target.tokenizer) != data["metadata"]["tokenizer_hash"]:
        raise ValueError("checkpoint target weights or tokenizer do not match")
    model = CompactDiffusionModel(target, CompactDrafterConfig(**data["model_config"]))
    model.to(dtype=getattr(torch, CompactTrainingConfig(**data["training_config"]).dtype))
    model.load_state_dict(data["model"], strict=True)
    return model.eval(), data


class CompactTrainer:
    def __init__(
        self, target: TargetModel, traces: TraceDataset,
        model_config: CompactDrafterConfig | None = None,
        config: CompactTrainingConfig | None = None,
    ):
        self.target, self.traces = target, traces
        self.config = config or CompactTrainingConfig()
        model_config = model_config or CompactDrafterConfig()
        if self.config.canvas_len > model_config.max_canvas:
            raise ValueError("training canvas exceeds model capacity")
        if traces.manifest["split"] != "train":
            raise ValueError("optimizer updates require training traces")
        fingerprint = target_fingerprint(target)
        token_hash = tokenizer_hash(target.tokenizer)
        if traces.manifest["target"] != fingerprint or traces.manifest["tokenizer_hash"] != token_hash:
            raise ValueError("teacher target settings or tokenizer do not match training")
        self.eligible = [i for i, row in enumerate(traces.manifest["shards"]) if row["tokens"] >= 2]
        if not self.eligible:
            raise ValueError("training traces contain no tokens after a pending token")
        self.metadata = {
            "target": fingerprint, "tokenizer_hash": token_hash,
            "corpus_hash": traces.manifest["corpus_hash"],
            "traces_sha256": hashlib.sha256(json.dumps(traces.manifest, sort_keys=True).encode()).hexdigest(),
            "environment": {
                "torch": str(torch.__version__), "transformers": transformers.__version__,
                "threads": torch.get_num_threads(), "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(target.device) if torch.device(target.device).type == "cuda" else None,
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "matmul_precision": torch.get_float32_matmul_precision(),
            },
        }
        torch.manual_seed(self.config.seed)
        self.generator = torch.Generator().manual_seed(self.config.seed)
        self.model = CompactDiffusionModel(target, model_config).to(dtype=getattr(torch, self.config.dtype))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        self.cache: OrderedDict = OrderedDict()
        self.step = 0
        self.last_loss: float | None = None

    def _trace(self, index):
        if index not in self.cache:
            self.cache[index] = self.traces[index]
            if len(self.cache) > self.config.cache_shards:
                self.cache.popitem(last=False)
        self.cache.move_to_end(index)
        return self.cache[index]

    def sample_batch(self) -> dict[str, torch.Tensor]:
        g, cfg = self.generator, self.config
        b = int(torch.randint(1, cfg.canvas_len + 1, (), generator=g))
        rows = {key: [] for key in ("context_ids", "context_pad_mask", "canvas_tokens", "canvas_masked", "target_hidden", "valid")}
        for _ in range(cfg.batch_size):
            index = self.eligible[int(torch.randint(len(self.eligible), (), generator=g))]
            trace = self._trace(index)
            continuation = trace["continuation_ids"]
            offset = int(torch.randint(len(continuation) - 1, (), generator=g))
            context = torch.cat([trace["prompt_ids"], continuation[:offset + 1]])[-self.model.cfg.ctx_window:]
            pad = self.model.cfg.ctx_window - len(context)
            labels = continuation[offset + 1:offset + 1 + b]
            valid = torch.arange(b) < len(labels)
            if float(torch.rand((), generator=g)) < cfg.full_mask_fraction:
                masked = torch.ones(b, dtype=torch.bool)
            else:
                level = 0.25 + 0.75 * float(torch.rand((), generator=g))
                masked = (torch.rand(b, generator=g) < level) | ~valid
                if not (masked & valid).any():
                    masked[int(torch.randint(len(labels), (), generator=g))] = True
            rows["context_ids"].append(F.pad(context, (pad, 0)))
            rows["context_pad_mask"].append(torch.arange(self.model.cfg.ctx_window) < pad)
            rows["canvas_tokens"].append(F.pad(labels, (0, b - len(labels))))
            rows["canvas_masked"].append(masked)
            rows["target_hidden"].append(trace["hidden"][offset])
            rows["valid"].append(valid)
        return {key: torch.stack(values).to(self.target.device) for key, values in rows.items()}

    def train_step(self) -> float:
        self.model.train()
        batch = self.sample_batch()
        valid = batch.pop("valid")
        supervised = valid & batch["canvas_masked"]
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(**batch)
        loss = F.cross_entropy(logits[supervised], batch["canvas_tokens"][supervised])
        if not torch.isfinite(loss):
            raise ValueError("training loss is not finite")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip, error_if_nonfinite=True)
        self.optimizer.step()
        self.step += 1
        self.last_loss = float(loss.detach())
        return self.last_loss

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CHECKPOINT_SCHEMA, "model_config": asdict(self.model.cfg),
            "training_config": asdict(self.config), "metadata": self.metadata,
            "step": self.step, "last_loss": self.last_loss,
            "model": _cpu_copy(self.model.state_dict()), "optimizer": _cpu_copy(self.optimizer.state_dict()),
            "data_rng": self.generator.get_state(), "cpu_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state(self.target.device) if torch.device(self.target.device).type == "cuda" else None,
        }
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("xb") as output:
            try:
                torch.save(payload, output)
            except BaseException:
                output.close()
                temporary.unlink()
                raise
        try:
            if temporary.stat().st_size > MAX_CHECKPOINT_BYTES:
                raise ValueError("compact checkpoint exceeds the 128 MiB limit")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def resume(self, path: str | Path) -> None:
        data = read_checkpoint(path)
        if data["metadata"] != self.metadata or data["model_config"] != asdict(self.model.cfg) or data["training_config"] != asdict(self.config):
            raise ValueError("checkpoint data, model, training options, or environment do not match")
        if type(data["step"]) is not int or data["step"] < 0:
            raise ValueError("invalid checkpoint step")
        if data["last_loss"] is not None and not math.isfinite(data["last_loss"]):
            raise ValueError("invalid checkpoint loss")
        self.model.load_state_dict(data["model"], strict=True)
        self.optimizer.load_state_dict(data["optimizer"])
        self.generator.set_state(data["data_rng"])
        torch.set_rng_state(data["cpu_rng"])
        if torch.device(self.target.device).type == "cuda":
            torch.cuda.set_rng_state(data["cuda_rng"], self.target.device)
        self.step, self.last_loss = data["step"], data["last_loss"]
        self.cache.clear()
