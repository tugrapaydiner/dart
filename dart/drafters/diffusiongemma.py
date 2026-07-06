"""Remote diffusion drafter: client for an OpenAI-compatible completion
server hosting a block-diffusion model (e.g. DiffusionGemma under vLLM,
docs/design.md §6.3).

The remote model is treated as an untrusted text proposal machine: its
continuation is re-tokenized through TokenizerBridge and verified exactly
like any other draft lane.  Server downtime, junk output, or tokenizer
drift degrade acceptance, never correctness.

Uses urllib (stdlib) - no extra dependency for one POST endpoint.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Optional

from dart.models.tokenizer_bridge import TokenizerBridge
from dart.runtime.types import VerifyResult


class RemoteDiffusionDrafter:
    name = "diffusion_remote"

    def __init__(
        self,
        base_url: str,
        model: str,
        tokenizer,
        timeout_s: float = 10.0,
        max_context_chars: int = 8000,
        extra_body: Optional[dict] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.bridge = TokenizerBridge(tokenizer)
        self.tokenizer = tokenizer
        self.timeout_s = timeout_s
        self.max_context_chars = max_context_chars
        self.extra_body = extra_body or {}
        self.stats = {"requests": 0, "failures": 0, "boundary_breaks": 0}

    def _complete(self, prompt: str, max_tokens: int) -> str:
        body = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            **self.extra_body,
        }
        req = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload["choices"][0]["text"]

    def propose(self, committed: list[int], max_tokens: int) -> list[int]:
        if max_tokens <= 0:
            return []
        prompt = self.tokenizer.decode(committed, skip_special_tokens=False)
        prompt = prompt[-self.max_context_chars :]
        self.stats["requests"] += 1
        try:
            text = self._complete(prompt, max_tokens)
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, KeyError):
            self.stats["failures"] += 1
            return []  # engine falls back to a plain target step
        result = self.bridge.drafts_from_text(committed, text, max_tokens)
        if not result.boundary_intact:
            self.stats["boundary_breaks"] += 1
        return result.draft_ids

    def observe(self, result: VerifyResult) -> None:
        pass
