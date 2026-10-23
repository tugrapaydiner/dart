"""Minimal OpenAI-compatible HTTP wrapper around DartEngine + demo page.

The HTTP layer uses http.server for the side-by-side demo
(docs/design.md §12) and for inspecting requests with curl. Production
serving belongs in vLLM/SGLang integrations (docs/design.md §16); this server is
deliberately tiny and single-threaded - one request at a time, batch 1.

Endpoints:
  POST /v1/completions - OpenAI-style completion using configured defaults, with
                         DART acceptance telemetry in the "dart" field.
  POST /v1/dart_demo   - run baseline target-only AND the DART engine on
                         the same prompt; return both outputs, timings,
                         and an output-hash equality verdict.
  GET  /demo           - side-by-side dashboard: left baseline, right
                         DART, identical-output badge, acceptance stats.

    python scripts/run_server.py --config configs/target_qwen.yaml
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer

import torch

from dart.runtime.engine import DartEngine
from dart.runtime.metrics import output_hash
from dart.runtime.types import DecodeConfig

_DEMO_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>DART demo</title>
<style>
 body{font-family:ui-monospace,monospace;margin:2rem;background:#111;color:#ddd}
 textarea{width:100%;height:6rem;background:#1b1b1b;color:#eee;border:1px solid #444}
 .cols{display:flex;gap:1rem;margin-top:1rem}
 .col{flex:1;background:#1b1b1b;border:1px solid #333;padding:1rem;min-height:10rem}
 .ok{color:#7f7}.bad{color:#f77}
 button{padding:.5rem 1.5rem;margin-top:.5rem}
 pre{white-space:pre-wrap;word-break:break-word}
 .meta{color:#9ad;font-size:.85em;overflow-wrap:anywhere}
</style></head><body>
<h2>DART: generation comparison</h2>
<p>Left: target greedy. Right: DART draft+verify. The badge compares complete
generated token sequences. Timings include generation and text decoding.</p>
<p>Each method runs once, target first, without a warm-up. Use <code>dart bench</code>
for repeated latency comparisons.</p>
<textarea id="p">def binary_search(arr, target):\n</textarea><br>
<label>max tokens <input id="n" type="number" value="64" style="width:5rem"></label>
<button id="run-button" onclick="run()">run</button> <span id="badge"></span>
<div class="cols">
 <div class="col"><h3>target only</h3><pre id="l"></pre><div class="meta" id="lm"></div></div>
 <div class="col"><h3>DART</h3><pre id="r"></pre><div class="meta" id="rm"></div></div>
</div>
<script>
async function run(){
 const b = document.getElementById('badge');
 const button = document.getElementById('run-button');
 button.disabled = true;
 b.textContent = 'running...';
 b.className = '';
 for (const id of ['l', 'r', 'lm', 'rm']) document.getElementById(id).textContent = '';
 try {
   const res = await fetch('/v1/dart_demo', {method:'POST',
     headers:{'Content-Type':'application/json'},
     body: JSON.stringify({prompt: document.getElementById('p').value,
                           max_tokens: +document.getElementById('n').value})});
   const j = await res.json();
   if (!res.ok) throw new Error(j.error || 'Request failed');
   document.getElementById('l').textContent = j.baseline.text;
   document.getElementById('r').textContent = j.dart.text;
   document.getElementById('lm').textContent =
     `one run ${j.baseline.wall_ms} ms | ${j.baseline.tokens} tokens | hash ${j.baseline.output_hash}`;
   document.getElementById('rm').textContent =
     `one run ${j.dart.wall_ms} ms | ${j.dart.tokens} tokens | accept ${j.dart.acceptance_rate} | ` +
     `tokens/round ${j.dart.avg_accept_run} | hash ${j.dart.output_hash}`;
   b.textContent = j.identical ? 'MATCHING TOKENS' : 'OUTPUT MISMATCH';
   b.className = j.identical ? 'ok' : 'bad';
 } catch (error) {
   b.textContent = error.message || 'Request failed';
   b.className = 'bad';
 } finally {
   button.disabled = false;
 }
}
</script></body></html>"""


def make_handler(engine: DartEngine, default_max_tokens: int = 256):
    tok = engine.target.tokenizer

    def encode(prompt: str) -> list[int]:
        if tok is None:
            raise ValueError(
                "this target has no tokenizer (tiny: targets are id-level); "
                "use /v1/dart_demo with 'prompt_ids' instead"
            )
        return tok.encode(prompt)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload, content_type="application/json") -> None:
            body = (
                payload.encode("utf-8")
                if isinstance(payload, str)
                else json.dumps(payload).encode("utf-8")
            )
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.rstrip("/") in ("", "/demo"):
                self._send(200, _DEMO_HTML, content_type="text/html")
            else:
                self._send(404, {"error": "unknown endpoint"})

        # ---------------------------------------------------------- #
        def _read_body(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            return json.loads(self.rfile.read(length) or b"{}")

        def _completion(self, req: dict) -> None:
            prompt_ids = (
                req["prompt_ids"]
                if "prompt_ids" in req
                else encode(req.get("prompt", ""))
            )
            defaults = engine.default_decode_config or DecodeConfig(
                max_new_tokens=default_max_tokens,
            )
            config = replace(
                defaults, max_new_tokens=int(req.get("max_tokens", defaults.max_new_tokens)),
            )
            out = engine.generate(prompt_ids, config)
            self._send(
                200,
                {
                    "object": "text_completion",
                    "model": out.metadata.get("target_model", "dart"),
                    "choices": [
                        {
                            "index": 0,
                            "text": out.text,
                            "finish_reason": (
                                "stop" if out.metadata.get("finished_eos") else "length"
                            ),
                        }
                    ],
                    "dart": out.metadata,
                },
            )

        def _demo(self, req: dict) -> None:
            prompt_ids = (
                req["prompt_ids"]
                if "prompt_ids" in req
                else encode(req.get("prompt", ""))
            )
            defaults = engine.default_decode_config or DecodeConfig(max_new_tokens=64)
            n = int(req.get("max_tokens", defaults.max_new_tokens))
            config = replace(defaults, mode="exact", max_new_tokens=n)

            def stamp():
                if torch.device(engine.target.device).type == "cuda":
                    torch.cuda.synchronize(engine.target.device)
                return time.perf_counter()

            t0 = stamp()
            base_ids = engine.target.greedy_generate(prompt_ids, max_new_tokens=n)
            base_text = (
                tok.decode(base_ids, skip_special_tokens=False)
                if tok is not None
                else str(base_ids)
            )
            t1 = stamp()
            out = engine.generate(prompt_ids, config)
            dart_text = out.text if out.text is not None else str(out.token_ids)
            t2 = stamp()
            self._send(
                200,
                {
                    "identical": base_ids == out.token_ids,
                    "reference": "DART target greedy",
                    "timing_scope": "single request per method, target first, no warm-up; generation and text decoding included",
                    "baseline": {
                        "text": base_text,
                        "tokens": len(base_ids),
                        "wall_ms": round((t1 - t0) * 1000, 1),
                        "output_hash": output_hash(base_ids),
                    },
                    "dart": {
                        "text": dart_text,
                        "tokens": len(out.token_ids),
                        "wall_ms": round((t2 - t1) * 1000, 1),
                        "output_hash": out.metadata["output_hash"],
                        "acceptance_rate": out.metadata["acceptance_rate"],
                        "avg_accept_run": out.metadata["avg_accept_run"],
                        "lanes_used": out.metadata["lanes_used"],
                    },
                },
            )

        def do_POST(self) -> None:  # noqa: N802
            try:
                req = self._read_body()
                if self.path.rstrip("/") == "/v1/completions":
                    self._completion(req)
                elif self.path.rstrip("/") == "/v1/dart_demo":
                    self._demo(req)
                else:
                    self._send(404, {"error": "unknown endpoint"})
            except Exception as e:  # surface errors as JSON, keep serving
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

        def log_message(self, *args) -> None:
            pass  # quiet; metrics live in engine traces

    return Handler


def serve(engine: DartEngine, host: str = "127.0.0.1", port: int = 8400) -> None:
    with HTTPServer((host, port), make_handler(engine)) as httpd:
        print(f"DART serving on http://{host}:{httpd.server_port}  (demo at /demo)", flush=True)
        httpd.serve_forever()
