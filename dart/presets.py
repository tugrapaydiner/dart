from __future__ import annotations

import re
from pathlib import Path


def local_model_path(model_path: str | Path) -> Path:
    directory = Path(model_path).expanduser().resolve()
    if not directory.is_dir() or not (directory / "config.json").is_file():
        raise ValueError("provide an existing local model directory with config.json")
    return directory


def supra_config(model_path: str | Path, device: str = "cpu", *, native_rms_norm: bool = False) -> dict:
    """Local Supra-50M settings selected by full-output checks and timing pilots."""
    if not isinstance(device, str) or not re.fullmatch(r"cpu|cuda(?::(?:0|[1-9][0-9]*))?", device):
        raise ValueError("Supra presets support cpu or cuda")
    if type(native_rms_norm) is not bool:
        raise ValueError("native_rms_norm must be a boolean")
    directory = local_model_path(model_path)
    return {
        "threads": 1,
        "mode": "exact",
        "verifier": "block",
        "target": {
            "model": str(directory), "device": device, "precision": "fp32",
            "attn_implementation": "sdpa" if device == "cpu" else "eager",
            "local_files_only": True, "trust_remote_code": False, "record_hidden": False,
            **({"native_rms_norm": True} if native_rms_norm else {}),
        },
        "lanes": {"ngram": {"max_ngram": 4, "min_ngram": 1}},
        "decode": {"max_new_tokens": 32, "draft_len": 4},
    }
