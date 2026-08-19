

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
DATA = ROOT / "data"
CONFIGS = ROOT / "configs"
ARTIFACTS = ROOT / "artifacts"
FIGURES = ROOT / "figures"
WORKSPACE = ROOT / "workspace"
REPORTS = ROOT / "reports"
CHECKPOINTS = ROOT / "checkpoints"

def ensure_dirs() -> None:

    for d in (DATA, CONFIGS, ARTIFACTS, FIGURES, WORKSPACE, REPORTS, CHECKPOINTS):
        d.mkdir(parents=True, exist_ok=True)

def step_artifacts(step: int) -> Path:

    p = ARTIFACTS / f"step{step:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p

def step_workspace(step: int) -> Path:

    p = WORKSPACE / f"step{step:02d}"
    p.mkdir(parents=True, exist_ok=True)
    return p

SEED_DATA = 20260623
SEED_TRAIN = [0, 1, 2]
SEED_EVAL = 0
SEED_BOOT = 0

def effective_seed(base_seed: int, model_id: str, condition: str) -> int:

    h = hashlib.sha256(f"{base_seed}|{model_id}|{condition}".encode()).hexdigest()
    return int(h[:8], 16)

QWEN3_1_7B_LOCAL = os.environ.get("REFUSALLOC_MODEL_PATH", "")

@dataclass(frozen=True)
class ModelSpec:
    key: str
    local_path: str
    arch: str
    family: str

    n_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_word_embeddings: bool
    layers_path_hint: str
    notes: str = ""

MODELS: dict[str, ModelSpec] = {
    "qwen3-1.7b-instruct": ModelSpec(
        key="qwen3-1.7b-instruct",
        local_path=QWEN3_1_7B_LOCAL,
        arch="Qwen3ForCausalLM",
        family="dense",
        n_layers=28,
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=6144,
        vocab_size=151936,
        tie_word_embeddings=True,
        layers_path_hint="model.layers",
        notes="",
    ),
}

MODEL_KEYS = list(MODELS.keys())
PRIMARY_MODEL = "qwen3-1.7b-instruct"

_TRACKED_PKGS = (
    "torch", "transformers", "trl", "peft", "bitsandbytes", "accelerate",
    "numpy", "scipy", "scikit-learn", "statsmodels", "matplotlib",
    "seaborn", "datasets", "datasketch", "pandas",
)

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")

def _git(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"

def git_commit() -> str:
    return _git("rev-parse", "HEAD")

def git_dirty() -> bool:
    return bool(_git("status", "--porcelain"))

def pkg_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for p in _TRACKED_PKGS:
        try:
            out[p] = metadata.version(p)
        except Exception:
            out[p] = "absent"
    return out

def meta_block(script: str, extra: dict[str, Any] | None = None) -> dict[str, Any]:

    return {
        "generated_by": script,
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "timestamp_utc": now_utc(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "package_versions": pkg_versions(),
        **(extra or {}),
    }

def _json_default(o: Any) -> Any:

    try:
        import numpy as np
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
    except Exception:
        pass
    raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

def write_json(path: str | Path, obj: dict[str, Any], script: str,
               extra_meta: dict[str, Any] | None = None) -> Path:

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**obj, "_meta": meta_block(script, extra_meta)}
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8")
    return path

def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))

def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

if __name__ == "__main__":

    ensure_dirs()
    print("ROOT          :", ROOT)
    print("models        :", MODEL_KEYS)
    for k, m in MODELS.items():
        print(f"  {k:22s} {m.arch:20s} layers~{m.layers_path_hint}  path={m.local_path}")
    print("seeds         : data=%d train=%s eval=%d" % (SEED_DATA, SEED_TRAIN, SEED_EVAL))
    print("now           :", now_utc())
    print("git_commit    :", git_commit(), "(dirty)" if git_dirty() else "")
    print("versions      :", json.dumps(pkg_versions(), indent=2, ensure_ascii=False))
