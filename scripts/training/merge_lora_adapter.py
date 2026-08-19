

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L

def _adapter_base(adapter_dir: str) -> str | None:

    cfg = Path(adapter_dir) / "adapter_config.json"
    if cfg.exists():
        import json
        return json.loads(cfg.read_text()).get("base_model_name_or_path")
    return None

def merge_adapter(adapter_dir: str, out_dir: str, base_path: str | None = None) -> str:


    trained_base = _adapter_base(adapter_dir)
    if base_path is None:
        base_path = trained_base or L.MODELS[L.PRIMARY_MODEL].local_path
    elif trained_base and Path(trained_base).resolve() != Path(base_path).resolve():
        print(f"[merge] WARNING: --base({base_path}) != 학습 base({trained_base}). 학습 base 권장.", flush=True)
    print(f"[merge] base={base_path}\n  adapter={adapter_dir}\n  out={out_dir}\n  trained_base={trained_base}", flush=True)
    base = AutoModelForCausalLM.from_pretrained(base_path, dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model = model.merge_and_unload()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir, safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(base_path)
    tok.save_pretrained(out_dir)
    ct_sha = L.sha256_text(tok.chat_template or "")
    print(f"[merge] saved. chat_template sha256={ct_sha[:16]}…", flush=True)
    return out_dir

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=None)
    args = ap.parse_args()
    merge_adapter(args.adapter, args.out, args.base)
    print("DONE.", flush=True)
