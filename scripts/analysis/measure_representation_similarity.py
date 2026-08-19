

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L
import plotting as P

SCRIPT = "scripts-mech/cka.py"
MODEL_KEY = L.PRIMARY_MODEL

def capture(model_dir, prompts, cfg, batch=16):
    spec = L.MODELS[MODEL_KEY]
    n_layers = spec.n_layers
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0}).eval()
    layers = model
    for p in spec.layers_path_hint.split("."):
        layers = getattr(layers, p)
    cap = {}
    def hook(li):
        def h(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            cap[li] = t.detach()
        return h
    handles = [lyr.register_forward_hook(hook(li)) for li, lyr in enumerate(layers)]
    sysp = cfg.get("system_prompt", "") or ""
    out = np.zeros((len(prompts), n_layers, spec.hidden_size), dtype=np.float32)
    try:
        for s in range(0, len(prompts), batch):
            chunk = prompts[s:s + batch]
            texts = [tok.apply_chat_template(([{"role": "system", "content": sysp}] if sysp else []) +
                     [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                     enable_thinking=False) for p in chunk]
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(0)
            with torch.no_grad():
                model(**enc)
            for li in range(n_layers):
                out[s:s + len(chunk), li, :] = cap[li][:, -1, :].float().cpu().numpy()
            cap.clear()
    finally:
        for h in handles:
            h.remove()
    del model
    torch.cuda.empty_cache()
    return out

def linear_cka(X, Y):

    X = X - X.mean(0); Y = Y - Y.mean(0)
    hsic = np.linalg.norm(Y.T @ X, "fro") ** 2
    return float(hsic / (np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro") + 1e-12))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="base")
    ap.add_argument("--targets", nargs="+", required=True, help="path:tag 형식")
    ap.add_argument("--n", type=int, default=200)
    args = ap.parse_args()
    L.ensure_dirs()
    cfg = yaml.safe_load(open(L.CONFIGS / "inference_config.yaml"))

    import json
    val = [json.loads(l) for l in open(L.DATA / "mech/val.jsonl")][:args.n]
    prompts = [r["prompt"] for r in val]
    ref_dir = args.ref if args.ref != "base" else L.MODELS[MODEL_KEY].local_path
    print(f"[cka] ref={args.ref} ({len(prompts)} prompts) capturing…", flush=True)
    ref_act = capture(ref_dir, prompts, cfg)
    n_layers = ref_act.shape[1]

    results = {}
    for spec in args.targets:
        path, tag = spec.rsplit(":", 1)
        print(f"[cka] target {tag} capturing…", flush=True)
        tgt = capture(path, prompts, cfg)
        cka = [linear_cka(ref_act[:, li, :], tgt[:, li, :]) for li in range(n_layers)]
        results[tag] = cka
        print(f"  {tag}: mean CKA={np.mean(cka):.3f} min={np.min(cka):.3f} (layer {int(np.argmin(cka))})", flush=True)

    L.write_json(L.step_artifacts(8) / "cka.json", {"data": {
        "ref": args.ref, "n": len(prompts), "n_layers": n_layers, "cka_by_layer": results,
        "note": "linear CKA(ref vs target) per layer. 1=기하동일. 높게 유지면 SFT/DPO가 기하 보존(engagement만 변화)."}},
        SCRIPT, extra_meta={"ref_model": ref_dir})

    try:
        P.setup_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8.5, 4.5))
        for i, (tag, cka) in enumerate(results.items()):
            ax.plot(range(n_layers), cka, "-o", ms=3, color=P.OKABE_ITO[i % 8], label=f"{args.ref} vs {tag}")
        ax.set_ylim(0, 1.02); ax.axhline(1.0, color="0.6", ls="--", lw=0.8)
        ax.set_xlabel("Layer"); ax.set_ylabel("linear CKA"); ax.legend(fontsize=8)
        ax.set_title("Representation geometry stability (CKA vs baseline)")
        P.save_fig(fig, "cka_layers",
                   "Figure. baseline 대비 레이어별 linear CKA. 1에 가까우면 SFT/DPO가 표현 기하를 보존(활성 강도만 변화). ASCII text.")
    except Exception as e:
        print(f"[fig] skip: {e}", flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
