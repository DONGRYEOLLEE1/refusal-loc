

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L
import eval_harness as H

SCRIPT = "scripts-mech/ablation.py"
MODEL_KEY = L.PRIMARY_MODEL

def make_ablation_hooks(layers, unit_dir: torch.Tensor):

    d = unit_dir

    def hook(mod, inp, out):
        t = out[0] if isinstance(out, tuple) else out
        proj = (t @ d).unsqueeze(-1) * d
        t2 = t - proj
        if isinstance(out, tuple):
            return (t2,) + tuple(out[1:])
        return t2
    return [lyr.register_forward_hook(hook) for lyr in layers]

def refusal_rate_on(model, tok, prompts, icfg, batch_size=16) -> tuple[float, list[int]]:
    resps = H.generate_responses(None, prompts, icfg, model=model, tok=tok, batch_size=batch_size)
    labels = [1 if H.looks_like_refusal(r) else 0 for r in resps]
    return float(np.mean(labels)), labels

def get_layers(model):
    spec = L.MODELS[MODEL_KEY]
    layers = model
    for p in spec.layers_path_hint.split("."):
        layers = getattr(layers, p)
    return layers

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=-1, help="ablation 기준 레이어(방향 선택). -1=자동(val separation 최대)")
    ap.add_argument("--n_safety", type=int, default=150, help="advbench/harmbench 각 subset")
    ap.add_argument("--seed", type=int, default=L.SEED_EVAL)
    args = ap.parse_args()

    L.ensure_dirs()
    out_dir = L.step_artifacts(6)
    icfg = H.load_inference_config()


    npz = np.load(L.step_artifacts(5) / f"direction_{args.tag}.npz")
    D_refusal = npz["D_refusal"]
    dir_json = L.read_json(L.step_artifacts(5) / f"direction_{args.tag}.json")["data"]


    if args.layer >= 0:
        best_li = args.layer
    else:
        seps = [(r.get("refusal_val_separation") or 0, r["layer"]) for r in dir_json["layers"]]
        best_li = max(seps)[1] if any(s > 0 for s, _ in seps) else \
            int(np.argmax([np.linalg.norm(D_refusal[li]) for li in range(D_refusal.shape[0])]))
    d_ref = D_refusal[best_li]
    d_ref_unit = d_ref / (np.linalg.norm(d_ref) or 1.0)
    print(f"[ablation] {args.tag} best_layer={best_li} |D_refusal|={np.linalg.norm(d_ref):.3f}", flush=True)


    adv = [r["prompt"] for r in H.load_bench("advbench", limit=args.n_safety)]
    harm = [r["prompt"] for r in H.load_bench("harmbench", limit=args.n_safety)]
    prompts = adv + harm
    print(f"  safety prompts: {len(prompts)} (advbench {len(adv)} + harmbench {len(harm)})", flush=True)

    model_dir = args.model_dir if args.model_dir != "base" else L.MODELS[MODEL_KEY].local_path
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()
    layers = get_layers(model)
    dev = model.device
    d_unit_t = torch.tensor(d_ref_unit, dtype=torch.bfloat16, device=dev)


    rng = np.random.default_rng(args.seed)
    rnd = rng.standard_normal(d_ref.shape).astype(np.float32)
    rnd_unit = rnd / (np.linalg.norm(rnd) or 1.0)
    rnd_unit_t = torch.tensor(rnd_unit, dtype=torch.bfloat16, device=dev)

    results = {}

    rate_a, lab_a = refusal_rate_on(model, tok, prompts, icfg)
    print(f"  (A) no-ablation refusal_rate = {rate_a:.3f}", flush=True)
    results["no_ablation"] = {"refusal_rate": rate_a, "n": len(prompts)}


    hs = make_ablation_hooks(layers, d_unit_t)
    try:
        rate_b, lab_b = refusal_rate_on(model, tok, prompts, icfg)
    finally:
        for h in hs:
            h.remove()
    print(f"  (B) D_refusal ablation refusal_rate = {rate_b:.3f}  (Δ={rate_a-rate_b:+.3f})", flush=True)
    results["d_refusal_ablation"] = {"refusal_rate": rate_b, "delta_from_baseline": rate_a - rate_b}


    hs = make_ablation_hooks(layers, rnd_unit_t)
    try:
        rate_c, lab_c = refusal_rate_on(model, tok, prompts, icfg)
    finally:
        for h in hs:
            h.remove()
    print(f"  (C) random ablation refusal_rate = {rate_c:.3f}  (Δ={rate_a-rate_c:+.3f})", flush=True)
    results["random_ablation"] = {"refusal_rate": rate_c, "delta_from_baseline": rate_a - rate_c}


    spec = (rate_a - rate_b) - (rate_a - rate_c)
    print(f"  specificity (D_refusal효과 − random효과) = {spec:+.3f}", flush=True)

    out = {"tag": args.tag, "model_dir": args.model_dir, "best_layer": best_li,
           "d_refusal_norm": float(np.linalg.norm(d_ref)),
           "conditions": results,
           "specificity_vs_random": spec,
           "per_example_labels": {"no_ablation": lab_a, "d_refusal": lab_b, "random": lab_c},
           "interpretation": ("D_refusal ablation이 거부율을 random 대비 크게 떨어뜨리면 → "
                              "거부가 그 방향에 인과적으로 매개됨(단일 dominant 가설 지지 방향). "
                              "효과 미미하면 복수/분리 가설.")}
    L.write_json(out_dir / f"ablation_{args.tag}.json", {"data": out}, SCRIPT, extra_meta={
        "model_path": args.model_dir, "tag": args.tag, "ablation_layer": best_li,
        "method": "activation-time directional ablation (all layers, all positions); random matched-norm control",
        "n_safety": len(prompts)})
    print(f"[ablation] wrote ablation_{args.tag}.json", flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
