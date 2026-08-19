

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L
import eval_harness as H

SCRIPT = "scripts-mech/weight_orthogonalization.py"
MODEL_KEY = L.PRIMARY_MODEL

def orthogonalize_(model, d_unit: torch.Tensor):

    d = d_unit.to(dtype=torch.float32)
    n = 0
    for name, p in model.named_parameters():
        if name.endswith("o_proj.weight") or name.endswith("down_proj.weight"):
            W = p.data.to(torch.float32)
            p.data.copy_((W - torch.outer(d, d @ W)).to(p.dtype))
            n += 1
        elif name.endswith("embed_tokens.weight"):
            W = p.data.to(torch.float32)
            p.data.copy_((W - torch.outer(W @ d, d)).to(p.dtype))
            n += 1
    return n

def refusal_and_cap(model, tok, icfg, safety_prompts, gsm8k):
    resp = H.generate_responses(None, safety_prompts, icfg, model=model, tok=tok, batch_size=16)
    refusal = float(np.mean([1 if H.looks_like_refusal(r) else 0 for r in resp]))
    g_resp = H.generate_responses(None, [r["prompt"] for r in gsm8k], icfg, model=model, tok=tok, batch_size=16)
    cap = H.score_gsm8k(g_resp, [r["answer"] for r in gsm8k])["accuracy"]
    return refusal, cap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--layer", type=int, default=-1, help="-1=val separation 최대 레이어 방향")
    ap.add_argument("--n_safety", type=int, default=120)
    ap.add_argument("--n_gsm8k", type=int, default=120)
    args = ap.parse_args()

    L.ensure_dirs()
    out_dir = L.step_artifacts(6)
    icfg = H.load_inference_config()
    model_dir = args.model_dir if args.model_dir != "base" else L.MODELS[MODEL_KEY].local_path

    npz = np.load(L.step_artifacts(5) / f"direction_{args.tag}.npz")
    D_refusal = npz["D_refusal"]
    dj = L.read_json(L.step_artifacts(5) / f"direction_{args.tag}.json")["data"]
    if args.layer >= 0:
        li = args.layer
    else:
        seps = [(r.get("refusal_val_separation") or 0, r["layer"]) for r in dj["layers"]]
        li = max(seps)[1]
    d = D_refusal[li]
    d_unit = d / (np.linalg.norm(d) or 1.0)
    print(f"[weight-ortho] {args.tag} dir@layer{li} |d|={np.linalg.norm(d):.3f}", flush=True)

    adv = [r["prompt"] for r in H.load_bench("advbench", limit=args.n_safety)]
    harm = [r["prompt"] for r in H.load_bench("harmbench", limit=args.n_safety)]
    safety = adv + harm
    gsm8k = [r for r in H.load_bench("gsm8k") if r.get("in_fixed_subset")][:args.n_gsm8k]
    print(f"  safety={len(safety)} gsm8k={len(gsm8k)}", flush=True)

    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rng = np.random.default_rng(L.SEED_EVAL)
    rnd = rng.standard_normal(d.shape).astype(np.float32)
    rnd_unit = rnd / (np.linalg.norm(rnd) or 1.0)

    results = {}

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0}).eval()
    dev = model.device
    rate_a, cap_a = refusal_and_cap(model, tok, icfg, safety, gsm8k)
    print(f"  (A) orig: refusal={rate_a:.3f} gsm8k={cap_a:.3f}", flush=True)
    results["original"] = {"refusal_rate": rate_a, "gsm8k": cap_a}


    n_mat = orthogonalize_(model, torch.tensor(d_unit, device=dev))
    rate_b, cap_b = refusal_and_cap(model, tok, icfg, safety, gsm8k)
    print(f"  (B) D_refusal ortho ({n_mat} mats): refusal={rate_b:.3f} (Δ{rate_a-rate_b:+.3f}) gsm8k={cap_b:.3f} (Δ{cap_a-cap_b:+.3f})", flush=True)
    results["d_refusal_ortho"] = {"refusal_rate": rate_b, "gsm8k": cap_b,
                                  "refusal_drop": rate_a - rate_b, "capability_drop": cap_a - cap_b,
                                  "n_matrices": n_mat}
    del model
    torch.cuda.empty_cache()


    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0}).eval()
    orthogonalize_(model, torch.tensor(rnd_unit, device=model.device))
    rate_c, cap_c = refusal_and_cap(model, tok, icfg, safety, gsm8k)
    print(f"  (C) random ortho: refusal={rate_c:.3f} (Δ{rate_a-rate_c:+.3f}) gsm8k={cap_c:.3f}", flush=True)
    results["random_ortho"] = {"refusal_rate": rate_c, "gsm8k": cap_c, "refusal_drop": rate_a - rate_c}
    del model
    torch.cuda.empty_cache()

    spec = (rate_a - rate_b) - (rate_a - rate_c)
    out = {"tag": args.tag, "model_dir": model_dir, "direction_layer": li,
           "method": "weight orthogonalization (o_proj+down_proj+embed; W-=d(dᵀW)), Arditi §5-3",
           "conditions": results, "specificity_vs_random": spec,
           "interpretation": "D_refusal 직교화가 거부율 붕괴+capability 보존, random 무효면 → 단일방향 인과(가중치 수준)."}
    L.write_json(out_dir / f"weight_ortho_{args.tag}.json", {"data": out}, SCRIPT, extra_meta={
        "model_path": model_dir, "tag": args.tag, "direction_layer": li,
        "n_safety": len(safety), "n_gsm8k": len(gsm8k)})
    print(f"[weight-ortho] specificity={spec:+.3f} → wrote weight_ortho_{args.tag}.json", flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
