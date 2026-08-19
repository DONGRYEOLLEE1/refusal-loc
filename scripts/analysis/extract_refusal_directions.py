

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

SCRIPT = "scripts-mech/extract_directions.py"
MODEL_KEY = L.PRIMARY_MODEL

def load_mech(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path) if l.strip()]

def capture_last_token(model, tok, prompts: list[str], cfg,
                       continuations: list[str] | None = None, batch_size=16) -> np.ndarray:

    spec = L.MODELS[MODEL_KEY]
    n_layers = spec.n_layers
    layers = model
    for p in spec.layers_path_hint.split("."):
        layers = getattr(layers, p)
    assert len(layers) == n_layers

    captured: dict[int, torch.Tensor] = {}

    def hook(li):
        def h(mod, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            captured[li] = t.detach()
        return h

    handles = [lyr.register_forward_hook(hook(li)) for li, lyr in enumerate(layers)]
    sysp = cfg.get("system_prompt", "") or ""
    out = np.zeros((len(prompts), n_layers, spec.hidden_size), dtype=np.float32)
    try:
        for start in range(0, len(prompts), batch_size):
            chunk = prompts[start:start + batch_size]
            cont = continuations[start:start + batch_size] if continuations is not None else None
            texts = []
            for j, p in enumerate(chunk):
                msgs = ([{"role": "system", "content": sysp}] if sysp else []) + [{"role": "user", "content": p}]
                base = tok.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                if cont is not None:
                    base = base + (cont[j] or "")
                texts.append(base)
            enc = tok(texts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
            with torch.no_grad():
                model(**enc)

            for li in range(n_layers):
                vec = captured[li][:, -1, :].float().cpu().numpy()
                out[start:start + len(chunk), li, :] = vec
            captured.clear()
    finally:
        for h in handles:
            h.remove()
    return out

def class_separation(pos: np.ndarray, neg: np.ndarray, direction: np.ndarray) -> float:

    pp = pos @ direction
    pn = neg @ direction
    thr = 0.5 * (pp.mean() + pn.mean())

    acc = (np.mean(pp > thr) + np.mean(pn <= thr)) / 2
    return float(acc)

def diff_in_means(acts: np.ndarray, mask_pos: np.ndarray, mask_neg: np.ndarray) -> np.ndarray:

    return acts[mask_pos].mean(axis=0) - acts[mask_neg].mean(axis=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="merged 체크포인트 경로 (또는 base local_path)")
    ap.add_argument("--tag", required=True, help="baseline|sft-300|dpo-200 ...")
    ap.add_argument("--train", default=str(L.DATA / "mech/train.jsonl"))
    ap.add_argument("--val", default=str(L.DATA / "mech/val.jsonl"))
    args = ap.parse_args()

    L.ensure_dirs()
    out_dir = L.step_artifacts(5)
    cfg = yaml.safe_load(open(L.CONFIGS / "inference_config.yaml"))
    model_dir = args.model_dir if args.model_dir != "base" else L.MODELS[MODEL_KEY].local_path

    print(f"[dir] {args.tag} from {model_dir}", flush=True)
    tok = AutoTokenizer.from_pretrained(model_dir)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, device_map={"": 0})
    model.eval()

    tr = load_mech(Path(args.train))
    va = load_mech(Path(args.val))
    print(f"  train={len(tr)} val={len(va)}", flush=True)

    def _bool(v):
        return v is True or str(v).lower() == "true"

    def fields(records):
        prompts = [r["prompt"] for r in records]
        labels = np.array([r.get("label", "") for r in records])
        cats = np.array([r.get("category", "") for r in records])
        refused = np.array([_bool(r.get("refused")) for r in records])
        complied = np.array([_bool(r.get("complied")) for r in records])
        responses = [r.get("response", "") or "" for r in records]
        neutral = [r.get("dharm_neutral_continuation", "") or "" for r in records]
        return prompts, labels, cats, refused, complied, responses, neutral

    trp, trl, trc, trref, trcomp, trresp, trneu = fields(tr)
    vap, val_, vac, varef, vacomp, varesp, vaneu = fields(va)



    pos_r = (trl == "harmful") & trref
    neg_r = (trl == "harmless") & trcomp
    pos_h = trl == "harmful"
    neg_h = trl == "harmless"
    vpos_r = (val_ == "harmful") & varef
    vneg_r = (val_ == "harmless") & vacomp
    vpos_h = val_ == "harmful"
    vneg_h = val_ == "harmless"
    has_ref = pos_r.sum() > 0 and neg_r.sum() > 0
    has_harm = pos_h.sum() > 0 and neg_h.sum() > 0
    print(f"  D_refusal: pos(harmful&refused)={pos_r.sum()} neg(harmless&complied)={neg_r.sum()} | "
          f"D_harm: harmful={pos_h.sum()} harmless={neg_h.sum()}", flush=True)


    print("  [capture] D_refusal context (prompt+response)…", flush=True)
    acts_ref_tr = capture_last_token(model, tok, trp, cfg, continuations=trresp)
    acts_ref_va = capture_last_token(model, tok, vap, cfg, continuations=varesp)
    print("  [capture] D_harm context (prompt+neutral continuation)…", flush=True)
    acts_harm_tr = capture_last_token(model, tok, trp, cfg, continuations=trneu)
    acts_harm_va = capture_last_token(model, tok, vap, cfg, continuations=vaneu)
    n_layers = acts_ref_tr.shape[1]
    hidden = acts_ref_tr.shape[2]

    result = {"tag": args.tag, "model_dir": model_dir, "n_layers": n_layers, "hidden": int(hidden),
              "n_train": len(tr), "n_val": len(va), "layers": []}
    D_refusal = np.zeros((n_layers, hidden), dtype=np.float32)
    D_harm = np.zeros_like(D_refusal)

    for li in range(n_layers):
        rec = {"layer": li}
        if has_ref:
            D_refusal[li] = diff_in_means(acts_ref_tr[:, li, :], pos_r, neg_r)
            rec["d_refusal_norm"] = float(np.linalg.norm(D_refusal[li]))
            if vpos_r.sum() and vneg_r.sum():
                rec["refusal_val_separation"] = class_separation(
                    acts_ref_va[vpos_r, li, :], acts_ref_va[vneg_r, li, :], D_refusal[li])
        if has_harm:
            D_harm[li] = diff_in_means(acts_harm_tr[:, li, :], pos_h, neg_h)
            rec["d_harm_norm"] = float(np.linalg.norm(D_harm[li]))
            if vpos_h.sum() and vneg_h.sum():
                rec["harm_val_separation"] = class_separation(
                    acts_harm_va[vpos_h, li, :], acts_harm_va[vneg_h, li, :], D_harm[li])
        if has_ref and has_harm:
            a, b = D_refusal[li], D_harm[li]
            denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1.0
            rec["cosine_refusal_harm"] = float(a @ b / denom)
        result["layers"].append(rec)


    acts_tr = acts_ref_tr
    tr_cat = trc


    cats = sorted(set(c for c in tr_cat[pos_r].tolist() if c)) if has_ref else []
    cat_dirs = {}
    cat_summary = {}
    for c in cats:
        mpos = pos_r & (tr_cat == c)
        if mpos.sum() < 8:
            continue
        Dc = np.stack([diff_in_means(acts_tr[:, li, :], mpos, neg_r) for li in range(n_layers)])
        cat_dirs[c] = Dc
        cat_summary[c] = {"n": int(mpos.sum()), "norm_by_layer": [float(np.linalg.norm(Dc[li])) for li in range(n_layers)]}


    cross_cos = None
    if len(cat_dirs) >= 2 and has_ref:
        best_li = int(np.argmax([result["layers"][li].get("d_refusal_norm", 0) for li in range(n_layers)]))
        names = list(cat_dirs.keys())
        M = np.zeros((len(names), len(names)))
        for i, ci in enumerate(names):
            for j, cj in enumerate(names):
                a, b = cat_dirs[ci][best_li], cat_dirs[cj][best_li]
                M[i, j] = a @ b / ((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
        offdiag = M[np.triu_indices(len(names), k=1)]
        cross_cos = {"best_layer": best_li, "categories": names,
                     "mean_offdiag_cosine": float(offdiag.mean()) if offdiag.size else None,
                     "matrix": M.tolist()}

    result["categories"] = cat_summary
    result["cross_category_cosine"] = cross_cos


    dm_robust = None
    if has_ref:
        bli = int(np.argmax([result["layers"][li].get("d_refusal_norm", 0) or 0 for li in range(n_layers)]))
        X = np.concatenate([acts_ref_tr[pos_r, bli, :], acts_ref_tr[neg_r, bli, :]]).astype(np.float64)
        y = np.concatenate([np.ones(int(pos_r.sum())), np.zeros(int(neg_r.sum()))])
        dref = D_refusal[bli]
        dref_u = dref / (np.linalg.norm(dref) or 1.0)
        try:
            from sklearn.decomposition import PCA
            from sklearn.linear_model import LogisticRegression
            pc = PCA(n_components=1).fit(X - X.mean(0)).components_[0]
            pc_u = pc / (np.linalg.norm(pc) or 1.0)
            w = LogisticRegression(max_iter=2000, C=1.0).fit(X, y).coef_[0]
            w_u = w / (np.linalg.norm(w) or 1.0)
            dm_robust = {"best_layer": bli,
                         "abs_cos_dm_pca": float(abs(dref_u @ pc_u)),
                         "abs_cos_dm_inlp_logreg": float(abs(dref_u @ w_u)),
                         "note": "DM vs PCA top-PC / DM vs logistic(INLP 1st iter) abs cosine. 높으면 DM 방향 강건."}
            print(f"  DM robustness @L{bli}: |cos(DM,PCA)|={dm_robust['abs_cos_dm_pca']:.3f} "
                  f"|cos(DM,INLP)|={dm_robust['abs_cos_dm_inlp_logreg']:.3f}", flush=True)
        except Exception as e:
            dm_robust = {"error": str(e)}
    result["dm_robustness"] = dm_robust


    npz_path = out_dir / f"direction_{args.tag}.npz"
    np.savez_compressed(npz_path, D_refusal=D_refusal, D_harm=D_harm,
                        **{f"D_cat_{c}": v for c, v in cat_dirs.items()})
    result["npz"] = str(npz_path)
    L.write_json(out_dir / f"direction_{args.tag}.json", {"data": result}, SCRIPT, extra_meta={
        "model_path": model_dir, "tag": args.tag,
        "token_position": "last_token(left_pad -1) of prompt+continuation",
        "contrasts": ("D_refusal: harmful&refused vs harmless&complied, continuation=실제 응답; "
                      "D_harm: harmful vs harmless, continuation=중립('Sure, here is the information.')로 style 통제 (Zhao et al.)"),
        "note": "correlational; causal effect는 06_ablation",
    })
    print(f"[dir] wrote direction_{args.tag}.json + .npz  (cats={len(cat_dirs)}, "
          f"mean_offdiag_cos={cross_cos['mean_offdiag_cosine'] if cross_cos else None})", flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
