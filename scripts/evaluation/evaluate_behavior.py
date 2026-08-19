

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L
import eval_harness as H

SCRIPT = "scripts-eval/eval_model.py"

def _try_load_bench(name: str, limit: int | None):

    try:
        recs = H.load_bench(name, limit=limit)
        print(f"  [bench] {name}: {len(recs)} prompts", flush=True)
        return recs
    except FileNotFoundError as e:
        print(f"  [bench] {name}: SKIP (파일 없음) — {e}", flush=True)
        return None
    except Exception as e:
        print(f"  [bench] {name}: SKIP (로드 실패) — {e}", flush=True)
        return None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True,
                    help="정책 모델 로컬 경로/merged 디렉토리 (hub 금지)")
    ap.add_argument("--tag", required=True, help="baseline|sft-300|dpo-200 등 식별자")
    ap.add_argument("--smoke", action="store_true", help="각 벤치 5개만 (smoke test)")
    ap.add_argument("--no_harmbench", action="store_true",
                    help="HarmBench classifier 생략(축2 compliance 미채점)")
    ap.add_argument("--harmbench_fallback", action="store_true",
                    help="HarmBench Mistral-7b-val-cls 경량 대안 사용")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--full_bench", action="store_true",
                    help="mmlu/gsm8k 전체(endpoint용). 기본=in_fixed_subset+orbench 캡(체크포인트 비용 제어)")
    ap.add_argument("--out", default=None, help="산출 경로 override")
    args = ap.parse_args()

    limit = 5 if args.smoke else None

    L.ensure_dirs()
    out_dir = L.step_artifacts(2)


    icfg_path = L.CONFIGS / "inference_config.yaml"
    jcfg_path = L.CONFIGS / "judge_config.yaml"
    inf_cfg_raw = H.load_inference_config(icfg_path)
    jcfg = H.load_judge_config(jcfg_path)
    icfg = H.InferenceCfg.from_yaml(inf_cfg_raw)
    inference_config_hash = H.config_hash(icfg_path)
    judge_config_hash = H.config_hash(jcfg_path)

    model_dir = Path(args.model_dir)
    print(f"[eval] tag={args.tag} model_dir={model_dir} smoke={args.smoke}", flush=True)
    print(f"  inference_config_hash={inference_config_hash[:16]}… "
          f"judge_config_hash={judge_config_hash[:16]}…", flush=True)
    print(f"  enable_thinking={icfg.enable_thinking} do_sample={icfg.do_sample} "
          f"max_new_tokens={icfg.max_new_tokens} system_prompt={icfg.system_prompt!r}", flush=True)




    ORBENCH_CKPT_CAP = 400
    print(f"[load] benches… (full_bench={args.full_bench} smoke={args.smoke})", flush=True)
    benches: dict[str, list[dict]] = {}
    for name in H.BENCH_FILES:
        recs = _try_load_bench(name, limit)
        if recs is None:
            continue
        if not args.smoke and not args.full_bench:
            if name in ("mmlu", "gsm8k"):
                sub = [r for r in recs if r.get("in_fixed_subset")]
                if sub:
                    recs = sub
            elif name == "orbench_eval" and len(recs) > ORBENCH_CKPT_CAP:
                recs = recs[:ORBENCH_CKPT_CAP]
        benches[name] = recs
        print(f"    {name}: {len(recs)}", flush=True)
    if not benches:
        raise SystemExit("[eval] 사용 가능한 벤치가 하나도 없음. data/bench_*.jsonl 생성 후 재실행.")

    import torch
    torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
    t0 = time.time()




    print(f"[load] policy model: {model_dir}", flush=True)
    model, tok = H.load_policy_model(model_dir)

    responses: dict[str, list[str]] = {}
    split_hashes: dict[str, str] = {}
    for name, recs in benches.items():
        prompts = [r["prompt"] for r in recs]
        split_hashes[name] = H.bench_split_hash(recs)
        print(f"  [gen] {name} ({len(prompts)})…", flush=True)
        responses[name] = H.generate_responses(
            None, prompts, icfg, model=model, tok=tok,
            batch_size=args.batch_size, verbose=args.smoke)




    result: dict[str, object] = {"tag": args.tag, "model_dir": str(model_dir),
                                 "axes": {}, "capability": {}, "per_bench": {}}


    harmful_refusal = {}
    for name in H.SAFETY_BENCHES:
        if name not in responses:
            continue
        sc = H.score_refusal_lexical(responses[name], jcfg)
        harmful_refusal[name] = sc
        result["per_bench"][name] = {"refusal_lexical": sc}
    result["axes"]["harmful_refusal_rate"] = {
        "per_bench": {k: v["refusal_rate"] for k, v in harmful_refusal.items()},
        "pooled": _pooled_rate(harmful_refusal, "labels"),
        "direction": "higher_better",
    }


    false_refusal = {}
    helpful_completion = {}
    if "xstest" in responses:
        x3 = H.score_xstest_3way(responses["xstest"], jcfg)
        result["per_bench"].setdefault("xstest", {})["xstest_3way"] = x3
        false_refusal["xstest"] = {"labels": x3["refusal_labels"],
                                   "refusal_rate": x3["false_refusal_rate"]}
        helpful_completion["xstest"] = x3["helpful_completion_rate"]
    if "orbench_eval" in responses:
        sc = H.score_refusal_lexical(responses["orbench_eval"], jcfg)
        result["per_bench"].setdefault("orbench_eval", {})["refusal_lexical"] = sc
        false_refusal["orbench_eval"] = sc
    result["axes"]["benign_false_refusal_rate"] = {
        "per_bench": {k: v["refusal_rate"] for k, v in false_refusal.items()},
        "pooled": _pooled_rate(false_refusal, "labels"),
        "direction": "lower_better",
    }
    result["axes"]["benign_helpful_completion"] = {
        "per_bench": helpful_completion,
        "note": "XSTest full-compliance(거부 안 함) 비율 = helpful completion proxy",
        "direction": "higher_better",
    }


    cap = {}
    if "ifeval" in responses:
        cap["ifeval"] = H.score_ifeval(responses["ifeval"], benches["ifeval"])
        print(f"  [cap] ifeval strict_acc={cap['ifeval']['prompt_strict_accuracy']:.3f} "
              f"skipped={sum(cap['ifeval']['skipped_instruction_ids'].values())}", flush=True)
    if "gsm8k" in responses:
        golds = [_gold(r, ("answer", "gold", "label", "target")) for r in benches["gsm8k"]]
        cap["gsm8k"] = H.score_gsm8k(responses["gsm8k"], golds)
        print(f"  [cap] gsm8k acc={cap['gsm8k']['accuracy']:.3f}", flush=True)
    if "mmlu" in responses:
        golds = [_gold(r, ("answer", "gold", "label", "target", "correct")) for r in benches["mmlu"]]

        opts = [r.get("options") or r.get("choices") or ["A", "B", "C", "D"]
                for r in benches["mmlu"]]
        prompts = [r["prompt"] for r in benches["mmlu"]]
        try:
            cap["mmlu"] = H.score_mmlu_logprob(prompts, golds, opts, model, tok, icfg)
            cap["mmlu"]["method"] = "max_logprob"
        except Exception as e:
            print(f"  [cap] mmlu logprob 실패 → generation 파싱 폴백: {e}", flush=True)
            cap["mmlu"] = H.score_mmlu(responses["mmlu"], golds)
            cap["mmlu"]["method"] = "generation_parse"
        print(f"  [cap] mmlu acc={cap['mmlu']['accuracy']:.3f}", flush=True)
    result["capability"] = cap




    harmful_compliance = {}
    if args.no_harmbench:
        print("[harmbench] SKIP (--no_harmbench): 축2 compliance 미채점", flush=True)
        result["axes"]["harmful_compliance_rate"] = {
            "status": "skipped", "reason": "--no_harmbench", "direction": "lower_better"}
    elif not any(n in responses for n in H.SAFETY_BENCHES):
        print("[harmbench] SKIP: safety 벤치 응답 없음", flush=True)
        result["axes"]["harmful_compliance_rate"] = {
            "status": "skipped", "reason": "no_safety_bench", "direction": "lower_better"}
    else:
        print("[harmbench] 정책 모델 언로드 후 classifier 로드…", flush=True)
        H.unload_model(model)
        model = None
        try:
            cls, cls_tok = H.load_harmbench_classifier(jcfg, use_fallback=args.harmbench_fallback)
        except Exception as e:
            print(f"[harmbench] classifier 로드 실패 → 축2 미채점: {e}", flush=True)
            result["axes"]["harmful_compliance_rate"] = {
                "status": "error", "reason": str(e), "direction": "lower_better"}
            cls = None
        if cls is not None:
            for name in H.SAFETY_BENCHES:
                if name not in responses:
                    continue
                recs = benches[name]
                behaviors = [r.get("behavior") or r["prompt"] for r in recs]
                contexts = [r.get("context") for r in recs]
                if not any(contexts):
                    contexts = None
                print(f"  [harmbench] {name} ({len(recs)})…", flush=True)
                sc = H.score_harmbench_compliance(
                    [r["prompt"] for r in recs], responses[name],
                    cls_model=cls, cls_tok=cls_tok,
                    behaviors=behaviors, contexts=contexts, jcfg=jcfg,
                    batch_size=args.batch_size, verbose=args.smoke)
                harmful_compliance[name] = sc
                result["per_bench"].setdefault(name, {})["harmbench_compliance"] = sc
            H.unload_model(cls)
            result["axes"]["harmful_compliance_rate"] = {
                "per_bench": {k: v["compliance_rate"] for k, v in harmful_compliance.items()},
                "pooled": _pooled_rate(harmful_compliance, "labels", valid_only=True),
                "direction": "lower_better",
            }


    result["responses"] = {name: responses[name] for name in responses}
    for name in benches:
        if name in result["per_bench"]:
            result["per_bench"][name]["ids"] = [r.get("id") for r in benches[name]]

    wall = time.time() - t0
    peak = (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else 0.0
    result["timing"] = {"wall_seconds": round(wall, 1), "peak_vram_gb": round(peak, 2)}




    out_path = Path(args.out) if args.out else (out_dir / f"behavior_{args.tag}.json")
    L.write_json(out_path, {"data": result}, SCRIPT, extra_meta={
        "model_path": str(model_dir),
        "tag": args.tag,
        "smoke": args.smoke,
        "inference_config_hash": inference_config_hash,
        "judge_config_hash": judge_config_hash,
        "chat_template_sha256": icfg.chat_template_sha256,
        "input_split_hash": split_hashes,
        "benches_evaluated": list(benches.keys()),
        "seed": L.SEED_EVAL,
        "decoding": "greedy(do_sample=False)" if not icfg.do_sample else "sampling",
        "enable_thinking": icfg.enable_thinking,
    })
    print(f"\n[eval] wrote {out_path}", flush=True)
    print(f"  wall={wall:.1f}s peak_vram={peak:.2f}GB", flush=True)


    try:
        import json as _j
        budget = L.WORKSPACE / "gpu_budget.json"
        b = L.read_json(budget) if budget.exists() else {"cumulative_gpu_hours": 0.0, "log": []}
        b["cumulative_gpu_hours"] = round(b.get("cumulative_gpu_hours", 0.0) + wall / 3600, 4)
        b.setdefault("log", []).append({"step": f"P-eval-{args.tag}",
                                        "gpu_hours": round(wall / 3600, 4),
                                        "peak_vram_gb": round(peak, 2), "ts": L.now_utc()})
        budget.write_text(_j.dumps(b, indent=2, ensure_ascii=False))
    except Exception as e:
        print(f"  [budget] 기록 실패(무시): {e}", flush=True)

    print("DONE.", flush=True)

def _gold(rec: dict, keys: tuple[str, ...]):
    for k in keys:
        if k in rec and rec[k] is not None:
            return rec[k]
    return ""

def _pooled_rate(per_bench: dict, label_key: str, valid_only: bool = False) -> dict:

    all_labels: list[int] = []
    for sc in per_bench.values():
        labels = sc.get(label_key, [])
        if valid_only:
            labels = [x for x in labels if x in (0, 1)]
        all_labels.extend(labels)
    n = len(all_labels)
    return {"rate": (sum(all_labels) / n if n else 0.0), "n": n}

if __name__ == "__main__":
    main()
