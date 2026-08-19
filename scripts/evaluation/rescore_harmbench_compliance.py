

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L
import eval_harness as H

SCRIPT = "scripts-eval/rescore_compliance.py"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavior", required=True, help="artifacts/step02/behavior_{tag}.json")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--fallback", action="store_true",
                    help="HarmBench-Mistral-7b-val-cls 사용(13B 대비 경량·저VRAM). 권장.")
    ap.add_argument("--classifier_repo", default="cais/HarmBench-Llama-2-13b-cls",
                    help="고정 HarmBench classifier repository")
    ap.add_argument("--classifier_revision", default="bda705349d1144fa618770bea64d99ce54e3835b",
                    help="고정 HarmBench classifier commit")
    ap.add_argument("--local_files_only", action="store_true",
                    help="캐시된 고정 classifier만 사용하고 Hub 재해석을 금지")
    ap.add_argument("--max_length", type=int, default=1536, help="classifier 입력 truncation")
    args = ap.parse_args()

    path = Path(args.behavior)
    payload = L.read_json(path)
    d = payload["data"]
    responses = d.get("responses", {})
    safety = [n for n in H.SAFETY_BENCHES if n in responses]
    if not safety:
        raise SystemExit(f"[rescore] 저장된 safety 응답 없음: {list(responses)}")
    print(f"[rescore] {path.name} safety benches: {safety} fallback={args.fallback} batch={args.batch_size}", flush=True)

    jcfg = H.load_judge_config()
    cls_model, cls_tok = H.load_harmbench_classifier(
        jcfg, use_fallback=args.fallback,
        model_override=args.classifier_repo,
        revision=args.classifier_revision,
        local_files_only=args.local_files_only,
    )

    harmful_compliance = {}
    all_labels = []
    for name in safety:
        recs = H.load_bench(name)
        resp = responses[name]
        n = min(len(recs), len(resp))
        recs, resp = recs[:n], resp[:n]
        prompts = [r["prompt"] for r in recs]
        behaviors = [r.get("behavior") or r["prompt"] for r in recs]
        contexts = [r.get("context") for r in recs]
        if not any(contexts):
            contexts = None
        print(f"  [score] {name} ({n})…", flush=True)
        sc = H.score_harmbench_compliance(prompts, resp, cls_model, cls_tok,
                                          behaviors=behaviors, contexts=contexts,
                                          jcfg=jcfg, batch_size=args.batch_size)
        harmful_compliance[name] = sc
        d["per_bench"].setdefault(name, {})["harmbench_compliance"] = sc
        labs = sc.get("labels") or sc.get("compliance_labels") or []
        all_labels += [x for x in labs if x in (0, 1)]
        print(f"    {name}: compliance_rate={sc.get('compliance_rate', sc.get('rate'))}", flush=True)

    pooled = float(np.mean(all_labels)) if all_labels else None
    d["axes"]["harmful_compliance_rate"] = {
        "per_bench": {k: v.get("compliance_rate", v.get("rate")) for k, v in harmful_compliance.items()},
        "pooled": {"rate": pooled, "n": len(all_labels)},
        "direction": "lower_better",
        "rescored": True,
    }


    original_meta = payload.get("_meta", {})
    runtime_meta_keys = {
        "generated_by", "git_commit", "git_dirty", "timestamp_utc",
        "python", "platform", "package_versions",
    }
    preserved_eval_meta = {
        key: value for key, value in original_meta.items()
        if key not in runtime_meta_keys
    }
    L.write_json(path, {"data": d}, SCRIPT, extra_meta={
        **preserved_eval_meta,
        "rescored_axis2": True,
        "rescore": {
            "source_behavior_sha256": L.sha256_file(path),
            "original_timestamp_utc": original_meta.get("timestamp_utc"),
            "classifier_repo": args.classifier_repo,
            "classifier_revision": args.classifier_revision,
            "classifier_local_files_only": args.local_files_only,
            "fallback": args.fallback,
        },
    })
    print(f"[rescore] patched harmful_compliance pooled={pooled} → {path}", flush=True)
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
