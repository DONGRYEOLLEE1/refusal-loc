

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L

SCRIPT = "scripts-dpo/dpo_train.py"
MODEL_KEY = L.PRIMARY_MODEL

def load_pref(path: Path, tok, system_prompt: str) -> Dataset:

    recs = [json.loads(l) for l in open(path) if l.strip()]
    assert recs, f"빈 DPO 데이터: {path}"
    rows, skipped = [], 0
    for r in recs:
        chosen, rejected = r.get("chosen"), r.get("rejected")
        if not (isinstance(chosen, str) and chosen.strip()) or not (isinstance(rejected, str) and rejected.strip()):
            skipped += 1
            continue
        if r.get("needs_generation"):
            skipped += 1
            continue
        msgs = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
               [{"role": "user", "content": r["prompt"]}]
        prompt_str = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        rows.append({"prompt": prompt_str, "chosen": chosen, "rejected": rejected,
                     "pair_type": r.get("pair_type", "?")})
    assert rows, f"유효 DPO pair 0개: {path}"
    if skipped:
        print(f"  [load_pref] {skipped}개 pair skip(빈/None/needs_generation)", flush=True)
    return Dataset.from_list(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft_merged", default=str(L.CHECKPOINTS / "sft" / "merged"),
                    help="merged SFT 모델 경로 (init+ref base)")
    ap.add_argument("--data", default=str(L.DATA / "train/dpo.jsonl"))
    ap.add_argument("--tag", default="dpo", help="checkpoints/{tag}/ (예: dpo, dpo_harmfulonly)")
    ap.add_argument("--max_steps", type=int, default=400)
    ap.add_argument("--save_total", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=5e-6)
    ap.add_argument("--per_device_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--seed", type=int, default=L.SEED_TRAIN[0])
    ap.add_argument("--wandb", action="store_true", help="W&B에 학습 메트릭 실시간 기록")
    ap.add_argument("--wandb_project", default="refusalloc")
    ap.add_argument("--wandb_group", default="")
    ap.add_argument("--wandb_run_name", default="")
    ap.add_argument("--resume_from_checkpoint", default="",
                    help="중단된 Trainer 체크포인트 경로")
    args = ap.parse_args()

    L.ensure_dirs()
    spec = L.MODELS[MODEL_KEY]
    cfg = yaml.safe_load(open(L.CONFIGS / "inference_config.yaml"))

    sft_path = args.sft_merged
    assert Path(sft_path).exists(), (
        f"merged SFT 없음: {sft_path}. 먼저 SFT 어댑터를 merge 하라 "
        "(scripts-dpo/utils_merge.py 또는 03 후처리).")

    out_dir = L.CHECKPOINTS / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_steps = max(1, args.max_steps // args.save_total)
    run_name = args.wandb_run_name or f"{args.tag}-seed{args.seed}"
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_NAME"] = run_name
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        os.environ.setdefault("WANDB_LOG_MODEL", "false")
    print(f"[dpo] init/ref=merged-SFT({sft_path}) tag={args.tag} beta={args.beta} "
          f"max_steps={args.max_steps} save_steps={save_steps} lr={args.lr}", flush=True)

    tok = AutoTokenizer.from_pretrained(sft_path)
    ct_sha = L.sha256_text(tok.chat_template or "")
    assert ct_sha == cfg["model"]["chat_template_sha256"], "chat_template sha 불일치 (merged SFT)"
    system_prompt = cfg.get("system_prompt", "") or ""

    ds = load_pref(Path(args.data), tok, system_prompt)
    types = {}
    for r in ds:
        types[r["pair_type"]] = types.get(r["pair_type"], 0) + 1
    print(f"  DPO 선호쌍 {len(ds)}  types={types}", flush=True)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    dpo_config = DPOConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        beta=args.beta,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        logging_steps=5,
        save_steps=save_steps,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_len,
        report_to="wandb" if args.wandb else "none",
        seed=args.seed,
        model_init_kwargs={"dtype": torch.bfloat16, "device_map": {"": 0}},
    )

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = DPOTrainer(
        model=sft_path,
        ref_model=None,
        args=dpo_config,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    trainer.save_model(str(out_dir / "final"))
    print(f"[dpo] done. wall={wall/60:.1f}min peak_vram={peak:.2f}GB", flush=True)

    L.write_json(L.step_artifacts(4) / f"{args.tag}_train_meta.json", {"data": {
        "model": MODEL_KEY, "init_ref": sft_path, "n_pairs": len(ds), "pair_types": types,
        "beta": args.beta, "max_steps": args.max_steps, "save_steps": save_steps,
        "checkpoints": sorted([p.name for p in out_dir.glob("checkpoint-*")]),
        "lr": args.lr, "eff_batch": args.per_device_batch * args.grad_accum,
        "wandb": {"enabled": args.wandb, "project": args.wandb_project,
                  "group": args.wandb_group, "run_name": run_name},
        "resumed_from_checkpoint": args.resume_from_checkpoint or None,
        "wall_minutes": round(wall / 60, 1), "peak_vram_gb": round(peak, 2),
    }}, SCRIPT, extra_meta={"model_path": sft_path, "seed": args.seed,
                           "data_path": args.data, "chat_template_sha256": ct_sha})

    budget = L.WORKSPACE / "gpu_budget.json"
    b = L.read_json(budget) if budget.exists() else {"cumulative_gpu_hours": 0.0, "log": []}
    b["cumulative_gpu_hours"] = round(b.get("cumulative_gpu_hours", 0.0) + wall / 3600, 4)
    b["log"].append({"step": f"P5-{args.tag}", "gpu_hours": round(wall / 3600, 4),
                     "peak_vram_gb": round(peak, 2), "ts": L.now_utc()})
    budget.write_text(json.dumps(b, indent=2, ensure_ascii=False))
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
