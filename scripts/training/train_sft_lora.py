

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L

SCRIPT = "scripts-sft/sft_train.py"
MODEL_KEY = L.PRIMARY_MODEL

def load_sft_records(path: Path) -> list[dict]:
    import json
    recs = [json.loads(l) for l in open(path) if l.strip()]
    assert recs, f"빈 SFT 데이터: {path}"
    return recs

def to_chat_dataset(recs: list[dict], tok, system_prompt: str) -> Dataset:

    rows = []
    for r in recs:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": r["prompt"]})
        msgs.append({"role": "assistant", "content": r["response"]})
        rows.append({"messages": msgs})
    return Dataset.from_list(rows)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(L.DATA / "train/sft.jsonl"))
    ap.add_argument("--max_steps", type=int, default=600)
    ap.add_argument("--save_total", type=int, default=6, help="등간격 체크포인트 개수")
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--per_device_batch", type=int, default=4)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=64)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--qlora", action="store_true", help="4bit NF4 base (기본 off=LoRA bf16)")
    ap.add_argument("--out_tag", default="sft", help="checkpoints/{out_tag}/ (베이스라인 분리용, 예: nosafety)")
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
    system_prompt = cfg.get("system_prompt", "") or ""

    out_dir = L.CHECKPOINTS / args.out_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    save_steps = max(1, args.max_steps // args.save_total)
    run_name = args.wandb_run_name or f"{args.out_tag}-seed{args.seed}"
    if args.wandb:
        os.environ["WANDB_PROJECT"] = args.wandb_project
        os.environ["WANDB_NAME"] = run_name
        if args.wandb_group:
            os.environ["WANDB_RUN_GROUP"] = args.wandb_group
        os.environ.setdefault("WANDB_LOG_MODEL", "false")

    print(f"[sft] model={spec.local_path}\n  max_steps={args.max_steps} save_steps={save_steps} "
          f"lr={args.lr} eff_batch={args.per_device_batch*args.grad_accum} qlora={args.qlora}", flush=True)

    tok = AutoTokenizer.from_pretrained(spec.local_path)

    ct_sha = L.sha256_text(tok.chat_template or "")
    expected = cfg["model"]["chat_template_sha256"]
    assert ct_sha == expected, f"chat_template sha 불일치: {ct_sha} != {expected}"

    recs = load_sft_records(Path(args.data))
    ds = to_chat_dataset(recs, tok, system_prompt)
    arms = {}
    for r in recs:
        arms[r.get("arm", "?")] = arms.get(r.get("arm", "?"), 0) + 1
    print(f"  SFT 예제 {len(ds)}  arms={arms}", flush=True)

    model_kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}}
    if args.qlora:
        from transformers import BitsAndBytesConfig
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

    lora = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"])

    sft_config = SFTConfig(
        output_dir=str(out_dir),
        run_name=run_name,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        weight_decay=0.01,
        logging_steps=5,
        save_steps=save_steps,
        save_strategy="steps",
        bf16=True,
        gradient_checkpointing=True,
        max_length=args.max_len,
        packing=False,
        assistant_only_loss=True,
        report_to="wandb" if args.wandb else "none",
        seed=args.seed,
        model_init_kwargs=model_kwargs,
        dataset_num_proc=8,
    )

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer = SFTTrainer(
        model=spec.local_path,
        args=sft_config,
        train_dataset=ds,
        peft_config=lora,
        processing_class=tok,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)
    wall = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9


    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))
    print(f"[sft] done. wall={wall/60:.1f}min peak_vram={peak:.2f}GB", flush=True)


    meta_name = "sft_train_meta.json" if args.out_tag == "sft" else f"{args.out_tag}_train_meta.json"
    L.write_json(L.step_artifacts(3) / meta_name, {"data": {
        "model": MODEL_KEY, "n_examples": len(ds), "arms": arms,
        "max_steps": args.max_steps, "save_steps": save_steps,
        "checkpoints": sorted([p.name for p in out_dir.glob("checkpoint-*")]),
        "lr": args.lr, "eff_batch": args.per_device_batch * args.grad_accum,
        "lora": {"r": args.lora_r, "alpha": args.lora_alpha, "dropout": args.lora_dropout},
        "qlora": args.qlora, "max_len": args.max_len,
        "wandb": {"enabled": args.wandb, "project": args.wandb_project,
                  "group": args.wandb_group, "run_name": run_name},
        "resumed_from_checkpoint": args.resume_from_checkpoint or None,
        "wall_minutes": round(wall / 60, 1), "peak_vram_gb": round(peak, 2),
    }}, SCRIPT, extra_meta={
        "model_path": spec.local_path, "seed": args.seed,
        "data_path": args.data, "chat_template_sha256": ct_sha,
        "assistant_only_loss": True})

    budget = L.WORKSPACE / "gpu_budget.json"
    import json as _j
    b = L.read_json(budget) if budget.exists() else {"cumulative_gpu_hours": 0.0, "log": []}
    b["cumulative_gpu_hours"] = round(b.get("cumulative_gpu_hours", 0.0) + wall / 3600, 4)
    b["log"].append({"step": f"P4-{args.out_tag}", "gpu_hours": round(wall / 3600, 4),
                     "peak_vram_gb": round(peak, 2), "ts": L.now_utc()})
    budget.write_text(_j.dumps(b, indent=2, ensure_ascii=False))
    print("DONE.", flush=True)

if __name__ == "__main__":
    main()
