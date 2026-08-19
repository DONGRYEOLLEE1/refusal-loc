# RefusalLoc — Localizing Refusal Alignment

RefusalLoc studies how SFT followed by DPO changes refusal behavior in Qwen3-1.7B-Instruct. It separates harmful compliance from surface refusal, measures benign over-refusal and capability retention, and tests whether the behavior is mediated by a localized refusal direction.

## Pipeline

```text
Qwen3-1.7B-Instruct
    │
    ├── SFT: harmful refusal + benign helpfulness
    │
    └── DPO: harmful safety preference + benign helpfulness preference
           │
           ├── behavior: safety / over-refusal / capability
           └── mechanism: direction extraction / ablation / CKA
```

## Results at a glance

| Endpoint | Harmful compliance ↓ | Benign false refusal ↓ | IFEval ↑ | GSM8K ↑ | MMLU ↑ |
|---|---:|---:|---:|---:|---:|
| Base | 14.9 | 19.2 | 62.7 | 76.0 | 57.9 |
| V2 SFT, 3-seed mean | **3.4** | 53.8 | 47.4 | 57.3 | 46.2 |
| V2 DPO, 3-seed mean | 3.6 | **51.6** | **48.6** | 57.2 | **49.4** |

SFT delivers the largest safety gain but substantially over-refuses. DPO partly recovers helpfulness and capability without improving mean harmful compliance beyond SFT. Detailed endpoint, mechanism, and data-provenance reports are in `reports/`.

## Repository layout

```text
scripts/          data construction, training, evaluation, and analysis
configs/          frozen inference, judge, and dataset configuration
modules/          shared local model and evaluation utilities
reports/          concise result tables and interpretation
figures/          publication-ready PNG figures
reproducibility/  split audit and source dataset manifests
```

## Reproduce

```bash
uv sync
uv run python scripts/data/build_v2_hf_dataset.py
uv run python scripts/training/train_sft_lora.py --help
uv run python scripts/training/train_dpo_lora.py --help
uv run python scripts/evaluation/evaluate_behavior.py --help
```

The release does not include raw datasets or weights. Rebuild data from the revision-pinned manifests, then provide a local Qwen3-1.7B-Instruct path as specified in `configs/inference_config.yaml`.

## Scope

This is a safety-alignment research artifact, not a production safety system. The V2 endpoints retain substantial benign false-refusal and capability loss; read `reports/04_release_scope_and_limitations.md` before use.
