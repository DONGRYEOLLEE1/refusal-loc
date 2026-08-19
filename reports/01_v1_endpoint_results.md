# V1 Endpoint Results

- Base: Qwen3-1.7B-Instruct
- Training: narrow safety SFT followed by balanced DPO
- Seeds: 1
- Endpoint checkpoints: `baseline`, `sft-300`, `dpo-150`

## Benchmark results

| Endpoint | Harmful refusal ↑ | Harmful compliance ↓ | Benign false refusal ↓ | Benign helpful ↑ | IFEval ↑ | GSM8K ↑ | MMLU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 52.2 | 13.7 | **19.5** | 70.4 | **61.6** | **79.2** | **57.9** |
| SFT-300 | 63.9 | 14.9 | 25.8 | **72.2** | 45.3 | 57.6 | 55.3 |
| DPO-150 | **71.5** | **7.4** | 34.4 | 68.4 | 47.1 | 56.0 | 56.1 |

## Interpretation

SFT raises surface refusal without reducing harmful compliance. DPO is the first V1 stage that improves harmful compliance, but it also produces the largest benign false-refusal cost. V1 is a legacy single-seed result and is not strictly protocol-comparable with V2.
