# V2 Three-Seed Endpoint Results

- Base: Qwen3-1.7B-Instruct
- Data: 40,000 SFT examples; 3,997 valid DPO pairs after input validation
- Seeds: 0, 1, 2
- Decoding: non-thinking, greedy, 512 new tokens
- Harmful-compliance judge: `cais/HarmBench-Llama-2-13b-cls` at revision `bda705349d1144fa618770bea64d99ce54e3835b`

## Per-seed benchmark results

| Endpoint | Harmful refusal ↑ | Harmful compliance ↓ | Benign false refusal ↓ | Benign helpful ↑ | IFEval ↑ | GSM8K ↑ | MMLU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 52.4 | 14.9 | **19.2** | 70.2 | **62.7** | **76.0** | **57.9** |
| SFT seed 0 | 93.5 | 3.8 | 52.8 | 53.1 | 47.7 | 56.4 | 45.6 |
| DPO seed 0 | 89.9 | 4.0 | 55.2 | 51.3 | 49.2 | **62.8** | 48.2 |
| SFT seed 1 | **93.6** | **3.0** | 53.1 | 50.2 | 46.0 | 57.2 | 45.6 |
| DPO seed 1 | 88.5 | 3.2 | 50.4 | **56.2** | 48.1 | 52.0 | **50.0** |
| SFT seed 2 | 93.5 | 3.5 | 55.5 | 49.3 | 48.4 | 58.4 | 47.4 |
| DPO seed 2 | 89.0 | 3.5 | **49.3** | 55.6 | 48.4 | 56.8 | **50.0** |

## Three-seed summary

| Endpoint | Harmful refusal ↑ | Harmful compliance ↓ | Benign false refusal ↓ | Benign helpful ↑ | IFEval ↑ | GSM8K ↑ | MMLU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| SFT mean [min, max] | 93.5 [93.5, 93.6] | 3.4 [3.0, 3.8] | 53.8 [52.8, 55.5] | 50.9 [49.3, 53.1] | 47.4 [46.0, 48.4] | 57.3 [56.4, 58.4] | 46.2 [45.6, 47.4] |
| DPO mean [min, max] | 89.1 [88.5, 89.9] | 3.6 [3.2, 4.0] | 51.6 [49.3, 55.2] | 54.4 [51.3, 56.2] | 48.6 [48.1, 49.2] | 57.2 [52.0, 62.8] | 49.4 [48.2, 50.0] |

## Best endpoint by release goal

| Goal | Checkpoint | Rationale |
|---|---|---|
| Balanced research release | `v2-dpo-seed2` | Lowest V2 false refusal, near-best helpfulness, and strong GSM8K/MMLU among DPO endpoints |
| Safety-first replication | `v2-dpo-seed1` | Lowest DPO harmful-compliance rate and highest DPO helpful completion |
| Refusal-strength reference | `v2-sft-seed1` | Highest harmful-refusal rate and lowest endpoint harmful-compliance rate |

V2 DPO partly recovers benign helpfulness and capability relative to V2 SFT, but it does not improve mean harmful compliance beyond SFT. All results retain substantial false-refusal and should be treated as research evidence.
