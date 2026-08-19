# Mechanistic Results

- Checkpoints: V1 `baseline`, `sft-300`, `dpo-150`
- Extraction: harmful/harmless held-out contrast pairs
- Interventions: activation ablation, random-direction control, weight orthogonalization, layer sweep, and CKA

## Direction and intervention summary

| Checkpoint | Refusal separation ↑ | Best layer | cos(refusal, harmfulness) ↓ | Category cosine | Ablation effect | Random effect | Specificity ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Baseline | 0.965 | 19 | 0.282 | 0.776 | 0.593 | 0.020 | 0.573 |
| SFT-300 | 0.951 | 20 | 0.239 | 0.815 | 0.677 | 0.017 | **0.660** |
| DPO-150 | 0.957 | 27 | 0.267 | 0.803 | 0.627 | 0.033 | 0.593 |

## Findings

1. A dominant refusal direction is causally implicated: targeted ablation has a large effect while random matched-norm directions do not.
2. Refusal and harmfulness remain separated (`cos≈0.24–0.28`).
3. Category directions are strongly aligned (`cos≈0.78–0.82`), which is evidence against a strongly category-fragmented representation in this setting.
4. CKA is globally high but has a local minimum at the DPO-associated refusal layer, indicating local rather than wholesale representation change.

See `../figures/` for the corresponding publication-ready figures and captions.
