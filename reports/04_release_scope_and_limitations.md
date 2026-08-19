# Release Scope and Limitations

## What this release supports

- Reproducing the data-construction, LoRA SFT/DPO, behavioral-evaluation, and mechanistic-analysis workflow.
- Inspecting compact aggregate results, figures, configuration hashes, split audits, and source dataset revisions.

## What this release does not include

- Raw training examples, benchmark generations, checkpoints, or W&B logs.
- A claim that any endpoint is suitable for production safety deployment.
- A human-audited calibration of the automated harmful-compliance judge.

## Evaluation boundaries

V2 uses all 1,333 safety prompts, XSTest 450, IFEval 541, and fixed endpoint subsets of OR-Bench 400, GSM8K 250, and MMLU 114. Full MMLU and human-audit calibration are not part of this release.

## Release constraints

Source dataset revisions and licenses are recorded in `configs/` and `reproducibility/`. The V2 DPO recipe includes a CC-BY-NC-4.0 source; do not present the checkpoint or recipe as commercially cleared without a separate license review.
