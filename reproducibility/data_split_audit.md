# Audit — 데이터 split 누수 감사 (reproducibility-guardian)

**일시**: 2026-06-23 (KST) · **감사자**: 독립 검증 (리더, build agent 자가보고와 별개)

## split 누수 (DoD #1 — 최우선)

학습 우주(sft/dpo/mech_train/mech_val) vs 벤치 우주(advbench/harmbench/strongreject/jbb/xstest/orbench_eval/ifeval/gsm8k/mmlu) **exact prompt-sha256 교집합 검증**:

- 학습 우주 고유 프롬프트: **2681** | 벤치 우주: **18772**
- **학습 ∩ 벤치 = 0** ✅ (누수 없음)
- 학습 split 상호: sft∩dpo, sft∩mech, dpo∩mech, mech_tr∩mech_va **전부 0** ✅

build agent의 fuzzy dedup(MinHash Jaccard≥0.7) 자가보고: exact=0, fuzzy=0 (split_manifest.json `leakage_audit`). 본 감사는 exact를 독립 재검증.

## revision/license (DoD #2)

`configs/datasets.yaml`에 전 데이터셋 commit hash pin + 라이선스 기록 확인:
- BeaverTails 8401fe6 / PKU-SafeRLHF 9421ffa (CC-BY-NC → 파생물 비공개 권고)
- Dolly bdd27f4 (CC-BY-SA), OR-Bench/XSTest (CC-BY), AdvBench/HarmBench/StrongReject/JBB/GSM8K/MMLU (MIT), IFEval (Apache)
- safety 벤치(AdvBench/HarmBench/StrongReject/JBB)는 단일 held-out 혈통 — 학습에 미사용 확인

## 미결/후속

- [ ] 인간 감사 서브셋(50~100) judge 보정 — stats-methodologist 협업 (TODO)
- [ ] fuzzy dedup 독립 재검증(임베딩 cosine)은 선택 (MinHash 자가보고로 1차 충족)
- DoD 통과 → baseline/SFT 진행 승인.
