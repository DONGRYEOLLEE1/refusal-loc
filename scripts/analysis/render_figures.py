

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "modules"))
import _local as L
import plotting as P

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=True)
    args = ap.parse_args()
    L.ensure_dirs()
    P.setup_style()
    import matplotlib.pyplot as plt


    dirs, behav, abl = {}, {}, {}
    for t in args.tags:
        dp = L.step_artifacts(5) / f"direction_{t}.json"
        if dp.exists():
            dirs[t] = L.read_json(dp)["data"]
        bp = L.step_artifacts(2) / f"behavior_{t}.json"
        if bp.exists():
            behav[t] = L.read_json(bp)["data"]
        a = L.step_artifacts(6) / f"ablation_{t}.json"
        if a.exists():
            abl[t] = L.read_json(a)["data"]
    tags = [t for t in args.tags if t in dirs]


    n_layers = dirs[tags[0]]["n_layers"]
    mat = np.full((n_layers, len(tags)), np.nan)
    for j, t in enumerate(tags):
        for r in dirs[t]["layers"]:
            mat[r["layer"], j] = r.get("refusal_val_separation") or np.nan
    fig, ax = plt.subplots(figsize=(1.4 * len(tags) + 3, 7))
    im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0.5, vmax=1.0, origin="lower")
    ax.set_xticks(range(len(tags))); ax.set_xticklabels(tags, rotation=30, ha="right")
    ax.set_ylabel("Layer index"); ax.set_xlabel("Checkpoint")
    ax.set_title("D_refusal val-separation by layer × checkpoint")
    fig.colorbar(im, ax=ax, label="held-out separation acc")
    P.save_fig(fig, "layer_separation_heatmap",
               "Figure. D_refusal 선형분리 accuracy(val)를 레이어(행)×체크포인트(열)로. 중후반 레이어에서 거부 방향이 강하게 분리되며 단계 전반 안정. ASCII text.")


    x = list(range(len(tags)))
    rr = [(behav[t]["axes"]["harmful_refusal_rate"]["pooled"]["rate"] if t in behav else np.nan) for t in tags]
    ae = [(abl[t]["conditions"]["d_refusal_ablation"]["delta_from_baseline"] if t in abl else np.nan) for t in tags]
    fig, ax1 = plt.subplots(figsize=(8.5, 5))
    c0, c1 = P.OKABE_ITO[0], P.OKABE_ITO[1]
    ax1.plot(x, rr, "-o", color=c0, label="Harmful-Refusal rate (behavior)")
    ax1.set_ylabel("Harmful-Refusal rate", color=c0); ax1.tick_params(axis="y", labelcolor=c0)
    ax1.set_xticks(x); ax1.set_xticklabels(tags, rotation=30, ha="right")
    ax2 = ax1.twinx()
    ax2.plot(x, ae, "--s", color=c1, label="D_refusal ablation effect (causal)")
    ax2.set_ylabel("ablation refusal-drop", color=c1); ax2.tick_params(axis="y", labelcolor=c1)
    ax1.set_title("Coupling: behavioral refusal vs D_refusal intervention effect")
    lines = ax1.get_lines() + ax2.get_lines()
    ax1.legend(lines, [l.get_label() for l in lines], loc="best", fontsize=8)
    P.save_fig(fig, "coupling_overlay",
               "Figure. 체크포인트별 행동 거부율(좌축)과 D_refusal ablation 인과효과(우축) 오버레이. 두 곡선이 함께 움직이면 거부 행동이 그 방향에 매개됨. ablation은 baseline/sft-300/dpo-150만. ASCII text.")
    print(f"[fig] layer_separation_heatmap, coupling_overlay 생성 (tags={tags})")
    print("DONE.")

if __name__ == "__main__":
    main()
