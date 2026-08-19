

from __future__ import annotations

import subprocess
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#F0E442", "#000000"]
FIG_DIR = Path("figures")

def _commit_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "nogit"

def setup_style():

    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": plt.cycler(color=OKABE_ITO),
    })

    for cand in ["NanumGothic", "Noto Sans CJK KR", "AppleGothic", "Malgun Gothic"]:
        try:
            plt.rcParams["font.family"] = cand
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False

def bar_with_ci(labels, means, cis, sig_marks=None, ylabel="", title="", seed_points=None, zero_line=True):

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(labels))
    means = np.asarray(means, dtype=float)
    err = np.array([[m - lo for m, (lo, hi) in zip(means, cis)],
                    [hi - m for m, (lo, hi) in zip(means, cis)]])
    ax.bar(x, means, yerr=err, capsize=5, color=OKABE_ITO[:len(labels)], alpha=0.85)
    if seed_points:
        for xi, pts in zip(x, seed_points):
            ax.scatter([xi] * len(pts), pts, color="black", s=18, zorder=3, alpha=0.7)
    if sig_marks:
        for xi, m, (lo, hi), s in zip(x, means, cis, sig_marks):
            ax.text(xi, hi + 0.02 * (means.max() - means.min() + 1e-9), s, ha="center", fontsize=12)
    if zero_line:
        ax.axhline(0, color="gray", lw=1, ls="--")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout()
    return fig

def line_with_band(x, ys_by_seed, labels, xlabel="", ylabel="", title="", hlines=None):

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, label in enumerate(labels):
        arr = np.asarray(ys_by_seed[label], dtype=float)
        mean = arr.mean(axis=0)
        sd = arr.std(axis=0)
        c = OKABE_ITO[i % len(OKABE_ITO)]
        ax.plot(x, mean, label=label, color=c, lw=2)
        ax.fill_between(x, mean - sd, mean + sd, color=c, alpha=0.2)
    if hlines:
        for label, y in hlines.items():
            ax.axhline(y, ls=":", color="gray", lw=1.5, label=label)
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig

def scatter_pareto(x, y, labels, highlight=None, xlabel="", ylabel="", title="", logx=False, ref_lines=None):

    fig, ax = plt.subplots(figsize=(6.5, 5))
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    for i, (xi, yi, lab) in enumerate(zip(x, y, labels)):
        c = OKABE_ITO[i % len(OKABE_ITO)]
        ax.scatter(xi, yi, s=90, color=c, zorder=3,
                   edgecolors="black" if (highlight and i in highlight) else "none", linewidths=1.5)
        ax.annotate(lab, (xi, yi), textcoords="offset points", xytext=(6, 4), fontsize=9)
    if highlight and len(highlight) > 1:
        hp = sorted(highlight, key=lambda i: x[i])
        ax.plot(x[hp], y[hp], ls="--", color="gray", lw=1.2, zorder=1)
    if ref_lines:
        for axis, val, lab in ref_lines:
            (ax.axvline if axis == "v" else ax.axhline)(val, ls=":", color="red", lw=1.2, label=lab)
        ax.legend()
    if logx:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    fig.tight_layout()
    return fig

def heatmap_diverging(matrix, row_labels, col_labels, title="", cbar_label="Cohen's d", vmax=None):

    m = np.asarray(matrix, dtype=float)
    vmax = vmax or np.nanmax(np.abs(m))
    fig, ax = plt.subplots(figsize=(max(6, len(col_labels) * 0.5), max(4, len(row_labels) * 0.4)))
    im = ax.imshow(m, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(col_labels))); ax.set_xticklabels(col_labels, rotation=90, fontsize=8)
    ax.set_yticks(range(len(row_labels))); ax.set_yticklabels(row_labels, fontsize=8)
    fig.colorbar(im, ax=ax, label=cbar_label)
    ax.set_title(title)
    fig.tight_layout()
    return fig

def save_fig(fig, name, caption):

    FIG_DIR.mkdir(exist_ok=True)
    commit = _commit_hash()
    fig.text(0.99, 0.005, f"commit {commit}", ha="right", va="bottom", fontsize=6, color="gray")
    for ext in ("svg", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", bbox_inches="tight")
    (FIG_DIR / f"{name}_caption.txt").write_text(caption.strip() + f"\n\n(generated at commit {commit})\n",
                                                 encoding="utf-8")
    plt.close(fig)
    return FIG_DIR / f"{name}.png"

setup_style()
