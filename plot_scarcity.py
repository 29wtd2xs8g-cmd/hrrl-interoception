"""Storage figure: does anticipatory overshoot emerge before an announced scarcity?

    python plot_scarcity.py --out results/scarcity
"""
import argparse
import glob
import os

import numpy as np

from hrrl.plotting import AQUA, BLUE, INK_2, ORANGE, YELLOW, mean_sem, plt, read_csv, setup

PRE, POST = 120, 260  # window around scarcity onset (steps)


def aligned_traces(files):
    """Internal state (mean over variables) aligned on every scarcity onset."""
    segs = []
    for f in files:
        with np.load(f) as d:
            h, sc = d["h"].mean(1), d["scarcity"]
        for o in np.flatnonzero(sc[1:] & ~sc[:-1]) + 1:
            if o >= PRE and o + POST <= len(h):
                segs.append(h[o - PRE:o + POST])
    return np.array(segs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/scarcity")
    a = p.parse_args()
    setup()
    rows = read_csv(os.path.join(a.out, "summary.csv"))
    names = list(dict.fromkeys(r["condition"] for r in rows))
    colors = dict(zip(names, [YELLOW, ORANGE, BLUE, AQUA]))
    setpoint, warning = 10.0, 60

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)

    ax = axes[0]
    x = np.arange(-PRE, POST)
    for name in names:
        segs = aligned_traces(glob.glob(os.path.join(a.out, f"{name}_seed*.npz")))
        if len(segs):
            ax.plot(x, segs.mean(0), color=colors[name], lw=1.5, label=f"{name}  (n={len(segs)})")
    ax.axvspan(-warning, 0, color=INK_2, alpha=0.08, lw=0)
    ax.axvspan(0, 250, color=INK_2, alpha=0.16, lw=0)
    ax.axhline(setpoint, color=INK_2, ls="--", lw=1)
    ax.text(-warning / 2, ax.get_ylim()[1], "cue", ha="center", va="top", color=INK_2, fontsize=8)
    ax.text(125, ax.get_ylim()[1], "scarcity", ha="center", va="top", color=INK_2, fontsize=8)
    ax.set(xlabel="Steps from scarcity onset", ylabel="Internal state (mean of variables)",
           title="a  State around scarcity")
    ax.legend(fontsize=7, loc="lower left")

    for ax, key, title in ((axes[1], "pre_scarcity_level", "b  Storage before onset"),
                           (axes[2], "survival", "c  Scarcity survival")):
        for i, name in enumerate(names):
            vals = [r[key] for r in rows if r["condition"] == name]
            mu, se = mean_sem(vals)
            ax.bar(i, mu, yerr=1.96 * se, color=colors[name], width=0.6, capsize=3)
            if key == "pre_scarcity_level":
                base, _ = mean_sem([r["baseline_level"] for r in rows if r["condition"] == name])
                ax.plot([i - 0.3, i + 0.3], [base, base], color=INK_2, lw=1.5)
        ax.set_xticks(range(len(names)), names, rotation=20, fontsize=8)
        ax.set_title(title)
    axes[1].set_ylabel("h − setpoint (bar)  vs. baseline (line)")
    axes[1].axhline(0, color=INK_2, lw=0.8)
    axes[2].set_ylabel("Fraction of scarcities survived")
    axes[2].set_ylim(0, 1.05)

    path = os.path.join(a.out, "fig_scarcity.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    main()
