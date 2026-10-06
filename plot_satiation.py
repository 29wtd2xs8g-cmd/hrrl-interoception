"""Central figure: how interoceptive noise degrades satiation (A-D crossed).

    python plot_satiation.py --out results/satiation
"""
import argparse
import glob
import os

import numpy as np

from hrrl.plotting import BLUE, COND_COLORS, COND_LABELS, INK_2, mean_sem, plt, read_csv, setup

METRICS = [
    ("near_setpoint", "Time near setpoint (fraction)"),
    ("overshoot", "Overshoot above setpoint"),
    ("resume_latency", "Resumption latency (steps)"),
    ("deaths", "Deaths per 5000-step life"),
]


def by_condition(rows, key):
    return {name: [r[key] for r in rows if r["condition"] == name]
            for name in dict.fromkeys(r["condition"] for r in rows)}


def panel_metric(ax, rows, key, label):
    hrrl = sorted({r["sigma"] for r in rows if r["reward"] == "hrrl"})
    stats = [mean_sem([r[key] for r in rows if r["reward"] == "hrrl" and r["sigma"] == s]) for s in hrrl]
    m, e = np.array(stats).T
    ax.errorbar(hrrl, m, yerr=1.96 * e, color=BLUE, marker="o", ms=5, capsize=3, label="HRRL (A, D)")
    # B and C have no interoceptive noise: drawn as reference levels across the sigma axis.
    for cond, ls in (("maximizer", "--"), ("saturating", ":")):
        vals = [r[key] for r in rows if r["reward"] == cond]
        if vals:
            mu, se = mean_sem(vals)
            ax.axhline(mu, color=COND_COLORS[cond], ls=ls, lw=1.5, label=COND_LABELS[cond])
            ax.axhspan(mu - 1.96 * se, mu + 1.96 * se, color=COND_COLORS[cond], alpha=0.12, lw=0)
    ax.set_xlabel("Interoceptive noise σ")
    ax.set_title(label)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="results/satiation")
    p.add_argument("--seed", type=int, default=0, help="seed used for example trajectories")
    a = p.parse_args()
    setup()
    rows = read_csv(os.path.join(a.out, "summary.csv"))
    setpoint = 10.0

    fig, axes = plt.subplots(2, 3, figsize=(12, 6.8), constrained_layout=True)

    # (a) example trajectories: energy variable during the evaluation life
    ax = axes[0, 0]
    for cond, name in (("hrrl", "A_hrrl"), ("maximizer", "B_maximizer"), ("saturating", "C_saturating")):
        path = os.path.join(a.out, f"{name}_seed{a.seed}.npz")
        if os.path.exists(path):
            with np.load(path) as d:
                ax.plot(d["h"][:600, 0], color=COND_COLORS[cond], lw=1.5, label=COND_LABELS[cond])
    ax.axhline(setpoint, color=INK_2, ls="--", lw=1)
    ax.text(598, setpoint + 0.3, "setpoint", ha="right", va="bottom", color=INK_2, fontsize=8)
    ax.set(xlabel="Evaluation step", ylabel="Internal energy h₀", title="a  Example trajectories")

    # (b) intake over training, mean across seeds
    ax = axes[0, 1]
    for cond, name in (("hrrl", "A_hrrl"), ("maximizer", "B_maximizer"), ("saturating", "C_saturating")):
        files = glob.glob(os.path.join(a.out, f"{name}_seed*.npz"))
        if not files:
            continue
        curves = []
        for f in files:
            with np.load(f) as d:
                curves.append(d["train_intake"].sum(1))
        curves = np.array(curves)
        x = np.arange(curves.shape[1]) + 1
        mu, se = curves.mean(0), curves.std(0, ddof=1 if len(curves) > 1 else 0) / np.sqrt(len(curves))
        ax.plot(x, mu, color=COND_COLORS[cond], lw=1.5, label=COND_LABELS[cond])
        ax.fill_between(x, mu - 1.96 * se, mu + 1.96 * se, color=COND_COLORS[cond], alpha=0.15, lw=0)
    ax.set(xlabel="Training step (×1000)", ylabel="Bites per step", title="b  Consumption rate over training")
    ax.set_yscale("log")

    # (c-f) satiation metrics vs interoceptive noise
    for ax, (key, label), letter in zip([axes[0, 2], *axes[1]], METRICS, "cdef"):
        panel_metric(ax, rows, key, f"{letter}  {label}")
    # one shared legend: the condition colours are the same in every panel
    handles, labels = axes[0, 2].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.06))

    n_seeds = len({r["seed"] for r in rows})
    fig.suptitle(f"Interoceptive noise and satiation  (mean ± 95% CI, {n_seeds} seeds)",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    path = os.path.join(a.out, "fig_satiation.png")
    fig.savefig(path, bbox_inches="tight")
    print("saved", path)


if __name__ == "__main__":
    main()
