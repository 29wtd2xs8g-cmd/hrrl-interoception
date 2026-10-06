"""Temporal variant: announced future scarcity, varying the discount factor gamma.

Homeostatic objectives are bounded at each instant but not over time: if the agent
anticipates a lethal shortage, overshooting the setpoint now (storage) can reduce the
total drive later. Does storage emerge, and does it depend on gamma?

Conditions: HRRL (sigma=0) with gamma in GAMMAS, plus a no-cue control at the highest
gamma (scarcity still happens, but the warning is hidden).

    python run_scarcity.py --seeds 30 --workers 8
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from hrrl.config import RunConfig
from hrrl.experiment import run_one, with_overrides
from hrrl.metrics import storage_summary

GAMMAS = (0.5, 0.9, 0.99)


def conditions(gammas, sigma):
    conds = {f"gamma{g:g}": {"gamma": g, "cue": True} for g in gammas}
    conds[f"gamma{max(gammas):g}_nocue"] = {"gamma": max(gammas), "cue": False}
    for c in conds.values():
        c["sigma"] = sigma
    return conds


def _job(args):
    name, c, seed, train_steps, out = args
    path = os.path.join(out, f"{name}_seed{seed}.npz")
    if os.path.exists(path):
        return name, seed, 0.0
    cfg = with_overrides(
        RunConfig(),
        run={"seed": seed, "train_steps": train_steps},
        env={"scarcity": True, "scarcity_cue": c["cue"], "max_steps": 2000},
        reward={"condition": "hrrl", "sigma": c["sigma"]},
        agent={"gamma": c["gamma"]},
    )
    t = time.time()
    run_one(cfg, path)
    return name, seed, time.time() - t


def summarize(out, conds, seeds, setpoint):
    rows = []
    for name, c in conds.items():
        for seed in range(seeds):
            path = os.path.join(out, f"{name}_seed{seed}.npz")
            if not os.path.exists(path):
                continue
            with np.load(path) as d:
                s = storage_summary(d, setpoint)
            rows.append({"condition": name, "gamma": c["gamma"], "cue": c["cue"], "seed": seed, **s})
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'condition':<20}{'pre-scarcity':>13}{'baseline':>10}{'survival':>10}{'deaths':>8}")
    for name in conds:
        r = [x for x in rows if x["condition"] == name]
        m = lambda k: np.nanmean([x[k] for x in r])
        print(f"{name:<20}{m('pre_scarcity_level'):13.2f}{m('baseline_level'):10.2f}"
              f"{m('survival'):10.2f}{m('deaths'):8.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--train-steps", type=int, default=RunConfig.train_steps)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--gammas", type=float, nargs="*", default=GAMMAS)
    p.add_argument("--sigma", type=float, default=0.0)
    p.add_argument("--out", default="results/scarcity")
    a = p.parse_args()

    conds = conditions(a.gammas, a.sigma)
    os.makedirs(a.out, exist_ok=True)
    jobs = [(n, c, s, a.train_steps, a.out) for s in range(a.seeds) for n, c in conds.items()]
    print(f"{len(jobs)} runs on {a.workers} workers -> {a.out}")
    with Pool(a.workers) as pool:
        for i, (name, seed, dt) in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name} seed {seed} ({dt:.0f}s)", flush=True)
    summarize(a.out, conds, a.seeds, RunConfig().env.setpoint)


if __name__ == "__main__":
    main()
