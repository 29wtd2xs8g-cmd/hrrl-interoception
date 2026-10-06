"""Satiation protocol (point 4): conditions A-D x seeds, inexhaustible resources.

    A  HRRL, exact interoception            (hrrl, sigma=0)
    B  Same agent, reward = amount consumed (maximizer)
    C  Same agent, hand-coded saturating reward (saturating)
    D  HRRL with increasing interoceptive noise sigma (obs + reward)

Each run is saved to <out>/<condition>_seed<k>.npz; existing files are skipped, so an
interrupted sweep can be resumed by re-running the same command.

    python run_satiation.py --seeds 30 --workers 8
    python run_satiation.py --seeds 2 --train-steps 30000 --out results/smoke   # smoke test
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from hrrl.config import RunConfig
from hrrl.experiment import run_one, with_overrides
from hrrl.metrics import satiation_summary

SIGMAS = (0.5, 1.0, 2.0, 4.0)


def conditions(sigmas):
    conds = {
        "A_hrrl": {"condition": "hrrl", "sigma": 0.0},
        "B_maximizer": {"condition": "maximizer", "sigma": 0.0},
        "C_saturating": {"condition": "saturating", "sigma": 0.0},
    }
    for s in sigmas:
        conds[f"D_hrrl_sigma{s:g}"] = {"condition": "hrrl", "sigma": s}
    return conds


def _job(args):
    name, reward, seed, train_steps, out = args
    path = os.path.join(out, f"{name}_seed{seed}.npz")
    if os.path.exists(path):
        return name, seed, 0.0
    cfg = with_overrides(RunConfig(), run={"seed": seed, "train_steps": train_steps}, reward=reward)
    t = time.time()
    run_one(cfg, path)
    return name, seed, time.time() - t


def summarize(out, conds, seeds, setpoint):
    rows = []
    for name, reward in conds.items():
        for seed in range(seeds):
            path = os.path.join(out, f"{name}_seed{seed}.npz")
            if not os.path.exists(path):
                continue
            with np.load(path) as d:
                s = satiation_summary(d, setpoint)
            rows.append({"condition": name, "reward": reward["condition"],
                         "sigma": reward["sigma"], "seed": seed, **s})
    with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\n{'condition':<22}{'near':>7}{'|dev|':>7}{'oversh':>8}{'intake':>8}{'latency':>9}{'deaths':>8}")
    for name in conds:
        r = [x for x in rows if x["condition"] == name]
        m = lambda k: np.nanmean([x[k] for x in r])
        print(f"{name:<22}{m('near_setpoint'):7.2f}{m('abs_dev'):7.2f}{m('overshoot'):8.2f}{m('intake'):8.3f}"
              f"{m('resume_latency'):9.1f}{m('deaths'):8.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=30)
    p.add_argument("--train-steps", type=int, default=RunConfig.train_steps)
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    p.add_argument("--sigmas", type=float, nargs="*", default=SIGMAS)
    p.add_argument("--only", nargs="*", help="subset of condition names")
    p.add_argument("--out", default="results/satiation")
    a = p.parse_args()

    conds = conditions(a.sigmas)
    if a.only:
        conds = {k: v for k, v in conds.items() if k in a.only}
    os.makedirs(a.out, exist_ok=True)
    jobs = [(n, r, s, a.train_steps, a.out) for s in range(a.seeds) for n, r in conds.items()]
    print(f"{len(jobs)} runs on {a.workers} workers -> {a.out}")
    with Pool(a.workers) as pool:
        for i, (name, seed, dt) in enumerate(pool.imap_unordered(_job, jobs), 1):
            print(f"[{i}/{len(jobs)}] {name} seed {seed} ({dt:.0f}s)", flush=True)
    summarize(a.out, conds, a.seeds, RunConfig().env.setpoint)


if __name__ == "__main__":
    main()
