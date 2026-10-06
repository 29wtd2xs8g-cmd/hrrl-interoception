"""Shared figure style: thin marks, recessive axes, fixed categorical colours."""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Categorical slots in fixed order (validated colour-blind-safe order).
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
COND_COLORS = {"hrrl": BLUE, "maximizer": ORANGE, "saturating": AQUA}
COND_LABELS = {"hrrl": "A  HRRL", "maximizer": "B  maximizer", "saturating": "C  saturating"}


def setup():
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 200, "font.size": 9,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "xtick.color": INK_2, "ytick.color": INK_2, "lines.linewidth": 2,
        "legend.frameon": False,
    })


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
    return rows


def mean_sem(values):
    v = np.asarray([x for x in values if not np.isnan(x)], float)
    if len(v) == 0:
        return np.nan, np.nan
    return v.mean(), (v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0)
