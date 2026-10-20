# -*- coding: utf-8 -*-
"""Replacement for huang2.pdf (IEEE Access submission).

The manuscript reports the re-initialisation repair as a coin flip over
perturbation draws on two collapsed runs.  The figure has to carry three
things at once in a single 3.4 in column:

  * every attempt lands in one of exactly two modes;
  * the band between the modes is empty (24.0 pp on the re-split, 30.5 pp on
    the official partition);
  * the initial dead-unit count does NOT predict which mode.

The earlier version of this figure ordered the draws along x by their epoch-0
dead count.  With 20 draws that ordering was tested and found not to predict
the outcome (Spearman rho = +0.22, p = 0.35; Mann-Whitney AUC = 0.64, p = 0.38),
so putting it on the x-axis would invite the reader to read a trend that is not
there.  x is therefore the outcome itself, accuracy, and the draws are stacked
within each partition so that all 40 attempts are countable.

Data source: analysis_results/rc_repair_perturbation.json (single source of
truth).  Nothing is recomputed here; every number is read back.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "analysis_results", "rc_repair_perturbation.json")
OUT = r"C:\Users\Administrator\ieee_access_submission\huang2.pdf"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9,
    "axes.linewidth": 1.0,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 300,
})

CHANCE = 100.0 / 11.0          # the constant predictor on the official split
ROW = 1.30                     # vertical distance between the two strip rows
STEP = 0.105                   # vertical step between stacked markers


def dodged(values, tol=1.0, step=STEP):
    """Stack values closer than `tol` so coincident points stay countable."""
    v = np.asarray(values, dtype=float)
    order = np.argsort(v, kind="stable")
    offs = np.zeros(len(v))
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] - v[order[i]] <= tol:
            j += 1
        grp = order[i:j + 1]
        n = len(grp)
        for k, idx in enumerate(grp):
            offs[idx] = (k - (n - 1) / 2.0) * step
        i = j + 1
    return offs


with open(SRC, encoding="utf-8") as f:
    R = json.load(f)

SEEDS = [int(k[1:].split("_")[0]) for k in R["resplit"]["cells"]
         if k.endswith("_reinit_fc1")]


def accs(run):
    return [R[run]["cells"][f"p{s}_reinit_fc1"]["epoch40"]["acc"] for s in SEEDS]


resplit = accs("resplit")
official = accs("official")

fig, ax = plt.subplots(figsize=(3.4, 2.15))

series = [
    ("Re-split", 0.0, resplit, "s", "white", "#a83232"),
    ("Official split", ROW, official, "o", "#1f4e79", "#1f4e79"),
]
for label, y0, vals, mk, fc, ec in series:
    y = y0 + dodged(vals)
    ax.plot(vals, y, mk, color=ec, mfc=fc, mec=ec, mew=1.1, markersize=4.2,
            label=label, zorder=4, linestyle="none")

ax.axvline(CHANCE, color="0.45", lw=0.9, ls="--", zorder=2)
ax.text(CHANCE, ROW + 1.02, "constant\npredictor", ha="center", va="top",
        fontsize=6.6, color="0.35", linespacing=1.1)

ax.set_yticks([0.0, ROW])
ax.set_yticklabels(["Re-split", "Official\nsplit"], fontsize=7.6)
ax.set_ylim(-0.72, ROW + 1.18)
ax.set_xlim(0, 50)
ax.set_xlabel("Test accuracy after the repair attempt (%)")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(axis="y", length=0)

# No legend: the row labels already name the partitions, the two markers differ
# in shape, and the only other graphic element (the dashed line) is annotated.

fig.savefig(OUT)
plt.close(fig)
print("wrote", OUT)
for run, v in (("resplit", resplit), ("official", official)):
    v = sorted(v)
    fail = [x for x in v if x < 20]
    esc = [x for x in v if x >= 20]
    print(f"  {run:9s} {len(esc)}/{len(v)} escaped   "
          f"failures {min(fail):.2f}-{max(fail):.2f}   "
          f"escapes {min(esc):.2f}-{max(esc):.2f}   "
          f"gap {min(esc) - max(fail):.2f} pp")
