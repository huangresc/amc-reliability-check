# -*- coding: utf-8 -*-
"""Regenerate huang1.pdf (IEEE Access submission, Fig. 1).

Why this file exists
--------------------
`huang1.pdf` in the submission was byte-identical to
`signal_processing_paper/elsarticle/figures/fig1_budget_effects.pdf`, produced by
the fig-1 block of `make_paper_figures.py` on 2026-10-04 (verified by SHA1:
6F59E6BF89108D9BB3A74A965DB64E156FDDC7A6 for both). This script reproduces that
block byte-for-byte except for one change.

The change: the two augmentation series were labelled "Augmentation, GAP" and
"Augmentation, GAP ($\\times4$)", which reads as "with augmentation" when the
plotted quantity is the opposite -- M2_GAPHead (no augmentation) minus
M5_GAPHead_PhaseAug (continuous phase rotation), i.e. a positive value is an
accuracy COST of augmentation. Check against tables/four_variants.csv:
    ep20  37.15977 - 24.85023 = +12.30954
    ep640 51.75909 - 50.41068 =  +1.34841
Reviewer concern 7 asks whether the legend means "with augmentation"; it does
not, so the labels now say what the ordinate is.

Every number is transcribed from tables/four_variants.csv and
tables/phase_augmentation.csv; nothing is recomputed here.

2026-10-13, reviewer concern 4.  The filled/open rule was "raw $p<0.05$",
which treats the nine checkpoints as nine independent tests.  It is now the
conjunction of the two corrections in Section IV-D: Holm-adjusted $p<0.05$
within the nine-checkpoint family AND a BCa cluster-bootstrap 95 % interval
over seeds excluding zero.  Both were computed by rc_inference.py and are in
analysis_results/rc_inference.json; the surviving sets are
    augmentation, continuous : all nine
    augmentation, x4         : epoch 20 only
    GAP minus FLAT, complete : 320, 480, 640
    GAP minus FLAT, subsample: 640 only
The raw-$p$ sets are supersets of these, so no marker was added, only removed.
"""
import io
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = r"C:\Users\Administrator\ieee_access_submission\huang1.pdf"

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
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.4,
    "lines.markersize": 4.5,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
    "figure.dpi": 300,
})

EP = [20, 40, 80, 120, 160, 240, 320, 480, 640]

aug = [12.310, 9.657, 5.446, 5.124, 3.630, 2.558, 1.983, 1.376, 1.348]
aug4 = [12.967, 7.421, 2.323, 1.952, 1.410, 0.840, 0.903, 1.067, 0.990]
arch_full = [-0.851, 2.162, 3.934, 5.327, 5.483, 5.533, 5.941, 6.479, 6.661]
arch_sub = [-3.949, -1.365, 0.291, 1.616, 1.648, 1.665, 2.046, 2.515, 2.734]

aug_sig = [True] * 9
aug4_sig = [True, False, False, False, False, False, False, False, False]
archf_sig = [False, False, False, False, False, False, True, True, True]
archs_sig = [False, False, False, False, False, False, False, False, True]


def series(ax, x, y, sig, color, marker, label, ls="-"):
    ax.plot(x, y, ls, color=color, marker=marker, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.1, label=label, zorder=3)
    xs = [xi for xi, s in zip(x, sig) if s]
    ys = [yi for yi, s in zip(y, sig) if s]
    ax.plot(xs, ys, "o", color=color, markersize=4.5, zorder=4)


fig, ax = plt.subplots(figsize=(3.4, 3.05))
series(ax, EP, aug, aug_sig, "#1f4e79", "s", "Augmentation cost, GAP")
series(ax, EP, aug4, aug4_sig, "#7ba7d0", "s", "Augmentation cost, GAP ($\\times4$)")
series(ax, EP, arch_full, archf_sig, "#a83232", "^", "GAP $-$ FLAT (complete)")
series(ax, EP, arch_sub, archs_sig, "#e09a9a", "^", "GAP $-$ FLAT (subsample)")

ax.axhline(0.0, color="0.55", lw=0.7, zorder=1)
ax.axvspan(20, 80, color="0.90", zorder=0)
ax.text(40, 13.4, "instability window", fontsize=6.4, color="0.35", ha="center")

ax.set_xscale("log")
ax.set_xticks([20, 40, 80, 160, 320, 640])
ax.set_xticklabels(["20", "40", "80", "160", "320", "640"])
ax.set_xlabel("Training budget (epochs)")
ax.set_ylabel("Accuracy difference (pp)")
ax.set_ylim(-7.0, 14.8)
ax.set_xlim(18.0, 720.0)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.245), ncol=2,
          fontsize=6.8, handlelength=1.5, columnspacing=1.0,
          handletextpad=0.5, borderaxespad=0.0)
ax.text(0.985, 0.02, "filled: Holm $p<0.05$ and BCa 95 % CI excludes 0",
        transform=ax.transAxes, fontsize=5.9, color="0.35",
        ha="right", va="bottom")
fig.savefig(OUT)
plt.close(fig)

print("wrote", OUT)
print("size", os.path.getsize(OUT), "bytes")
