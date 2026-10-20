# -*- coding: utf-8 -*-
"""
Figures for the Signal Processing draft.

  fig1_budget_effects.pdf   effect size vs training budget, four contrasts
  fig2_cascade.pdf          dead fraction per ReLU layer over the first 20 epochs
  fig3_escape_repair.pdf    terminal state, escape by LR, escape by re-init

Every number is transcribed from the analysis outputs; nothing is recomputed here.
Source: tables/*.csv, results/escape_repair_sweep.json,
        results/control_layers_and_constant_predictor.json.
"""
import io, os, sys
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "signal_processing_paper", "elsarticle", "figures")
os.makedirs(OUT, exist_ok=True)

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

# ------------------------------------------------------------------ fig 1
aug = [12.310, 9.657, 5.446, 5.124, 3.630, 2.558, 1.983, 1.376, 1.348]
aug4 = [12.967, 7.421, 2.323, 1.952, 1.410, 0.840, 0.903, 1.067, 0.990]
arch_full = [-0.851, 2.162, 3.934, 5.327, 5.483, 5.533, 5.941, 6.479, 6.661]
arch_sub = [-3.949, -1.365, 0.291, 1.616, 1.648, 1.665, 2.046, 2.515, 2.734]

aug_sig = [True] * 9
aug4_sig = [True, True, False, False, False, False, False, False, True]
archf_sig = [False, False, False, True, True, True, True, True, True]
archs_sig = [True, False, False, False, False, True, True, True, True]


def series(ax, x, y, sig, color, marker, label, ls="-"):
    ax.plot(x, y, ls, color=color, marker=marker, markerfacecolor="white",
            markeredgecolor=color, markeredgewidth=1.1, label=label, zorder=3)
    xs = [xi for xi, s in zip(x, sig) if s]
    ys = [yi for yi, s in zip(y, sig) if s]
    ax.plot(xs, ys, "o", color=color, markersize=4.5, zorder=4)


fig, ax = plt.subplots(figsize=(3.4, 2.6))
series(ax, EP, aug, aug_sig, "#1f4e79", "s", "Augmentation, GAP")
series(ax, EP, aug4, aug4_sig, "#7ba7d0", "s", "Augmentation, GAP ($\\times4$ discrete)")
series(ax, EP, arch_full, archf_sig, "#a83232", "^", "GAP $-$ FLAT, complete")
series(ax, EP, arch_sub, archs_sig, "#e09a9a", "^", "GAP $-$ FLAT, subsample")

ax.axhline(0.0, color="0.55", lw=0.7, zorder=1)
ax.axvspan(20, 80, color="0.90", zorder=0)
ax.text(38, 11.4, "instability\nwindow", fontsize=6.5, color="0.35", ha="center")

ax.set_xscale("log")
ax.set_xticks([20, 40, 80, 160, 320, 640])
ax.set_xticklabels(["20", "40", "80", "160", "320", "640"])
ax.set_xlabel("Training budget (epochs)")
ax.set_ylabel("Accuracy difference (pp)")
ax.set_ylim(-6.0, 14.0)
ax.legend(loc="upper right", handlelength=1.6)
ax.text(0.015, 0.03, "filled marker: $p<0.05$   open: n.s.",
        transform=ax.transAxes, fontsize=6.2, color="0.35")
fig.savefig(os.path.join(OUT, "fig1_budget_effects.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ fig 2
ep_c = [0, 1, 2, 3, 4, 5, 7, 10, 15, 20]
c1 = [0.6249993, 0.6249994, 0.6249983, 0.6249925, 0.6249978, 0.6249915,
      0.5793839, 0.5793840, 0.5793840, 0.5793840]
c2 = [0.4077158, 0.9995113, 0.9995113, 0.9999997, 0.9999998, 0.9999998,
      0.9999942, 0.9999942, 0.9999942, 0.9999942]
fc = [0.515625, 0.734375, 0.828125, 0.9375, 0.9375, 0.96875,
      1.0, 1.0, 1.0, 1.0]
acc = [9.268, 9.264, 8.961, 8.961, 9.052, 8.961, 9.020, 8.964, 9.052, 9.143]

fig, ax = plt.subplots(figsize=(3.4, 2.6))
ax.plot(ep_c, c1, "-o", color="#1f4e79", label="conv1 ReLU", ms=3.5)
ax.plot(ep_c, c2, "-s", color="#a83232", label="conv2 ReLU", ms=3.5)
ax.plot(ep_c, fc, "-^", color="#7a5c1e", label="fc1 ReLU", ms=3.5)
ax.axhline(1.0, color="0.7", lw=0.7, ls=":", zorder=1)
ax.set_xlabel("Epoch")
ax.set_ylabel("Dead fraction")
ax.set_ylim(0.30, 1.06)
ax.set_xlim(-0.6, 20.6)

ax2 = ax.twinx()
ax2.plot(ep_c, acc, "--", color="0.45", lw=1.1, zorder=2)
ax2.set_ylabel("Test accuracy (\\%)", color="0.45")
ax2.set_ylim(0, 100)
ax2.tick_params(axis="y", colors="0.45")
ax2.axhline(9.09, color="0.75", lw=0.6, ls="-.", zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax2.spines["top"].set_visible(False)
ax2.spines["left"].set_visible(False)

ax.annotate("conv2 collapses\nwithin epoch 1",
            xy=(1, 0.9995), xytext=(3.4, 0.80),
            fontsize=6.5, color="#a83232",
            arrowprops=dict(arrowstyle="->", color="#a83232", lw=0.8))
ax.text(0.98, 0.06, "chance (9.09\\%)", transform=ax.transAxes,
        fontsize=6.2, color="0.5", ha="right")
ax.legend(loc="center left", bbox_to_anchor=(0.02, 0.44), handlelength=1.6)
fig.savefig(os.path.join(OUT, "fig2_cascade.pdf"))
plt.close(fig)

# ------------------------------------------------------------------ fig 3
labels = ["Collapsed\n(ep 640)", "+1000$\\times$LR\n(10 ep)",
          "Re-init fc1\n(40 ep)", "Re-init fc1+fc2\n(40 ep)",
          "Healthy M2\n(raw)"]
vals = [9.02, 9.14, 40.79, 35.05, 51.03]
dead = [100.0, 100.0, 93.1, 89.0, 83.1]
colors = ["#a83232", "#a83232", "#2e7d32", "#2e7d32", "#1f4e79"]

fig, ax = plt.subplots(figsize=(3.4, 2.5))
x = range(len(vals))
ax.bar(x, vals, color=colors, width=0.62, zorder=3)
for xi, (v, d) in enumerate(zip(vals, dead)):
    ax.text(xi, v + 1.4, "%.2f" % v, ha="center", fontsize=7.5, zorder=4)
    ax.text(xi, 1.6, "dead %.0f\\%%" % d, ha="center", fontsize=6.2,
            color="white", zorder=4)
ax.axhline(9.09, color="0.5", lw=0.8, ls="--", zorder=2)
ax.text(4.42, 10.4, "chance", fontsize=6.2, color="0.45", ha="right")
ax.set_xticks(list(x))
ax.set_xticklabels(labels, fontsize=6.6)
ax.set_ylabel("Test accuracy (\\%)")
ax.set_ylim(0, 62)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.savefig(os.path.join(OUT, "fig3_escape_repair.pdf"))
plt.close(fig)

print("wrote 3 figures to", OUT)
for f in sorted(os.listdir(OUT)):
    print("   ", f)
