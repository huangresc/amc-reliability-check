# -*- coding: utf-8 -*-
"""Three checks on rc_repair_perturbation.json before anything is written down.

(1) Integrality.  The epoch-0 dead fractions are multiples of 1/64, the fc1
    width.  If the dead COUNT is an exact multiple of n_test then each fc1 unit
    is dead for every test vector or for none, i.e. the sign of the
    re-initialised pre-activation is input-independent.  (The counts are only
    near-integral: a unit can be off by a handful of vectors, because the layer
    input is the near-constant output of the collapsed convolutional trunk.)

(2) Bimodality.  Are there really only two outcomes, or does the sweep fill in
    the gap between them?

(3) Does the epoch-0 dead count predict the escape?  This is a LEAD, and the
    test has to be threshold-free.  A Fisher table needs a cut point, and with
    five draws the cut was chosen after seeing the data -- the p = 0.0476 that
    the earlier version of this file reported is a post-hoc minimum over cuts,
    not a p-value.  So the primary test here is rank-based and needs no cut:

      * each draw has one dead count (it is bit-identical in both partitions)
        and one mean final accuracy over the two partitions, n = #draws;
        Spearman rho between them;
      * Mann-Whitney U on dead count, escaped draws vs re-collapsed draws,
        reported as an AUC = P(dead of an escaped draw > dead of a failed one),
        which is the common-language effect size.

    The Fisher table is still printed, but as a table view at the median split,
    labelled as post hoc.
"""
import json

import numpy as np
import scipy.stats as stats

with open("analysis_results/rc_repair_perturbation.json", encoding="utf-8") as f:
    R = json.load(f)

SEEDS = [int(k[1:].split("_")[0]) for k in R["resplit"]["cells"]
         if k.endswith("_reinit_fc1")]
THRESH = 20.0
NTEST = {"resplit": 44000, "official": 110000}


def ep0(run, p):
    return R[run]["cells"][f"p{p}_reinit_fc1"]["epoch0"]


def ep40(run, p):
    return R[run]["cells"][f"p{p}_reinit_fc1"]["epoch40"]


print("=" * 78)
print(f"(1) Is each fc1 unit dead for all test vectors or none?  ({len(SEEDS)} draws)")
print("=" * 78)
worst = 0.0
for run in ("resplit", "official"):
    n = NTEST[run]
    offs = []
    for p in SEEDS:
        units = ep0(run, p)["dead_fc1"] * 64
        offs.append(abs(units - round(units)))
    worst = max(worst, max(offs))
    print(f"  {run:9s} n_test {n:>6d}: units-from-integer "
          f"min {min(offs):.4f}  median {np.median(offs):.4f}  max {max(offs):.4f}")
    print(f"  {'':9s} (max {max(offs):.4f} units = {max(offs) * n:.0f} test vectors "
          f"out of {64 * n:,})")
print(f"\n  => dead counts are NOT exactly integral; a unit can be dead for up to")
print(f"     a handful of vectors more or fewer than the rest.  Report the dead")
print(f"     count as 'to within a handful of test vectors', not as exact.")

print("\n" + "=" * 78)
print("(2) Bimodality: does anything land between the two modes?")
print("=" * 78)
allacc = []
for run in ("resplit", "official"):
    a = sorted(ep40(run, p)["acc"] for p in SEEDS)
    allacc += a
    below = [x for x in a if x <= THRESH]
    above = [x for x in a if x > THRESH]
    gap = (min(above) - max(below)) if below and above else float("nan")
    print(f"  {run:9s} re-collapsed max {max(below) if below else float('nan'):6.2f}  "
          f"| escaped min {min(above) if above else float('nan'):6.2f}  "
          f"gap {gap:5.2f} pp   ({len(below)}/{len(a)} re-collapsed)")
mid = [x for x in allacc if THRESH / 2 < x < 30.0]
print(f"  cells anywhere in the (10 %, 30 %) band: {len(mid)}  {['%.2f' % x for x in mid]}")

print("\n" + "=" * 78)
print("(3) Does the epoch-0 dead count predict the escape?  (lead, threshold-free)")
print("=" * 78)
# One dead count per draw: identical in both partitions (checked below), so take
# the re-split value.  One outcome per draw: mean final accuracy over the two
# partitions.  n = number of DRAWS, not cells -- the 2 cells of a draw share the
# dead count and are not independent.
dead, meanacc, per = [], [], []
for p in SEEDS:
    d = ep0("resplit", p)["dead_fc1"] * 64
    a_r, a_o = ep40("resplit", p)["acc"], ep40("official", p)["acc"]
    dead.append(d)
    meanacc.append(0.5 * (a_r + a_o))
    per.append((p, d, a_r, a_o))

# consistency of the dead count across partitions
sh = [abs(ep0("official", p)["dead_fc1"] - ep0("resplit", p)["dead_fc1"]) * 64
      for p in SEEDS]
print(f"  dead-count agreement across partitions: max drift {max(sh):.4f} units "
      f"(fc1 sha1 identical by construction)")

rho, prho = stats.spearmanr(dead, meanacc)
print(f"\n  Spearman(ep0 dead units, mean final acc)  n={len(SEEDS)} draws")
print(f"    rho = {rho:+.3f}   p = {prho:.4f}")

esc_d = [d for d, m in zip(dead, meanacc) if m > THRESH]
fail_d = [d for d, m in zip(dead, meanacc) if m <= THRESH]
print(f"\n  draws: {len(esc_d)} escaped (mean acc > {THRESH:.0f} %), "
      f"{len(fail_d)} re-collapsed")
if esc_d and fail_d:
    u, pmw = stats.mannwhitneyu(esc_d, fail_d, alternative="two-sided")
    auc = u / (len(esc_d) * len(fail_d))
    print(f"    dead units | escaped  min {min(esc_d):5.2f}  median "
          f"{np.median(esc_d):5.2f}  max {max(esc_d):5.2f}")
    print(f"    dead units | failed   min {min(fail_d):5.2f}  median "
          f"{np.median(fail_d):5.2f}  max {max(fail_d):5.2f}")
    print(f"    Mann-Whitney U two-sided p = {pmw:.4f}   "
          f"AUC = P(dead[escaped] > dead[failed]) = {auc:.3f}")

# the same test at cell level, for reference (do not quote: cells within a draw
# are not independent)
cell_d, cell_esc = [], []
for run in ("resplit", "official"):
    for p in SEEDS:
        cell_d.append(ep0(run, p)["dead_fc1"] * 64)
        cell_esc.append(ep40(run, p)["acc"] > THRESH)
rho_c, prho_c = stats.spearmanr(cell_d, cell_esc)
print(f"\n  (cell level, n={len(cell_d)} -- cells share draws, NOT independent)")
print(f"    Spearman(dead, escaped) rho = {rho_c:+.3f}  p = {prho_c:.4f}")

print("\n" + "=" * 78)
print("Table view: Fisher at the median split (POST HOC cut -- a table, not a test)")
print("=" * 78)
cut = float(np.median(dead))
hi_e = sum(1 for d, m in zip(dead, meanacc) if d > cut and m > THRESH)
hi_f = sum(1 for d, m in zip(dead, meanacc) if d > cut and m <= THRESH)
lo_e = sum(1 for d, m in zip(dead, meanacc) if d <= cut and m > THRESH)
lo_f = sum(1 for d, m in zip(dead, meanacc) if d <= cut and m <= THRESH)
odds, pval = stats.fisher_exact([[hi_e, hi_f], [lo_e, lo_f]])
print(f"  cut at the median dead count = {cut:.2f} units (of 64)")
print(f"    dead >  cut : {hi_e} escaped, {hi_f} failed")
print(f"    dead <= cut : {lo_e} escaped, {lo_f} failed")
print(f"    Fisher two-sided p = {pval:.4f}  (odds ratio {odds:g})")
print(f"  Report the Spearman/Mann-Whitney above as the result; quote this p only")
print(f"  with the cut named and flagged as chosen after seeing the outcome.")

print("\n" + "=" * 78)
print("Per-draw table")
print("=" * 78)
print(f"  {'seed':>7} {'ep0 dead/64':>12} {'re-split':>9} {'official':>9} "
      f"{'mean':>8}  verdict")
for p, d, a_r, a_o in sorted(per, key=lambda r: r[1]):
    m = 0.5 * (a_r + a_o)
    v = "escaped" if m > THRESH else ("re-collapsed" if a_r <= THRESH and a_o <= THRESH
                                      else "SPLIT")
    print(f"  {p:>7} {d:>12.2f} {a_r:>9.2f} {a_o:>9.2f} {m:>8.2f}  {v}")
