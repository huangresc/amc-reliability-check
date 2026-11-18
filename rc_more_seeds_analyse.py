# -*- coding: utf-8 -*-
"""Analyse the 30-seed re-split run (results_more_seeds.json)."""
import json
import numpy as np
from scipy.stats import wilcoxon, wilcoxon as wilx

CHECKPOINTS = ["20", "40", "80", "120", "160", "240", "320", "480", "640"]

with open("results_more_seeds.json", encoding="utf-8") as f:
    d = json.load(f)

seeds = sorted({int(k.split("_")[0][4:]) for k in d if k.startswith("seed")})
gap = {s: {e: d[f"seed{s}_GAP"][e] for e in CHECKPOINTS} for s in seeds}
flat = {s: {e: d[f"seed{s}_FLAT"][e] for e in CHECKPOINTS} for s in seeds}

print(f"n seeds = {len(seeds)}")
fails = [s for s in seeds if flat[s]["640"] < 15.0]
print(f"FLAT failures (ep640 < 15%): {len(fails)}/{len(seeds)} -> {fails}")

# Wilson 95% for failure rate
from scipy.stats import norm
def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z*z/n
    center = (p + z*z/(2*n)) / denom
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return max(0, center-half), min(1, center+half)

k, n = len(fails), len(seeds)
lo, hi = wilson(k, n)
print(f"failure rate: {k}/{n} = {100*k/n:.1f}%, Wilson 95% [{100*lo:.1f}%, {100*hi:.1f}%]")

print("\ncheckpoint | mean gap | Wilcoxon p | n significant sign")
for e in CHECKPOINTS:
    diffs = np.array([gap[s][e] - flat[s][e] for s in seeds])
    w = wilcoxon(diffs, zero_method="wilcox")
    neg = int((diffs < 0).sum())
    pos = int((diffs > 0).sum())
    print(f"  ep{e:>3} | {diffs.mean():+7.3f} | {w.pvalue:.4f} | GAP>FLAT {pos}, FLAT>GAP {neg}")

# full trajectory of any failures
for s in fails:
    print(f"\nseed {s} FLAT trajectory:")
    for e in CHECKPOINTS:
        print(f"  ep{e:>3}: {flat[s][e]:.2f}")
