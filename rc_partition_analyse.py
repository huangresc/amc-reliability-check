# -*- coding: utf-8 -*-
"""Analyse the partition-isolation arm (perm_seed 43) and contrast with the
paper's re-split (perm_seed 42).  Reports, per checkpoint: the paired
GAP-minus-FLAT difference (mean), the two-sided Wilcoxon signed-rank p, and the
failure count (final accuracy < 15 %)."""
import csv
import io
import json
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

import numpy as np
from scipy.stats import wilcoxon

CHECKPOINTS = ["20", "40", "80", "120", "160", "240", "320", "480", "640"]
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]


def load43(path="analysis_results/rc_partition_isolate.json"):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    gap, flat = {}, {}
    for s in SEEDS:
        gap[s] = {e: d[f"seed{s}_GAP"][e] for e in CHECKPOINTS}
        flat[s] = {e: d[f"seed{s}_FLAT"][e] for e in CHECKPOINTS}
    return gap, flat


def load42(path="tables/main_architecture_per_seed.csv"):
    gap, flat = {}, {}
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        if int(r["seed"]) not in SEEDS:
            continue
        ep = {e: float(r["ep" + e]) for e in CHECKPOINTS}
        if r["variant"] == "M2_GAPHead":
            gap[int(r["seed"])] = ep
        elif r["variant"] == "M4_FlattenFC":
            flat[int(r["seed"])] = ep
    return gap, flat


def report(tag, gap, flat):
    print(f"\n=== {tag} ===")
    fails = [s for s in SEEDS if flat[s]["640"] < 15.0]
    print(f"  FLAT failures (final<15%): {len(fails)}/10 -> {fails}")
    print(f"  {'epoch':>6} {'mean gap':>9} {'Wilcoxon p':>11}")
    for e in CHECKPOINTS:
        d = np.array([gap[s][e] - flat[s][e] for s in SEEDS])
        w = wilcoxon(d, zero_method="wilcox")
        print(f"  {e:>6} {d.mean():>+9.3f} {w.pvalue:>11.4f}")


if __name__ == "__main__":
    g43, f43 = load43()
    g42, f42 = load42()
    report("perm_seed=43 (new draw, same size/method)", g43, f43)
    report("perm_seed=42 (paper re-split)", g42, f42)
    # seed 2021 trajectory on both draws
    print("\n=== seed 2021 FLAT trajectory ===")
    for e in CHECKPOINTS:
        print(f"  ep{e:>3}: seed42={f42[2021][e]:6.2f}  seed43={f43[2021][e]:6.2f}")
