"""
Corrected analysis: strictly the official 110 000/110 000 split, 640 epochs,
mini-batch order left to the ambient RNG stream (i.e. NOT the pinned-order arm).

The *_batch2a / *_gate1 runs use the old 176 000/44 000 re-split and are
excluded -- mixing them with the rc30 runs would compare two protocols.
"""
import os
import json
import collections

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OFFICIAL_IDX_SHA1 = "0ff510c64bc09b86e1a8224246292c9095c75d89"
COLLAPSE = 15.0

ORIGINAL = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
NEW = [7, 13, 55, 101, 256, 512, 777, 1001, 1500, 2048,
       3141, 4096, 5150, 6666, 7777, 8888, 9999, 12345, 24680, 31337]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def collect(variant):
    """Return {seed: (final_acc, tag)} for official-split, 640-epoch,
    ambient-order runs."""
    out = {}
    root = os.path.join("runs", variant)
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "manifest.json")
        kp = os.path.join(root, d, "metrics.json")
        if not (os.path.exists(mp) and os.path.exists(kp)):
            continue
        with open(mp) as f:
            man = json.load(f)
        nu = man["nuisance"]
        if nu["data"]["train_idx_sha1"] != OFFICIAL_IDX_SHA1:
            continue
        if nu["total_epochs"] != 640:
            continue
        if "shuffle_seed" in nu:          # pinned-order arm, analysed apart
            continue
        with open(kp) as f:
            met = json.load(f)
        last = max(int(k) for k in met["epochs"])
        if last != 640:
            continue
        seed = man["seed"]
        prev = out.get(seed)
        if prev is None or d < prev[2]:
            out[seed] = (met["epochs"]["640"]["acc"], d, d)
    return {s: (v[0], v[1]) for s, v in out.items()}


def show(label, table, seeds):
    rows = [(s, table[s]) for s in seeds if s in table]
    missing = [s for s in seeds if s not in table]
    print(f"\n=== {label}   n={len(rows)} ===")
    for s, (a, tag) in sorted(rows, key=lambda r: r[1][0]):
        print(f"  seed{s:<8} {a:6.2f}%   [{tag}]"
              + ("   <-- COLLAPSE" if a < COLLAPSE else ""))
    bad = sum(1 for _, (a, _) in rows if a < COLLAPSE)
    lo, hi = wilson(bad, len(rows))
    good = [a for _, (a, _) in rows if a >= COLLAPSE]
    print(f"  collapsed {bad}/{len(rows)}  Wilson95=[{lo*100:.1f}%, {hi*100:.1f}%]")
    if good:
        print(f"  healthy mean={sum(good)/len(good):.2f}%  "
              f"range=[{min(good):.2f}, {max(good):.2f}]")
    if missing:
        print(f"  MISSING official-split runs for seeds: {missing}")
    return bad, len(rows)


flat = collect("M4_FlattenFC")
gap = collect("M2_GAPHead")

print("official split 110k/110k, 640 epochs, ambient mini-batch order")

bf, nf = show("FLAT head (M4_FlattenFC)", flat, ORIGINAL + NEW)
bg, ng = show("GAP head  (M2_GAPHead)", gap, ORIGINAL + NEW)

shared = [s for s in ORIGINAL + NEW if s in flat and s in gap]
print(f"\n=== paired FLAT - GAP, n={len(shared)} ===")
d = []
for s in shared:
    af = flat[s][0]
    ag = gap[s][0]
    d.append((s, af, ag, af - ag))
for s, af, ag, dd in sorted(d, key=lambda r: r[3]):
    print(f"  seed{s:<8} FLAT={af:6.2f}  GAP={ag:6.2f}  d={dd:+6.2f}"
          + ("   <-- FLAT collapsed" if af < COLLAPSE else ""))
healthy = [dd for _, af, _, dd in d if af >= COLLAPSE]
allof = [dd for _, _, _, dd in d]
for name, vals in [("healthy only", healthy), ("all runs", allof)]:
    n = len(vals)
    mean = sum(vals) / n
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    se = (var / n) ** 0.5
    print(f"  {name:<13} n={n}  mean={mean:+.2f}pp  "
          f"95%CI=[{mean-1.96*se:+.2f}, {mean+1.96*se:+.2f}]  "
          f"negative {sum(1 for x in vals if x < 0)}/{n}")
