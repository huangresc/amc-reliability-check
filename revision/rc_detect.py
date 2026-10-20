"""
Detectability: at which checkpoint can the collapse be separated from runs
that will recover? Uses only the official-split, 640-epoch, ambient-order
FLAT runs, so the sample is protocol-consistent.

Reports, per early milestone, the highest accuracy among collapsing runs and
the lowest among runs that end healthy. Separation exists at milestone e iff
max(collapsing) < min(healthy).
"""
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OFFICIAL = "0ff510c64bc09b86e1a8224246292c9095c75d89"
COLLAPSE = 15.0
EARLY = [1, 2, 3, 4, 5, 7, 10, 20, 40]


def runs(variant):
    out = []
    root = os.path.join("runs", variant)
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "manifest.json")
        kp = os.path.join(root, d, "metrics.json")
        if not (os.path.exists(mp) and os.path.exists(kp)):
            continue
        with open(mp) as f:
            man = json.load(f)
        nu = man["nuisance"]
        if nu["data"]["train_idx_sha1"] != OFFICIAL:
            continue
        if nu["total_epochs"] != 640 or "shuffle_seed" in nu:
            continue
        with open(kp) as f:
            met = json.load(f)
        ep = {int(k): v["acc"] for k, v in met["epochs"].items()}
        if 640 not in ep:
            continue
        out.append((man["seed"], d, ep))
    return out


for variant in ["M4_FlattenFC", "M2_GAPHead"]:
    rs = runs(variant)
    # if a seed has several runs, keep the alphabetically-first (matches other script)
    seen = {}
    for s, d, ep in rs:
        if s not in seen or d < seen[s][0]:
            seen[s] = (d, ep)
    rs = [(s, d, ep) for s, (d, ep) in sorted(seen.items())]

    bad = [(s, ep) for s, d, ep in rs if ep[640] < COLLAPSE]
    good = [(s, ep) for s, d, ep in rs if ep[640] >= COLLAPSE]
    print(f"\n=== {variant}   n={len(rs)}  collapsing={len(bad)} ===")

    print(f"  {'milestone':<10} {'max(collapse)':>14} {'min(healthy)':>13} "
          f"{'gap':>7}  separable")
    first = None
    for e in EARLY:
        cb = [ep[e] for _, ep in bad if e in ep]
        gd = [ep[e] for _, ep in good if e in ep]
        if not cb or not gd:
            continue
        hi = max(cb)
        lo = min(gd)
        sep = hi < lo
        if sep and first is None:
            first = e
        print(f"  {e:<10} {hi:>13.2f}% {lo:>12.2f}% {lo-hi:>7.2f}  "
              f"{'YES' if sep else 'no'}")

    print(f"  earliest separating milestone: {first}")

    # which healthy runs are still at chance early on -- the false-positive cost
    for e in EARLY[:6]:
        at_chance = [s for s, ep in good if e in ep and ep[e] < COLLAPSE]
        if at_chance:
            print(f"  ep{e}: {len(at_chance)}/{len(good)} healthy runs "
                  f"still < {COLLAPSE}%  -> {at_chance}")
