"""
Disentangle two things the headline recomputation changes at once.

rc_recompute_headline.py varies the data partition AND the identity of the
failed run: on the 176 000/44 000 re-split the collapse is seed 2021, on the
official 110 000/110 000 split it is seed 42. The two subsample columns
therefore drop different seeds, so the columns differ in two ways at once and
the difference cannot be attributed to the partition alone.

This script separates them:

  (1) per-seed GAP-FLAT at the shortest budget, both partitions, so the
      structure of the paired differences is visible rather than summarised;
  (2) leave-one-out over ALL ten seeds (not only the surviving nine) at each
      partition -- i.e. what the contrast looks like if each seed in turn is
      the one discarded, which is the correct placebo for "which run failed";
  (3) the explicit cross pair: drop seed 42 from the re-split and seed 2021
      from the official split, so that in both columns the SAME seed is
      removed and only the partition differs.

Reads metrics.json only; no training.
"""
import os
import json

import scipy.stats as stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OFFICIAL = "0ff510c64bc09b86e1a8224246292c9095c75d89"
RESPLIT = "1b078bc88d4677b5aa16a72093b044a251e0679d"
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
TAG = {"official": "minverify", "resplit": "batch2a"}
COLLAPSE = 15.0


def load(variant, split, seed):
    root = os.path.join("runs", variant)
    for d in sorted(os.listdir(root)):
        mp, kp = os.path.join(root, d, "manifest.json"), \
            os.path.join(root, d, "metrics.json")
        if not (os.path.exists(mp) and os.path.exists(kp)):
            continue
        with open(mp) as f:
            man = json.load(f)
        if (man["seed"] != seed
                or man["nuisance"]["total_epochs"] != 640
                or man["nuisance"]["normalization"] != "none"
                or man["treatment"]["use_aug"]
                or TAG[split] not in d):
            continue
        want = OFFICIAL if split == "official" else RESPLIT
        if man["nuisance"]["data"]["train_idx_sha1"] != want:
            continue
        with open(kp) as f:
            met = json.load(f)
        ep = {k: v["acc"] for k, v in met["epochs"].items()}
        if "640" in ep:
            return ep
    return None


def build(split):
    data = {}
    for v in ("M4_FlattenFC", "M2_GAPHead"):
        for s in SEEDS:
            ep = load(v, split, s)
            if ep is None:
                raise SystemExit(f"missing {split} {v} seed {s}")
            data[(v, s)] = ep
    return data


def diffs(data, seeds, cp):
    return [data[("M2_GAPHead", s)][cp] - data[("M4_FlattenFC", s)][cp]
            for s in seeds]


def summarize(d):
    m = sum(d) / len(d)
    try:
        p = stats.wilcoxon(d).pvalue
    except Exception:
        p = float("nan")
    return m, p


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


res = build("resplit")
off = build("official")

print("=" * 74)
print("(1) per-seed GAP minus FLAT at 20 and 640 epochs")
print("=" * 74)
print(f"  {'seed':>6} | {'re-split ep20':>13} {'ep640':>8} {'final FLAT':>11}"
      f" | {'official ep20':>13} {'ep640':>8} {'final FLAT':>11}")
print("  " + "-" * 88)
for s in SEEDS:
    r20 = res[("M2_GAPHead", s)]["20"] - res[("M4_FlattenFC", s)]["20"]
    r64 = res[("M2_GAPHead", s)]["640"] - res[("M4_FlattenFC", s)]["640"]
    o20 = off[("M2_GAPHead", s)]["20"] - off[("M4_FlattenFC", s)]["20"]
    o64 = off[("M2_GAPHead", s)]["640"] - off[("M4_FlattenFC", s)]["640"]
    rf = res[("M4_FlattenFC", s)]["640"]
    of = off[("M4_FlattenFC", s)]["640"]
    mark_r = " <-- collapse" if rf < COLLAPSE else ""
    mark_o = " <-- collapse" if of < COLLAPSE else ""
    print(f"  {s:>6} | {r20:>+13.3f} {r64:>+8.3f} {rf:>10.2f}{mark_r:<12}"
          f" | {o20:>+13.3f} {o64:>+8.3f} {of:>10.2f}{mark_o}")

print()
print("=" * 74)
print("(2) leave-one-out over ALL ten seeds (placebo: each seed discarded in turn)")
print("=" * 74)
for label, data in (("re-split", res), ("official", off)):
    for cp in ("20", "640"):
        ps = []
        for drop in SEEDS:
            keep = [s for s in SEEDS if s != drop]
            m, p = summarize(diffs(data, keep, cp))
            ps.append((drop, m, p))
        sig = [x for x in ps if x[2] < 0.05]
        lo = min(x[1] for x in ps)
        hi = max(x[1] for x in ps)
        print(f"\n  {label}, epoch {cp}: mean ranges {lo:+.2f} to {hi:+.2f} pp; "
              f"p < 0.05 in {len(sig)}/10 drop-one configurations")
        for drop, m, p in ps:
            tag = ""
            if drop in (42, 2021):
                tag = "   <-- the seed that collapses on THIS partition"
            print(f"    drop {drop:>6}  mean {m:>+7.3f} pp  p={p:.4f}"
                  f"{'  *' if p < 0.05 else '   '}{tag}")

print()
print("=" * 74)
print("(3) same seed dropped on both partitions -- partition is the only change")
print("=" * 74)
print(f"  {'column':>34} | {'ep20 mean':>10} {'p':>7} | {'ep640 mean':>11} {'p':>7}")
print("  " + "-" * 80)
for label, data, drop in (("re-split, drop seed 2021 (real collapse)", res, 2021),
                          ("official, drop seed 2021 (placebo)", off, 2021),
                          ("official, drop seed 42 (real collapse)", off, 42),
                          ("re-split, drop seed 42 (placebo)", res, 42)):
    keep = [s for s in SEEDS if s != drop]
    m20, p20 = summarize(diffs(data, keep, "20"))
    m64, p64 = summarize(diffs(data, keep, "640"))
    print(f"  {label:>34} | {m20:>+10.3f} {p20:>7.4f} | {m64:>+11.3f} {p64:>7.4f}")

print()
print("=" * 74)
print("(4) failure rate per partition, FLAT head, identical hyperparameters")
print("=" * 74)
for label, data in (("re-split", res), ("official", off)):
    fin = [data[("M4_FlattenFC", s)]["640"] for s in SEEDS]
    bad = sum(1 for a in fin if a < COLLAPSE)
    lo, hi = wilson(bad, len(fin))
    ids = [s for s in SEEDS if data[("M4_FlattenFC", s)]["640"] < COLLAPSE]
    print(f"  {label:>9}: {bad}/10 collapsed  Wilson95=[{lo*100:.1f}%, "
          f"{hi*100:.1f}%]  seeds={ids}")
