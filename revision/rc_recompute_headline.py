"""
Recompute the paper's headline comparison (Table tab:arch) on the OFFICIAL
110 000/110 000 split instead of the 176 000/44 000 re-split, using runs that
are already on disk.

Motivation: the manuscript's headline (n=10, re-split) and its mechanism /
detectability results (n=30, official split) currently sit on two different
data partitions, which a reviewer can read as post-hoc. The ten original seeds
of both heads were also trained on the official split under identical
hyperparameters (`*_minverify_*`, normalization=none, 640 epochs, no
augmentation, same nuisance_hash). If the headline conclusion survives the
partition change, the paper can be unified onto one split at zero new compute.

Nothing here trains or touches a GPU: it reads metrics.json only.

Methodology is matched to the project's existing scripts -- paired two-sided
Wilcoxon signed-rank via scipy ``stats.wilcoxon(d).pvalue`` (see _stats.py:48).
"""
import os
import json
import math
import itertools

import scipy.stats as stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))

OFFICIAL = "0ff510c64bc09b86e1a8224246292c9095c75d89"
RESPLIT = "1b078bc88d4677b5aa16a72093b044a251e0679d"
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
CHECKPOINTS = ["20", "40", "80", "120", "160", "240", "320", "480", "640"]
COLLAPSE = 15.0

TAG = {
    "official": "minverify",
    "resplit": "batch2a",
}


def load(variant, split, seed):
    """Return {epoch: acc} for the run matching (variant, split, seed)."""
    root = os.path.join("runs", variant)
    for d in sorted(os.listdir(root)):
        mp = os.path.join(root, d, "manifest.json")
        kp = os.path.join(root, d, "metrics.json")
        if not (os.path.exists(mp) and os.path.exists(kp)):
            continue
        with open(mp) as f:
            man = json.load(f)
        if man["seed"] != seed:
            continue
        if man["nuisance"]["total_epochs"] != 640:
            continue
        if man["nuisance"]["normalization"] != "none":
            continue
        if man["treatment"]["use_aug"]:
            continue
        want = OFFICIAL if split == "official" else RESPLIT
        if man["nuisance"]["data"]["train_idx_sha1"] != want:
            continue
        if TAG[split] not in d:
            continue
        with open(kp) as f:
            met = json.load(f)
        ep = {k: v["acc"] for k, v in met["epochs"].items()}
        if "640" not in ep:
            continue
        return d, ep
    return None, None


def collect(split):
    out = {}
    for variant in ("M4_FlattenFC", "M2_GAPHead"):
        for seed in SEEDS:
            d, ep = load(variant, split, seed)
            if d is None:
                print(f"  MISSING {split} {variant} seed {seed}")
                continue
            out[(variant, seed)] = (d, ep)
    return out


def diff_series(data, seed_list, cp):
    """GAP minus FLAT per seed at one checkpoint."""
    return [data[("M2_GAPHead", s)][1][cp] - data[("M4_FlattenFC", s)][1][cp]
            for s in seed_list]


def report(split, data):
    print(f"\n{'=' * 72}\n{split.upper()} split\n{'=' * 72}")

    flat = {s: data[("M4_FlattenFC", s)][1]["640"] for s in SEEDS}
    gap = {s: data[("M2_GAPHead", s)][1]["640"] for s in SEEDS}
    failed = [s for s in SEEDS if flat[s] < COLLAPSE or gap[s] < COLLAPSE]
    for s in failed:
        which = []
        if flat[s] < COLLAPSE:
            which.append(f"FLAT {flat[s]:.2f}%")
        if gap[s] < COLLAPSE:
            which.append(f"GAP {gap[s]:.2f}%")
        print(f"  failed run: seed {s}  ->  " + ", ".join(which))
    if not failed:
        print("  failed runs: none")
    surviving = [s for s in SEEDS if s not in failed]

    print(f"\n  {'epochs':>7} | {'complete (n=10)':>26} | "
          f"{'subsample (n=' + str(len(surviving)) + ')':>26}")
    print(f"  {'':>7} | {'diff':>9} {'p':>7} {'':>8} | {'diff':>9} {'p':>7}")
    print("  " + "-" * 68)

    rows = {}
    for cp in CHECKPOINTS:
        d_all = diff_series(data, SEEDS, cp)
        d_sub = diff_series(data, surviving, cp)
        m_all, m_sub = sum(d_all) / len(d_all), sum(d_sub) / len(d_sub)
        try:
            p_all = stats.wilcoxon(d_all).pvalue
        except Exception:
            p_all = float("nan")
        try:
            p_sub = stats.wilcoxon(d_sub).pvalue
        except Exception:
            p_sub = float("nan")
        star_all = "*" if p_all < 0.05 else " "
        star_sub = "*" if p_sub < 0.05 else " "
        print(f"  {cp:>7} | {m_all:>+9.3f} {p_all:>7.4f}{star_all} {'':>7} | "
              f"{m_sub:>+9.3f} {p_sub:>7.4f}{star_sub}")
        rows[cp] = (m_all, p_all, m_sub, p_sub)

    print("  (* = p < 0.05; diff is GAP minus FLAT, percentage points)")

    # effect-size inflation at the longest budget
    m_all, p_all, m_sub, p_sub = rows["640"]
    if abs(m_sub) > 1e-9:
        infl = (abs(m_all) - abs(m_sub)) / abs(m_sub) * 100
        print(f"\n  at 640 epochs: complete {m_all:+.3f} pp, subsample "
              f"{m_sub:+.3f} pp  ->  inflation {infl:+.0f}%")

    # low-budget reversal
    m_all, p_all, m_sub, p_sub = rows["20"]
    print(f"  at  20 epochs: complete {m_all:+.3f} pp (p={p_all:.4f}), "
          f"subsample {m_sub:+.3f} pp (p={p_sub:.4f})")
    if m_sub < 0 < m_all:
        print("    -> the failed run converts a significant FLAT lead into a "
              "non-significant GAP lead: reversal reproduced")

    # crossover: last checkpoint negative -> first positive, on both columns
    def crossover(series_seeds):
        prev = None
        for cp in CHECKPOINTS:
            ds = diff_series(data, series_seeds, cp)
            m = sum(ds) / len(ds)
            if prev is not None and prev[1] < 0 <= m:
                return prev[0], cp
            prev = (cp, m)
        return None

    print(f"\n  sign change (negative -> positive), complete : {crossover(SEEDS)}")
    print(f"  sign change (negative -> positive), subsample: "
          f"{crossover(surviving)}")

    # leave-one-out at the shortest budget, matching §V-B's caveat
    if surviving:
        print(f"\n  leave-one-out over the {len(surviving)} surviving seeds at "
              f"20 epochs:")
        ps = []
        for drop in surviving:
            keep = [s for s in surviving if s != drop]
            d = diff_series(data, keep, "20")
            try:
                p = stats.wilcoxon(d).pvalue
            except Exception:
                p = float("nan")
            ps.append((drop, sum(d) / len(d), p))
        sig = [x for x in ps if x[2] < 0.05]
        print(f"    p < 0.05 in {len(sig)}/{len(ps)} drop-one configurations; "
              f"remaining at p = "
              f"{sorted({round(x[2], 3) for x in ps if x[2] >= 0.05})}")
        print(f"    signs negative in "
              f"{sum(1 for x in ps if x[1] < 0)}/{len(ps)}; range "
              f"{min(x[1] for x in ps):+.2f} to {max(x[1] for x in ps):+.2f} pp")
    return rows


def main():
    results = {}
    for split in ("resplit", "official"):
        data = collect(split)
        results[split] = report(split, data)

    print(f"\n{'=' * 72}\nVERDICT\n{'=' * 72}")
    a = results["resplit"]
    b = results["official"]
    for cp in CHECKPOINTS:
        ra, rb = a[cp], b[cp]
        same_sign = (ra[0] > 0) == (rb[0] > 0)
        same_sig = (ra[1] < 0.05) == (rb[1] < 0.05)
        flag = "same" if (same_sign and same_sig) else "DIFFERS"
        print(f"  ep{cp:>4}  re-split {ra[0]:>+6.3f} pp p={ra[1]:.4f} | "
              f"official {rb[0]:>+6.3f} pp p={rb[1]:.4f}   {flag}")

    out = {split: {cp: {"complete_pp": r[0], "complete_p": r[1],
                        "subsample_pp": r[2], "subsample_p": r[3]}
                   for cp, r in rows.items()}
           for split, rows in results.items()}
    with open(os.path.join("analysis_results", "rc_headline_official.json"),
              "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
