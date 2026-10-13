# -*- coding: utf-8 -*-
"""Reviewer concern 4: multiplicity, and inference that respects the design.

THE COMPLAINT
Nine checkpoints of the same ten paired runs are reported as nine separate
Wilcoxon tests.  The checkpoints are repeated measures on the same seeds, so
they are neither independent nor a family of nine unrelated hypotheses; no
multiplicity correction is applied anywhere; and at n=10 a Wilcoxon p-value is
a coarse object (the smallest attainable two-sided value is 0.00195).

WHAT THIS SCRIPT COMPUTES
For every contrast the manuscript actually tables, on the already-written runs:

  1. the current test, unchanged  -- paired two-sided Wilcoxon, n as reported;
  2. Holm-adjusted p within the family the manuscript presents as one table
     (nine checkpoints per column), which is the correction that makes the nine
     p-values tell one story instead of nine;
  3. a paired cluster bootstrap over seeds (BCa and percentile intervals) for
     the per-checkpoint mean difference, which does not assume normality and
     is therefore the right interval for a distribution the paper itself
     documents as bimodal;
  4. a linear mixed model on the paired-difference series with a random
     intercept per seed and checkpoint as a repeated-measures factor.  The test
     of the checkpoint factor is exactly the reviewer's model for "does the
     effect size depend on training budget" and, unlike nine separate tests,
     carries one p-value for the whole budget-dependence claim.

Nothing is trained, resampled on disk, or rewritten.  Read-only over
metrics.json / the per-seed CSVs.

CONCLUSION CARRIED INTO THE MANUSCRIPT: see the SUMMARY block at the end.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import scipy.stats as stats

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from tables_io import load_per_seed, load_official  # noqa: E402

CHECKPOINTS = ["20", "40", "80", "120", "160", "240", "320", "480", "640"]
NBOOT = 20000
BOOT_SEED = 20261013
COLLAPSE = 15.0


# ---------------------------------------------------------------- statistics
def holm(p):
    """Holm-Bonferroni adjusted p-values, order preserved."""
    p = np.asarray(p, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * p[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    return adj


def boot_mean(d, rng, n=NBOOT):
    d = np.asarray(d, dtype=float)
    n_obs = len(d)
    idx = rng.integers(0, n_obs, size=(n, n_obs))
    return d[idx].mean(axis=1)


def bca(d, theta_hat, boots, rng):
    """BCa interval.  Jackknife acceleration + bias correction."""
    d = np.asarray(d, dtype=float)
    n = len(d)
    jack = np.array([np.delete(d, i).mean() for i in range(n)])
    jm = jack.mean()
    num = ((jm - jack) ** 3).sum()
    den = 6.0 * (((jm - jack) ** 2).sum() ** 1.5)
    a = num / den if den != 0 else 0.0
    prop = np.mean(boots < theta_hat)
    prop = min(max(prop, 1.0 / (len(boots) + 1)), 1.0 - 1.0 / (len(boots) + 1))
    z0 = stats.norm.ppf(prop)
    zl, zu = stats.norm.ppf(0.025), stats.norm.ppf(0.975)

    def adj(z):
        return stats.norm.cdf(z0 + (z0 + z) / (1.0 - a * (z0 + z)))

    lo = np.percentile(boots, 100 * adj(zl))
    hi = np.percentile(boots, 100 * adj(zu))
    return lo, hi, a


def lmm_epoch_test(d, seeds):
    """LMM on the paired-difference series.

    d_{s,e} = mu_e + b_s + eps,  b_s ~ N(0, sigma_b^2)

    The joint Wald test on the checkpoint factor is the single hypothesis
    "the effect depends on training budget", spanning all nine checkpoints and
    accounting for the fact that they are repeated measures on the same seed.
    """
    import pandas as pd
    import statsmodels.formula.api as smf

    rows = []
    for s in seeds:
        for e in CHECKPOINTS:
            rows.append({"seed": s, "epoch": e, "d": d[s][e]})
    df = pd.DataFrame(rows)
    md = smf.mixedlm("d ~ C(epoch)", df, groups=df["seed"]).fit(reml=True)
    names = [n for n in md.params.index if n.startswith("C(epoch)")]
    wald = md.f_test(" = 0, ".join(names) + " = 0")
    var_b = float(np.asarray(md.cov_re)[0, 0])
    var_e = float(md.scale)
    b = md.random_effects
    return {
        "p_epoch": float(wald.pvalue),
        "f": float(np.asarray(wald.fvalue).ravel()[0]),
        "var_seed": var_b,
        "var_resid": var_e,
        "icc": var_b / (var_b + var_e) if (var_b + var_e) > 0 else float("nan"),
        "seed_effects": {str(k): float(np.asarray(v).ravel()[0])
                         for k, v in b.items()},
    }


# ------------------------------------------------------------------ reporting
def family(name, diffs, seeds, note=""):
    """diffs: {seed: {epoch: paired difference}}."""
    print("\n" + "=" * 78)
    print(f"{name}   (n={len(seeds)}, {len(CHECKPOINTS)} checkpoints)"
          + (f"   {note}" if note else ""))
    print("=" * 78)
    rng = np.random.default_rng(BOOT_SEED)
    raw, rows = [], []
    for e in CHECKPOINTS:
        d = np.array([diffs[s][e] for s in seeds], dtype=float)
        m = d.mean()
        try:
            p = stats.wilcoxon(d).pvalue
        except Exception:
            p = float("nan")
        boots = boot_mean(d, rng)
        lo_p, hi_p = np.percentile(boots, [2.5, 97.5])
        lo_b, hi_b, acc = bca(d, m, boots, rng)
        raw.append(p)
        rows.append({"epoch": e, "mean": m, "p_raw": p,
                     "ci_lo": lo_p, "ci_hi": hi_p,
                     "bca_lo": lo_b, "bca_hi": hi_b,
                     "boot_se": boots.std(ddof=1), "accel": acc,
                     "per_seed": d.tolist()})
    adj = holm(raw)
    for r, a in zip(rows, adj):
        r["p_holm"] = a
        r["sig_raw"] = r["p_raw"] < 0.05
        r["sig_holm"] = a < 0.05

    print(f"  {'ep':>5} | {'mean':>7} {'wilcoxon p':>10} {'Holm p':>7} |"
          f" {'95% CI (percentile)':>21} | {'95% CI (BCa)':>19} | flips")
    print("  " + "-" * 96)
    for r in rows:
        flag = "" if r["sig_raw"] == r["sig_holm"] else (
            "LOST" if r["sig_raw"] else "GAINED")
        print(f"  {r['epoch']:>5} | {r['mean']:>+7.3f} {r['p_raw']:>10.4f} "
              f"{r['p_holm']:>7.4f} | "
              f"[{r['ci_lo']:>+7.3f}, {r['ci_hi']:>+7.3f}]     | "
              f"[{r['bca_lo']:>+7.3f}, {r['bca_hi']:>+7.3f}] | {flag}")
    n_raw = sum(r["sig_raw"] for r in rows)
    n_holm = sum(r["sig_holm"] for r in rows)
    print(f"  significant at p<0.05: {n_raw}/9 raw, {n_holm}/9 Holm-adjusted")
    ci_excl0 = sum(1 for r in rows if r["bca_lo"] > 0 or r["bca_hi"] < 0)
    print(f"  BCa 95% CI excluding 0: {ci_excl0}/9")

    lmm = lmm_epoch_test(diffs, seeds)
    print(f"\n  LMM (random intercept per seed, checkpoint as repeated measure):")
    print(f"    joint test of the checkpoint factor: F = {lmm['f']:.2f}, "
          f"p = {lmm['p_epoch']:.4f}")
    print(f"    variance: seed {lmm['var_seed']:.3f} pp^2, residual "
          f"{lmm['var_resid']:.3f} pp^2, ICC = {lmm['icc']:.3f}")
    top = sorted(lmm["seed_effects"].items(), key=lambda kv: -abs(kv[1]))[:3]
    print(f"    largest seed effects (pp): "
          + ", ".join(f"{k} {v:+.2f}" for k, v in top))
    return {"rows": rows, "lmm": lmm, "n_sig_raw": n_raw, "n_sig_holm": n_holm,
            "n_bca_excl0": ci_excl0}


def main():
    perseed = load_per_seed()
    official, _ = load_official()
    all_seeds = sorted(perseed["M2_GAPHead"])

    # --- which seeds are collapsed at the end, per split (same rule as
    #     rc_recompute_headline.py: final accuracy < 15 %)
    def failed(flat, gap):
        return [s for s in all_seeds
                if flat[s]["640"] < COLLAPSE or gap[s]["640"] < COLLAPSE]

    def diff_map(a, b, seeds):
        return {s: {e: a[s][e] - b[s][e] for e in CHECKPOINTS} for s in seeds}

    out = {}

    # resplit families (per-seed CSVs)
    r_flat, r_gap = perseed["M4_FlattenFC"], perseed["M2_GAPHead"]
    r_fail = failed(r_flat, r_gap)
    r_keep = [s for s in all_seeds if s not in r_fail]
    print(f"\n[resplit] collapsed runs at 640 epochs: {r_fail or 'none'}"
          f"  -> subsample n={len(r_keep)}")

    out["resplit_arch_complete"] = family(
        "tab:arch col 1 -- GAP minus FLAT, complete",
        diff_map(r_gap, r_flat, all_seeds), all_seeds, "re-split")
    out["resplit_arch_subsample"] = family(
        "tab:arch col 2 -- GAP minus FLAT, subsample",
        diff_map(r_gap, r_flat, r_keep), r_keep, f"re-split, {r_fail} dropped")

    for v, label in (("M5_GAPHead_PhaseAug", "continuous"),
                     ("M5b_GAPHead_PhaseAug4", "discrete x4")):
        out[f"resplit_aug_{label.split()[0]}"] = family(
            f"tab:aug -- unaugmented GAP minus phase-rotated GAP ({label})",
            diff_map(r_gap, perseed[v], all_seeds), all_seeds, "re-split")

    # official split
    o_flat = {s: official[("M4_FlattenFC", s)] for s in all_seeds}
    o_gap = {s: official[("M2_GAPHead", s)] for s in all_seeds}
    o_fail = failed(o_flat, o_gap)
    o_keep = [s for s in all_seeds if s not in o_fail]
    print(f"\n[official] collapsed runs at 640 epochs: {o_fail or 'none'}"
          f"  -> subsample n={len(o_keep)}")

    out["official_arch_complete"] = family(
        "tab:partition col 1 -- GAP minus FLAT, complete",
        diff_map(o_gap, o_flat, all_seeds), all_seeds, "official split")
    out["official_arch_subsample"] = family(
        "tab:partition col 2 -- GAP minus FLAT, subsample",
        diff_map(o_gap, o_flat, o_keep), o_keep, f"official, {o_fail} dropped")

    # ------------------------------------------------------------- summary
    print("\n" + "=" * 78)
    print("SUMMARY -- what a corrected single family changes")
    print("=" * 78)
    print(f"  {'family':<32} {'raw sig':>8} {'Holm sig':>9} "
          f"{'BCa excl 0':>11} {'LMM p(budget)':>14}")
    for k, v in out.items():
        print(f"  {k:<32} {v['n_sig_raw']:>8} {v['n_sig_holm']:>9} "
              f"{v['n_bca_excl0']:>11} {v['lmm']['p_epoch']:>14.4f}")

    dst = os.path.join("analysis_results", "rc_inference.json")
    os.makedirs("analysis_results", exist_ok=True)
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"\nwrote {dst}")


if __name__ == "__main__":
    main()
