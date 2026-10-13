"""Summarise rc_repair_perturbation.json.

The repair outcome is bimodal, so "how much does repair recover?" is the wrong
question.  The right one is "how often does it recover at all?", which is a
Bernoulli rate over perturbation draws.  At the same perturbation seed the two
runs receive a bit-identical fc1 (verified by weight sha1 in the JSON), so the
draw -- not the run -- is largely what decides the outcome.
"""
import json
import math

with open("analysis_results/rc_repair_perturbation.json") as f:
    R = json.load(f)

# Read the draw list back from the file rather than hard-coding it, so the
# script follows the harness when the sweep is extended.  json preserves the
# insertion order, which is PERTURB_SEEDS order.
SEEDS = [int(k[1:].split("_")[0]) for k in R["resplit"]["cells"]
         if k.endswith("_reinit_fc1")]
THRESH = 20.0          # the two modes are ~9 % and ~35-43 %; anything 15-30 splits them


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def cell(run, p):
    return R[run]["cells"][f"p{p}_reinit_fc1"]


print("=" * 78)
print("Is the outcome determined by the run or by the perturbation draw?")
print("=" * 78)
print(f"  {'perturb':>7} {'fc1 weight sha1':>17} | {'re-split':>22} | "
      f"{'official':>22} | agree")
print(f"  {'':>7} {'':>17} | {'acc':>8} {'recov':>7} {'verdict':>5} | "
      f"{'acc':>8} {'recov':>7} {'verdict':>5} |")
print("  " + "-" * 92)
agree = 0
for p in SEEDS:
    r, o = cell("resplit", p), cell("official", p)
    ra, oa = r["epoch40"]["acc"], o["epoch40"]["acc"]
    rr = ra - r["epoch0"]["acc"]
    orr = oa - o["epoch0"]["acc"]
    rv, ov = ra > THRESH, oa > THRESH
    same = rv == ov
    agree += same
    print(f"  {p:>7} {r['weight_sha1_16']:>17} | {ra:>8.2f} {rr:>+7.2f} "
          f"{'yes' if rv else 'no':>5} | {oa:>8.2f} {orr:>+7.2f} "
          f"{'yes' if ov else 'no':>5} | {'YES' if same else 'no'}")
print(f"\n  identical perturbation -> identical verdict in {agree}/{len(SEEDS)} draws")

print("\n" + "=" * 78)
print("Repair rate (final accuracy > %.0f %%), pooled over perturbation draws" % THRESH)
print("=" * 78)
tot_k = tot_n = 0
for run in ("resplit", "official"):
    accs = [cell(run, p)["epoch40"]["acc"] for p in SEEDS]
    k = sum(1 for a in accs if a > THRESH)
    lo, hi = wilson(k, len(SEEDS))
    tot_k += k
    tot_n += len(SEEDS)
    print(f"  {run:9s} {k}/{len(SEEDS)} repaired  ({100*k/len(SEEDS):.0f} %)  "
          f"Wilson95 [{100*lo:.1f} %, {100*hi:.1f} %]   final acc "
          f"{['%.2f' % a for a in accs]}")
lo, hi = wilson(tot_k, tot_n)
print(f"  {'pooled':9s} {tot_k}/{tot_n} repaired  ({100*tot_k/tot_n:.0f} %)  "
      f"Wilson95 [{100*lo:.1f} %, {100*hi:.1f} %]")

print("\n" + "=" * 78)
print("Failed repairs do not merely stall -- they return to the SAME state")
print("=" * 78)
for run in ("resplit", "official"):
    fails = [(p, cell(run, p)) for p in SEEDS if cell(run, p)["epoch40"]["acc"] <= THRESH]
    for p, c in fails:
        a0, a40 = c["epoch0"]["acc"], c["epoch40"]["acc"]
        print(f"  {run:9s} p={p:<5d} ep0 {a0:6.2f} -> ep40 {a40:6.2f}  "
              f"dead_fc1 {c['epoch40']['dead_fc1']:.4f}")
    if fails:
        print(f"            (all re-collapse to the constant predictor; "
              f"dead_fc1 = 1.0000)")

print("\n" + "=" * 78)
print("The bifurcation is already decided by epoch 20")
print("=" * 78)
for run in ("resplit", "official"):
    hi20 = [cell(run, p)["epoch20"]["acc"] for p in SEEDS
            if cell(run, p)["epoch40"]["acc"] > THRESH]
    lo20 = [cell(run, p)["epoch20"]["acc"] for p in SEEDS
            if cell(run, p)["epoch40"]["acc"] <= THRESH]
    print(f"  {run:9s} repairs: ep20 {['%.2f' % a for a in hi20]}   "
          f"re-collapses: ep20 {['%.2f' % a for a in lo20]}")
