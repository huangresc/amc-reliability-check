"""
Weight-space detectability restricted to exactly the SEED30 sample, i.e. the
same 30 official-split / 640-epoch / ambient-order runs per head that
rc_analyse3.py and rc_detect.py use for the accuracy statements.

rc_weights2.py swept every official-split FLAT run that happened to retain
checkpoints, which pulls in gate/probe runs outside the thirty and is not the
sample the manuscript's population claims refer to. This script keeps the two
halves of the detectability claim on one population.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from src.models import M4_FlattenFC  # noqa: E402
from rc_analyse3 import collect, ORIGINAL, NEW, COLLAPSE  # noqa: E402

MILESTONES = [1, 2, 3, 4, 5, 7, 10, 20, 40]


def init_fc1(seed):
    torch.manual_seed(seed)
    m = M4_FlattenFC(num_classes=11)
    return m.state_dict()["classifier.1.weight"].detach().clone()


def main():
    table = collect("M4_FlattenFC")
    runs = []
    for seed in ORIGINAL + NEW:
        if seed not in table:
            continue
        acc, tag = table[seed]
        d = os.path.join("runs", "M4_FlattenFC", tag)
        cks = {}
        for fn in os.listdir(d):
            if fn.startswith("ckpt_ep") and fn.endswith(".pt"):
                e = int(fn[7:10])
                cks[e] = torch.load(os.path.join(d, fn), map_location="cpu",
                                    weights_only=False)["model_state_dict"]
        if not cks:
            continue
        W0 = init_fc1(seed)
        n0 = W0.norm().item()
        rel = {e: ((sd["classifier.1.weight"] - W0).norm().item() / n0)
               for e, sd in cks.items()}
        runs.append({"seed": seed, "tag": tag, "final": acc,
                     "outcome": "COLLAPSE" if acc < COLLAPSE else "healthy",
                     "rel": rel})
        del cks

    bad = [r for r in runs if r["outcome"] == "COLLAPSE"]
    good = [r for r in runs if r["outcome"] == "healthy"]
    print(f"SEED30 FLAT runs with checkpoints: {len(runs)}"
          f"  (collapsing {len(bad)}, healthy {len(good)})")
    print("collapsing runs and the checkpoints they retain:")
    for r in bad:
        print(f"  {r['tag']:<34} seed{r['seed']:<7} final={r['final']:5.2f}%"
              f"  eps={sorted(r['rel'])}")
    print(f"healthy runs retaining an epoch-1 checkpoint: "
          f"{sum(1 for r in good if 1 in r['rel'])}/{len(good)}")

    print("\n=== fc1.W relative displacement ||W(e)-W(0)||/||W(0)|| ===")
    print(f"  {'ep':>4} {'max(collapse)':>14} {'min(healthy)':>13} "
          f"{'ratio':>8}  separable   n_col  n_hlth")
    for e in MILESTONES:
        cb = [r["rel"][e] for r in bad if e in r["rel"]]
        gd = [r["rel"][e] for r in good if e in r["rel"]]
        if not cb or not gd:
            print(f"  {e:>4} {'--':>14} {'--':>13} {'':>8}  no data"
                  f"      {len(cb):>5}  {len(gd):>6}")
            continue
        hi, lo = max(cb), min(gd)
        print(f"  {e:>4} {hi:>14.4f} {lo:>13.4f} {lo/hi:>8.2f}x  "
              f"{'YES' if hi < lo else 'no':<10}  {len(cb):>5}  {len(gd):>6}")

    print("\n=== trajectory of the collapsing runs vs the healthy envelope ===")
    print(f"  {'ep':>4} " + "".join(f"{'s' + str(r['seed']):>10}" for r in bad)
          + f"{'healthy min':>14}{'healthy max':>14}")
    for e in MILESTONES:
        row = f"  {e:>4} "
        for r in bad:
            v = r["rel"].get(e)
            row += f"{'--':>10}" if v is None else f"{v:>10.4f}"
        gd = [r["rel"][e] for r in good if e in r["rel"]]
        if gd:
            row += f"{min(gd):>14.4f}{max(gd):>14.4f}"
        print(row)


if __name__ == "__main__":
    main()
