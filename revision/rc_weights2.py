"""
Weight-space detectability across every official-split FLAT run that has
early checkpoints.

For each run: rel_fc1(e) = ||W_fc1(e) - W_fc1(0)||_F / ||W_fc1(0)||_F, where
W_fc1(0) is reconstructed from the seed (see rc_weights.py for why that is
exact). Runs are grouped by outcome; a milestone separates the groups iff
max(collapsing) < min(healthy).
"""
import os
import sys
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from src.models import M4_FlattenFC  # noqa: E402

OFFICIAL = "0ff510c64bc09b86e1a8224246292c9095c75d89"
ROOT = os.path.join("runs", "M4_FlattenFC")
MILESTONES = [1, 2, 3, 4, 5, 7, 10, 20, 40]


def init_fc1(seed):
    torch.manual_seed(seed)
    m = M4_FlattenFC(num_classes=11)
    return m.state_dict()["classifier.1.weight"].detach().clone()


def collect():
    out = []
    for d in sorted(os.listdir(ROOT)):
        dirp = os.path.join(ROOT, d)
        mp, kp = os.path.join(dirp, "manifest.json"), os.path.join(dirp, "metrics.json")
        if not (os.path.exists(mp) and os.path.exists(kp)):
            continue
        with open(mp) as f:
            man = json.load(f)
        if man["nuisance"]["data"]["train_idx_sha1"] != OFFICIAL:
            continue
        with open(kp) as f:
            met = json.load(f)
        eps = {int(k): v["acc"] for k, v in met["epochs"].items()}
        final = eps[max(eps)]
        cks = {}
        for fn in os.listdir(dirp):
            if fn.startswith("ckpt_ep") and fn.endswith(".pt"):
                e = int(fn[7:10])
                cks[e] = torch.load(os.path.join(dirp, fn), map_location="cpu",
                                    weights_only=False)["model_state_dict"]
        if not cks:
            continue
        W0 = init_fc1(man["seed"])
        n0 = W0.norm().item()
        rels = {e: ((sd["classifier.1.weight"] - W0).norm().item() / n0)
                for e, sd in cks.items()}
        out.append({
            "run": d, "seed": man["seed"],
            "outcome": "COLLAPSE" if final < 15.0 else "healthy",
            "final": final, "rel": rels,
            "pinned": "shuffle_seed" in man["nuisance"],
        })
        del cks
    return out


def main():
    runs = collect()
    bad = [r for r in runs if r["outcome"] == "COLLAPSE"]
    good = [r for r in runs if r["outcome"] == "healthy"]
    print(f"official-split FLAT runs with checkpoints: {len(runs)}"
          f"  (collapsing {len(bad)}, healthy {len(good)})")
    print("\ncollapsing runs:")
    for r in bad:
        print(f"  {r['run']:<34} final={r['final']:5.2f}%  eps={sorted(r['rel'])}")

    print("\n=== fc1.W relative displacement ||W(e)-W(0)||/||W(0)|| ===")
    print(f"  {'ep':>4} {'max(collapse)':>14} {'min(healthy)':>13} "
          f"{'ratio':>8}  separable")
    first = None
    for e in MILESTONES:
        cb = [r["rel"][e] for r in bad if e in r["rel"]]
        gd = [r["rel"][e] for r in good if e in r["rel"]]
        if not cb or not gd:
            continue
        hi, lo = max(cb), min(gd)
        sep = hi < lo
        if sep and first is None:
            first = e
        print(f"  {e:>4} {hi:>14.4f} {lo:>13.4f} {lo/hi:>8.1f}x  "
              f"{'YES' if sep else 'no'}")
    print(f"  earliest separating milestone (weight space): {first}")

    print("\n=== trajectory of the collapsing runs vs the healthy envelope ===")
    print(f"  {'ep':>4} " + "".join(f"{r['run'][:14]:>16}" for r in bad)
          + f"{'healthy min':>14}{'healthy max':>14}")
    for e in MILESTONES:
        row = f"  {e:>4} "
        for r in bad:
            row += f"{r['rel'].get(e, float('nan')):>16.4f}"
        gd = [r["rel"][e] for r in good if e in r["rel"]]
        if gd:
            row += f"{min(gd):>14.4f}{max(gd):>14.4f}"
        print(row)

    # the key pair: same init seed 42
    print("\n=== same initialisation (seed 42), different outcome ===")
    for r in runs:
        if r["seed"] == 42:
            tag = "pinned order" if r["pinned"] else "ambient order"
            print(f"  {r['run']:<34} [{tag:<13}] final={r['final']:5.2f}%  "
                  + "  ".join(f"ep{e}={r['rel'][e]:.4f}"
                              for e in sorted(r["rel"])[:7]))

    with open(os.path.join("analysis_results", "rc_weights_traj.json"), "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "rel"} |
                   {"rel": {str(k): v for k, v in r["rel"].items()}}
                   for r in runs], f, indent=2)


if __name__ == "__main__":
    main()
