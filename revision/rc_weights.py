"""
Weight-displacement and dead-unit analysis (CPU only, reads existing .pt).

The claim under test is the manuscript's: in a collapsed FLAT run the first
fully-connected layer stops receiving gradient and therefore stays frozen at
its initialisation. That is a falsifiable prediction, because the
initialisation is reconstructible exactly:

    src/train.py:145  set_seed(seed)
    src/train.py:147  model = MODEL_REGISTRY[variant](num_classes=11)

Model construction happens immediately after the seed is set and before the
DataLoader exists (line 154), so the initial weights are a function of the
seed alone -- independent of shuffle order, dropout stream, or anything else.
A layer that receives zero gradient must therefore match these weights
bit-for-bit at epoch 640.

Run: python -I rc_weights.py
"""
import os
import sys
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402

from src.models import M4_FlattenFC  # noqa: E402

RUNS = [
    # (label, run_dir, seed, outcome)
    ("FLAT seed42 ambient (minverify)", "seed42_minverify_M4_seed42_ep640", 42, "COLLAPSE"),
    ("FLAT seed1500 ambient",           "seed1500_rc30_s1500",             1500, "COLLAPSE"),
    ("FLAT seed42 order=00 pinned",     "seed42_rcord000",                  42, "healthy"),
    ("FLAT seed42 order=07 pinned",     "seed42_rcord007",                  42, "healthy"),
    ("FLAT seed7  ambient",             "seed7_rc30_s7",                     7, "healthy"),
    ("FLAT seed1001 ambient",           "seed1001_rc30_s1001",            1001, "healthy"),
]

LAYERS = [
    ("features.0.weight", "conv1.W"),
    ("features.0.bias",   "conv1.b"),
    ("features.3.weight", "conv2.W"),
    ("features.3.bias",   "conv2.b"),
    ("classifier.1.weight", "fc1.W"),
    ("classifier.1.bias",   "fc1.b"),
    ("classifier.4.weight", "fc2.W"),
    ("classifier.4.bias",   "fc2.b"),
]

VARIANT_DIR = os.path.join("runs", "M4_FlattenFC")


def init_state(seed):
    """Reconstruct the initial state dict for a run of this seed."""
    torch.manual_seed(seed)
    m = M4_FlattenFC(num_classes=11)
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


def load_ckpts(run_dir):
    out = {}
    for fn in os.listdir(run_dir):
        if fn.startswith("ckpt_ep") and fn.endswith(".pt"):
            ep = int(fn[7:10])
            ck = torch.load(os.path.join(run_dir, fn), map_location="cpu",
                            weights_only=False)
            out[ep] = ck["model_state_dict"]
    return out


def rel(W, W0):
    """Relative Frobenius displacement, plus exact-equality and max element diff."""
    d = (W - W0)
    denom = W0.norm().item()
    return (d.norm().item() / denom if denom > 0 else float("inf"),
            bool(torch.equal(W, W0)),
            d.abs().max().item())


def main():
    print("=" * 100)
    print("PART 1  do all runs with the same seed start from the same weights?")
    print("=" * 100)
    # Two independent reconstructions must agree; and the two seed-42 runs
    # (one collapsing, one healthy) must have started identically. The second
    # is checked indirectly: both are compared against the same reconstructed W0.
    a = init_state(42)
    b = init_state(42)
    same = all(torch.equal(a[k], b[k]) for k in a)
    print(f"  two reconstructions of seed 42 identical: {same}")
    print(f"  fc1.W init: shape={tuple(a['classifier.1.weight'].shape)} "
          f"|W|_F={a['classifier.1.weight'].norm().item():.6f} "
          f"bias range=[{a['classifier.1.bias'].min().item():.6f}, "
          f"{a['classifier.1.bias'].max().item():.6f}]")

    print()
    print("=" * 100)
    print("PART 2  ||W(t) - W(0)|| / ||W(0)||   and   exact-equality with W(0)")
    print("=" * 100)

    summary = {}
    for label, run, seed, outcome in RUNS:
        run_dir = os.path.join(VARIANT_DIR, run)
        if not os.path.isdir(run_dir):
            print(f"  !! missing {run_dir}")
            continue
        W0 = init_state(seed)
        cks = load_ckpts(run_dir)
        if not cks:
            print(f"  !! no checkpoints in {run_dir}")
            continue
        eps = sorted(cks)

        print(f"\n--- {label}   [{outcome}]   checkpoints: {eps}")
        print(f"    {'layer':<9}", end="")
        for e in eps:
            print(f"{'ep'+str(e):>9}", end="")
        print()
        frozen = {}
        for key, short in LAYERS:
            print(f"    {short:<9}", end="")
            any_moved = False
            for e in eps:
                r, eq, mx = rel(cks[e][key], W0[key])
                if not eq:
                    any_moved = True
                mark = "=" if eq else ""
                # for bias layers a relative number is meaningless when |b0|~0:
                # print the absolute max element change instead
                if key.endswith(".bias"):
                    print(f"{mx:>9.2e}", end="")
                else:
                    print(f"{r:>8.4f}{mark}", end="")
            frozen[short] = not any_moved
            print()
        last = eps[-1]
        print(f"    at epoch {last}: " + ", ".join(
            f"{s}={'FROZEN at init' if f else 'moved'}"
            for s, f in frozen.items()
            if s in ("fc1.W", "fc1.b", "fc2.W", "fc2.b")))
        summary[run] = {"outcome": outcome, "seed": seed,
                        "frozen_at_640": {s: f for s, f in frozen.items()}}

    print()
    print("=" * 100)
    print("PART 3  is the fc1 ReLU output dead?  (fraction of the 64 units that")
    print("        are zero for every one of 4096 test vectors, at epoch 640)")
    print("=" * 100)

    torch.manual_seed(0)
    from src.data import load_data
    data = load_data(normalization="none")
    X = data["X_test_t"][:4096].float()

    for label, run, seed, outcome in RUNS:
        run_dir = os.path.join(VARIANT_DIR, run)
        cks = load_ckpts(run_dir)
        if not cks:
            continue
        sd = cks[max(cks)]
        with torch.no_grad():
            z = _pre(sd, X)                                   # [N, 1024]
            h = torch.nn.functional.relu(
                torch.nn.functional.linear(z, sd["classifier.1.weight"],
                                           sd["classifier.1.bias"]))
        dead = (h == 0).all(dim=0).sum().item()
        print(f"  {label:<34} [{outcome:<8}] dead fc1 units = "
              f"{dead:2d}/64 ({dead/64*100:5.1f}%)   mean|act| = "
              f"{h.abs().mean().item():.4e}")

    out = os.path.join("analysis_results", "rc_weights.json")
    os.makedirs("analysis_results", exist_ok=True)
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nwrote {out}")


def _pre(sd, X):
    """features -> flatten, i.e. the input to classifier.1."""
    f = torch.nn.functional
    x = X
    x = f.relu(f.conv2d(x, sd["features.0.weight"], sd["features.0.bias"],
                        padding=1))
    x = f.max_pool2d(x, 2, 2)
    x = f.relu(f.conv2d(x, sd["features.3.weight"], sd["features.3.bias"],
                        padding=1))
    x = f.max_pool2d(x, kernel_size=(1, 2))
    return x.flatten(1)


if __name__ == "__main__":
    main()
