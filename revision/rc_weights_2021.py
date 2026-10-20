"""
Displacement of the paper's own collapsed run (M4_FlattenFC seed 2021,
176 000/44 000 split) from its own initialisation.

The manuscript infers "the parameters of this layer never moved" from the dead
layer's pre-activation range lying inside the bias support. That inference is
not valid, and this measures the displacement directly. The initialisation is
reconstructed from the seed (see rc_weights.py for why that is exact), which
does not depend on the data partition.
"""
import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())

import torch  # noqa: E402
from src.models import M4_FlattenFC  # noqa: E402

RUNS = [
    ("seed2021_batch2a", 2021, "COLLAPSE (paper's run, 176k split)"),
    ("seed42_batch2a",     42, "healthy   (176k split)"),
    ("seed123_batch2a",   123, "healthy   (176k split)"),
]

KEYS = [("classifier.1.weight", "fc1.W"), ("classifier.1.bias", "fc1.b"),
        ("classifier.4.weight", "fc2.W"), ("classifier.4.bias", "fc2.b")]


def w0(seed):
    torch.manual_seed(seed)
    return M4_FlattenFC(num_classes=11).state_dict()


for run, seed, note in RUNS:
    d = os.path.join("runs", "M4_FlattenFC", run)
    if not os.path.isdir(d):
        print(f"missing {d}")
        continue
    W0 = w0(seed)
    cks = {}
    for fn in os.listdir(d):
        if fn.startswith("ckpt_ep") and fn.endswith(".pt"):
            cks[int(fn[7:10])] = torch.load(
                os.path.join(d, fn), map_location="cpu",
                weights_only=False)["model_state_dict"]
    eps = sorted(cks)
    print(f"\n--- {run}   {note}")
    print(f"    {'layer':<8}" + "".join(f"{'ep'+str(e):>12}" for e in eps))
    for key, short in KEYS:
        n0 = W0[key].norm().item()
        row = f"    {short:<8}"
        for e in eps:
            if short.endswith(".W"):
                row += f"{(cks[e][key]-W0[key]).norm().item()/n0:>12.5f}"
            else:
                row += f"{(cks[e][key]-W0[key]).abs().max().item():>12.3e}"
        print(row)
    # is it frozen across the whole recorded range?
    for key, short in KEYS:
        same = all(torch.equal(cks[eps[0]][key], cks[e][key]) for e in eps)
        moved = not torch.equal(cks[eps[0]][key], W0[key])
        print(f"    {short}: identical across ep{eps[0]}..ep{eps[-1]}: {same}"
              f"   differs from init: {moved}")
    b0 = W0["classifier.1.bias"]
    print(f"    initial fc1 bias support: +/-{1/32:.5f} "
          f"(max |b_init| = {b0.abs().max().item():.5f})")
