# -*- coding: utf-8 -*-
"""Equalised-initialisation ablation on the two collapsed FLAT cells.

Reviewer 2, point 3.  GAP and FLAT differ not only in the pooling decision but
in the first-FC fan-in (32 vs 1024), and therefore in the default bias support
(+-0.1768 vs +-0.03125), a factor of 5.7.  The causal attribution "the head
comparison isolates the pooling decision" is therefore not clean: the measured
difference could be an initialisation-scale difference, not a pooling
difference.  This script tests that directly by re-initialising fc1 of the FLAT
head at the GAP convention (fan-in 32) and re-running the collapsed cells.  If
the collapse survives, pooling is causal; if it does not, the collapse is an
initialisation-scale effect and the contrast must be reframed as
head-plus-initialisation-scale.

Cells (both FLAT head, raw inputs, 640 epochs, seed-paired):
  seed 42,  official split (110k/110k)   -> collapsed, 9.09 %  (tab:partition)
  seed 2021, re-split (176k/44k)         -> collapsed, 9.02 %  (tab:arch)

Variants, identical except fc1 initialisation scale:
  none  : original M4_FlattenFC (control; must reproduce the collapse)
  fc1   : fc1 weight AND bias re-drawn at bound 1/sqrt(32), the GAP fc1 scale

Equalisation is exact.  PyTorch initialises Linear(1024->64) with
kaiming_uniform_(a=sqrt(5)) weight ~ U(-1/sqrt(1024), 1/sqrt(1024)) and bias
~ U(-1/sqrt(1024), 1/sqrt(1024)).  Linear(32->64) gives the same distributions
with fan-in 32, i.e. bound 1/sqrt(32).  kaiming_uniform_(a=sqrt(5)) reduces to
uniform(-1/sqrt(fan_in), 1/sqrt(fan_in)) exactly, so nn.init.uniform_ with bound
1/sqrt(32) reproduces the GAP fc1 distribution bit-for-bit.

RNG.  Re-drawing fc1 consumes RNG, which would shift the mini-batch order and
re-roll the "draw".  The re-draw is wrapped in _restored() so the NET RNG
consumption equals the control's (conv1, conv2, fc1, fc2 in order), leaving the
data shuffle and dropout masks identical across variants.  set_seed(seed) runs
BEFORE model creation exactly as in src/train.py:146-148.
"""
import gc
import io
import json
import math
import os
import pickle
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from src.data import load_data, make_loader  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PKL = "rml2016_data/RML2016.10a_dict.pkl"
MILESTONES = [20, 40, 80, 120, 160, 240, 320, 480, 640]
OUT = "analysis_results/rc_equalinit_ablation.json"
EQ_BOUND = 1.0 / math.sqrt(32)  # 0.17678..., the GAP fc1 (Linear 32->64) bound


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restored(fn):
    """Run fn() while leaving the global RNG stream untouched."""
    state = torch.get_rng_state()
    fn()
    torch.set_rng_state(state)


class FlattenFC_EqualInit(nn.Module):
    """M4_FlattenFC with optional fc1 initialisation-scale equalisation."""

    def __init__(self, num_classes=11, eq="none"):
        super().__init__()
        assert eq in ("none", "fc1")
        self.eq = eq
        # conv trunk (identical to M4_FlattenFC / FlattenFC_Residual "none")
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        # classifier
        self.fc1 = nn.Linear(32 * 1 * 32, 64)
        if eq == "fc1":
            def _reinit():
                nn.init.uniform_(self.fc1.weight, -EQ_BOUND, EQ_BOUND)
                nn.init.uniform_(self.fc1.bias, -EQ_BOUND, EQ_BOUND)
            _restored(_reinit)
        self.relu_fc1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        c1 = self.pool1(self.relu1(self.conv1(x)))
        c2 = self.pool2(self.relu2(self.conv2(c1)))
        f = torch.flatten(c2, 1)
        h = self.relu_fc1(self.fc1(f))
        return self.fc2(self.dropout(h))


def evaluate(model, X, y, bs=2048):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(DEVICE)
            yb = y[i:i + bs].to(DEVICE)
            correct += (torch.argmax(model(xb), dim=1) == yb).sum().item()
            total += yb.shape[0]
    return correct / total * 100


def official_split():
    d = load_data(normalization="none", seed=42)
    return d["X_train_t"], d["y_train_t"], d["X_test_t"], d["y_test_t"]


def resplit():
    with open(PKL, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in raw})
    m2i = {m: i for i, m in enumerate(mods)}
    X, y = [], []
    for (mod, snr), s in raw.items():
        X.append(s)
        y.append(np.full(s.shape[0], m2i[mod]))
    Xcat = np.concatenate(X).astype("float32")
    ycat = np.concatenate(y).astype("int64")
    N = len(Xcat)
    np.random.seed(42)
    perm = np.random.permutation(N)
    cut = int(N * 0.8)
    tr, te = perm[:cut], perm[cut:]
    Xtr = torch.from_numpy(Xcat[tr]).unsqueeze(1)
    ytr = torch.from_numpy(ycat[tr])
    Xte = torch.from_numpy(Xcat[te]).unsqueeze(1)
    yte = torch.from_numpy(ycat[te])
    prev = 100.0 * float((ycat[te] == 9).mean())
    assert abs(prev - 9.0205) < 0.0005, f"re-split drifted: {prev:.4f}"
    return Xtr, ytr, Xte, yte


def train(eq, Xtr, ytr, Xte, yte, seed):
    set_seed(seed)  # BEFORE model init, matching src/train.py:146
    model = FlattenFC_EqualInit(num_classes=11, eq=eq).to(DEVICE)
    wb = float(model.fc1.weight.abs().max())
    bb = float(model.fc1.bias.abs().max())
    print(f"    fc1 init: max|W|={wb:.5f}  max|b|={bb:.5f}", flush=True)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0.0)
    loader = make_loader(Xtr, ytr, batch_size=256, num_workers=0,
                         shuffle_seed=None)
    _ = next(iter(loader))
    accs = {}
    for ep in range(1, 641):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if ep in MILESTONES:
            accs[str(ep)] = evaluate(model, Xte, yte)
    return accs


def main():
    os.makedirs("analysis_results", exist_ok=True)
    results = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            results = json.load(f)

    cells = [
        ("official_seed42", official_split, 42),
        ("resplit_seed2021", resplit, 2021),
    ]
    only = sys.argv[1] if len(sys.argv) > 1 else None
    if only:
        cells = [(n, l, s) for n, l, s in cells if only in n]

    for name, loader, seed in cells:
        Xtr, ytr, Xte, yte = loader()
        print(f"\n{'=' * 70}\n{name}  (n_train={len(Xtr)})\n{'=' * 70}",
              flush=True)
        for eq in ("none", "fc1"):
            key = f"{name}_{eq}"
            if key in results:
                print(f"  {eq:>8}: cached, skipping", flush=True)
                continue
            accs = train(eq, Xtr, ytr, Xte, yte, seed)
            results[key] = accs
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            line = "  ".join(f"ep{k}={v:.2f}" for k, v in accs.items()
                             if k in ("20", "40", "80", "320", "640"))
            print(f"  {eq:>8}: {line}", flush=True)
        del Xtr, ytr, Xte, yte
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
