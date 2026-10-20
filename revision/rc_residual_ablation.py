# -*- coding: utf-8 -*-
"""Residual-connection ablation on the two collapsed FLAT cells.

Advisor comment (MAJOR 1, ecological validity): do modern-architecture features
make the collapse go away?  The BN ablation already showed batch normalisation
does.  This script tests the OTHER canonical modern feature --- a residual
(skip) connection --- on the same two collapsed cells, so the two safeguards can
be read side by side.

Cells (FLAT head, raw inputs, 640 epochs):
  seed 42,  official split -> collapsed 9.09 %
  seed 2021, re-split      -> collapsed 9.02 %

Variants, all identical except the skip connection:
  none  : the original M4_FlattenFC (control; already recorded in
          analysis_results/rc_bn_ablation.json under *_none)
  fc1   : residual around the fc1 linear:  h = ReLU(fc1(x)) + W_shortcut x
  full  : residual around conv2 AND fc1 (the standard recipe)

RNG.  Unlike BatchNorm (deterministic init, no RNG), a projection shortcut is a
nn.Linear / nn.Conv2d and consumes RNG at construction, which would shift the
mini-batch order relative to the control and re-roll the "draw".  We neutralise
that by saving torch.get_rng_state() before each shortcut and restoring it
immediately after, so the NET RNG consumption is exactly the control's (conv1,
conv2, fc1, fc2 in order).  The data shuffle and dropout masks are therefore
identical across variants, and the skip connection is the only difference.
"""
import gc
import io
import json
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
OUT = "analysis_results/rc_residual_ablation.json"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _restored(fn):
    """Build fn() while leaving the global RNG stream untouched."""
    state = torch.get_rng_state()
    obj = fn()
    torch.set_rng_state(state)
    return obj


class FlattenFC_Residual(nn.Module):
    """M4_FlattenFC with an optional residual, 'none' being the original."""

    def __init__(self, num_classes=11, res="none"):
        super().__init__()
        assert res in ("none", "fc1", "full")
        self.res = res
        # conv trunk
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2))
        if res == "full":
            # projection + downsample to match conv2's output shape (32,1,32)
            self.conv2_shortcut = _restored(
                lambda: nn.Conv2d(16, 32, 1, stride=(1, 2)))
        # classifier
        self.fc1 = nn.Linear(32 * 1 * 32, 64)
        self.relu_fc1 = nn.ReLU()
        if res in ("fc1", "full"):
            self.fc1_shortcut = _restored(lambda: nn.Linear(32 * 1 * 32, 64))
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        c1 = self.pool1(self.relu1(self.conv1(x)))
        c2 = self.pool2(self.relu2(self.conv2(c1)))
        if self.res == "full":
            c2 = c2 + self.conv2_shortcut(c1)
        f = torch.flatten(c2, 1)
        h = self.relu_fc1(self.fc1(f))
        if self.res in ("fc1", "full"):
            h = h + self.fc1_shortcut(f)
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


def train(res, Xtr, ytr, Xte, yte, seed):
    set_seed(seed)  # BEFORE model creation, matching src/train.py:146
    model = FlattenFC_Residual(num_classes=11, res=res).to(DEVICE)
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
        for res in ("fc1", "full"):
            key = f"{name}_{res}"
            if key in results:
                print(f"  {res:>8}: cached, skipping", flush=True)
                continue
            accs = train(res, Xtr, ytr, Xte, yte, seed)
            results[key] = accs
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            line = "  ".join(f"ep{k}={v:.2f}" for k, v in accs.items()
                             if k in ("20", "40", "80", "320", "640"))
            print(f"  {res:>8}: {line}", flush=True)
        del Xtr, ytr, Xte, yte
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
