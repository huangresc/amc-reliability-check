# -*- coding: utf-8 -*-
"""BN ablation on the two collapsed FLAT cells.

Reviewer concern 3 / advisor comment 3.  The paper identifies the collapse
mechanism as the narrow bias support of fc1 (Linear 1024->64, fan-in 1024,
default bias support +-0.031): a perturbation pushes the fc1 pre-activations
entirely to one side, every ReLU unit dies, and the trunk gradient vanishes.
Batch normalisation renormalises pre-activations to zero mean / unit variance,
so it is the direct counter-intervention to that exact mechanism.  This script
tests the prediction: does BN prevent the collapse in the cell that collapsed?

Cells (both FLAT head, raw inputs, 640 epochs, seed-paired configs):
  seed 42,  official split (110k/110k)   -> collapsed, 9.09 %  (tab:partition)
  seed 2021, re-split (176k/44k)         -> collapsed, 9.02 %  (tab:arch,
                                             the paper's mechanism example)

Variants, all identical except the BN layers:
  control  : no BN  (must reproduce the collapse, or the ablation is void)
  fc1      : BatchNorm1d(64) after the fc1 linear, before ReLU -- the
             mechanism-targeted intervention
  full     : BatchNorm after conv1, conv2 and fc1 -- the standard recipe

RNG.  BatchNorm initialises weight=1, bias=0 deterministically and consumes no
RNG, so inserting it between the existing layers leaves the order of
RNG-consuming initialisations (two convs, then two linears) unchanged.  The
data shuffle and dropout masks are therefore identical across the three
variants for a given (seed, split).  set_seed(seed) runs BEFORE model creation
exactly as in src/train.py:146-148, so the control reproduces the historical
run's mini-batch order (the "draw") rather than re-rolling it.
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
OUT = "analysis_results/rc_bn_ablation.json"


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class FlattenFC_BN(nn.Module):
    """M4_FlattenFC with optional BatchNorm, 'none' being the original."""

    def __init__(self, num_classes=11, bn="none"):
        super().__init__()
        assert bn in ("none", "fc1", "full")
        feats = [nn.Conv2d(1, 16, 3, padding=1)]
        if bn == "full":
            feats.append(nn.BatchNorm2d(16))
        feats += [nn.ReLU(), nn.MaxPool2d(2, 2),
                  nn.Conv2d(16, 32, 3, padding=1)]
        if bn == "full":
            feats.append(nn.BatchNorm2d(32))
        feats += [nn.ReLU(), nn.MaxPool2d(kernel_size=(1, 2))]
        self.features = nn.Sequential(*feats)

        cls = [nn.Flatten(), nn.Linear(32 * 1 * 32, 64)]
        if bn in ("fc1", "full"):
            cls.append(nn.BatchNorm1d(64))
        cls += [nn.ReLU(), nn.Dropout(0.5), nn.Linear(64, num_classes)]
        self.classifier = nn.Sequential(*cls)

    def forward(self, x):
        return self.classifier(self.features(x))


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
    """The split the current src.data.load_data produces (110k/110k)."""
    d = load_data(normalization="none", seed=42)
    return d["X_train_t"], d["y_train_t"], d["X_test_t"], d["y_test_t"]


def resplit():
    """The historical 176k/44k unstratified re-split (legacy setup_project)."""
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


def train(bn, Xtr, ytr, Xte, yte, seed):
    set_seed(seed)  # BEFORE model init, matching src/train.py:146
    model = FlattenFC_BN(num_classes=11, bn=bn).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0.0)
    loader = make_loader(Xtr, ytr, batch_size=256, num_workers=0,
                         shuffle_seed=None)  # ambient order, as in train.py
    _ = next(iter(loader))  # replicate train.py's GPU assert draw
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
        for bn in ("none", "fc1", "full"):
            key = f"{name}_{bn}"
            if key in results:
                print(f"  {bn:>8}: cached, skipping", flush=True)
                continue
            accs = train(bn, Xtr, ytr, Xte, yte, seed)
            results[key] = accs
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            line = "  ".join(f"ep{k}={v:.2f}" for k, v in accs.items()
                             if k in ("20", "40", "80", "320", "640"))
            print(f"  {bn:>8}: {line}", flush=True)
        del Xtr, ytr, Xte, yte
        gc.collect()
        torch.cuda.empty_cache()
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
