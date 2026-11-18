# -*- coding: utf-8 -*-
"""Self-contained 30-seed GAP/FLAT raw-input contrast on the RML2016.10a re-split.

Run:
    python cloud_more_seeds.py           # all 30 seeds (GAP + FLAT each)
    python cloud_more_seeds.py 2021      # single seed (smoke test)

Reproduces the paper's re-split (permutation seed 42, 176000/44000) and the
paper's M2_GAPHead / M4_FlattenFC training loop (Adam lr 1e-3, no BN, raw
inputs, 640 epochs, 9 checkpoints).  set_seed runs BEFORE model creation, and
the RNG-consuming layer order (conv1, conv2, fc1, fc2) matches src/models.py, so
a given seed reproduces the laptop run up to GPU floating-point differences.
Results are cached incrementally in results_more_seeds.json.
"""
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
from torch.utils.data import DataLoader, TensorDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PKL = "RML2016.10a_dict.pkl"
MILESTONES = [20, 40, 80, 120, 160, 240, 320, 480, 640]
OUT = "results_more_seeds.json"
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021,
         7, 13, 55, 101, 256, 512, 777, 1001, 1500, 2048,
         3141, 4096, 5150, 6666, 7777, 8888, 9999, 12345, 24680, 31337]


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class GAP(nn.Module):
    def __init__(self, nc=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32, 64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, nc))

    def forward(self, x):
        return self.classifier(nn.AdaptiveAvgPool2d(1)(self.features(x)))


class FLAT(nn.Module):
    def __init__(self, nc=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d((1, 2)))
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(32 * 1 * 32, 64), nn.ReLU(),
            nn.Dropout(0.5), nn.Linear(64, nc))

    def forward(self, x):
        return self.classifier(self.features(x))


def resplit(perm_seed=42):
    """The paper's 176000/44000 unstratified 80/20 re-split."""
    print(f"loading {PKL} ...", flush=True)
    with open(PKL, "rb") as f:
        raw = pickle.load(f, encoding="latin1")
    print(f"pkl loaded, {len(raw)} cells", flush=True)
    mods = sorted({k[0] for k in raw})
    m2i = {m: i for i, m in enumerate(mods)}
    X, y = [], []
    for (mod, snr), s in raw.items():
        X.append(s)
        y.append(np.full(s.shape[0], m2i[mod]))
    Xcat = np.concatenate(X).astype("float32")
    ycat = np.concatenate(y).astype("int64")
    N = len(Xcat)
    np.random.seed(perm_seed)
    perm = np.random.permutation(N)
    cut = int(N * 0.8)
    tr, te = perm[:cut], perm[cut:]
    Xtr = torch.from_numpy(Xcat[tr]).unsqueeze(1)
    ytr = torch.from_numpy(ycat[tr])
    Xte = torch.from_numpy(Xcat[te]).unsqueeze(1)
    yte = torch.from_numpy(ycat[te])
    prev = 100.0 * float((ycat[te] == 9).mean())
    assert abs(prev - 9.0205) < 0.0005, f"re-split drifted: {prev:.4f}"
    print(f"split done: n_train={len(Xtr)} n_test={len(Xte)} "
          f"label9={prev:.4f}%", flush=True)
    return Xtr, ytr, Xte, yte


def evaluate(model, X, y, bs=2048):
    model.eval()
    c = t = 0
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(DEVICE)
            yb = y[i:i + bs].to(DEVICE)
            c += (torch.argmax(model(xb), dim=1) == yb).sum().item()
            t += yb.shape[0]
    return c / t * 100


def train(model_cls, Xtr, ytr, Xte, yte, seed):
    set_seed(seed)  # BEFORE model creation, matching src/train.py
    model = model_cls().to(DEVICE)
    crit = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                     weight_decay=0.0)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=256, shuffle=True,
                        num_workers=8)
    _ = next(iter(loader))
    accs = {}
    for ep in range(1, 641):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            opt.zero_grad()
        if ep in MILESTONES:
            accs[str(ep)] = evaluate(model, Xte, yte)
        if ep % 100 == 0:
            last = max((m for m in MILESTONES if m <= ep), default=20)
            print(f"    ...epoch {ep}, acc@{last}={accs[str(last)]:.2f}%",
                  flush=True)
    return accs


def main():
    Xtr, ytr, Xte, yte = resplit(42)
    results = {}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            results = json.load(f)

    only = sys.argv[1] if len(sys.argv) > 1 else None
    for seed in SEEDS:
        if only and str(seed) != only:
            continue
        for head, cls in (("GAP", GAP), ("FLAT", FLAT)):
            key = f"seed{seed}_{head}"
            if key in results:
                print(f"  {key}: cached, skip", flush=True)
                continue
            print(f"  training {key} ...", flush=True)
            accs = train(cls, Xtr, ytr, Xte, yte, seed)
            results[key] = accs
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            line = "  ".join(f"ep{k}={v:.2f}" for k, v in accs.items()
                             if k in ("20", "40", "80", "320", "640"))
            print(f"  {key}: {line}", flush=True)
    print(f"wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
