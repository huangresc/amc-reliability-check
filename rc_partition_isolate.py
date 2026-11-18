# -*- coding: utf-8 -*-
"""Partition isolation arm: a same-size, different-draw re-split.

Reviewer 3, point 5.  The paper compares a 176k/44k non-stratified re-split
(permutation seed 42) against the 110k/110k stratified official split.  The two
differ in BOTH training-set size and splitting method, so the partition effect
is not isolated from the size/method effect.  This script adds a third arm: the
SAME 176k/44k size and the SAME unstratified 80/20 method, but a DIFFERENT
permutation draw (seed 43).  Comparing re-split@42 against re-split@43 holds
size and method fixed and isolates the draw; comparing either against the
official split then isolates size/method.

The existing re-split (paper) is:
    np.random.seed(42); perm = np.random.permutation(220000); cut at 176000.
The new split uses permutation seed 43 and records its own test-set label-9
prevalence (the chance level of the constant predictor), which need not equal
the 9.0205 % of seed 42.

Runs: 10 main seeds x (GAP, FLAT), raw inputs, 640 epochs, the paper's nine
checkpoints.  Training loop, model classes and RNG order are identical to
src.train.train_one_variant so the only variable is the data partition.
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

from src.data import make_loader  # noqa: E402
from src.models import M2_GAPHead, M4_FlattenFC  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PKL = "rml2016_data/RML2016.10a_dict.pkl"
MILESTONES = [20, 40, 80, 120, 160, 240, 320, 480, 640]
OUT = "analysis_results/rc_partition_isolate.json"
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
PERM_SEED = 43  # the new split draw; the paper's re-split uses 42

MODELS = {
    "GAP": M2_GAPHead,
    "FLAT": M4_FlattenFC,
}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resplit(perm_seed=PERM_SEED):
    """176k/44k unstratified 80/20 re-split with a chosen permutation seed."""
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
    np.random.seed(perm_seed)
    perm = np.random.permutation(N)
    cut = int(N * 0.8)
    tr, te = perm[:cut], perm[cut:]
    Xtr = torch.from_numpy(Xcat[tr]).unsqueeze(1)
    ytr = torch.from_numpy(ycat[tr])
    Xte = torch.from_numpy(Xcat[te]).unsqueeze(1)
    yte = torch.from_numpy(ycat[te])
    prev9 = 100.0 * float((ycat[te] == 9).mean())
    return Xtr, ytr, Xte, yte, prev9


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


def train(model_cls, Xtr, ytr, Xte, yte, seed):
    set_seed(seed)  # BEFORE model init, matching src/train.py:146
    model = model_cls(num_classes=11).to(DEVICE)
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

    Xtr, ytr, Xte, yte, prev9 = resplit()
    meta = {
        "perm_seed": PERM_SEED,
        "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)),
        "test_label9_prevalence_pct": prev9,
    }
    results["_meta"] = meta
    print(f"\n{'=' * 70}")
    print(f"re-split perm_seed={PERM_SEED}  n_train={len(Xtr)} "
          f"n_test={len(Xte)}  label9_prevalence={prev9:.4f}%")
    print(f"{'=' * 70}", flush=True)

    only = sys.argv[1] if len(sys.argv) > 1 else None
    for seed in SEEDS:
        if only and str(seed) not in only:
            continue
        for head, model_cls in MODELS.items():
            key = f"seed{seed}_{head}"
            if key in results:
                print(f"  {key:>20}: cached, skipping", flush=True)
                continue
            accs = train(model_cls, Xtr, ytr, Xte, yte, seed)
            results[key] = accs
            with open(OUT, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            line = "  ".join(f"ep{k}={v:.2f}" for k, v in accs.items()
                             if k in ("20", "40", "80", "320", "640"))
            print(f"  {key:>20}: {line}", flush=True)
            gc.collect()
            torch.cuda.empty_cache()
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
