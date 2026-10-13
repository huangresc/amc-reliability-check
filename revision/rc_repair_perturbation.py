# -*- coding: utf-8 -*-
"""
Does the "repair" claim survive a change of perturbation draw?

The manuscript (abstract, and tab:repair via results/escape_repair_sweep.json)
states that re-initialising the dead fc1 layer and training 40 more epochs at
1e-3 recovers the collapsed run from 9.05 % to 40.79 % -- a 31.7 pp recovery.
That number is a single cell: one run, one perturbation draw.

Re-running the identical protocol on the *other* collapsed run (seed 42,
official split) recovers nothing (analysis_results/exp4_repair.json, 9.0909 %).
The two protocols differ in WHICH numbers the re-initialisation draws: both
re-init with kaiming_uniform(a=sqrt(5)) + U(-b,b), but the global RNG sits at a
different position, so the perturbation tensor is different.  tab:repair is
therefore n=1 per cell over perturbation draws, and its generality is untested.

This harness holds the perturbation draw COMMON across the two runs:

    set_seed(p) -> M4_FlattenFC(11) -> load_state_dict(ckpt) -> reset fc1

`load_state_dict` consumes no RNG and both runs use the same model class, so at
a given p the re-initialised fc1 is bit-identical in both runs.  A SHA-1 of the
reset weight is recorded to make that checkable rather than asserted.

Validation cell: (re-split, p=2021, reinit_fc1) re-executes P3's exact RNG path
and must reproduce 40.79090909090909 %.  Also (re-split, p=2021,
reinit_fc1_and_fc2) must reproduce 35.04772727272727 %.

Data partitions (both rebuilt from source, see rc_verify_split.py):
  re-split  setup_project.py:131-136 -- np.random.seed(42), permutation of all
            220 000, 80/20 cut.  Unstratified: label 9 has prevalence 9.0205 %,
            which is exactly the collapsed run's accuracy.  n_test 44 000.
  official  current src/data.py -- per-(mod,snr) half/half, fully stratified
            (10 000 per class), n_test 110 000.

Writes analysis_results/rc_repair_perturbation.json.
"""
import hashlib
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_data
from src.models import M4_FlattenFC

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_LR = 1e-3
EPOCHS = 40
BS = 256
PKL = "rml2016_data/RML2016.10a_dict.pkl"
OUT = os.path.join("analysis_results", "rc_repair_perturbation.json")

# The first five draws were run on 2026-10-13 and are already in OUT; the run
# resumes from disk, so extending this list only pays for the new draws.  The
# five are kept in front so the previously validated cells are the first thing
# the harness re-checks.
PERTURB_SEEDS = [2021, 42, 7, 123, 555,
                 1, 3, 11, 17, 29, 77, 256, 777, 1001, 1500,
                 2048, 4096, 6666, 12345, 31337]
VALIDATION_SEED = 2021          # reproduces the recorded P3 measurement

RUNS = [
    ("resplit", "runs/M4_FlattenFC/seed2021_batch2a/ckpt_ep640.pt"),
    ("official", "runs/M4_FlattenFC/seed42_minverify_M4_seed42_ep640/ckpt_ep640.pt"),
]

VARIANTS = {"reinit_fc1": ["classifier.1"],
            "reinit_fc1_and_fc2": ["classifier.1", "classifier.4"]}


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_resplit():
    """Rebuild setup_project.py:131-136 exactly."""
    with open(PKL, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in data})
    m2i = {m: i for i, m in enumerate(mods)}
    Xs, ys = [], []
    for (mod, snr), s in data.items():
        Xs.append(s)
        ys.append(np.full(s.shape[0], m2i[mod]))
    X = np.concatenate(Xs).astype("float32")
    y = np.concatenate(ys).astype("int64")
    np.random.seed(42)
    idx = np.random.permutation(len(X))
    cut = int(len(X) * 0.8)
    return (torch.from_numpy(X[idx[:cut]]).unsqueeze(1),
            torch.from_numpy(y[idx[:cut]]),
            torch.from_numpy(X[idx[cut:]]).unsqueeze(1),
            torch.from_numpy(y[idx[cut:]]))


def load_official():
    d = load_data(normalization="none")
    return d["X_train_t"], d["y_train_t"], d["X_test_t"], d["y_test_t"]


@torch.no_grad()
def evaluate(model, X, y, batch=2048):
    """Test accuracy and dead fraction of the fc1 ReLU, over the whole split."""
    model.eval()
    dead = [0]
    total = [0]

    def hook(m, i, o):
        pre = i[0]
        dead[0] += int((pre <= 0).sum())
        total[0] += pre.numel()

    h = model.classifier[2].register_forward_hook(hook)
    correct = 0
    n = 0
    for i in range(0, len(X), batch):
        xb = X[i:i + batch].to(DEVICE)
        yb = y[i:i + batch].to(DEVICE)
        correct += int((model(xb).argmax(1) == yb).sum())
        n += yb.shape[0]
    h.remove()
    return 100.0 * correct / n, dead[0] / total[0]


def get_module(model, dotted):
    mod = model
    for part in dotted.split("."):
        mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
    return mod


def weight_sha1(model):
    w = model.classifier[1].weight.detach().cpu().numpy().tobytes()
    return hashlib.sha1(w).hexdigest()[:16]


def main():
    t_start = time.time()
    print(f"device {DEVICE}")
    datasets = {}
    for name, fn in (("resplit", load_resplit), ("official", load_official)):
        Xtr, ytr, Xte, yte = fn()
        datasets[name] = (Xtr, ytr, Xte, yte)
        print(f"  {name:9s} n_train {len(Xtr):>6d}  n_test {len(Xte):>6d}")
    sys.stdout.flush()

    criterion = nn.CrossEntropyLoss()
    results = {"device": str(DEVICE), "base_lr": BASE_LR, "epochs": EPOCHS}
    if os.path.exists(OUT):
        with open(OUT, encoding="utf-8") as f:
            results = json.load(f)
        results["device"] = str(DEVICE)
        results["base_lr"] = BASE_LR
        results["epochs"] = EPOCHS
        cached = sum(len(results.get(r, {}).get("cells", {})) for r, _ in RUNS)
        print(f"  resuming from {OUT}: {cached} cells already on disk, "
              f"{len(PERTURB_SEEDS)} draws requested per run")

    for run, ckpt_path in RUNS:
        Xtr, ytr, Xte, yte = datasets[run]
        ck = torch.load(ckpt_path, map_location=DEVICE)
        results.setdefault(run, {})
        results[run]["ckpt"] = ckpt_path
        results[run].setdefault("cells", {})
        print("\n" + "=" * 78)
        print(f"{run}  <-  {ckpt_path}")
        print("=" * 78)
        print(f"  {'perturb':>7} {'variant':<20} {'ep0 acc':>8} {'ep0 dead':>9} "
              f"{'ep20 acc':>9} {'ep40 acc':>9} {'ep40 dead':>10} {'wsha1':>17} {'s':>5}")
        print("  " + "-" * 100)

        for p in PERTURB_SEEDS:
            # reinit_fc1 is the manuscript's claim, swept over draws.  The
            # two-layer variant is only the second P3 validation cell, so it is
            # run once (p=2021) rather than doubling the cost of every draw.
            todo = ["reinit_fc1"]
            if p == VALIDATION_SEED and run == "resplit":
                todo.append("reinit_fc1_and_fc2")
            for vname in todo:
                layers = VARIANTS[vname]
                if f"p{p}_{vname}" in results[run]["cells"]:
                    print(f"  {p:>7} {vname:<20} cached, skipped")
                    sys.stdout.flush()
                    continue
                # P3's exact RNG path: seed -> construct -> load -> reset
                set_seed(p)
                model = M4_FlattenFC(num_classes=11).to(DEVICE)
                model.load_state_dict(ck["model_state_dict"])
                for lname in layers:
                    layer = get_module(model, lname)
                    if isinstance(layer, nn.Linear):
                        layer.reset_parameters()
                wsha = weight_sha1(model)

                acc0, dead0 = evaluate(model, Xte, yte)
                opt = optim.Adam(model.parameters(), lr=BASE_LR)
                loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS,
                                    shuffle=True, num_workers=0)
                curve = [{"epoch": 0, "acc": acc0, "dead_fc1": dead0}]
                t0 = time.time()
                for ep in range(EPOCHS):
                    if ep + 1 == 20:
                        a, d = evaluate(model, Xte, yte)
                        curve.append({"epoch": 20, "acc": a, "dead_fc1": d})
                    model.train()
                    for x, y in loader:
                        x = x.to(DEVICE, non_blocking=True)
                        y = y.to(DEVICE, non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        loss = criterion(model(x), y)
                        loss.backward()
                        opt.step()
                dt = time.time() - t0
                acc40, dead40 = evaluate(model, Xte, yte)
                curve.append({"epoch": 40, "acc": acc40, "dead_fc1": dead40})

                mid = next(c["acc"] for c in curve if c["epoch"] == 20)
                results[run]["cells"][f"p{p}_{vname}"] = {
                    "perturb_seed": p, "variant": vname, "layers": layers,
                    "weight_sha1_16": wsha,
                    "epoch0": {"acc": acc0, "dead_fc1": dead0},
                    "epoch20": {"acc": mid},
                    "epoch40": {"acc": acc40, "dead_fc1": dead40},
                    "curve": curve, "seconds": dt,
                }
                print(f"  {p:>7} {vname:<20} {acc0:>8.2f} {dead0:>9.4f} "
                      f"{mid:>9.2f} {acc40:>9.2f} {dead40:>10.4f} {wsha:>17} {dt:>5.0f}")
                sys.stdout.flush()

    # ---- validation against the recorded P3 numbers ------------------------
    print("\n" + "=" * 78)
    print("VALIDATION against results/escape_repair_sweep.json P3_repair")
    print("=" * 78)
    exp = {"p2021_reinit_fc1": 40.79090909090909,
           "p2021_reinit_fc1_and_fc2": 35.04772727272727}
    val = {}
    for key, want in exp.items():
        got = results["resplit"]["cells"][key]["epoch40"]["acc"]
        ok = abs(got - want) < 1e-6
        val[key] = {"expected": want, "got": got, "reproduced": ok}
        print(f"  {key:<28} expected {want:.4f}  got {got:.4f}  "
              f"{'OK' if ok else 'MISMATCH'}")

    # ---- the question: does the recovery replicate across draws? -----------
    print("\n" + "=" * 78)
    print("RECOVERY across perturbation draws (reinit_fc1, epoch 40 minus epoch 0)")
    print("=" * 78)
    for run in ("resplit", "official"):
        cells = results[run]["cells"]
        rec = [(p, cells[f"p{p}_reinit_fc1"]["epoch40"]["acc"]
                - cells[f"p{p}_reinit_fc1"]["epoch0"]["acc"]) for p in PERTURB_SEEDS]
        accs = [cells[f"p{p}_reinit_fc1"]["epoch40"]["acc"] for p in PERTURB_SEEDS]
        print(f"  {run:9s} final acc {['%.2f' % a for a in accs]}")
        print(f"  {'':9s} recovery  {['%+.2f' % r for _, r in rec]}")
    results["validation"] = val
    results["runtime_seconds"] = time.time() - t_start

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {OUT}   total {results['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
