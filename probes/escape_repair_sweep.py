# -*- coding: utf-8 -*-
"""
Escape / repair sweep for the collapsed M4 run (seed 2021, no normalization).

Answers three questions the current draft cannot:
  P1  Is the gradient into the dead layer exactly zero?      (direct evidence)
  P2  Can any learning rate revive it?                        (LR sweep 1x..1000x)
  P3  Does re-initialising the dead layer repair the run?     (falsifiable repair)

Also measures dead fractions per ReLU layer over the FULL test set, replacing the
single-256-sample-batch estimate used in the original probe.

Writes results/escape_repair_sweep.json and prints a compact summary.
"""
import io
import json
import os
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.data import load_data
from src.models import M4_FlattenFC, M2_GAPHead

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_LR = 1e-3
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "results", "escape_repair_sweep.json")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def relu_layers():
    """The three ReLU modules, in forward order."""
    return [("features.1", 1), ("features.4", 4), ("classifier.2", 2)]


@torch.no_grad()
def layer_stats(model, X, y, batch=2048):
    """Dead fraction and pre-activation range per ReLU layer, over a whole split."""
    model.eval()
    feats = ["features.1", "features.4", "classifier.2"]
    acc = {k: {"dead": 0, "total": 0, "pre_min": float("inf"), "pre_max": -float("inf")}
           for k in feats}
    hooks = []

    def mk(name):
        def h(m, i, o):
            pre = i[0]
            acc[name]["dead"] += int((pre <= 0).sum().item())
            acc[name]["total"] += pre.numel()
            acc[name]["pre_min"] = min(acc[name]["pre_min"], float(pre.min()))
            acc[name]["pre_max"] = max(acc[name]["pre_max"], float(pre.max()))
        return h

    for name, idx in relu_layers():
        mod = model
        for part in name.split("."):
            mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
        hooks.append(mod.register_forward_hook(mk(name)))

    logit_std_sum, n_batch, correct, total = 0.0, 0, 0, 0
    for i in range(0, len(X), batch):
        xb, yb = X[i:i + batch].to(DEVICE), y[i:i + batch].to(DEVICE)
        logits = model(xb)
        logit_std_sum += float(logits.std(dim=1).mean())
        n_batch += 1
        correct += int((logits.argmax(1) == yb).sum())
        total += yb.shape[0]

    for h in hooks:
        h.remove()

    out = {}
    for k, v in acc.items():
        out[k] = {"dead_frac": v["dead"] / v["total"],
                  "dead": v["dead"], "total": v["total"],
                  "alive": v["total"] - v["dead"],
                  "preact_min": v["pre_min"], "preact_max": v["pre_max"]}
    out["logit_std"] = logit_std_sum / n_batch
    out["test_acc"] = 100.0 * correct / total
    out["n_test"] = total
    return out


def grad_probe(model, loader, criterion):
    """One forward+backward; return grad-norm per parameter group, before any step."""
    model.train()
    xb, yb = next(iter(loader))
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    model.zero_grad(set_to_none=True)
    loss = criterion(model(xb), yb)
    loss.backward()

    groups = {}
    for name, p in model.named_parameters():
        g = 0.0 if p.grad is None else float(p.grad.norm())
        key = ("trunk" if name.startswith("features") else "classifier")
        groups.setdefault(key, 0.0)
        groups[key] += g * g
    # per-tensor detail for the dead layer
    detail = {n: (0.0 if p.grad is None else float(p.grad.norm()))
              for n, p in model.named_parameters() if n.startswith("classifier.")}
    groups = {k: v ** 0.5 for k, v in groups.items()}
    return loss.item(), groups, detail


def build_loader(X, y, seed, bs=256):
    set_seed(seed)
    return DataLoader(TensorDataset(X, y), batch_size=bs, shuffle=True, num_workers=0)


def train_epochs(model, loader, criterion, opt, n_epochs, tag, log):
    t0 = time.time()
    for ep in range(n_epochs):
        model.train()
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            opt.step()
        log.append({"tag": tag, "epoch": ep + 1, "loss": loss.item()})
    return time.time() - t0


def main():
    results = {"device": str(DEVICE), "base_lr": BASE_LR}
    data = load_data(normalization="none")          # same split as every run (seed 42)
    Xtr, ytr = data["X_train_t"], data["y_train_t"]
    Xte, yte = data["X_test_t"], data["y_test_t"]
    print("data: train %d  test %d  (split seed 42, matching the original runs)"
          % (len(Xtr), len(Xte)))

    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------------------- P1
    print("\n" + "=" * 72)
    print("P1  gradient probe at the collapsed checkpoint (M4 seed2021, ep640)")
    print("=" * 72)
    set_seed(2021)
    m4 = M4_FlattenFC(num_classes=11).to(DEVICE)
    ck = torch.load("runs/M4_FlattenFC/seed2021_batch2a/ckpt_ep640.pt",
                    map_location=DEVICE)
    m4.load_state_dict(ck["model_state_dict"])
    loader4 = build_loader(Xtr, ytr, 2021)

    st = layer_stats(m4, Xte, yte)
    print("  per-layer dead fraction over the FULL test set (n=%d):" % st["n_test"])
    for k in ["features.1", "features.4", "classifier.2"]:
        v = st[k]
        print("    %-14s dead %d/%d (alive %d)  frac %.6f  preact in [%.6f, %.6f]"
              % (k, v["dead"], v["total"], v["alive"], v["dead_frac"],
                 v["preact_min"], v["preact_max"]))
    print("  logit_std %.6f   test_acc %.4f %%" % (st["logit_std"], st["test_acc"]))

    loss0, groups, detail = grad_probe(m4, loader4, criterion)
    print("\n  loss %.6f  (ln 11 = %.6f)" % (loss0, np.log(11)))
    print("  grad-norm  trunk %.6e   classifier %.6e" % (groups["trunk"], groups["classifier"]))
    for n, g in detail.items():
        print("    %-24s %.6e" % (n, g))
    results["P1"] = {"layer_stats": st, "loss": loss0, "grad_groups": groups,
                     "grad_detail": detail}

    # ---------------------------------------------------------------- P2
    print("\n" + "=" * 72)
    print("P2  LR sweep from the collapsed checkpoint (10 epochs each)")
    print("=" * 72)
    sweep = {}
    for mult in (1, 10, 100, 1000):
        set_seed(2021)
        m = M4_FlattenFC(num_classes=11).to(DEVICE)
        m.load_state_dict(ck["model_state_dict"])
        opt = optim.Adam(m.parameters(), lr=BASE_LR * mult)
        log = []
        dt = train_epochs(m, loader4, criterion, opt, 10, "lr%g" % mult, log)
        a = layer_stats(m, Xte, yte)
        c1 = a["classifier.2"]
        sweep["x%d" % mult] = {
            "lr": BASE_LR * mult, "epochs": 10,
            "dead_c1": c1["dead"], "total_c1": c1["total"], "alive_c1": c1["alive"],
            "dead_frac_c1": c1["dead_frac"],
            "logit_std": a["logit_std"], "test_acc": a["test_acc"],
            "final_loss": log[-1]["loss"], "seconds": dt,
        }
        print("  LR=%.4f (%4dx)  dead_c1=%d/%d  logit_std=%9.6f  acc=%6.2f%%  loss=%.6f  (%0.0fs)"
              % (BASE_LR * mult, mult, c1["dead"], c1["total"],
                 a["logit_std"], a["test_acc"], log[-1]["loss"], dt))
    results["P2"] = sweep

    # ---------------------------------------------------------------- P2b control
    print("\n" + "=" * 72)
    print("P2b control: same sweep on the HEALTHY M2 run (seed 2021)")
    print("=" * 72)
    set_seed(2021)
    m2 = M2_GAPHead(num_classes=11).to(DEVICE)
    ck2 = torch.load("runs/M2_GAPHead/seed2021_batch2a/ckpt_ep640.pt",
                     map_location=DEVICE)
    m2.load_state_dict(ck2["model_state_dict"])
    loader2 = build_loader(Xtr, ytr, 2021)
    a0 = layer_stats(m2, Xte, yte)
    print("  baseline: dead_c1=%.4f  acc=%.2f%%"
          % (a0["classifier.2"]["dead_frac"], a0["test_acc"]))
    ctrl = {"baseline": {"dead_frac_c1": a0["classifier.2"]["dead_frac"],
                         "test_acc": a0["test_acc"]}}
    for mult in (1, 100):
        set_seed(2021)
        m = M2_GAPHead(num_classes=11).to(DEVICE)
        m.load_state_dict(ck2["model_state_dict"])
        opt = optim.Adam(m.parameters(), lr=BASE_LR * mult)
        log = []
        train_epochs(m, loader2, criterion, opt, 10, "ctl%g" % mult, log)
        a = layer_stats(m, Xte, yte)
        ctrl["x%d" % mult] = {"dead_frac_c1": a["classifier.2"]["dead_frac"],
                              "test_acc": a["test_acc"]}
        print("  LR=%6.4f (%4dx)  dead_c1=%d/%d  acc=%6.2f%%"
              % (BASE_LR * mult, mult, a["classifier.2"]["dead"],
                 a["classifier.2"]["total"], a["test_acc"]))
    results["P2b_control"] = ctrl

    # ---------------------------------------------------------------- P3
    print("\n" + "=" * 72)
    print("P3  repair: re-initialise the dead layer, then train at the base LR")
    print("=" * 72)
    repair = {}
    variants = {
        "reinit_fc1": ["classifier.1"],
        "reinit_fc1_and_fc2": ["classifier.1", "classifier.4"],
    }
    for vname, layers in variants.items():
        set_seed(2021)
        m = M4_FlattenFC(num_classes=11).to(DEVICE)
        m.load_state_dict(ck["model_state_dict"])

        def reset(mod):
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

        for lname in layers:
            mod = m
            for part in lname.split("."):
                mod = mod[int(part)] if part.isdigit() else getattr(mod, part)
            reset(mod)

        pre = layer_stats(m, Xte, yte)
        opt = optim.Adam(m.parameters(), lr=BASE_LR)
        log = []
        dt = train_epochs(m, loader4, criterion, opt, 40, vname, log)
        post = layer_stats(m, Xte, yte)
        repair[vname] = {
            "at_repair_epoch0": {"dead_frac_c1": pre["classifier.2"]["dead_frac"],
                                 "test_acc": pre["test_acc"]},
            "after_40_epochs": {"dead_frac_c1": post["classifier.2"]["dead_frac"],
                                "test_acc": post["test_acc"],
                                "logit_std": post["logit_std"]},
            "seconds": dt,
        }
        print("  %-20s  epoch0: dead=%d/%d acc=%5.2f%%   ->  ep40: dead=%d/%d acc=%6.2f%%  (%0.0fs)"
              % (vname, pre["classifier.2"]["dead"], pre["classifier.2"]["total"],
                 pre["test_acc"],
                 post["classifier.2"]["dead"], post["classifier.2"]["total"],
                 post["test_acc"], dt))
    results["P3_repair"] = repair

    # ---------------------------------------------------------------- P4
    print("\n" + "=" * 72)
    print("P4  cascade trace: train M4 from scratch (seed 2021), watch all 3 ReLU layers")
    print("=" * 72)
    set_seed(2021)
    m = M4_FlattenFC(num_classes=11).to(DEVICE)
    opt = optim.Adam(m.parameters(), lr=BASE_LR)
    loader = build_loader(Xtr, ytr, 2021)
    watch = (0, 1, 2, 3, 4, 5, 7, 10, 15, 20)
    cascade = []
    for ep in range(21):
        if ep in watch:
            a = layer_stats(m, Xte, yte)
            row = {"epoch": ep,
                   "features.1": a["features.1"]["dead_frac"],
                   "features.4": a["features.4"]["dead_frac"],
                   "classifier.2": a["classifier.2"]["dead_frac"],
                   "test_acc": a["test_acc"], "logit_std": a["logit_std"]}
            cascade.append(row)
            print("  ep%-3d  conv1 %.4f   conv2 %.4f   fc1 %.4f   acc %5.2f%%  logit_std %.6f"
                  % (ep, row["features.1"], row["features.4"],
                     row["classifier.2"], row["test_acc"], row["logit_std"]))
        if ep == 20:
            break
        m.train()
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = criterion(m(x), y)
            loss.backward()
            opt.step()
    results["P4_cascade_seed2021"] = cascade

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)
    print("\nwrote %s" % OUT)


if __name__ == "__main__":
    main()
