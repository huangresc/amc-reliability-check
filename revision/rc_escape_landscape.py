# -*- coding: utf-8 -*-
"""Loss-landscape signatures at the re-initialised state vs the escape outcome.

Advisor comment (MAJOR 2): the escape is reported as a black box -- re-init
leaves the collapsed state in ~half of draws and no measured quantity predicts
which half.  The manuscript notes that a Hessian-spectrum analysis is future
work.  This script does the cheap version of that analysis now.

For each of the 20 perturbation draws on each collapsed cell (the exact RNG path
of rc_repair_perturbation.py -- set_seed(p) -> construct -> load -> reset fc1),
we measure, at epoch 0 BEFORE any training, three loss-landscape signatures of
the re-initialised fc1 layer:

    loss    : cross-entropy on a fixed training batch
    grad    : Frobenius norm of the fc1 gradient on that batch
    sharp   : top eigenvalue of the fc1 loss Hessian (power iteration)

and ask whether any of them predicts the recorded epoch-40 accuracy (the escape
outcome).  "Initial dead count does not predict" is already established
(Spearman rho=+0.22, p=0.35); this tests whether curvature does.

Everything is read-only over rc_repair_perturbation.json and the two collapsed
checkpoints.  Nothing is trained here.
"""
import io
import json
import os
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8",
                              line_buffering=True)

import numpy as np
import torch
import torch.nn as nn

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from scipy import stats

from src.data import load_data
from src.models import M4_FlattenFC
from rc_repair_perturbation import (PERTURB_SEEDS, load_resplit, load_official,
                                    set_seed, get_module, DEVICE)

REPAIR = "analysis_results/rc_repair_perturbation.json"
OUT = "analysis_results/rc_escape_landscape.json"

RUNS = [
    ("resplit", "runs/M4_FlattenFC/seed2021_batch2a/ckpt_ep640.pt"),
    ("official", "runs/M4_FlattenFC/seed42_minverify_M4_seed42_ep640/ckpt_ep640.pt"),
]
POWER_ITERS = 60
BS = 256


def reinit_fc1(ck, p):
    """Reproduce rc_repair_perturbation.py's exact re-init for draw p."""
    set_seed(p)
    model = M4_FlattenFC(num_classes=11).to(DEVICE)
    model.load_state_dict(ck["model_state_dict"])
    get_module(model, "classifier.1").reset_parameters()
    return model


def flat_params(model, which="fc1"):
    if which == "fc1":
        layer = model.classifier[1]
        return [layer.weight, layer.bias]
    return [p for p in model.parameters() if p.requires_grad]


def signatures(model, xb, yb, criterion):
    """loss, fc1 gradient norm, and fc1 Hessian top eigenvalue."""
    model.train()
    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
    params = flat_params(model, "fc1")

    loss = criterion(model(xb), yb)
    grads = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
    grad_norm = float(sum((g.detach() ** 2).sum() for g in grads) ** 0.5)

    # power iteration for the top eigenvalue of the fc1 Hessian
    vs = [torch.randn_like(p) for p in params]
    n = (sum((v ** 2).sum() for v in vs) ** 0.5).item()
    vs = [v / n for v in vs]
    eig = 0.0
    for _ in range(POWER_ITERS):
        gv = sum((g * v).sum() for g, v in zip(grads, vs))
        hv = torch.autograd.grad(gv, params, retain_graph=True)
        eig = float(sum((h * v).sum() for h, v in zip(hv, vs)))
        n = (sum((h ** 2).sum() for h in hv) ** 0.5).item()
        if n < 1e-12:
            eig = 0.0
            break
        vs = [h / n for h in hv]
    return float(loss.item()), grad_norm, eig


def main():
    with open(REPAIR, encoding="utf-8") as f:
        repair = json.load(f)

    criterion = nn.CrossEntropyLoss()
    out = {}

    for run, ckpt_path in RUNS:
        Xtr, ytr, Xte, yte = (load_resplit() if run == "resplit"
                              else load_official())
        # fixed batch so all 20 draws are measured on the SAME data
        xb = Xtr[:BS]
        yb = ytr[:BS]
        ck = torch.load(ckpt_path, map_location=DEVICE)
        cells = repair[run]["cells"]

        rows = {}
        for p in PERTURB_SEEDS:
            key = f"p{p}_reinit_fc1"
            acc40 = cells[key]["epoch40"]["acc"]
            dead0 = cells[key]["epoch0"]["dead_fc1"]
            model = reinit_fc1(ck, p)
            loss, gnorm, sharp = signatures(model, xb, yb, criterion)
            rows[p] = {"epoch0_dead_fc1": dead0, "loss": loss,
                       "grad_norm": gnorm, "sharpness": sharp, "acc40": acc40}
            print(f"  {run:9s} p={p:>5}  dead0={dead0:.3f}  loss={loss:.4f}  "
                  f"|g|={gnorm:.4f}  sharp={sharp:+.4f}  acc40={acc40:.2f}",
                  flush=True)
        out[run] = rows

        # ---- does any epoch-0 signature predict escape? -----------------
        print(f"\n  {run}: Spearman rho vs epoch-40 accuracy (n={len(rows)}):")
        for name in ("epoch0_dead_fc1", "loss", "grad_norm", "sharpness"):
            xs = [rows[p][name] for p in rows]
            ys = [rows[p]["acc40"] for p in rows]
            rho, pval = stats.spearmanr(xs, ys)
            print(f"    {name:>16}: rho={rho:+.3f}  p={pval:.3f}")

        # ---- sharpness vs the binary escape outcome --------------------
        thr = 15.0
        sharp_esc = [rows[p]["sharpness"] for p in rows if rows[p]["acc40"] >= thr]
        sharp_col = [rows[p]["sharpness"] for p in rows if rows[p]["acc40"] < thr]
        if sharp_esc and sharp_col:
            u, pu = stats.mannwhitneyu(sharp_esc, sharp_col, alternative="two-sided")
            print(f"    sharpness  escape median={np.median(sharp_esc):+.4f} "
                  f"collapse median={np.median(sharp_col):+.4f}  "
                  f"Mann-Whitney p={pu:.3f}")

    os.makedirs("analysis_results", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
