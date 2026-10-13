# -*- coding: utf-8 -*-
"""Loaders for the two on-disk results sources, with no recomputation.

`load_per_seed`  -- tables/*_per_seed.csv, the re-split (176000/44000) runs.
`load_official`  -- runs/*/seed*_minverify_*/metrics.json, the official
                    stratified (110000/110000) runs, located with the same
                    filter as rc_recompute_headline.py (640 epochs,
                    normalization=none, no augmentation, split fingerprint
                    OFFICIAL, directory tag "minverify").

Both return {seed: {epoch_str: accuracy_percent}}.
"""
import csv
import json
import os

CHECKPOINTS = ["20", "40", "80", "120", "160", "240", "320", "480", "640"]

OFFICIAL = "0ff510c64bc09b86e1a8224246292c9095c75d89"
SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

FILES = {
    "M2_GAPHead": "main_architecture_per_seed.csv",
    "M4_FlattenFC": "main_architecture_per_seed.csv",
    "M5_GAPHead_PhaseAug": "phase_augmentation_per_seed.csv",
    "M5b_GAPHead_PhaseAug4": "phase_augmentation_per_seed.csv",
    "M2n_g": "normalization_ablation_M2_per_seed.csv",
    "M2n_p": "normalization_ablation_M2_per_seed.csv",
    "M4n_g": "normalization_ablation_M4_per_seed.csv",
    "M4n_p": "normalization_ablation_M4_per_seed.csv",
}


def load_per_seed(root="."):
    """{variant: {seed: {epoch: acc}}} from tables/."""
    out = {}
    cache = {}
    for variant, fname in FILES.items():
        path = os.path.join(root, "tables", fname)
        if fname not in cache:
            rows = []
            with open(path, encoding="utf-8") as fh:
                for r in csv.DictReader(fh):
                    rows.append(r)
            cache[fname] = rows
        per = {}
        for r in cache[fname]:
            if r["variant"] != variant:
                continue
            per[int(r["seed"])] = {e: float(r["ep" + e]) for e in CHECKPOINTS}
        if per:
            out[variant] = per
    return out


def load_official(root=".", seeds=SEEDS):
    """(data, survivors) for the official split, read from metrics.json only.

    data[(variant, seed)] = {epoch: acc}
    """
    root = os.path.abspath(root)
    data = {}
    for variant in ("M2_GAPHead", "M4_FlattenFC"):
        vroot = os.path.join(root, "runs", variant)
        for d in sorted(os.listdir(vroot)):
            if "minverify" not in d:
                continue
            mp = os.path.join(vroot, d, "manifest.json")
            kp = os.path.join(vroot, d, "metrics.json")
            if not (os.path.exists(mp) and os.path.exists(kp)):
                continue
            with open(mp, encoding="utf-8") as f:
                man = json.load(f)
            nu = man["nuisance"]
            if (man["seed"] not in seeds or nu["total_epochs"] != 640
                    or nu["normalization"] != "none"
                    or man["treatment"]["use_aug"]
                    or nu["data"]["train_idx_sha1"] != OFFICIAL):
                continue
            with open(kp, encoding="utf-8") as f:
                met = json.load(f)
            ep = {k: v["acc"] for k, v in met["epochs"].items()}
            if not all(c in ep for c in CHECKPOINTS):
                continue
            data[(variant, man["seed"])] = ep

    flat = {s: data[("M4_FlattenFC", s)]["640"] for s in seeds}
    gap = {s: data[("M2_GAPHead", s)]["640"] for s in seeds}
    survivors = [s for s in seeds if flat[s] >= 15.0 and gap[s] >= 15.0]
    return data, survivors
