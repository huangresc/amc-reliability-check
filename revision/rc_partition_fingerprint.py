# -*- coding: utf-8 -*-
"""Fix and verify the partition-provenance field.

THE DEFECT
`src/provenance.py:build_nuisance` records
    "train_idx_sha1": sha1_json(sorted(train_idx.tolist()))
and both loaders hand it `np.arange(n_train)` -- positions in the already-split
arrays.  So the field is sha1(range(n_train)): it records the SIZE of the
training set and nothing about which samples are in it.  Every one of the 185
manifests on disk carries one of exactly two values (95 x sha1(range(176000))
for the re-split, 90 x sha1(range(110000)) for the official split), and two
different splits of the same size collide.  Seven analysis scripts filter runs
on this value (`rc_analyse3.py`, `rc_detect.py`, `rc_weights2.py`,
`rc_crosscheck_partition.py`, `rc_recompute_headline.py`, `rc_protocol_check.py`,
`rc_verify_split.py`), so the collision would silence them rather than fail.

THE FIX (additive, in src/)
`load_data` now also returns, per selected sample, the triple
(mod_id, snr, local_index) -- WHICH sample, not just how many.  `provenance`'s
`sha1_source_keys` sorts those triples and hashes them, and `build_partition`
wraps the result.  `train.py` writes it as a TOP-LEVEL `manifest["partition"]`
key, deliberately outside the nuisance block, so `nuisance_hash` does not move
and all 185 existing manifests stay valid.  Nothing on disk is rewritten: a
manifest is a provenance record, and editing one would falsify it.

THIS SCRIPT
  (1) reproduces `nuisance_hash` for a real official-split manifest using the
      new code path -- the end-to-end proof that the fix changed no hash;
  (2) reconstructs the historical re-split and checks it against its own
      recorded evidence (label-9 prevalence 9.0205 %);
  (3) computes the true fingerprint of both historical partitions and writes
      analysis_results/partition_fingerprints.json, so that the legacy value in
      an old manifest can be resolved to the partition it actually came from.
"""
import io
import json
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import numpy as np
import torch

from src.data import load_data
from src.models import M4_FlattenFC
from src.provenance import build_nuisance, build_partition, sha1_json

PKL = "rml2016_data/RML2016.10a_dict.pkl"
OFFICIAL_MANIFEST = ("runs/M4_FlattenFC/seed42_minverify_M4_seed42_ep640/"
                     "manifest.json")
OUT = os.path.join("analysis_results", "partition_fingerprints.json")

print("=" * 78)
print("(1) Does the new code path still produce the SAME nuisance_hash?")
print("=" * 78)

with open(OFFICIAL_MANIFEST, encoding="utf-8") as f:
    man = json.load(f)
nu = man["nuisance"]

# Rebuild the cfg that produced this manifest, from the manifest itself.
cfg = {k: nu[k] for k in ("optimizer", "lr", "betas", "eps", "weight_decay",
                          "scheduler", "loss", "batch_size", "total_epochs",
                          "seed", "dropout", "normalization", "use_bn")}
cfg.update({
    "dataset": nu["data"]["dataset"],
    "eval_batch_size": nu["eval"]["eval_batch_size"],
    "num_workers": nu.get("num_workers", 0),
    "pin_memory": nu.get("pin_memory", False),
})
if "shuffle_seed" in nu:
    cfg["shuffle_seed"] = nu["shuffle_seed"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data = load_data(normalization=cfg["normalization"])
model = M4_FlattenFC(num_classes=11).to(DEVICE)
x_sample = data["X_train_t"][:8].to(DEVICE)

rebuilt = build_nuisance(cfg, data["train_idx"], data["test_idx"], model,
                         x_sample)
h = sha1_json(rebuilt)
print(f"  manifest      : {OFFICIAL_MANIFEST}")
print(f"  recorded hash : {man['nuisance_hash']}")
print(f"  rebuilt  hash : {h}")
if h != man["nuisance_hash"]:
    for k in sorted(set(rebuilt) | set(nu)):
        if rebuilt.get(k) != nu.get(k):
            print(f"  MISMATCH at {k!r}:\n    recorded {nu.get(k)!r}\n"
                  f"    rebuilt  {rebuilt.get(k)!r}")
    raise SystemExit("nuisance_hash MOVED -- the fix is not additive. Stop.")
print("  OK: byte-identical. The fix moved no hash.")
print(f"  legacy data field is size-only: train_idx_sha1={nu['data']['train_idx_sha1']}")
print(f"     = sha1(list(range({nu['data']['n_train']}))) -> "
      f"{sha1_json(list(range(nu['data']['n_train'])))}")

official = build_partition(data["partition"])
print(f"\n  new top-level partition block:")
for k, v in official.items():
    print(f"    {k}: {v}")

print("\n" + "=" * 78)
print("(2) Reconstruct the historical 176000/44000 re-split")
print("=" * 78)
with open(PKL, "rb") as f:
    raw = pickle.load(f, encoding="latin1")
mods = sorted({k[0] for k in raw})
m2i = {m: i for i, m in enumerate(mods)}

X, src_mod, src_snr, src_local = [], [], [], []
for (mod, snr), s in raw.items():
    X.append(s)
    src_mod.append(np.full(s.shape[0], m2i[mod]))
    src_snr.append(np.full(s.shape[0], int(snr)))
    src_local.append(np.arange(s.shape[0]))
Xcat = np.concatenate(X)
src_mod = np.concatenate(src_mod)
src_snr = np.concatenate(src_snr)
src_local = np.concatenate(src_local)
y = src_mod.copy()
N = len(Xcat)

np.random.seed(42)
perm = np.random.permutation(N)
cut = int(N * 0.8)
keys_all = np.stack([src_mod, src_snr, src_local], axis=1).astype(np.int32)

resplit = {
    "rule": "legacy_unstratified_re-split_80_20",
    "seed": 42,
    "n_groups": len(raw),
    "n_train": int(cut),
    "n_test": int(N - cut),
    "train_source_keys": keys_all[perm[:cut]],
    "test_source_keys": keys_all[perm[cut:]],
}
prev = 100.0 * float((y[perm[cut:]] == 9).mean())
print(f"  rebuilt test set n={N - cut}, label-9 prevalence {prev:.4f} %")
assert abs(prev - 9.0205) < 0.0005, "re-split reconstruction drifted"
print("  OK: matches the recorded collapsed accuracy (9.0205 %).")
resplit_block = build_partition(resplit)
print(f"  fingerprint: train_source_sha1={resplit_block['train_source_sha1']}")

print("\n" + "=" * 78)
print("(3) Do the two historical partitions now separate?")
print("=" * 78)
print(f"  official  train {official['train_source_sha1']}")
print(f"  re-split  train {resplit_block['train_source_sha1']}")
assert official["train_source_sha1"] != resplit_block["train_source_sha1"]
print("  OK: distinct.")
print(f"  (These two also differ in n_train, {official['n_train']} vs "
      f"{resplit_block['n_train']}, so the legacy field separated them by "
      f"accident. The case it cannot separate is a same-size pair -- below.)")

print("\n" + "=" * 78)
print("(4) The collision itself: two official splits of the SAME size")
print("=" * 78)
print("  Re-running the same rule with a different seed gives a different")
print("  partition of identical size. This is the case the legacy field cannot")
print("  see and the new one can.")
alt = load_data(normalization=cfg["normalization"], seed=cfg["seed"] + 1)
alt_block = build_partition(alt["partition"])
legacy_same = (sha1_json(list(range(alt_block["n_train"])))
               == sha1_json(list(range(official["n_train"]))))
same_pixels = bool((alt["X_train_t"] == data["X_train_t"]).all())
print(f"  partition A (seed {official['seed']:>3}): n_train {official['n_train']}")
print(f"  partition B (seed {alt_block['seed']:>3}): n_train {alt_block['n_train']}")
print(f"    identical training tensors?      {same_pixels}")
print(f"    legacy train_idx_sha1 identical? {legacy_same}"
      f"      <- the defect: indistinguishable")
print(f"      A {sha1_json(list(range(official['n_train'])))}")
print(f"      B {sha1_json(list(range(alt_block['n_train'])))}  (same string)")
print(f"    new fingerprint identical?       "
      f"{official['train_source_sha1'] == alt_block['train_source_sha1']}"
      f"      <- fixed: separated")
print(f"      A {official['train_source_sha1']}")
print(f"      B {alt_block['train_source_sha1']}")
if same_pixels or not legacy_same or \
        official["train_source_sha1"] == alt_block["train_source_sha1"]:
    raise SystemExit("collision demonstration failed -- investigate")
print("  OK: the legacy field collides, the new field does not.")

sidecar = {
    "note": ("True fingerprints of the two historical partitions. Maps the "
             "legacy, size-only nuisance.data.train_idx_sha1 found in the 185 "
             "manifests already on disk to the partition it actually came "
             "from. Manifests were NOT rewritten."),
    "field": "manifest.partition.train_source_sha1",
    "legacy_field": "manifest.nuisance.data.train_idx_sha1",
    "legacy_note": "sha1(list(range(n_train))); identifies size only.",
    "partitions": {
        "official_stratified_110000_110000": {
            "legacy_train_idx_sha1":
                "0ff510c64bc09b86e1a8224246292c9095c75d89",
            "manifests_affected": 90,
            "build_partition": official,
        },
        "legacy_resplit_176000_44000": {
            "legacy_train_idx_sha1":
                "1b078bc88d4677b5aa16a72093b044a251e0679d",
            "manifests_affected": 95,
            "build_partition": resplit_block,
        },
    },
}
os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(sidecar, f, indent=2)
print(f"\nwrote {OUT}")
