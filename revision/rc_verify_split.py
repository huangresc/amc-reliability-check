"""Rebuild the 176 000/44 000 re-split and check it against the two observed
constant-predictor accuracies.

Why this is needed: `escape_repair_sweep.py` (the manuscript's mechanism
source) ran on 2026-10-04 against a `src/data.py` that implemented the OLD
re-split.  `src/data.py` was later replaced by the official 110 000/110 000
stratified split, so the current loader cannot reproduce the seed-2021/batch2a
checkpoint's own training data.  No file on disk records the re-split's actual
indices -- `train_idx_sha1` is only sha1(list(range(n_train))), so it encodes
n_train and nothing else (verified: range(176000) -> 1b078bc8..., range(110000)
-> 0ff510c6...).

The only re-split generator in the repository is setup_project.py:131-136:
    np.random.seed(seed); indices = np.random.permutation(len(X))
    train = indices[:int(0.8 * len(X))]; test = the rest
with the default seed=42.  This script rebuilds it and reports the class
prevalences of the test set, which a constant predictor reproduces exactly.

Checks:
  * re-split  -> the emitted label's test-set prevalence should be 9.0205 %
  * official  -> every class is exactly 9.090909... % (= 1/11, stratified)
"""
import pickle

import numpy as np

PKL = "rml2016_data/RML2016.10a_dict.pkl"
RESPLIT_ACC = 9.0205          # the manuscript's collapsed accuracy
OFFICIAL_ACC = 100.0 / 11.0   # 9.090909090909092
EMITTED = 9                   # label the seed-2021 collapse emits


def raw():
    with open(PKL, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    mods = sorted({k[0] for k in data})
    m2i = {m: i for i, m in enumerate(mods)}
    X, y = [], []
    for (mod, snr), s in data.items():
        X.append(s)
        y.append(np.full(s.shape[0], m2i[mod]))
    return np.concatenate(X).astype("float32"), np.concatenate(y).astype("int64")


def prevalence(y):
    c = np.bincount(y, minlength=11)
    return c, 100.0 * c / len(y)


X, y = raw()
print(f"total samples {len(X)}   classes {sorted(set(y.tolist()))}")

# ---- re-split, exactly as setup_project.py generated it -------------------
np.random.seed(42)
idx = np.random.permutation(len(X))
cut = int(len(X) * 0.8)
y_rt = y[idx[cut:]]
c_rt, p_rt = prevalence(y_rt)
print(f"\nre-split  : n_train {cut}  n_test {len(y_rt)}")
print(f"  per-class test counts {c_rt.tolist()}")
print(f"  max prevalence {p_rt.max():.4f} %  (argmax label {int(p_rt.argmax())})")
print(f"  prevalence of emitted label {EMITTED}: {p_rt[EMITTED]:.4f} %")
print(f"  observed collapsed accuracy        : {RESPLIT_ACC:.4f} %")

# ---- official split, as the current src/data.py builds it -----------------
from src.data import load_data  # noqa: E402

d = load_data(normalization="none")
y_off = d["y_test_t"].numpy()
c_of, p_of = prevalence(y_off)
print(f"\nofficial  : n_train {len(d['y_train_t'])}  n_test {len(y_off)}")
print(f"  per-class test counts {c_of.tolist()}")
print(f"  max prevalence {p_of.max():.4f} %  (argmax label {int(p_of.argmax())})")
print(f"  prevalence of emitted label {EMITTED}: {p_of[EMITTED]:.4f} %")
print(f"  expected (stratified)              : {OFFICIAL_ACC:.4f} %")

print("\n" + "=" * 66)
ok_r = abs(p_rt[EMITTED] - RESPLIT_ACC) < 5e-3
ok_o = abs(p_of[EMITTED] - OFFICIAL_ACC) < 1e-9
print(f"re-split reconstruction matches 9.0205 % on label {EMITTED}: {ok_r}")
print(f"official split matches 1/11 exactly                       : {ok_o}")
