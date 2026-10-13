"""
Zero-GPU analysis over the completed revision queue.

Reads only runs/*/metrics.json. Reports, per arm, the final-epoch accuracy of
every run, the collapse count, and the Wilson 95% interval on the failure rate.

Usage:
    python -I rc_analyse.py
"""
import os
import sys
import json
import glob

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)

COLLAPSE_THRESHOLD = 15.0   # chance = 9.09%, healthy FLAT final = 45-50%

ORIGINAL_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
NEW_SEEDS = [7, 13, 55, 101, 256, 512, 777, 1001, 1500, 2048,
             3141, 4096, 5150, 6666, 7777, 8888, 9999, 12345, 24680, 31337]


def load(variant, run_id):
    path = os.path.join("runs", variant, run_id, "metrics.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def final_acc(m):
    epochs = m["epochs"]
    last = max(int(k) for k in epochs)
    return last, epochs[str(last)]["acc"]


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))


def report(label, rows):
    """rows: list of (name, final_acc)."""
    accs = [a for _, a in rows]
    bad = [(n, a) for n, a in rows if a < COLLAPSE_THRESHOLD]
    print(f"\n=== {label}   n={len(rows)} ===")
    for n, a in sorted(rows, key=lambda r: r[1]):
        flag = "  <-- COLLAPSE" if a < COLLAPSE_THRESHOLD else ""
        print(f"  {n:<28} {a:6.2f}%{flag}")
    if accs:
        good = [a for a in accs if a >= COLLAPSE_THRESHOLD]
        print(f"  collapsed {len(bad)}/{len(rows)}"
              f"   Wilson95=[{wilson(len(bad), len(rows))[0]*100:.1f}%,"
              f" {wilson(len(bad), len(rows))[1]*100:.1f}%]")
        if good:
            print(f"  healthy: mean={sum(good)/len(good):.2f}%  "
                  f"min={min(good):.2f}%  max={max(good):.2f}%")
    return len(bad), len(rows)


def main():
    print("collapse threshold: < %.2f%% (chance = 9.09%%)" % COLLAPSE_THRESHOLD)

    # ---- A1: FLAT head, raw inputs, 30 initialisation seeds -------------
    a1 = []
    for s in ORIGINAL_SEEDS:
        m = load("M4_FlattenFC", f"seed{s}_batch2a")
        if m:
            _, a = final_acc(m)
            a1.append((f"seed{s} (orig)", a))
    for s in NEW_SEEDS:
        m = load("M4_FlattenFC", f"seed{s}_rc30_s{s}")
        if m:
            _, a = final_acc(m)
            a1.append((f"seed{s} (new)", a))
    report("A1  M4_FlattenFC  raw  (30 init seeds)", a1)

    # ---- A2: GAP head, raw inputs, same 30 seeds ------------------------
    a2 = []
    for s in ORIGINAL_SEEDS:
        m = load("M2_GAPHead", f"seed{s}_batch2a")
        if m:
            _, a = final_acc(m)
            a2.append((f"seed{s} (orig)", a))
    for s in NEW_SEEDS:
        m = load("M2_GAPHead", f"seed{s}_rc30_s{s}")
        if m:
            _, a = final_acc(m)
            a2.append((f"seed{s} (new)", a))
    report("A2  M2_GAPHead  raw  (30 init seeds)", a2)

    # ---- B: FLAT head, init seed pinned at 42, 20 explicit orders -------
    b = []
    for o in range(20):
        m = load("M4_FlattenFC", f"seed42_rcord{o:03d}")
        if m:
            _, a = final_acc(m)
            b.append((f"order{o:02d}", a))
    report("B   M4_FlattenFC  raw  init=42, pinned order", b)

    # ---- contrast: the same init seed under the ambient RNG stream ------
    print("\n=== anchor: init seed 42 under the ambient RNG stream ===")
    for tag in ["batch2a", "rc30_s42"]:
        m = load("M4_FlattenFC", f"seed42_{tag}")
        if m:
            _, a = final_acc(m)
            print(f"  seed42_{tag:<12} {a:6.2f}%")

    # ---- paired FLAT vs GAP on the shared 30 seeds ----------------------
    print("\n=== paired FLAT - GAP (raw), per seed, pp ===")
    d = []
    for s in ORIGINAL_SEEDS + NEW_SEEDS:
        mf = load("M4_FlattenFC", f"seed{s}_batch2a") or \
            load("M4_FlattenFC", f"seed{s}_rc30_s{s}")
        mg = load("M2_GAPHead", f"seed{s}_batch2a") or \
            load("M2_GAPHead", f"seed{s}_rc30_s{s}")
        if mf and mg:
            _, af = final_acc(mf)
            _, ag = final_acc(mg)
            d.append(af - ag)
            print(f"  seed{s:<8} FLAT={af:6.2f}  GAP={ag:6.2f}  d={af-ag:+6.2f}")
    if d:
        n = len(d)
        mean = sum(d) / n
        var = sum((x - mean) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
        se = (var / n) ** 0.5
        print(f"  n={n}  mean diff={mean:+.2f}pp  sd={var**0.5:.2f}  "
              f"se={se:.2f}  t={mean/se:.2f}" if se > 0 else "")


if __name__ == "__main__":
    main()
