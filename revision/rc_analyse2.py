"""
Follow-up checks: (1) does init seed 42 ever collapse under any run tag we
have; (2) the epoch trace of the two collapsing runs; (3) robust paired
FLAT-vs-GAP statistics with the collapses handled explicitly.
"""
import os
import json

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ORIGINAL = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
NEW = [7, 13, 55, 101, 256, 512, 777, 1001, 1500, 2048,
       3141, 4096, 5150, 6666, 7777, 8888, 9999, 12345, 24680, 31337]


def m(variant, run_id):
    p = os.path.join("runs", variant, run_id, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def curve(mo):
    return [(int(k), v["acc"]) for k, v in sorted(mo["epochs"].items(),
                                                 key=lambda kv: int(kv[0]))]


print("=== every completed run with init seed 42 ===")
root = os.path.join("runs", "M4_FlattenFC")
for d in sorted(os.listdir(root)):
    if not d.startswith("seed42"):
        continue
    mo = m("M4_FlattenFC", d)
    if mo:
        c = curve(mo)
        print(f"  {d:<45} final={c[-1][1]:6.2f}%  ep1={c[0][1]:6.2f}%")
    else:
        print(f"  {d:<45} (no metrics.json)")

print("\n=== epoch trace of the two collapsing runs (FLAT, raw) ===")
for tag, run in [("seed2021", "seed2021_batch2a"),
                 ("seed1500", "seed1500_rc30_s1500")]:
    mo = m("M4_FlattenFC", run)
    if mo:
        print(f"  {tag}: " + "  ".join(f"ep{e}={a:.2f}" for e, a in curve(mo)))
    org = m("M2_GAPHead", run)
    if org:
        print(f"  {tag} GAP: " + "  ".join(f"ep{e}={a:.2f}" for e, a in curve(org)))

print("\n=== how many healthy FLAT runs sit below 47% at epoch 1-7? ===")
early = {}
for s in ORIGINAL:
    mo = m("M4_FlattenFC", f"seed{s}_batch2a")
    if mo:
        early[s] = curve(mo)
for s in NEW:
    mo = m("M4_FlattenFC", f"seed{s}_rc30_s{s}")
    if mo:
        early[s] = curve(mo)
for s, c in sorted(early.items(), key=lambda kv: kv[1][6][1]):
    collapse = c[-1][1] < 15.0
    print(f"  seed{s:<8} ep1={c[0][1]:6.2f}  ep7={c[6][1]:6.2f}  "
          f"final={c[-1][1]:6.2f}{'  <-- COLLAPSE' if collapse else ''}")

print("\n=== paired FLAT - GAP, collapses excluded ===")
d = []
for s in ORIGINAL + NEW:
    mf = m("M4_FlattenFC", f"seed{s}_batch2a") or \
        m("M4_FlattenFC", f"seed{s}_rc30_s{s}")
    mg = m("M2_GAPHead", f"seed{s}_batch2a") or \
        m("M2_GAPHead", f"seed{s}_rc30_s{s}")
    if mf and mg:
        af = curve(mf)[-1][1]
        ag = curve(mg)[-1][1]
        if af >= 15.0:
            d.append(af - ag)
n = len(d)
mean = sum(d) / n
var = sum((x - mean) ** 2 for x in d) / (n - 1)
se = (var / n) ** 0.5
sd = sorted(d)
med = (sd[n // 2] if n % 2 else (sd[n // 2 - 1] + sd[n // 2]) / 2)
print(f"  n={n}  mean={mean:+.2f}pp  median={med:+.2f}pp  sd={var**0.5:.2f}  "
      f"se={se:.2f}  95%CI=[{mean-1.96*se:+.2f}, {mean+1.96*se:+.2f}]")
print(f"  sign test: {(sum(1 for x in d if x < 0))}/{n} seeds negative")
