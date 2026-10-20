"""Check which data protocol each run used -- the arms must not be mixed."""
import os
import json
import collections

os.chdir(os.path.dirname(os.path.abspath(__file__)))

groups = collections.defaultdict(list)
for variant in ["M4_FlattenFC", "M2_GAPHead"]:
    root = os.path.join("runs", variant)
    if not os.path.isdir(root):
        continue
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, "manifest.json")
        if not os.path.exists(p):
            continue
        with open(p) as f:
            man = json.load(f)
        n = man["nuisance"]["data"]
        key = (n["n_train"], n["n_test"], n["train_idx_sha1"][:8])
        groups[(variant, key)].append(d)

for (variant, key), runs in sorted(groups.items()):
    print(f"\n{variant}  n_train={key[0]}  n_test={key[1]}  idx_sha1={key[2]}..."
          f"  -> {len(runs)} runs")
    for r in runs[:6]:
        print(f"     {r}")
    if len(runs) > 6:
        print(f"     ... and {len(runs)-6} more")
