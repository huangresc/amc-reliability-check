import json
import os
import numpy as np
import hashlib

def sha1_json(obj):
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

# 11个run的完整列表
runs = [
    'seed42_gate1',
    'seed42_gate1_rep2',
    'seed123_gate1',
    'seed456_gate1',
    'seed789_gate1',
    'seed1011_gate1',
    'seed1213_gate1',
    'seed1415_gate1',
    'seed1617_gate1',
    'seed1819_gate1',
    'seed2021_gate1',
]

variant = 'M2_GAPHead'

print("=" * 70)
print("闸门1 核对")
print("=" * 70)

# === 核对1：不含seed的nuisance_hash一致性 ===
print("\n1. nuisance_hash 一致性（不含seed）")
print("-" * 70)
hashes_no_seed = set()
for rid in runs:
    with open(f'runs/{variant}/{rid}/manifest.json', 'r') as f:
        m = json.load(f)
    n = m['nuisance'].copy()
    n.pop('seed', None)
    hashes_no_seed.add(sha1_json(n))
print(f"   不含seed的nuisance_hash: {len(hashes_no_seed)} 个不同值")
if len(hashes_no_seed) == 1:
    print(f"   ✅ 全部相同: {list(hashes_no_seed)[0]}")
else:
    print(f"   ❌ 不一致: {hashes_no_seed}")

# === 核对2：160→320增量 ===
print("\n2. 160→320 增量")
print("-" * 70)
accs_160, accs_320 = [], []
for rid in runs:
    with open(f'runs/{variant}/{rid}/metrics.json', 'r') as f:
        m = json.load(f)
    accs_160.append(m['epochs']['160']['acc'])
    accs_320.append(m['epochs']['320']['acc'])
mean_160 = np.mean(accs_160)
mean_320 = np.mean(accs_320)
print(f"   160ep均值: {mean_160:.2f}%")
print(f"   320ep均值: {mean_320:.2f}%")
print(f"   增量: {mean_320 - mean_160:+.2f}pp")
if abs(mean_320 - mean_160) < 0.5:
    print(f"   ✅ 在+0.5pp以内（平台）")
else:
    print(f"   ⚠️ 超过+0.5pp，可能未平台")

# === 核对3：单epoch耗时是否随时间上升 ===
print("\n3. 单epoch耗时")
print("-" * 70)
for rid in runs[:5]:
    with open(f'runs/{variant}/{rid}/metrics.json', 'r') as f:
        m = json.load(f)
    times = [m['epochs'][str(ep)]['epoch_time'] for ep in [20, 40, 80, 120, 160, 240, 320]]
    print(f"   {rid}: {[f'{t:.1f}' for t in times]}")

# === 核对4：十种子均值 ===
print("\n4. M2十种子均值")
print("-" * 70)
print(f"   M2@320均值: {mean_320:.2f} ± {np.std(accs_320, ddof=1):.2f}")
if 47 <= mean_320 <= 51:
    print(f"   ✅ 落在47-51%区间")
else:
    print(f"   ⚠️ 不在47-51%区间")

# === rep1 vs rep2 差异 ===
print("\n5. rep1 vs rep2 逐epoch差异")
print("-" * 70)
with open(f'runs/{variant}/seed42_gate1/metrics.json', 'r') as f:
    m1 = json.load(f)
with open(f'runs/{variant}/seed42_gate1_rep2/metrics.json', 'r') as f:
    m2 = json.load(f)
for ep in ['20', '40', '80', '120', '160', '240', '320']:
    a1 = m1['epochs'][ep]['acc']
    a2 = m2['epochs'][ep]['acc']
    print(f"   Epoch {ep}: rep1={a1:.4f}%, rep2={a2:.4f}%, 差={a2-a1:+.4f}pp")

print("\n" + "=" * 70)
print("闸门1 核对完成")
print("=" * 70)