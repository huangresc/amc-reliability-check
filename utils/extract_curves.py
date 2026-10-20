import json
import numpy as np

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
milestones = ['20', '40', '80', '120', '160', '240', '320']

print("=" * 80)
print("M2 的 7 点完整曲线（每个种子的值 + 均值 ± std）")
print("=" * 80)

# 收集每个run在每个milestone的acc
data = {ep: [] for ep in milestones}
for rid in runs:
    with open(f'runs/{variant}/{rid}/metrics.json', 'r') as f:
        m = json.load(f)
    for ep in milestones:
        if ep in m['epochs']:
            data[ep].append(m['epochs'][ep]['acc'])

# 打印每个种子的值
print("\n每个种子的值:")
print(f"{'Run':<25}" + "".join([f"{'ep'+ep:<12}" for ep in milestones]))
print("-" * 110)
for rid in runs:
    with open(f'runs/{variant}/{rid}/metrics.json', 'r') as f:
        m = json.load(f)
    row = f"{rid:<25}"
    for ep in milestones:
        v = m['epochs'].get(ep, {}).get('acc', None)
        row += f"{v:<12.4f}" if v else f"{'N/A':<12}"
    print(row)

# 打印均值±std
print("\n均值 ± std:")
print(f"{'':<25}" + "".join([f"{'ep'+ep:<12}" for ep in milestones]))
print("-" * 110)
row = f"{'Mean ± std':<25}"
for ep in milestones:
    vals = data[ep]
    row += f"{np.mean(vals):.4f}±{np.std(vals, ddof=1):.2f}"
row = f"{'':<25}" + "".join([f"{np.mean(data[ep]):.4f}±{np.std(data[ep], ddof=1):.2f}" for ep in milestones])
print(row)