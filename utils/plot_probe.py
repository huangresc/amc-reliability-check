"""
探针结果可视化 + 对拍检查（修复版）
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设置中文字体（避免方块）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


# ============================================
# 1. 读取三个 probe.jsonl
# ============================================
probe_runs = {
    'M4_seed2021 (collapse)': 'runs/probe/M4_seed2021/probe.jsonl',
    'M4_seed42 (control)': 'runs/probe/M4_seed42/probe.jsonl',
    'M2_seed2021 (control)': 'runs/probe/M2_seed2021/probe.jsonl',
}

data = {}
for name, path in probe_runs.items():
    if not os.path.exists(path):
        print(f"[WARN] 找不到 {path}")
        continue
    records = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    data[name] = records
    print(f"OK {name}: {len(records)} 条记录")


# ============================================
# 2. 画三张曲线
# ============================================
os.makedirs('figures', exist_ok=True)

metrics_to_plot = [
    ('dead_frac_c1', 'Dead ReLU Fraction (classifier.1)', 'dead_relu_fraction.png'),
    ('grad_norm_c1', 'Gradient Norm (classifier.1)', 'grad_norm_c1.png'),
    ('logit_std', 'Logit Std (per-sample)', 'logit_std.png'),
]

colors = {
    'M4_seed2021 (collapse)': 'red',
    'M4_seed42 (control)': 'green',
    'M2_seed2021 (control)': 'blue',
}

for metric_key, title, filename in metrics_to_plot:
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, records in data.items():
        epochs = [r['epoch'] for r in records if metric_key in r and r[metric_key] is not None]
        values = [r[metric_key] for r in records if metric_key in r and r[metric_key] is not None]
        if not epochs:
            continue
        ax.plot(epochs, values, 'o-', label=name, color=colors.get(name, 'black'),
                linewidth=2, markersize=6)

    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel(metric_key, fontsize=13)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(fontsize=11)
    if metric_key == 'grad_norm_c1':
        ax.set_yscale('symlog', linthresh=1e-8)
    if metric_key == 'logit_std':
        ax.set_yscale('symlog', linthresh=1e-3)

    plt.tight_layout()
    plt.savefig(f'figures/{filename}', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"OK 已保存: figures/{filename}")


# ============================================
# 3. 对拍 M4 seed2021 探针精度 vs batch2a 精度
# ============================================
print("\n" + "=" * 70)
print("对拍检查：M4 seed2021 探针精度 vs batch2a 精度")
print("=" * 70)

# 读取探针精度
probe_path = 'runs/probe/M4_seed2021/probe.jsonl'
probe_records = []
with open(probe_path, 'r') as f:
    for line in f:
        if line.strip():
            probe_records.append(json.loads(line))

# 读取 batch2a 精度
batch2a_path = 'runs/M4_FlattenFC/seed2021_batch2a/metrics.json'
with open(batch2a_path, 'r') as f:
    batch2a = json.load(f)

# 打印探针记录的epoch和test_acc
print("\n探针记录的epoch和test_acc:")
for r in probe_records:
    if 'test_acc' in r and r['test_acc'] is not None:
        print(f"  epoch {r['epoch']}: test_acc = {r['test_acc']:.6f}")

print("\nbatch2a记录的epoch和acc:")
for ep_str in sorted(batch2a['epochs'].keys(), key=int):
    print(f"  epoch {ep_str}: acc = {batch2a['epochs'][ep_str]['acc']:.6f}")

# 比对
probe_dict = {}
for r in probe_records:
    if 'test_acc' in r and r['test_acc'] is not None:
        probe_dict[r['epoch']] = r['test_acc']

batch2a_dict = {}
for ep_str, val in batch2a['epochs'].items():
    batch2a_dict[int(ep_str)] = val['acc']

common_epochs = sorted(set(probe_dict.keys()) & set(batch2a_dict.keys()))

print(f"\n共同epoch: {common_epochs}")

print(f"\n{'epoch':<10}{'probe_acc':<15}{'batch2a_acc':<15}{'diff':<15}{'status':<10}")
print("-" * 70)

all_match = True
for ep in common_epochs:
    if ep == 0:
        continue
    p_acc = probe_dict[ep]
    b_acc = batch2a_dict[ep]
    diff = p_acc - b_acc
    status = 'OK' if abs(diff) < 0.01 else 'FAIL'
    if status == 'FAIL':
        all_match = False
    print(f"{ep:<10}{p_acc:<15.6f}{b_acc:<15.6f}{diff:<+15.6f}{status:<10}")

print("-" * 70)
if all_match and common_epochs:
    print("\n[RESULT] 逐位相同！")
    print("         -> 探针仪器不改变数值")
    print("         -> 崩溃可逐位复现（确定性训练失败）")
else:
    print("\n[RESULT] 不匹配或没有共同epoch！")
    print("         -> 需排查")


# ============================================
# 4. 判定：死ReLU假说
# ============================================
print("\n" + "=" * 70)
print("判定：死ReLU假说")
print("=" * 70)

m4_2021 = data.get('M4_seed2021 (collapse)', [])

last = next((r for r in m4_2021 if r['epoch'] == 640), None)

if last:
    dead = last.get('dead_frac_c1')
    grad = last.get('grad_norm_c1')
    logit = last.get('logit_std')
    loss = last.get('train_loss')
    acc = last.get('test_acc')

    print(f"\nM4 seed2021 @ epoch 640:")
    print(f"  dead_frac_c1 = {dead:.4f}")
    print(f"  grad_norm_c1 = {grad:.4e}")
    print(f"  logit_std    = {logit:.4f}")
    print(f"  train_loss   = {loss:.4f}")
    print(f"  test_acc     = {acc:.2f}%")

    conditions = {
        'dead_frac_c1 ~= 1.0': dead is not None and dead > 0.99,
        'grad_norm_c1 = 0': grad is not None and grad < 1e-8,
        'logit_std ~= 0': logit is not None and logit < 0.05,
        'loss ~= 2.3979': loss is not None and abs(loss - 2.3979) < 0.01,
    }

    print(f"\n判定条件:")
    for cond, met in conditions.items():
        status = '成立' if met else '不成立'
        print(f"  [{'OK' if met else 'FAIL'}] {cond}: {status}")

    if all(conditions.values()):
        print(f"\n[RESULT] 死ReLU假说成立")
    else:
        print(f"\n[RESULT] 死ReLU假说否证")


# ============================================
# 5. 额外：崩溃率对比
# ============================================
print("\n" + "=" * 70)
print("额外：崩溃率对比")
print("=" * 70)

for name, records in data.items():
    last = next((r for r in records if r['epoch'] == 640), None)
    if last:
        acc = last.get('test_acc', 0)
        failed = acc < 15.0
        print(f"  {name}: final_acc={acc:.2f}%, {'FAILED' if failed else 'OK'}")


print("\n" + "=" * 70)
print("完成")
print("=" * 70)