"""
批量运行 M2@320, n=10 + seed42重复
按 grid_gate1_m2.yaml 的顺序执行
"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import subprocess
import sys
import time
import os
import shutil
import yaml

with open('grid_gate1_m2.yaml', 'r') as f:
    grid = yaml.safe_load(f)

variant = grid['variant']
total_epochs = grid['total_epochs']
batch_tag = grid['batch_tag']
runs = grid['runs']

print("=" * 70)
print(f"批量运行 {variant} @ {total_epochs} epoch, {len(runs)} 次")
print("=" * 70)

total_start = time.time()

for i, run_cfg in enumerate(runs):
    seed = run_cfg['seed']
    repeat_id = run_cfg.get('repeat_id', 1)

    if repeat_id == 1:
        run_id = f"seed{seed}_{batch_tag}"
    else:
        run_id = f"seed{seed}_{batch_tag}_rep{repeat_id}"

    run_dir = f"runs/{variant}/{run_id}"
    metrics_path = f"{run_dir}/metrics.json"

    print(f"\n{'='*70}")
    print(f"[{i+1}/{len(runs)}] {run_id}")
    print(f"{'='*70}")

    # 已完成则跳过
    if os.path.exists(metrics_path):
        print(f"  ⏭️ {run_id} 已完成，跳过")
        continue

    # 崩溃残留则删除
    if os.path.exists(run_dir):
        print(f"  ⚠️ {run_id} 有崩溃残留，删除后重跑")
        shutil.rmtree(run_dir)

    cmd = [sys.executable, '-m', 'src.train',
           '--variant', variant,
           '--seed', str(seed),
           '--total_epochs', str(total_epochs),
           '--repeat_id', str(repeat_id),
           '--batch_tag', batch_tag]

    start = time.time()
    result = subprocess.run(cmd, cwd='.')
    elapsed = time.time() - start

    if result.returncode != 0:
        print(f"  ❌ {run_id} 失败")
        sys.exit(1)

    print(f"  ✅ {run_id} 完成 ({elapsed/60:.1f}分钟)")

total_elapsed = time.time() - total_start
print(f"\n{'='*70}")
print(f"✅ 全部完成，总耗时: {total_elapsed/3600:.2f}h")
print(f"{'='*70}")