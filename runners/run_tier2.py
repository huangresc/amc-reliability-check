"""
批量运行 Tier 2: M2n_g, M4n_g, M2n_p, M4n_p @ 640 epoch, n=10
"""

import subprocess
import sys
import time
import os
import shutil
import yaml

import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_grid(grid_path='grid_tier2.yaml'):
    with open(grid_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_one(variant, seed, total_epochs, batch_tag, repeat_id=1):
    if repeat_id == 1:
        run_id = f"seed{seed}_{batch_tag}"
    else:
        run_id = f"seed{seed}_{batch_tag}_rep{repeat_id}"

    run_dir = f"runs/{variant}/{run_id}"
    metrics_path = f"{run_dir}/metrics.json"

    print(f"\n{'='*70}")
    print(f"[RUN] {variant} / {run_id}")
    print(f"{'='*70}")

    if os.path.exists(metrics_path):
        print(f"  [SKIP] {run_id} 已完成，跳过")
        return True

    if os.path.exists(run_dir):
        print(f"  [WARN] {run_id} 有崩溃残留，删除后重跑")
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
        print(f"  [FAIL] {run_id} 失败，退出码={result.returncode}")
        return False

    print(f"  [OK] {run_id} 完成 ({elapsed/60:.1f}分钟)")
    return True


def main():
    grid = load_grid()
    batch_tag = grid['batch_tag']
    total_epochs = grid['total_epochs']
    runs = grid['runs']

    print("=" * 70)
    print(f"Tier 2 - 全部 {len(runs)} 个 run")
    print("=" * 70)

    total_start = time.time()
    completed = 0
    failed = 0

    for i, run_cfg in enumerate(runs):
        variant = run_cfg['variant']
        seed = run_cfg['seed']
        repeat_id = run_cfg.get('repeat_id', 1)

        print(f"\n[{i+1}/{len(runs)}]")

        success = run_one(variant, seed, total_epochs, batch_tag, repeat_id)

        if success:
            completed += 1
        else:
            failed += 1
            print(f"\n[ABORT] 在第 {i+1} 个run失败")
            break

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"[DONE] Tier 2 完成")
    print(f"  成功: {completed}, 失败: {failed}")
    print(f"  总耗时: {total_elapsed/3600:.2f}h")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()