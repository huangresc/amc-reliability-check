"""
批量运行 Batch 2a: M2 + M4 @ 640 epoch, n=10 (共20 runs)
先跑 priority_runs 中的2个（seed42），再跑 remaining_runs 中的18个。

用法:
    python run_batch2a.py              # 跑全部（先2个，再18个）
    python run_batch2a.py --priority   # 只跑前2个（seed42）
    python run_batch2a.py --remaining  # 只跑剩余18个
"""

import subprocess
import sys
import time
import os
import shutil
import argparse
import yaml

# 设置UTF-8输出，避免emoji报错
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def load_grid(grid_path='grid_batch2a.yaml'):
    with open(grid_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def run_one(variant, seed, total_epochs, batch_tag, repeat_id=1):
    """跑一个run，失败则整段重跑"""
    if repeat_id == 1:
        run_id = f"seed{seed}_{batch_tag}"
    else:
        run_id = f"seed{seed}_{batch_tag}_rep{repeat_id}"

    run_dir = f"runs/{variant}/{run_id}"
    metrics_path = f"{run_dir}/metrics.json"

    print(f"\n{'='*70}")
    print(f"[RUN] {variant} / {run_id}")
    print(f"{'='*70}")

    # 已完成则跳过
    if os.path.exists(metrics_path):
        print(f"  [SKIP] {run_id} 已完成，跳过")
        return True

    # 崩溃残留则删除
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--priority', action='store_true', help='只跑前2个seed42')
    parser.add_argument('--remaining', action='store_true', help='只跑剩余18个')
    args = parser.parse_args()

    grid = load_grid()
    batch_tag = grid['batch_tag']
    total_epochs = grid['total_epochs']
    priority_runs = grid['priority_runs']
    remaining_runs = grid['remaining_runs']

    # 决定跑哪些
    if args.priority:
        runs_to_execute = priority_runs
        print("=" * 70)
        print(f"Batch 2a - 只跑 priority (seed42)")
        print("=" * 70)
    elif args.remaining:
        runs_to_execute = remaining_runs
        print("=" * 70)
        print(f"Batch 2a - 只跑 remaining (18个)")
        print("=" * 70)
    else:
        runs_to_execute = priority_runs + remaining_runs
        print("=" * 70)
        print(f"Batch 2a - 全部 {len(runs_to_execute)} 个 run")
        print(f"  priority: {len(priority_runs)}")
        print(f"  remaining: {len(remaining_runs)}")
        print("=" * 70)

    total_start = time.time()
    completed = 0
    failed = 0

    for i, run_cfg in enumerate(runs_to_execute):
        variant = run_cfg['variant']
        seed = run_cfg['seed']
        repeat_id = run_cfg.get('repeat_id', 1)

        print(f"\n[{i+1}/{len(runs_to_execute)}]")
        success = run_one(variant, seed, total_epochs, batch_tag, repeat_id)

        if success:
            completed += 1
        else:
            failed += 1
            print(f"\n[ABORT] 在第 {i+1} 个run失败，停止批量运行")
            break

        # 每跑完一个priority run，如果是第一个，提示对拍
        if args.priority or (i == 0 and not args.remaining):
            if i == 0:
                print("\n" + "=" * 70)
                print("[NEXT] M2@640 (seed42) 完成")
                print("请运行对拍检查:")
                print("  python check_batch2a_match.py")
                print("=" * 70)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*70}")
    print(f"[DONE] Batch 2a 完成")
    print(f"  成功: {completed}, 失败: {failed}")
    print(f"  总耗时: {total_elapsed/3600:.2f}h")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()