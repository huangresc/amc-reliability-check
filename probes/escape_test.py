"""
逃逸测试：从崩溃checkpoint出发，把LR放大100×再跑5个epoch
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from src.data import load_data, make_loader
from src.models import M4_FlattenFC

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================
# 在 M4_FlattenFC 上加 hook
# ============================================
def attach_hooks(model):
    """在 classifier[1] (Linear) 和 classifier[2] (ReLU) 上注册 hook"""
    cache = {}

    def hook_fc1(module, input, output):
        cache['preact'] = output.detach()

    def hook_relu1(module, input, output):
        cache['postact'] = output.detach()

    # classifier 结构:
    # [0] Flatten
    # [1] Linear(1024->64)
    # [2] ReLU
    # [3] Dropout
    # [4] Linear(64->11)
    model.classifier[1].register_forward_hook(hook_fc1)
    model.classifier[2].register_forward_hook(hook_relu1)

    return cache


# ============================================
# 评估
# ============================================
def evaluate(model, X_test_t, y_test_t):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X_test_t), 2048):
            xb = X_test_t[i:i+2048].to(DEVICE)
            yb = y_test_t[i:i+2048].to(DEVICE)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    return correct / total * 100


# ============================================
# 探针指标
# ============================================
def compute_probe(model, cache, x_batch):
    model.eval()
    with torch.no_grad():
        _ = model(x_batch)
        postact = cache['postact']
        preact = cache['preact']

        dead_count = int((postact == 0).sum().item())
        total_count = postact.numel()

        return {
            'dead_count': dead_count,
            'total_count': total_count,
            'dead_frac': dead_count / total_count,
            'preact_min': preact.min().item(),
        }


# ============================================
# 逃逸测试
# ============================================
def escape_test(seed, lr_multiplier=100.0, n_epochs=5):
    set_seed(seed)

    # 数据
    data = load_data()
    X_train_t = data['X_train_t']
    y_train_t = data['y_train_t']
    X_test_t = data['X_test_t']
    y_test_t = data['y_test_t']

    # 加载崩溃checkpoint
    ckpt_path = f'runs/M4_FlattenFC/seed{seed}_batch2a/ckpt_ep640.pt'
    if not os.path.exists(ckpt_path):
        print(f"[FAIL] 找不到 {ckpt_path}")
        return None

    model = M4_FlattenFC(num_classes=11).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"[OK] 加载checkpoint: {ckpt_path}")

    # 注册hook
    cache = attach_hooks(model)

    # 新优化器：LR放大100×
    original_lr = 0.001
    new_lr = original_lr * lr_multiplier
    optimizer = optim.Adam(model.parameters(), lr=new_lr)

    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(X_train_t, y_train_t, batch_size=256, num_workers=4)

    # 取一个batch用于探针
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    records = []

    # Epoch 0：逃逸前
    acc = evaluate(model, X_test_t, y_test_t)
    probe = compute_probe(model, cache, x_batch)
    record = {
        'epoch': 0,
        'phase': 'before_escape',
        'lr': original_lr,
        'test_acc': acc,
        'dead_frac': probe['dead_frac'],
        'dead_count': probe['dead_count'],
        'preact_min': probe['preact_min'],
    }
    records.append(record)
    print(f"  [Before] epoch=0, lr={original_lr}, "
          f"acc={acc:.2f}%, dead={probe['dead_count']}/{probe['total_count']}, "
          f"preact_min={probe['preact_min']:.4f}")

    # 训练
    for epoch in range(n_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        acc = evaluate(model, X_test_t, y_test_t)
        probe = compute_probe(model, cache, x_batch)
        record = {
            'epoch': epoch + 1,
            'phase': 'escape',
            'lr': new_lr,
            'test_acc': acc,
            'dead_frac': probe['dead_frac'],
            'dead_count': probe['dead_count'],
            'preact_min': probe['preact_min'],
        }
        records.append(record)
        print(f"  [Escape] epoch={epoch+1}, lr={new_lr}, "
              f"acc={acc:.2f}%, dead={probe['dead_count']}/{probe['total_count']}, "
              f"preact_min={probe['preact_min']:.4f}")

    return records


# ============================================
# 主程序
# ============================================
def main():
    print("=" * 70)
    print("逃逸测试：从崩溃checkpoint出发，LR放大100×再跑5个epoch")
    print("=" * 70)

    seed = 2021
    print(f"\n--- Seed={seed} ---")

    records = escape_test(seed, lr_multiplier=100.0, n_epochs=5)

    if records is None:
        print("[FAIL] 逃逸测试失败")
        return

    print("\n" + "=" * 70)
    print("判定")
    print("=" * 70)

    before = records[0]
    after = records[-1]

    print(f"\n逃逸前: acc={before['test_acc']:.2f}%, dead={before['dead_count']}/{before['total_count']}")
    print(f"逃逸后: acc={after['test_acc']:.2f}%, dead={after['dead_count']}/{after['total_count']}")

    dead_drop = before['dead_frac'] - after['dead_frac']
    acc_gain = after['test_acc'] - before['test_acc']

    print(f"\ndead_frac 变化: {before['dead_frac']:.4f} -> {after['dead_frac']:.4f} "
          f"(差={dead_drop:+.4f})")
    print(f"test_acc 变化: {before['test_acc']:.2f}% -> {after['test_acc']:.2f}% "
          f"(差={acc_gain:+.2f}pp)")

    if after['dead_count'] == after['total_count'] and abs(acc_gain) < 1.0:
        print(f"\n[RESULT] 逃不出 -> '吸收态'坐实")
    elif dead_drop > 0.1 or acc_gain > 5.0:
        print(f"\n[RESULT] 逃出 -> '极窄的吸引盆'")
    else:
        print(f"\n[RESULT] 部分逃逸，需进一步分析")

    out_dir = f'runs/escape_test/M4_seed{seed}'
    os.makedirs(out_dir, exist_ok=True)
    with open(f'{out_dir}/escape_records.json', 'w') as f:
        json.dump(records, f, indent=2)
    print(f"\nOK 记录已保存: {out_dir}/escape_records.json")


if __name__ == '__main__':
    main()