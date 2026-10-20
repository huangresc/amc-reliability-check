"""
机制探针：死ReLU假说验证（归一化臂）
- 独立脚本，不碰 train.py
- 只跑 M4n_g 和 M2n_g（全局归一化）
- 记录 dead_frac_c1 / grad_norm_c1 / logit_std / test_acc / train_loss
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

from src.data import load_data, make_loader
from src.models import M4_FlattenFC, M2_GAPHead

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================
# 在模型上注册 hook
# ============================================
def attach_hooks(model, variant):
    """根据变体在 fc1 和 relu1 上注册 hook"""
    cache = {}

    if variant in ('M4_FlattenFC', 'M4n_g'):
        # classifier 结构: [0]Flatten [1]Linear [2]ReLU [3]Dropout [4]Linear
        model.classifier[1].register_forward_hook(
            lambda m, i, o: cache.update({'preact': o.detach()})
        )
        model.classifier[2].register_forward_hook(
            lambda m, i, o: cache.update({'postact': o.detach()})
        )
    elif variant in ('M2_GAPHead', 'M2n_g'):
        # 结构相同，classifier 与 M4 一致
        model.classifier[1].register_forward_hook(
            lambda m, i, o: cache.update({'preact': o.detach()})
        )
        model.classifier[2].register_forward_hook(
            lambda m, i, o: cache.update({'postact': o.detach()})
        )
    return cache


# ============================================
# 评估
# ============================================
def evaluate(model, X_test_t, y_test_t, batch_size=2048):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X_test_t), batch_size):
            xb = X_test_t[i:i+batch_size].to(DEVICE)
            yb = y_test_t[i:i+batch_size].to(DEVICE)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    return correct / total * 100


# ============================================
# 探针指标
# ============================================
def compute_probe(model, cache, x_batch, criterion, y_batch):
    model.eval()
    with torch.no_grad():
        logits = model(x_batch)
        postact = cache['postact']
        preact = cache['preact']

        dead_count = int((postact == 0).sum().item())
        total_count = postact.numel()

        return {
            'dead_count': dead_count,
            'total_count': total_count,
            'dead_frac': dead_count / total_count,
            'preact_min': preact.min().item(),
            'preact_mean': preact.mean().item(),
            'logit_std': logits.std(dim=1).mean().item(),
            'train_loss': criterion(logits, y_batch).item(),
        }


# ============================================
# 跑一个探针run
# ============================================
def run_probe(variant, model_name, seed, total_epochs=640,
              probe_epochs=(1, 2, 3, 5, 10, 20, 40, 80, 120, 160, 240, 320, 480, 640),
              normalization='global'):
    set_seed(seed)

    # 数据
    data = load_data(normalization=normalization)
    X_train_t = data['X_train_t']
    y_train_t = data['y_train_t']
    X_test_t = data['X_test_t']
    y_test_t = data['y_test_t']

    # 模型
    if variant in ('M4_FlattenFC', 'M4n_g'):
        model = M4_FlattenFC(num_classes=11).to(DEVICE)
    else:
        model = M2_GAPHead(num_classes=11).to(DEVICE)

    cache = attach_hooks(model, variant)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_loader = make_loader(X_train_t, y_train_t, batch_size=256, num_workers=4)

    # 输出目录
    out_dir = f'runs/probe/{model_name}_seed{seed}'
    os.makedirs(out_dir, exist_ok=True)

    manifest = {
        'run_id': f'{model_name}_seed{seed}',
        'variant': variant,
        'seed': seed,
        'experiment': 'mechanism_probe',
        'normalization': normalization,
        'total_epochs': total_epochs,
        'note': 'probe only; never enters main tables',
    }
    with open(f'{out_dir}/manifest.json', 'w') as f:
        json.dump(manifest, f, indent=2)

    jsonl_path = f'{out_dir}/probe.jsonl'
    with open(jsonl_path, 'w') as f:
        pass

    # 取一个固定 batch 用于探针
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    # Epoch 0（初始化时刻）
    probe = compute_probe(model, cache, x_batch, criterion, y_batch)
    probe['epoch'] = 0
    probe['phase'] = 'init'
    with open(jsonl_path, 'a') as f:
        f.write(json.dumps(probe) + '\n')
    print(f"[INIT] dead_frac_c1={probe['dead_frac']:.4f}, "
          f"preact_min={probe['preact_min']:.4f}, "
          f"logit_std={probe['logit_std']:.4f}")

    # 训练
    start = time.time()
    for epoch in range(total_epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        if (epoch + 1) in probe_epochs:
            probe = compute_probe(model, cache, x_batch, criterion, y_batch)
            probe['epoch'] = epoch + 1
            probe['phase'] = 'train'

            # 梯度范数
            grad_c1_sq = 0.0
            grad_c4_sq = 0.0
            for name, p in model.named_parameters():
                if p.grad is not None:
                    if 'classifier.1' in name:
                        grad_c1_sq += p.grad.norm().item() ** 2
                    elif 'classifier.4' in name:
                        grad_c4_sq += p.grad.norm().item() ** 2
            probe['grad_norm_c1'] = grad_c1_sq ** 0.5
            probe['grad_norm_c4'] = grad_c4_sq ** 0.5

            # test_acc
            probe['test_acc'] = evaluate(model, X_test_t, y_test_t)

            with open(jsonl_path, 'a') as f:
                f.write(json.dumps(probe) + '\n')

            elapsed = time.time() - start
            print(f"[Epoch {epoch+1}] dead={probe['dead_count']}/{probe['total_count']}, "
                  f"grad_c1={probe['grad_norm_c1']:.4e}, "
                  f"logit_std={probe['logit_std']:.4f}, "
                  f"acc={probe['test_acc']:.2f}%, "
                  f"loss={probe['train_loss']:.4f} ({elapsed:.0f}s)")

    print(f"[DONE] {model_name} seed={seed}, total={time.time()-start:.0f}s")


# ============================================
# 主程序
# ============================================
def main():
    print("=" * 70)
    print("机制探针：归一化臂的死ReLU验证")
    print("=" * 70)
    print(f"设备: {DEVICE}")
    print()

    runs = [
        ('M4n_g', 'M4n_g', 2021, 'global'),   # 归一化臂（崩溃组）
        ('M2n_g', 'M2n_g', 2021, 'global'),   # 归一化臂（对照组）
    ]

    for variant, model_name, seed, normalization in runs:
        print(f"\n{'='*70}")
        print(f"探针 run: {variant} seed={seed} (normalization={normalization})")
        print(f"{'='*70}")
        run_probe(variant, model_name, seed, total_epochs=640,
                  normalization=normalization)

    print("\n" + "=" * 70)
    print("全部探针完成")
    print("=" * 70)


if __name__ == '__main__':
    main()