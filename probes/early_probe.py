"""
早期探针：10个M4种子 × 前15个epoch
每个epoch记录 dead_frac_c1（整数k/64）、preact_min、grad_norm_c1
成本：约8分钟
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
from src.models import MODEL_REGISTRY

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================
# 带钩子的M4模型
# ============================================
class ProbeM4(nn.Module):
    def __init__(self, num_classes=11):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(kernel_size=(1, 2)),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 1 * 32, 64)
        self.relu1 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

        self.probe_cache = {}
        self.fc1.register_forward_hook(self._hook_fc1)
        self.relu1.register_forward_hook(self._hook_relu1)

    def _hook_fc1(self, module, input, output):
        self.probe_cache['preact'] = output.detach()

    def _hook_relu1(self, module, input, output):
        self.probe_cache['postact'] = output.detach()

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# ============================================
# 跑一个种子的前15个epoch
# ============================================
def run_early_probe(seed, total_epochs=15):
    set_seed(seed)

    data = load_data()
    X_train_t = data['X_train_t']
    y_train_t = data['y_train_t']

    model = ProbeM4(num_classes=11).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_loader = make_loader(X_train_t, y_train_t, batch_size=256, num_workers=4)

    out_dir = f'runs/early_probe/M4_seed{seed}'
    os.makedirs(out_dir, exist_ok=True)

    jsonl_path = f'{out_dir}/early_probe.jsonl'
    with open(jsonl_path, 'w') as f:
        pass

    # 取一个batch用于探针
    x_batch, y_batch = next(iter(train_loader))
    x_batch = x_batch.to(DEVICE)
    y_batch = y_batch.to(DEVICE)

    records = []

    # Epoch 0（初始化时刻）
    model.eval()
    with torch.no_grad():
        _ = model(x_batch)
        postact = model.probe_cache['postact']
        preact = model.probe_cache['preact']
        dead_count = int((postact == 0).sum().item())
        total_count = postact.numel()
        record = {
            'epoch': 0,
            'dead_frac_c1_int': f'{dead_count}/{total_count}',
            'dead_frac_c1': dead_count / total_count,
            'preact_min': preact.min().item(),
        }
        records.append(record)
        print(f"  Seed={seed} Epoch 0: dead={dead_count}/{total_count}, preact_min={record['preact_min']:.4f}")

    # 训练
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

        # 探针
        model.eval()
        with torch.no_grad():
            _ = model(x_batch)
            postact = model.probe_cache['postact']
            preact = model.probe_cache['preact']

            dead_count = int((postact == 0).sum().item())
            total_count = postact.numel()

            # 梯度范数
            grad_norm_c1 = 0.0
            for name, p in model.named_parameters():
                if 'fc1' in name and p.grad is not None:
                    grad_norm_c1 += p.grad.norm().item() ** 2
            grad_norm_c1 = grad_norm_c1 ** 0.5

            record = {
                'epoch': epoch + 1,
                'dead_frac_c1_int': f'{dead_count}/{total_count}',
                'dead_frac_c1': dead_count / total_count,
                'preact_min': preact.min().item(),
                'grad_norm_c1': grad_norm_c1,
            }
            records.append(record)

            print(f"  Seed={seed} Epoch {epoch+1}: dead={dead_count}/{total_count}, "
                  f"preact_min={record['preact_min']:.4f}, grad={grad_norm_c1:.4e}")

    with open(jsonl_path, 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')

    return records


# ============================================
# 主程序
# ============================================
def main():
    print("=" * 70)
    print("早期探针：10个M4种子 × 前15个epoch")
    print("=" * 70)

    seeds = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]

    all_records = {}
    for seed in seeds:
        print(f"\n--- Seed={seed} ---")
        records = run_early_probe(seed, total_epochs=15)
        all_records[seed] = records

    # 汇总：每个种子的 dead_frac_c1 随epoch的变化
    print("\n" + "=" * 70)
    print("汇总：dead_frac_c1 (k/64) 随epoch的变化")
    print("=" * 70)

    epochs = list(range(0, 16))
    print(f"{'Seed':<8}" + "".join([f"ep{e:<5}" for e in epochs]))
    print("-" * 100)

    for seed in seeds:
        row = f"{seed:<8}"
        for ep in epochs:
            r = next((r for r in all_records[seed] if r['epoch'] == ep), None)
            if r:
                row += f"{r['dead_frac_c1_int']:<8}"
            else:
                row += f"{'N/A':<8}"
        print(row)


if __name__ == '__main__':
    main()