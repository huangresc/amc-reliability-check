import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collapse_tools import (
    load_official_data, find_checkpoint, load_model,
    evaluate, DEVICE, set_seed
)
from src.data import make_loader

VARIANT = 'M4_FlattenFC'
SEED = 42
EPOCH = 640
RUN_DIR = f'runs/{VARIANT}/seed{SEED}_minverify_M4_seed{SEED}_ep{EPOCH}'
REPAIR_EPOCHS = 40
BATCH_SIZE = 256
LR = 1e-3


def reinit_layer(module):
    """重新初始化一个 Linear 层。"""
    if isinstance(module, nn.Linear):
        nn.init.kaiming_uniform_(module.weight, a=np.sqrt(5))
        fan_in = module.weight.size(1)
        bound = 1 / np.sqrt(fan_in)
        nn.init.uniform_(module.bias, -bound, bound)


def main():
    data = load_official_data(normalization='none')
    X_train_t = data['X_train_t']
    y_train_t = data['y_train_t']
    X_test_t = data['X_test_t']
    y_test_t = data['y_test_t']

    ckpt_path = find_checkpoint(RUN_DIR, EPOCH)
    print(f"Loading checkpoint: {ckpt_path}")

    results = []

    # 修复方案 1：只 re-init fc1
    for reinit_fc2 in [False, True]:
        set_seed(SEED)
        model = load_model(VARIANT, ckpt_path)
        # 找到所有 Linear 层
        linears = [m for m in model.modules() if isinstance(m, nn.Linear)]
        if len(linears) < 2:
            print("Warning: less than 2 Linear layers found.")
            continue
        # 第一个 Linear 是 fc1
        reinit_layer(linears[0])
        if reinit_fc2 and len(linears) >= 2:
            reinit_layer(linears[1])

        optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.CrossEntropyLoss()
        train_loader = make_loader(
            X_train_t, y_train_t,
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        for epoch in range(REPAIR_EPOCHS):
            model.train()
            for x, y in train_loader:
                x = x.to(DEVICE, non_blocking=True)
                y = y.to(DEVICE, non_blocking=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

        acc, _ = evaluate(model, X_test_t, y_test_t)
        results.append({
            'reinit_fc1': True,
            'reinit_fc2': reinit_fc2,
            'final_acc': acc,
        })
        print(f"Re-init fc1={True}, fc2={reinit_fc2}: acc={acc:.2f}%")

    os.makedirs('analysis_results', exist_ok=True)
    with open('analysis_results/exp4_repair.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to analysis_results/exp4_repair.json")


if __name__ == '__main__':
    main()