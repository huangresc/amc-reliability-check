import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collapse_tools import (
    load_official_data, find_checkpoint, load_model,
    evaluate, get_relu_stats, DEVICE, set_seed
)
from src.data import make_loader

VARIANT = 'M4_FlattenFC'
SEED = 42
EPOCH = 640
RUN_DIR = f'runs/{VARIANT}/seed{SEED}_minverify_M4_seed{SEED}_ep{EPOCH}'
LR_MULTIPLIERS = [1, 10, 100, 1000]
EXTRA_EPOCHS = 10
BATCH_SIZE = 256

def main():
    data = load_official_data(normalization='none')
    X_train_t = data['X_train_t']
    y_train_t = data['y_train_t']
    X_test_t = data['X_test_t']
    y_test_t = data['y_test_t']

    ckpt_path = find_checkpoint(RUN_DIR, EPOCH)
    print(f"Loading checkpoint: {ckpt_path}")

    results = []
    for mult in LR_MULTIPLIERS:
        set_seed(SEED)
        model = load_model(VARIANT, ckpt_path)
        lr = 1e-3 * mult
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
        criterion = nn.CrossEntropyLoss()

        train_loader = make_loader(
            X_train_t, y_train_t,
            batch_size=BATCH_SIZE, shuffle=True, num_workers=0
        )

        for epoch in range(EXTRA_EPOCHS):
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
        relu_stats = get_relu_stats(model, X_test_t, y_test_t)
        dead_fc1 = relu_stats[2]['dead_fraction'] if len(relu_stats) > 2 else None

        # logit 标准差
        model.eval()
        with torch.no_grad():
            xb = X_test_t[:2048].to(DEVICE)
            logits = model(xb)
            logit_std = logits.std().item()

        results.append({
            'lr_multiplier': mult,
            'final_acc': acc,
            'dead_fc1': dead_fc1,
            'logit_std': logit_std,
        })
        print(f"LR x{mult}: acc={acc:.2f}%  dead_fc1={dead_fc1:.6f}  logit_std={logit_std:.6f}")

    os.makedirs('analysis_results', exist_ok=True)
    with open('analysis_results/exp3_lr_escape.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved to analysis_results/exp3_lr_escape.json")


if __name__ == '__main__':
    main()