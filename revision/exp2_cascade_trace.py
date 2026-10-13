import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collapse_tools import (
    load_official_data, set_seed, get_relu_stats, evaluate, DEVICE
)
from src.models import MODEL_REGISTRY
from torch.utils.data import DataLoader, TensorDataset

VARIANT = 'M4_FlattenFC'
SEED = 42
MAX_EPOCHS = 640
BATCH_SIZE = 256
LR = 1e-3
SAVE_EVERY = 5  # 每 5 个 epoch 保存一次中间结果


def main():
    set_seed(SEED)
    data = load_official_data(normalization='none')
    X_train_t = data['X_train_t']
    y_train_t = data['y_train_t']
    X_test_t = data['X_test_t']
    y_test_t = data['y_test_t']

    model_class = MODEL_REGISTRY[VARIANT]
    model = model_class(num_classes=11).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )

    os.makedirs('analysis_results', exist_ok=True)
    trace = []
    out_path = 'analysis_results/exp2_cascade_trace.json'

    for epoch in range(MAX_EPOCHS):
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
        dead_fracs = [s['dead_fraction'] for s in relu_stats]
        entry = {
            'epoch': epoch,
            'acc': acc,
            'dead_conv1': dead_fracs[0] if len(dead_fracs) > 0 else None,
            'dead_conv2': dead_fracs[1] if len(dead_fracs) > 1 else None,
            'dead_fc1': dead_fracs[2] if len(dead_fracs) > 2 else None,
        }
        trace.append(entry)
        print(f"Epoch {epoch:3d}: acc={acc:6.2f}%  "
              f"dead_conv1={entry['dead_conv1']:.4f}  "
              f"dead_conv2={entry['dead_conv2']:.4f}  "
              f"dead_fc1={entry['dead_fc1']:.4f}")

        # 每 SAVE_EVERY 个 epoch 保存一次中间结果
        if (epoch + 1) % SAVE_EVERY == 0 or epoch == MAX_EPOCHS - 1:
            with open(out_path, 'w') as f:
                json.dump(trace, f, indent=2)
            print(f"  -> 中间结果已保存到 {out_path}")

    print(f"\n完成 {MAX_EPOCHS} 个 epoch，最终结果已保存到 {out_path}")


if __name__ == '__main__':
    main()