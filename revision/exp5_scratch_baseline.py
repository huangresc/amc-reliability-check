import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collapse_tools import (
    load_official_data, set_seed, evaluate, DEVICE
)
from src.models import MODEL_REGISTRY
from src.data import make_loader

VARIANT = 'M4_FlattenFC'
SEED = 42
EPOCHS = 40
BATCH_SIZE = 256
LR = 1e-3


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

    train_loader = make_loader(
        X_train_t, y_train_t,
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    for epoch in range(EPOCHS):
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
    print(f"From-scratch 40 epoch baseline: acc={acc:.2f}%")

    os.makedirs('analysis_results', exist_ok=True)
    with open('analysis_results/exp5_scratch_baseline.json', 'w') as f:
        json.dump({'acc': acc}, f, indent=2)
    print("Saved to analysis_results/exp5_scratch_baseline.json")


if __name__ == '__main__':
    main()