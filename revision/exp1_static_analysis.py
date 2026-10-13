import os
import json
import numpy as np
import torch
from collapse_tools import (
    load_official_data, find_checkpoint, load_model,
    evaluate, get_relu_stats, get_grad_norms, DEVICE
)

VARIANT = 'M4_FlattenFC'
SEED = 42
EPOCH = 640
RUN_DIR = f'runs/{VARIANT}/seed{SEED}_minverify_{VARIANT.split("_")[0]}_seed{SEED}_ep{EPOCH}'

# 如果上面的路径不对，直接指定实际路径：
RUN_DIR = f'runs/{VARIANT}/seed{SEED}_minverify_M4_seed{SEED}_ep{EPOCH}'

def main():
    data = load_official_data(normalization='none')
    X_test_t = data['X_test_t']
    y_test_t = data['y_test_t']

    ckpt_path = find_checkpoint(RUN_DIR, EPOCH)
    print(f"Loading checkpoint: {ckpt_path}")
    model = load_model(VARIANT, ckpt_path)

    acc, preds = evaluate(model, X_test_t, y_test_t)
    print(f"Test accuracy: {acc:.4f}%")

    # 常数预测器检查
    unique, counts = np.unique(preds, return_counts=True)
    print("Predicted label distribution:")
    for u, c in zip(unique, counts):
        print(f"  label {u}: {c} ({c/len(preds)*100:.2f}%)")

    # 损失
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    total_loss = 0.0
    n = 0
    with torch.no_grad():
        for i in range(0, len(X_test_t), 2048):
            xb = X_test_t[i:i+2048].to(DEVICE)
            yb = y_test_t[i:i+2048].to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * yb.shape[0]
            n += yb.shape[0]
    avg_loss = total_loss / n
    print(f"Average test loss: {avg_loss:.6f}  (ln 11 = {np.log(11):.6f})")

    # ReLU 统计
    relu_stats = get_relu_stats(model, X_test_t, y_test_t)
    print("\nReLU stats:")
    for s in relu_stats:
        print(f"  ReLU {s['relu_index']}: dead={s['dead_fraction']*100:.5f}%  "
              f"pre_range=[{s['pre_min']:.6f}, {s['pre_max']:.6f}]")

    # 梯度范数
    grad_norms = get_grad_norms(model, X_test_t, y_test_t)
    print("\nGradient norms:")
    for name, gn in grad_norms.items():
        print(f"  {name}: {gn:.8f}")

    # 保存结果
    result = {
        'variant': VARIANT,
        'seed': SEED,
        'epoch': EPOCH,
        'test_acc': acc,
        'avg_loss': avg_loss,
        'pred_dist': {int(u): int(c) for u, c in zip(unique, counts)},
        'relu_stats': relu_stats,
        'grad_norms': grad_norms,
    }
    os.makedirs('analysis_results', exist_ok=True)
    with open('analysis_results/exp1_static_analysis.json', 'w') as f:
        json.dump(result, f, indent=2)
    print("\nSaved to analysis_results/exp1_static_analysis.json")


if __name__ == '__main__':
    main()