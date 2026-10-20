import json
import os
import numpy as np
import torch

from src.provenance import sha1_file
from src.models import random_phase_rotation


def evaluate_checkpoint(
    checkpoint_path,
    model_class,
    X_test_t,
    y_test_t,
    device,
    n_rotations=1,
    rotate=False,
    eval_batch_size=2048,
    num_classes=11,
):
    """以 (checkpoint_path, sha1) 为入参的评估函数。"""
    assert os.path.exists(checkpoint_path), f"Checkpoint not found: {checkpoint_path}"
    ckpt_sha1 = sha1_file(checkpoint_path)

    model = model_class(num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    all_accs = []

    for rot_idx in range(n_rotations):
        if rotate:
            np.random.seed(rot_idx)
        correct, total = 0, 0
        with torch.no_grad():
            for i in range(0, len(X_test_t), eval_batch_size):
                xb = X_test_t[i:i + eval_batch_size].to(device)
                yb = y_test_t[i:i + eval_batch_size].to(device)
                if rotate:
                    xb = random_phase_rotation(xb)
                logits = model(xb)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == yb).sum().item()
                total += yb.shape[0]
        all_accs.append(correct / total * 100)

    return {
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha1': ckpt_sha1,
        'n_rotations': n_rotations,
        'rotate': rotate,
        'mean_acc': float(np.mean(all_accs)),
        'std_acc': float(np.std(all_accs)),
        'all_accs': [float(a) for a in all_accs],
    }


def save_eval_result(result, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"OK 评估结果已保存: {out_path}")
