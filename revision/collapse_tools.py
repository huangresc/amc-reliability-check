import os
import json
import numpy as np
import torch
import torch.nn as nn
from src.data import load_data, make_loader
from src.models import MODEL_REGISTRY

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_official_data(normalization='none'):
    """加载官方 110k/110k 划分数据。"""
    return load_data(normalization=normalization)


def find_checkpoint(run_dir, epoch):
    """在 run_dir 中查找指定 epoch 的 checkpoint 文件。"""
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run dir not found: {run_dir}")
    for fname in os.listdir(run_dir):
        if fname.endswith(('.pt', '.pth')) and str(epoch) in fname:
            return os.path.join(run_dir, fname)
    # 如果找不到，尝试找任意 checkpoint
    for fname in os.listdir(run_dir):
        if fname.endswith(('.pt', '.pth')):
            return os.path.join(run_dir, fname)
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def load_model(variant, checkpoint_path=None, num_classes=11):
    model_class = MODEL_REGISTRY[variant]
    model = model_class(num_classes=num_classes).to(DEVICE)
    if checkpoint_path:
        ckpt = torch.load(checkpoint_path, map_location=DEVICE)
        # 处理多种 checkpoint 格式
        if isinstance(ckpt, dict):
            if 'model_state_dict' in ckpt:
                state = ckpt['model_state_dict']
            elif 'model' in ckpt:
                state = ckpt['model']
            elif 'state_dict' in ckpt:
                state = ckpt['state_dict']
            else:
                state = ckpt
        else:
            state = ckpt
        # 去掉可能的 'module.' 前缀（DataParallel 保存的）
        state = {k.replace('module.', ''): v for k, v in state.items()}
        model.load_state_dict(state)
    model.eval()
    return model


def evaluate(model, X_test_t, y_test_t, eval_batch_size=2048):
    model.eval()
    correct, total = 0, 0
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_t), eval_batch_size):
            xb = X_test_t[i:i+eval_batch_size].to(DEVICE)
            yb = y_test_t[i:i+eval_batch_size].to(DEVICE)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
            all_preds.append(pred.cpu().numpy())
    acc = correct / total * 100
    all_preds = np.concatenate(all_preds)
    return acc, all_preds


def get_relu_stats(model, X, y, eval_batch_size=2048):
    """
    返回每个 ReLU 层的 dead fraction 和 pre-activation 范围。
    通过 forward hook 捕获 ReLU 的输入和输出。
    """
    relu_infos = []
    hooks = []

    def make_hook(idx):
        def hook_fn(module, input, output):
            # input[0] 是 pre-activation
            pre = input[0].detach()
            dead = (output <= 1e-6).float().mean().item()
            relu_infos.append({
                'idx': idx,
                'dead_fraction': dead,
                'pre_min': pre.min().item(),
                'pre_max': pre.max().item(),
            })
        return hook_fn

    relu_idx = 0
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            hooks.append(module.register_forward_hook(make_hook(relu_idx)))
            relu_idx += 1

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X), eval_batch_size):
            xb = X[i:i+eval_batch_size].to(DEVICE)
            model(xb)

    for h in hooks:
        h.remove()

    # 合并多次 batch 的结果：取平均 dead fraction，取全局 min/max
    if not relu_infos:
        return []
    n_relu = relu_idx
    merged = []
    for k in range(n_relu):
        entries = [e for e in relu_infos if e['idx'] == k]
        dead_avg = np.mean([e['dead_fraction'] for e in entries])
        pre_min = min(e['pre_min'] for e in entries)
        pre_max = max(e['pre_max'] for e in entries)
        merged.append({
            'relu_index': k,
            'dead_fraction': dead_avg,
            'pre_min': pre_min,
            'pre_max': pre_max,
        })
    return merged


def get_grad_norms(model, X, y, batch_size=256):
    model.train()
    criterion = nn.CrossEntropyLoss()
    xb = X[:batch_size].to(DEVICE)
    yb = y[:batch_size].to(DEVICE)
    logits = model(xb)
    loss = criterion(logits, yb)
    model.zero_grad()
    loss.backward()
    grad_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms[name] = param.grad.norm().item()
        else:
            grad_norms[name] = 0.0
    return grad_norms