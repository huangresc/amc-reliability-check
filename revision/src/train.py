"""
唯一训练入口。
用法：
    python -m src.train --variant M2_GAPHead --seed 42 --total_epochs 320
"""

import argparse
import json
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.provenance import (
    build_nuisance, build_partition, build_treatment, save_ckpt, sha1_json,
    assert_gpu
)
from src.data import load_data, make_loader
from src.models import (
    MODEL_REGISTRY, random_phase_rotation, discrete_phase_rotation
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_quick(model, X_test_t, y_test_t, eval_batch_size=2048):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i in range(0, len(X_test_t), eval_batch_size):
            xb = X_test_t[i:i + eval_batch_size].to(DEVICE)
            yb = y_test_t[i:i + eval_batch_size].to(DEVICE)
            logits = model(xb)
            pred = torch.argmax(logits, dim=1)
            correct += (pred == yb).sum().item()
            total += yb.shape[0]
    return correct / total * 100


def get_variant_config(variant):
    config = {
        'M2_GAPHead': {
            'head_spec': 'GAP -> Linear(32->64) -> Linear(64->11)',
            'use_aug': False, 'use_discrete': False, 'aug_spec': None,
            'normalization': 'none',
        },
        'M4_FlattenFC': {
            'head_spec': 'Flatten -> Linear(1024->64) -> Linear(64->11)',
            'use_aug': False, 'use_discrete': False, 'aug_spec': None,
            'normalization': 'none',
        },
        'M5_GAPHead_PhaseAug': {
            'head_spec': 'GAP -> Linear(32->64) -> Linear(64->11)',
            'use_aug': True, 'use_discrete': False, 'aug_spec': 'phase_rotate_uniform_2pi',
            'normalization': 'none',
        },
        'M5b_GAPHead_PhaseAug4': {
            'head_spec': 'GAP -> Linear(32->64) -> Linear(64->11)',
            'use_aug': True, 'use_discrete': True, 'aug_spec': 'phase_rotate_discrete4',
            'normalization': 'none',
        },
        # Tier 2: 归一化消融
        'M2n_g': {
            'head_spec': 'GAP -> Linear(32->64) -> Linear(64->11)',
            'use_aug': False, 'use_discrete': False, 'aug_spec': None,
            'normalization': 'global',
        },
        'M4n_g': {
            'head_spec': 'Flatten -> Linear(1024->64) -> Linear(64->11)',
            'use_aug': False, 'use_discrete': False, 'aug_spec': None,
            'normalization': 'global',
        },
        'M2n_p': {
            'head_spec': 'GAP -> Linear(32->64) -> Linear(64->11)',
            'use_aug': False, 'use_discrete': False, 'aug_spec': None,
            'normalization': 'per_sample',
        },
        'M4n_p': {
            'head_spec': 'Flatten -> Linear(1024->64) -> Linear(64->11)',
            'use_aug': False, 'use_discrete': False, 'aug_spec': None,
            'normalization': 'per_sample',
        },
    }
    return config[variant]


def train_one_variant(variant, seed, total_epochs, milestones,
                       repeat_id=1, batch_tag="gate1", shuffle_seed=None):
    # === 1. 变体配置 ===
    vcfg = get_variant_config(variant)

    # === 2. 数据加载 ===
    data = load_data(normalization=vcfg['normalization'])
    X_train_t = data['X_train_t']
    y_train_t = data['y_train_t']
    X_test_t = data['X_test_t']
    y_test_t = data['y_test_t']
    train_idx = data['train_idx']
    test_idx = data['test_idx']

    # === 3. 构建 cfg ===
    cfg = {
        'variant': variant,
        'seed': seed,
        'optimizer': 'Adam',
        'lr': 0.001,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.0,
        'scheduler': None,
        'loss': 'CrossEntropyLoss',
        'batch_size': 256,
        'total_epochs': total_epochs,
        'dropout': 0.5,
        'normalization': vcfg['normalization'],
        'use_bn': False,
        'dataset': 'RML2016.10a',
        'eval_batch_size': 2048,
        'head_spec': vcfg['head_spec'],
        'use_aug': vcfg['use_aug'],
        'use_discrete': vcfg['use_discrete'],
        'aug_spec': vcfg['aug_spec'],
        'num_workers': 0,
        'pin_memory': False,
        'shuffle_seed': shuffle_seed,
    }

    # === 4. 创建运行目录 ===
    if repeat_id == 1:
        run_id = f"seed{seed}_{batch_tag}"
    else:
        run_id = f"seed{seed}_{batch_tag}_rep{repeat_id}"
    run_dir = os.path.join('runs', variant, run_id)
    os.makedirs(run_dir, exist_ok=True)

    # === 5. 初始化模型 ===
    set_seed(seed)
    model_class = MODEL_REGISTRY[variant]
    model = model_class(num_classes=11).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=cfg['lr'],
        betas=tuple(cfg['betas']), eps=cfg['eps'],
        weight_decay=cfg['weight_decay']
    )
    train_loader = make_loader(
        X_train_t, y_train_t,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle_seed=cfg['shuffle_seed'],
    )

    # === 6. GPU 硬断言 ===
    x_sample, _ = next(iter(train_loader))
    x_sample = x_sample.to(DEVICE, non_blocking=True)
    assert_gpu(model, x_sample)
    print(f"OK GPU断言通过: model on {next(model.parameters()).device}, "
          f"input on {x_sample.device}")

    # === 7. 构建 nuisance 和 treatment ===
    nuisance = build_nuisance(cfg, train_idx, test_idx, model, x_sample)
    treatment = build_treatment(cfg)
    nuisance_hash = sha1_json(nuisance)
    treatment_hash = sha1_json(treatment)

    # The partition fingerprint is a TOP-LEVEL key, not part of nuisance: it
    # identifies which samples the split selected, and putting it inside
    # nuisance would move nuisance_hash and orphan every manifest already on
    # disk.  See src/provenance.py:sha1_source_keys.
    partition = build_partition(data['partition'])

    # === 8. 训练前落盘 manifest.json ===
    manifest = {
        'run_id': run_id,
        'variant': variant,
        'seed': seed,
        'repeat_id': repeat_id,
        'batch_tag': batch_tag,
        'partition': partition,
        'nuisance': nuisance,
        'nuisance_hash': nuisance_hash,
        'treatment': treatment,
        'treatment_hash': treatment_hash,
    }
    with open(os.path.join(run_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    print(f"OK Manifest 已保存: {run_dir}/manifest.json")
    print(f"   partition: {partition['rule']} seed={partition['seed']} "
          f"train_source_sha1={partition['train_source_sha1'][:16]}")
    print(f"   run_id: {run_id}")
    print(f"   nuisance_hash: {nuisance_hash}")
    print(f"   treatment_hash: {treatment_hash}")

    # === 9. 训练 ===
    metrics = {
        'variant': variant,
        'seed': seed,
        'run_id': run_id,
        'nuisance_hash': nuisance_hash,
        'treatment_hash': treatment_hash,
        'epochs': {},
    }

    start = time.time()
    epoch_times = []

    for epoch in range(total_epochs):
        model.train()
        epoch_start = time.time()

        for x, y in train_loader:
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)
            if cfg['use_discrete']:
                x = discrete_phase_rotation(x)
            elif cfg['use_aug']:
                x = random_phase_rotation(x)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        if (epoch + 1) in milestones:
            ckpt_sha1 = save_ckpt(model, optimizer, run_dir, epoch + 1)
            acc = evaluate_quick(model, X_test_t, y_test_t, cfg['eval_batch_size'])

            metrics['epochs'][str(epoch + 1)] = {
                'acc': float(acc),
                'ckpt_sha1': ckpt_sha1,
                'epoch_time': float(epoch_time),
            }
            elapsed = time.time() - start
            print(f"      Epoch {epoch+1}: acc={acc:.2f}%  "
                  f"ckpt_sha1={ckpt_sha1[:8]}  "
                  f"epoch_time={epoch_time:.1f}s  total={elapsed:.0f}s")

    # === 10. 训练后落盘 metrics.json ===
    metrics['total_time'] = float(time.time() - start)
    metrics['mean_epoch_time'] = float(np.mean(epoch_times))
    metrics['std_epoch_time'] = float(np.std(epoch_times))

    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"OK Metrics 已保存: {run_dir}/metrics.json")
    print(f"   总耗时: {metrics['total_time']:.0f}s  "
          f"平均epoch: {metrics['mean_epoch_time']:.1f}s  "
          f"std: {metrics['std_epoch_time']:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--total_epochs', type=int, required=True)
    parser.add_argument('--repeat_id', type=int, default=1)
    parser.add_argument('--batch_tag', type=str, default='gate1')
    args = parser.parse_args()

    if args.total_epochs == 320:
        milestones = [20, 40, 80, 120, 160, 240, 320]
    elif args.total_epochs == 240:
        milestones = [20, 40, 80, 120, 160, 240]
    elif args.total_epochs == 640:
        milestones = [20, 40, 80, 120, 160, 240, 320, 480, 640]
    else:
        milestones = [args.total_epochs]

    print("=" * 70)
    print(f"训练 {args.variant}, seed={args.seed}, total_epochs={args.total_epochs}")
    print(f"   里程碑: {milestones}")
    print(f"   设备: {DEVICE}")
    print(f"   repeat_id: {args.repeat_id}, batch_tag: {args.batch_tag}")
    print("=" * 70)

    train_one_variant(args.variant, args.seed, args.total_epochs, milestones,
                      repeat_id=args.repeat_id, batch_tag=args.batch_tag)


if __name__ == '__main__':
    main()