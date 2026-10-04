import json
import os
import sys
import numpy as np
import hashlib


TABLE_SPECS = {
    "main_architecture": {
        "batch_tag": "batch2a",
        "variants": ["M2_GAPHead", "M4_FlattenFC"],
        "nuisance": ["framework", "device_used", "cuda", "optimizer", "lr", "betas",
                     "eps", "weight_decay", "scheduler", "loss", "batch_size",
                     "total_epochs", "dropout", "normalization", "use_bn",
                     "num_workers", "pin_memory", "data"],
        "treatment": ["head"],
    },
    "phase_augmentation": {
        "batch_tag": "batch2b",
        "variants": ["M5_GAPHead_PhaseAug", "M5b_GAPHead_PhaseAug4"],
        "nuisance": ["framework", "device_used", "cuda", "optimizer", "lr", "betas",
                     "eps", "weight_decay", "scheduler", "loss", "batch_size",
                     "total_epochs", "dropout", "normalization", "use_bn",
                     "num_workers", "pin_memory", "data"],
        "treatment": ["head", "aug_spec"],
    },
    "four_variants": {
        "batch_tag": None,
        "sources": {
            "M2_GAPHead": "batch2a",
            "M4_FlattenFC": "batch2a",
            "M5_GAPHead_PhaseAug": "batch2b",
            "M5b_GAPHead_PhaseAug4": "batch2b",
        },
        "nuisance": ["framework", "device_used", "cuda", "optimizer", "lr", "betas",
                     "eps", "weight_decay", "scheduler", "loss", "batch_size",
                     "total_epochs", "dropout", "normalization", "use_bn",
                     "num_workers", "pin_memory", "data"],
        "treatment": ["head", "aug_spec"],
    },
    "normalization_ablation_M2": {
        "batch_tag": None,
        "sources": {
            "M2_GAPHead": "batch2a",
            "M2n_g": "tier2",
            "M2n_p": "tier2",
        },
        "nuisance": ["framework", "device_used", "cuda", "optimizer", "lr", "betas",
                     "eps", "weight_decay", "scheduler", "loss", "batch_size",
                     "total_epochs", "dropout", "use_bn",
                     "num_workers", "pin_memory", "data", "head"],
        "treatment": ["normalization"],
    },
    "normalization_ablation_M4": {
        "batch_tag": None,
        "sources": {
            "M4_FlattenFC": "batch2a",
            "M4n_g": "tier2",
            "M4n_p": "tier2",
        },
        "nuisance": ["framework", "device_used", "cuda", "optimizer", "lr", "betas",
                     "eps", "weight_decay", "scheduler", "loss", "batch_size",
                     "total_epochs", "dropout", "use_bn",
                     "num_workers", "pin_memory", "data", "head"],
        "treatment": ["normalization"],
    },
}


def sha1_json(obj):
    return hashlib.sha1(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def load_metrics(runs_dir='runs', batch_tag=None):
    rows = []
    if not os.path.exists(runs_dir):
        return rows
    for variant in os.listdir(runs_dir):
        variant_dir = os.path.join(runs_dir, variant)
        if not os.path.isdir(variant_dir):
            continue
        if variant in ('probe', 'early_probe', 'escape_test'):
            continue
        for run_id in os.listdir(variant_dir):
            if batch_tag is not None and batch_tag not in run_id:
                continue
            if 'smoketest' in run_id:
                continue
            run_dir = os.path.join(variant_dir, run_id)
            metrics_path = os.path.join(run_dir, 'metrics.json')
            manifest_path = os.path.join(run_dir, 'manifest.json')
            if not (os.path.exists(metrics_path) and os.path.exists(manifest_path)):
                continue
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)
                with open(manifest_path) as f:
                    manifest = json.load(f)
            except json.JSONDecodeError:
                continue
            rows.append({
                'variant': variant,
                'run_id': run_id,
                'seed': manifest['seed'],
                'manifest': manifest,
                'epochs': metrics['epochs'],
            })
    return rows


def admissible(rows, spec):
    if len(rows) < 2:
        print("  OK 只有 %d 行数据，跳过可比性检查" % len(rows))
        return True
    nh_set = set()
    for r in rows:
        nuisance_dict = {}
        for k in spec["nuisance"]:
            if k == "data":
                nuisance_dict["data"] = r['manifest']['nuisance']['data']
            elif k == "head":
                nuisance_dict["head"] = r['manifest']['treatment'].get('head')
            else:
                nuisance_dict[k] = r['manifest']['nuisance'].get(k)
        nh_set.add(sha1_json(nuisance_dict))
    assert len(nh_set) == 1, "REFUSING: %d distinct nuisance hashes" % len(nh_set)
    tk = spec["treatment"]
    for t in tk:
        values = set()
        for r in rows:
            v = r['manifest']['treatment'].get(t)
            values.add(json.dumps(v, sort_keys=True))
        if len(values) > 1:
            print("  OK treatment factor '%s' 有 %d 种取值" % (t, len(values)))
        else:
            print("  WARN treatment factor '%s' 只有 1 种取值" % t)
    print("  OK nuisance_hash 一致（按spec计算）")
    return True


def build_table(rows, spec_name, out_dir='tables'):
    os.makedirs(out_dir, exist_ok=True)
    all_epochs = set()
    for r in rows:
        all_epochs.update(int(e) for e in r['epochs'].keys())
    all_epochs = sorted(all_epochs)
    variants = sorted(set(r['variant'] for r in rows))

    per_seed_rows = []
    for r in rows:
        is_replicate = '_rep' in r['run_id']
        row = {
            'variant': r['variant'],
            'run_id': r['run_id'],
            'seed': r['seed'],
            'is_replicate': is_replicate,
        }
        for ep in all_epochs:
            v = r['epochs'].get(str(ep), {}).get('acc', None)
            row['ep%d' % ep] = v
        per_seed_rows.append(row)

    per_seed_path = os.path.join(out_dir, '%s_per_seed.csv' % spec_name)
    with open(per_seed_path, 'w', encoding='utf-8') as f:
        header = ['variant', 'run_id', 'seed', 'is_replicate'] + ['ep%d' % ep for ep in all_epochs]
        f.write(','.join(header) + '\n')
        for row in per_seed_rows:
            line = [str(row['variant']), row['run_id'], str(row['seed']),
                    str(row['is_replicate'])]
            for ep in all_epochs:
                v = row['ep%d' % ep]
                line.append(repr(v) if v is not None else '')
            f.write(','.join(line) + '\n')
    print("  OK 逐种子表已保存: %s" % per_seed_path)

    main_rows = []
    for v in variants:
        v_rows = [r for r in per_seed_rows if r['variant'] == v]
        v_rows_unique = [r for r in v_rows if not r['is_replicate']]
        n_seeds = len(v_rows_unique)
        row = {'variant': v, 'n_seeds': n_seeds}
        for ep in all_epochs:
            vals = [r['ep%d' % ep] for r in v_rows_unique if r['ep%d' % ep] is not None]
            if len(vals) > 0:
                row['ep%d_mean' % ep] = float(np.mean(vals))
                row['ep%d_std' % ep] = float(np.std(vals, ddof=1)) if len(vals) > 1 else None
            else:
                row['ep%d_mean' % ep] = None
                row['ep%d_std' % ep] = None
        main_rows.append(row)

    main_path = os.path.join(out_dir, '%s.csv' % spec_name)
    with open(main_path, 'w', encoding='utf-8') as f:
        header = ['variant', 'n_seeds'] + ['ep%d_mean' % ep for ep in all_epochs] + ['ep%d_std' % ep for ep in all_epochs]
        f.write(','.join(header) + '\n')
        for row in main_rows:
            line = [row['variant'], str(row['n_seeds'])]
            for ep in all_epochs:
                v = row['ep%d_mean' % ep]
                line.append(repr(v) if v is not None else '')
            for ep in all_epochs:
                v = row['ep%d_std' % ep]
                line.append(repr(v) if v is not None else '')
            f.write(','.join(line) + '\n')
    print("  OK 主表已保存: %s" % main_path)
    return main_rows, per_seed_rows


def main():
    print("=" * 70)
    print("build_tables - 多 batch_tag 分组出表")
    print("=" * 70)
    for spec_name, spec in TABLE_SPECS.items():
        print("\n" + "=" * 70)
        print("--- 生成表: %s ---" % spec_name)
        print("=" * 70)

        if spec['batch_tag'] is not None:
            rows = load_metrics(batch_tag=spec['batch_tag'])
            variants_in_spec = spec.get('variants', None)
            if variants_in_spec:
                rows = [r for r in rows if r['variant'] in variants_in_spec]
        else:
            sources = spec['sources']
            rows = []
            for variant, batch_tag in sources.items():
                v_rows = load_metrics(batch_tag=batch_tag)
                v_rows = [r for r in v_rows if r['variant'] == variant]
                rows.extend(v_rows)

        if len(rows) < 2:
            print("  跳过: 只有 %d 行有效数据" % len(rows))
            continue

        print("  加载了 %d 行数据" % len(rows))
        for r in rows[:5]:
            print("    %s/%s (seed=%d)" % (r['variant'], r['run_id'], r['seed']))
        if len(rows) > 5:
            print("    ... 还有 %d 行" % (len(rows) - 5))

        try:
            admissible(rows, spec)
            main_rows, per_seed_rows = build_table(rows, spec_name)
            print("  --- 断言 ---")
            for v in sorted(set(r['variant'] for r in rows)):
                v_unique = [r for r in per_seed_rows
                           if r['variant'] == v and not r['is_replicate']]
                print("  OK %s: n_seeds=%d" % (v, len(v_unique)))
        except AssertionError as e:
            print("  [FAIL] %s" % e)

    print("\n" + "=" * 70)
    print("完成")
    print("=" * 70)


if __name__ == '__main__':
    main()