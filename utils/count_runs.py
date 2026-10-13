import os

runs_dir = 'runs'
total = 0
stats = {}

for variant in os.listdir(runs_dir):
    variant_dir = os.path.join(runs_dir, variant)
    if not os.path.isdir(variant_dir):
        continue
    if variant in ('probe', 'early_probe', 'escape_test'):
        continue
    for run_id in os.listdir(variant_dir):
        run_dir = os.path.join(variant_dir, run_id)
        metrics_path = os.path.join(run_dir, 'metrics.json')
        if not os.path.exists(metrics_path):
            continue
        total += 1
        parts = run_id.split('_')
        batch_tag = parts[1] if len(parts) >= 2 else 'unknown'
        stats.setdefault(batch_tag, {}).setdefault(variant, 0)
        stats[batch_tag][variant] += 1

print(f"总计: {total} 个 run")
for batch_tag, variants in sorted(stats.items()):
    print(f"\n[{batch_tag}]")
    for variant, count in sorted(variants.items()):
        print(f"  {variant}: {count}")