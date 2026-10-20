import json

for variant in ['M2_GAPHead', 'M4_FlattenFC']:
    with open(f'runs/{variant}/seed42/manifest.json', 'r') as f:
        m = json.load(f)
    print(f"{variant}: {m['nuisance_hash']}")