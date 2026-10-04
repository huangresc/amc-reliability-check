with open('tables_all.txt', encoding='gbk', errors='ignore') as f:
    content = f.read()

idx = content.find('normalization_ablation_M2')
if idx >= 0:
    print(content[idx:idx+1500])
else:
    print('未找到 normalization_ablation_M2')