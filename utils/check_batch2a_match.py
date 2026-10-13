"""
对拍检查：M2@640 (seed42) 的前320 epoch，是否与 M2@320 (seed42) 逐位相同？
"""

import json
import os
import sys

# M2@320 的结果路径
path_320 = 'runs/M2_GAPHead/seed42_gate1/metrics.json'

# M2@640 的结果路径
path_640 = 'runs/M2_GAPHead/seed42_batch2a/metrics.json'

if not os.path.exists(path_320):
    print(f"[FAIL] 找不到 {path_320}")
    sys.exit(1)

if not os.path.exists(path_640):
    print(f"[FAIL] 找不到 {path_640}，M2@640 还没跑完？")
    sys.exit(1)

with open(path_320, 'r') as f:
    m320 = json.load(f)
with open(path_640, 'r') as f:
    m640 = json.load(f)

milestones_320 = ['20', '40', '80', '120', '160', '240', '320']

print("=" * 80)
print("对拍检查：M2@640 (seed42) 的前320 epoch vs M2@320 (seed42)")
print("=" * 80)
print(f"{'epoch':<8}{'M2@320':<20}{'M2@640':<20}{'diff':<20}{'status':<10}")
print("-" * 80)

all_match = True
for ep in milestones_320:
    a320 = m320['epochs'][ep]['acc']
    a640 = m640['epochs'][ep]['acc']
    diff = a640 - a320
    status = 'OK' if abs(diff) < 1e-9 else 'FAIL'
    if status == 'FAIL':
        all_match = False
    print(f"{ep:<8}{a320:<20.15f}{a640:<20.15f}{diff:<+20.2e}{status:<10}")

print("=" * 80)
if all_match:
    print("[RESULT] 逐位相同！跨批次、跨预算的逐位复现成立。")
    print("         -> 管道内不存在对 total_epochs 的隐藏依赖。")
else:
    print("[RESULT] 不匹配！管道内可能存在对 total_epochs 的隐藏依赖。")
    print("         -> 立刻停机，报告导师。")
print("=" * 80)