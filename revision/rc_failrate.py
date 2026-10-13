"""Failure rate under two different units of randomisation.

The same hyperparameter set fails once in ten on the 176 000/44 000 re-split
(seed 2021) and once in ten on the official 110 000/110 000 split (seed 42) --
a different seed each time. So the failure is not a property of a seed; it is a
property of a (seed, partition) draw. Pooling every draw we have measured gives
the honest rate.
"""
import math


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


DRAWS = [
    ("re-split, 10 seeds, FLAT head, raw inputs", 10, 1, "seed 2021"),
    ("official, 30 seeds, FLAT head, raw inputs", 30, 2, "seeds 42, 1500"),
    ("official, 30 seeds, GAP head, raw inputs", 30, 0, "none"),
]

print(f"{'sample':<44} {'k/n':>7} {'rate':>8}   Wilson 95%")
print("-" * 84)
for label, n, k, ids in DRAWS:
    lo, hi = wilson(k, n)
    print(f"{label:<44} {k:>3}/{n:<3} {100 * k / n:>7.2f}%   "
          f"[{100 * lo:.1f}%, {100 * hi:.1f}%]   ({ids})")

flat_n = 10 + 30
flat_k = 1 + 2
lo, hi = wilson(flat_k, flat_n)
print("-" * 84)
print(f"{'FLAT head, all (seed, partition) draws':<44} {flat_k:>3}/{flat_n:<3} "
      f"{100 * flat_k / flat_n:>7.2f}%   [{100 * lo:.1f}%, {100 * hi:.1f}%]")
