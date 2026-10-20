"""
Reliability-check training driver.

Wraps src.train.train_one_variant so that
  (a) the process working directory is always the project root,
  (b) CUDA determinism can be switched on per run,
  (c) the milestone set always includes the early epochs needed to study
      when the collapse becomes detectable.

Usage (from anywhere):
    python rc_train.py --variant M4_FlattenFC --seed 42 --tag rc_r1
    python rc_train.py --variant M4_FlattenFC --seed 42 --tag rc_d1 --deterministic 1
    python rc_train.py --variant M4_FlattenFC --seed 42 --epochs 10 --tag probe
"""
import os
import sys
import json
import time
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

# Must be set before CUDA is initialised or torch.use_deterministic_algorithms
# cannot make cuBLAS reproducible.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402

from src.train import train_one_variant  # noqa: E402

# Early checkpoints are the point of the rerun: the failure is decided by
# epoch ~7, so a study of detectability needs those epochs on every run,
# not only on the runs that happened to fail.
EARLY = [1, 2, 3, 4, 5, 7, 10]
LATE = [20, 40, 80, 160, 320, 640]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--epochs", type=int, default=640)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--deterministic", type=int, default=0)
    ap.add_argument("--shuffle_seed", type=int, default=None,
                    help="pin the mini-batch order to an explicit seed instead "
                         "of inheriting the ambient RNG stream")
    args = ap.parse_args()

    if args.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        # Explicitly recorded rather than left to the default, because the
        # value is part of what the run is testing.
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

    milestones = [e for e in EARLY if e < args.epochs]
    milestones += [e for e in LATE if e <= args.epochs]
    if args.epochs not in milestones:
        milestones.append(args.epochs)
    milestones = sorted(set(milestones))

    print("=" * 70)
    print(f"rc_train: variant={args.variant} seed={args.seed} "
          f"epochs={args.epochs} tag={args.tag} repeat={args.repeat}")
    print(f"  deterministic={args.deterministic} "
          f"cudnn.deterministic={torch.backends.cudnn.deterministic} "
          f"cudnn.benchmark={torch.backends.cudnn.benchmark} "
          f"shuffle_seed={args.shuffle_seed}")
    print(f"  milestones={milestones}")
    print(f"  cwd={os.getcwd()}")
    print("=" * 70)

    t0 = time.time()
    train_one_variant(
        variant=args.variant,
        seed=args.seed,
        total_epochs=args.epochs,
        milestones=milestones,
        repeat_id=args.repeat,
        batch_tag=args.tag,
        shuffle_seed=args.shuffle_seed,
    )
    dt = time.time() - t0

    run_id = (f"seed{args.seed}_{args.tag}" if args.repeat == 1
              else f"seed{args.seed}_{args.tag}_rep{args.repeat}")
    run_dir = os.path.join("runs", args.variant, run_id)
    stamp = {
        "variant": args.variant,
        "seed": args.seed,
        "epochs": args.epochs,
        "tag": args.tag,
        "repeat": args.repeat,
        "deterministic": bool(args.deterministic),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "torch_version": torch.__version__,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        "shuffle_seed": args.shuffle_seed,
        "wall_seconds": dt,
        "seconds_per_epoch": dt / max(args.epochs, 1),
        "milestones": milestones,
    }
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "rc_stamp.json"), "w") as f:
        json.dump(stamp, f, indent=2)
    print(f"rc_train done in {dt:.1f}s ({dt/max(args.epochs,1):.2f}s/epoch)")


if __name__ == "__main__":
    main()
