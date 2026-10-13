"""
Revision-experiment queue for the reliability paper.

Three arms, run sequentially on the one GPU, resumable: a run whose
metrics.json already exists is skipped, so the script can be restarted after
an interruption without redoing finished work.

  A1_flat_seeds  FLAT head, raw inputs, 20 further initialisation seeds.
                 Extends the failure-rate cell from 10 to 30 seeds.
  B_order        FLAT head, raw inputs, initialisation seed fixed at 42,
                 mini-batch order pinned to 20 explicit seeds. Separates the
                 effect of data order from the effect of initialisation.
  A2_gap_seeds   GAP head, raw inputs, the same 20 further seeds, so the
                 architecture contrast is paired at n=30.

Every run stores checkpoints at epochs 1,2,3,4,5,7,10,20,40,80,160,320,640 so
the detectability analysis has the early epochs for runs that did NOT fail,
not only for the ones that did.

Usage:
    python rc_queue.py --arm A1_flat_seeds
    python rc_queue.py --arm all
"""
import os
import sys
import time
import json
import argparse
import traceback

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.train import train_one_variant  # noqa: E402
from rc_train import EARLY, LATE  # noqa: E402

EXISTING_SEEDS = [42, 123, 456, 789, 1011, 1213, 1415, 1617, 1819, 2021]
NEW_SEEDS = [7, 13, 55, 101, 256, 512, 777, 1001, 1500, 2048,
             3141, 4096, 5150, 6666, 7777, 8888, 9999, 12345, 24680, 31337]
ORDER_SEEDS = list(range(20))
ANCHOR_SEED = 42          # the initialisation seed known to be vulnerable
EPOCHS = 640

ARMS = {
    "A1_flat_seeds": [("M4_FlattenFC", s, None, f"rc30_s{s}") for s in NEW_SEEDS],
    "A2_gap_seeds": [("M2_GAPHead", s, None, f"rc30_s{s}") for s in NEW_SEEDS],
    "B_order": [("M4_FlattenFC", ANCHOR_SEED, o, f"rcord{o:03d}")
                for o in ORDER_SEEDS],
}


def run_one(variant, seed, shuffle_seed, tag):
    milestones = [e for e in EARLY if e < EPOCHS] + [e for e in LATE if e <= EPOCHS]
    milestones = sorted(set(milestones))

    train_one_variant(
        variant=variant,
        seed=seed,
        total_epochs=EPOCHS,
        milestones=milestones,
        repeat_id=1,
        batch_tag=tag,
        shuffle_seed=shuffle_seed,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    help="one of %s, or 'all'" % ", ".join(ARMS))
    ap.add_argument("--max_runs", type=int, default=0,
                    help="stop after N runs this invocation (0 = no limit)")
    args = ap.parse_args()

    todo = []
    arms = list(ARMS) if args.arm == "all" else [args.arm]
    for arm in arms:
        todo += [(arm,) + t for t in ARMS[arm]]

    print(f"queue: {len(todo)} runs requested, arm(s)={arms}", flush=True)

    done = skipped = failed = 0
    start = time.time()
    for i, (arm, variant, seed, shuffle_seed, tag) in enumerate(todo, 1):
        run_dir = os.path.join("runs", variant, f"seed{seed}_{tag}")
        metrics = os.path.join(run_dir, "metrics.json")
        if os.path.exists(metrics):
            print(f"[{i}/{len(todo)}] SKIP (done) {arm} {variant} "
                  f"seed={seed} shuffle={shuffle_seed}", flush=True)
            skipped += 1
            continue

        print(f"[{i}/{len(todo)}] START {arm} {variant} seed={seed} "
              f"shuffle_seed={shuffle_seed} tag={tag}", flush=True)
        t0 = time.time()
        try:
            run_one(variant, seed, shuffle_seed, tag)
            done += 1
            print(f"[{i}/{len(todo)}] OK    {tag} in {time.time()-t0:.0f}s",
                  flush=True)
        except Exception:
            failed += 1
            print(f"[{i}/{len(todo)}] FAIL  {tag}", flush=True)
            traceback.print_exc()

        if args.max_runs and (done + failed) >= args.max_runs:
            print("max_runs reached, stopping", flush=True)
            break

    print(f"\nqueue finished: done={done} skipped={skipped} failed={failed} "
          f"in {(time.time()-start)/3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
